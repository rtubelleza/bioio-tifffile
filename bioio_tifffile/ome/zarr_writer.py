#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OME-NGFF v0.5 zarr exporter for QPTIFF multiscale data.

Takes an xr.DataTree (from Reader.xarray_dask_datatree_data) and an
ome_types.model.OME object (from Reader.ome_metadata), writes a valid
OME-NGFF v0.5 zarr v3 store with optional sharding.

Write path:
  1. Open zarr v3 store, create scale0/image, scale1/image, ... arrays
     with per-level chunk and shard shapes.
  2. Write pixel data from each DataTree level's dask-backed DataArray.
  3. Write OME-NGFF v0.5 .zattrs to the root group.

Default chunking strategy (aligned with typical whole-slide access patterns):
  - shard_shape: (1, H, W) per level — one full channel plane per shard file.
    Reads that load a single channel (e.g. DAPI) touch exactly one shard.
  - chunk_shape: (1, tile, tile) per level — 1024×1024 spatial tiles inside
    each shard. Supports efficient sub-region access within a channel plane.
"""

from __future__ import annotations

import logging
import typing
from pathlib import Path
from typing import Any, Dict, List, MutableMapping, Optional, Tuple, Union

import dask.array as da
import numpy as np
import xarray as xr
import zarr

from ..qptiff_types import QptiffMetadata

#: bioformats2raw: numbered group holding the (single) image of this fileset.
IMAGE_GROUP = "0"
#: bioformats2raw: group holding METADATA.ome.xml.
OME_GROUP = "OME"
OME_XML_NAME = "METADATA.ome.xml"

log = logging.getLogger(__name__)

# Default inner tile size. Applied per spatial axis up to the array dimension.
_DEFAULT_TILE = 1024


def _default_chunk(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """One channel plane, tiled spatially: (1, tile, tile) for CYX."""
    if len(shape) == 3:
        return (1, min(_DEFAULT_TILE, shape[1]), min(_DEFAULT_TILE, shape[2]))
    # Fallback for non-CYX layouts
    return tuple(min(_DEFAULT_TILE, s) for s in shape)


def _default_shard(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """One full channel plane per shard: (1, H, W) for CYX."""
    if len(shape) == 3:
        return (1, shape[1], shape[2])
    return shape


def _snap_shard_to_chunk(
    shard: Optional[Tuple[int, ...]],
    chunk: Optional[Tuple[int, ...]],
) -> Optional[Tuple[int, ...]]:
    """Round each shard dim up to the nearest multiple of the chunk dim.

    zarr v3 requires shard_shape to be divisible by chunk_shape.  The default
    shard is (1, H, W) but H/W are rarely multiples of the 1024-tile chunk,
    so we ceil-round each dimension to the next multiple.
    """
    if shard is None or chunk is None:
        return shard
    return tuple(
        ((s + c - 1) // c) * c
        for s, c in zip(shard, chunk)
    )


def _normalise_shapes(
    shapes_arg: Optional[Union[Tuple, List[Tuple]]],
    n_levels: int,
    default_fn: typing.Callable[[Tuple[int, ...]], Tuple[int, ...]],
    level_shapes: List[Tuple[int, ...]],
) -> List[Optional[Tuple[int, ...]]]:
    """
    Expand a per-level shape spec to a list of length *n_levels*.

    - ``None`` → compute default for each level from *level_shapes*
    - A single tuple → same shape for all levels
    - A list of tuples → use as-is (must have length *n_levels*)
    """
    if shapes_arg is None:
        return [default_fn(s) for s in level_shapes]
    if isinstance(shapes_arg, tuple) and not isinstance(shapes_arg[0], tuple):
        # Single tuple provided — broadcast to all levels
        return [shapes_arg] * n_levels  # type: ignore[list-item]
    shapes = list(shapes_arg)  # type: ignore[arg-type]
    if len(shapes) != n_levels:
        raise ValueError(
            f"Expected {n_levels} shapes (one per pyramid level), got {len(shapes)}"
        )
    return shapes  # type: ignore[return-value]


def _ome_color_to_hex(color: object) -> Optional[str]:
    """Convert an ome_types Color to a 6-char uppercase hex string (RRGGBB).

    ome_types Color stores colour as a packed RGBA int regardless of whether it
    was constructed from a hex string or a CSS name (e.g. "blue").  Extracting
    the int is the only reliable path; string inspection fails for named colours.
    """
    if color is None:
        return None
    try:
        rgba = int(color)  # packed RGBA: bits 31-24=R, 23-16=G, 15-8=B, 7-0=A
        r = (rgba >> 24) & 0xFF
        g = (rgba >> 16) & 0xFF
        b = (rgba >> 8) & 0xFF
        return f"{r:02X}{g:02X}{b:02X}"
    except Exception:
        pass
    return None


def _write_ome_xml(store: Any, ome_grp: Any, xml: str) -> None:
    """Write METADATA.ome.xml beside the OME group.

    bioformats2raw stores it as a plain file inside the OME group, not as a
    zarr array. For a directory store that is a real file on disk; for other
    stores fall back to stashing it in the group attrs so nothing is lost.
    """
    root_path = None
    if isinstance(store, (str, Path)):
        root_path = Path(store)
    else:
        sp = getattr(store, "root", None) or getattr(store, "path", None)
        if sp is not None:
            root_path = Path(str(sp))

    if root_path is not None:
        try:
            target = root_path / OME_GROUP / OME_XML_NAME
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(xml, encoding="utf-8")
            return
        except OSError as exc:  # pragma: no cover - permissions/remote stores
            log.warning("could not write %s: %s", OME_XML_NAME, exc)

    # non-filesystem store: keep the XML addressable rather than dropping it
    ome_grp.attrs.update({OME_XML_NAME: xml})


def _build_qpi_block(meta: "QptiffMetadata") -> Optional[Dict[str, Any]]:
    """Build the structured ``qpi`` zarr attribute block from the dataclasses.

    Reads :class:`QptiffMetadata` directly. The previous implementation took the
    OME object and regex-split ``ch<N>_`` keys back out of the stringified
    MapAnnotation to rebuild this dict — a lossy round-trip through a transport
    format to recover data that was typed all along.

    Returns ``{"image": {...}, "channels": [...]}``, or None when empty. The
    channel list is ordered by ``ChannelInfo.index`` so it lines up with the
    ``c`` axis regardless of how the channels were discovered.
    """
    from dataclasses import asdict

    def _clean(d: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in d.items() if v is not None}

    image_block: Dict[str, Any] = _clean(asdict(meta.slide))
    fr = meta.full_resolution or (meta.images[0] if meta.images else None)
    if fr is not None:
        image_block.update(_clean(asdict(fr.image_info)))
    for k in ("acquisition_format", "channel_locus"):
        v = getattr(meta, k, None)
        if v is not None:
            image_block[k] = v
    if meta.structure_signature:
        image_block["structure_signature"] = meta.structure_signature

    channels_list: List[Dict[str, Any]] = [
        _clean(asdict(ch)) for ch in sorted(meta.channels, key=lambda c: c.index)
    ]
    # color_rgb tuples -> "#rrggbb", matching the xarray channel coords
    for entry in channels_list:
        rgb = entry.get("color_rgb")
        if isinstance(rgb, (list, tuple)) and len(rgb) == 3:
            entry["color_rgb"] = "#%02x%02x%02x" % tuple(rgb)

    if not image_block and not channels_list:
        return None
    result: Dict[str, Any] = {}
    if image_block:
        result["image"] = image_block
    if channels_list:
        result["channels"] = channels_list
    return result


def _build_ngff_zattrs(
    datatree: xr.DataTree,
    ome: object,
    meta: Optional[QptiffMetadata] = None,
    *,
    ome_only: bool = False,
) -> Dict[str, Any]:
    """
    Build the root .zattrs dict for an OME-NGFF v0.5 store.

    Reads ``pixel_size_um`` from each DataTree level's ``image.attrs``
    (written by ``compute_scale_attrs`` in multiscale.py) to build per-level
    scale coordinate transforms.

    Channel names and optional colors are stored under ``ome.omero.channels``
    as plain dicts — ome-zarr-models-py's ``Channel`` model requires both
    ``color`` and ``window`` which we cannot guarantee from QPTIFF metadata,
    so the omero block is built manually.

    QPI vendor fields (from ``qpi://vectra`` MapAnnotation) are stored under
    a top-level ``"qpi"`` key alongside ``"ome"``, unless ``ome_only=True``.
    """
    from ome_zarr_models.v05.axes import Axis
    from ome_zarr_models.v05.image import ImageAttrs
    from ome_zarr_models.v05.multiscales import Dataset, Multiscale

    axes = [
        Axis(name="c", type="channel"),
        Axis(name="y", type="space", unit="micrometer"),
        Axis(name="x", type="space", unit="micrometer"),
    ]

    level_names: List[str] = sorted(
        (k for k in datatree.children if k.startswith("scale")),
        key=lambda s: int(s[5:]),
    )

    datasets: List[Dataset] = []
    for name in level_names:
        node = datatree[name]
        image_da = node.ds["image"]
        px_sizes = image_da.attrs.get("pixel_size_um")
        if px_sizes is not None:
            py_um = float(px_sizes[0])
            px_um = float(px_sizes[1])
        else:
            py_um, px_um = 1.0, 1.0

        datasets.append(
            Dataset.build(
                path=f"{name}/image",
                scale=[1.0, py_um, px_um],
                translation=None,
            )
        )

    image_name: Optional[str] = None
    try:
        image_name = ome.images[0].name  # type: ignore[union-attr]
    except (AttributeError, IndexError):
        pass

    multiscale = Multiscale(axes=axes, datasets=tuple(datasets), name=image_name)
    image_attrs = ImageAttrs(version="0.5", multiscales=[multiscale])
    ome_dict: Dict[str, Any] = image_attrs.model_dump(exclude_none=True, mode="json")

    # omero.channels — label + optional color per channel.
    # ome-zarr-models-py Channel requires both color and window (which we may
    # not have), so the omero block is constructed manually.
    try:
        channels = ome.images[0].pixels.channels  # type: ignore[union-attr]
        ch_entries: List[Dict[str, Any]] = []
        for ch in channels:
            entry: Dict[str, Any] = {}
            if getattr(ch, "name", None):
                entry["label"] = ch.name
            hex_color = _ome_color_to_hex(getattr(ch, "color", None))
            if hex_color:
                entry["color"] = hex_color
            if entry:
                ch_entries.append(entry)
        if ch_entries:
            ome_dict["omero"] = {"channels": ch_entries}
    except (AttributeError, IndexError):
        pass

    zattrs: Dict[str, Any] = {"ome": ome_dict}

    if not ome_only and meta is not None:
        qpi_block = _build_qpi_block(meta)
        if qpi_block:
            zattrs["qpi"] = qpi_block

    return zattrs


def write_ome_zarr(
    datatree: xr.DataTree,
    ome: object,
    store: Union[str, Path, MutableMapping[str, Any]],
    *,
    overwrite: bool = False,
    ome_only: bool = False,
    meta: Optional[QptiffMetadata] = None,
    zarr_format: int = 3,
    chunk_shape: Optional[Union[Tuple[int, ...], List[Tuple[int, ...]]]] = None,
    shard_shape: Optional[Union[Tuple[int, ...], List[Tuple[int, ...]]]] = None,
    compressor: Optional[Any] = None,
    validate: bool = True,
    consolidate: bool = True,
) -> zarr.Group:
    """
    Write a multiscale DataTree to an OME-NGFF v0.5 zarr store.

    Parameters
    ----------
    datatree:
        Multiscale pyramid DataTree, as produced by
        ``Reader.xarray_dask_datatree_data``. Children must be named
        ``scale0``, ``scale1``, ... and each must contain an ``image``
        DataArray with dims ``(c, y, x)``.
    ome:
        OME metadata object (``ome_types.model.OME``), as produced by
        ``Reader.ome_metadata``.
    store:
        Output zarr path (str / Path) or an existing zarr MutableMapping.
    overwrite:
        If True, replace an existing store. If False (default) and the store
        already exists, raises ``zarr.errors.ContainsGroupError``.
    ome_only:
        If True, omit QPI vendor fields (``qpi://vectra`` MapAnnotation) from
        the store's root ``.zattrs``. Default: False.
    zarr_format:
        Zarr format version. Default: 3 (supports sharding).
    chunk_shape:
        Inner chunk shape, shared by all levels or one tuple per level.
        For CYX arrays, defaults to ``(1, 1024, 1024)`` — one channel,
        1024×1024 spatial tile.
    shard_shape:
        Shard shape (zarr v3 only), shared by all levels or one tuple per level.
        Defaults to ``(1, H, W)`` per level — one full channel plane per shard
        file, matching the typical whole-slide access pattern of loading one
        channel at a time.
        Pass ``None`` explicitly to disable sharding.
    compressor:
        Zarr compressor codec. Defaults to Blosc/zstd when ``zarr_format=3``,
        or the zarr default otherwise. Pass ``[]`` or ``None`` to disable.
    validate:
        If True (default), validate the written store against the OME-NGFF v0.5
        spec using ``ome_zarr_models.open_ome_zarr``. Raises
        ``pydantic.ValidationError`` or ``ValueError`` if the output is invalid.
    consolidate:
        If True (default), call ``zarr.consolidate_metadata`` after writing.
        This merges all per-group ``zarr.json`` files into a single root-level
        entry so the full store metadata can be read in one I/O — important
        for stores accessed through tarballs or remote object storage.

    Returns
    -------
    zarr.Group
        Opened root group of the written store.
    """
    level_names: List[str] = sorted(
        (k for k in datatree.children if k.startswith("scale")),
        key=lambda s: int(s[5:]),
    )
    if not level_names:
        raise ValueError("datatree has no scale children (scale0, scale1, ...)")

    # Collect DataArrays and their shapes up front
    level_arrays: List[xr.DataArray] = []
    level_shapes: List[Tuple[int, ...]] = []
    for name in level_names:
        da_arr = datatree[name].ds["image"]
        level_arrays.append(da_arr)
        level_shapes.append(tuple(da_arr.shape))

    n_levels = len(level_names)
    chunks_per_level = _normalise_shapes(chunk_shape, n_levels, _default_chunk, level_shapes)
    shards_per_level = _normalise_shapes(shard_shape, n_levels, _default_shard, level_shapes)
    # zarr v3: shard dims must be exact multiples of chunk dims — snap up.
    shards_per_level = [
        _snap_shard_to_chunk(s, c)
        for s, c in zip(shards_per_level, chunks_per_level)
    ]

    # Default compressor for zarr v3: clevel=1 for fast writes; bump to 5+
    # via the compressor kwarg when storage size matters more than write speed.
    if compressor is None and zarr_format == 3:
        try:
            from zarr.codecs import BloscCodec, BloscShuffle
            compressor = BloscCodec(cname="zstd", clevel=1, shuffle=BloscShuffle.bitshuffle)
        except ImportError:
            compressor = None

    mode = "w" if overwrite else "w-"
    root = zarr.open_group(store, mode=mode, zarr_format=zarr_format)

    # bioformats2raw transitional layout: the fileset root is a *container*,
    # each image lives in its own numbered group. This is the only place the
    # NGFF spec sanctions for full OME-XML, which the `ome` attribute key
    # (version + multiscales only) cannot hold.
    img_grp = root.create_group(IMAGE_GROUP)

    # Create all zarr arrays first, then write pixel data in a single batched
    # da.store call so dask can schedule reads and writes across all pyramid
    # levels simultaneously rather than processing them one at a time.
    zarr_arrays: List[Any] = []
    dask_arrays: List[da.Array] = []
    numpy_writes: List[tuple] = []  # (zarr_arr, np_data) for non-dask levels

    for name, xr_arr, chunks, shards in zip(
        level_names, level_arrays, chunks_per_level, shards_per_level
    ):
        shape = tuple(xr_arr.shape)
        dtype = xr_arr.dtype

        create_kwargs: Dict[str, Any] = dict(
            shape=shape,
            dtype=dtype,
            chunks=chunks,
            overwrite=False,
        )
        if zarr_format == 3:
            # dimension_names is required by the OME-NGFF v0.5 Image validator
            create_kwargs["dimension_names"] = list(xr_arr.dims)
            if shards is not None:
                create_kwargs["shards"] = shards
        if compressor is not None:
            create_kwargs["compressors"] = [compressor]

        zarr_arr = img_grp.create_array(f"{name}/image", **create_kwargs)

        # Per-scale group attrs and dimension coordinate arrays.
        # Writing c/y/x as sibling arrays to `image` lets xarray (and any
        # other reader that opens the group directly) auto-detect them as
        # dimension coordinates without needing to read the root OME attrs.
        px_sizes = xr_arr.attrs.get("pixel_size_um")
        py_um = float(px_sizes[0]) if px_sizes is not None else 1.0
        px_um = float(px_sizes[1]) if px_sizes is not None else 1.0
        level_idx = int(name[5:])  # "scaleN" → N

        img_grp[name].attrs.update({
            "scale_level": level_idx,
            "pixel_size_um": [py_um, px_um],
        })

        data = xr_arr.data
        if isinstance(data, da.Array):
            dask_arrays.append(data)
            zarr_arrays.append(zarr_arr)
        else:
            numpy_writes.append((zarr_arr, np.asarray(data)))

        log.debug(
            "Created %s/image shape=%s chunks=%s shards=%s",
            name, shape, chunks, shards,
        )

    # enable batch writing all dask-backed levels in one scheduler pass; can config with
    # dask context managers
    if dask_arrays:
        da.store(dask_arrays, zarr_arrays, lock=False)

    for zarr_arr, np_data in numpy_writes:
        zarr_arr[:] = np_data

    # write OME-NGFF v0.5 root .zattrs
    zattrs = _build_ngff_zattrs(datatree, ome, meta, ome_only=ome_only)

    img_grp.attrs.update(zattrs)

    # Declare the transitional layout and write the full OME-XML where the
    # spec says it goes, so bioformats/OMERO actually find it. The previous
    # root "OME" attribute key was invented and no standard reader looked for
    # it; the v0.4 root "multiscales" alias is likewise gone — SpatialData
    # should use the in-memory datatree_to_image2d path instead.
    root.attrs.update({"bioformats2raw.layout": 3})
    try:
        xml = ome.to_xml()  # type: ignore[union-attr]
    except Exception as exc:  # pragma: no cover - non-OME inputs
        log.warning("could not serialise OME-XML, store will carry none: %s", exc)
        xml = None
    if xml is not None:
        ome_grp = root.create_group(OME_GROUP)
        ome_grp.attrs.update({"series": [IMAGE_GROUP]})
        _write_ome_xml(store, ome_grp, xml)

    # validate against OME-NGFF v0.5 spec — Pydantic-validates the ome attrs
    # and confirms every declared dataset path exists as a zarr v3 array.
    if validate:
        from ome_zarr_models import open_ome_zarr
        open_ome_zarr(img_grp, version="0.5")  # raises on any spec violation

    # consolidate all per-group zarr.json files into a single root-level entry
    # so the full store metadata is readable in one I/O (critical for tarballs
    # and remote object storage where directory traversal is expensive).
    if consolidate:
        zarr.consolidate_metadata(root.store)

    log.debug("Wrote OME-NGFF v0.5 zarr store: %s (%d levels)", store, n_levels)
    return root
