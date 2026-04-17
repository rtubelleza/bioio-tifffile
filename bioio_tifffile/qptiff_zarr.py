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
from typing import Any, Dict, List, MutableMapping, Optional, Sequence, Tuple, Union

import dask.array as da
import numpy as np
import xarray as xr
import zarr

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


def _build_qpi_block(ome: object) -> Optional[Dict[str, Any]]:
    """Build the structured ``qpi`` zarr attribute block from an OME object.

    Returns a dict with two keys:

    ``"image"``
        Image-level metadata: instrument model, objective, detector, experimenter,
        acquisition software, and any image-scope fields from the ``qpi://vectra``
        MapAnnotation that lack an OME equivalent.

    ``"channels"``
        A list of per-channel dicts, one entry per channel (indexed to match the
        ``c`` axis).  Each dict combines:

        - OME fields that are not part of the OME-NGFF spec (emission/excitation
          wavelengths, exposure time, detector gain/binning, filter names resolved
          via ``LightPath``, fluorophore, colour name).
        - QPTIFF-vendor fields from the ``qpi://vectra`` MapAnnotation (responsivity,
          filter part numbers, ROI, autofluorescence flag, etc.).

    Returns ``None`` when there is no meaningful QPI data (non-QPTIFF files).
    """
    # --- resolve MapAnnotation ch<N>_* keys into per-channel dicts ----------
    ann_image: Dict[str, str] = {}
    ann_channels: Dict[int, Dict[str, str]] = {}
    import re as _re
    _ch_re = _re.compile(r"^ch(\d+)_(.+)$")
    for ann in getattr(ome, "structured_annotations", []):
        if getattr(ann, "namespace", None) != "qpi://vectra":
            continue
        for k, v in (ann.value or {}).items():
            m = _ch_re.match(str(k))
            if m:
                ann_channels.setdefault(int(m.group(1)), {})[m.group(2)] = str(v)
            else:
                ann_image[str(k)] = str(v)

    # --- build filter id → model lookup from Instrument ---------------------
    filter_model: Dict[str, str] = {}
    try:
        for f in ome.instruments[0].filters:  # type: ignore[union-attr]
            if getattr(f, "id", None) and getattr(f, "model", None):
                filter_model[str(f.id)] = str(f.model)
    except (AttributeError, IndexError, TypeError):
        pass

    # --- image-level block ---------------------------------------------------
    image_block: Dict[str, Any] = {}

    # instrument
    try:
        mic = ome.instruments[0].microscope  # type: ignore[union-attr]
        if getattr(mic, "model", None):
            image_block["microscope_model"] = str(mic.model)
    except (AttributeError, IndexError, TypeError):
        pass
    try:
        obj = ome.instruments[0].objectives[0]  # type: ignore[union-attr]
        if getattr(obj, "model", None):
            image_block["objective_model"] = str(obj.model)
        if getattr(obj, "nominal_magnification", None) is not None:
            image_block["objective_magnification"] = float(obj.nominal_magnification)
    except (AttributeError, IndexError, TypeError):
        pass
    try:
        det = ome.instruments[0].detectors[0]  # type: ignore[union-attr]
        if getattr(det, "model", None):
            image_block["detector_model"] = str(det.model)
    except (AttributeError, IndexError, TypeError):
        pass

    # experimenter
    try:
        exp = ome.experimenters[0]  # type: ignore[union-attr]
        if getattr(exp, "user_name", None):
            image_block["experimenter"] = str(exp.user_name)
    except (AttributeError, IndexError, TypeError):
        pass

    # acquisition date
    try:
        acq = ome.images[0].acquisition_date  # type: ignore[union-attr]
        if acq:
            image_block["acquisition_date"] = str(acq)
    except (AttributeError, IndexError, TypeError):
        pass

    # vendor fields without OME equivalent
    image_block.update(ann_image)

    # --- per-channel list ----------------------------------------------------
    channels_list: List[Dict[str, Any]] = []
    try:
        px = ome.images[0].pixels  # type: ignore[union-attr]
        ome_channels = getattr(px, "channels", [])
        ome_planes = getattr(px, "planes", [])
        plane_by_c: Dict[int, Any] = {int(p.the_c): p for p in ome_planes}

        for i, ch in enumerate(ome_channels):
            entry: Dict[str, Any] = {}

            # OME channel fields not in OME-NGFF spec
            if getattr(ch, "fluor", None):
                entry["fluor"] = str(ch.fluor)

            # colour: always store hex; also store the human name when it's a
            # CSS keyword (e.g. "blue") rather than a "#rrggbb" string.
            color = getattr(ch, "color", None)
            if color is not None:
                hex_c = _ome_color_to_hex(color)
                if hex_c:
                    entry["color_hex"] = hex_c
                color_str = str(color).strip()
                if color_str and not color_str.startswith("#"):
                    entry["color_name"] = color_str

            if getattr(ch, "emission_wavelength", None) is not None:
                entry["emission_wavelength_nm"] = float(ch.emission_wavelength)
            if getattr(ch, "excitation_wavelength", None) is not None:
                entry["excitation_wavelength_nm"] = float(ch.excitation_wavelength)

            # detector settings
            ds = getattr(ch, "detector_settings", None)
            if ds is not None:
                if getattr(ds, "gain", None) is not None:
                    entry["gain"] = float(ds.gain)
                if getattr(ds, "binning", None) is not None:
                    # ds.binning is a Binning enum; .value gives "2x2" etc.
                    bval = getattr(ds.binning, "value", None) or str(ds.binning)
                    entry["binning"] = str(bval)

            # filter names resolved via LightPath
            lp = getattr(ch, "light_path", None)
            if lp is not None:
                exc_refs = getattr(lp, "excitation_filters", [])
                if exc_refs:
                    name = filter_model.get(str(exc_refs[0].id))
                    if name:
                        entry["excitation_filter"] = name
                emi_refs = getattr(lp, "emission_filters", [])
                if emi_refs:
                    name = filter_model.get(str(emi_refs[0].id))
                    if name:
                        entry["emission_filter"] = name

            # exposure time from Plane
            plane = plane_by_c.get(i)
            if plane is not None and getattr(plane, "exposure_time", None) is not None:
                entry["exposure_time_us"] = float(plane.exposure_time)

            # vendor-specific fields from MapAnnotation (responsivity, part nos, ROI, …)
            entry.update(ann_channels.get(i, {}))

            channels_list.append(entry)
    except (AttributeError, IndexError, TypeError):
        pass

    if not image_block and not channels_list and not ann_channels:
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

    if not ome_only:
        qpi_block = _build_qpi_block(ome)
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

        zarr_arr = root.create_array(f"{name}/image", **create_kwargs)

        # Per-scale group attrs and dimension coordinate arrays.
        # Writing c/y/x as sibling arrays to `image` lets xarray (and any
        # other reader that opens the group directly) auto-detect them as
        # dimension coordinates without needing to read the root OME attrs.
        px_sizes = xr_arr.attrs.get("pixel_size_um")
        py_um = float(px_sizes[0]) if px_sizes is not None else 1.0
        px_um = float(px_sizes[1]) if px_sizes is not None else 1.0
        level_idx = int(name[5:])  # "scaleN" → N

        root[name].attrs.update({
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
    zattrs = _build_ngff_zattrs(datatree, ome, ome_only=ome_only)

    # embed full OME-XML so the complete OME object (instrument, filters,
    # detector settings, exposure times, experimenter, …) can be round-tripped
    # via ome_types.from_xml(z.attrs["OME"]) — fields not in the OME-NGFF spec
    # are otherwise inaccessible to standard readers.
    try:
        zattrs["OME"] = ome.to_xml()  # type: ignore[union-attr]
    except Exception:
        pass  # non-OME files: skip silently

    # OME-NGFF v0.4 compatibility: SpatialData and multiscale-spatial-image
    # look for "multiscales" at the root level (not nested under "ome").
    # Writing both lets our zarr v3 store be parsed by Image2DModel.parse()
    # without any structural changes.
    try:
        zattrs["multiscales"] = zattrs["ome"]["multiscales"]
        zattrs["multiscaleSpatialImageVersion"] = 1
    except (KeyError, TypeError):
        pass

    root.attrs.update(zattrs)

    # validate against OME-NGFF v0.5 spec — Pydantic-validates the ome attrs
    # and confirms every declared dataset path exists as a zarr v3 array.
    if validate:
        from ome_zarr_models import open_ome_zarr
        open_ome_zarr(root, version="0.5")  # raises on any spec violation

    # consolidate all per-group zarr.json files into a single root-level entry
    # so the full store metadata is readable in one I/O (critical for tarballs
    # and remote object storage where directory traversal is expensive).
    if consolidate:
        zarr.consolidate_metadata(root.store)

    log.debug("Wrote OME-NGFF v0.5 zarr store: %s (%d levels)", store, n_levels)
    return root
