"""
Helpers for building multiscale xarray.DataTree pyramids from QPTIFF data.

    DataTree
    ├── scale0  Dataset(image: (c, y, x))
    ├── scale1  Dataset(image: (c, y, x))
    └── ...

Every scale node carries the same per-channel coordinates and an attrs dict
with `scale_factors`, `pixel_size_um`, and `level`.
"""

from __future__ import annotations

import typing

import multiscale_spatial_image  # registers the .msi accessor on xr.DataTree
import numpy as np
import xarray as xr

from .qptiff_metadata import CHANNEL_COORD_SCHEMA, ChannelInfo

_UPPER_TO_LOWER = {"C": "c", "S": "c", "Y": "y", "X": "x", "Z": "z", "T": "t"}


def squeeze_to_cyx(arr: xr.DataArray) -> xr.DataArray:
    """
    Drop length-1 T and Z axes and rename dims to the lowercase
    (c, y, x) convention used by `Image2DModel`.

    Brightfield TIFFs expose an `S` (samples) axis rather than `C`; we rename
    it to `c` so downstream consumers see a single channel axis name.
    """
    out = arr
    for squeezable in ("T", "Z"):
        if squeezable in out.dims and out.sizes[squeezable] == 1:
            out = out.squeeze(squeezable, drop=True)

    rename = {d: _UPPER_TO_LOWER[d] for d in out.dims if d in _UPPER_TO_LOWER}
    if rename:
        out = out.rename(rename)

    # compat. with sdat Image2DModel expects (c, y, x). Brightfield TIFFs natively store YXS,
    # which becomes yxc after rename, so transpose the channel axis to front.
    if "c" in out.dims and out.dims[0] != "c":
        desired = ("c",) + tuple(d for d in out.dims if d != "c")
        out = out.transpose(*desired)
    return out


def compute_scale_attrs(
    level_idx: int,
    level0_shape_yx: typing.Tuple[int, int],
    level_shape_yx: typing.Tuple[int, int],
    pixel_size_yx_um: typing.Optional[typing.Tuple[float, float]],
) -> typing.Dict[str, typing.Any]:
    """
    Build the per-level attrs describing this scale's relationship to scale0.

    `scale_factors` is [sy, sx] relative to level 0 (so [1.0, 1.0] at scale0).
    `pixel_size_um` is the absolute pixel size at this level (if known).
    """
    y0, x0 = level0_shape_yx
    y, x = level_shape_yx
    sy = float(y0) / float(y) if y else 1.0
    sx = float(x0) / float(x) if x else 1.0

    attrs: typing.Dict[str, typing.Any] = {
        "level": level_idx,
        "scale_factors": [sy, sx],
    }
    if pixel_size_yx_um is not None:
        py, px = pixel_size_yx_um
        attrs["pixel_size_um"] = [py * sy, px * sx]
    return attrs


def channel_coord_dict(
    channel_names: typing.Optional[typing.List[str]],
    channel_infos: typing.Optional[typing.List[ChannelInfo]],
    channel_dim: str = "c",
    ome_only: bool = False,
) -> typing.Dict[str, typing.Any]:
    """
    Build the coords dict attaching per-channel OME-style metadata to a
    channel axis. Returns an empty dict when no channel info is available.

    Keys are emitted from :data:`CHANNEL_COORD_SCHEMA` so the set of keys
    stays in lockstep with the flat attrs emitted by
    :meth:`QptiffMetadata.to_dict`. Keys follow OME-XML field naming
    (``Channel:Fluor``, ``Plane:ExposureTime``, ``DetectorSettings:Gain``,
    ...); QPTIFF-specific fields with no OME equivalent use a ``qpi_``
    prefix.
    """
    coords: typing.Dict[str, typing.Any] = {}
    if not channel_names:
        return coords
    coords[channel_dim] = list(channel_names)

    if not channel_infos:
        return coords

    n = len(channel_names)
    infos = channel_infos[:n]

    for ome_key, attr, formatter in CHANNEL_COORD_SCHEMA:
        if ome_only and ome_key.startswith("qpi_"):
            continue
        raw = [getattr(ci, attr, None) for ci in infos]
        if formatter is not None:
            vals = [formatter(v) if v is not None else None for v in raw]
        else:
            vals = raw
        if any(v is not None for v in vals):
            coords[ome_key] = (channel_dim, vals)

    return coords


def _sanitise_attrs(
    attrs: typing.Dict[str, typing.Any],
    ome_only: bool = False,
) -> typing.Dict[str, typing.Any]:
    """
    xarray's netCDF-leaning attrs treat plain dicts and None poorly, which
    causes headaches for consumers serialising the tree. Strip those values
    and stringify any remaining non-primitive types. When ``ome_only`` is
    True, also drop vendor keys that use the ``qpi_`` prefix.
    """
    clean: typing.Dict[str, typing.Any] = {}
    for k, v in attrs.items():
        if v is None:
            continue
        if ome_only and isinstance(k, str) and k.startswith("qpi_"):
            continue
        if isinstance(v, (str, int, float, bool, list, tuple, np.ndarray)):
            clean[k] = v
    return clean


def build_datatree_from_levels(
    levels_arrays: typing.List[xr.DataArray],
    channel_names: typing.Optional[typing.List[str]],
    channel_infos: typing.Optional[typing.List[ChannelInfo]],
    pixel_size_yx_um: typing.Optional[typing.Tuple[float, float]],
    base_attrs: typing.Dict[str, typing.Any],
    ome_only: bool = False,
) -> xr.DataTree:
    """
    Assemble an xr.DataTree with `/scale0`, `/scale1`, ... children, to
    follow spec of multiscale_spatial_image

    Each child wraps a single `image` data variable in an xr.Dataset. Input
    arrays are expected to already be in (c, y, x) layout; pass them through
    `squeeze_to_cyx` upstream.

    When ``pixel_size_yx_um`` is known, each level additionally carries
    physical ``y`` / ``x`` coordinates (in um, aligned to the pixel grid at
    that level).

    Attribute placement:

    - Image-global metadata (``base_attrs``) goes on the root DataTree
      attrs — it applies to every scale identically.
    - Per-level metadata (``level``, ``scale_factors``, ``pixel_size_um``)
      goes on the ``image`` DataArray's attrs, not on the Dataset or node
      wrapping it. This keeps all scale-varying info travelling with the
      array when callers slice out a single level.
    """
    if not levels_arrays:
        raise ValueError("levels_arrays must contain at least one level")

    level0_yx = (
        int(levels_arrays[0].sizes.get("y", 1)),
        int(levels_arrays[0].sizes.get("x", 1)),
    )

    coords_base = channel_coord_dict(
        channel_names, channel_infos, channel_dim="c", ome_only=ome_only
    )
    clean_base = _sanitise_attrs(base_attrs, ome_only=ome_only)

    datasets: typing.Dict[str, xr.DataTree] = {}
    for i, arr in enumerate(levels_arrays):
        level_yx = (int(arr.sizes.get("y", 1)), int(arr.sizes.get("x", 1)))
        scale_attrs = compute_scale_attrs(i, level0_yx, level_yx, pixel_size_yx_um)

        # physical y / x coordinates at this level. Spacing varies per level
        # and is what distinguishes scale0 (fine) from scaleN (coarse) in
        # physical units. When pixel size is unknown we leave y / x without
        coords_spatial: typing.Dict[str, typing.Any] = {}
        if pixel_size_yx_um is not None and "pixel_size_um" in scale_attrs:
            py_um, px_um = scale_attrs["pixel_size_um"]
            coords_spatial["y"] = xr.Variable(
                "y",
                np.arange(level_yx[0], dtype=np.float64) * py_um,
                attrs={"units": "um", "pixel_size_um": py_um},
            )
            coords_spatial["x"] = xr.Variable(
                "x",
                np.arange(level_yx[1], dtype=np.float64) * px_um,
                attrs={"units": "um", "pixel_size_um": px_um},
            )

        all_coords = {**coords_base, **coords_spatial}
        arr_with_coords = arr.assign_coords(all_coords) if all_coords else arr
        # per-level attrs live on the DataArray so they ride with a sliced
        # `.image` and don't pollute the Dataset node attrs.
        arr_with_coords.attrs = {**arr_with_coords.attrs, **scale_attrs}
        ds = xr.Dataset(data_vars={"image": arr_with_coords})
        datasets[f"scale{i}"] = xr.DataTree(dataset=ds)

    dt = xr.DataTree(children=datasets)
    # image-global attrs belong on the root, not repeated per scale node
    dt.attrs = clean_base
    _validate_multiscale_spec(dt)
    return dt


def _validate_multiscale_spec(dt: xr.DataTree) -> None:
    """
    Assert that *dt* conforms to the multiscale_spatial_image schema.

    Rules checked (derived from MultiscaleSpatialImage.to_zarr):
    - children are named scale0, scale1, ... with no gaps.
    - every child Dataset contains the variable ``image``.
    - all scales expose the same set of dimensions on ``image``.
    - numeric y/x coordinates, when present, are strictly monotone (spacing > 0).
    """
    children = list(dt.children.keys())
    expected = [f"scale{i}" for i in range(len(children))]
    if children != expected:
        raise ValueError(
            f"DataTree children must be named scale0, scale1, ...; got {children}"
        )

    ref_dims: typing.Optional[typing.Tuple[str, ...]] = None
    for name in children:
        ds = dt[name].dataset
        if "image" not in ds.data_vars:
            raise ValueError(
                f"Scale node '{name}' is missing the required 'image' data variable"
            )
        dims = tuple(ds["image"].dims)
        if ref_dims is None:
            ref_dims = dims
        elif dims != ref_dims:
            raise ValueError(
                f"Dimension mismatch across scales: scale0 has {ref_dims}, "
                f"'{name}' has {dims}"
            )
        for spatial_dim in ("y", "x"):
            if spatial_dim in ds.coords:
                coord = ds.coords[spatial_dim].values
                if len(coord) > 1 and not (coord[1:] > coord[:-1]).all():
                    raise ValueError(
                        f"Coordinate '{spatial_dim}' in '{name}' is not strictly "
                        "monotone increasing — physical spacing must be positive"
                    )
