"""
Helpers for building multiscale xarray.DataTree pyramids from QPTIFF data.

    DataTree
    ├── scale0  Dataset(image: (c, y, x))
    ├── scale1  Dataset(image: (c, y, x))
    └── ...

Every scale node carries the same per-channel coordinates and an attrs dict
with `level`, `dims`, `scale_factors`, and `pixel_size_um`.
"""

from __future__ import annotations

import typing

import multiscale_spatial_image  # registers the .msi accessor on xr.DataTree
import numpy as np
import xarray as xr

from .qptiff_types import ChannelInfo

_UPPER_TO_LOWER = {"C": "c", "S": "c", "Y": "y", "X": "x", "Z": "z", "T": "t"}

# ChannelInfo fields to promote as per-channel xarray coordinates.
# color_rgb tuples are converted to "#rrggbb" hex strings for JSON safety.
_CHANNEL_COORD_FIELDS = [
    "fluorophore",
    "exposure_time_us",
    "emission_wavelength_nm",
    "excitation_wavelength_nm",
    "is_brightfield",
    "color_rgb",
    "is_unmixed_component",
    "gain",
    "binning",
    "bit_depth",
    "excitation_filter_name",
    "emission_filter_name",
    "excitation_filter_manufacturer",
    "emission_filter_manufacturer",
    "responsivity",
    "autofluorescence_subtracted",
    "signal_units",
]


def channel_infos_to_coords(
    channel_infos: typing.List[ChannelInfo],
    channel_dim: str = "c",
) -> typing.Dict[str, typing.Any]:
    """Build xarray coords from a list of ChannelInfo objects.

    The primary ``channel_dim`` coordinate holds channel names. Additional
    per-channel fields from ``_CHANNEL_COORD_FIELDS`` are attached when at
    least one channel has a non-None value. ``color_rgb`` tuples are
    serialised as ``"#rrggbb"`` hex strings.
    """
    coords: typing.Dict[str, typing.Any] = {}
    coords[channel_dim] = xr.Variable(channel_dim, [ch.name for ch in channel_infos])

    for field in _CHANNEL_COORD_FIELDS:
        raw = [getattr(ch, field, None) for ch in channel_infos]
        if not any(v is not None for v in raw):
            continue
        if field == "color_rgb":
            values: typing.List[typing.Any] = [
                f"#{v[0]:02x}{v[1]:02x}{v[2]:02x}" if v is not None else None
                for v in raw
            ]
        else:
            values = raw
        coords[field] = xr.Variable(channel_dim, values)

    return coords


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
        "dims": ["y", "x"],
        "scale_factors": [sy, sx],
    }
    if pixel_size_yx_um is not None:
        py, px = pixel_size_yx_um
        attrs["pixel_size_um"] = [py * sy, px * sx]
    return attrs



def _sanitise_attrs(
    attrs: typing.Dict[str, typing.Any],
) -> typing.Dict[str, typing.Any]:
    """Strip None values from an attrs dict.

    Nested dicts (structured metadata blocks) are kept as-is — they are
    JSON-serializable and zarr / xarray handle them correctly when stored.
    Non-JSON-safe types (numpy arrays, dataclass instances, etc.) are dropped.
    """
    result: typing.Dict[str, typing.Any] = {}
    for k, v in attrs.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool, list, tuple, dict)):
            result[k] = v
    return result


def build_datatree_from_levels(
    levels_arrays: typing.List[xr.DataArray],
    pixel_size_yx_um: typing.Optional[typing.Tuple[float, float]],
    base_attrs: typing.Dict[str, typing.Any],
    channel_infos: typing.Optional[typing.List[ChannelInfo]] = None,
    channel_names: typing.Optional[typing.List[str]] = None,
) -> xr.DataTree:
    """
    Assemble an xr.DataTree with `/scale0`, `/scale1`, ... children.

    Input arrays must already be in ``(c, y, x)`` layout (pass through
    ``squeeze_to_cyx`` upstream). Channel coordinates are built from
    ``channel_infos`` (full ChannelInfo per channel) when provided, falling
    back to bare ``channel_names`` strings for non-QPTIFF files.

    Physical ``y``/``x`` coordinates are added when ``pixel_size_yx_um`` is
    known. Per-level attrs (``level``, ``scale_factors``, ``pixel_size_um``)
    travel on the ``image`` DataArray; image-global ``base_attrs`` go on the
    root DataTree node.
    """
    if not levels_arrays:
        raise ValueError("levels_arrays must contain at least one level")

    level0_yx = (
        int(levels_arrays[0].sizes.get("y", 1)),
        int(levels_arrays[0].sizes.get("x", 1)),
    )

    if channel_infos:
        coords_base = channel_infos_to_coords(channel_infos, channel_dim="c")
    elif channel_names:
        coords_base = {"c": xr.Variable("c", channel_names)}
    else:
        coords_base = {}
    clean_base = _sanitise_attrs(base_attrs)

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

        # Normalise to explicit xr.Variable objects — the (dims, data) tuple
        # shorthand is ambiguous in newer xarray and can cause TypeError when
        # any data value is None or when xarray tries to infer coord types.
        safe_coords: typing.Dict[str, typing.Any] = {}
        for k, v in {**coords_base, **coords_spatial}.items():
            if v is None:
                continue
            if isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], str):
                dims_v, data_v = v
                if data_v is not None:
                    safe_coords[k] = xr.Variable(dims_v, data_v)
            else:
                safe_coords[k] = v

        arr_with_coords = arr.assign_coords(safe_coords) if safe_coords else arr
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
