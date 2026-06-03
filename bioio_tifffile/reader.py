#!/usr/bin/env python
# -*- coding: utf-8 -*-


import logging
import typing
import warnings
import xml.etree.ElementTree as ET

import dask.array as da
import numpy as np
import ome_types
import tifffile
import xarray as xr
from bioio_base import constants, dimensions, exceptions, io, reader, types
from dask import delayed
from fsspec.spec import AbstractFileSystem
from tifffile import TiffFile, imread
from tifffile.tifffile import TiffTags

from .multiscale import build_datatree_from_levels, channel_infos_to_coords, squeeze_to_cyx
from .qptiff_zarr import write_ome_zarr as _write_ome_zarr
from .qptiff_metadata import (
    ChannelInfo,
    QptiffMetadata,
    SlideInfo,
    extract_datetime_from_page,
    ome_metadata_from_qptiff,
    parse_qpi_xml,
    qptiff_meta_to_root_attrs,
    valid_qpi_series,
)
from .utils import generate_ome_channel_id, generate_ome_image_id

###############################################################################

# "Q" is used by tifffile to say "unknown dimension"
# "I" is used to mean a generic image sequence
UNKNOWN_DIM_CHARS = ["Q", "I"]
TIFF_IMAGE_DESCRIPTION_TAG_INDEX = 270

# TIFF tags excluded from the `unprocessed` metadata attrs dict (QPTIFF path):
# - 270 ImageDescription: the QPI XML, fully parsed into the structured schema
#   (slide_info / image_info / channel coords) and preserved on meta.raw_xml.
# - 273/279/324/325: Strip/Tile offset + byte-count arrays — per-tile file byte
#   locations, not metadata, and tens of KB each. Useless once pixels are read.
_UNPROCESSED_TAG_EXCLUDE = {
    TIFF_IMAGE_DESCRIPTION_TAG_INDEX,  # 270 ImageDescription (QPI XML)
    273,  # StripOffsets
    279,  # StripByteCounts
    324,  # TileOffsets
    325,  # TileByteCounts
}

FULL_RESOLUTION_TYPE = "FullResolution"  # qptiff

log = logging.getLogger(__name__)

###############################################################################


class Reader(reader.Reader):
    """
    Wraps the tifffile API to provide the same bioio Reader API but for
    volumetric Tiff (and other tifffile supported) images.

    Parameters
    ----------
    image: types.PathLike
        Path to image file to construct Reader for.
    chunk_dims: Union[str, List[str]]
        Which dimensions to create chunks for.
        Default: DEFAULT_CHUNK_DIMS
        Note: Dimensions.SpatialY, Dimensions.SpatialX, and DimensionNames.Samples,
        will always be added to the list if not present during dask array
        construction.
    dim_order: Optional[Union[List[str], str]]
        A string of dimensions to be applied to all array(s) or a
        list of string dimension names to be mapped onto the list of arrays
        provided to image. I.E. "TYX".
        Default: None (guess dimensions for single array or multiple arrays)
    fs_kwargs: Dict[str, Any]
        Any specific keyword arguments to pass down to the fsspec created filesystem.
        Default: {}
    ome_metadata: bool
        When True, restrict per-channel coords and flat attrs to fields that
        map to an OME-XML equivalent (``Channel:*``, ``Plane:*``,
        ``DetectorSettings:*``, ``Pixels:*``, ``Objective:*``, ...). Vendor
        extensions using the ``qpi_`` prefix are dropped. Default: False
        (emit every available field). Also accepted via
        ``reader_kwargs={"ome_metadata": True}`` when used with BioImage.
    """

    _scenes: typing.Optional[typing.Tuple[str, ...]] = None
    _physical_pixel_sizes: typing.Optional[types.PhysicalPixelSizes] = None

    # qptiff
    # qptiff: maps scene index → tifffile series index (thumbnails, labels, etc.)
    _scene_series_map: typing.Optional[typing.Dict[int, int]] = None
    # qptiff: maps scene index → pyramid level index (0 = full resolution)
    _scene_level_map: typing.Optional[typing.Dict[int, int]] = None
    # qptiff: parsed XML metadata per tifffile series index
    _qpi_meta_cache: typing.Dict[int, QptiffMetadata]
    # qptiff: OME object per tifffile series index (avoids double-parse when
    # write_ome_zarr calls both xarray_dask_datatree_data and ome_metadata)
    _ome_cache: typing.Dict[int, object]

    @staticmethod
    def _is_supported_image(
        fs: AbstractFileSystem, path: str, **kwargs: typing.Any
    ) -> bool:
        try:
            with fs.open(path) as open_resource:
                with TiffFile(open_resource):
                    return True

        except Exception as e:
            raise exceptions.UnsupportedFileFormatError(
                "bioio-tifffile",
                path,
                str(e),
            )

    def __init__(
        self,
        image: types.PathLike,
        chunk_dims: typing.Union[str, typing.List[str]] = dimensions.DEFAULT_CHUNK_DIMS,
        dim_order: typing.Optional[typing.Union[typing.List[str], str]] = None,
        fs_kwargs: typing.Dict[str, typing.Any] = {},
        **kwargs: typing.Any,
    ):
        if ".ome.tif" in str(image):
            log.warning(
                "The image ends with .ome.tiff, which might indicate an OME-TIFF "
                "file format. You might want to install the "
                "`bioio-ome-tiff` plug-in for improved metadata Processing."
                "You can also use 'bioio.plugin_feasibility_report(image)' "
                "method to check if a specific image can be handled by the "
                "available plugins."
            )

        self._fs, self._path = io.pathlike_to_fs(
            image,
            enforce_exists=True,
            fs_kwargs=fs_kwargs,
        )

        # store params
        if isinstance(chunk_dims, str):
            chunk_dims = list(chunk_dims)

        # basic checks on dims and channel names
        if isinstance(dim_order, list):
            if len(dim_order) != len(self.scenes):
                raise exceptions.ConflictingArgumentsError(
                    f"Number of dimension strings provided does not match the "
                    f"number of scenes found in the file. "
                    f"Number of scenes: {len(self.scenes)}, "
                    f"Number of provided dimension order strings: {len(dim_order)}"
                )

        self.chunk_dims = chunk_dims
        self._dim_order = dim_order

        # qptiff

        _rk: typing.Dict[str, typing.Any] = kwargs.get("reader_kwargs", {})
        self._include_aux_series = kwargs.get(
            "include_aux_series", _rk.get("include_aux_series", False)
        )
        # when True, each pyramid level of the FullResolution series is
        # exposed as its own scene.  Disabled by default so that
        # get_xarray_dask_stack() and similar bioio utilities work correctly
        # query the number of levels without enabling this.
        self._include_pyramid_levels = kwargs.get(
            "include_pyramid_levels", _rk.get("include_pyramid_levels", False)
        )
        self._ome_only: bool = bool(
            kwargs.get("ome_metadata", _rk.get("ome_metadata", False))
        )
        self._qpi_meta_cache = {}
        self._ome_cache = {}

        # Enforce valid image
        self._is_supported_image(self._fs, self._path)

    @property
    def scenes(self) -> typing.Tuple[str, ...]:
        if self._scenes is None:
            with self._fs.open(self._path) as open_resource:
                with TiffFile(open_resource, is_mmstack=False) as tiff:
                    (
                        self._scenes,
                        self._scene_series_map,
                        self._scene_level_map,
                    ) = self._build_scene_map(tiff)
        return self._scenes

    @property
    def physical_pixel_sizes(self) -> types.PhysicalPixelSizes:
        """
        Physical pixel sizes in micrometres (Z, Y, X).

        Prefers PixelSizeMicrons from the QPI XML; falls back to TIFF
        XResolution / YResolution tags.
        """
        if self._physical_pixel_sizes is None:
            tiff_series_idx = self._tiff_series_index(self.current_scene_index)
            meta = self.qpi_metadata
            tiff_z = tiff_y = tiff_x = None
            with self._fs.open(self._path) as open_resource:
                try:
                    tiff_z, tiff_y, tiff_x = _get_pixel_size(
                        open_resource, tiff_series_idx
                    )
                except Exception as exc:
                    warnings.warn(f"Could not parse QPTIFF pixel size: {exc}")
            if meta.pixel_size_um is not None:
                px = meta.pixel_size_um
                # warn if TIFF resolution tags disagree with the XML by >1%.
                for axis, tiff_val in (("Y", tiff_y), ("X", tiff_x)):
                    if tiff_val is not None and abs(tiff_val - px) > 0.01 * px:
                        warnings.warn(
                            f"QPTIFF pixel size mismatch on {axis}: XML={px}um, "
                            f"TIFF tag={tiff_val}um; using XML value."
                        )
                self._physical_pixel_sizes = types.PhysicalPixelSizes(None, px, px)
            else:
                self._physical_pixel_sizes = types.PhysicalPixelSizes(
                    tiff_z, tiff_y, tiff_x
                )
        return self._physical_pixel_sizes

    @staticmethod
    def _get_image_data(
        fs: AbstractFileSystem,
        path: str,
        scene: int,
        retrieve_indices: typing.Tuple[typing.Union[int, slice]],
        transpose_indices: typing.List[int],
        level: int = 0,
    ) -> np.ndarray:
        """
        Open a file for reading, construct a Zarr store, select data, and compute to
        numpy.
        Parameters
        ----------
        fs: AbstractFileSystem
            The file system to use for reading.
        path: str
            The path to file to read.
        scene: int
            The scene index to pull the chunk from.
        retrieve_indices: Tuple[Union[int, slice]]
            The image indices to retrieve.
        transpose_indices: List[int]
            The indices to transpose to prior to requesting data.
        Returns
        -------
        chunk: np.ndarray
            The image chunk as a numpy array.
        """
        with fs.open(path) as open_resource:
            with imread(
                open_resource,
                aszarr=True,
                series=scene,
                level=level,
                chunkmode="page",
                is_mmstack=False,
            ) as store:
                arr = da.from_zarr(store)
                arr = arr.transpose(transpose_indices)

                # By setting the compute call to always use a "synchronous" scheduler,
                # it informs Dask not to look for an existing scheduler / client
                # and instead simply read the data using the current thread / process.
                # In doing so, we shouldn't run into any worker data transfer and
                # handoff _during_ a read.
                return arr[retrieve_indices].compute(scheduler="synchronous")

    def _get_tiff_tags(
        self, tiff: TiffFile, process: bool = True
    ) -> typing.Union[TiffTags, typing.Dict[int, typing.Any]]:
        tiff_series_idx = self._tiff_series_index(self.current_scene_index)
        unprocessed = tiff.series[tiff_series_idx].pages[0].tags
        if not process:
            return unprocessed
        return {code: tag.value for code, tag in unprocessed.items()}

    @staticmethod
    def _merge_dim_guesses(dims_from_meta: str, guessed_dims: str) -> str:
        # Construct a "best guess" (super naive)
        best_guess = []
        for dim_from_meta in dims_from_meta:
            # Dim from meta is recognized, add it
            if dim_from_meta not in UNKNOWN_DIM_CHARS:
                best_guess.append(dim_from_meta)

            # Dim from meta isn't recognized
            # Find next dim that isn't already in best guess or dims from meta
            else:
                appended_dim = False
                for guessed_dim in guessed_dims:
                    if (
                        guessed_dim not in best_guess
                        and guessed_dim not in dims_from_meta
                    ):
                        best_guess.append(guessed_dim)
                        appended_dim = True
                        break

                # All of our guess dims were already in the best guess list,
                # append the dim read from meta
                if not appended_dim:
                    best_guess.append(dim_from_meta)

        return "".join(best_guess)

    def _guess_tiff_dim_order(
        self, tiff: TiffFile, tiff_series_idx: typing.Optional[int] = None
    ) -> typing.List[str]:
        if tiff_series_idx is None:
            tiff_series_idx = self._tiff_series_index(self.current_scene_index)
        scene = tiff.series[tiff_series_idx]
        dims_from_meta = scene.pages.axes

        # If all dims are known, simply return as list
        if all(i not in UNKNOWN_DIM_CHARS for i in dims_from_meta):
            return [d for d in dims_from_meta]

        # Otherwise guess the dimensions and return merge
        else:
            # Get basic guess from shape size
            guessed_dims = Reader._guess_dim_order(scene.shape)
            return [d for d in self._merge_dim_guesses(dims_from_meta, guessed_dims)]

    def _get_dims_for_scene(self, tiff: TiffFile) -> typing.List[str]:
        # Get / guess dims
        if self._dim_order is None:
            return self._guess_tiff_dim_order(tiff)

        # Provided list get or guess based
        if isinstance(self._dim_order, list):
            # This list index has a value, use it
            if self._dim_order[self.current_scene_index] is not None:
                return list(self._dim_order[self.current_scene_index])

            # Otherwise guess
            return self._guess_tiff_dim_order(tiff)

        # Provided the same string for all, use
        return list(self._dim_order)

    def _get_channel_names_for_scene(
        self,
        image_shape: typing.Tuple[int, ...],
        dims: typing.List[str],
        channel_infos: typing.Optional[typing.List[ChannelInfo]] = None,
    ) -> typing.Optional[typing.List[str]]:
        """Derive channel names from ChannelInfo objects for the active scene."""
        names: typing.List[str] = []
        if channel_infos is not None:
            names = [ch.name for ch in channel_infos if ch.name]

        if not names:
            return None

        ch_dim = dimensions.DimensionNames.Channel
        samples_dim = "S"  # brightfield RGB uses S samples, not C

        if ch_dim in dims:
            n = image_shape[dims.index(ch_dim)]
            if len(names) >= n:
                return names[:n]
            return names + [f"Channel_{i}" for i in range(len(names), n)]

        if samples_dim in dims:
            n = image_shape[dims.index(samples_dim)]
            if len(names) == n:
                return names

        return None

    def _get_coords(
        self,
        dims: typing.List[str],
        shape: typing.Tuple[int, ...],
        scene_index: int,
        channel_names: typing.Optional[typing.List[str]],
        channel_infos: typing.Optional[typing.List[ChannelInfo]] = None,
    ) -> typing.Dict[str, typing.Any]:
        coords: typing.Dict[str, typing.Any] = {}
        ch_dim = dimensions.DimensionNames.Channel  # "C"
        samples_dim = "S"

        # Determine which dimension actually holds the channel/sample axis
        if ch_dim in dims:
            active_dim = ch_dim
        elif samples_dim in dims:
            active_dim = samples_dim
        else:
            active_dim = None

        if channel_infos is not None and active_dim == ch_dim:
            # Full per-channel coords from ChannelInfo — use uppercase "C" dim
            extra = channel_infos_to_coords(channel_infos, channel_dim=ch_dim)
            coords.update(extra)
        elif channel_names is None:
            if active_dim == ch_dim:
                image_id = generate_ome_image_id(scene_index)
                coords[ch_dim] = [
                    generate_ome_channel_id(image_id=image_id, channel_id=i)
                    for i in range(shape[dims.index(ch_dim)])
                ]
            # No auto-coords for Samples dim — leave unlabelled
        else:
            if active_dim is not None:
                coords[active_dim] = channel_names

        return coords

    def _create_dask_array(
        self, tiff: TiffFile, selected_scene_dims_list: typing.List[str]
    ) -> da.Array:
        """
        Creates a delayed dask array for the active scene at its selected level.
        Thin wrapper around _create_dask_array_for_level for backward compatibility.
        """
        tiff_series_idx = self._tiff_series_index(self.current_scene_index)
        level_idx = self._tiff_level_index(self.current_scene_index)
        return self._create_dask_array_for_level(
            tiff, selected_scene_dims_list, tiff_series_idx, level_idx
        )

    def _create_dask_array_for_level(
        self,
        tiff: TiffFile,
        selected_scene_dims_list: typing.List[str],
        tiff_series_idx: int,
        level_idx: int,
    ) -> da.Array:
        """
        Creates a delayed dask array for an explicit (series, pyramid level) pair.
        Used both by the single-scene read path and the multiscale DataTree
        builder, which needs to materialise every pyramid level lazily.
        """

        # Always add the plane dimensions if not present already
        for dim in dimensions.REQUIRED_CHUNK_DIMS:
            if dim not in self.chunk_dims:
                self.chunk_dims.append(dim)

        # Safety measure / "feature"
        self.chunk_dims = [d.upper() for d in self.chunk_dims]

        selected_scene = tiff.series[tiff_series_idx]
        level_series = selected_scene.levels[level_idx]
        selected_scene_dims = "".join(selected_scene_dims_list)

        # Raise invalid dims error
        if len(level_series.shape) != len(selected_scene_dims):
            raise exceptions.ConflictingArgumentsError(
                f"Dimension string provided does not match the "
                f"number of dimensions found for this scene. "
                f"This scene shape: {level_series.shape}, "
                f"Provided dims string: {selected_scene_dims}"
            )

        # Constuct the chunk and non-chunk shapes one dim at a time
        # We also collect the chunk and non-chunk dimension order so that
        # we can swap the dimensions after we block out the array
        non_chunk_dim_order: typing.List[str] = []
        non_chunk_shape: typing.List[int] = []
        chunk_dim_order: typing.List[str] = []
        chunk_shape: typing.List[int] = []

        for dim, size in zip(selected_scene_dims, level_series.shape):
            if dim in self.chunk_dims:
                chunk_dim_order.append(dim)
                chunk_shape.append(size)
            else:
                non_chunk_dim_order.append(dim)
                non_chunk_shape.append(size)

        # Fill out the rest of the blocked shape with dimension sizes of 1 to
        # match the length of the sample chunk
        # When dask.block happens it fills the dimensions from inner-most to
        # outer-most with the chunks as long as the dimension is size 1
        blocked_dim_order = non_chunk_dim_order + chunk_dim_order
        blocked_shape = tuple(non_chunk_shape) + ((1,) * len(chunk_shape))

        # Construct the transpose indices that will be used to
        # transpose the array prior to pulling the chunk dims
        match_map = {d: selected_scene_dims.find(d) for d in selected_scene_dims}
        transposer = []
        for dim in blocked_dim_order:
            transposer.append(match_map[dim])

        # Make ndarray for lazy arrays to fill
        lazy_arrays: np.ndarray = np.ndarray(blocked_shape, dtype=object)
        for np_index, _ in np.ndenumerate(lazy_arrays):
            # All dimensions get their normal index except for chunk dims
            # which get filled with "full" slices
            indices_with_slices = np_index[: len(non_chunk_shape)] + (
                (slice(None, None, None),) * len(chunk_shape)
            )

            # Fill the numpy array with the delayed arrays
            lazy_arrays[np_index] = da.from_delayed(
                delayed(Reader._get_image_data)(
                    fs=self._fs,
                    path=self._path,
                    scene=tiff_series_idx,
                    retrieve_indices=indices_with_slices,
                    transpose_indices=transposer,
                    level=level_idx,
                ),
                shape=chunk_shape,
                dtype=level_series.dtype,
            )

        # Convert the numpy array of lazy readers into a dask array
        image_data = da.block(lazy_arrays.tolist())

        # Because we have set certain dimensions to be chunked and others not
        # we will need to transpose back to original dimension ordering
        # Example, if the original dimension ordering was "TZYX" and we
        # chunked by "T", "Y", and "X"
        # we created an array with dimensions ordering "ZTYX"
        transpose_indices = []
        for i, d in enumerate(selected_scene_dims):
            new_index = blocked_dim_order.index(d)
            if new_index != i:
                transpose_indices.append(new_index)
            else:
                transpose_indices.append(i)

        # Transpose back to normal
        image_data = da.transpose(image_data, tuple(transpose_indices))

        return image_data

    def _read_delayed(self) -> xr.DataArray:
        """
        Construct the delayed xarray DataArray object for the image.
        Returns
        -------
        image: xr.DataArray
            The fully constructed and fully delayed image as a DataArray object.
            Metadata is attached in some cases as coords, dims, and attrs.
        Raises
        ------
        exceptions.UnsupportedFileFormatError
            The file could not be read or is not supported.
        """
        with self._fs.open(self._path) as open_resource:
            with TiffFile(open_resource, is_mmstack=False) as tiff:
                dims = self._get_dims_for_scene(tiff)
                image_data = self._create_dask_array(tiff, dims)
                tiff_tags = self._get_tiff_tags(tiff)

                # get the current image being processed;
                # below change to modular pattern:
                # qptiff -> parsers -> qptiffmetadata canonical dataclass
                # qptiffmetadata -> ome_types OME object
                # then for now put everything in xarray dataarray format
                tiff_series_idx = self._tiff_series_index(self.current_scene_index)
                series = tiff.series[tiff_series_idx]
                if valid_qpi_series(series.pages):
                    meta = self._get_or_parse_meta(tiff_series_idx, series)
                    attrs = self._build_attrs(tiff_tags, meta, series)
                    channels = self._get_channel_names_for_scene(
                        image_data.shape, dims, channel_infos=meta.channels
                    )
                    coords = self._get_coords(
                        dims,
                        image_data.shape,
                        scene_index=self.current_scene_index,
                        channel_names=channels,
                        channel_infos=meta.channels,
                    )
                else:
                    # non-QPTIFF tifffile fallback; NOTE: can remove if this becomes standalone plugin,
                    channels = self._get_channel_names_for_scene(image_data.shape, dims)
                    coords = self._get_coords(
                        dims,
                        image_data.shape,
                        scene_index=self.current_scene_index,
                        channel_names=channels,
                    )
                    try:
                        attrs = {
                            constants.METADATA_UNPROCESSED: tiff_tags,
                            constants.METADATA_PROCESSED: tiff_tags[
                                TIFF_IMAGE_DESCRIPTION_TAG_INDEX
                            ],
                        }
                    except KeyError:
                        attrs = {constants.METADATA_UNPROCESSED: tiff_tags}

                return xr.DataArray(
                    image_data,
                    dims=dims,
                    coords=coords,
                    attrs=attrs,
                )

    def _read_immediate(self) -> xr.DataArray:
        """
        Construct the in-memory xarray DataArray object for the image.
        Returns
        -------
        image: xr.DataArray
            The fully constructed and fully read into memory image as a DataArray
            object. Metadata is attached in some cases as coords, dims, and attrs.
        Raises
        ------
        exceptions.UnsupportedFileFormatError
            The file could not be read or is not supported.
        """
        with self._fs.open(self._path) as open_resource:
            with TiffFile(open_resource, is_mmstack=False) as tiff:
                dims = self._get_dims_for_scene(tiff)
                tiff_tags = self._get_tiff_tags(tiff)

                tiff_series_idx = self._tiff_series_index(self.current_scene_index)
                level_idx = self._tiff_level_index(self.current_scene_index)
                series = tiff.series[tiff_series_idx]

                if valid_qpi_series(series.pages):
                    # qptiff
                    meta = self._get_or_parse_meta(tiff_series_idx, series)
                    image_data = series.levels[level_idx].asarray()
                    attrs = self._build_attrs(tiff_tags, meta, series)
                    channels = self._get_channel_names_for_scene(
                        image_data.shape, dims, channel_infos=meta.channels
                    )
                    coords = self._get_coords(
                        dims,
                        image_data.shape,
                        scene_index=self.current_scene_index,
                        channel_names=channels,
                        channel_infos=meta.channels,
                    )
                else:
                    # tifffile
                    image_data = series.asarray()
                    channels = self._get_channel_names_for_scene(image_data.shape, dims)
                    coords = self._get_coords(
                        dims,
                        image_data.shape,
                        scene_index=self.current_scene_index,
                        channel_names=channels,
                    )
                    try:
                        attrs = {
                            constants.METADATA_UNPROCESSED: tiff_tags,
                            constants.METADATA_PROCESSED: tiff_tags[
                                TIFF_IMAGE_DESCRIPTION_TAG_INDEX
                            ],
                        }
                    except KeyError:
                        attrs = {constants.METADATA_UNPROCESSED: tiff_tags}

                return xr.DataArray(
                    image_data,
                    dims=dims,
                    coords=coords,
                    attrs=attrs,
                )

    # qptiff
    @staticmethod
    def _is_pyramidal(series: typing.Any) -> bool:
        """True when the tifffile series has more than one resolution level."""
        return len(getattr(series, "levels", [])) > 1

    # qptiff
    def _build_scene_map(
        self, tiff: TiffFile
    ) -> typing.Tuple[
        typing.Tuple[str, ...],
        typing.Dict[int, int],
        typing.Dict[int, int],
    ]:
        """
        Build scene names and mappings {scene_index: tifffile_series_index} and
        {scene_index: pyramid_level_index}.

        For qptiff files, tiff series correspond to image types (FullResolution,
        Thumbnail, Label, etc.). For pyramidal FullResolution series, each
        pyramid level becomes its own scene so callers can select resolution via
        current_scene_index.

        Three cases are handled:

        1. Normal tifffile (non-qpi): each tiff series → one scene at level 0.
        2. Single-scale qptiff: FullResolution series has one level → one scene.
        3. Pyramidal qptiff: FullResolution series has N levels → N scenes
           (FullResolution, FullResolution_level1, …).

        Args:
        tiff: TiffFile
            The opened tifffile.TiffFile object.

        Returns:
            Tuple of (scene_names, scene_series_map, scene_level_map).
        """
        # each entry: (tiff_series_idx, pyramid_level_idx, display_name)
        full_res: typing.List[typing.Tuple[int, int, str]] = []
        aux: typing.List[typing.Tuple[int, int, str]] = []

        for tiff_idx, series in enumerate(tiff.series):
            if valid_qpi_series(series.pages):
                meta = self._get_or_parse_meta(tiff_idx, series)
                image_type = meta.image_type or f"Series_{tiff_idx}"
                if image_type == FULL_RESOLUTION_TYPE:
                    if self._is_pyramidal(series) and self._include_pyramid_levels:
                        # one scene per resolution level.
                        # disabled by default because scenes at different resolutions
                        # have different shapes, which breaks get_xarray_dask_stack() downstream
                        for level_idx in range(len(series.levels)):
                            name = (
                                image_type
                                if level_idx == 0
                                else f"{image_type}_level{level_idx}"
                            )
                            full_res.append((tiff_idx, level_idx, name))
                    else:
                        # ss or pyramidal qptiff (default): one scene at
                        # full resolution (level 0).  Use pyramid_level_count to
                        # query available levels.
                        full_res.append((tiff_idx, 0, image_type))
                else:
                    aux.append((tiff_idx, 0, image_type))
            else:
                # normal tifffile (non-qpi): backward-compatible, level 0 only
                full_res.append((tiff_idx, 0, generate_ome_image_id(tiff_idx)))

        candidates = full_res + (aux if self._include_aux_series else [])
        if not candidates:
            candidates = [(0, 0, generate_ome_image_id(0))]

        scene_names: typing.List[str] = []
        scene_series_map: typing.Dict[int, int] = {}
        scene_level_map: typing.Dict[int, int] = {}
        seen: typing.Dict[str, int] = {}

        for tiff_idx, level_idx, name in candidates:
            count = seen.get(name, 0)
            seen[name] = count + 1
            unique = name if count == 0 else f"{name}_{count}"
            scene_idx = len(scene_names)
            scene_names.append(unique)
            scene_series_map[scene_idx] = tiff_idx
            scene_level_map[scene_idx] = level_idx

        return tuple(scene_names), scene_series_map, scene_level_map

    # qptiff
    def _tiff_series_index(self, scene_idx: int) -> int:
        """Map scene index to the underlying tifffile series index."""
        if self._scene_series_map is None:
            _ = self.scenes
        assert self._scene_series_map is not None
        return self._scene_series_map.get(scene_idx, scene_idx)

    # qptiff
    def _tiff_level_index(self, scene_idx: int) -> int:
        """Map scene index to the pyramid level index (0 = full resolution)."""
        if self._scene_level_map is None:
            _ = self.scenes
        assert self._scene_level_map is not None
        return self._scene_level_map.get(scene_idx, 0)

    # qptiff
    def _get_or_parse_meta(
        self,
        tiff_series_idx: int,
        series: typing.Any,
    ) -> QptiffMetadata:
        if tiff_series_idx not in self._qpi_meta_cache:
            # collect per-page XMLs; for Fusion paged format each page carries
            # its own channel XML, for Polaris every page has the same root XML.
            per_page_xmls: typing.List[str] = []
            try:
                for page in series.pages:
                    xml_val = ""
                    try:
                        tag = page.tags.get(TIFF_IMAGE_DESCRIPTION_TAG_INDEX)
                        if tag is not None and isinstance(tag.value, str) and tag.value:
                            root = ET.fromstring(tag.value)
                            if (
                                "PerkinElmer-QPI" in root.tag
                                or "PerkinElmerQPI" in root.tag
                            ):
                                xml_val = tag.value
                    except Exception:
                        pass
                    per_page_xmls.append(xml_val)
            except Exception:
                per_page_xmls = []

            # n_channels from axes when tifffile resolves them; fall back to
            # page count (valid for CYX layout where pages == channels).
            axes = getattr(series.pages, "axes", "")
            shape = series.shape
            n_channels: typing.Optional[int] = None
            if "C" in axes:
                n_channels = shape[axes.index("C")]
            elif "S" in axes:
                n_channels = shape[axes.index("S")]
            elif per_page_xmls:
                n_channels = len(per_page_xmls)

            # Brightfield = RGB samples (S dimension / >=3 samples per pixel).
            # This is the reliable discriminator: a <BFLampType> element alone is
            # not, since Fusion 2.x fluorescence scans also emit it.
            samples_per_pixel = getattr(series.pages[0], "samplesperpixel", 1) or 1
            is_rgb = samples_per_pixel >= 3 or ("S" in axes and "C" not in axes)

            datetime_str = extract_datetime_from_page(series.pages[0])

            self._qpi_meta_cache[tiff_series_idx] = parse_qpi_xml(
                per_page_xmls[0] if per_page_xmls else "",
                n_channels=n_channels,
                datetime_str=datetime_str,
                per_page_xmls=per_page_xmls if per_page_xmls else None,
                is_rgb=is_rgb,
            )
        return self._qpi_meta_cache[tiff_series_idx]

    # qptiff
    @property
    def qpi_metadata(self) -> QptiffMetadata:
        """
        Structured QPI metadata for the active scene.

        Returns the QptiffMetadata parsed from the PerkinElmer QPI XML of
        the active scene's tifffile series.  Returns an empty QptiffMetadata
        if the file contains no QPI XML.
        """
        tiff_series_idx = self._tiff_series_index(self.current_scene_index)
        if tiff_series_idx in self._qpi_meta_cache:
            return self._qpi_meta_cache[tiff_series_idx]

        with self._fs.open(self._path) as open_resource:
            with TiffFile(open_resource, is_mmstack=False) as tiff:
                series = tiff.series[tiff_series_idx]
                return self._get_or_parse_meta(tiff_series_idx, series)

    # qptiff
    @property
    def pyramid_level_count(self) -> int:
        """
        Number of resolution levels in the current scene's pyramid (1 = flat).

        For pyramidal QPTIFF files the FullResolution series typically has
        several levels (full res, half, quarter, …).  By default, only level 0
        (full resolution) is used.  Pass ``include_pyramid_levels=True`` to the
        Reader constructor to expose each level as its own scene instead.
        """
        tiff_series_idx = self._tiff_series_index(self.current_scene_index)
        with self._fs.open(self._path) as open_resource:
            with TiffFile(open_resource, is_mmstack=False) as tiff:
                return len(tiff.series[tiff_series_idx].levels)

    # qptiff
    @property
    def ome_metadata(self) -> "ome_types.model.OME":
        """
        OME-structured metadata for the current scene as an ``ome_types.OME``
        object.

        Channel, Plane, DetectorSettings, Instrument, Objective, Detector, and
        Filter objects are populated from the QPI XML wherever a direct OME-XML
        mapping exists. Vendor-specific fields with no OME equivalent are
        collected in a ``MapAnnotation`` (namespace ``qpi://vectra``) on
        ``OME.structured_annotations``.

        For non-QPTIFF files the Pixels element is populated from TIFF
        resolution tags alone and the channel list will be empty.
        """
        tiff_series_idx = self._tiff_series_index(self.current_scene_index)
        if tiff_series_idx in self._ome_cache:
            return self._ome_cache[tiff_series_idx]  # type: ignore[return-value]

        qpi = self.qpi_metadata
        level_idx = self._tiff_level_index(self.current_scene_index)

        with self._fs.open(self._path) as open_resource:
            with TiffFile(open_resource, is_mmstack=False) as tiff:
                level_series = tiff.series[tiff_series_idx].levels[level_idx]
                axes = getattr(level_series, "axes", "")
                shape = level_series.shape

        shape_map = dict(zip(axes.upper(), shape))
        ome = ome_metadata_from_qptiff(
            qpi=qpi,
            scene_name=self.scenes[self.current_scene_index],
            size_x=shape_map.get("X"),
            size_y=shape_map.get("Y"),
            size_z=shape_map.get("Z"),
            size_c=shape_map.get("C", shape_map.get("S")),
            size_t=shape_map.get("T"),
        )
        self._ome_cache[tiff_series_idx] = ome
        return ome

    # qptiff
    @property
    def xarray_dask_datatree_data(self) -> xr.DataTree:
        """
        Lazy, dask-backed multiscale pyramid for the active scene.

        Returns an ``xarray.DataTree`` with child nodes ``scale0``, ``scale1``,
        ... — one per pyramid level present in the underlying TIFF series.
        Each child wraps a single ``image`` data variable in a ``Dataset``
        with dims ``(c, y, x)`` (T and Z are squeezed when length 1), channel
        coords populated with per-channel QPTIFF metadata, and per-level
        ``scale_factors`` / ``pixel_size_um`` in its attrs.

        Non-pyramidal files return a single-node tree with just ``scale0``.
        """
        with self._fs.open(self._path) as open_resource:
            with TiffFile(open_resource, is_mmstack=False) as tiff:
                return self._build_multiscale_datatree(tiff, lazy=True)

    @property
    def xarray_datatree_data(self) -> xr.DataTree:
        """
        In-memory multiscale pyramid for the active scene.

        Same schema as :attr:`xarray_dask_datatree_data`, but each level is
        materialised as a numpy array at call time.
        """
        with self._fs.open(self._path) as open_resource:
            with TiffFile(open_resource, is_mmstack=False) as tiff:
                return self._build_multiscale_datatree(tiff, lazy=False)

    # qptiff
    @property
    def xarray_dask_scene_datatree(self) -> xr.DataTree:
        """
        Lazy DataTree whose top-level nodes are the image scenes of the file.

        Unlike :attr:`xarray_dask_datatree_data` (the *active* scene only, with
        ``scale0``/``scale1``/... at the root), this returns **every** QPI image
        scene (``FullResolution``, ``Label``, ``Macro``, ``Thumbnail``, ...) in a
        single tree, regardless of the ``include_aux_series`` flag::

            root
            ├── FullResolution/      (pyramidal -> scale sub-tree)
            │   ├── scale0   (image)
            │   ├── scale1
            │   └── ...
            ├── Label        (single image -> holds `image` directly)
            ├── Macro
            └── Thumbnail

        A pyramidal scene becomes an inner node with ``scale0..N`` children; a
        single-resolution scene is a leaf node holding the ``image`` data variable
        directly. Per-scene ``slide_info`` / ``image_info`` / ``processed`` live on
        the scene node's attrs; per-channel coords ride on each ``image``.
        """
        with self._fs.open(self._path) as open_resource:
            with TiffFile(open_resource, is_mmstack=False) as tiff:
                return self._build_scene_datatree(tiff, lazy=True)

    # qptiff
    @property
    def xarray_scene_datatree(self) -> xr.DataTree:
        """
        In-memory variant of :attr:`xarray_dask_scene_datatree` (each level is
        materialised as a numpy array at call time).
        """
        with self._fs.open(self._path) as open_resource:
            with TiffFile(open_resource, is_mmstack=False) as tiff:
                return self._build_scene_datatree(tiff, lazy=False)

    # qptiff
    def _build_scene_datatree(self, tiff: TiffFile, *, lazy: bool) -> xr.DataTree:
        """Assemble a scene-keyed DataTree from every image series in the file.

        Parameterised by series index (no ``current_scene_index`` mutation), so
        it reuses the per-scene array / coord / attr machinery for each series in
        turn. Pyramidal series get ``<scene>/scale0..N`` nodes; single series get
        a ``<scene>`` leaf holding ``image`` directly.
        """
        nodes: typing.Dict[str, xr.Dataset] = {}
        seen: typing.Dict[str, int] = {}
        slide_info: typing.Optional[SlideInfo] = None  # slide-scoped, root-level

        for series_idx, series in enumerate(tiff.series):
            is_qpi = valid_qpi_series(series.pages)
            if is_qpi:
                meta = self._get_or_parse_meta(series_idx, series)
                name = meta.image_type or f"Series_{series_idx}"
                channel_infos = meta.channels or None
                tiff_tags = {
                    code: tag.value for code, tag in series.pages[0].tags.items()
                }
                base_attrs = self._build_attrs(tiff_tags, meta, series)
                # SlideInfo is slide-constant; capture the FullResolution series'
                # copy (parsed first) as the canonical root-level slide metadata.
                if slide_info is None or name == FULL_RESOLUTION_TYPE:
                    slide_info = meta.slide
            else:
                meta = None
                name = generate_ome_image_id(series_idx)
                channel_infos = None
                base_attrs = {}

            # de-duplicate repeated image types (e.g. two Macro series)
            count = seen.get(name, 0)
            seen[name] = count + 1
            if count:
                name = f"{name}_{count}"

            dims = self._guess_tiff_dim_order(tiff, series_idx)

            pixel_size_yx_um: typing.Optional[typing.Tuple[float, float]] = None
            if meta is not None and meta.pixel_size_um is not None:
                pixel_size_yx_um = (float(meta.pixel_size_um), float(meta.pixel_size_um))

            n_levels = len(getattr(series, "levels", [series]))
            level_arrays: typing.List[xr.DataArray] = []
            channel_names_ref: typing.Optional[typing.List[str]] = None
            for level_idx in range(n_levels):
                if lazy:
                    data = self._create_dask_array_for_level(
                        tiff, dims, series_idx, level_idx
                    )
                else:
                    data = series.levels[level_idx].asarray()
                if channel_names_ref is None:
                    channel_names_ref = self._get_channel_names_for_scene(
                        data.shape, dims, channel_infos=channel_infos
                    )
                arr = squeeze_to_cyx(xr.DataArray(data, dims=dims))
                level_arrays.append(arr)

            sub = build_datatree_from_levels(
                levels_arrays=level_arrays,
                pixel_size_yx_um=pixel_size_yx_um,
                base_attrs=base_attrs,
                channel_infos=channel_infos,
                channel_names=channel_names_ref,
            )

            # Image-scoped metadata rides on the `image` DataArray itself (like
            # the per-channel coords already do), so a sliced-out image is
            # self-describing: `image_info` dict + the scene's
            # QptiffImageSceneMetadata under `processed`.
            image_info = dict(sub.attrs.get("image_info", {})) or None
            scene_meta = meta._primary if meta is not None else None

            if n_levels > 1:
                # pyramidal: scene becomes an inner node with scale children.
                nodes[name] = xr.Dataset(attrs=dict(sub.attrs))
                for child_name, child in sub.children.items():
                    cds = child.to_dataset()
                    _stamp_image_metadata(cds, image_info, scene_meta)
                    nodes[f"{name}/{child_name}"] = cds
            else:
                # single image: scene is a leaf holding `image` directly, with
                # the scene-level attrs merged onto it.
                ds = sub["scale0"].to_dataset()
                ds.attrs = {**dict(sub.attrs), **dict(ds.attrs)}
                _stamp_image_metadata(ds, image_info, scene_meta)
                nodes[name] = ds

            # Scope `processed` to this level: a scene node carries only its own
            # QptiffImageSceneMetadata (image_info + channels + scales), not the
            # whole QptiffMetadata. SlideInfo is slide-constant and lives on the
            # root node (below), so drop the per-scene slide_info copy.
            nodes[name].attrs.pop("slide_info", None)
            if scene_meta is not None:
                nodes[name].attrs[constants.METADATA_PROCESSED] = scene_meta

        dt = xr.DataTree.from_dict(nodes)
        # SlideInfo sits at the same level as the scenes — on the root node.
        if slide_info is not None:
            from dataclasses import asdict

            dt.attrs["slide_info"] = asdict(slide_info)
            dt.attrs[constants.METADATA_PROCESSED] = slide_info
        return dt

    def _build_multiscale_datatree(self, tiff: TiffFile, *, lazy: bool) -> xr.DataTree:
        tiff_series_idx = self._tiff_series_index(self.current_scene_index)
        series = tiff.series[tiff_series_idx]
        dims = self._get_dims_for_scene(tiff)

        channel_infos: typing.Optional[typing.List[ChannelInfo]] = None
        base_attrs: typing.Dict[str, typing.Any] = {}
        meta: typing.Optional[QptiffMetadata] = None
        if valid_qpi_series(series.pages):
            meta = self._get_or_parse_meta(tiff_series_idx, series)
            base_attrs = self._build_attrs(self._get_tiff_tags(tiff), meta, series)
            channel_infos = meta.channels or None
        else:
            base_attrs = {}

        px = self.physical_pixel_sizes
        pixel_size_yx_um: typing.Optional[typing.Tuple[float, float]] = None
        if px.Y is not None and px.X is not None:
            pixel_size_yx_um = (float(px.Y), float(px.X))

        n_levels = len(getattr(series, "levels", [series]))
        level_arrays: typing.List[xr.DataArray] = []
        channel_names_ref: typing.Optional[typing.List[str]] = None

        for level_idx in range(n_levels):
            level_series = series.levels[level_idx]
            if lazy:
                data = self._create_dask_array_for_level(
                    tiff, dims, tiff_series_idx, level_idx
                )
            else:
                data = level_series.asarray()

            if channel_names_ref is None:
                channel_names_ref = self._get_channel_names_for_scene(
                    data.shape, dims, channel_infos=channel_infos
                )

            arr = xr.DataArray(data, dims=dims)
            arr = squeeze_to_cyx(arr)
            level_arrays.append(arr)

        dt = build_datatree_from_levels(
            levels_arrays=level_arrays,
            pixel_size_yx_um=pixel_size_yx_um,
            base_attrs=base_attrs,
            channel_infos=channel_infos,
            channel_names=channel_names_ref,
        )

        # Retain the parsed QptiffMetadata object on the root attrs for parity
        # with the single DataArray path (so attrs[METADATA_PROCESSED] is the
        # same object on both). build_datatree_from_levels drops it because
        # _sanitise_attrs keeps only JSON-safe types for a clean serialisable
        # tree; re-attach it here. The zarr writer builds its own .zattrs from
        # the OME object and never serialises the root attrs, so this does not
        # affect write_ome_zarr output. (A direct dt.to_zarr() would need it
        # stripped, but the supported write paths do not call that.)
        if meta is not None:
            dt.attrs[constants.METADATA_PROCESSED] = meta
        return dt

    # qptiff
    def _build_attrs(
        self,
        tiff_tags: typing.Dict[int, typing.Any],
        meta: QptiffMetadata,
        series: typing.Any = None,
    ) -> typing.Dict[str, typing.Any]:
        # extract XPosition / YPosition from TIFF tags (tag 286 / 287) and
        # convert to um using the same scalar table as pixel size.
        _TIFF_XPOS = 286
        _TIFF_YPOS = 287
        _TIFF_RESUNIT = 296
        if _TIFF_XPOS in tiff_tags and _TIFF_YPOS in tiff_tags:
            res_unit = tiff_tags.get(_TIFF_RESUNIT)
            scalar = _NAME_TO_MICRONS.get(res_unit, 1.0)
            xpos = tiff_tags[_TIFF_XPOS]
            ypos = tiff_tags[_TIFF_YPOS]
            img_info = meta._primary.image_info
            if isinstance(xpos, tuple) and len(xpos) == 2 and xpos[1]:
                img_info.xposition_um = scalar * xpos[0] / xpos[1]
            if isinstance(ypos, tuple) and len(ypos) == 2 and ypos[1]:
                img_info.yposition_um = scalar * ypos[0] / ypos[1]

        # stored pixel precision from TIFF tag 258 BitsPerSample. RGB stores a
        # (8, 8, 8) tuple, single-sample an int; the samples share one depth so
        # we keep the first. Distinct from camera.bit_depth (the ADC depth).
        _TIFF_BITSPERSAMPLE = 258
        bps = tiff_tags.get(_TIFF_BITSPERSAMPLE)
        if isinstance(bps, (tuple, list)) and bps:
            bps = bps[0]
        if isinstance(bps, int):
            meta._primary.image_info.stored_bits_per_sample = int(bps)

        # scale_factor fallback for scenes whose QPI XML lacks <PixelSizeMicrons>
        # (Label / Macro / Overview / Thumbnail): every TIFF page still defines a
        # pixel->physical scale via XResolution (tag 282) + ResolutionUnit (296).
        # XML keeps precedence (FullResolution already set in parse_qpi_xml).
        img_info = meta._primary.image_info
        if img_info.scale_factor is None:
            xres = tiff_tags.get(282)  # (numerator, denominator): pixels per unit
            resunit = tiff_tags.get(296)
            if (
                isinstance(xres, tuple)
                and len(xres) == 2
                and xres[0]
                and resunit not in (None, tifffile.RESUNIT.NONE)
            ):
                scalar = _NAME_TO_MICRONS.get(resunit)
                if scalar:
                    px_um = scalar * xres[1] / xres[0]
                    if px_um > 0:
                        img_info.scale_factor = px_um
                        img_info.scale_factor_unit = "um"

        attrs = qptiff_meta_to_root_attrs(meta)

        if series is not None:
            n_levels = len(getattr(series, "levels", [series]))
            attrs["pyramid_level_count"] = n_levels

        # bioio-base Reader.metadata looks for these keys in xarray_dask_data.attrs.
        # `unprocessed` holds the raw TIFF tags EXCEPT the bulky, non-metadata
        # ones (see _UNPROCESSED_TAG_EXCLUDE): the ImageDescription XML is fully
        # parsed into the structured schema and kept on `meta.raw_xml`, and the
        # Strip/Tile offset arrays are just file byte locations. Dropping them
        # avoids duplicating tens of KB per scene (and per pyramid level once
        # promoted onto the DataTree). Read the XML back via
        # `reader.qpi_metadata.raw_xml` if needed.
        attrs[constants.METADATA_UNPROCESSED] = {
            code: value
            for code, value in tiff_tags.items()
            if code not in _UNPROCESSED_TAG_EXCLUDE
        }
        attrs[constants.METADATA_PROCESSED] = meta

        return attrs

    # qptiff
    def write_ome_zarr(
        self,
        store: typing.Union[str, "typing.MutableMapping[str, typing.Any]"],
        *,
        overwrite: bool = False,
        ome_only: typing.Optional[bool] = None,
        validate: bool = True,
        **kwargs: typing.Any,
    ) -> "typing.Any":
        """
        Write the current scene's multiscale pyramid to an OME-NGFF v0.5 zarr store.

        Convenience wrapper around :func:`~bioio_tifffile.qptiff_zarr.write_ome_zarr`.

        Parameters
        ----------
        store:
            Output path (str) or zarr MutableMapping.
        overwrite:
            Replace an existing store if True. Default: False.
        ome_only:
            Omit QPI vendor fields from the output. Defaults to the reader's
            ``ome_metadata`` constructor flag.
        validate:
            Validate the written store against OME-NGFF v0.5 spec. Default: True.
        **kwargs:
            Forwarded to :func:`~bioio_tifffile.qptiff_zarr.write_ome_zarr`
            (e.g. ``chunk_shape``, ``shard_shape``, ``compressor``,
            ``zarr_format``).

        Returns
        -------
        zarr.Group
            Root group of the written store.
        """
        return _write_ome_zarr(
            self.xarray_dask_datatree_data,
            self.ome_metadata,
            store,
            overwrite=overwrite,
            ome_only=ome_only if ome_only is not None else self._ome_only,
            validate=validate,
            **kwargs,
        )


def _stamp_image_metadata(
    ds: xr.Dataset,
    image_info: typing.Optional[typing.Dict[str, typing.Any]],
    scene_meta: typing.Any,  # QptiffImageSceneMetadata (meta._primary)
) -> None:
    """Attach image-scoped metadata onto a dataset's ``image`` DataArray attrs.

    Keeps the image self-describing when sliced out of the scene DataTree: the
    ``image_info`` dict and the scene's ``QptiffImageSceneMetadata`` ride on the
    array alongside the per-channel coords already present.
    """
    if "image" not in ds.data_vars:
        return
    img = ds["image"]
    if image_info is not None:
        img.attrs["image_info"] = image_info
    if scene_meta is not None:
        img.attrs[constants.METADATA_PROCESSED] = scene_meta


_NAME_TO_MICRONS = {
    "pm": 1e-6,
    "picometer": 1e-6,
    "nm": 1e-3,
    "nanometer": 1e-3,
    "micron": 1,
    "um": 1,
    "um": 1,
    "\u00b5m": 1,  # µm as unicode literal → treat as um
    tifffile.RESUNIT.NONE: 1,
    tifffile.RESUNIT.MICROMETER: 1,
    None: 1,
    "mm": 1e3,
    "millimeter": 1e3,
    tifffile.RESUNIT.MILLIMETER: 1e3,
    "cm": 1e4,
    "centimeter": 1e4,
    tifffile.RESUNIT.CENTIMETER: 1e4,
    "cal": 2.54 * 1e4,
    tifffile.RESUNIT.INCH: 2.54 * 1e4,
}


def _get_pixel_size(
    path_or_file: typing.Any, series_index: int
) -> typing.Tuple[
    typing.Optional[float], typing.Optional[float], typing.Optional[float]
]:
    """Return the pixel size in microns (z,y,x) for the given series in a tiff path."""

    with TiffFile(path_or_file, is_mmstack=False) as tiff:
        tags = tiff.series[series_index].pages[0].tags

    if tiff.is_imagej:
        unit = tiff.imagej_metadata["unit"]
        z_size = tiff.imagej_metadata.get("spacing", None)
    else:
        unit = tags["ResolutionUnit"].value
        z_size = None

    scalar = _NAME_TO_MICRONS.get(unit, 1)

    # Resolution tags are two LONGs: representing a fraction
    # "The number of pixels per ResolutionUnit"
    x_npix, x_res_units = tags["XResolution"].value
    y_npix, y_res_units = tags["YResolution"].value
    # the inverse of the fraction is the size of a pixel
    x_size = scalar * x_res_units / x_npix
    y_size = scalar * y_res_units / y_npix
    if z_size is not None:
        z_size *= scalar

    return z_size, y_size, x_size
