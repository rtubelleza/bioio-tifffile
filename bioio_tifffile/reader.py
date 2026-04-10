#!/usr/bin/env python
# -*- coding: utf-8 -*-


import logging
import typing
import warnings

import dask.array as da
import numpy as np
import tifffile
import xarray as xr
from bioio_base import constants, dimensions, exceptions, io, reader, types
from dask import delayed
from fsspec.spec import AbstractFileSystem
from tifffile import TiffFile, imread
from tifffile.tifffile import TiffTags

from .qptiff_metadata import (
    QptiffMetadata,
    extract_datetime_from_page,
    extract_qpi_xml_from_page,
    parse_qpi_xml,
)
from .utils import generate_ome_channel_id, generate_ome_image_id

###############################################################################

# "Q" is used by tifffile to say "unknown dimension"
# "I" is used to mean a generic image sequence
UNKNOWN_DIM_CHARS = ["Q", "I"]
TIFF_IMAGE_DESCRIPTION_TAG_INDEX = 270

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
    channel_names: Optional[Union[List[str], List[List[str]]]]
        A list of string channel names to be applied to all array(s) or a
        list of lists of string channel names to be mapped onto the list of arrays
        provided to image.
        Default: None (create OME channel IDs for names for single or multiple arrays)
    fs_kwargs: Dict[str, Any]
        Any specific keyword arguments to pass down to the fsspec created filesystem.
        Default: {}
    """

    _scenes: typing.Optional[typing.Tuple[str, ...]] = None
    _physical_pixel_sizes: typing.Optional[types.PhysicalPixelSizes] = None

    # qptiff
    # qptiff: maps scene index → tifffile series index (thumbnails, labels, etc.)
    _scene_series_map: typing.Optional[typing.Dict[int, int]] = None
    # qptiff: parsed XML metadata per tifffile series index
    _qpi_meta_cache: typing.Dict[int, QptiffMetadata]

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
        channel_names: typing.Optional[
            typing.Union[typing.List[str], typing.List[typing.List[str]]]
        ] = None,
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

        # Store params
        if isinstance(chunk_dims, str):
            chunk_dims = list(chunk_dims)

        # Run basic checks on dims and channel names
        if isinstance(dim_order, list):
            if len(dim_order) != len(self.scenes):
                raise exceptions.ConflictingArgumentsError(
                    f"Number of dimension strings provided does not match the "
                    f"number of scenes found in the file. "
                    f"Number of scenes: {len(self.scenes)}, "
                    f"Number of provided dimension order strings: {len(dim_order)}"
                )

        # If provided a list
        if isinstance(channel_names, list):
            # If provided a list of lists
            if len(channel_names) > 0 and isinstance(channel_names[0], list):
                # Ensure that the outer list is the number of scenes
                if len(channel_names) != len(self.scenes):
                    raise exceptions.ConflictingArgumentsError(
                        f"Number of channel name lists provided does not match the "
                        f"number of scenes found in the file. "
                        f"Number of scenes: {len(self.scenes)}, "
                        f"Provided channel name lists: {dim_order}"
                    )

        self.chunk_dims = chunk_dims
        self._dim_order = dim_order
        self._channel_names = channel_names

        # qptiff
        if "include_aux_series" in kwargs:
            self._include_aux_series = kwargs["include_aux_series"]
        else:
            self._include_aux_series = False
        self._qpi_meta_cache = {}

        # Enforce valid image
        self._is_supported_image(self._fs, self._path)

    @property
    def scenes(self) -> typing.Tuple[str, ...]:
        if self._scenes is None:
            with self._fs.open(self._path) as open_resource:
                with TiffFile(open_resource, is_mmstack=False) as tiff:
                    self._scenes, self._scene_series_map = self._build_scene_map(tiff)
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
            if meta.pixel_size_um is not None:
                px = meta.pixel_size_um
                self._physical_pixel_sizes = types.PhysicalPixelSizes(None, px, px)
            else:
                with self._fs.open(self._path) as open_resource:
                    try:
                        z, y, x = _get_pixel_size(open_resource, tiff_series_idx)
                    except Exception as exc:
                        warnings.warn(f"Could not parse QPTIFF pixel size: {exc}")
                        z, y, x = None, None, None
                self._physical_pixel_sizes = types.PhysicalPixelSizes(z, y, x)
        return self._physical_pixel_sizes

    @staticmethod
    def _get_image_data(
        fs: AbstractFileSystem,
        path: str,
        scene: int,
        retrieve_indices: typing.Tuple[typing.Union[int, slice]],
        transpose_indices: typing.List[int],
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
                level=0,
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

    def _guess_tiff_dim_order(self, tiff: TiffFile) -> typing.List[str]:
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
        self, image_shape: typing.Tuple[int, ...], dims: typing.List[str]
    ) -> typing.Optional[typing.List[str]]:
        # Fast return in None case
        if self._channel_names is not None:
            # If channels was provided as a list of lists
            if isinstance(self._channel_names[0], list):
                scene_channels = self._channel_names[self.current_scene_index]
            elif all(isinstance(c, str) for c in self._channel_names):
                scene_channels = self._channel_names  # type: ignore
            else:
                return None

            # If scene channels isn't None and no channel dimension raise error
            if dimensions.DimensionNames.Channel not in dims:
                raise exceptions.ConflictingArgumentsError(
                    f"Provided channel names for scene with no channel dimension. "
                    f"Scene dims: {dims}, "
                    f"Provided channel names: {scene_channels}"
                )

            # If scene channels isn't the same length as the size of channel dim
            if (
                len(scene_channels)
                != image_shape[dims.index(dimensions.DimensionNames.Channel)]
            ):
                raise exceptions.ConflictingArgumentsError(
                    f"Number of channel names provided does not match the "
                    f"size of the channel dimension for this scene. "
                    f"Scene shape: {image_shape}, "
                    f"Dims: {dims}, "
                    f"Provided channel names: {self._channel_names}",
                )

            return scene_channels  # type: ignore

        # derive channel names from qptiff metadata if not user-provided
        meta = self.qpi_metadata
        if not meta.channels:
            return None

        ch_dim = dimensions.DimensionNames.Channel
        samples_dim = "S"  # qptiff; brightfield rgb uses S samples not C

        if ch_dim in dims:
            n = image_shape[dims.index(ch_dim)]
            names = meta.channel_names
            if len(names) >= n:
                return names[:n]
            return names + [f"Channel_{i}" for i in range(len(names), n)]

        if samples_dim in dims:
            n = image_shape[dims.index(samples_dim)]
            names = meta.channel_names
            if len(names) == n:
                return names

        return None

    @staticmethod
    def _get_coords(
        dims: typing.List[str],
        shape: typing.Tuple[int, ...],
        scene_index: int,
        channel_names: typing.Optional[typing.List[str]],
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

        if channel_names is None:
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
        Creates a delayed dask array for the file.
        Parameters
        ----------
        tiff: TiffFile
            An open TiffFile for processing.
        selected_scene_dims_list: List[str]
            The dimensions to use for constructing the array with.
            Required for managing chunked vs non-chunked dimensions.
        Returns
        -------
        image_data: da.Array
            The fully constructed and fully delayed image as a Dask Array object.
        """

        # Always add the plane dimensions if not present already
        for dim in dimensions.REQUIRED_CHUNK_DIMS:
            if dim not in self.chunk_dims:
                self.chunk_dims.append(dim)

        # Safety measure / "feature"
        self.chunk_dims = [d.upper() for d in self.chunk_dims]

        # Construct delayed dask array for the current remapped scene index
        tiff_series_idx = self._tiff_series_index(self.current_scene_index)
        selected_scene = tiff.series[tiff_series_idx]
        selected_scene_dims = "".join(selected_scene_dims_list)

        # Raise invalid dims error
        if len(selected_scene.shape) != len(selected_scene_dims):
            raise exceptions.ConflictingArgumentsError(
                f"Dimension string provided does not match the "
                f"number of dimensions found for this scene. "
                f"This scene shape: {selected_scene.shape}, "
                f"Provided dims string: {selected_scene_dims}"
            )

        # Constuct the chunk and non-chunk shapes one dim at a time
        # We also collect the chunk and non-chunk dimension order so that
        # we can swap the dimensions after we block out the array
        non_chunk_dim_order: typing.List[str] = []
        non_chunk_shape: typing.List[int] = []
        chunk_dim_order: typing.List[str] = []
        chunk_shape: typing.List[int] = []

        for dim, size in zip(selected_scene_dims, selected_scene.shape):
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
                ),
                shape=chunk_shape,
                dtype=selected_scene.dtype,
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
                # Get dims from provided or guess
                dims = self._get_dims_for_scene(tiff)

                # Create the delayed dask array
                image_data = self._create_dask_array(tiff, dims)

                # Get unprocessed metadata from tags
                tiff_tags = self._get_tiff_tags(tiff)

                # Get channel names for this scene or generate
                channels = self._get_channel_names_for_scene(image_data.shape, dims)

                # Create coords
                coords = self._get_coords(
                    dims,
                    image_data.shape,
                    scene_index=self.current_scene_index,
                    channel_names=channels,
                )

                # Try accepted processed metadata
                # qptiff: if qpi xml present, rich attrs;
                tiff_series_idx = self._tiff_series_index(self.current_scene_index)
                series = tiff.series[tiff_series_idx]
                xml = extract_qpi_xml_from_page(series.pages[0])

                if xml:
                    meta = self._get_or_parse_meta(tiff_series_idx, xml, series)
                    attrs = self._build_attrs(tiff_tags, meta)
                else:  # default non-qpi
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
                # Get dims from provided or guess
                dims = self._get_dims_for_scene(tiff)

                # Read image into memory
                tiff_series_idx = self._tiff_series_index(self.current_scene_index)
                image_data = tiff.series[tiff_series_idx].asarray()

                # Get unprocessed metadata from tags
                tiff_tags = self._get_tiff_tags(tiff)

                # Get channel names for this scene or generate
                channels = self._get_channel_names_for_scene(image_data.shape, dims)

                # Create dims and coords
                coords = self._get_coords(
                    dims,
                    image_data.shape,
                    scene_index=self.current_scene_index,
                    channel_names=channels,
                )

                # Try accepted processed metadata
                # qptiff: if qpi xml present, rich attrs;
                tiff_series_idx = self._tiff_series_index(self.current_scene_index)
                series = tiff.series[tiff_series_idx]
                xml = extract_qpi_xml_from_page(series.pages[0])

                if xml:
                    meta = self._get_or_parse_meta(tiff_series_idx, xml, series)
                    attrs = self._build_attrs(tiff_tags, meta)
                else:
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
    def _build_scene_map(
        self, tiff: TiffFile
    ) -> typing.Tuple[typing.Tuple[str, ...], typing.Dict[int, int]]:
        """
        Build scene names and a mapping {scene_index: tifffile_series_index}.

        Mainly for qptiff images, where multiple tiff files are stored as
        tiff series elements. These usually correspond to thumbnails, overviews,
        labels, and the full image. This function orders the full res image to
        the 0th index in the mapping. Aux images follow in the original order.

        Parameters
        ----------
        tiff: TiffFile
            The opened tifffile.TiffFile object.

        Returns
        ----------
        Tuple:
            scene_names: Tuple[str, ...]
                Ordered tuple of human readable scene names.
            scene_series_map: Dict[[int, int]]
                Map of each scene index to a tiff file series index.
        """
        full_res: typing.List[typing.Tuple[int, str]] = []  # Full resolution image
        aux: typing.List[
            typing.Tuple[int, str]
        ] = []  # Auxiliary images; Thumbnails, Labels, etc

        for tiff_idx, series in enumerate(tiff.series):
            xml = extract_qpi_xml_from_page(series.pages[0])
            if xml:
                meta = self._get_or_parse_meta(tiff_idx, xml, series)
                image_type = meta.image_type or f"Series_{tiff_idx}"
                if image_type == FULL_RESOLUTION_TYPE:
                    full_res.append((tiff_idx, image_type))
                else:
                    aux.append((tiff_idx, image_type))
            else:  # fallback: non-qpi tiff, treat as full resolution scene
                full_res.append((tiff_idx, generate_ome_image_id(tiff_idx)))

        candidates = full_res + (aux if self._include_aux_series else [])
        if not candidates:  # fall back to defaults
            candidates = [(0, generate_ome_image_id(0))]

        scene_names: typing.List[str] = []
        scene_series_map: typing.Dict[int, int] = {}
        seen: typing.Dict[str, int] = {}  # in case of duplicate tags

        for tiff_idx, name in candidates:
            # dedupe names, ie FullResolution -> FullResolution_1
            count = seen.get(name, 0)
            seen[name] = count + 1
            unique = name if count == 0 else f"{name}_{count}"
            scene_idx = len(scene_names)
            scene_names.append(unique)
            scene_series_map[scene_idx] = tiff_idx

        return tuple(scene_names), scene_series_map

    # qptiff
    def _tiff_series_index(self, scene_idx: int) -> int:
        """Map scene index to the underlying tifffile series index."""
        if self._scene_series_map is None:
            _ = self.scenes
        assert self._scene_series_map is not None
        return self._scene_series_map.get(scene_idx, scene_idx)

    # qptiff
    def _get_or_parse_meta(
        self,
        tiff_series_idx: int,
        xml: str,
        series: typing.Any,
    ) -> QptiffMetadata:
        if tiff_series_idx not in self._qpi_meta_cache:
            axes = getattr(series.pages, "axes", "")
            shape = series.shape
            n_channels: typing.Optional[int] = None
            if "C" in axes:
                n_channels = shape[axes.index("C")]
            elif "S" in axes:
                n_channels = shape[axes.index("S")]
            datetime_str = extract_datetime_from_page(series.pages[0])

            # Collect per-page XMLs for newer Fusion files where biomarker
            # names are stored one-per-page rather than in ScanBands-i.
            per_page_xmls: typing.List[str] = []
            try:
                for page in series.pages:
                    per_page_xmls.append(extract_qpi_xml_from_page(page))
            except Exception:
                per_page_xmls = []

            self._qpi_meta_cache[tiff_series_idx] = parse_qpi_xml(
                xml,
                n_channels=n_channels,
                datetime_str=datetime_str,
                per_page_xmls=per_page_xmls if per_page_xmls else None,
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
                xml = extract_qpi_xml_from_page(series.pages[0])
                return self._get_or_parse_meta(tiff_series_idx, xml, series)

    # qptiff
    def _build_attrs(
        self,
        tiff_tags: typing.Dict[int, typing.Any],
        meta: QptiffMetadata,
    ) -> typing.Dict[str, typing.Any]:
        attrs: typing.Dict[str, typing.Any] = {
            constants.METADATA_UNPROCESSED: tiff_tags,
        }
        if meta.raw_xml:
            attrs[constants.METADATA_PROCESSED] = meta.raw_xml
        attrs.update(meta.to_dict())
        return attrs


_NAME_TO_MICRONS = {
    "pm": 1e-6,
    "picometer": 1e-6,
    "nm": 1e-3,
    "nanometer": 1e-3,
    "micron": 1,
    "µm": 1,
    "um": 1,
    "\\u00B5m": 1,  # µm unicode
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
