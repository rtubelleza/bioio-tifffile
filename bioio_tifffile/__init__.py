# -*- coding: utf-8 -*-

"""Top-level package for bioio_tifffile."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("bioio-tifffile")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Sean Meharry"
__email__ = "seanm@alleninstitute.org"

from .multiscale import (
    build_datatree_from_levels,
    compute_scale_attrs,
    squeeze_to_cyx,
)
from .qptiff_metadata import (
    FORMAT_BRIGHTFIELD,
    FORMAT_FUSION_PAGED,
    FORMAT_POLARIS_SCANBAND,
    FORMAT_UNKNOWN,
    ChannelInfo,
    QptiffMetadata,
    ScanResolutionInfo,
    ome_metadata_from_qptiff,
    ome_to_channel_coords,
    ome_to_flat_attrs,
)
from .reader import Reader
from .reader_metadata import ReaderMetadata

__all__ = [
    "Reader",
    "ReaderMetadata",
    "QptiffMetadata",
    "ChannelInfo",
    "ScanResolutionInfo",
    "ome_metadata_from_qptiff",
    "ome_to_channel_coords",
    "ome_to_flat_attrs",
    "build_datatree_from_levels",
    "compute_scale_attrs",
    "squeeze_to_cyx",
    "FORMAT_BRIGHTFIELD",
    "FORMAT_FUSION_PAGED",
    "FORMAT_POLARIS_SCANBAND",
    "FORMAT_UNKNOWN",
]
