# -*- coding: utf-8 -*-

"""Top-level package for bioio_tifffile."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("bioio-tifffile")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Sean Meharry"
__email__ = "seanm@alleninstitute.org"

from .qptiff_metadata import (
    FORMAT_BRIGHTFIELD,
    FORMAT_FUSION_PAGED,
    FORMAT_POLARIS_SCANBAND,
    FORMAT_UNKNOWN,
    ChannelInfo,
    OMEChannel,
    OMEInstrument,
    OMEMetadata,
    QptiffMetadata,
    ScanResolutionInfo,
    ome_metadata_from_qptiff,
)
from .reader import Reader
from .reader_metadata import ReaderMetadata

__all__ = [
    "Reader",
    "ReaderMetadata",
    "QptiffMetadata",
    "ChannelInfo",
    "ScanResolutionInfo",
    "OMEMetadata",
    "OMEChannel",
    "OMEInstrument",
    "ome_metadata_from_qptiff",
    "FORMAT_BRIGHTFIELD",
    "FORMAT_FUSION_PAGED",
    "FORMAT_POLARIS_SCANBAND",
    "FORMAT_UNKNOWN",
]
