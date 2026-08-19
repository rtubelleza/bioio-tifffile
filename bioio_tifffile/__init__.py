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
    LOCUS_PER_PAGE_ROOT,
    LOCUS_RGB_SAMPLES,
    LOCUS_SHARED_SCANBANDS,
    LOCUS_UNKNOWN,
    ChannelInfo,
    QptiffMetadata,
    ScanResolutionInfo,
)
from .reader import Reader
from .reader_metadata import ReaderMetadata

#: Names served lazily from the OME transport layer. Importing
#: ``bioio_tifffile.ome`` pulls in ome_types (200+ modules), so the parse and
#: xarray paths must not pay for it just by importing the package. Touching any
#: of these names loads the layer on demand.
_OME_LAZY = {
    "ome_metadata_from_qptiff": "model",
    "qptiff_to_ome_zarr": "convert",
    "write_ome_zarr": "zarr_writer",
}


def __getattr__(name: str) -> object:
    """PEP 562 lazy access to the OME transport layer."""
    module = _OME_LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f".ome.{module}", __name__), name)


def __dir__() -> list:
    return sorted(set(globals()) | set(_OME_LAZY))

__all__ = [
    "Reader",
    "ReaderMetadata",
    "qptiff_to_ome_zarr",
    "write_ome_zarr",
    "QptiffMetadata",
    "ChannelInfo",
    "ScanResolutionInfo",
    "ome_metadata_from_qptiff",
    "build_datatree_from_levels",
    "compute_scale_attrs",
    "squeeze_to_cyx",
    "FORMAT_BRIGHTFIELD",
    "FORMAT_FUSION_PAGED",
    "FORMAT_POLARIS_SCANBAND",
    "FORMAT_UNKNOWN",
    "LOCUS_PER_PAGE_ROOT",
    "LOCUS_RGB_SAMPLES",
    "LOCUS_SHARED_SCANBANDS",
    "LOCUS_UNKNOWN",
]
