#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
metadata split into three main components;
  qptiff_types.py: dataclasses and FORMAT constants
  qptiff_parser.py: tiff series/pages/xml parsing
  ome/model.py:  OME conversion (transport layer)
"""

from .multiscale import qptiff_meta_to_root_attrs
from .qptiff_parser import (
    TIFF_DATETIME_TAG,
    TIFF_IMAGE_DESCRIPTION_TAG,
    extract_datetime_from_page,
    parse_qpi_xml,
    valid_qpi_series,
)
from .qptiff_types import (
    FORMAT_BRIGHTFIELD,
    FORMAT_FUSION_PAGED,
    FORMAT_POLARIS_SCANBAND,
    FORMAT_UNKNOWN,
    LOCUS_PER_PAGE_ROOT,
    LOCUS_RGB_SAMPLES,
    LOCUS_SHARED_SCANBANDS,
    LOCUS_UNKNOWN,
    CameraInfo,
    ChannelInfo,
    ImageInfo,
    QptiffImageSceneMetadata,
    QptiffMetadata,
    ScaleInfo,
    ScanResolutionInfo,
    SlideInfo,
)

__all__ = [
    "FORMAT_BRIGHTFIELD",
    "FORMAT_FUSION_PAGED",
    "FORMAT_POLARIS_SCANBAND",
    "FORMAT_UNKNOWN",
    "LOCUS_PER_PAGE_ROOT",
    "LOCUS_RGB_SAMPLES",
    "LOCUS_SHARED_SCANBANDS",
    "LOCUS_UNKNOWN",
    "TIFF_DATETIME_TAG",
    "TIFF_IMAGE_DESCRIPTION_TAG",
    "CameraInfo",
    "ChannelInfo",
    "ImageInfo",
    "QptiffImageSceneMetadata",
    "QptiffMetadata",
    "ScaleInfo",
    "ScanResolutionInfo",
    "SlideInfo",
    "extract_datetime_from_page",
    "valid_qpi_series",
    "ome_metadata_from_qptiff",
    "qptiff_meta_to_root_attrs",
    "parse_qpi_xml",
]


def __getattr__(name: str) -> object:
    """Lazily serve the OME transport layer so importing this shim (and hence
    the package) does not drag in ome_types."""
    if name == "ome_metadata_from_qptiff":
        from .ome.model import ome_metadata_from_qptiff

        return ome_metadata_from_qptiff
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
