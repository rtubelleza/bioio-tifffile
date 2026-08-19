#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
metadata split into three main components;
  qptiff_types.py: dataclasses and FORMAT constants
  qptiff_parser.py: tiff series/pages/xml parsing
  qptiff_ome.py: OME conversion and xarray extraction helpers
"""

from .qptiff_ome import ome_metadata_from_qptiff, ome_to_channel_coords, ome_to_flat_attrs, qptiff_meta_to_root_attrs
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
    "ome_to_channel_coords",
    "ome_to_flat_attrs",
    "qptiff_meta_to_root_attrs",
    "parse_qpi_xml",
    "_format_binning",
]
