#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OME transport layer.

Serialisation only: QPTIFF metadata in, OME-XML / OME-Zarr out.
:class:`~bioio_tifffile.qptiff_types.QptiffMetadata` is the source of truth —
this package is a leaf, and nothing outside it should import ``ome_types`` or
``ome_zarr_models``. The xarray/DataTree path deliberately does not depend on
any of this.
"""

from .convert import qptiff_to_ome_zarr
from .model import (
    VENDOR_CHANNEL_KEYS,
    VENDOR_IMAGE_KEYS,
    ome_metadata_from_qptiff,
)
from .zarr_writer import IMAGE_GROUP, OME_GROUP, OME_XML_NAME, write_ome_zarr

__all__ = [
    "IMAGE_GROUP",
    "OME_GROUP",
    "OME_XML_NAME",
    "VENDOR_CHANNEL_KEYS",
    "VENDOR_IMAGE_KEYS",
    "ome_metadata_from_qptiff",
    "qptiff_to_ome_zarr",
    "write_ome_zarr",
]
