#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List

import bioio_base.reader_metadata


class ReaderMetadata(bioio_base.reader_metadata.ReaderMetadata):
    """
    Metadata about the bioio-tifffile reader plugin itself
    (not the image being read).
    """

    @staticmethod
    def get_supported_extensions() -> List[str]:
        """Return file extensions this plugin supports."""
        return ["tif", "tiff", "lsm", "qptiff"]

    @staticmethod
    def get_reader() -> bioio_base.reader.Reader:
        """Return the Reader class for this plugin."""
        from .reader import Reader

        return Reader
