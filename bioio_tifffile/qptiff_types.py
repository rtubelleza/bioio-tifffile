#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PerkinElmer QPTIFF metadata dataclasses and schema.

Provides intermediate representations as dataclasses. Filled in by parsers,
used by ome converters/main reader. 

Organised to resemble ome-ngff hierarchies:
    QptiffMetadata: OME root
      SlideInfo: Experimenter + Instrument  (slide-constant)
      QptiffImageSceneMetadata[]: Image[]
        ImageInfo:Image / ObjectiveSettings
          CameraInfo:Detector
          ScanResolutionInfo:Objective + Pixels.PhysicalSize*
        ChannelInfo[]: Channel[] + Plane[]
        ScaleInfo[]:multiscales[].datasets[]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

#: H&E / IHC brightfield scan. scan_mode contains "Brightfield" or BFLampType
#: is present. Channels are RGB samples stored in the S dimension.
FORMAT_BRIGHTFIELD = "brightfield"

#: Older Vectra / Polaris / OPAL format. Channel metadata lives in
#: <ScanBands-i> elements inside an XML <ScanProfile>.
FORMAT_POLARIS_SCANBAND = "polaris_scanband"

#: Newer Akoya Biosciences / Fusion 1.x format. ScanProfile is a JSON blob;
#: each TIFF page carries its own <Biomarker> / <ExposureTime> in tag 270.
FORMAT_FUSION_PAGED = "fusion_paged"

#: No recognised structural signals. Parser will try strategies in order.
FORMAT_UNKNOWN = "unknown"


@dataclass
class ScanResolutionInfo:
    """Objective/scan settings that apply to the whole scene.

    ``base_pixel_size_um`` is the physical pixel size of the full resolution page (scale0).
    """

    magnification: Optional[float] = None
    objective_name: Optional[str] = None
    binning: Optional[int] = None
    base_pixel_size_um: Optional[float] = None


@dataclass
class CameraInfo:
    """Camera / detector settings for the whole scene.

    Fusion 1.x embeds ``<CameraSettings>`` in every page XML; in practice the
    values are identical across channels, so these live at the image scope.

    If a file does genuinely vary per channel the parser warns and keeps the
    varying copy on :class:`ChannelInfo`.
    """

    camera_name: Optional[str] = None
    camera_type: Optional[str] = None
    gain: Optional[float] = None
    bit_depth: Optional[int] = None
    binning: Optional[int] = None


@dataclass
class SlideInfo:
    """Physical-slide identity. Usually constant across every image on this slide."""

    slide_id: Optional[str] = None
    barcode: Optional[str] = None
    study_name: Optional[str] = None
    operator_name: Optional[str] = None
    computer_name: Optional[str] = None
    datetime: Optional[str] = None  # TIFF tag 306
    acquisition_software: Optional[str] = None
    description_version: Optional[str] = None
    instrument_type: Optional[str] = None
    identifier: Optional[str] = None  # slide-level UUID


@dataclass
class ImageInfo:
    """Metadata for a given image scene.

    For qptiffs;
        - FullResolution
        - Thumbnail
        - Macro
        - Label

    One ``ImageInfo`` per distinct image (FullResolution / Label / Macro /
    Thumbnail).
    """

    image_type: Optional[str] = None  # FullResolution / Thumbnail / Macro / Label
    # optics / acquisition
    objective: Optional[str] = None # potentiall SlideInfo
    bf_lamp_type: Optional[str] = None
    scan_profile_name: Optional[str] = None
    scan_mode: Optional[str] = None
    is_tma: Optional[bool] = None # qptiff specific; most probably a qpi metadata
    opal_kit_type: Optional[str] = None
    # stage position of this image
    xposition_um: Optional[float] = None
    yposition_um: Optional[float] = None
    # nested per-image blocks
    camera: CameraInfo = field(default_factory=CameraInfo)
    scan_resolution: ScanResolutionInfo = field(default_factory=ScanResolutionInfo)


@dataclass
class ChannelInfo:
    """Canonical metadata for a single image channel.

    This defines variables along the "c"/"channel" axis. 

    In qptiffs this is usually organised the other around (Attr: <chanx>, <chany>),
    so here trying to pivot the order to chanx: attr, chany: attr.
    """

    index: int
    name: str  # biomarker / stain or fluorophore name / rgb for birghtfield/he
    fluorophore: Optional[str] = None
    # ExposureTime in the PerkinElmer spec is in microseconds
    exposure_time_us: Optional[float] = None
    emission_wavelength_nm: Optional[float] = None
    excitation_wavelength_nm: Optional[float] = None
    is_brightfield: bool = False
    # Per spec: <Color>r,g,b</Color> — display colour for this band
    color_rgb: Optional[Tuple[int, int, int]] = None
    # Per spec: <IsUnmixedComponent>True/False</IsUnmixedComponent>
    is_unmixed_component: Optional[bool] = None
    # Per spec: <SignalUnits> — packed byte: high nibble = weighting, low = unit type
    signal_units: Optional[int] = None
    # Per-channel detector settings (from CameraSettings or per-page XML)
    gain: Optional[float] = None
    binning: Optional[int] = None
    # Additional per-channel fields present in Fusion 1.x page XMLs
    objective: Optional[str] = None
    autofluorescence_subtracted: Optional[bool] = None
    responsivity: Optional[float] = None
    responsivity_filter_id: Optional[str] = None
    responsivity_date: Optional[str] = None
    responsivity_filter_name: Optional[str] = None
    excitation_filter_name: Optional[str] = None
    excitation_filter_manufacturer: Optional[str] = None
    excitation_filter_part_no: Optional[str] = None
    emission_filter_name: Optional[str] = None
    emission_filter_manufacturer: Optional[str] = None
    emission_filter_part_no: Optional[str] = None
    bit_depth: Optional[int] = None
    offset_counts: Optional[int] = None
    camera_orientation: Optional[str] = None
    roi_x: Optional[int] = None
    roi_y: Optional[int] = None
    roi_width: Optional[int] = None
    roi_height: Optional[int] = None


@dataclass
class ScaleInfo:
    """Image metadata which changes with pyramidal scale.

    ``dims`` names each spatial axis (e.g. ``["y", "x"]``).  ``downsample_factors``
    and ``pixel_sizes_um`` are parallel lists indexed by position in ``dims``.

    Stored as lists rather than named per-axis fields so that anisotropic
    downsampling (e.g. 2x in Y, 4x in X) and future extra axes are handled
    without schema changes.

    ie (2x isotropic level 1):
        dims = ["y", "x"]
        downsample_factors = [2.0, 2.0]
        pixel_sizes_um = [0.5, 0.5]
    """

    level: int
    dims: List[str] = field(default_factory=list)
    downsample_factors: List[float] = field(default_factory=list)
    pixel_sizes_um: List[float] = field(default_factory=list)

@dataclass
class QptiffImageSceneMetadata:
    """Metadata for one logical image from a QPTIFF (FullRes / Label / Macro / Thumbnail).

    Metadata-only.
    """

    image_info: ImageInfo = field(default_factory=ImageInfo)
    channels: List[ChannelInfo] = field(default_factory=list)  # [] for Label/Macro
    scales: List[ScaleInfo] = field(default_factory=list)  # len 1 if not pyramidal
    raw_xml: str = ""

    @property
    def image_type(self) -> Optional[str]:
        return self.image_info.image_type

    @property
    def is_pyramidal(self) -> bool:
        return len(self.scales) > 1

    @property
    def channel_names(self) -> List[str]:
        return [ch.name for ch in self.channels]

    @property
    def is_brightfield(self) -> bool:
        sm = self.image_info.scan_mode
        return any(ch.is_brightfield for ch in self.channels) or (
            sm is not None and "Brightfield" in sm
        )


@dataclass
class QptiffMetadata:
    """
    Structured metadata extracted from a PerkinElmer QPI QPTIFF file,
    across each image scenes, dimensions and scales (if multiscale).

    ``slide`` holds fields constant across every image on the slide.
    ``images`` holds one :class:`QptiffImageSceneMetadata` per distinct image (the
    main FullResolution scan plus any Label / Macro / Thumbnail siblings).
    """

    slide: SlideInfo = field(default_factory=SlideInfo)
    images: List[QptiffImageSceneMetadata] = field(default_factory=list)
    acquisition_format: Optional[str] = None  # one of the FORMAT_* constants; defines how we parse in xmls
    raw_xml: str = ""

    @property
    def full_resolution(self) -> Optional[QptiffImageSceneMetadata]:
        return next(
            (im for im in self.images if im.image_type == "FullResolution"),
            None,
        )

    def by_type(self, image_type: str) -> Optional[QptiffImageSceneMetadata]:
        return next(
            (im for im in self.images if im.image_type == image_type),
            None,
        )

    @property
    def _primary(self) -> QptiffImageSceneMetadata:
        """Image used by back-compat accessors — FullRes if present else first."""
        return self.full_resolution or (
            self.images[0] if self.images else QptiffImageSceneMetadata()
        )

    # for convenience, denormalise and have everything easily accessible for downstream
    # ideally, when constructing ie omengff object, it would call by attribute hierarchy;
    # ie for Slide info -> QptiffMetadata.slide
    # then for the multiscale object constructor;
    # call im = QptiffMetadata.image[scene_index]
    # then im.image_info, .scales, .channels

    @property
    def slide_id(self) -> Optional[str]:
        return self.slide.slide_id

    @property
    def barcode(self) -> Optional[str]:
        return self.slide.barcode

    @property
    def study_name(self) -> Optional[str]:
        return self.slide.study_name

    @property
    def operator_name(self) -> Optional[str]:
        return self.slide.operator_name

    @property
    def computer_name(self) -> Optional[str]:
        return self.slide.computer_name

    @property
    def datetime(self) -> Optional[str]:
        return self.slide.datetime

    @property
    def acquisition_software(self) -> Optional[str]:
        return self.slide.acquisition_software

    @property
    def description_version(self) -> Optional[str]:
        return self.slide.description_version

    @property
    def instrument_type(self) -> Optional[str]:
        return self.slide.instrument_type

    @property
    def identifier(self) -> Optional[str]:
        return self.slide.identifier

    @property
    def image_type(self) -> Optional[str]:
        return self._primary.image_info.image_type

    @property
    def objective(self) -> Optional[str]:
        return self._primary.image_info.objective

    @property
    def bf_lamp_type(self) -> Optional[str]:
        return self._primary.image_info.bf_lamp_type

    @property
    def scan_profile_name(self) -> Optional[str]:
        return self._primary.image_info.scan_profile_name

    @property
    def scan_mode(self) -> Optional[str]:
        return self._primary.image_info.scan_mode

    @property
    def is_tma(self) -> Optional[bool]:
        return self._primary.image_info.is_tma

    @property
    def opal_kit_type(self) -> Optional[str]:
        return self._primary.image_info.opal_kit_type

    @property
    def xposition_um(self) -> Optional[float]:
        return self._primary.image_info.xposition_um

    @property
    def yposition_um(self) -> Optional[float]:
        return self._primary.image_info.yposition_um

    @property
    def camera(self) -> CameraInfo:
        return self._primary.image_info.camera

    @property
    def scan_resolution(self) -> ScanResolutionInfo:
        return self._primary.image_info.scan_resolution

    @property
    def channels(self) -> List[ChannelInfo]:
        return self._primary.channels

    @property
    def pixel_size_um(self) -> Optional[float]:
        return self._primary.image_info.scan_resolution.base_pixel_size_um

    @property
    def channel_names(self) -> List[str]:
        return self._primary.channel_names

    @property
    def is_brightfield(self) -> bool:
        return self._primary.is_brightfield
