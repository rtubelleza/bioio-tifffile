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

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


def _strip_key(obj: Any, key: str) -> None:
    """Recursively remove ``key`` from nested dicts/lists (in place)."""
    if isinstance(obj, dict):
        obj.pop(key, None)
        for v in obj.values():
            _strip_key(v, key)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _strip_key(v, key)


class JsonReprMixin:
    """Gives every metadata dataclass a JSON-/dict-serializable representation.

    xarray / zarr attrs cannot store arbitrary Python objects, so anything that
    ends up in ``DataArray.attrs`` (e.g. the ``processed`` metadata) must be a
    plain dict. ``to_dict`` recurses through nested dataclasses and lists; the
    bulky ``raw_xml`` is dropped by default (it is the unparsed source, redundant
    with the structured fields, and tens of KB).
    """

    def to_dict(self, include_raw_xml: bool = False) -> Dict[str, Any]:
        d = asdict(self)  # recursive over nested dataclasses / lists
        if not include_raw_xml:
            _strip_key(d, "raw_xml")
        return d

    def to_json(self, *, include_raw_xml: bool = False, **kwargs: Any) -> str:
        return json.dumps(
            self.to_dict(include_raw_xml=include_raw_xml), default=str, **kwargs
        )

# ---------------------------------------------------------------------------
# Legacy acquisition-format labels (DEPRECATED as a dispatch key).
#
# These names describe the vendor product we *guessed* wrote the file, and they
# have turned out not to predict how it must be parsed. Empirically the channel
# dialect is identical across Fusion 1.0.6, 1.0.8 and 2.3.1 (DescriptionVersion
# 4 and 6 alike), while optional field presence varies at patch level — so
# vendor/version granularity is wrong in both directions. Worse, files that land
# on ``polaris_scanband`` because a filter-cube <ScanBands-i> block exists must
# still be parsed page-by-page.
#
# Parsing now dispatches on LOCUS_* below. These constants are retained solely
# so ``QptiffMetadata.acquisition_format`` keeps emitting the exact strings that
# already reached written zarr stores; do not branch on them in new code.
# ---------------------------------------------------------------------------

#: RGB brightfield scan (H&E / IHC): samples stored in the S dimension.
FORMAT_BRIGHTFIELD = "brightfield"

#: A <ScanBands-i> block is present anywhere in the tree. NOTE: this says only
#: that a filter-cube description exists, NOT that channels are described there
#: — newer paged files carry both. See :data:`LOCUS_SHARED_SCANBANDS`.
FORMAT_POLARIS_SCANBAND = "polaris_scanband"

#: <ScanProfile> content is a JSON blob (Fusion 1.x and 2.x alike).
FORMAT_FUSION_PAGED = "fusion_paged"

#: No recognised structural signals. Parser will try strategies in order.
FORMAT_UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Channel-metadata locus — where per-channel fields actually live.
#
# This is the axis that predicts parsing behaviour, and the one the parser
# dispatches on. The field dialect (what a given tag *means*) is a function of
# the locus, not of the acquisition software version.
# ---------------------------------------------------------------------------

#: Channels are the RGB samples of a single acquisition, stored in the S
#: dimension. There is no per-channel biomarker/filter metadata to find.
LOCUS_RGB_SAMPLES = "rgb_samples"

#: Per-channel metadata lives in <ScanBands-i> elements of one shared XML that
#: every page repeats (classic Vectra / Polaris / OPAL). In this dialect
#: <Fluorophore> is explicit and <Name> means the biomarker.
LOCUS_SHARED_SCANBANDS = "shared_scanbands"

#: Each page's own XML root carries its channel's metadata (Fusion 1.x and 2.x,
#: and "paged Polaris"). In this dialect there is no <Fluorophore> element at
#: all: <Biomarker> is the stain and <Name> is the fluorophore/filter.
LOCUS_PER_PAGE_ROOT = "per_page_root"

#: No recognised locus; the parser falls back to trying strategies in order.
LOCUS_UNKNOWN = "unknown"


@dataclass
class ScanResolutionInfo(JsonReprMixin):
    """Objective/scan settings that apply to the whole scene.

    ``base_pixel_size_um`` is the physical pixel size of the full resolution page (scale0).
    """

    magnification: Optional[float] = None
    objective_name: Optional[str] = None
    binning: Optional[int] = None
    #: DEPRECATED: micron-specific. Use the unit-agnostic
    #: ``ImageInfo.scale_factor`` + ``ImageInfo.scale_factor_unit`` instead, which
    #: carry the full-resolution pixel->physical scale and its unit as declared by
    #: the source metadata. Kept (still populated) only for backward compatibility.
    base_pixel_size_um: Optional[float] = None


@dataclass
class CameraInfo(JsonReprMixin):
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
class SlideInfo(JsonReprMixin):
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
    validation_code: Optional[str] = None  # file integrity hash, root <ValidationCode>
    sample_description: Optional[str] = None  # free-text sample label, root <SampleDescription>  # noqa: E501


@dataclass
class ImageInfo(JsonReprMixin):
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
    # stored pixel precision: TIFF tag 258 BitsPerSample (8 for 8-bit RGB H&E,
    # 16 for fluorescence). Distinct from camera.bit_depth (the ADC depth).
    stored_bits_per_sample: Optional[int] = None
    # optics / acquisition
    objective: Optional[str] = None # potentiall SlideInfo
    bf_lamp_type: Optional[str] = None  # brightfield lamp, root <BFLampType>
    lamp_type: Optional[str] = None  # fluorescence excitation lamp, root <LampType>
    scan_profile_name: Optional[str] = None
    scan_mode: Optional[str] = None
    is_tma: Optional[bool] = None # qptiff specific; most probably a qpi metadata
    opal_kit_type: Optional[str] = None
    # Full-resolution pixel -> physical scaling of THIS image (value only; one per
    # image, NOT per pyramid level). A pyramid level's physical pixel size is
    # inferred as scale_factor * (that level's downsample factor). Unit-agnostic:
    # the unit lives in scale_factor_unit.
    scale_factor: Optional[float] = None
    # physical unit of scale_factor, as declared by the source metadata (e.g. "um"
    # for QPI <PixelSizeMicrons>). ASCII (not "µm") to stay query-friendly.
    scale_factor_unit: Optional[str] = None
    # ScanProfile acquisition settings (Polaris/Fusion <ScanProfile><root>)
    compression: Optional[str] = None  # <Compression> e.g. "LZW"
    jpeg_quality: Optional[int] = None  # <JPEGQuality>
    saturation_protection_type: Optional[str] = None  # <SaturationProtectionType>
    coverslip_thickness: Optional[str] = None  # <CoverslipThickness>
    is_rna: Optional[bool] = None  # <IsRNA> — RNAscope-style assay flag
    # camera orientation flags (<ScanProfile><root><CameraSettings>)
    rotate_image: Optional[bool] = None  # <RotateImage>
    mirror_image: Optional[bool] = None  # <MirrorImage>
    # stage position of this image
    xposition_um: Optional[float] = None
    yposition_um: Optional[float] = None
    # nested per-image blocks
    camera: CameraInfo = field(default_factory=CameraInfo)
    scan_resolution: ScanResolutionInfo = field(default_factory=ScanResolutionInfo)


@dataclass
class ChannelInfo(JsonReprMixin):
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
    #: Midpoint of the active band — the single value OME's
    #: Channel.emission_wavelength / .excitation_wavelength take. Derived from
    #: the cut-on/cut-off edges below when a <Band> is present.
    emission_wavelength_nm: Optional[float] = None
    excitation_wavelength_nm: Optional[float] = None
    #: Raw passband edges of the *active* band, feeding OME's
    #: Filter.transmittance_range (cut_in / cut_out). Averaging these into the
    #: midpoint alone loses the passband width.
    emission_cut_on_nm: Optional[float] = None
    emission_cut_off_nm: Optional[float] = None
    excitation_cut_on_nm: Optional[float] = None
    excitation_cut_off_nm: Optional[float] = None
    #: Number of bands the filter cube declares; >1 means a multi-pass cube
    #: (OME Filter.type MULTI_PASS rather than BAND_PASS).
    n_filter_bands: Optional[int] = None
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
    # Polaris ScanBands-i auto-exposure mode, e.g. "aet_Fluorescence"
    auto_expose_type: Optional[str] = None
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
class ScaleInfo(JsonReprMixin):
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
class QptiffImageSceneMetadata(JsonReprMixin):
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
class QptiffMetadata(JsonReprMixin):
    """
    Structured metadata extracted from a PerkinElmer QPI QPTIFF file,
    across each image scenes, dimensions and scales (if multiscale).

    ``slide`` holds fields constant across every image on the slide.
    ``images`` holds one :class:`QptiffImageSceneMetadata` per distinct image (the
    main FullResolution scan plus any Label / Macro / Thumbnail siblings).
    """

    slide: SlideInfo = field(default_factory=SlideInfo)
    images: List[QptiffImageSceneMetadata] = field(default_factory=list)
    #: DEPRECATED as a dispatch key — one of the FORMAT_* constants. Retained
    #: verbatim for back-compat with zarr stores already written; parsing is
    #: driven by :attr:`channel_locus`.
    acquisition_format: Optional[str] = None
    #: One of the LOCUS_* constants: where per-channel metadata actually lives.
    #: This is what the parser dispatches on.
    channel_locus: Optional[str] = None
    #: Primitive structural signals the locus was derived from. Recorded so the
    #: real-world corpus can be measured without re-opening files.
    structure_signature: Dict[str, object] = field(default_factory=dict)
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
