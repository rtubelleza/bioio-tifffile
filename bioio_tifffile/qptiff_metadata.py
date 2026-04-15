#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PerkinElmer QPI (QPTIFF) metadata parser.

Parses the XML embedded in ImageDescription tag (TIFF tag 270) of QPTIFF files
produced by PerkinElmer Vectra/Polaris/Fusion/CODEX instruments.
"""

from __future__ import annotations

import json
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

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
    """Objective / scan-profile settings that apply to the whole scene.

    ``base_pixel_size_um`` is the physical pixel size of the full resolution tile.
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
    offset_counts: Optional[int] = None
    orientation: Optional[str] = None
    roi_x: Optional[int] = None
    roi_y: Optional[int] = None
    roi_width: Optional[int] = None
    roi_height: Optional[int] = None


@dataclass
class SlideInfo:
    """Physical-slide identity. Constant across every image on the slide.

    Surfaced on each image's root DataTree attrs (with a ``slide:`` prefix) so
    detached NGFF groups remain independently interpretable.
    """
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
    Thumbnail). Dumped flat into that image's root ``DataTree.attrs``.
    """
    image_type: Optional[str] = None  # FullResolution / Thumbnail / Macro / Label
    # optics / acquisition
    objective: Optional[str] = None
    bf_lamp_type: Optional[str] = None
    scan_profile_name: Optional[str] = None
    scan_mode: Optional[str] = None
    is_tma: Optional[bool] = None
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

    Attempt to standardise incoming qptiff metadata fields here for ome_metadata compatability
    as well.

    This defines variables along the "c"/"channel" axis.
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
    """Image metadata which changes with pyramidal scale. """
    level: int
    scale_factor: Optional[float] = None # ds factor relative to full
    # physical pixel size in micrometres; 
    # this can be implicitly found by scale_factor * base_pixel_size_um, 
    # but denormalise here for convenience
    pixel_size_um: Optional[float] = None  

def _format_color(c: Optional[Tuple[int, int, int]]) -> Optional[str]:
    return f"{c[0]},{c[1]},{c[2]}" if c else None


def _format_binning(b: Optional[int]) -> Optional[str]:
    return f"{b}x{b}" if b is not None else None


# declarative schema describing how each ChannelInfo attribute is surfaced
# as an OME-style key in the flat attrs dict and as an xarray coordinate on
# the channel axis. Order is preserved when iterating so xarray coord order
# matches to_dict() output.
#
# Tuple layout: (ome_key, channel_info_attr, formatter_or_None)
CHANNEL_COORD_SCHEMA: List[Tuple[str, str, Optional[object]]] = [
    # Name is emitted as the channel coord itself, not a sibling coord, so it
    # is intentionally absent from this schema.
    ("Channel:Fluor", "fluorophore", None),
    ("Channel:Color", "color_rgb", _format_color),
    ("Channel:EmissionWavelength", "emission_wavelength_nm", None),
    ("Channel:ExcitationWavelength", "excitation_wavelength_nm", None),
    ("Plane:ExposureTime", "exposure_time_us", None),
    ("DetectorSettings:Gain", "gain", None),
    ("DetectorSettings:Binning", "binning", _format_binning),
    # QPTIFF-specific per-channel fields — no OME equivalent
    ("qpi_IsUnmixedComponent", "is_unmixed_component", None),
    ("qpi_SignalUnits", "signal_units", None),
    ("qpi_Objective", "objective", None),
    ("qpi_ScaleFactor", "scale_factor", None),
    ("qpi_AutofluorescenceSubtracted", "autofluorescence_subtracted", None),
    ("qpi_Responsivity", "responsivity", None),
    ("qpi_ResponsivityFilterId", "responsivity_filter_id", None),
    ("qpi_ResponsivityDate", "responsivity_date", None),
    ("qpi_ResponsivityFilterName", "responsivity_filter_name", None),
    ("qpi_ExcitationFilterName", "excitation_filter_name", None),
    ("qpi_ExcitationFilterManufacturer", "excitation_filter_manufacturer", None),
    ("qpi_ExcitationFilterPartNo", "excitation_filter_part_no", None),
    ("qpi_EmissionFilterName", "emission_filter_name", None),
    ("qpi_EmissionFilterManufacturer", "emission_filter_manufacturer", None),
    ("qpi_EmissionFilterPartNo", "emission_filter_part_no", None),
    ("qpi_BitDepth", "bit_depth", None),
    ("qpi_OffsetCounts", "offset_counts", None),
    ("qpi_CameraOrientation", "camera_orientation", None),
    ("qpi_ROIX", "roi_x", None),
    ("qpi_ROIY", "roi_y", None),
    ("qpi_ROIWidth", "roi_width", None),
    ("qpi_ROIHeight", "roi_height", None),
]


@dataclass
class QptiffImageScene:
    """One logical image from a QPTIFF (FullRes / Label / Macro / Thumbnail).

    Metadata-only. Pixel data is attached by the reader when building the
    per-scene DataTree.
    """
    image_info: ImageInfo = field(default_factory=ImageInfo)
    channels: List[ChannelInfo] = field(default_factory=list)  # [] for Label/Macro
    scales: List[ScaleInfo] = field(default_factory=list)       # len 1 if not pyramidal
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
    ``images`` holds one :class:`QptiffImageScene` per distinct image (the
    main FullResolution scan plus any Label / Macro / Thumbnail siblings).

    """
    slide: SlideInfo = field(default_factory=SlideInfo)
    images: List[QptiffImageScene] = field(default_factory=list)
    acquisition_format: Optional[str] = None  # one of the FORMAT_* constants
    raw_xml: str = ""

    @property
    def full_resolution(self) -> Optional[QptiffImageScene]:
        return next(
            (im for im in self.images if im.image_type == "FullResolution"),
            None,
        )

    def by_type(self, image_type: str) -> Optional[QptiffImageScene]:
        return next(
            (im for im in self.images if im.image_type == image_type),
            None,
        )

    @property
    def _primary(self) -> QptiffImageScene:
        """Image used by back-compat accessors — FullRes if present else first."""
        return self.full_resolution or (self.images[0] if self.images else QptiffImageScene())

    @property
    def slide_id(self) -> Optional[str]: return self.slide.slide_id
    @property
    def barcode(self) -> Optional[str]: return self.slide.barcode
    @property
    def study_name(self) -> Optional[str]: return self.slide.study_name
    @property
    def operator_name(self) -> Optional[str]: return self.slide.operator_name
    @property
    def computer_name(self) -> Optional[str]: return self.slide.computer_name
    @property
    def datetime(self) -> Optional[str]: return self.slide.datetime
    @property
    def acquisition_software(self) -> Optional[str]: return self.slide.acquisition_software
    @property
    def description_version(self) -> Optional[str]: return self.slide.description_version
    @property
    def instrument_type(self) -> Optional[str]: return self.slide.instrument_type
    @property
    def identifier(self) -> Optional[str]: return self.slide.identifier
    @property
    def image_type(self) -> Optional[str]: return self._primary.image_info.image_type
    @property
    def objective(self) -> Optional[str]: return self._primary.image_info.objective
    @property
    def bf_lamp_type(self) -> Optional[str]: return self._primary.image_info.bf_lamp_type
    @property
    def scan_profile_name(self) -> Optional[str]: return self._primary.image_info.scan_profile_name
    @property
    def scan_mode(self) -> Optional[str]: return self._primary.image_info.scan_mode
    @property
    def is_tma(self) -> Optional[bool]: return self._primary.image_info.is_tma
    @property
    def opal_kit_type(self) -> Optional[str]: return self._primary.image_info.opal_kit_type
    @property
    def xposition_um(self) -> Optional[float]: return self._primary.image_info.xposition_um
    @property
    def yposition_um(self) -> Optional[float]: return self._primary.image_info.yposition_um
    @property
    def camera(self) -> CameraInfo: return self._primary.image_info.camera
    @property
    def scan_resolution(self) -> ScanResolutionInfo: return self._primary.image_info.scan_resolution

    @property
    def channels(self) -> List[ChannelInfo]: return self._primary.channels

    @property
    def pixel_size_um(self) -> Optional[float]:
        return self._primary.image_info.scan_resolution.base_pixel_size_um

    @property
    def channel_names(self) -> List[str]:
        return self._primary.channel_names

    @property
    def is_brightfield(self) -> bool:
        return self._primary.is_brightfield

    def to_dict(self, *, ome_only: bool = False) -> Dict:
        """
        Flat dictionary for xarray attrs.

        OME-compliant field names are used where an OME equivalent exists
        (e.g. ``Pixels:PhysicalSizeX``, ``Channel:0:Name``).  Fields that are
        specific to the PerkinElmer QPTIFF format with no OME equivalent retain
        the ``qpi_`` prefix.

        When ``ome_only`` is True, entries with the ``qpi_`` prefix are
        omitted, leaving only keys that map to an OME-XML field.

        """
        d: Dict = {
            # ome pixels
            "Pixels:PhysicalSizeX": self.pixel_size_um,
            "Pixels:PhysicalSizeXUnit": "um"
            if self.pixel_size_um is not None
            else None,
            "Pixels:PhysicalSizeY": self.pixel_size_um,
            "Pixels:PhysicalSizeYUnit": "um"
            if self.pixel_size_um is not None
            else None,
            # ome plane (baseline / first plane positional metadata)
            "Plane:PositionX": self.xposition_um,
            "Plane:PositionXUnit": "um" if self.xposition_um is not None else None,
            "Plane:PositionY": self.yposition_um,
            "Plane:PositionYUnit": "um" if self.yposition_um is not None else None,
            # ome experimenter
            "Experimenter:UserName": self.operator_name,
            # ome instrument / microscope
            "Microscope:Model": self.instrument_type,
            # ome objective
            "Objective:Model": self.scan_resolution.objective_name,
            "Objective:NominalMagnification": self.scan_resolution.magnification,
            # ome detector
            "Detector:Model": self.camera.camera_type,
            # below seems to be qptiff specific. TODO: check if any map to an ome_types object
            "qpi_description_version": self.description_version,
            "qpi_acquisition_software": self.acquisition_software,
            "qpi_image_type": self.image_type,
            "qpi_identifier": self.identifier,
            "qpi_slide_id": self.slide_id,
            "qpi_barcode": self.barcode,
            "qpi_study_name": self.study_name,
            "qpi_computer_name": self.computer_name,
            "qpi_datetime": self.datetime,
            "qpi_bf_lamp_type": self.bf_lamp_type,
            "qpi_scan_profile_name": self.scan_profile_name,
            "qpi_scan_mode": self.scan_mode,
            "qpi_is_tma": self.is_tma,
            "qpi_opal_kit_type": self.opal_kit_type,
            "qpi_acquisition_format": self.acquisition_format,
            "qpi_camera_name": self.camera.camera_name,
            "qpi_camera_gain": self.camera.gain,
            "qpi_camera_bit_depth": self.camera.bit_depth,
            "qpi_channel_count": len(self.channels),
            "qpi_is_brightfield": self.is_brightfield,
            "qpi_objective": self.objective,
        }

        for ch in self.channels:
            i = ch.index
            d[f"Channel:{i}:Name"] = ch.name

            # units for the two fields where the unit matters
            if ch.emission_wavelength_nm is not None:
                d[f"Channel:{i}:EmissionWavelengthUnit"] = "nm"
            if ch.excitation_wavelength_nm is not None:
                d[f"Channel:{i}:ExcitationWavelengthUnit"] = "nm"
            if ch.exposure_time_us is not None:
                d[f"Plane:{i}:ExposureTimeUnit"] = "us"

            for ome_key, attr, formatter in CHANNEL_COORD_SCHEMA:
                if ome_only and ome_key.startswith("qpi_"):
                    continue
                raw = getattr(ch, attr, None)
                value = formatter(raw) if formatter else raw  # type: ignore[operator]
                if value is None:
                    continue
                #  ome_key has no index in it (e.g. "Channel:Fluor"); insert
                # the channel index after the first colon to form
                # "Channel:N:Fluor" / "qpi_chN_<name>" etc.
                if ome_key.startswith("qpi_"):
                    flat_key = f"qpi_ch{i}_{ome_key[4:]}"
                else:
                    prefix, _, suffix = ome_key.partition(":")
                    flat_key = f"{prefix}:{i}:{suffix}"
                d[flat_key] = value
        return {
            k: v
            for k, v in d.items()
            if v is not None and not (ome_only and k.startswith("qpi_"))
        }


def _ome_color(rgb: Optional[Tuple[int, int, int]]) -> Optional[object]:
    if rgb is None:
        return None
    try:
        from ome_types.model import Color

        return Color(f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}")
    except Exception:
        return None


def _str(v: object) -> Optional[str]:
    return str(v) if v is not None else None


_TIFF_DATETIME_RE = re.compile(
    r"^(\d{4}):(\d{2}):(\d{2})[ T](\d{2}:\d{2}:\d{2}(?:\.\d+)?)"
)


def _tiff_datetime_to_iso(dt: Optional[str]) -> Optional[str]:
    """Convert a TIFF DateTime string ("YYYY:MM:DD HH:MM:SS") to ISO 8601.

    Pydantic's datetime validator rejects the colon-separated date form used
    by the TIFF spec. Pass-through any value that already parses (ISO, with
    timezone, etc.).
    """
    if not dt:
        return dt
    m = _TIFF_DATETIME_RE.match(dt.strip())
    if m is None:
        return dt
    y, mo, d, t = m.groups()
    return f"{y}-{mo}-{d}T{t}"


def ome_metadata_from_qptiff(
    qpi: "QptiffMetadata",
    scene_name: Optional[str] = None,
    size_x: Optional[int] = None,
    size_y: Optional[int] = None,
    size_z: Optional[int] = None,
    size_c: Optional[int] = None,
    size_t: Optional[int] = None,
) -> object:
    """
    Map a :class:`QptiffMetadata` to a fully typed :class:`ome_types.model.OME`
    object.

    Fields that have a direct OME-XML equivalent are placed on the canonical
    OME objects (``Channel``, ``Plane``, ``DetectorSettings``, ``Pixels``,
    ``Instrument``, ``Objective``, ``Detector``, ``Filter``).

    Fields specific to the PerkinElmer QPI format with no OME equivalent are
    collected into a ``MapAnnotation`` (namespace ``qpi://vectra``) on the
    ``OME.structured_annotations`` list — one annotation per image and one per
    channel (prefixed ``ch<N>_``).

    Excitation and emission filter names / manufacturers map to OME ``Filter``
    objects in ``Instrument.filters``, linked from ``Channel.light_path``.

    Parameters
    ----------
    qpi:
        Parsed QPTIFF metadata.
    scene_name:
        Human-readable name of the scene (becomes ``Image.Name``).
    size_x, size_y, size_z, size_c, size_t:
        Pixel dimensions of the image at the selected pyramid level.
    """
    from ome_types.model import (
        OME,
        AnnotationRef,
        Channel,
        Detector,
        DetectorSettings,
        Experimenter,
        ExperimenterRef,
        Filter,
        FilterRef,
        Image,
        Instrument,
        InstrumentRef,
        LightPath,
        MapAnnotation,
        Microscope,
        Objective,
        ObjectiveSettings,
        Pixels,
        Plane,
    )

    instrument_id = "Instrument:0"
    objective_id = "Objective:0"
    detector_id = "Detector:0"
    image_id = "Image:0"
    pixels_id = "Pixels:0"
    experimenter_id = "Experimenter:0"

    # instrument sub objects
    microscope = Microscope(model=qpi.instrument_type) if qpi.instrument_type else None

    objective = None
    if qpi.scan_resolution.objective_name or qpi.scan_resolution.magnification:
        objective = Objective(
            id=objective_id,
            model=qpi.scan_resolution.objective_name,
            nominal_magnification=qpi.scan_resolution.magnification,
        )

    detector = None
    if qpi.camera.camera_type or qpi.camera.camera_name:
        detector = Detector(
            id=detector_id,
            model=qpi.camera.camera_type or qpi.camera.camera_name,
            gain=qpi.camera.gain,
        )

    # one pair per channel declaring filter info
    filters: List[Filter] = []
    filter_idx = 0

    def _make_filter(
        name: Optional[str],
        manufacturer: Optional[str],
        fid: str,
    ) -> Optional[Filter]:
        if not name and not manufacturer:
            return None
        return Filter(id=fid, model=name, manufacturer=manufacturer)

    # filter objs, track id per chan
    exc_filter_ids: Dict[int, str] = {}
    emi_filter_ids: Dict[int, str] = {}

    for ch in qpi.channels:
        exc = _make_filter(
            ch.excitation_filter_name,
            ch.excitation_filter_manufacturer,
            f"Filter:{filter_idx}",
        )
        if exc:
            exc_filter_ids[ch.index] = exc.id
            filters.append(exc)
            filter_idx += 1

        emi = _make_filter(
            ch.emission_filter_name,
            ch.emission_filter_manufacturer,
            f"Filter:{filter_idx}",
        )
        if emi:
            emi_filter_ids[ch.index] = emi.id
            filters.append(emi)
            filter_idx += 1

    instrument = Instrument(
        id=instrument_id,
        microscope=microscope,
        objectives=[objective] if objective else [],
        detectors=[detector] if detector else [],
        filters=filters,
    )

    # experimenter
    experimenter = None
    if qpi.operator_name:
        experimenter = Experimenter(id=experimenter_id, user_name=qpi.operator_name)

    # channels, planes, detector settings
    ome_channels: List[Channel] = []
    ome_planes: List[Plane] = []

    for ch in qpi.channels:
        det_settings = None
        if ch.gain is not None or ch.binning is not None:
            det_settings = DetectorSettings(
                id=detector_id,
                gain=ch.gain,
                binning=_format_binning(ch.binning),
            )

        exc_fid = exc_filter_ids.get(ch.index)
        emi_fid = emi_filter_ids.get(ch.index)
        light_path = None
        if exc_fid or emi_fid:
            light_path = LightPath(
                excitation_filters=[FilterRef(id=exc_fid)] if exc_fid else [],
                emission_filters=[FilterRef(id=emi_fid)] if emi_fid else [],
            )

        ch_kwargs: Dict[str, object] = dict(
            id=f"Channel:0:{ch.index}",
            name=ch.name,
            fluor=ch.fluorophore,
            detector_settings=det_settings,
            light_path=light_path,
        )
        color = _ome_color(ch.color_rgb)
        if color is not None:
            ch_kwargs["color"] = color
        if ch.emission_wavelength_nm is not None:
            ch_kwargs["emission_wavelength"] = ch.emission_wavelength_nm
            ch_kwargs["emission_wavelength_unit"] = "nm"
        if ch.excitation_wavelength_nm is not None:
            ch_kwargs["excitation_wavelength"] = ch.excitation_wavelength_nm
            ch_kwargs["excitation_wavelength_unit"] = "nm"
        ome_channels.append(Channel(**ch_kwargs))  # type: ignore[arg-type]

        # One Plane per channel (Z=0, T=0).
        plane_kwargs: Dict[str, object] = dict(the_z=0, the_t=0, the_c=ch.index)
        if ch.exposure_time_us is not None:
            plane_kwargs["exposure_time"] = ch.exposure_time_us
            plane_kwargs["exposure_time_unit"] = "\u00b5s"
        if qpi.xposition_um is not None:
            plane_kwargs["position_x"] = qpi.xposition_um
            plane_kwargs["position_x_unit"] = "\u00b5m"
        if qpi.yposition_um is not None:
            plane_kwargs["position_y"] = qpi.yposition_um
            plane_kwargs["position_y_unit"] = "\u00b5m"
        ome_planes.append(Plane(**plane_kwargs))  # type: ignore[arg-type]

    px_kwargs: Dict[str, object] = dict(
        id=pixels_id,
        dimension_order="XYZCT",
        type="uint16",
        size_x=size_x or 1,
        size_y=size_y or 1,
        size_z=size_z or 1,
        size_c=size_c or max(1, len(qpi.channels)),
        size_t=size_t or 1,
    )
    if qpi.pixel_size_um is not None:
        px_kwargs["physical_size_x"] = qpi.pixel_size_um
        px_kwargs["physical_size_x_unit"] = "\u00b5m"
        px_kwargs["physical_size_y"] = qpi.pixel_size_um
        px_kwargs["physical_size_y_unit"] = "\u00b5m"
    px_kwargs["channels"] = ome_channels
    px_kwargs["planes"] = ome_planes
    pixels = Pixels(**px_kwargs)  # type: ignore[arg-type]

    # image level
    img_qpi: Dict[str, str] = {}
    _qpi_map = [
        ("description_version", qpi.description_version),
        ("acquisition_software", qpi.acquisition_software),
        ("image_type", qpi.image_type),
        ("identifier", qpi.identifier),
        ("slide_id", qpi.slide_id),
        ("barcode", qpi.barcode),
        ("study_name", qpi.study_name),
        ("computer_name", qpi.computer_name),
        ("datetime", qpi.datetime),
        ("bf_lamp_type", qpi.bf_lamp_type),
        ("scan_profile_name", qpi.scan_profile_name),
        ("scan_mode", qpi.scan_mode),
        ("is_tma", _str(qpi.is_tma)),
        ("opal_kit_type", qpi.opal_kit_type),
        ("acquisition_format", qpi.acquisition_format),
        ("camera_name", qpi.camera.camera_name),
        ("camera_gain", _str(qpi.camera.gain)),
        ("camera_bit_depth", _str(qpi.camera.bit_depth)),
    ]
    for k, v in _qpi_map:
        if v is not None:
            img_qpi[k] = v

    # channel level
    for ch in qpi.channels:
        ch_map = [
            ("is_unmixed_component", _str(ch.is_unmixed_component)),
            ("signal_units", _str(ch.signal_units)),
            ("objective", ch.objective),
            ("scale_factor", _str(ch.scale_factor)),
            ("autofluorescence_subtracted", _str(ch.autofluorescence_subtracted)),
            ("responsivity", _str(ch.responsivity)),
            ("responsivity_filter_id", ch.responsivity_filter_id),
            ("responsivity_date", ch.responsivity_date),
            ("responsivity_filter_name", ch.responsivity_filter_name),
            ("excitation_filter_part_no", ch.excitation_filter_part_no),
            ("emission_filter_part_no", ch.emission_filter_part_no),
            ("bit_depth", _str(ch.bit_depth)),
            ("offset_counts", _str(ch.offset_counts)),
            ("camera_orientation", ch.camera_orientation),
            ("roi_x", _str(ch.roi_x)),
            ("roi_y", _str(ch.roi_y)),
            ("roi_width", _str(ch.roi_width)),
            ("roi_height", _str(ch.roi_height)),
        ]
        for k, v in ch_map:
            if v is not None:
                img_qpi[f"ch{ch.index}_{k}"] = v

    annotations = []
    if img_qpi:
        annotations.append(
            MapAnnotation(
                id="Annotation:0",
                namespace="qpi://vectra",
                value=img_qpi,
            )
        )

    image = Image(
        id=image_id,
        name=scene_name,
        acquisition_date=_tiff_datetime_to_iso(qpi.datetime),
        instrument_ref=InstrumentRef(id=instrument_id),
        objective_settings=ObjectiveSettings(id=objective_id) if objective else None,
        experimenter_ref=ExperimenterRef(id=experimenter_id) if experimenter else None,
        pixels=pixels,
        annotation_refs=[AnnotationRef(id="Annotation:0")] if annotations else [],
    )

    ome_kwargs: Dict[str, object] = {
        "images": [image],
        "instruments": [instrument],
        "structured_annotations": annotations,
    }
    if experimenter:
        ome_kwargs["experimenters"] = [experimenter]

    return OME(**ome_kwargs)  # type: ignore[arg-type]


# XML tag candidates for biomarker name, in priority order
_BIOMARKER_TAGS = [
    "Biomarker",
    "BioMarker",
    "StainName",
    "Marker",
    "Name",
]

# XML tag candidates for fluorophore name
_FLUOROPHORE_TAGS = [
    "Fluorophore",
    "Fluor",
]


def _text(element: Optional[ET.Element]) -> Optional[str]:
    """Return stripped text of an Element, or None."""
    if element is None:
        return None
    t = element.text
    return t.strip() if t else None


def _float(element: Optional[ET.Element]) -> Optional[float]:
    t = _text(element)
    if t is None:
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _int(element: Optional[ET.Element]) -> Optional[int]:
    t = _text(element)
    if t is None:
        return None
    try:
        return int(t)
    except ValueError:
        return None


def _bool(element: Optional[ET.Element]) -> Optional[bool]:
    t = _text(element)
    if t is None:
        return None
    return t.lower() in ("true", "1", "yes")


def _parse_color(element: Optional[ET.Element]) -> Optional[Tuple[int, int, int]]:
    """Parse a <Color>r,g,b</Color> element into an (r, g, b) int tuple."""
    t = _text(element)
    if t is None:
        return None
    parts = t.split(",")
    if len(parts) == 3:
        try:
            return (int(parts[0].strip()), int(parts[1].strip()), int(parts[2].strip()))
        except ValueError:
            return None
    return None


def _find_first(root: ET.Element, *tags: str) -> Optional[str]:
    """Search for any of the given tags anywhere in *root* and return first text."""
    for tag in tags:
        el = root.find(f".//{tag}")
        if el is not None and el.text:
            return el.text.strip()
    return None


def _parse_scan_resolution(root: ET.Element) -> ScanResolutionInfo:
    """Extract pixel size and objective from ScanResolution element."""
    sr = root.find(".//ScanResolution")
    if sr is None:
        return ScanResolutionInfo()

    pixel_size = _float(sr.find("PixelSizeMicrons"))
    magnification = _float(sr.find("Magnification"))
    objective_name = _text(sr.find("ObjectiveName"))
    binning = _int(sr.find("Binning"))

    return ScanResolutionInfo(
        base_pixel_size_um=pixel_size,
        magnification=magnification,
        objective_name=objective_name,
        binning=binning,
    )


def _parse_camera(root: ET.Element) -> CameraInfo:
    """Extract camera settings from the top-level CameraSettings element."""
    cs = root.find("CameraSettings")
    if cs is None:
        return CameraInfo()
    return CameraInfo(
        camera_name=_text(root.find("CameraName")),
        camera_type=_text(root.find("CameraType")),
        gain=_float(cs.find("Gain")),
        bit_depth=_int(cs.find("BitDepth")),
        binning=_int(cs.find("Binning")),
    )


def _parse_exposure_times(root: ET.Element) -> List[Optional[float]]:
    """Return list of per-channel exposure times (us) from ExposureTimeArray.

    Per the PerkinElmer spec, ExposureTime values are integer microseconds.
    """
    eta = root.find("ExposureTimeArray")
    if eta is None:
        single = _float(root.find("ExposureTime"))
        return [single]
    times = []
    for val_el in eta.findall("Value"):
        times.append(_float(val_el))
    return times


def _parse_brightfield_channels(n_samples: int) -> List[ChannelInfo]:
    """
    For brightfield RGB images, return R/G/B channel descriptors.
    n_samples is the number of colour planes (usually 3).
    """
    if n_samples == 3:
        names = ["Red", "Green", "Blue"]
    else:
        names = [f"Sample_{i}" for i in range(n_samples)]
    return [
        ChannelInfo(index=i, name=names[i], is_brightfield=True)
        for i in range(n_samples)
    ]


def _parse_fluorescence_channels(
    root: ET.Element,
    exposure_times: List[Optional[float]],
) -> List[ChannelInfo]:
    """
    Parse per-channel biomarker / fluorophore info from the ScanBands section.

    In multiplexed fluorescence QPI files each channel is described inside a
    ScanBands-i element.  The per-band element carries everything the
    Fusion-paged format stores at its page root, so the same extractor is used.
    """
    channels: List[ChannelInfo] = []
    scan_bands = root.findall(".//ScanBands-i")

    for idx, band in enumerate(scan_bands):
        fields: Dict[str, object] = {"index": idx, "name": f"Channel_{idx}"}
        _populate_channel_fields_from_element(band, fields)

        # Emission wavelength inference from fluorophore name is specific to
        # the OPAL scan-band format (e.g. "OPAL520" → 520 nm).
        if "emission_wavelength_nm" not in fields:
            fluor = fields.get("fluorophore")
            if isinstance(fluor, str):
                m = re.search(r"(\d{3,4})", fluor)
                if m:
                    fields["emission_wavelength_nm"] = float(m.group(1))

        # Exposure times can arrive via ExposureTimeArray rather than per-band.
        if "exposure_time_us" not in fields:
            if idx < len(exposure_times) and exposure_times[idx] is not None:
                fields["exposure_time_us"] = exposure_times[idx]

        channels.append(ChannelInfo(**fields))  # type: ignore[arg-type]

    return channels


def _parse_channels_from_per_page_xmls(
    per_page_xmls: List[str],
) -> List[ChannelInfo]:
    """
    Extract per-channel info from individual page XMLs.

    Newer PerkinElmer Fusion QPTIFF files embed per-channel metadata directly
    at the root level of each page's ImageDescription XML (one page per channel),
    rather than grouping them in ScanBands-i elements.
    """
    channels: List[ChannelInfo] = []
    for idx, xml in enumerate(per_page_xmls):
        fields: Dict[str, object] = {"index": idx, "name": f"Channel_{idx}"}

        if xml:
            try:
                page_root = ET.fromstring(xml)
                _populate_channel_fields_from_element(page_root, fields)
            except ET.ParseError:
                pass

        channels.append(ChannelInfo(**fields))  # type: ignore[arg-type]
    return channels


def _populate_channel_fields_from_element(
    elem: ET.Element, fields: Dict[str, object]
) -> None:
    """
    Pull every per-channel field we know how to extract out of a single XML
    element (either a <ScanBands-i> or a page root) and write it into ``fields``.

    Mutates ``fields`` in place. ``fields`` is a dict that will be passed to
    ``ChannelInfo(**fields)``; keys must match ChannelInfo attribute names.
    Values are only set when they parse successfully, so defaults on the
    dataclass remain in effect when a field is missing from this file.
    """
    # Biomarker / stain name
    for tag in _BIOMARKER_TAGS:
        el = elem.find(tag)
        if el is not None and el.text:
            val = el.text.strip()
            if val and val not in ("None", "--"):
                fields["name"] = val
                break

    # fluorophore
    for tag in _FLUOROPHORE_TAGS:
        el = elem.find(tag)
        if el is not None and el.text and el.text.strip():
            fields["fluorophore"] = el.text.strip()
            break

    # exposure time (us per spec)
    et_text = _text(elem.find("ExposureTime"))
    if et_text is not None:
        try:
            fields["exposure_time_us"] = float(et_text)
        except ValueError:
            pass

    # emission / excitation wavelengths. Filters may list multiple bands (one
    # per channel); prefer the one flagged <Active>true</Active>, else the
    # first band.
    em_band = _select_active_band(elem.find(".//EmissionFilter"))
    if em_band is not None:
        cuton = _float(em_band.find("Cuton"))
        cutoff = _float(em_band.find("Cutoff"))
        if cuton is not None and cutoff is not None:
            fields["emission_wavelength_nm"] = (cuton + cutoff) / 2.0

    ex_band = _select_active_band(elem.find(".//ExcitationFilter"))
    if ex_band is not None:
        cuton = _float(ex_band.find("Cuton"))
        cutoff = _float(ex_band.find("Cutoff"))
        if cuton is not None and cutoff is not None:
            fields["excitation_wavelength_nm"] = (cuton + cutoff) / 2.0

    # fallback to a <HomeWavelength> element for old ScanBands format
    if "emission_wavelength_nm" not in fields:
        hw = _float(elem.find(".//HomeWavelength"))
        if hw is not None:
            fields["emission_wavelength_nm"] = hw
    if "excitation_wavelength_nm" not in fields:
        ex = _float(elem.find(".//ExcitationWavelength"))
        if ex is not None:
            fields["excitation_wavelength_nm"] = ex

    # display colour, signal type, unmixed flag, per-channel detector gain/binning
    color = _parse_color(elem.find("Color"))
    if color is None:
        color = _parse_color(elem.find(".//Color"))
    if color is not None:
        fields["color_rgb"] = color

    _set_if(fields, "is_unmixed_component", _bool(elem.find("IsUnmixedComponent")))
    _set_if(fields, "signal_units", _int(elem.find("SignalUnits")))

    # per chan objective, scale factor, AF subtraction
    _set_if(fields, "objective", _text(elem.find("Objective")))
    _set_if(fields, "scale_factor", _float(elem.find("ScaleFactor")))
    _set_if(
        fields,
        "autofluorescence_subtracted",
        _bool(elem.find("AutofluorescenceSubtracted")),
    )

    # responsivity / calibration
    resp = elem.find("Responsivity")
    if resp is not None:
        rf = resp.find("Filter")
        if rf is not None:
            _set_if(fields, "responsivity", _float(rf.find("Response")))
            _set_if(fields, "responsivity_filter_id", _text(rf.find("FilterID")))
            _set_if(fields, "responsivity_date", _text(rf.find("Date")))
            _set_if(fields, "responsivity_filter_name", _text(rf.find("Name")))

    # xcitation / emission filter identification
    ex_filter = elem.find("ExcitationFilter")
    if ex_filter is not None:
        _set_if(fields, "excitation_filter_name", _text(ex_filter.find("Name")))
        _set_if(
            fields,
            "excitation_filter_manufacturer",
            _text(ex_filter.find("Manufacturer")),
        )
        _set_if(fields, "excitation_filter_part_no", _text(ex_filter.find("PartNo")))

    em_filter = elem.find("EmissionFilter")
    if em_filter is not None:
        _set_if(fields, "emission_filter_name", _text(em_filter.find("Name")))
        _set_if(
            fields,
            "emission_filter_manufacturer",
            _text(em_filter.find("Manufacturer")),
        )
        _set_if(fields, "emission_filter_part_no", _text(em_filter.find("PartNo")))

    # per channel detectors
    cs = elem.find("CameraSettings")
    if cs is not None:
        _set_if(fields, "gain", _float(cs.find("Gain")))
        _set_if(fields, "binning", _int(cs.find("Binning")))
        _set_if(fields, "bit_depth", _int(cs.find("BitDepth")))
        _set_if(fields, "offset_counts", _int(cs.find("OffsetCounts")))
        _set_if(fields, "camera_orientation", _text(cs.find("Orientation")))
        roi = cs.find("ROI")
        if roi is not None:
            _set_if(fields, "roi_x", _int(roi.find("X")))
            _set_if(fields, "roi_y", _int(roi.find("Y")))
            _set_if(fields, "roi_width", _int(roi.find("Width")))
            _set_if(fields, "roi_height", _int(roi.find("Height")))
    else:
        # older ScanBands format stores Gain/Binning directly on the band
        _set_if(fields, "gain", _float(elem.find(".//Gain")))
        _set_if(fields, "binning", _int(elem.find(".//Binning")))


def _select_active_band(
    filter_elem: Optional[ET.Element],
) -> Optional[ET.Element]:
    """Return the <Band> flagged <Active>true</Active>, else the first band.

    Fusion 1.x per-page XMLs list every band of the multi-band filter; only one
    is marked active for that channel. Older formats omit <Active> entirely, so
    we fall back to the first band to preserve prior behaviour.
    """
    if filter_elem is None:
        return None
    bands = filter_elem.findall("Bands/Band")
    if not bands:
        return None
    for b in bands:
        active = b.find("Active")
        if active is not None and active.text and active.text.strip().lower() in (
            "true",
            "1",
            "yes",
        ):
            return b
    return bands[0]


def _parse_scan_profile_json(
    image_info: ImageInfo, scan_profile_text: str
) -> None:
    """Pull structured fields out of a Fusion 1.x JSON ScanProfile.

    Fusion stores ScanProfile as a JSON blob rather than nested XML. Extract
    the fields that map onto existing ImageInfo slots; leave the rest for
    callers who parse ``raw_xml`` themselves.
    """
    try:
        sp = json.loads(scan_profile_text)
    except (ValueError, TypeError) as exc:
        log.debug("ScanProfile JSON parse failed: %s", exc)
        return
    if not isinstance(sp, dict):
        return

    if image_info.is_tma is None and isinstance(sp.get("isTma"), bool):
        image_info.is_tma = sp["isTma"]

    if image_info.scan_resolution.binning is None:
        binning = sp.get("binning")
        if isinstance(binning, int):
            image_info.scan_resolution.binning = binning

    exp = sp.get("experimentDescription")
    if isinstance(exp, dict):
        if image_info.scan_profile_name is None:
            name = exp.get("name")
            if isinstance(name, str) and name:
                image_info.scan_profile_name = name


def _set_if(target: Dict[str, object], key: str, value: object) -> None:
    """Write ``value`` into ``target[key]`` only when it is not None."""
    if value is not None:
        target[key] = value


def _detect_format(
    root: ET.Element,
    scan_mode: str,
    bf_lamp_type: Optional[str],
) -> str:
    """
    Identify which QPTIFF format variant produced this file.. a couple of versions
    as qptiff evolved from vectra

    1. **brightfield** — scan_mode contains "Brightfield" (case-insensitive) or
       a ``<BFLampType>`` element is present (brightfield-only tag).
    2. **polaris_scanband** — ``<ScanBands-i>`` elements are present anywhere in
       the tree (older Vectra/Polaris/OPAL XML ScanProfile format).
    3. **fusion_paged** — ``<ScanProfile>`` content is a JSON object (newer
       Akoya Biosciences / Fusion 1.x format where channel metadata is
       embedded per-page rather than in a shared ScanProfile).
    4. **unknown** — no recognised structural signal; caller should attempt
       strategies in sequence.
    """
    if "brightfield" in scan_mode.lower() or bf_lamp_type is not None:
        return FORMAT_BRIGHTFIELD

    if root.findall(".//ScanBands-i"):
        return FORMAT_POLARIS_SCANBAND

    sp = root.find("ScanProfile")
    if sp is not None and sp.text and sp.text.strip().startswith("{"):
        return FORMAT_FUSION_PAGED

    return FORMAT_UNKNOWN


def parse_qpi_xml(
    xml_string: str,
    n_channels: Optional[int] = None,
    datetime_str: Optional[str] = None,
    per_page_xmls: Optional[List[str]] = None,
) -> QptiffMetadata:
    """
    Parse the PerkinElmer QPI ImageDescription XML and return a QptiffMetadata.

    xml_string:
        The raw XML string from TIFF tag 270 of the first page.
    n_channels:
        Number of channels in the image, used as fallback when XML channel
        descriptions are absent (e.g. brightfield RGB → 3).
    datetime_str:
        Value from TIFF DateTime tag (306), added to metadata as-is.
    per_page_xmls:
        List of raw XML strings, one per channel page.  Used for newer
        PerkinElmer Fusion files where channel-level metadata (Biomarker,
        ExposureTime, filters) is stored per-page rather than in ScanBands-i.


    QptiffMetadata
        Fully populated (where data is available) metadata object.
    """
    slide = SlideInfo(datetime=datetime_str)
    image_info = ImageInfo()
    channels: List[ChannelInfo] = []

    if not xml_string:
        return QptiffMetadata(
            slide=slide,
            images=[QptiffImageScene(image_info=image_info, raw_xml=xml_string)],
            raw_xml=xml_string,
        )

    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError as exc:
        log.warning("Failed to parse QPI XML: %s", exc)
        return QptiffMetadata(
            slide=slide,
            images=[QptiffImageScene(image_info=image_info, raw_xml=xml_string)],
            raw_xml=xml_string,
        )

    if "PerkinElmer-QPI" not in root.tag and "PerkinElmerQPI" not in root.tag:
        log.debug("XML root %r not a recognised QPI description", root.tag)

    # slide-scope fields
    slide.description_version = _text(root.find("DescriptionVersion"))
    slide.acquisition_software = _text(root.find("AcquisitionSoftware"))
    slide.identifier = _text(root.find("Identifier"))
    slide.slide_id = _text(root.find("SlideID"))
    slide.barcode = _text(root.find("Barcode")) or None
    slide.study_name = _text(root.find("StudyName"))
    slide.operator_name = _text(root.find("OperatorName"))
    slide.computer_name = _text(root.find("ComputerName"))
    slide.instrument_type = _text(root.find("InstrumentType"))

    # image-scope fields
    image_info.image_type = _text(root.find("ImageType"))
    image_info.bf_lamp_type = _text(root.find("BFLampType"))
    image_info.objective = _text(root.find("Objective"))

    sp = root.find("ScanProfile")
    if sp is not None:
        _nested = sp.find("root")
        sp_root = _nested if _nested is not None else sp
        image_info.scan_profile_name = _text(sp_root.find("Name"))
        image_info.scan_mode = _text(sp_root.find("Mode"))
        image_info.is_tma = _bool(sp_root.find("SampleIsTMA"))
        image_info.opal_kit_type = _text(sp_root.find("OpalKitType"))

    image_info.scan_resolution = _parse_scan_resolution(root)

    # Fusion 1.x stores ScanProfile as a JSON blob; fill any still-missing
    # fields from it. Done after _parse_scan_resolution so the XML-form values
    # take priority.
    if sp is not None and sp.text and sp.text.strip().startswith("{"):
        _parse_scan_profile_json(image_info, sp.text)

    image_info.camera = _parse_camera(root)

    exposure_times = _parse_exposure_times(root)

    fmt = _detect_format(root, image_info.scan_mode or "", image_info.bf_lamp_type)
    log.debug("Detected QPTIFF format: %s (slide=%s)", fmt, slide.slide_id)

    if fmt == FORMAT_BRIGHTFIELD:
        n = n_channels if n_channels and n_channels > 0 else 3
        channels = _parse_brightfield_channels(n)

    elif fmt == FORMAT_POLARIS_SCANBAND:
        channels = _parse_fluorescence_channels(root, exposure_times)

    elif fmt == FORMAT_FUSION_PAGED:
        if per_page_xmls:
            channels = _parse_channels_from_per_page_xmls(per_page_xmls)
        elif n_channels and n_channels > 0:
            channels = [
                ChannelInfo(
                    index=i,
                    name=f"Channel_{i}",
                    exposure_time_us=(
                        exposure_times[i] if i < len(exposure_times) else None
                    ),
                )
                for i in range(n_channels)
            ]

    else:
        flu_channels = _parse_fluorescence_channels(root, exposure_times)
        if flu_channels:
            channels = flu_channels
        elif per_page_xmls:
            channels = _parse_channels_from_per_page_xmls(per_page_xmls)
        elif n_channels and n_channels > 0:
            channels = [
                ChannelInfo(
                    index=i,
                    name=f"Channel_{i}",
                    exposure_time_us=(
                        exposure_times[i] if i < len(exposure_times) else None
                    ),
                )
                for i in range(n_channels)
            ]

    # supplement missing exposure times where possible
    for ch in channels:
        if ch.exposure_time_us is None and ch.index < len(exposure_times):
            ch.exposure_time_us = exposure_times[ch.index]

    image = QptiffImageScene(
        image_info=image_info,
        channels=channels,
        raw_xml=xml_string,
    )
    return QptiffMetadata(
        slide=slide,
        images=[image],
        acquisition_format=fmt,
        raw_xml=xml_string,
    )


TIFF_IMAGE_DESCRIPTION_TAG = 270
TIFF_DATETIME_TAG = 306


def extract_qpi_xml_from_page(page: object) -> str:
    """
    Return the raw XML string from a tifffile page's ImageDescription tag,
    only if it is a PerkinElmer QPI XML block.
    Returns empty string if the tag is absent, malformed, or not QPI XML.
    """
    try:
        tags = page.tags  # type: ignore[attr-defined]
        tag = tags.get(TIFF_IMAGE_DESCRIPTION_TAG)
        if tag is None:
            return ""
        val = tag.value
        if not isinstance(val, str) or not val:
            return ""
        # only return XML recognised as a QPI description
        try:
            root = ET.fromstring(val)
            if "PerkinElmer-QPI" not in root.tag and "PerkinElmerQPI" not in root.tag:
                return ""
        except ET.ParseError:
            return ""
        return val
    except Exception as exc:
        log.debug("Could not read ImageDescription tag: %s", exc)
        return ""


def extract_datetime_from_page(page: object) -> Optional[str]:
    """Return the DateTime string from a tifffile page, or None."""
    try:
        tags = page.tags  # type: ignore[attr-defined]
        tag = tags.get(TIFF_DATETIME_TAG)
        if tag is None:
            return None
        return tag.value if isinstance(tag.value, str) else None
    except Exception:
        return None
