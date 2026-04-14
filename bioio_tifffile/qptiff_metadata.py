#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PerkinElmer QPI (QPTIFF) metadata parser.

Parses the XML embedded in ImageDescription tag (TIFF tag 270) of QPTIFF files
produced by PerkinElmer Vectra/Polaris/Fusion/CODEX instruments.
"""

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
class ChannelInfo:
    """Metadata for a single image channel."""

    index: int
    name: str  # biomarker / stain or fluorophore name, or rgb for brightfield, H&E, etc.
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


@dataclass
class ScanResolutionInfo:
    """Pixel size and objective information from the scan profile."""

    pixel_size_um: Optional[float] = None  # physical pixel size in micrometres
    magnification: Optional[float] = None
    objective_name: Optional[str] = None
    binning: Optional[int] = None


@dataclass
class CameraInfo:
    """Camera / detector settings."""

    camera_name: Optional[str] = None
    camera_type: Optional[str] = None
    gain: Optional[float] = None
    bit_depth: Optional[int] = None
    binning: Optional[int] = None


@dataclass
class QptiffMetadata:
    """
    Structured metadata extracted from a PerkinElmer QPI QPTIFF file.

    All fields are optional and populated only when the corresponding XML
    element is present in the file.
    """

    # --- File / acquisition identity ---
    description_version: Optional[str] = None
    acquisition_software: Optional[str] = None
    image_type: Optional[str] = None  # "FullResolution", "Thumbnail", "Macro", "Label"
    identifier: Optional[str] = None  # UUID
    slide_id: Optional[str] = None
    barcode: Optional[str] = None
    study_name: Optional[str] = None
    operator_name: Optional[str] = None
    computer_name: Optional[str] = None
    datetime: Optional[str] = None  # from TIFF tag 306

    # --- Instrument ---
    instrument_type: Optional[str] = None
    bf_lamp_type: Optional[str] = None
    objective: Optional[str] = None
    scan_profile_name: Optional[str] = None  # e.g. "Brightfield_TMA"
    scan_mode: Optional[str] = None  # e.g. "im_Brightfield"
    is_tma: Optional[bool] = None
    opal_kit_type: Optional[str] = None

    # --- Camera ---
    camera: CameraInfo = field(default_factory=CameraInfo)

    # --- Pixel geometry ---
    scan_resolution: ScanResolutionInfo = field(default_factory=ScanResolutionInfo)

    # --- Physical position (from TIFF XPosition/YPosition tags, in µm) ---
    xposition_um: Optional[float] = None
    yposition_um: Optional[float] = None

    # --- Channels ---
    channels: List[ChannelInfo] = field(default_factory=list)

    # --- Detected format ---
    acquisition_format: Optional[str] = None  # one of the FORMAT_* constants

    # --- Raw ---
    raw_xml: str = ""

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def pixel_size_um(self) -> Optional[float]:
        """Physical pixel size in µm (shortcut for scan_resolution.pixel_size_um)."""
        return self.scan_resolution.pixel_size_um

    @property
    def channel_names(self) -> List[str]:
        """Ordered list of channel names."""
        return [ch.name for ch in self.channels]

    @property
    def is_brightfield(self) -> bool:
        """True when the image is a brightfield (H&E / IHC) scan."""
        return any(ch.is_brightfield for ch in self.channels) or (
            self.scan_mode is not None and "Brightfield" in self.scan_mode
        )

    def to_dict(self) -> Dict:
        """
        Flat dictionary for xarray attrs.

        OME-compliant field names are used where an OME equivalent exists
        (e.g. ``Pixels:PhysicalSizeX``, ``Channel:0:Name``).  Fields that are
        specific to the PerkinElmer QPTIFF format with no OME equivalent retain
        the ``qpi_`` prefix.

        Key naming conventions
        ----------------------
        ``Pixels:*``             → OME Pixels attributes
        ``Plane:N:*``            → OME Plane attributes for channel index N
        ``Channel:N:*``          → OME Channel attributes for channel index N
        ``DetectorSettings:N:*`` → OME DetectorSettings for channel index N
        ``Objective:*``          → OME Objective attributes
        ``Microscope:*``         → OME Microscope / Instrument attributes
        ``Detector:*``           → OME Detector attributes
        ``Experimenter:*``       → OME Experimenter attributes
        ``qpi_*``                → QPTIFF-specific, no OME equivalent
        """
        d: Dict = {
            # ---- OME Pixels ----
            "Pixels:PhysicalSizeX": self.pixel_size_um,
            "Pixels:PhysicalSizeXUnit": "µm" if self.pixel_size_um is not None else None,
            "Pixels:PhysicalSizeY": self.pixel_size_um,
            "Pixels:PhysicalSizeYUnit": "µm" if self.pixel_size_um is not None else None,
            # ---- OME Plane (baseline / first plane positional metadata) ----
            "Plane:PositionX": self.xposition_um,
            "Plane:PositionXUnit": "µm" if self.xposition_um is not None else None,
            "Plane:PositionY": self.yposition_um,
            "Plane:PositionYUnit": "µm" if self.yposition_um is not None else None,
            # ---- OME Experimenter ----
            "Experimenter:UserName": self.operator_name,
            # ---- OME Instrument / Microscope ----
            "Microscope:Model": self.instrument_type,
            # ---- OME Objective ----
            "Objective:Model": self.scan_resolution.objective_name,
            "Objective:NominalMagnification": self.scan_resolution.magnification,
            # ---- OME Detector ----
            "Detector:Model": self.camera.camera_type,
            # ---- QPTIFF-specific (no OME equivalent) ----
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
        # Per-channel OME fields (Channel, Plane, DetectorSettings)
        for ch in self.channels:
            i = ch.index
            # OME Channel
            d[f"Channel:{i}:Name"] = ch.name
            d[f"Channel:{i}:Fluor"] = ch.fluorophore
            d[f"Channel:{i}:Color"] = (
                f"{ch.color_rgb[0]},{ch.color_rgb[1]},{ch.color_rgb[2]}"
                if ch.color_rgb
                else None
            )
            d[f"Channel:{i}:EmissionWavelength"] = ch.emission_wavelength_nm
            d[f"Channel:{i}:EmissionWavelengthUnit"] = (
                "nm" if ch.emission_wavelength_nm is not None else None
            )
            d[f"Channel:{i}:ExcitationWavelength"] = ch.excitation_wavelength_nm
            d[f"Channel:{i}:ExcitationWavelengthUnit"] = (
                "nm" if ch.excitation_wavelength_nm is not None else None
            )
            # OME Plane (Z=0, T=0, C=i)
            d[f"Plane:{i}:ExposureTime"] = ch.exposure_time_us
            d[f"Plane:{i}:ExposureTimeUnit"] = (
                "µs" if ch.exposure_time_us is not None else None
            )
            # OME DetectorSettings
            d[f"DetectorSettings:{i}:Gain"] = ch.gain
            d[f"DetectorSettings:{i}:Binning"] = (
                f"{ch.binning}x{ch.binning}" if ch.binning is not None else None
            )
            # QPTIFF-specific per-channel fields
            d[f"qpi_ch{i}_is_unmixed"] = ch.is_unmixed_component
            d[f"qpi_ch{i}_signal_units"] = ch.signal_units
        return {k: v for k, v in d.items() if v is not None}


###############################################################################
# OME-structured metadata dataclasses
###############################################################################


@dataclass
class OMEChannel:
    """
    OME Channel + DetectorSettings + per-plane metadata for one channel.

    Field names follow the OME-XML specification:
    https://ome-model.readthedocs.io/en/stable/ome-xml/
    """

    index: int
    # OME Channel
    name: Optional[str] = None  # Channel.Name
    fluor: Optional[str] = None  # Channel.Fluor
    color: Optional[Tuple[int, int, int]] = None  # Channel.Color (r, g, b)
    emission_wavelength_nm: Optional[float] = None  # Channel.EmissionWavelength
    excitation_wavelength_nm: Optional[float] = None  # Channel.ExcitationWavelength
    # OME Plane (Z=0, T=0, C=index)
    exposure_time_us: Optional[float] = None  # Plane.ExposureTime (µs)
    # OME DetectorSettings
    gain: Optional[float] = None  # DetectorSettings.Gain
    binning: Optional[str] = None  # DetectorSettings.Binning ("NxN")
    # QPTIFF-specific (no OME equivalent)
    is_unmixed_component: Optional[bool] = None
    signal_units: Optional[int] = None


@dataclass
class OMEInstrument:
    """OME Instrument / Objective / Detector metadata."""

    microscope_model: Optional[str] = None  # Microscope.Model ← InstrumentType
    detector_model: Optional[str] = None  # Detector.Model ← CameraType
    objective_model: Optional[str] = None  # Objective.Model
    objective_magnification: Optional[float] = None  # Objective.NominalMagnification
    # Extra — camera name not in OME but useful for identification
    camera_name: Optional[str] = None


@dataclass
class OMEMetadata:
    """
    OME-structured metadata for a single QPTIFF scene.

    Populated from the PerkinElmer QPI XML and TIFF tags.  Fields mirror the
    OME-XML data model (https://ome-model.readthedocs.io/en/stable/ome-xml/).
    Fields with no OME equivalent are kept with a ``qpi_`` prefix on the
    :class:`QptiffMetadata` source object.
    """

    # ---- OME Pixels ----
    physical_size_x_um: Optional[float] = None  # Pixels.PhysicalSizeX
    physical_size_y_um: Optional[float] = None  # Pixels.PhysicalSizeY
    size_x: Optional[int] = None  # Pixels.SizeX
    size_y: Optional[int] = None  # Pixels.SizeY
    size_z: Optional[int] = None  # Pixels.SizeZ
    size_c: Optional[int] = None  # Pixels.SizeC
    size_t: Optional[int] = None  # Pixels.SizeT

    # ---- OME Image ----
    image_name: Optional[str] = None  # Image.Name

    # ---- OME Experimenter ----
    experimenter: Optional[str] = None  # Experimenter.UserName ← OperatorName

    # ---- OME Plane (baseline plane, first channel) ----
    position_x_um: Optional[float] = None  # Plane.PositionX
    position_y_um: Optional[float] = None  # Plane.PositionY

    # ---- OME Instrument / Objective / Detector ----
    instrument: OMEInstrument = field(default_factory=OMEInstrument)

    # ---- OME Channels ----
    channels: List[OMEChannel] = field(default_factory=list)

    # ---- Source ----
    # The raw QptiffMetadata that produced this object
    _source: Optional["QptiffMetadata"] = field(default=None, repr=False, compare=False)


def ome_metadata_from_qptiff(
    qpi: "QptiffMetadata",
    scene_name: Optional[str] = None,
    size_x: Optional[int] = None,
    size_y: Optional[int] = None,
    size_z: Optional[int] = None,
    size_c: Optional[int] = None,
    size_t: Optional[int] = None,
) -> OMEMetadata:
    """
    Build an :class:`OMEMetadata` from a :class:`QptiffMetadata` instance.

    Parameters
    ----------
    qpi:
        Parsed QPTIFF metadata.
    scene_name:
        Human-readable name of the scene (becomes ``Image.Name``).
    size_x, size_y, size_z, size_c, size_t:
        Pixel dimensions of the image at the selected pyramid level.
    """
    channels = [
        OMEChannel(
            index=ch.index,
            name=ch.name,
            fluor=ch.fluorophore,
            color=ch.color_rgb,
            emission_wavelength_nm=ch.emission_wavelength_nm,
            excitation_wavelength_nm=ch.excitation_wavelength_nm,
            exposure_time_us=ch.exposure_time_us,
            gain=ch.gain,
            binning=(
                f"{ch.binning}x{ch.binning}" if ch.binning is not None else None
            ),
            is_unmixed_component=ch.is_unmixed_component,
            signal_units=ch.signal_units,
        )
        for ch in qpi.channels
    ]

    instrument = OMEInstrument(
        microscope_model=qpi.instrument_type,
        detector_model=qpi.camera.camera_type,
        objective_model=qpi.scan_resolution.objective_name,
        objective_magnification=qpi.scan_resolution.magnification,
        camera_name=qpi.camera.camera_name,
    )

    return OMEMetadata(
        physical_size_x_um=qpi.pixel_size_um,
        physical_size_y_um=qpi.pixel_size_um,
        size_x=size_x,
        size_y=size_y,
        size_z=size_z,
        size_c=size_c,
        size_t=size_t,
        image_name=scene_name,
        experimenter=qpi.operator_name,
        position_x_um=qpi.xposition_um,
        position_y_um=qpi.yposition_um,
        instrument=instrument,
        channels=channels,
        _source=qpi,
    )


###############################################################################
# Parser
###############################################################################

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
        pixel_size_um=pixel_size,
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
    """Return list of per-channel exposure times (µs) from ExposureTimeArray.

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
    ScanBands-i element.  We extract:
    - biomarker name  (Biomarker / BioMarker / StainName / Marker)
    - fluorophore     (Fluorophore / Fluor)
    - emission wavelength (from filter band or fluorophore name)
    """
    channels: List[ChannelInfo] = []
    scan_bands = root.findall(".//ScanBands-i")

    for idx, band in enumerate(scan_bands):
        # Try to find biomarker name
        name: Optional[str] = None
        for tag in _BIOMARKER_TAGS:
            el = band.find(f".//{tag}")
            if el is not None and el.text and el.text.strip() not in ("", "None"):
                name = el.text.strip()
                break
        if name is None:
            name = f"Channel_{idx}"

        # Fluorophore
        fluorophore: Optional[str] = None
        for tag in _FLUOROPHORE_TAGS:
            el = band.find(f".//{tag}")
            if el is not None and el.text and el.text.strip() not in ("", "None"):
                fluorophore = el.text.strip()
                break

        # Emission wavelength — try HomeWavelength first, infer from fluorophore name
        emission_nm = _float(band.find(".//HomeWavelength"))
        if emission_nm is None and fluorophore:
            m = re.search(r"(\d{3,4})", fluorophore)
            if m:
                emission_nm = float(m.group(1))

        # Excitation wavelength
        excitation_nm = _float(band.find(".//ExcitationWavelength"))

        exposure_us = exposure_times[idx] if idx < len(exposure_times) else None

        color_rgb = _parse_color(band.find(".//Color"))
        is_unmixed = _bool(band.find(".//IsUnmixedComponent"))
        signal_units_el = band.find(".//SignalUnits")
        signal_units = _int(signal_units_el)
        gain = _float(band.find(".//Gain"))
        binning = _int(band.find(".//Binning"))

        channels.append(
            ChannelInfo(
                index=idx,
                name=name,
                fluorophore=fluorophore,
                exposure_time_us=exposure_us,
                emission_wavelength_nm=emission_nm,
                excitation_wavelength_nm=excitation_nm,
                is_brightfield=False,
                color_rgb=color_rgb,
                is_unmixed_component=is_unmixed,
                signal_units=signal_units,
                gain=gain,
                binning=binning,
            )
        )

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
        name = f"Channel_{idx}"
        fluorophore: Optional[str] = None
        exposure_us: Optional[float] = None
        emission_nm: Optional[float] = None
        excitation_nm: Optional[float] = None
        color_rgb: Optional[Tuple[int, int, int]] = None
        is_unmixed: Optional[bool] = None
        signal_units: Optional[int] = None
        gain: Optional[float] = None
        binning: Optional[int] = None

        if xml:
            try:
                page_root = ET.fromstring(xml)
                # Search only at direct children to avoid nested <Name> tags
                for tag in _BIOMARKER_TAGS:
                    el = page_root.find(tag)
                    if el is not None and el.text:
                        val = el.text.strip()
                        if val and val not in ("None", "--"):
                            name = val
                            break
                # Fluorophore
                for tag in _FLUOROPHORE_TAGS:
                    el = page_root.find(tag)
                    if el is not None and el.text and el.text.strip():
                        fluorophore = el.text.strip()
                        break
                # Exposure time (µs per spec)
                et_el = page_root.find("ExposureTime")
                if et_el is not None and et_el.text:
                    try:
                        exposure_us = float(et_el.text.strip())
                    except ValueError:
                        pass
                # Emission wavelength — midpoint of first emission band
                em_band = page_root.find(".//EmissionFilter/Bands/Band")
                if em_band is not None:
                    cuton = _float(em_band.find("Cuton"))
                    cutoff = _float(em_band.find("Cutoff"))
                    if cuton is not None and cutoff is not None:
                        emission_nm = (cuton + cutoff) / 2.0
                # Excitation wavelength — midpoint of first excitation band
                ex_band = page_root.find(".//ExcitationFilter/Bands/Band")
                if ex_band is not None:
                    cuton = _float(ex_band.find("Cuton"))
                    cutoff = _float(ex_band.find("Cutoff"))
                    if cuton is not None and cutoff is not None:
                        excitation_nm = (cuton + cutoff) / 2.0
                # Display colour, signal type, unmixed flag, detector settings
                color_rgb = _parse_color(page_root.find("Color"))
                is_unmixed = _bool(page_root.find("IsUnmixedComponent"))
                signal_units = _int(page_root.find("SignalUnits"))
                gain = _float(page_root.find("Gain"))
                binning = _int(page_root.find("Binning"))
            except ET.ParseError:
                pass

        channels.append(
            ChannelInfo(
                index=idx,
                name=name,
                fluorophore=fluorophore,
                exposure_time_us=exposure_us,
                emission_wavelength_nm=emission_nm,
                excitation_wavelength_nm=excitation_nm,
                is_brightfield=False,
                color_rgb=color_rgb,
                is_unmixed_component=is_unmixed,
                signal_units=signal_units,
                gain=gain,
                binning=binning,
            )
        )
    return channels


def _detect_format(
    root: ET.Element,
    scan_mode: str,
    bf_lamp_type: Optional[str],
) -> str:
    """
    Identify which QPTIFF format variant produced this file.

    Detection priority
    ------------------
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

    Parameters
    ----------
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

    Returns
    -------
    QptiffMetadata
        Fully populated (where data is available) metadata object.
    """
    meta = QptiffMetadata(raw_xml=xml_string, datetime=datetime_str)

    if not xml_string:
        return meta

    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError as exc:
        log.warning("Failed to parse QPI XML: %s", exc)
        return meta

    # Validate root tag
    if "PerkinElmer-QPI" not in root.tag and "PerkinElmerQPI" not in root.tag:
        log.debug("XML root %r not a recognised QPI description", root.tag)

    # --- Top-level scalars ---
    meta.description_version = _text(root.find("DescriptionVersion"))
    meta.acquisition_software = _text(root.find("AcquisitionSoftware"))
    meta.image_type = _text(root.find("ImageType"))
    meta.identifier = _text(root.find("Identifier"))
    meta.slide_id = _text(root.find("SlideID"))
    meta.barcode = _text(root.find("Barcode")) or None
    meta.study_name = _text(root.find("StudyName"))
    meta.operator_name = _text(root.find("OperatorName"))
    meta.computer_name = _text(root.find("ComputerName"))
    meta.instrument_type = _text(root.find("InstrumentType"))
    meta.bf_lamp_type = _text(root.find("BFLampType"))
    meta.objective = _text(root.find("Objective"))

    # --- Scan profile fields ---
    sp = root.find("ScanProfile")
    if sp is not None:
        _nested = sp.find("root")
        sp_root = _nested if _nested is not None else sp
        meta.scan_profile_name = _text(sp_root.find("Name"))
        meta.scan_mode = _text(sp_root.find("Mode"))
        meta.is_tma = _bool(sp_root.find("SampleIsTMA"))
        meta.opal_kit_type = _text(sp_root.find("OpalKitType"))

    # --- Pixel size / objective ---
    meta.scan_resolution = _parse_scan_resolution(root)

    # --- Camera ---
    meta.camera = _parse_camera(root)

    # --- Exposure times ---
    exposure_times = _parse_exposure_times(root)

    # --- Detect format and parse channels conditionally ---
    fmt = _detect_format(root, meta.scan_mode or "", meta.bf_lamp_type)
    meta.acquisition_format = fmt
    log.debug("Detected QPTIFF format: %s (slide=%s)", fmt, meta.slide_id)

    if fmt == FORMAT_BRIGHTFIELD:
        # Brightfield H&E / IHC: R/G/B samples stored in the S dimension.
        n = n_channels if n_channels and n_channels > 0 else 3
        meta.channels = _parse_brightfield_channels(n)

    elif fmt == FORMAT_POLARIS_SCANBAND:
        # Older Vectra/Polaris/OPAL: channel info in <ScanBands-i> XML elements.
        meta.channels = _parse_fluorescence_channels(root, exposure_times)

    elif fmt == FORMAT_FUSION_PAGED:
        # Newer Akoya Biosciences / Fusion 1.x: each TIFF page carries its own
        # <Biomarker> tag — collect names from the per-page XML list.
        if per_page_xmls:
            meta.channels = _parse_channels_from_per_page_xmls(per_page_xmls)
        elif n_channels and n_channels > 0:
            meta.channels = [
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
        # Unknown format: try each strategy in sequence.
        flu_channels = _parse_fluorescence_channels(root, exposure_times)
        if flu_channels:
            meta.channels = flu_channels
        elif per_page_xmls:
            meta.channels = _parse_channels_from_per_page_xmls(per_page_xmls)
        elif n_channels and n_channels > 0:
            meta.channels = [
                ChannelInfo(
                    index=i,
                    name=f"Channel_{i}",
                    exposure_time_us=(
                        exposure_times[i] if i < len(exposure_times) else None
                    ),
                )
                for i in range(n_channels)
            ]

    # Supplement missing exposure times where possible
    for ch in meta.channels:
        if ch.exposure_time_us is None and ch.index < len(exposure_times):
            ch.exposure_time_us = exposure_times[ch.index]

    return meta


###############################################################################
# Utility: extract XML from a TiffFile page
###############################################################################

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
        # Only return XML recognised as a QPI description
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
