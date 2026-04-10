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
from typing import Dict, List, Optional

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
    name: str  # biomarker / stain or fluorphore name, or rgb for brightfield, HE,etc.
    fluorophore: Optional[str] = None
    exposure_time_ms: Optional[float] = None
    emission_wavelength_nm: Optional[float] = None
    excitation_wavelength_nm: Optional[float] = None
    is_brightfield: bool = False


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
        """Flat dictionary representation suitable for xarray attrs."""
        d: Dict = {
            "qpi_description_version": self.description_version,
            "qpi_acquisition_software": self.acquisition_software,
            "qpi_image_type": self.image_type,
            "qpi_identifier": self.identifier,
            "qpi_slide_id": self.slide_id,
            "qpi_barcode": self.barcode,
            "qpi_study_name": self.study_name,
            "qpi_operator_name": self.operator_name,
            "qpi_computer_name": self.computer_name,
            "qpi_datetime": self.datetime,
            "qpi_instrument_type": self.instrument_type,
            "qpi_bf_lamp_type": self.bf_lamp_type,
            "qpi_objective": self.objective,
            "qpi_scan_profile_name": self.scan_profile_name,
            "qpi_scan_mode": self.scan_mode,
            "qpi_is_tma": self.is_tma,
            "qpi_opal_kit_type": self.opal_kit_type,
            "qpi_acquisition_format": self.acquisition_format,
            "qpi_pixel_size_um": self.pixel_size_um,
            "qpi_magnification": self.scan_resolution.magnification,
            "qpi_objective_name": self.scan_resolution.objective_name,
            "qpi_camera_name": self.camera.camera_name,
            "qpi_camera_type": self.camera.camera_type,
            "qpi_camera_gain": self.camera.gain,
            "qpi_camera_bit_depth": self.camera.bit_depth,
            "qpi_channel_names": self.channel_names,
            "qpi_channel_count": len(self.channels),
            "qpi_is_brightfield": self.is_brightfield,
        }
        # Per-channel details
        for ch in self.channels:
            prefix = f"qpi_ch{ch.index}"
            d[f"{prefix}_name"] = ch.name
            d[f"{prefix}_fluorophore"] = ch.fluorophore
            d[f"{prefix}_exposure_ms"] = ch.exposure_time_ms
            d[f"{prefix}_emission_nm"] = ch.emission_wavelength_nm
        return {k: v for k, v in d.items() if v is not None}


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


def _find_first(root: ET.Element, *tags: str) -> Optional[str]:
    """Search for any of the given tags anywhere in *root* and return first text found."""
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
    """Return list of per-channel exposure times (ms) from ExposureTimeArray."""
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

        # Emission wavelength - try to get from HomeWavelength
        emission_nm = _float(band.find(".//HomeWavelength"))
        if emission_nm is None:
            # Try to infer from fluorophore name, e.g. "OPAL520" -> 520
            if fluorophore:
                m = re.search(r"(\d{3,4})", fluorophore)
                if m:
                    emission_nm = float(m.group(1))

        exposure_ms = exposure_times[idx] if idx < len(exposure_times) else None

        channels.append(
            ChannelInfo(
                index=idx,
                name=name,
                fluorophore=fluorophore,
                exposure_time_ms=exposure_ms,
                emission_wavelength_nm=emission_nm,
                is_brightfield=False,
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
        exposure_ms: Optional[float] = None
        emission_nm: Optional[float] = None

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
                # Exposure time
                et_el = page_root.find("ExposureTime")
                if et_el is not None and et_el.text:
                    try:
                        exposure_ms = float(et_el.text.strip())
                    except ValueError:
                        pass
                # Emission wavelength — midpoint of first emission band
                band = page_root.find(".//EmissionFilter/Bands/Band")
                if band is not None:
                    cuton = _float(band.find("Cuton"))
                    cutoff = _float(band.find("Cutoff"))
                    if cuton is not None and cutoff is not None:
                        emission_nm = (cuton + cutoff) / 2.0
            except ET.ParseError:
                pass

        channels.append(
            ChannelInfo(
                index=idx,
                name=name,
                exposure_time_ms=exposure_ms,
                emission_wavelength_nm=emission_nm,
                is_brightfield=False,
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
                    exposure_time_ms=(
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
                    exposure_time_ms=(
                        exposure_times[i] if i < len(exposure_times) else None
                    ),
                )
                for i in range(n_channels)
            ]

    # Supplement missing exposure times where possible
    for ch in meta.channels:
        if ch.exposure_time_ms is None and ch.index < len(exposure_times):
            ch.exposure_time_ms = exposure_times[ch.index]

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
