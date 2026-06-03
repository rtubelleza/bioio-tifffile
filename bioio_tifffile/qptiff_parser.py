#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PerkinElmer QPTIFF parsers for its tiff pages/series and contained XML strings.

Extracts structured metadata from TIFF ImageDescription tag (tag 270) XML
and populates the dataclasses defined in qptiff_types.
"""

from __future__ import annotations

import json
import logging
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

from .qptiff_types import (
    FORMAT_BRIGHTFIELD,
    FORMAT_FUSION_PAGED,
    FORMAT_POLARIS_SCANBAND,
    FORMAT_UNKNOWN,
    CameraInfo,
    ChannelInfo,
    ImageInfo,
    QptiffImageSceneMetadata,
    QptiffMetadata,
    ScanResolutionInfo,
    SlideInfo,
)

log = logging.getLogger(__name__)

TIFF_IMAGE_DESCRIPTION_TAG = 270
TIFF_DATETIME_TAG = 306

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


def _set_if(target: Dict[str, object], key: str, value: object) -> None:
    """Write ``value`` into ``target[key]`` only when it is not None."""
    if value is not None:
        target[key] = value

def _parse_scan_resolution(root: ET.Element) -> ScanResolutionInfo:
    """Extract pixel size and objective from ScanResolution element."""
    sr = root.find(".//ScanResolution")
    if sr is None:
        return ScanResolutionInfo()

    base_px = _float(sr.find("PixelSizeMicrons"))
    return ScanResolutionInfo(
        base_pixel_size_um=base_px,
        # <PixelSizeMicrons> is microns by definition; record the unit explicitly
        # so downstream consumers don't rely on the field name alone. Use ASCII
        # "um" (not "µm") so it's easy to string-query/filter.
        pixel_size_unit="um" if base_px is not None else None,
        magnification=_float(sr.find("Magnification")),
        objective_name=_text(sr.find("ObjectiveName")),
        binning=_int(sr.find("Binning")),
    )


def _parse_camera(root: ET.Element) -> CameraInfo:
    """Extract camera settings from the top-level CameraSettings element.

    CameraName and CameraType are direct children of root; Gain/BitDepth/Binning
    live inside CameraSettings. We always try both so a missing CameraSettings
    does not silently drop the camera identity fields.
    """
    cs = root.find("CameraSettings")
    return CameraInfo(
        camera_name=_text(root.find("CameraName")),
        camera_type=_text(root.find("CameraType")),
        gain=_float(cs.find("Gain")) if cs is not None else None,
        bit_depth=_int(cs.find("BitDepth")) if cs is not None else None,
        binning=_int(cs.find("Binning")) if cs is not None else None,
    )


def _parse_exposure_times(root: ET.Element) -> List[Optional[float]]:
    """Return list of per-channel exposure times (us) from ExposureTimeArray.

    Per the PerkinElmer spec, ExposureTime values are integer microseconds.
    """
    eta = root.find("ExposureTimeArray")
    if eta is None:
        return [_float(root.find("ExposureTime"))]
    return [_float(val_el) for val_el in eta.findall("Value")]


def _parse_scan_profile_json(image_info: ImageInfo, scan_profile_text: str) -> None:
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
    if isinstance(exp, dict) and image_info.scan_profile_name is None:
        name = exp.get("name")
        if isinstance(name, str) and name:
            image_info.scan_profile_name = name


def _detect_format(
    root: ET.Element,
    scan_mode: str,
    is_rgb: bool,
) -> str:
    """
    Identify which QPTIFF format variant produced this file.. a couple of versions
    as qptiff evolved from vectra

    1. **brightfield** — scan_mode contains "Brightfield" (case-insensitive) or
       the pixel data is RGB (``is_rgb``, i.e. samples stored in the S dimension).

       NOTE: a ``<BFLampType>`` element is *not* sufficient on its own. Newer
       Fusion 2.x fluorescence scans also emit ``<BFLampType>WhiteLED</BFLampType>``
       alongside per-channel fluorescence pages, so keying off it misclassified
       multiplex panels as 3-sample brightfield and dropped every channel's
       biomarker / fluorophore metadata.
    2. **polaris_scanband** — ``<ScanBands-i>`` elements are present anywhere in
       the tree (older Vectra/Polaris/OPAL XML ScanProfile format).
    3. **fusion_paged** — ``<ScanProfile>`` content is a JSON object (newer
       Akoya Biosciences / Fusion 1.x format where channel metadata is
       embedded per-page rather than in a shared ScanProfile).
    4. **unknown** — no recognised structural signal; caller should attempt
       strategies in sequence.
    """
    if "brightfield" in scan_mode.lower() or is_rgb:
        return FORMAT_BRIGHTFIELD

    if root.findall(".//ScanBands-i"):
        return FORMAT_POLARIS_SCANBAND

    sp = root.find("ScanProfile")
    if sp is not None and sp.text and sp.text.strip().startswith("{"):
        return FORMAT_FUSION_PAGED

    return FORMAT_UNKNOWN


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

    # display colour, signal type, unmixed flag
    color = _parse_color(elem.find("Color"))
    if color is None:
        color = _parse_color(elem.find(".//Color"))
    if color is not None:
        fields["color_rgb"] = color

    _set_if(fields, "is_unmixed_component", _bool(elem.find("IsUnmixedComponent")))
    _set_if(fields, "signal_units", _int(elem.find("SignalUnits")))

    # per-channel objective, AF subtraction
    _set_if(fields, "objective", _text(elem.find("Objective")))
    _set_if(
        fields,
        "autofluorescence_subtracted",
        _bool(elem.find("AutofluorescenceSubtracted")),
    )
    # Polaris per-band auto-exposure mode (nested under the band's filter spec)
    _set_if(fields, "auto_expose_type", _text(elem.find(".//AutoExposeType")))

    # responsivity / calibration
    resp = elem.find("Responsivity")
    if resp is not None:
        rf = resp.find("Filter")
        if rf is not None:
            _set_if(fields, "responsivity", _float(rf.find("Response")))
            _set_if(fields, "responsivity_filter_id", _text(rf.find("FilterID")))
            _set_if(fields, "responsivity_date", _text(rf.find("Date")))
            _set_if(fields, "responsivity_filter_name", _text(rf.find("Name")))

    # excitation / emission filter identification
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

    # per-channel detector settings
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


def _parse_brightfield_channels(
    n_samples: int, root: Optional[ET.Element] = None
) -> List[ChannelInfo]:
    """For brightfield RGB images, return R/G/B channel descriptors.

    The RGB samples are a single acquisition, so the image-root ``ExposureTime`` /
    ``SignalUnits`` / ``Objective`` / ``IsUnmixedComponent`` and ``CameraSettings``
    (gain, bit depth, binning, offset, orientation, ROI) apply equally to every
    sample. Pull them in so brightfield channels carry the same per-channel
    metadata as fluorescence ones rather than bare names.

    Only *direct* children of ``root`` are read: the deep ``<ScanProfile>`` filter
    config (HomeWavelength, TransmissionBands, ...) describes the tunable-filter
    hardware, not the RGB samples, and must not leak onto the channels.
    """
    names = (
        ["Red", "Green", "Blue"]
        if n_samples == 3
        else [f"Sample_{i}" for i in range(n_samples)]
    )
    shared: Dict[str, object] = {}
    if root is not None:
        et = _text(root.find("ExposureTime"))
        if et is not None:
            try:
                shared["exposure_time_us"] = float(et)
            except ValueError:
                pass
        _set_if(shared, "signal_units", _int(root.find("SignalUnits")))
        _set_if(shared, "is_unmixed_component", _bool(root.find("IsUnmixedComponent")))
        _set_if(shared, "objective", _text(root.find("Objective")))
        cs = root.find("CameraSettings")
        if cs is not None:
            _set_if(shared, "gain", _float(cs.find("Gain")))
            _set_if(shared, "binning", _int(cs.find("Binning")))
            _set_if(shared, "bit_depth", _int(cs.find("BitDepth")))
            _set_if(shared, "offset_counts", _int(cs.find("OffsetCounts")))
            _set_if(shared, "camera_orientation", _text(cs.find("Orientation")))
            roi = cs.find("ROI")
            if roi is not None:
                _set_if(shared, "roi_x", _int(roi.find("X")))
                _set_if(shared, "roi_y", _int(roi.find("Y")))
                _set_if(shared, "roi_width", _int(roi.find("Width")))
                _set_if(shared, "roi_height", _int(roi.find("Height")))
    return [
        ChannelInfo(index=i, name=names[i], is_brightfield=True, **shared)  # type: ignore[arg-type]
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
    for idx, band in enumerate(root.findall(".//ScanBands-i")):
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


def _parse_channels_from_per_page_xmls(per_page_xmls: List[str]) -> List[ChannelInfo]:
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


def parse_qpi_xml(
    xml_string: str,
    n_channels: Optional[int] = None,
    datetime_str: Optional[str] = None,
    per_page_xmls: Optional[List[str]] = None,
    is_rgb: bool = False,
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
    is_rgb:
        True when the pixel data is RGB (samples in the S dimension). This is
        the reliable brightfield signal — a ``<BFLampType>`` element alone is
        not, since Fusion 2.x fluorescence scans also carry it.

    Returns
    -------
    QptiffMetadata
        Fully populated (where data is available) metadata object.
    """
    slide = SlideInfo(datetime=datetime_str)
    image_info = ImageInfo()
    channels: List[ChannelInfo] = []

    if not xml_string:
        return QptiffMetadata(
            slide=slide,
            images=[QptiffImageSceneMetadata(image_info=image_info, raw_xml=xml_string)],
            raw_xml=xml_string,
        )

    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError as exc:
        log.warning("Failed to parse QPI XML: %s", exc)
        return QptiffMetadata(
            slide=slide,
            images=[QptiffImageSceneMetadata(image_info=image_info, raw_xml=xml_string)],
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
    slide.validation_code = _text(root.find("ValidationCode"))
    slide.sample_description = _text(root.find("SampleDescription"))

    # image-scope fields
    image_info.image_type = _text(root.find("ImageType"))
    image_info.bf_lamp_type = _text(root.find("BFLampType"))
    image_info.lamp_type = _text(root.find("LampType"))
    image_info.objective = _text(root.find("Objective"))
    image_info.scale_factor = _float(root.find("ScaleFactor"))

    sp = root.find("ScanProfile")
    if sp is not None:
        _nested = sp.find("root")
        sp_root = _nested if _nested is not None else sp
        image_info.scan_profile_name = _text(sp_root.find("Name"))
        image_info.scan_mode = _text(sp_root.find("Mode"))
        image_info.is_tma = _bool(sp_root.find("SampleIsTMA"))
        image_info.opal_kit_type = _text(sp_root.find("OpalKitType"))
        # ScanProfile acquisition settings (Polaris/Vectra XML ScanProfile)
        image_info.compression = _text(sp_root.find("Compression"))
        image_info.jpeg_quality = _int(sp_root.find("JPEGQuality"))
        image_info.saturation_protection_type = _text(
            sp_root.find("SaturationProtectionType")
        ) or _text(sp_root.find("FieldSaturationProtectionType"))
        image_info.coverslip_thickness = _text(sp_root.find("CoverslipThickness"))
        image_info.is_rna = _bool(sp_root.find("IsRNA"))
        cam = sp_root.find("CameraSettings")
        if cam is not None:
            image_info.rotate_image = _bool(cam.find("RotateImage"))
            image_info.mirror_image = _bool(cam.find("MirrorImage"))

    image_info.scan_resolution = _parse_scan_resolution(root)

    # Fusion 1.x stores ScanProfile as a JSON blob; fill any still-missing
    # fields from it. Done after _parse_scan_resolution so XML-form values
    # take priority.
    if sp is not None and sp.text and sp.text.strip().startswith("{"):
        _parse_scan_profile_json(image_info, sp.text)

    image_info.camera = _parse_camera(root)

    exposure_times = _parse_exposure_times(root)
    fmt = _detect_format(root, image_info.scan_mode or "", is_rgb)
    log.debug("Detected QPTIFF format: %s (slide=%s)", fmt, slide.slide_id)

    if fmt == FORMAT_BRIGHTFIELD:
        n = n_channels if n_channels and n_channels > 0 else 3
        channels = _parse_brightfield_channels(n, root)

    elif fmt == FORMAT_POLARIS_SCANBAND:
        # Newer "paged Polaris" files (Polaris/PhenoCycler, Fusion 2.x) carry the
        # per-channel metadata — biomarker <Name>, <ExcitationFilter>/<EmissionFilter>,
        # <Responsivity>, <ExposureTime>, <SignalUnits>, <Objective> — at each
        # *page* XML root, like fusion_paged, while still exposing <ScanBands-i>
        # filter-cube config in the shared ScanProfile. Detection lands on
        # polaris_scanband because <ScanBands-i> is present, but the ScanBands-i
        # describe the filter cube (not one stain each), so reading them yields
        # unnamed channels and drops the filter/responsivity metadata.
        #
        # When the per-page XMLs are distinct, parse the page roots (rich
        # per-channel data, handled by _populate_channel_fields_from_element).
        # Fall back to the classic shared-XML ScanBands-i layout (old Vectra /
        # Polaris OPAL, where every page repeats one root XML and the per-channel
        # Biomarker/Fluorophore live inside ScanBands-i).
        distinct_pages = {x for x in (per_page_xmls or []) if x}
        if len(distinct_pages) > 1:
            channels = _parse_channels_from_per_page_xmls(per_page_xmls)
        else:
            channels = _parse_fluorescence_channels(root, exposure_times)

    elif fmt == FORMAT_FUSION_PAGED:
        if per_page_xmls:
            channels = _parse_channels_from_per_page_xmls(per_page_xmls)
        elif n_channels and n_channels > 0:
            channels = [
                ChannelInfo(
                    index=i,
                    name=f"Channel_{i}",
                    exposure_time_us=exposure_times[i] if i < len(exposure_times) else None,
                )
                for i in range(n_channels)
            ]

    else:  # FORMAT_UNKNOWN — try strategies in order
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
                    exposure_time_us=exposure_times[i] if i < len(exposure_times) else None,
                )
                for i in range(n_channels)
            ]

    # supplement missing exposure times where possible
    for ch in channels:
        if ch.exposure_time_us is None and ch.index < len(exposure_times):
            ch.exposure_time_us = exposure_times[ch.index]

    # Fusion paged: fluorophore lives in ScanProfile JSON, not per-page XML.
    # experimentDescription.channels[] lists the filter cycle in order
    # (e.g. ["DAPI", "ATTO550", "CY5", "AF750"]); channel i uses filter i % n.
    if fmt == FORMAT_FUSION_PAGED and any(ch.fluorophore is None for ch in channels):
        sp_elem = root.find("ScanProfile")
        if sp_elem is not None and sp_elem.text and sp_elem.text.strip().startswith("{"):
            try:
                sp = json.loads(sp_elem.text)
                exp = sp.get("experimentDescription", {})
                filter_fluors = [
                    c["name"] for c in exp.get("channels", [])
                    if isinstance(c, dict) and c.get("name")
                ]
                if filter_fluors:
                    n = len(filter_fluors)
                    for ch in channels:
                        if ch.fluorophore is None:
                            ch.fluorophore = filter_fluors[ch.index % n]
            except (ValueError, TypeError, KeyError):
                pass

    image = QptiffImageSceneMetadata(
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


def valid_qpi_series(pages: object) -> bool:
    """Return True if the tifffile series pages carry PerkinElmer QPI XML.

    Uses pages[0] as a fast proxy — all pages in a QPI series share the same
    root-level XML structure, so one check is sufficient for format detection.
    """
    try:
        page = pages[0]  # type: ignore[index]
        tags = page.tags  # type: ignore[attr-defined]
        tag = tags.get(TIFF_IMAGE_DESCRIPTION_TAG)
        if tag is None:
            return False
        val = tag.value
        if not isinstance(val, str) or not val:
            return False
        try:
            root = ET.fromstring(val)
            return "PerkinElmer-QPI" in root.tag or "PerkinElmerQPI" in root.tag
        except ET.ParseError:
            return False
    except Exception as exc:
        log.debug("Could not read ImageDescription tag: %s", exc)
        return False
