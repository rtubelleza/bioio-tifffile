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
from typing import Dict, List, NamedTuple, Optional, Tuple

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


# ---------------------------------------------------------------------------
# Channel field dialects.
#
# A tag does not mean the same thing in every QPTIFF layout, seem to see these
# patterns:
#   scanband — classic Vectra/Polaris/OPAL <ScanBands-i>. <Fluorophore> is
#              explicit, so <Name> is free to mean the biomarker.
#   paged    — Fusion 1.x/2.x page roots. There is no <Fluorophore> element at
#              all: <Biomarker> is the stain and <Name> is the fluorophore,
#              echoed by <Responsivity><Filter><Name>.
# ---------------------------------------------------------------------------
DIALECT_SCANBAND = "scanband"
DIALECT_PAGED = "paged"


class _DialectTags(NamedTuple):
    #: Tags consulted, in order, for the channel's biomarker/stain name.
    name: Tuple[str, ...]
    #: Consulted for the name only if none of ``name`` matched.
    name_fallback: Tuple[str, ...]
    #: Tags/paths consulted, in order, for the fluorophore.
    fluorophore: Tuple[str, ...]


_DIALECTS: Dict[str, _DialectTags] = {
    DIALECT_SCANBAND: _DialectTags(
        name=tuple(_BIOMARKER_TAGS),
        name_fallback=(),
        fluorophore=tuple(_FLUOROPHORE_TAGS),
    ),
    DIALECT_PAGED: _DialectTags(
        # deliberately excludes "Name" — here it is the fluorophore
        name=tuple(t for t in _BIOMARKER_TAGS if t != "Name"),
        # ...but an unnamed page is worse than a duplicated one
        name_fallback=("Name",),
        fluorophore=tuple(_FLUOROPHORE_TAGS) + ("Name", "Responsivity/Filter/Name"),
    ),
}


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

    return ScanResolutionInfo(
        base_pixel_size_um=_float(sr.find("PixelSizeMicrons")),
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


def _page_root_carries_channel_tags(page_root: ET.Element) -> bool:
    """True if this page's XML root describes a channel *in its own right*.

    The discriminator is **where the tag sits**, not what it says. Classic
    shared-XML layouts nest <Biomarker> inside <ScanBands-i>; paged layouts put
    it (or a <Responsivity><Filter>) as a direct child of the page root. Testing
    position is strictly more informative than the older test of whether the
    page XML strings happened to differ, which said nothing when a scan's pages
    were legitimately identical.
    """
    for tag in ("Biomarker", "BioMarker", "StainName", "Marker"):
        if page_root.find(tag) is not None:
            return True
    return page_root.find("Responsivity/Filter") is not None


class _StructureSignals(NamedTuple):
    """Primitive structural facts about a QPTIFF

    Both the channel-metadata locus (which drives parsing) and the legacy
    ``acquisition_format`` label (kept only for back-compat) are pure functions
    of these, so the two can never drift apart.
    """

    is_rgb: bool
    scan_mode_is_brightfield: bool
    has_scanbands: bool
    n_scanbands: int
    has_json_scanprofile: bool
    n_pages: int
    n_distinct_pages: int
    pages_with_root_channel_tags: int

    def to_dict(self) -> Dict[str, object]:
        return dict(self._asdict())


def _collect_signals(
    root: ET.Element,
    scan_mode: str,
    is_rgb: bool,
    per_page_xmls: Optional[List[str]],
    page_roots: List[Optional[ET.Element]],
) -> _StructureSignals:
    """Gather every structural signal to check format of the xml fields"""
    sp = root.find("ScanProfile")
    sp_text = sp.text if sp is not None else None
    scanbands = root.findall(".//ScanBands-i")
    present = [x for x in (per_page_xmls or []) if x]
    return _StructureSignals(
        is_rgb=is_rgb,
        scan_mode_is_brightfield="brightfield" in scan_mode.lower(),
        has_scanbands=bool(scanbands),
        n_scanbands=len(scanbands),
        has_json_scanprofile=(
            sp_text is not None and sp_text.strip().startswith("{")
        ),
        n_pages=len(present),
        n_distinct_pages=len(set(present)),
        pages_with_root_channel_tags=sum(
            1 for r in page_roots if r is not None and _page_root_carries_channel_tags(r)
        ),
    )


def _legacy_acquisition_format(sig: _StructureSignals) -> str:
    """Reproduce the historical ``acquisition_format`` label exactly.

    DEPRECATED as a dispatch key — see the FORMAT_* block in qptiff_types. This
    exists only so files already converted to zarr keep reporting the same
    string. The precedence here is verbatim the original ``_detect_format``.
    """
    if sig.scan_mode_is_brightfield or sig.is_rgb:
        return FORMAT_BRIGHTFIELD
    if sig.has_scanbands:
        return FORMAT_POLARIS_SCANBAND
    if sig.has_json_scanprofile:
        return FORMAT_FUSION_PAGED
    return FORMAT_UNKNOWN


def _determine_locus(sig: _StructureSignals) -> str:
    """Decide where this file's per-channel metadata actually lives.

    1. **RGB samples** — brightfield scan mode or RGB pixel data. There is no
       per-channel metadata to locate.
    2. **Per-page roots** — the pages describe themselves. When no <ScanBands-i>
       exists this is unambiguous.
    3. **Shared scanbands** — nothing at the page roots, but a <ScanBands-i>
       block is there to read.

    When *both* sources are present (newer "paged Polaris" files expose a
    filter-cube <ScanBands-i> in the shared ScanProfile while describing each
    stain at its page root) the page roots win unless the bands look like a
    genuine one-per-channel list: identical page XMLs *and* exactly one band per
    page. Those bands describe the filter cube rather than one stain each, so a
    count mismatch is evidence against reading them as channels.
    """
    if sig.is_rgb or sig.scan_mode_is_brightfield:
        return LOCUS_RGB_SAMPLES

    if sig.pages_with_root_channel_tags:
        if not sig.has_scanbands:
            return LOCUS_PER_PAGE_ROOT
        if sig.n_distinct_pages > 1 or sig.n_scanbands != sig.n_pages:
            return LOCUS_PER_PAGE_ROOT
        return LOCUS_SHARED_SCANBANDS

    if sig.has_scanbands:
        return LOCUS_SHARED_SCANBANDS
    if sig.has_json_scanprofile and sig.n_pages:
        return LOCUS_PER_PAGE_ROOT
    return LOCUS_UNKNOWN


def _active_band_index(filter_elem: Optional[ET.Element]) -> Optional[int]:
    """Position of the <Band> flagged <Active>true</Active>, or None."""
    if filter_elem is None:
        return None
    for i, b in enumerate(filter_elem.findall("Bands/Band")):
        active = b.find("Active")
        if active is not None and active.text and active.text.strip().lower() in (
            "true",
            "1",
            "yes",
        ):
            return i
    return None


def _select_band(
    filter_elem: Optional[ET.Element],
    active_idx: Optional[int] = None,
) -> Optional[ET.Element]:
    """Return the band this channel actually used.

    Fusion per-page XMLs list every band of the multi-band cube; only one
    applies to the channel. Crucially, **only the excitation filter flags it** —
    the emission filter lists the same cube's bands in the same order with no
    <Active> element at all. So the active position has to carry across from
    whichever filter declares it. Taking each filter's own first band instead
    silently assigns every non-first channel the wrong emission passband (a
    75-plex ATTO 550 channel would report DAPI's 440-466 nm).

    ``active_idx`` is that carried-over position. Falls back to the first band
    when nothing anywhere declares one, preserving behaviour for older formats
    that omit <Active> entirely.
    """
    if filter_elem is None:
        return None
    bands = filter_elem.findall("Bands/Band")
    if not bands:
        return None
    own = _active_band_index(filter_elem)
    if own is not None:
        return bands[own]
    if active_idx is not None and 0 <= active_idx < len(bands):
        return bands[active_idx]
    return bands[0]


def _populate_channel_fields_from_element(
    elem: ET.Element, fields: Dict[str, object], dialect: str = DIALECT_SCANBAND
) -> None:
    """
    Pull every per-channel field we know how to extract out of a single XML
    element (either a <ScanBands-i> or a page root) and write it into ``fields``.

    Mutates ``fields`` in place. ``fields`` is a dict that will be passed to
    ``ChannelInfo(**fields)``; keys must match ChannelInfo attribute names.
    Values are only set when they parse successfully, so defaults on the
    dataclass remain in effect when a field is missing from this file.
    """
    tags = _DIALECTS.get(dialect, _DIALECTS[DIALECT_SCANBAND])

    # Biomarker / stain name. The fallback tier only runs if the dedicated
    # biomarker tags found nothing, so in the paged dialect a page carrying
    # only <Name> still gets a name rather than "Channel_i".
    for tag in tags.name + tags.name_fallback:
        el = elem.find(tag)
        if el is not None and el.text:
            val = el.text.strip()
            if val and val not in ("None", "--"):
                fields["name"] = val
                break

    # Fluorophore. In the paged dialect this list reaches <Name> and the
    # <Responsivity><Filter><Name> echo; in the scanband dialect it stops at
    # the explicit <Fluorophore>/<Fluor>, leaving <Name> to mean the biomarker.
    for tag in tags.fluorophore:
        fluor = _text(elem.find(tag))
        if fluor:
            fields["fluorophore"] = fluor
            break

    # exposure time (us per spec)
    et_text = _text(elem.find("ExposureTime"))
    if et_text is not None:
        try:
            fields["exposure_time_us"] = float(et_text)
        except ValueError:
            pass

    # emission / excitation bands. Filters may list every band of a multi-band
    # cube (one per channel); prefer the one flagged <Active>true</Active>, else
    # the first band.
    #
    # Keep the raw cut-on/cut-off edges as well as the midpoint: OME's
    # Filter.transmittance_range wants the edges, and averaging them away loses
    # the passband width. The midpoint stays as the derived single wavelength
    # that Channel.emission_wavelength/excitation_wavelength need.
    em_filter = elem.find(".//EmissionFilter")
    ex_filter = elem.find(".//ExcitationFilter")
    # Only the excitation filter flags its active band; the emission filter
    # lists the same cube's bands in the same order without <Active>, so the
    # position carries across. See _select_band.
    active_idx = _active_band_index(ex_filter)
    if active_idx is None:
        active_idx = _active_band_index(em_filter)

    for prefix, filter_elem in (
        ("emission", em_filter),
        ("excitation", ex_filter),
    ):
        band = _select_band(filter_elem, active_idx)
        if band is None:
            continue
        cuton = _float(band.find("Cuton"))
        cutoff = _float(band.find("Cutoff"))
        if cuton is None or cutoff is None:
            continue
        fields[f"{prefix}_cut_on_nm"] = cuton
        fields[f"{prefix}_cut_off_nm"] = cutoff
        fields[f"{prefix}_wavelength_nm"] = (cuton + cutoff) / 2.0

    # How many bands the cube declares, so OME can distinguish a plain
    # band-pass filter from a multi-pass one. Both filters describe the same
    # physical cube, so either is a valid source.
    for filter_elem in (em_filter, ex_filter):
        if filter_elem is not None:
            n_bands = len(filter_elem.findall("Bands/Band"))
            if n_bands:
                fields["n_filter_bands"] = n_bands
                break

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
        _populate_channel_fields_from_element(band, fields, DIALECT_SCANBAND)

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


def _parse_page_roots(
    per_page_xmls: Optional[List[str]],
) -> List[Optional[ET.Element]]:
    """Parse each page's XML once, tolerating empty or malformed entries.

    Shared by locus detection and channel extraction so a 75-page panel is not
    parsed twice.
    """
    roots: List[Optional[ET.Element]] = []
    for xml in per_page_xmls or []:
        if not xml:
            roots.append(None)
            continue
        try:
            roots.append(ET.fromstring(xml))
        except ET.ParseError:
            roots.append(None)
    return roots


def _parse_channels_from_page_roots(
    page_roots: List[Optional[ET.Element]],
) -> List[ChannelInfo]:
    """
    Extract per-channel info from individual page XML roots.

    Fusion QPTIFF files (1.x and 2.x alike) embed per-channel metadata directly
    at the root level of each page's ImageDescription XML (one page per channel),
    rather than grouping them in ScanBands-i elements.
    """
    channels: List[ChannelInfo] = []
    for idx, page_root in enumerate(page_roots):
        fields: Dict[str, object] = {"index": idx, "name": f"Channel_{idx}"}
        if page_root is not None:
            _populate_channel_fields_from_element(page_root, fields, DIALECT_PAGED)
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
            channel_locus=LOCUS_UNKNOWN,
            raw_xml=xml_string,
        )

    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError as exc:
        log.warning("Failed to parse QPI XML: %s", exc)
        return QptiffMetadata(
            slide=slide,
            images=[QptiffImageSceneMetadata(image_info=image_info, raw_xml=xml_string)],
            channel_locus=LOCUS_UNKNOWN,
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

    # Canonical full-res pixel -> physical scale (unit-agnostic). The QPI XML
    # declares it as <PixelSizeMicrons>, i.e. microns, so the unit is "um".
    # (Supersedes the deprecated scan_resolution.base_pixel_size_um.)
    base_px = image_info.scan_resolution.base_pixel_size_um
    if base_px is not None:
        image_info.scale_factor = base_px
        image_info.scale_factor_unit = "um"

    # Fusion 1.x stores ScanProfile as a JSON blob; fill any still-missing
    # fields from it. Done after _parse_scan_resolution so XML-form values
    # take priority.
    if sp is not None and sp.text and sp.text.strip().startswith("{"):
        _parse_scan_profile_json(image_info, sp.text)

    image_info.camera = _parse_camera(root)

    exposure_times = _parse_exposure_times(root)

    page_roots = _parse_page_roots(per_page_xmls)
    signals = _collect_signals(
        root, image_info.scan_mode or "", is_rgb, per_page_xmls, page_roots
    )
    locus = _determine_locus(signals)
    fmt = _legacy_acquisition_format(signals)
    log.debug(
        "QPTIFF channel locus: %s (legacy format=%s, slide=%s, signals=%s)",
        locus,
        fmt,
        slide.slide_id,
        signals.to_dict(),
    )

    def _generic_channels() -> List[ChannelInfo]:
        if not (n_channels and n_channels > 0):
            return []
        return [
            ChannelInfo(
                index=i,
                name=f"Channel_{i}",
                exposure_time_us=(
                    exposure_times[i] if i < len(exposure_times) else None
                ),
            )
            for i in range(n_channels)
        ]

    if locus == LOCUS_RGB_SAMPLES:
        n = n_channels if n_channels and n_channels > 0 else 3
        channels = _parse_brightfield_channels(n, root)

    elif locus == LOCUS_PER_PAGE_ROOT:
        channels = _parse_channels_from_page_roots(page_roots) or _generic_channels()

    elif locus == LOCUS_SHARED_SCANBANDS:
        channels = _parse_fluorescence_channels(root, exposure_times)

    else:  # LOCUS_UNKNOWN — fail open, try strategies in order
        channels = (
            _parse_fluorescence_channels(root, exposure_times)
            or _parse_channels_from_page_roots(page_roots)
            or _generic_channels()
        )

    # supplement missing exposure times where possible
    for ch in channels:
        if ch.exposure_time_us is None and ch.index < len(exposure_times):
            ch.exposure_time_us = exposure_times[ch.index]

    # Paged files: when the page XMLs carry no fluorophore at all, fall back to
    # experimentDescription.channels[] in the ScanProfile JSON, which usually lists the
    # filter set in acquisition order (e.g. ["DAPI", "ATTO550", "CY5", "AF750"]).
    #
    # This is only a positional guess, so it is applied strictly 1:1 and only when
    # the counts match exactly. Multi-cycle scans do NOT repeat the filter list in
    # order — a 75-page panel may run DAPI, ATTO550, AF750, Cy5, ATTO550, ... — so
    # an `index % n` mapping silently invents wrong fluorophores. Leave None instead.
    missing = [ch for ch in channels if ch.fluorophore is None]
    if locus == LOCUS_PER_PAGE_ROOT and missing:
        sp_elem = root.find("ScanProfile")
        if sp_elem is not None and sp_elem.text and sp_elem.text.strip().startswith("{"):
            try:
                sp = json.loads(sp_elem.text)
                exp = sp.get("experimentDescription", {})
                filter_fluors = [
                    c["name"] for c in exp.get("channels", [])
                    if isinstance(c, dict) and c.get("name")
                ]
                if filter_fluors and len(filter_fluors) == len(channels):
                    for ch in missing:
                        ch.fluorophore = filter_fluors[ch.index]
                elif filter_fluors:
                    log.warning(
                        "QPTIFF %s: %d channel(s) have no fluorophore in their page "
                        "XML and the ScanProfile filter list (%d entries) cannot be "
                        "mapped 1:1 to %d channels; leaving fluorophore unset rather "
                        "than guessing.",
                        slide.slide_id,
                        len(missing),
                        len(filter_fluors),
                        len(channels),
                    )
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
        channel_locus=locus,
        structure_signature=signals.to_dict(),
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
