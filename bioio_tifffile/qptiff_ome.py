#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Maps QptiffMetadata to a fully typed ome_types.model.OME object, and
provides helpers to extract flat xarray attrs / channel coords from that OME
object. The OME object is the single source of truth for all downstream
metadata.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .qptiff_types import QptiffMetadata

# Matches per-channel keys in the qpi://vectra MapAnnotation: "chN_fieldName"
_CH_ANN_RE = re.compile(r"^ch(\d+)_(.+)$")

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

def _format_binning(b: Optional[int]) -> Optional[str]:
    return f"{b}x{b}" if b is not None else None

def ome_metadata_from_qptiff(
    qpi: QptiffMetadata,
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

    # instrument sub-objects
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

    # one filter pair per channel
    filters: List[Filter] = []
    filter_idx = 0

    def _make_filter(
        name: Optional[str],
        manufacturer: Optional[str],
        part_no: Optional[str],
        fid: str,
    ) -> Optional[Filter]:
        if not name and not manufacturer and not part_no:
            return None
        return Filter(id=fid, model=name, manufacturer=manufacturer, lot_number=part_no)

    exc_filter_ids: Dict[int, str] = {}
    emi_filter_ids: Dict[int, str] = {}

    for ch in qpi.channels:
        exc = _make_filter(
            ch.excitation_filter_name,
            ch.excitation_filter_manufacturer,
            ch.excitation_filter_part_no,
            f"Filter:{filter_idx}",
        )
        if exc:
            exc_filter_ids[ch.index] = exc.id
            filters.append(exc)
            filter_idx += 1

        emi = _make_filter(
            ch.emission_filter_name,
            ch.emission_filter_manufacturer,
            ch.emission_filter_part_no,
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

    experimenter = None
    if qpi.operator_name:
        experimenter = Experimenter(id=experimenter_id, user_name=qpi.operator_name)

    # channels + planes
    ome_channels: List[Channel] = []
    ome_planes: List[Plane] = []

    for ch in qpi.channels:
        det_settings = None
        if ch.gain is not None or ch.binning is not None or ch.offset_counts is not None:
            det_settings = DetectorSettings(
                id=detector_id,
                gain=ch.gain,
                binning=_format_binning(ch.binning),
                offset=float(ch.offset_counts) if ch.offset_counts is not None else None,
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

        # one Plane per channel (Z=0, T=0)
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

    # significant_bits: use the first channel's bit_depth (camera ADC depth,
    # e.g. 14-bit data stored in uint16). All channels share the same camera.
    sig_bits: Optional[int] = next(
        (ch.bit_depth for ch in qpi.channels if ch.bit_depth is not None), None
    ) or getattr(qpi.camera, "bit_depth", None)

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
    if sig_bits is not None:
        px_kwargs["significant_bits"] = sig_bits
    if qpi.pixel_size_um is not None:
        px_kwargs["physical_size_x"] = qpi.pixel_size_um
        px_kwargs["physical_size_x_unit"] = "\u00b5m"
        px_kwargs["physical_size_y"] = qpi.pixel_size_um
        px_kwargs["physical_size_y_unit"] = "\u00b5m"
    px_kwargs["channels"] = ome_channels
    px_kwargs["planes"] = ome_planes
    pixels = Pixels(**px_kwargs)  # type: ignore[arg-type]

    # image-level QPI annotation
    sl = qpi.slide
    ii = qpi._primary.image_info
    img_qpi: Dict[str, str] = {}
    for k, v in [
        ("description_version", qpi.description_version),
        ("acquisition_software", qpi.acquisition_software),
        ("image_type", qpi.image_type),
        ("identifier", qpi.identifier),
        ("slide_id", qpi.slide_id),
        ("barcode", qpi.barcode),
        ("study_name", qpi.study_name),
        ("computer_name", qpi.computer_name),
        ("datetime", qpi.datetime),
        ("validation_code", sl.validation_code),
        ("sample_description", sl.sample_description),
        ("bf_lamp_type", qpi.bf_lamp_type),
        ("lamp_type", ii.lamp_type),
        ("scan_profile_name", qpi.scan_profile_name),
        ("scan_mode", qpi.scan_mode),
        ("is_tma", _str(qpi.is_tma)),
        ("opal_kit_type", qpi.opal_kit_type),
        ("acquisition_format", qpi.acquisition_format),
        ("scale_factor", _str(ii.scale_factor)),
        ("compression", ii.compression),
        ("jpeg_quality", _str(ii.jpeg_quality)),
        ("saturation_protection_type", ii.saturation_protection_type),
        ("coverslip_thickness", ii.coverslip_thickness),
        ("is_rna", _str(ii.is_rna)),
        ("rotate_image", _str(ii.rotate_image)),
        ("mirror_image", _str(ii.mirror_image)),
        ("camera_name", qpi.camera.camera_name),
        ("camera_gain", _str(qpi.camera.gain)),
        ("camera_bit_depth", _str(qpi.camera.bit_depth)),
        ("channel_count", _str(len(qpi.channels))),
        ("is_brightfield", _str(qpi.is_brightfield)),
        ("objective", qpi.objective),
    ]:
        if v is not None:
            img_qpi[k] = v

    # channel-level QPI annotation
    for ch in qpi.channels:
        for k, v in [
            ("is_unmixed_component", _str(ch.is_unmixed_component)),
            ("signal_units", _str(ch.signal_units)),
            ("objective", ch.objective),
            ("autofluorescence_subtracted", _str(ch.autofluorescence_subtracted)),
            ("auto_expose_type", ch.auto_expose_type),
            ("responsivity", _str(ch.responsivity)),
            ("responsivity_filter_id", ch.responsivity_filter_id),
            ("responsivity_date", ch.responsivity_date),
            ("responsivity_filter_name", ch.responsivity_filter_name),
            # excitation/emission_filter_part_no → Filter.lot_number (mapped to OME)
            # bit_depth → Pixels.significant_bits (mapped to OME)
            # offset_counts → DetectorSettings.offset (mapped to OME)
            ("camera_orientation", ch.camera_orientation),
            ("roi_x", _str(ch.roi_x)),
            ("roi_y", _str(ch.roi_y)),
            ("roi_width", _str(ch.roi_width)),
            ("roi_height", _str(ch.roi_height)),
        ]:
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


def ome_to_flat_attrs(ome: object) -> Dict[str, Any]:
    """Slide-level flat attrs for the DataTree root, derived from an OME object.

    Only image/slide-scoped fields are included here — instrument, experimenter,
    pixel size, and stage position.  Per-channel metadata (wavelengths, exposure
    times, detector settings, vendor fields) lives in ``ome_to_channel_coords``
    and the OME object itself; it must not be flattened onto the root attrs.
    """
    d: Dict[str, Any] = {}

    # --- Instrument ---------------------------------------------------------
    if getattr(ome, "instruments", None):
        inst = ome.instruments[0]  # type: ignore[union-attr]
        if getattr(inst, "microscope", None) and inst.microscope.model:
            d["Microscope:Model"] = inst.microscope.model
        objectives = getattr(inst, "objectives", [])
        if objectives:
            obj = objectives[0]
            if getattr(obj, "model", None) is not None:
                d["Objective:Model"] = obj.model
            if getattr(obj, "nominal_magnification", None) is not None:
                d["Objective:NominalMagnification"] = float(obj.nominal_magnification)
        detectors = getattr(inst, "detectors", [])
        if detectors:
            det = detectors[0]
            if getattr(det, "model", None) is not None:
                d["Detector:Model"] = det.model

    # --- Experimenter -------------------------------------------------------
    experimenters = getattr(ome, "experimenters", [])
    if experimenters and getattr(experimenters[0], "user_name", None):
        d["Experimenter:UserName"] = experimenters[0].user_name

    images = getattr(ome, "images", [])
    if not images:
        return d

    px = images[0].pixels

    # --- Physical pixel size ------------------------------------------------
    if getattr(px, "physical_size_x", None) is not None:
        d["Pixels:PhysicalSizeX"] = float(px.physical_size_x)
        d["Pixels:PhysicalSizeXUnit"] = str(px.physical_size_x_unit or "\u00b5m")
    if getattr(px, "physical_size_y", None) is not None:
        d["Pixels:PhysicalSizeY"] = float(px.physical_size_y)
        d["Pixels:PhysicalSizeYUnit"] = str(px.physical_size_y_unit or "\u00b5m")

    # --- Stage position (slide origin) from the first plane -----------------
    planes = getattr(px, "planes", [])
    if planes:
        p0 = planes[0]
        if getattr(p0, "position_x", None) is not None:
            d["Plane:PositionX"] = float(p0.position_x)
            d["Plane:PositionXUnit"] = str(p0.position_x_unit or "\u00b5m")
        if getattr(p0, "position_y", None) is not None:
            d["Plane:PositionY"] = float(p0.position_y)
            d["Plane:PositionYUnit"] = str(p0.position_y_unit or "\u00b5m")

    return d


def ome_to_channel_coords(
    ome: object,
    channel_dim: str = "c",
    *,
    ome_only: bool = False,
) -> Dict[str, Any]:
    """xarray coords dict for the channel axis, derived from an OME object.

    Replaces ``channel_coord_dict()`` + ``CHANNEL_COORD_SCHEMA``. The
    *channel_dim* entry (the channel name list) is always included when
    channels are present. Additional per-channel metadata is attached as
    ``(channel_dim, values)`` tuples so xarray indexes them correctly.

    QPTIFF-specific fields are read from the ``qpi://vectra`` annotation
    and omitted when *ome_only* is ``True``.
    """
    coords: Dict[str, Any] = {}

    images = getattr(ome, "images", [])
    if not images:
        return coords

    px = images[0].pixels
    channels = getattr(px, "channels", [])
    if not channels:
        return coords

    n = len(channels)
    coords[channel_dim] = [ch.name or "" for ch in channels]

    planes = getattr(px, "planes", [])
    plane_by_c: Dict[int, Any] = {int(p.the_c): p for p in planes}

    def _add(key: str, vals: List[Any]) -> None:
        if any(v is not None for v in vals):
            coords[key] = (channel_dim, vals)

    _add("Channel:Fluor", [getattr(ch, "fluor", None) for ch in channels])
    _add(
        "Channel:Color",
        [
            str(ch.color) if getattr(ch, "color", None) is not None else None
            for ch in channels
        ],
    )
    _add(
        "Channel:EmissionWavelength",
        [
            float(ch.emission_wavelength)
            if getattr(ch, "emission_wavelength", None) is not None
            else None
            for ch in channels
        ],
    )
    _add(
        "Channel:ExcitationWavelength",
        [
            float(ch.excitation_wavelength)
            if getattr(ch, "excitation_wavelength", None) is not None
            else None
            for ch in channels
        ],
    )
    _add(
        "DetectorSettings:Gain",
        [
            ch.detector_settings.gain
            if getattr(ch, "detector_settings", None) is not None
            and ch.detector_settings.gain is not None
            else None
            for ch in channels
        ],
    )
    _add(
        "DetectorSettings:Binning",
        [
            str(ch.detector_settings.binning)
            if getattr(ch, "detector_settings", None) is not None
            and ch.detector_settings.binning is not None
            else None
            for ch in channels
        ],
    )
    _add(
        "Plane:ExposureTime",
        [
            float(plane_by_c[i].exposure_time)
            if i in plane_by_c
            and getattr(plane_by_c[i], "exposure_time", None) is not None
            else None
            for i in range(n)
        ],
    )

    # --- QPTIFF per-channel annotation fields --------------------------------
    if not ome_only:
        ch_ann: Dict[int, Dict[str, str]] = {}
        for ann in getattr(ome, "structured_annotations", []):
            if getattr(ann, "namespace", None) == "qpi://vectra":
                for k, v in (ann.value or {}).items():
                    m = _CH_ANN_RE.match(k)
                    if m:
                        ch_ann.setdefault(int(m.group(1)), {})[m.group(2)] = v

        all_fields: Set[str] = set()
        for ch_fields in ch_ann.values():
            all_fields.update(ch_fields.keys())
        for field_name in sorted(all_fields):
            vals: List[Any] = [ch_ann.get(i, {}).get(field_name) for i in range(n)]
            if any(v is not None for v in vals):
                coords[f"qpi_{field_name}"] = (channel_dim, vals)


def _drop_none(obj: Any) -> Any:
    """Recursively remove None values from dicts/lists (for clean attrs)."""
    if isinstance(obj, dict):
        return {k: _drop_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_drop_none(v) for v in obj]
    return obj


def qptiff_meta_to_root_attrs(meta: "QptiffMetadata") -> Dict[str, Any]:
    """
    Build structured nested attrs for the DataTree root from a QptiffMetadata.

    Returns a dict with two top-level keys:

    ``"slide_info"``
        SlideInfo fields — instrument type, operator, study name, acquisition
        software, datetime, barcode, slide ID.

    ``"image_info"``
        ImageInfo fields for the FullResolution scene — scan mode, objective,
        pixel size, stage position, camera and scan resolution info.

    Per-channel metadata lives on the ``c`` xarray coordinates (see
    ``channel_infos_to_coords`` in multiscale.py), not in root attrs.
    None-valued fields are omitted. The result is JSON-serializable.
    """
    from dataclasses import asdict

    result: Dict[str, Any] = {}

    slide_dict = _drop_none(asdict(meta.slide))
    if slide_dict:
        result["slide_info"] = slide_dict

    fr = meta.full_resolution or (meta.images[0] if meta.images else None)
    if fr is not None:
        image_dict = _drop_none(asdict(fr.image_info))
        if image_dict:
            result["image_info"] = image_dict

    return result
