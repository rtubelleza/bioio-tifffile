#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
QptiffMetadata -> ome_types.model.OME.

This is a *transport* layer: OME is a serialisation target, not the in-memory
model. :class:`~bioio_tifffile.qptiff_types.QptiffMetadata` is the source of
truth; nothing outside ``bioio_tifffile.ome`` should import ``ome_types``.

The mapping favours typed OME fields over the vendor annotation. Only fields
with genuinely no OME slot end up in the ``qpi://vectra`` MapAnnotation — see
:data:`_VENDOR_ONLY_NOTE`.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from ..qptiff_types import ChannelInfo, QptiffMetadata

_TIFF_DATETIME_RE = re.compile(
    r"^(\d{4}):(\d{2}):(\d{2})[ T](\d{2}:\d{2}:\d{2}(?:\.\d+)?)"
)

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-" r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

#: OME's Binning enum only admits these. Anything else must degrade to OTHER
#: rather than raising out of Reader.ome_metadata.
_VALID_BINNING = {1: "1x1", 2: "2x2", 4: "4x4", 8: "8x8"}

_VENDOR_ONLY_NOTE = """\
Fields kept in the qpi://vectra MapAnnotation because OME has no slot for them.
Everything else must go to a typed field; the test suite asserts this set does
not grow silently."""

#: Image-scope keys permitted in the vendor annotation.
VENDOR_IMAGE_KEYS = frozenset(
    {
        "description_version",
        "image_type",
        "computer_name",
        "validation_code",
        "jpeg_quality",
        "saturation_protection_type",
        "is_rna",
        "rotate_image",
        "mirror_image",
        "is_tma",
        "opal_kit_type",
        "slide_id",
        "barcode",
        "compression",
        # parser provenance, deliberately not OME concepts
        "acquisition_format",
        "channel_locus",
        "structure_signature",
    }
)

#: Per-channel keys permitted in the vendor annotation (without the ch<N>_ prefix).
VENDOR_CHANNEL_KEYS = frozenset(
    {
        "is_unmixed_component",
        "signal_units",
        "objective",
        "autofluorescence_subtracted",
        "auto_expose_type",
        "responsivity",
        "responsivity_filter_id",
        "responsivity_filter_name",
        "camera_orientation",
    }
)


def _tiff_datetime_to_iso(dt: Optional[str]) -> Optional[str]:
    """Convert a TIFF DateTime ("YYYY:MM:DD HH:MM:SS") to ISO 8601.

    Pydantic's datetime validator rejects the colon-separated date form the
    TIFF spec uses. Anything that already parses is passed through.
    """
    if not dt:
        return dt
    m = _TIFF_DATETIME_RE.match(dt.strip())
    if m is None:
        return dt
    y, mo, d, t = m.groups()
    return f"{y}-{mo}-{d}T{t}"


def _str(v: object) -> Optional[str]:
    return str(v) if v is not None else None


def _ome_color(rgb: Optional[Tuple[int, int, int]]) -> Optional[object]:
    if rgb is None:
        return None
    try:
        from ome_types.model import Color

        return Color(f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}")
    except Exception:
        return None


def _format_binning(b: Optional[int]) -> Optional[str]:
    """Map a binning factor onto OME's Binning enum.

    OME admits only 1x1/2x2/4x4/8x8/Other. Returning "3x3" for a 3x binned
    acquisition raises a pydantic ValidationError out of Reader.ome_metadata,
    so anything unrecognised degrades to "Other".
    """
    if b is None:
        return None
    return _VALID_BINNING.get(b, "Other")


def _pixel_type(stored_bits: Optional[int], dtype_name: Optional[str]) -> str:
    """Choose Pixels.type from the real storage width.

    Brightfield QPTIFFs store 8-bit RGB; hardcoding uint16 contradicts the
    array actually written to zarr. An explicit dtype wins when supplied.
    """
    if dtype_name in (
        "uint8",
        "uint16",
        "uint32",
        "int8",
        "int16",
        "int32",
        "float",
        "double",
        "complex",
        "double-complex",
        "bit",
    ):
        return dtype_name
    if stored_bits is not None and stored_bits <= 8:
        return "uint8"
    return "uint16"


def _light_source_for(lamp: str, lsid: str) -> object:
    """Build the LightSource subclass that best matches a vendor lamp string."""
    from ome_types.model import (
        Arc,
        Filament,
        GenericExcitationSource,
        LightEmittingDiode,
    )

    low = lamp.lower()
    if "led" in low:
        return LightEmittingDiode(id=lsid, model=lamp)
    if "hg" in low or "mercury" in low:
        return Arc(id=lsid, model=lamp, type="Hg")
    if "xe" in low or "xenon" in low:
        return Arc(id=lsid, model=lamp, type="Xe")
    if "halogen" in low or "tungsten" in low or "filament" in low:
        return Filament(id=lsid, model=lamp, type="Halogen")
    return GenericExcitationSource(id=lsid, model=lamp)


def _channel_modes(ch: ChannelInfo, scan_mode: Optional[str]) -> Dict[str, str]:
    """Map scan mode / brightfield flag onto the three Channel enums."""
    brightfield = bool(ch.is_brightfield) or (
        scan_mode is not None and "brightfield" in scan_mode.lower()
    )
    if brightfield:
        return {
            "acquisition_mode": "BrightField",
            "contrast_method": "Brightfield",
            "illumination_type": "Transmitted",
        }
    return {
        "acquisition_mode": "WideField",
        "contrast_method": "Fluorescence",
        "illumination_type": "Epifluorescence",
    }


def _transmittance_range(
    cut_on: Optional[float], cut_off: Optional[float]
) -> Optional[object]:
    if cut_on is None or cut_off is None:
        return None
    from ome_types.model import TransmittanceRange

    return TransmittanceRange(
        cut_in=cut_on, cut_in_unit="nm", cut_out=cut_off, cut_out_unit="nm"
    )


def ome_metadata_from_qptiff(
    qpi: QptiffMetadata,
    scene_name: Optional[str] = None,
    size_x: Optional[int] = None,
    size_y: Optional[int] = None,
    size_z: Optional[int] = None,
    size_c: Optional[int] = None,
    size_t: Optional[int] = None,
    dtype_name: Optional[str] = None,
    include_raw_xml: bool = False,
) -> object:
    """
    Map a :class:`QptiffMetadata` onto a fully typed ``ome_types.model.OME``.

    Everything with an OME equivalent lands on a typed field: filters carry
    their real passbands via ``TransmittanceRange``, filter pairs become a
    ``FilterSet``, the lamp becomes a ``LightSource``, per-channel ROIs become
    ``Rectangle`` shapes, and stage position becomes a ``StageLabel``. Only the
    residue in :data:`VENDOR_IMAGE_KEYS` / :data:`VENDOR_CHANNEL_KEYS` goes to
    the ``qpi://vectra`` MapAnnotation.

    Parameters
    ----------
    qpi:
        Parsed QPTIFF metadata.
    scene_name:
        Human-readable scene name (becomes ``Image.name``).
    size_x, size_y, size_z, size_c, size_t:
        Pixel dimensions at the selected pyramid level.
    dtype_name:
        Array dtype (e.g. ``"uint8"``), used to pick ``Pixels.type`` instead of
        assuming 16-bit.
    include_raw_xml:
        Attach the original vendor XML as an ``XMLAnnotation``. Off by default
        because it is large.
    """
    from ome_types.model import (
        OME,
    )
    from ome_types.model import ROI as OMEROI
    from ome_types.model import (
        AnnotationRef,
        Channel,
        Detector,
        DetectorSettings,
        Experiment,
        Experimenter,
        ExperimenterRef,
        ExperimentRef,
        Filter,
        FilterRef,
        FilterSet,
        FilterSetRef,
        Image,
        Instrument,
        InstrumentRef,
        LightPath,
        LightSourceSettings,
        MapAnnotation,
        Microscope,
        Objective,
        ObjectiveSettings,
        Pixels,
        Plane,
        Rectangle,
        StageLabel,
        TimestampAnnotation,
        XMLAnnotation,
    )

    instrument_id = "Instrument:0"
    objective_id = "Objective:0"
    detector_id = "Detector:0"
    lightsource_id = "LightSource:0"
    image_id = "Image:0"
    pixels_id = "Pixels:0"
    experimenter_id = "Experimenter:0"
    experiment_id = "Experiment:0"

    sl = qpi.slide
    ii = qpi._primary.image_info

    # ---------------------------------------------------------------- instrument
    microscope = Microscope(model=qpi.instrument_type) if qpi.instrument_type else None

    objective = None
    if qpi.scan_resolution.objective_name or qpi.scan_resolution.magnification:
        objective = Objective(
            id=objective_id,
            model=qpi.scan_resolution.objective_name or ii.objective,
            nominal_magnification=qpi.scan_resolution.magnification,
        )

    detector = None
    if qpi.camera.camera_type or qpi.camera.camera_name:
        detector = Detector(
            id=detector_id,
            model=qpi.camera.camera_type,
            manufacturer=qpi.camera.camera_name,
            gain=qpi.camera.gain,
        )

    # lamp -> a typed LightSource, referenced from every channel
    light_sources: List[object] = []
    lamp = ii.lamp_type or ii.bf_lamp_type
    if lamp:
        light_sources.append(_light_source_for(lamp, lightsource_id))

    # ------------------------------------------------------------------ filters
    # One Filter per excitation/emission cube per channel, carrying its real
    # passband, plus a FilterSet tying the pair together.
    filters: List[object] = []
    filter_sets: List[object] = []
    exc_filter_ids: Dict[int, str] = {}
    emi_filter_ids: Dict[int, str] = {}
    filter_set_ids: Dict[int, str] = {}
    fidx = 0

    def _filter_type(n_bands: Optional[int]) -> str:
        return "MultiPass" if (n_bands or 1) > 1 else "BandPass"

    for ch in qpi.channels:
        made: List[str] = []
        for name, manuf, part, cut_on, cut_off, bucket in (
            (
                ch.excitation_filter_name,
                ch.excitation_filter_manufacturer,
                ch.excitation_filter_part_no,
                ch.excitation_cut_on_nm,
                ch.excitation_cut_off_nm,
                exc_filter_ids,
            ),
            (
                ch.emission_filter_name,
                ch.emission_filter_manufacturer,
                ch.emission_filter_part_no,
                ch.emission_cut_on_nm,
                ch.emission_cut_off_nm,
                emi_filter_ids,
            ),
        ):
            if not (name or manuf or part or cut_on):
                made.append("")
                continue
            fid = f"Filter:{fidx}"
            fidx += 1
            filters.append(
                Filter(
                    id=fid,
                    model=name,
                    manufacturer=manuf,
                    lot_number=part,
                    type=_filter_type(ch.n_filter_bands),
                    transmittance_range=_transmittance_range(cut_on, cut_off),
                )
            )
            bucket[ch.index] = fid
            made.append(fid)

        exc_fid, emi_fid = (made + ["", ""])[:2]
        if exc_fid or emi_fid:
            fsid = f"FilterSet:{ch.index}"
            filter_sets.append(
                FilterSet(
                    id=fsid,
                    model=ch.excitation_filter_name or ch.emission_filter_name,
                    manufacturer=ch.excitation_filter_manufacturer,
                    excitation_filters=[FilterRef(id=exc_fid)] if exc_fid else [],
                    emission_filters=[FilterRef(id=emi_fid)] if emi_fid else [],
                )
            )
            filter_set_ids[ch.index] = fsid

    instrument = Instrument(
        id=instrument_id,
        microscope=microscope,
        objectives=[objective] if objective else [],
        detectors=[detector] if detector else [],
        filters=filters,
        filter_sets=filter_sets,
        **_light_source_kwargs(light_sources),
    )
    # B4: only reference an instrument that actually carries something
    has_instrument = bool(
        microscope or objective or detector or filters or light_sources
    )

    experimenter = (
        Experimenter(id=experimenter_id, user_name=qpi.operator_name)
        if qpi.operator_name
        else None
    )

    experiment = None
    exp_desc = " / ".join(p for p in (qpi.study_name, qpi.scan_profile_name) if p)
    if exp_desc:
        experiment = Experiment(id=experiment_id, description=exp_desc)

    # ------------------------------------------------------- channels and planes
    ome_channels: List[object] = []
    ome_planes: List[object] = []
    rois: List[object] = []
    timestamp_anns: List[object] = []

    for ch in qpi.channels:
        # B4: DetectorSettings must not reference a Detector that was not built
        det_settings = None
        if detector is not None and (
            ch.gain is not None
            or ch.binning is not None
            or ch.offset_counts is not None
        ):
            det_settings = DetectorSettings(
                id=detector_id,
                gain=ch.gain,
                binning=_format_binning(ch.binning),
                offset=(
                    float(ch.offset_counts) if ch.offset_counts is not None else None
                ),
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
            **_channel_modes(ch, ii.scan_mode),
        )
        if light_sources:
            ch_kwargs["light_source_settings"] = LightSourceSettings(id=lightsource_id)
        if filter_set_ids.get(ch.index):
            ch_kwargs["filter_set_ref"] = FilterSetRef(id=filter_set_ids[ch.index])
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

        # one Plane per channel (Z=0, T=0). Stage position is per-image, so it
        # lives on Image.stage_label rather than being copied onto every plane.
        plane_kwargs: Dict[str, object] = dict(the_z=0, the_t=0, the_c=ch.index)
        if ch.exposure_time_us is not None:
            plane_kwargs["exposure_time"] = ch.exposure_time_us
            plane_kwargs["exposure_time_unit"] = "µs"
        ome_planes.append(Plane(**plane_kwargs))  # type: ignore[arg-type]

        # camera ROI -> a typed Rectangle bound to this channel
        if ch.roi_width is not None and ch.roi_height is not None:
            rois.append(
                OMEROI(
                    id=f"ROI:{ch.index}",
                    union=[
                        Rectangle(
                            id=f"Shape:{ch.index}",
                            x=float(ch.roi_x or 0),
                            y=float(ch.roi_y or 0),
                            width=float(ch.roi_width),
                            height=float(ch.roi_height),
                            the_c=ch.index,
                        )
                    ],
                )
            )

        if ch.responsivity_date:
            timestamp_anns.append(
                TimestampAnnotation(
                    id=f"Annotation:ts:{ch.index}",
                    namespace="qpi://vectra/responsivity",
                    value=_tiff_datetime_to_iso(ch.responsivity_date),
                )
            )

    # -------------------------------------------------------------------- pixels
    sig_bits = next(
        (ch.bit_depth for ch in qpi.channels if ch.bit_depth is not None), None
    )
    if sig_bits is None:
        sig_bits = getattr(qpi.camera, "bit_depth", None)

    px_kwargs: Dict[str, object] = dict(
        id=pixels_id,
        dimension_order="XYZCT",
        type=_pixel_type(ii.stored_bits_per_sample, dtype_name),
        size_x=size_x or 1,
        size_y=size_y or 1,
        size_z=size_z or 1,
        size_c=size_c or max(1, len(qpi.channels)),
        size_t=size_t or 1,
    )
    if sig_bits is not None:
        px_kwargs["significant_bits"] = sig_bits
    scale = ii.scale_factor if ii.scale_factor is not None else qpi.pixel_size_um
    if scale is not None:
        px_kwargs["physical_size_x"] = scale
        px_kwargs["physical_size_x_unit"] = "µm"
        px_kwargs["physical_size_y"] = scale
        px_kwargs["physical_size_y_unit"] = "µm"
    px_kwargs["channels"] = ome_channels
    px_kwargs["planes"] = ome_planes
    pixels = Pixels(**px_kwargs)  # type: ignore[arg-type]

    # ------------------------------------------------------- vendor leftovers
    img_qpi = _vendor_annotation_values(qpi)

    annotations: List[object] = []
    annotation_refs: List[object] = []
    if img_qpi:
        annotations.append(
            MapAnnotation(id="Annotation:0", namespace="qpi://vectra", value=img_qpi)
        )
        annotation_refs.append(AnnotationRef(id="Annotation:0"))
    annotations.extend(timestamp_anns)
    if include_raw_xml and qpi.raw_xml:
        annotations.append(
            XMLAnnotation(
                id="Annotation:xml",
                namespace="qpi://vectra/raw",
                value=qpi.raw_xml,
            )
        )

    stage_label = None
    if qpi.xposition_um is not None or qpi.yposition_um is not None:
        stage_label = StageLabel(
            name=qpi.slide_id or "stage",
            x=qpi.xposition_um,
            x_unit="µm" if qpi.xposition_um is not None else None,
            y=qpi.yposition_um,
            y_unit="µm" if qpi.yposition_um is not None else None,
        )

    image = Image(
        id=image_id,
        name=scene_name,
        description=sl.sample_description,
        acquisition_date=_tiff_datetime_to_iso(qpi.datetime),
        instrument_ref=InstrumentRef(id=instrument_id) if has_instrument else None,
        objective_settings=(
            ObjectiveSettings(
                id=objective_id,
                correction_collar=_as_float(ii.coverslip_thickness),
            )
            if objective
            else None
        ),
        experimenter_ref=ExperimenterRef(id=experimenter_id) if experimenter else None,
        experiment_ref=ExperimentRef(id=experiment_id) if experiment else None,
        stage_label=stage_label,
        pixels=pixels,
        annotation_refs=annotation_refs,
        roi_refs=[],
    )

    ome_kwargs: Dict[str, object] = {
        "images": [image],
        "instruments": [instrument] if has_instrument else [],
        "structured_annotations": annotations,
        "rois": rois,
        "creator": qpi.acquisition_software,
    }
    if qpi.identifier and _UUID_RE.match(qpi.identifier):
        ome_kwargs["uuid"] = f"urn:uuid:{qpi.identifier}"
    if experimenter:
        ome_kwargs["experimenters"] = [experimenter]
    if experiment:
        ome_kwargs["experiments"] = [experiment]

    return OME(**ome_kwargs)  # type: ignore[arg-type]


def _light_source_kwargs(light_sources: List[object]) -> Dict[str, List[object]]:
    """Route each LightSource instance to the Instrument list its class needs."""
    if not light_sources:
        return {}
    from ome_types.model import (
        Arc,
        Filament,
        GenericExcitationSource,
        LightEmittingDiode,
    )

    buckets: Dict[str, List[object]] = {}
    for ls in light_sources:
        if isinstance(ls, LightEmittingDiode):
            key = "light_emitting_diodes"
        elif isinstance(ls, Arc):
            key = "arcs"
        elif isinstance(ls, Filament):
            key = "filaments"
        elif isinstance(ls, GenericExcitationSource):
            key = "generic_excitation_sources"
        else:  # pragma: no cover - defensive
            continue
        buckets.setdefault(key, []).append(ls)
    return buckets


def _as_float(v: object) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(str(v).strip().rstrip("m").strip())
    except (ValueError, TypeError):
        return None


def _vendor_annotation_values(qpi: QptiffMetadata) -> Dict[str, str]:
    """Build the qpi://vectra map from the leftovers only.

    Anything with a typed OME home is deliberately absent, including the four
    values that used to be emitted twice (datetime, camera gain, camera bit
    depth, channel count).
    """
    import json

    ii = qpi._primary.image_info
    sl = qpi.slide

    out: Dict[str, str] = {}
    image_src: Dict[str, object] = {
        "description_version": qpi.description_version,
        "image_type": qpi.image_type,
        "computer_name": qpi.computer_name,
        "validation_code": sl.validation_code,
        "jpeg_quality": ii.jpeg_quality,
        "saturation_protection_type": ii.saturation_protection_type,
        "is_rna": ii.is_rna,
        "rotate_image": ii.rotate_image,
        "mirror_image": ii.mirror_image,
        "is_tma": ii.is_tma,
        "opal_kit_type": ii.opal_kit_type,
        "slide_id": qpi.slide_id,
        "barcode": qpi.barcode,
        "compression": ii.compression,
        "acquisition_format": qpi.acquisition_format,
        "channel_locus": qpi.channel_locus,
        "structure_signature": (
            json.dumps(qpi.structure_signature, sort_keys=True)
            if qpi.structure_signature
            else None
        ),
    }
    for k, v in image_src.items():
        if v is not None:
            assert k in VENDOR_IMAGE_KEYS, f"undeclared vendor key {k!r}"
            out[k] = _str(v) or ""

    for ch in qpi.channels:
        ch_src: Dict[str, object] = {
            "is_unmixed_component": ch.is_unmixed_component,
            "signal_units": ch.signal_units,
            "objective": ch.objective,
            "autofluorescence_subtracted": ch.autofluorescence_subtracted,
            "auto_expose_type": ch.auto_expose_type,
            "responsivity": ch.responsivity,
            "responsivity_filter_id": ch.responsivity_filter_id,
            "responsivity_filter_name": ch.responsivity_filter_name,
            "camera_orientation": ch.camera_orientation,
        }
        for k, v in ch_src.items():
            if v is not None:
                assert k in VENDOR_CHANNEL_KEYS, f"undeclared vendor key {k!r}"
                out[f"ch{ch.index}_{k}"] = _str(v) or ""

    return out
