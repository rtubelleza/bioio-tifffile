#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the OME transport layer (bioio_tifffile.ome)."""

import ast
import pathlib
import warnings

import pytest
from ome_types.model import MapAnnotation

from bioio_tifffile.ome.model import (
    VENDOR_CHANNEL_KEYS,
    VENDOR_IMAGE_KEYS,
    ome_metadata_from_qptiff,
)
from bioio_tifffile.qptiff_types import (
    CameraInfo,
    ChannelInfo,
    ImageInfo,
    QptiffImageSceneMetadata,
    QptiffMetadata,
    SlideInfo,
)

_PKG = pathlib.Path(__file__).resolve().parent.parent

#: Modules that model, parse or assemble. None of them may reach for the
#: transport layer — that dependency direction is the whole point of ome/.
CORE_MODULES = ["qptiff_types.py", "qptiff_parser.py", "multiscale.py"]


def _meta(**ch_kwargs: object) -> QptiffMetadata:
    ch = ChannelInfo(index=0, name="CI.PARP-Atto550", **ch_kwargs)  # type: ignore[arg-type]
    return QptiffMetadata(
        slide=SlideInfo(acquisition_software="Fusion 2.3.1", operator_name="Admin"),
        images=[
            QptiffImageSceneMetadata(
                image_info=ImageInfo(scan_mode="im_Fluorescence"), channels=[ch]
            )
        ],
    )


class TestLayerBoundary:
    """The core must not import the transport layer.

    Checked in the source rather than via sys.modules: bioio_base itself
    imports ome_types, so runtime absence is unachievable and would make the
    test vacuous. The source-level rule is the one that actually stops the
    layering rotting.
    """

    @pytest.mark.parametrize("module", CORE_MODULES)
    def test_core_does_not_import_ome(self, module: str) -> None:
        tree = ast.parse((_PKG / module).read_text())
        imported: list = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        banned = [
            m
            for m in imported
            if m.split(".")[0] in ("ome_types", "ome_zarr_models")
            or m.startswith("ome.")
            or m == "ome"
        ]
        assert not banned, f"{module} must not import {banned}"


class TestVerifiedBugs:
    """Each of these reproduced a real failure before the fix."""

    def test_non_power_of_two_binning_degrades(self) -> None:
        """binning=3 used to raise a pydantic ValidationError straight out of
        Reader.ome_metadata."""
        m = _meta(binning=3, gain=1.0)
        m.images[0].image_info.camera = CameraInfo(camera_type="IMX421")
        ome = ome_metadata_from_qptiff(m)
        binning = ome.images[0].pixels.channels[0].detector_settings.binning
        assert binning.value == "Other"

    @pytest.mark.parametrize(
        "stored_bits, dtype, expected",
        [(8, None, "uint8"), (14, None, "uint16"), (None, "uint8", "uint8")],
    )
    def test_pixel_type_follows_storage(
        self, stored_bits: int, dtype: str, expected: str
    ) -> None:
        """Pixels.type was hardcoded uint16, contradicting 8-bit brightfield."""
        m = _meta()
        m.images[0].image_info.stored_bits_per_sample = stored_bits
        ome = ome_metadata_from_qptiff(m, dtype_name=dtype)
        assert ome.images[0].pixels.type.value == expected

    def test_no_detector_settings_without_detector(self) -> None:
        """DetectorSettings used to reference a Detector:0 that was never
        built, producing 'Reference to unknown ID' on parse."""
        ome = ome_metadata_from_qptiff(_meta(gain=1.5))
        assert ome.instruments == [] or not ome.instruments[0].detectors
        assert ome.images[0].pixels.channels[0].detector_settings is None

    def test_roundtrip_has_no_dangling_references(self) -> None:
        import ome_types

        m = _meta(
            gain=1.0,
            binning=2,
            emission_cut_on_nm=570.0,
            emission_cut_off_nm=640.0,
            emission_filter_name="DAPI / ATTO550 / AF750",
        )
        m.images[0].image_info.camera = CameraInfo(camera_type="IMX421")
        ome = ome_metadata_from_qptiff(m)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ome_types.from_xml(ome.to_xml())
        dangling = [w for w in caught if "unknown ID" in str(w.message)]
        assert not dangling, [str(w.message) for w in dangling]


class TestTypedMapping:
    """Metadata that used to be stringified now lands on typed OME fields."""

    def test_transmittance_range_from_band_edges(self) -> None:
        ome = ome_metadata_from_qptiff(
            _meta(
                emission_filter_name="DAPI / ATTO550 / AF750",
                emission_cut_on_nm=570.0,
                emission_cut_off_nm=640.0,
                n_filter_bands=3,
            )
        )
        filt = ome.instruments[0].filters[0]
        assert filt.transmittance_range.cut_in == 570.0
        assert filt.transmittance_range.cut_out == 640.0
        assert filt.type.value == "MultiPass"

    def test_single_band_cube_is_band_pass(self) -> None:
        ome = ome_metadata_from_qptiff(
            _meta(
                emission_filter_name="F",
                emission_cut_on_nm=570.0,
                emission_cut_off_nm=640.0,
                n_filter_bands=1,
            )
        )
        assert ome.instruments[0].filters[0].type.value == "BandPass"

    def test_channel_mode_enums(self) -> None:
        fluor = ome_metadata_from_qptiff(_meta()).images[0].pixels.channels[0]
        assert fluor.contrast_method.value == "Fluorescence"
        assert fluor.acquisition_mode.value == "WideField"
        assert fluor.illumination_type.value == "Epifluorescence"

        bf = _meta(is_brightfield=True)
        bf.images[0].image_info.scan_mode = "im_Brightfield"
        bfc = ome_metadata_from_qptiff(bf).images[0].pixels.channels[0]
        assert bfc.contrast_method.value == "Brightfield"
        assert bfc.illumination_type.value == "Transmitted"

    def test_lamp_becomes_a_light_source(self) -> None:
        m = _meta()
        m.images[0].image_info.lamp_type = "XCiteMultiBandLed"
        ome = ome_metadata_from_qptiff(m)
        leds = ome.instruments[0].light_emitting_diodes
        assert len(leds) == 1 and leds[0].model == "XCiteMultiBandLed"
        assert ome.images[0].pixels.channels[0].light_source_settings is not None

    def test_roi_from_camera_region(self) -> None:
        ome = ome_metadata_from_qptiff(
            _meta(roi_x=0, roi_y=8, roi_width=1920, roi_height=1440)
        )
        rect = ome.rois[0].union[0]
        assert (rect.x, rect.y, rect.width, rect.height) == (0.0, 8.0, 1920.0, 1440.0)
        assert rect.the_c == 0

    def test_filter_set_pairs_excitation_and_emission(self) -> None:
        ome = ome_metadata_from_qptiff(
            _meta(excitation_filter_name="Ex", emission_filter_name="Em")
        )
        fs = ome.instruments[0].filter_sets
        assert len(fs) == 1
        assert fs[0].excitation_filters and fs[0].emission_filters
        assert ome.images[0].pixels.channels[0].filter_set_ref is not None

    def test_creator_and_uuid(self) -> None:
        m = _meta()
        m.slide.identifier = "39a42dfe-bbd8-4136-a48f-f03ed563f9de"
        ome = ome_metadata_from_qptiff(m)
        assert ome.creator == "Fusion 2.3.1"
        assert ome.uuid.endswith("39a42dfe-bbd8-4136-a48f-f03ed563f9de")

    def test_non_uuid_identifier_is_not_forced_into_uuid(self) -> None:
        m = _meta()
        m.slide.identifier = "test-uuid-1234"
        assert ome_metadata_from_qptiff(m).uuid != "urn:uuid:test-uuid-1234"

    def test_stage_label_replaces_per_plane_position(self) -> None:
        m = _meta()
        m.images[0].image_info.xposition_um = 1000.0
        m.images[0].image_info.yposition_um = 2000.0
        ome = ome_metadata_from_qptiff(m)
        assert ome.images[0].stage_label.x == 1000.0
        assert ome.images[0].pixels.planes[0].position_x is None


class TestVendorBlockShrinks:
    """The MapAnnotation is now the residue, and must not silently regrow."""

    def _vendor_keys(self, ome: object) -> set:
        maps = [a for a in ome.structured_annotations if isinstance(a, MapAnnotation)]
        return {kv.k for kv in maps[0].value.ms} if maps else set()

    def test_only_declared_keys_are_emitted(self) -> None:
        m = _meta(signal_units=64, auto_expose_type="aet_Fluorescence", gain=1.0)
        m.slide.computer_name = "DESKTOP-9H4V0RJ"
        m.images[0].image_info.jpeg_quality = 90
        keys = self._vendor_keys(ome_metadata_from_qptiff(m))
        undeclared = [
            k
            for k in keys
            if k not in VENDOR_IMAGE_KEYS
            and not (k.startswith("ch0_") and k[4:] in VENDOR_CHANNEL_KEYS)
        ]
        assert not undeclared, undeclared

    @pytest.mark.parametrize(
        "duplicated", ["datetime", "camera_gain", "camera_bit_depth", "channel_count"]
    )
    def test_values_with_typed_homes_are_not_duplicated(self, duplicated: str) -> None:
        m = _meta(gain=1.0, bit_depth=14)
        m.slide.datetime = "2026:08:07 10:00:00"
        m.images[0].image_info.camera = CameraInfo(camera_type="IMX421", gain=1.0)
        assert duplicated not in self._vendor_keys(ome_metadata_from_qptiff(m))
