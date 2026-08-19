#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for bioio_tifffile Reader and metadata parser.
Include qptiff file cases.
"""

import pathlib
import re
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pytest
from bioio_base import exceptions, test_utilities
from distributed import Client, LocalCluster

from bioio_tifffile import Reader
from bioio_tifffile.qptiff_metadata import (
    FORMAT_BRIGHTFIELD,
    FORMAT_FUSION_PAGED,
    FORMAT_POLARIS_SCANBAND,
    LOCUS_PER_PAGE_ROOT,
    LOCUS_RGB_SAMPLES,
    LOCUS_SHARED_SCANBANDS,
    LOCUS_UNKNOWN,
    ome_metadata_from_qptiff,
    ome_to_flat_attrs,
    parse_qpi_xml,
)

from .conftest import LOCAL_RESOURCES_DIR


@pytest.mark.parametrize(
    "filename, "
    "set_scene, "
    "expected_scenes, "
    "expected_shape, "
    "expected_dtype, "
    "expected_dims_order, "
    "expected_channel_names, "
    "expected_physical_pixel_sizes",
    [
        # brightfield RGB (YXS): pixel_size from QPI XML (0.25 µm)
        (
            "s_1_bf_yx3.qptiff",
            "FullResolution",
            ("FullResolution",),
            (64, 64, 3),
            np.uint8,
            "YXS",
            None,  # S-dim names not surfaced via bioio channel_names
            (None, 0.25, 0.25),
        ),
        # fluorescence CYX-written-as-QYX: tifffile guesses Q->Z
        # channel_names is None because no C coordinate exists (dim is Z)
        (
            "s_1_fluor_cyx.qptiff",
            "FullResolution",
            ("FullResolution",),
            (3, 64, 64),
            np.uint16,
            "ZYX",
            None,
            (None, 0.5, 0.5),
        ),
    ],
)
def test_qptiff_reader(
    filename: str,
    set_scene: str,
    expected_scenes: Tuple[str, ...],
    expected_shape: Tuple[int, ...],
    expected_dtype: np.dtype,
    expected_dims_order: str,
    expected_channel_names: Optional[List[str]],
    expected_physical_pixel_sizes: Tuple[
        Optional[float], Optional[float], Optional[float]
    ],
) -> None:
    """
    Standard bioio reader checks via test_utilities.run_image_file_checks.

    Covers: scene names, shape, dtype, dims order, channel_names,
    physical_pixel_sizes, metadata type, dask vs in-memory read consistency,
    file handle leak detection, and Dask distributed serialization.
    """
    uri = LOCAL_RESOURCES_DIR / filename
    test_utilities.run_image_file_checks(
        ImageContainer=Reader,
        image=uri,
        set_scene=set_scene,
        expected_scenes=expected_scenes,
        expected_current_scene=set_scene,
        expected_shape=expected_shape,
        expected_dtype=expected_dtype,
        expected_dims_order=expected_dims_order,
        expected_channel_names=expected_channel_names,
        expected_physical_pixel_sizes=expected_physical_pixel_sizes,
        expected_metadata_type=str,  # METADATA_PROCESSED is the raw QPI XML string
        reader_kwargs={},
    )


def test_qptiff_reader_rejects_non_qptiff(sample_text_file: pathlib.Path) -> None:
    with pytest.raises(exceptions.UnsupportedFileFormatError):
        Reader(sample_text_file)


BRIGHTFIELD_XML = """\
<?xml version="1.0" encoding="utf-16"?>
<PerkinElmer-QPI-ImageDescription>
  <DescriptionVersion>TestDescriptionVersion</DescriptionVersion>
  <AcquisitionSoftware>TestSoftware X.X.X</AcquisitionSoftware>
  <ImageType>FullResolution</ImageType>
  <Identifier>test-uuid-1234</Identifier>
  <SlideID>TestSlide_001</SlideID>
  <Barcode>BC123</Barcode>
  <StudyName>TestStudy</StudyName>
  <OperatorName>Tester</OperatorName>
  <ComputerName>TEST-PC</ComputerName>
  <Objective>20x</Objective>
  <ExposureTime>5000</ExposureTime>
  <ExposureTimeArray>
    <Value>5000</Value>
    <Value>5000</Value>
    <Value>5000</Value>
  </ExposureTimeArray>
  <InstrumentType>TestInstrumentType</InstrumentType>
  <BFLampType>TestBFLampType</BFLampType>
  <CameraType>TestCameraType</CameraType>
  <CameraName>TestCameraName</CameraName>
  <CameraSettings>
    <Gain>1</Gain>
    <Binning>1</Binning>
    <BitDepth>12</BitDepth>
  </CameraSettings>
  <ScanProfile>
    <root s_v="1" ref="ref0">
      <Mode>im_Brightfield</Mode>
      <Name>TestBrightfieldName</Name>
      <SampleIsTMA>true</SampleIsTMA>
      <OpalKitType>TestOpalKitType</OpalKitType>
      <ScanResolution ref="ref1">
        <PixelSizeMicrons>0.25</PixelSizeMicrons>
        <Magnification>40</Magnification>
        <ObjectiveName>20x</ObjectiveName>
        <Binning>1</Binning>
      </ScanResolution>
    </root>
  </ScanProfile>
</PerkinElmer-QPI-ImageDescription>
"""

FLUORESCENCE_XML = """\
<?xml version="1.0" encoding="utf-16"?>
<PerkinElmer-QPI-ImageDescription>
  <DescriptionVersion>TestDescriptionVersion</DescriptionVersion>
  <AcquisitionSoftware>TestSoftware Y.Y.Y</AcquisitionSoftware>
  <ImageType>FullResolution</ImageType>
  <Identifier>test-uuid-5678</Identifier>
  <SlideID>TestSlide_002</SlideID>
  <ExposureTimeArray>
    <Value>100</Value>
    <Value>200</Value>
    <Value>150</Value>
  </ExposureTimeArray>
  <ScanProfile>
    <root>
      <Mode>im_Fluorescence</Mode>
      <Name>TestFluorescenceName</Name>
      <SampleIsTMA>false</SampleIsTMA>
      <ScanResolution>
        <PixelSizeMicrons>0.5</PixelSizeMicrons>
        <Magnification>20</Magnification>
        <ObjectiveName>20x</ObjectiveName>
      </ScanResolution>
      <ScanBands>
        <ScanBands-i>
          <Biomarker>CD3</Biomarker>
          <Fluorophore>OPAL570</Fluorophore>
          <HomeWavelength>570</HomeWavelength>
        </ScanBands-i>
        <ScanBands-i>
          <Biomarker>CD8</Biomarker>
          <Fluorophore>OPAL690</Fluorophore>
          <HomeWavelength>690</HomeWavelength>
        </ScanBands-i>
        <ScanBands-i>
          <BioMarker>FOXP3</BioMarker>
          <Fluorophore>OPAL780</Fluorophore>
          <HomeWavelength>780</HomeWavelength>
        </ScanBands-i>
      </ScanBands>
    </root>
  </ScanProfile>
</PerkinElmer-QPI-ImageDescription>
"""


def _fusion_page_xml(biomarker: str, filter_name: str) -> str:
    """Build one Fusion 2.x per-page ImageDescription.

    ``<Name>`` is the filter / fluorophore and ``<Biomarker>`` the stain;
    neither ``<Fluorophore>`` nor ``<Fluor>`` is ever emitted by these files.
    """
    return f"""\
<?xml version="1.0" encoding="utf-16"?>
<PerkinElmer-QPI-ImageDescription>
  <ImageType>FullResolution</ImageType>
  <SlideID>TestSlide_003</SlideID>
  <Name>{filter_name}</Name>
  <Biomarker>{biomarker}</Biomarker>
  <Responsivity>
    <Filter>
      <Name>{filter_name}</Name>
      <Response>244.87</Response>
    </Filter>
  </Responsivity>
  <ScanProfile>{{"binning":2,"experimentDescription":{{"channels":[
    {{"name":"DAPI"}},{{"name":"ATTO550"}},{{"name":"CY5"}},{{"name":"AF750"}}]}}}}</ScanProfile>
</PerkinElmer-QPI-ImageDescription>
"""


# Acquisition order is NOT a repeating cycle of the four-filter set: index 2 is
# AF 750, not CY5. This is what makes an ``index % n_filters`` mapping unsafe.
_FUSION_FILTERS = ["DAPI", "ATTO 550", "AF 750", "Cy5", "ATTO 550"]
_FUSION_BIOMARKERS = [
    "DAPI",
    "CD38-Atto550",
    "CD107a-AF750",
    "CD4-AF647",
    "CI.PARP-Atto550",
]
FUSION_PAGED_XMLS = [
    _fusion_page_xml(bm, filt) for bm, filt in zip(_FUSION_BIOMARKERS, _FUSION_FILTERS)
]


def _strip_fluorophore_sources(xml: str, filter_name: str) -> str:
    """Remove both fluorophore sources (<Name> and the Responsivity echo)."""
    xml = xml.replace(f"<Name>{filter_name}</Name>", "<Name />", 1)
    return re.sub(r"<Responsivity>.*?</Responsivity>", "", xml, flags=re.S)


def _paged_polaris_xml(biomarker: str, filter_name: str, mode: str) -> str:
    """A page from a "paged Polaris" file.

    These carry per-channel metadata at the page root *and* a filter-cube
    <ScanBands-i> block in a shared XML ScanProfile, so both candidate sources
    are present at once.
    """
    return f"""\
<?xml version="1.0" encoding="utf-16"?>
<PerkinElmer-QPI-ImageDescription>
  <ImageType>FullResolution</ImageType>
  <SlideID>TestSlide_004</SlideID>
  <Name>{filter_name}</Name>
  <Biomarker>{biomarker}</Biomarker>
  <ScanProfile><root>
    <Mode>{mode}</Mode>
    <ScanBands>
      <ScanBands-i><Name>Cube_A</Name></ScanBands-i>
      <ScanBands-i><Name>Cube_B</Name></ScanBands-i>
      <ScanBands-i><Name>Cube_C</Name></ScanBands-i>
    </ScanBands>
  </root></ScanProfile>
</PerkinElmer-QPI-ImageDescription>
"""


class TestChannelLocus:
    """Dispatch keys off where channel metadata lives, not the vendor version."""

    def test_legacy_acquisition_format_unchanged(self) -> None:
        """Back-compat guard: these strings already reached written zarr stores
        and must keep their exact historical values."""
        assert (
            parse_qpi_xml(BRIGHTFIELD_XML, n_channels=3).acquisition_format
            == FORMAT_BRIGHTFIELD
        )
        assert (
            parse_qpi_xml(FLUORESCENCE_XML).acquisition_format
            == FORMAT_POLARIS_SCANBAND
        )
        assert (
            parse_qpi_xml(
                FUSION_PAGED_XMLS[0],
                n_channels=len(FUSION_PAGED_XMLS),
                per_page_xmls=FUSION_PAGED_XMLS,
            ).acquisition_format
            == FORMAT_FUSION_PAGED
        )

    def test_locus_of_each_fixture(self) -> None:
        assert (
            parse_qpi_xml(BRIGHTFIELD_XML, n_channels=3).channel_locus
            == LOCUS_RGB_SAMPLES
        )
        assert parse_qpi_xml(FLUORESCENCE_XML).channel_locus == LOCUS_SHARED_SCANBANDS
        assert (
            parse_qpi_xml(
                FUSION_PAGED_XMLS[0],
                n_channels=len(FUSION_PAGED_XMLS),
                per_page_xmls=FUSION_PAGED_XMLS,
            ).channel_locus
            == LOCUS_PER_PAGE_ROOT
        )

    def test_paged_polaris_splits_legacy_label_from_locus(self) -> None:
        """ScanBands-i present, but the stains are at the page roots: the legacy
        label stays polaris_scanband while the locus tells the truth."""
        pages = [
            _paged_polaris_xml(bm, f, "im_Fluorescence")
            for bm, f in [("CD3-Opal570", "OPAL 570"), ("CD8-Opal690", "OPAL 690")]
        ]
        meta = parse_qpi_xml(pages[0], n_channels=2, per_page_xmls=pages)
        assert meta.acquisition_format == FORMAT_POLARIS_SCANBAND
        assert meta.channel_locus == LOCUS_PER_PAGE_ROOT
        assert meta.channel_names == ["CD3-Opal570", "CD8-Opal690"]
        assert [ch.fluorophore for ch in meta.channels] == ["OPAL 570", "OPAL 690"]

    def test_identical_page_xmls_still_read_as_paged(self) -> None:
        """The old discriminator asked whether page XML strings differed, which
        said nothing when a scan's pages were legitimately identical. Position of
        the tag — root child vs nested in ScanBands-i — is the real signal."""
        page = _fusion_page_xml("CD3-Atto550", "ATTO 550")
        pages = [page, page]
        meta = parse_qpi_xml(page, n_channels=2, per_page_xmls=pages)
        assert meta.channel_locus == LOCUS_PER_PAGE_ROOT
        assert meta.channel_names == ["CD3-Atto550", "CD3-Atto550"]
        assert [ch.fluorophore for ch in meta.channels] == ["ATTO 550"] * 2

    def test_shared_scanband_dialect_keeps_name_as_biomarker(self) -> None:
        """In the scanband dialect <Name> means the stain, not the fluorophore —
        the opposite of the paged dialect. Guards against the fix for one
        leaking into the other."""
        xml = """\
        <PerkinElmer-QPI-ImageDescription>
          <ScanProfile><root><Mode>im_Fluorescence</Mode>
            <ScanBands>
              <ScanBands-i><Name>CD20</Name></ScanBands-i>
            </ScanBands>
          </root></ScanProfile>
        </PerkinElmer-QPI-ImageDescription>"""
        meta = parse_qpi_xml(xml)
        assert meta.channel_locus == LOCUS_SHARED_SCANBANDS
        assert meta.channel_names == ["CD20"]
        assert meta.channels[0].fluorophore is None

    def test_structure_signature_recorded(self) -> None:
        meta = parse_qpi_xml(
            FUSION_PAGED_XMLS[0],
            n_channels=len(FUSION_PAGED_XMLS),
            per_page_xmls=FUSION_PAGED_XMLS,
        )
        sig = meta.structure_signature
        assert sig["has_json_scanprofile"] is True
        assert sig["has_scanbands"] is False
        assert sig["pages_with_root_channel_tags"] == 5
        assert sig["n_pages"] == 5

    def test_unknown_structure_fails_open(self) -> None:
        """An unrecognised file must degrade, never raise."""
        xml = "<PerkinElmer-QPI-ImageDescription><ImageType>FullResolution</ImageType></PerkinElmer-QPI-ImageDescription>"  # noqa: E501
        meta = parse_qpi_xml(xml, n_channels=2)
        assert meta.channel_locus == LOCUS_UNKNOWN
        assert meta.channel_names == ["Channel_0", "Channel_1"]


# A classic Vectra/Polaris OPAL scan: ONE shared XML that every page repeats,
# with the per-channel metadata inside <ScanBands-i>. Note <Name> here is the
# biomarker and <Fluorophore> is explicit — the exact inverse of the paged
# dialect above, which is what makes this the cross-dialect regression guard.
SHARED_SCANBAND_XML = """\
<?xml version="1.0" encoding="utf-16"?>
<PerkinElmer-QPI-ImageDescription>
  <DescriptionVersion>2</DescriptionVersion>
  <AcquisitionSoftware>Vectra 3.0.3</AcquisitionSoftware>
  <ImageType>FullResolution</ImageType>
  <SlideID>TestSlide_OPAL</SlideID>
  <ExposureTimeArray>
    <Value>1000</Value>
    <Value>2000</Value>
    <Value>3000</Value>
  </ExposureTimeArray>
  <ScanProfile><root>
    <Mode>im_Fluorescence</Mode>
    <Name>OPAL 3-plex</Name>
    <OpalKitType>Opal7</OpalKitType>
    <ScanResolution>
      <PixelSizeMicrons>0.496</PixelSizeMicrons>
      <Magnification>20</Magnification>
      <ObjectiveName>20x</ObjectiveName>
    </ScanResolution>
    <ScanBands>
      <ScanBands-i>
        <Name>CD3</Name>
        <Fluorophore>OPAL570</Fluorophore>
        <HomeWavelength>570</HomeWavelength>
        <ExcitationWavelength>550</ExcitationWavelength>
        <Color>255,0,0</Color>
        <Gain>1.5</Gain>
        <Binning>2</Binning>
        <AutoExposeType>aet_Fluorescence</AutoExposeType>
      </ScanBands-i>
      <ScanBands-i>
        <Biomarker>CD8</Biomarker>
        <Name>ShouldNotBeTheFluorophore</Name>
        <Fluorophore>OPAL690</Fluorophore>
        <HomeWavelength>690</HomeWavelength>
        <Color>0,255,0</Color>
        <Gain>2.0</Gain>
        <Binning>2</Binning>
      </ScanBands-i>
      <ScanBands-i>
        <Name>DAPI</Name>
        <Fluorophore>DAPI</Fluorophore>
        <HomeWavelength>460</HomeWavelength>
        <Color>0,0,255</Color>
        <Gain>1.0</Gain>
        <Binning>2</Binning>
      </ScanBands-i>
    </ScanBands>
  </root></ScanProfile>
</PerkinElmer-QPI-ImageDescription>
"""


class TestSharedScanbandDialect:
    """Classic Vectra/Polaris OPAL: channel metadata lives in <ScanBands-i>.

    In this dialect <Fluorophore> is explicit, so <Name> means the biomarker —
    the opposite of the paged dialect. These tests exist so the paged fix can
    never leak across and start reading <Name> as a fluorophore here.
    """

    def _meta(self, n_pages: int = 3) -> object:
        # every page repeats the one shared XML, as these files really do
        pages = [SHARED_SCANBAND_XML] * n_pages
        return parse_qpi_xml(
            SHARED_SCANBAND_XML, n_channels=n_pages, per_page_xmls=pages
        )

    def test_locus_and_legacy_label(self) -> None:
        """Identical page XMLs carrying no root-level channel tags must resolve
        to the shared-scanbands locus, not the paged one."""
        meta = self._meta()
        assert meta.channel_locus == LOCUS_SHARED_SCANBANDS
        assert meta.acquisition_format == FORMAT_POLARIS_SCANBAND

    def test_name_is_the_biomarker_not_the_fluorophore(self) -> None:
        meta = self._meta()
        assert meta.channel_names == ["CD3", "CD8", "DAPI"]
        # <Biomarker> still outranks <Name> when both are present
        assert meta.channels[1].name == "CD8"

    def test_fluorophore_comes_only_from_the_explicit_tag(self) -> None:
        meta = self._meta()
        assert [ch.fluorophore for ch in meta.channels] == [
            "OPAL570",
            "OPAL690",
            "DAPI",
        ]
        # the paged rule must not leak in: <Name> is never a fluorophore here
        assert all(
            ch.fluorophore != "ShouldNotBeTheFluorophore" for ch in meta.channels
        )

    def test_wavelengths_from_home_wavelength(self) -> None:
        meta = self._meta()
        assert [ch.emission_wavelength_nm for ch in meta.channels] == [
            pytest.approx(570.0),
            pytest.approx(690.0),
            pytest.approx(460.0),
        ]
        assert meta.channels[0].excitation_wavelength_nm == pytest.approx(550.0)

    def test_exposure_times_fall_back_to_the_root_array(self) -> None:
        """<ScanBands-i> carries no <ExposureTime>; the positional
        <ExposureTimeArray> supplies it."""
        meta = self._meta()
        assert [ch.exposure_time_us for ch in meta.channels] == [
            pytest.approx(1000.0),
            pytest.approx(2000.0),
            pytest.approx(3000.0),
        ]

    def test_detector_fields_read_directly_off_the_band(self) -> None:
        """This layout has no <CameraSettings>: Gain/Binning hang off the band."""
        meta = self._meta()
        assert [ch.gain for ch in meta.channels] == [1.5, 2.0, 1.0]
        assert [ch.binning for ch in meta.channels] == [2, 2, 2]
        assert meta.channels[0].auto_expose_type == "aet_Fluorescence"
        assert meta.channels[0].color_rgb == (255, 0, 0)

    def test_opal_wavelength_inferred_when_home_wavelength_absent(self) -> None:
        xml = SHARED_SCANBAND_XML.replace(
            "<HomeWavelength>570</HomeWavelength>", ""
        )
        meta = parse_qpi_xml(xml)
        assert meta.channels[0].emission_wavelength_nm == pytest.approx(570.0)

    def test_slide_provenance_recorded(self) -> None:
        meta = self._meta()
        assert meta.acquisition_software == "Vectra 3.0.3"
        assert meta.description_version == "2"
        assert meta.pixel_size_um == pytest.approx(0.496)


class TestFusionPagedFluorophore:
    """Fusion 2.x paged QPTIFFs: fluorophore comes from each page's <Name>."""

    def test_fluorophore_from_page_name(self) -> None:
        meta = parse_qpi_xml(
            FUSION_PAGED_XMLS[0],
            n_channels=len(FUSION_PAGED_XMLS),
            per_page_xmls=FUSION_PAGED_XMLS,
        )
        assert meta.acquisition_format == FORMAT_FUSION_PAGED
        assert [ch.fluorophore for ch in meta.channels] == _FUSION_FILTERS

    def test_biomarker_still_wins_for_name(self) -> None:
        """<Name> feeds fluorophore only — the channel name stays the stain."""
        meta = parse_qpi_xml(
            FUSION_PAGED_XMLS[0],
            n_channels=len(FUSION_PAGED_XMLS),
            per_page_xmls=FUSION_PAGED_XMLS,
        )
        assert meta.channel_names == _FUSION_BIOMARKERS

    def test_responsivity_filter_name_fallback(self) -> None:
        """With <Name> empty, the Responsivity filter name still supplies it."""
        pages = [
            x.replace(f"<Name>{f}</Name>", "<Name />", 1)
            for x, f in zip(FUSION_PAGED_XMLS, _FUSION_FILTERS)
        ]
        meta = parse_qpi_xml(pages[0], n_channels=len(pages), per_page_xmls=pages)
        assert [ch.fluorophore for ch in meta.channels] == _FUSION_FILTERS

    def test_no_positional_guess_when_counts_mismatch(self) -> None:
        """5 pages against a 4-filter ScanProfile must not be cycled."""
        pages = [
            _strip_fluorophore_sources(x, f)
            for x, f in zip(FUSION_PAGED_XMLS, _FUSION_FILTERS)
        ]
        meta = parse_qpi_xml(pages[0], n_channels=len(pages), per_page_xmls=pages)
        assert [ch.fluorophore for ch in meta.channels] == [None] * 5

    def test_positional_fallback_when_counts_match(self) -> None:
        """A single-cycle scan (n_pages == n_filters) may map 1:1 in order."""
        pages = [
            _strip_fluorophore_sources(x, f)
            for x, f in zip(FUSION_PAGED_XMLS[:4], _FUSION_FILTERS[:4])
        ]
        meta = parse_qpi_xml(pages[0], n_channels=4, per_page_xmls=pages)
        assert [ch.fluorophore for ch in meta.channels] == [
            "DAPI",
            "ATTO550",
            "CY5",
            "AF750",
        ]


class TestParseQpiXml:
    def test_brightfield_top_level_fields(self) -> None:
        meta = parse_qpi_xml(BRIGHTFIELD_XML, n_channels=3)
        assert meta.description_version == "TestDescriptionVersion"
        assert meta.acquisition_software == "TestSoftware X.X.X"
        assert meta.image_type == "FullResolution"
        assert meta.identifier == "test-uuid-1234"
        assert meta.slide_id == "TestSlide_001"
        assert meta.barcode == "BC123"
        assert meta.study_name == "TestStudy"
        assert meta.operator_name == "Tester"
        assert meta.computer_name == "TEST-PC"
        assert meta.objective == "20x"
        assert meta.instrument_type == "TestInstrumentType"
        assert meta.bf_lamp_type == "TestBFLampType"

    def test_brightfield_scan_profile(self) -> None:
        meta = parse_qpi_xml(BRIGHTFIELD_XML, n_channels=3)
        assert meta.scan_profile_name == "TestBrightfieldName"
        assert meta.scan_mode == "im_Brightfield"
        assert meta.is_tma is True
        assert meta.opal_kit_type == "TestOpalKitType"

    def test_brightfield_pixel_size(self) -> None:
        meta = parse_qpi_xml(BRIGHTFIELD_XML, n_channels=3)
        assert meta.pixel_size_um == pytest.approx(0.25)
        assert meta.scan_resolution.magnification == pytest.approx(40.0)
        assert meta.scan_resolution.objective_name == "20x"
        assert meta.scan_resolution.binning == 1

    def test_brightfield_camera(self) -> None:
        meta = parse_qpi_xml(BRIGHTFIELD_XML, n_channels=3)
        assert meta.camera.camera_name == "TestCameraName"
        assert meta.camera.camera_type == "TestCameraType"
        assert meta.camera.gain == pytest.approx(1.0)
        assert meta.camera.bit_depth == 12
        assert meta.camera.binning == 1

    def test_brightfield_channels_rgb(self) -> None:
        meta = parse_qpi_xml(BRIGHTFIELD_XML, n_channels=3)
        assert meta.is_brightfield is True
        assert len(meta.channels) == 3
        assert meta.channel_names == ["Red", "Green", "Blue"]
        for ch in meta.channels:
            assert ch.is_brightfield is True

    def test_brightfield_to_dict_keys(self) -> None:
        meta = parse_qpi_xml(BRIGHTFIELD_XML, n_channels=3)
        d = ome_to_flat_attrs(ome_metadata_from_qptiff(meta))
        assert d["qpi_slide_id"] == "TestSlide_001"
        assert d["Pixels:PhysicalSizeX"] == pytest.approx(0.25)
        assert d["Channel:0:Name"] == "Red"
        assert d["qpi_is_brightfield"] == "True"

    def test_fluorescence_channels(self) -> None:
        meta = parse_qpi_xml(FLUORESCENCE_XML)
        assert meta.is_brightfield is False
        assert meta.channel_names == ["CD3", "CD8", "FOXP3"]

    def test_fluorescence_channel_details(self) -> None:
        meta = parse_qpi_xml(FLUORESCENCE_XML)
        cd3 = meta.channels[0]
        assert cd3.name == "CD3"
        assert cd3.fluorophore == "OPAL570"
        assert cd3.emission_wavelength_nm == pytest.approx(570.0)
        assert cd3.exposure_time_us == pytest.approx(100.0)

    def test_fluorescence_biomarker_tag_variants(self) -> None:
        meta = parse_qpi_xml(FLUORESCENCE_XML)
        assert "FOXP3" in meta.channel_names

    def test_fluorescence_pixel_size(self) -> None:
        meta = parse_qpi_xml(FLUORESCENCE_XML)
        assert meta.pixel_size_um == pytest.approx(0.5)
        assert meta.scan_resolution.magnification == pytest.approx(20.0)

    def test_empty_xml(self) -> None:
        meta = parse_qpi_xml("")
        assert meta.slide_id is None
        assert meta.pixel_size_um is None
        assert len(meta.channels) == 0

    def test_malformed_xml(self) -> None:
        meta = parse_qpi_xml("<not-closed>")
        assert meta.slide_id is None

    def test_datetime_passthrough(self) -> None:
        meta = parse_qpi_xml(BRIGHTFIELD_XML, datetime_str="YEAR:MM:DD HH:MM:SS")
        assert meta.datetime == "YEAR:MM:DD HH:MM:SS"

    def test_n_channels_fallback_generic(self) -> None:
        """When no ScanBands and mode is fluorescence, generate generic names."""
        xml = """\
        <PerkinElmer-QPI-ImageDescription>
          <ImageType>FullResolution</ImageType>
          <ScanProfile><root><Mode>im_Fluorescence</Mode></root></ScanProfile>
        </PerkinElmer-QPI-ImageDescription>"""
        meta = parse_qpi_xml(xml, n_channels=4)
        assert meta.channel_names == [
            "Channel_0",
            "Channel_1",
            "Channel_2",
            "Channel_3",
        ]

    def test_opal_wavelength_inferred_from_name(self) -> None:
        """Emission wavelength should be extracted from fluorophore name e.g. OPAL520."""  # noqa: E501
        xml = """\
        <PerkinElmer-QPI-ImageDescription>
          <ScanProfile><root><Mode>im_Fluorescence</Mode>
            <ScanBands>
              <ScanBands-i>
                <Biomarker>TestBiomarker</Biomarker>
                <Fluorophore>OPAL520</Fluorophore>
              </ScanBands-i>
            </ScanBands>
          </root></ScanProfile>
        </PerkinElmer-QPI-ImageDescription>"""
        meta = parse_qpi_xml(xml)
        assert meta.channels[0].emission_wavelength_nm == pytest.approx(520.0)


@pytest.mark.parametrize(
    "filename, "
    "set_scene, "
    "expected_scenes, "
    "expected_shape, "
    "expected_dtype, "
    "expected_dims_order, "
    "expected_channel_names, "
    "expected_physical_pixel_sizes",
    [
        (
            "s_1_t_1_c_1_z_1.ome.tiff",
            "Image:0",
            ("Image:0",),
            (325, 475),
            np.uint16,
            "YX",
            None,
            (None, 1.0833333604166673, 1.0833333604166673),
        ),
        (
            "s_1_t_1_c_1_z_1.tiff",
            "Image:0",
            ("Image:0",),
            (325, 475),
            np.uint16,
            "YX",
            None,
            (None, 1.08333441666775, 1.08333441666775),
        ),
        (
            "s_1_t_1_c_10_z_1.ome.tiff",
            "Image:0",
            ("Image:0",),
            (10, 1736, 1776),
            np.uint16,
            "CYX",
            [f"Channel:0:{i}" for i in range(10)],
            (None, 1, 1),
        ),
        (
            "s_1_t_10_c_3_z_1.tiff",
            "Image:0",
            ("Image:0",),
            (10, 3, 325, 475),
            np.uint16,
            "TCYX",
            ["Channel:0:0", "Channel:0:1", "Channel:0:2"],
            (None, 1.08333441666775, 1.08333441666775),
        ),
        (
            "s_3_t_1_c_3_z_5.ome.tiff",
            "Image:0",
            ("Image:0", "Image:1", "Image:2"),
            (5, 3, 325, 475),
            np.uint16,
            "ZCYX",
            ["Channel:0:0", "Channel:0:1", "Channel:0:2"],
            (None, 1.0833333604166673, 1.0833333604166673),
        ),
        (
            "s_3_t_1_c_3_z_5.ome.tiff",
            "Image:1",
            ("Image:0", "Image:1", "Image:2"),
            (5, 3, 325, 475),
            np.uint16,
            "ZCYX",
            ["Channel:1:0", "Channel:1:1", "Channel:1:2"],
            (None, 1.0833333604166673, 1.0833333604166673),
        ),
        (
            "s_3_t_1_c_3_z_5.ome.tiff",
            "Image:2",
            ("Image:0", "Image:1", "Image:2"),
            (5, 3, 325, 475),
            np.uint16,
            "ZCYX",
            ["Channel:2:0", "Channel:2:1", "Channel:2:2"],
            (None, 1.0833333604166673, 1.0833333604166673),
        ),
        (
            "s_1_t_1_c_1_z_1_RGB.tiff",
            "Image:0",
            ("Image:0",),
            (7548, 7548, 3),
            np.uint16,
            "YXS",  # S stands for samples dimension
            None,
            (None, None, None),
        ),
        (
            # Doesn't affect this test but this is actually an OME-TIFF file
            "s_1_t_1_c_2_z_1_RGB.tiff",
            "Image:0",
            ("Image:0",),
            (2, 32, 32, 3),
            np.uint8,
            "CYXS",  # S stands for samples dimension
            ["Channel:0:0", "Channel:0:1"],
            (None, 1, 1),
        ),
        pytest.param(
            "s_1_t_1_c_1_z_1.ome.tiff",
            "Image:1",
            None,
            None,
            None,
            None,
            None,
            (None, None, None),
            marks=pytest.mark.xfail(raises=IndexError),
        ),
        pytest.param(
            "s_3_t_1_c_3_z_5.ome.tiff",
            "Image:3",
            None,
            None,
            None,
            None,
            None,
            (None, None, None),
            marks=pytest.mark.xfail(raises=IndexError),
        ),
    ],
)
def test_tiff_reader(
    filename: str,
    set_scene: str,
    expected_scenes: Tuple[str, ...],
    expected_shape: Tuple[int, ...],
    expected_dtype: np.dtype,
    expected_dims_order: str,
    expected_channel_names: List[str],
    expected_physical_pixel_sizes: Tuple[
        Optional[float], Optional[float], Optional[float]
    ],
) -> None:
    # construct full filepath
    uri = LOCAL_RESOURCES_DIR / filename

    # run checks
    test_utilities.run_image_file_checks(
        ImageContainer=Reader,
        image=uri,
        set_scene=set_scene,
        expected_scenes=expected_scenes,
        expected_current_scene=set_scene,
        expected_shape=expected_shape,
        expected_dtype=expected_dtype,
        expected_dims_order=expected_dims_order,
        expected_channel_names=expected_channel_names,
        expected_physical_pixel_sizes=expected_physical_pixel_sizes,
        expected_metadata_type=str,
    )


def test_tiff_reader_with_non_tiff_file(sample_text_file: pathlib.Path) -> None:
    with pytest.raises(exceptions.UnsupportedFileFormatError):
        Reader(sample_text_file)


@pytest.mark.parametrize(
    "filename, first_scene_id, first_scene_shape, second_scene_id, second_scene_shape",
    [
        (
            "s_3_t_1_c_3_z_5.ome.tiff",
            "Image:0",
            (5, 3, 325, 475),
            "Image:1",
            (5, 3, 325, 475),
        ),
        (
            "s_3_t_1_c_3_z_5.ome.tiff",
            "Image:1",
            (5, 3, 325, 475),
            "Image:2",
            (5, 3, 325, 475),
        ),
    ],
)
def test_multi_scene_tiff_reader(
    filename: str,
    first_scene_id: str,
    first_scene_shape: Tuple[int, ...],
    second_scene_id: str,
    second_scene_shape: Tuple[int, ...],
) -> None:
    # construct full filepath
    uri = LOCAL_RESOURCES_DIR / filename

    test_utilities.run_multi_scene_image_read_checks(
        ImageContainer=Reader,
        image=uri,
        first_scene_id=first_scene_id,
        first_scene_shape=first_scene_shape,
        first_scene_dtype=np.dtype(np.uint16),
        second_scene_id=second_scene_id,
        second_scene_shape=second_scene_shape,
        second_scene_dtype=np.dtype(np.uint16),
    )


@pytest.mark.parametrize(
    "dims_from_meta, guessed_dims, expected",
    [
        ("QZYX", "CZYX", "CZYX"),
        ("ZQYX", "CZYX", "ZCYX"),
        ("ZYXC", "CZYX", "ZYXC"),
        ("TQQYX", "TCZYX", "TCZYX"),
        ("QTQYX", "TCZYX", "CTZYX"),
        # testing that nothing happens when Q not present
        ("LTCYX", "DIMOK", "LTCYX"),
    ],
)
def test_merge_dim_guesses(
    dims_from_meta: str, guessed_dims: str, expected: str
) -> None:
    assert Reader._merge_dim_guesses(dims_from_meta, guessed_dims) == expected


def test_micromanager_ome_tiff_binary_file() -> None:
    # Construct full filepath
    uri = (
        LOCAL_RESOURCES_DIR
        / "image_stack_tpzc_50tp_2p_5z_3c_512k_1_MMStack_2-Pos001_000.ome.tif"
    )

    # vven though the file name says it is an OME TIFF, this is
    # a binary TIFF file where the actual metadata for all scenes
    # lives in a different image file.
    # (image_stack_tpzc_50tp_2p_5z_3c_512k_1_MMStack_2-Pos000_000.ome.tif)
    # Because of this, we will read "non-main" micromanager files as just
    # normal TIFFs

    # run image read checks on the first scene
    test_utilities.run_image_file_checks(
        ImageContainer=Reader,
        image=uri,
        set_scene="Image:0",
        expected_scenes=("Image:0",),
        expected_current_scene="Image:0",
        expected_shape=(50, 5, 3, 256, 256),
        expected_dtype=np.dtype(np.uint16),
        expected_dims_order="TZCYX",
        expected_channel_names=["Channel:0:0", "Channel:0:1", "Channel:0:2"],
        expected_physical_pixel_sizes=(1.75, 0.0002, 0.0002),
        expected_metadata_type=str,
    )


@pytest.mark.parametrize(
    "filename, set_scene, get_dims, get_specific_dims, expected_shape",
    [
        (
            "s_1_t_1_c_2_z_1_RGB.tiff",
            "Image:0",
            "CYXS",
            {},
            (2, 32, 32, 3),
        ),
    ],
)
@pytest.mark.parametrize("chunk_dims", ["YX", "ZYX"])
@pytest.mark.parametrize("processes", [True, False])
def test_parallel_read(
    filename: str,
    set_scene: str,
    chunk_dims: str,
    processes: bool,
    get_dims: str,
    get_specific_dims: Dict[str, Union[int, slice, range, Tuple[int, ...], List[int]]],
    expected_shape: Tuple[int, ...],
) -> None:
    """
    This test ensures that our produced dask array can be read in parallel.
    """
    # construct full filepath
    uri = LOCAL_RESOURCES_DIR / filename

    img = Reader(uri, chunk_dims=chunk_dims)
    img.set_scene(set_scene)

    cluster = LocalCluster(processes=processes)
    client = Client(cluster)

    # Select data
    out = img.get_image_dask_data(get_dims, **get_specific_dims).compute()
    assert out.shape == expected_shape

    cluster.close()
    client.close()
    time.sleep(5)


@pytest.mark.parametrize(
    "filename, "
    "first_scene, "
    "expected_first_chunk_shape, "
    "second_scene, "
    "expected_second_chunk_shape",
    [
        (
            "image_stack_tpzc_50tp_2p_5z_3c_512k_1_MMStack_2-Pos000_000.ome.tif",
            0,
            (50, 5, 256, 256),
            1,
            (50, 5, 256, 256),
        ),
        (
            "image_stack_tpzc_50tp_2p_5z_3c_512k_1_MMStack_2-Pos000_000.ome.tif",
            1,
            (50, 5, 256, 256),
            0,
            (50, 5, 256, 256),
        ),
    ],
)
@pytest.mark.parametrize("processes", [True, False])
def test_parallel_multifile_tiff_read(
    filename: str,
    first_scene: int,
    expected_first_chunk_shape: Tuple[int, ...],
    second_scene: int,
    expected_second_chunk_shape: Tuple[int, ...],
    processes: bool,
) -> None:
    """
    This test ensures that we can serialize and read 'multi-file multi-scene' formats.
    See: https://github.com/AllenCellModeling/aicsimageio/issues/196

    We specifically test with a Distributed cluster to ensure that we serialize and
    read properly from each file.
    """
    uri = LOCAL_RESOURCES_DIR / filename

    img = Reader(uri)

    cluster = LocalCluster(processes=processes)
    client = Client(cluster)

    img.set_scene(first_scene)
    first_out = img.get_image_dask_data("TZYX").compute()
    assert first_out.shape == expected_first_chunk_shape

    img.set_scene(second_scene)
    second_out = img.get_image_dask_data("TZYX").compute()
    assert second_out.shape == expected_second_chunk_shape

    cluster.close()
    client.close()
    time.sleep(5)


@pytest.mark.parametrize(
    "filename, expected_shape",
    [
        ("s_1_t_10_c_3_z_1.tiff", (10, 3, 325, 475)),
    ],
)
def test_no_scene_prop_access(
    filename: str,
    expected_shape: Tuple[int, ...],
) -> None:
    # ccnstruct full filepath
    uri = LOCAL_RESOURCES_DIR / filename

    # construct image and check no scene call with property access
    img = Reader(uri)
    assert img.shape == expected_shape
