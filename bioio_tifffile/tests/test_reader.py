#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for bioio_tifffile Reader and metadata parser."""

import pathlib
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pytest
from bioio_base import exceptions, test_utilities
from distributed import Client, LocalCluster

from bioio_tifffile import Reader
from bioio_tifffile.qptiff_metadata import parse_qpi_xml

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
        # Brightfield RGB (YXS): pixel_size from QPI XML (0.25 µm)
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
        # Fluorescence CYX-written-as-QYX: tifffile guesses Q→Z
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
        d = meta.to_dict()
        assert d["qpi_slide_id"] == "TestSlide_001"
        assert d["Pixels:PhysicalSizeX"] == pytest.approx(0.25)
        assert d["Channel:0:Name"] == "Red"
        assert d["qpi_is_brightfield"] is True

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
    # Construct full filepath
    uri = LOCAL_RESOURCES_DIR / filename

    # Run checks
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
    "filename, "
    "first_scene_id, "
    "first_scene_shape, "
    "second_scene_id, "
    "second_scene_shape",
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
    # Construct full filepath
    uri = LOCAL_RESOURCES_DIR / filename

    # Run checks
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
        # testing that nothing happens when Q isn't present
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

    # Even though the file name says it is an OME TIFF, this is
    # a binary TIFF file where the actual metadata for all scenes
    # lives in a different image file.
    # (image_stack_tpzc_50tp_2p_5z_3c_512k_1_MMStack_2-Pos000_000.ome.tif)
    # Because of this, we will read "non-main" micromanager files as just
    # normal TIFFs

    # Run image read checks on the first scene
    # (this files binary data)
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
    # Construct full filepath
    uri = LOCAL_RESOURCES_DIR / filename

    # Init image
    img = Reader(uri, chunk_dims=chunk_dims)
    img.set_scene(set_scene)

    # Init cluster
    cluster = LocalCluster(processes=processes)
    client = Client(cluster)

    # Select data
    out = img.get_image_dask_data(get_dims, **get_specific_dims).compute()
    assert out.shape == expected_shape

    # Shutdown and then safety measure timeout
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
    # Construct full filepath
    uri = LOCAL_RESOURCES_DIR / filename

    # Init image
    img = Reader(uri)

    # Init cluster
    cluster = LocalCluster(processes=processes)
    client = Client(cluster)

    # Select data
    img.set_scene(first_scene)
    first_out = img.get_image_dask_data("TZYX").compute()
    assert first_out.shape == expected_first_chunk_shape

    # Update scene and select data
    img.set_scene(second_scene)
    second_out = img.get_image_dask_data("TZYX").compute()
    assert second_out.shape == expected_second_chunk_shape

    # Shutdown and then safety measure timeout
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
    # Construct full filepath
    uri = LOCAL_RESOURCES_DIR / filename

    # Construct image and check no scene call with property access
    img = Reader(uri)
    assert img.shape == expected_shape
