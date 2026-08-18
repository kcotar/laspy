"""
Tests related to extra bytes
"""

import numpy as np
import pytest

import laspy
from laspy.lib import write_then_read_again
from tests.conftest import (
    EXTRA_BYTES_LAS_FILE_PATH,
    SIMPLE_LAS_FILE_PATH,
    UNREGISTERED_EXTRA_BYTES_LAS,
)


def test_read_example_extra_bytes_las(las_file_path_with_extra_bytes):
    """
    Test that we can read the files with extra bytes with have as examples
    """
    las = laspy.read(las_file_path_with_extra_bytes)
    expected_names = [
        "Colors",
        "Reserved",
        "Flags",
        "Intensity",
        "Time",
    ]
    assert expected_names == list(las.point_format.extra_dimension_names)


def test_read_write_example_extra_bytes_file(las_file_path_with_extra_bytes):
    """
    Test that we can write extra bytes without problem
    """
    original = laspy.read(las_file_path_with_extra_bytes)
    las = write_then_read_again(original)

    for name in original.point_format.dimension_names:
        assert np.allclose(las[name], original[name])


def test_adding_extra_bytes_keeps_values_of_all_existing_fields(
    extra_bytes_params, simple_las_path
):
    """
    Test that when extra bytes are added, the existing fields keep their
    values and then we don't somehow drop them
    """
    las = laspy.read(simple_las_path)
    las.add_extra_dim(extra_bytes_params)

    original = laspy.read(simple_las_path)

    for name in original.point_format.dimension_names:
        assert np.allclose(las[name], original[name])


def test_creating_extra_bytes(extra_bytes_params, simple_las_path):
    """
    Test that we can create extra byte dimensions for each
    data type. And they can be written then read.
    """
    las = laspy.read(simple_las_path)
    las.add_extra_dim(extra_bytes_params)

    assert np.allclose(las[extra_bytes_params.name], 0)

    las[extra_bytes_params.name][:] = 42
    assert np.allclose(las[extra_bytes_params.name], 42)

    las = write_then_read_again(las)
    assert np.allclose(las[extra_bytes_params.name], 42)


def test_creating_scaled_extra_bytes(extra_bytes_params, simple_las_path):
    las = laspy.read(simple_las_path)

    if extra_bytes_params.type.ndim == 1:
        num_elements = extra_bytes_params.type.shape[0]
    else:
        num_elements = 1

    params = laspy.ExtraBytesParams(
        extra_bytes_params.name,
        extra_bytes_params.type,
        offsets=np.array([2.0] * num_elements),
        scales=np.array([1.0] * num_elements),
    )
    las.add_extra_dim(params)

    assert np.allclose(las[extra_bytes_params.name], 2.0)

    las[params.name][:] = 42.0
    assert np.allclose(las[extra_bytes_params.name], 42.0)

    las = write_then_read_again(las)
    assert np.allclose(las[extra_bytes_params.name], 42.0)


def test_scaled_extra_byte_array_type(simple_las_path):
    """
    To make sure we handle scaled extra bytes
    """
    las = laspy.read(simple_las_path)

    las.add_extra_dim(
        laspy.ExtraBytesParams(
            name="test_dim",
            type="3int32",
            scales=np.array([1.0, 2.0, 3.0], np.float64),
            offsets=np.array([10.0, 20.0, 30.0], np.float64),
        )
    )

    assert np.allclose(las.test_dim[..., 0], 10.0)
    assert np.allclose(las.test_dim[..., 1], 20.0)
    assert np.allclose(las.test_dim[..., 2], 30.0)

    las.test_dim[..., 0][:] = 42.0
    las.test_dim[..., 1][:] = 82.0
    las.test_dim[..., 2][:] = 123.0

    assert np.allclose(las.test_dim[..., 0], 42.0)
    assert np.allclose(las.test_dim[..., 1], 82.0)
    assert np.allclose(las.test_dim[..., 2], 123.0)

    las = write_then_read_again(las)
    assert np.allclose(las.test_dim[..., 0], 42.0)
    assert np.allclose(las.test_dim[..., 1], 82.0)
    assert np.allclose(las.test_dim[..., 2], 123.0)


def test_scaled_extra_byte_min_max(simple_las_path):
    """
    To make sure we handle scaled extra bytes
    """
    MIN = -10
    MAX = 1000
    NODATA = -10000
    SCALE = np.array([1.0, 2.0, 3.0])
    OFFSET = np.array([10.0, 20.0, 30.0])
    las = laspy.read(simple_las_path)

    las.add_extra_dim(
        laspy.ExtraBytesParams(
            name="test_dim",
            type="3int32",
            scales=np.array(SCALE, np.float64),
            offsets=np.array(OFFSET, np.float64),
        )
    )

    assert np.allclose(las.test_dim[..., 0], 10.0)
    assert np.allclose(las.test_dim[..., 1], 20.0)
    assert np.allclose(las.test_dim[..., 2], 30.0)

    las.test_dim[..., 0][:] = 42.0
    las.test_dim[..., 1][:] = 82.0
    las.test_dim[..., 2][:] = 123.0

    las.test_dim[0, 0] = MIN
    las.test_dim[0, 1] = MIN
    las.test_dim[0, 2] = MIN

    las.test_dim[1, 0] = MAX
    las.test_dim[1, 1] = MAX
    las.test_dim[1, 2] = MAX

    las.test_dim[2, 0] = NODATA
    las.test_dim[2, 1] = NODATA
    las.test_dim[2, 2] = NODATA

    las.header.vlrs[0].extra_bytes_structs[0].no_data = [NODATA, NODATA, NODATA]

    las.update_header()

    assert las.header.vlrs[0].extra_bytes_structs[0].data_type == 26  # 3*int32
    assert len(las.header.vlrs[0].extra_bytes_structs[0].min) == 3
    assert las.header.vlrs[0].extra_bytes_structs[0].min.dtype == np.float64

    ebs = las.header.vlrs[0].extra_bytes_structs[0]
    assert np.allclose(ebs.min[:], MIN, atol=1)
    assert np.allclose(ebs.max[:], MAX, atol=1)

    las = write_then_read_again(las)
    assert np.allclose(las.test_dim[..., 0][3:], 42.0)
    assert np.allclose(las.test_dim[..., 1][3:], 82.0)
    assert np.allclose(las.test_dim[..., 2][3:], 123.0)

    ebs = las.header.vlrs[0].extra_bytes_structs[0]
    assert np.allclose(ebs.min[:], MIN, atol=1)
    assert np.allclose(ebs.max[:], MAX, atol=1)


def test_extra_bytes_description_is_ok(extra_bytes_params, simple_las_path):
    """
    Test that the description in ok
    """
    las = laspy.read(simple_las_path)
    las.add_extra_dim(extra_bytes_params)

    extra_dim_info = list(las.point_format.extra_dimensions)
    assert len(extra_dim_info) == 1
    assert extra_dim_info[0].description == extra_bytes_params.description

    las = write_then_read_again(las)

    extra_dim_info = list(las.point_format.extra_dimensions)
    assert len(extra_dim_info) == 1
    assert extra_dim_info[0].description == extra_bytes_params.description


def test_extra_bytes_with_spaces_in_name(simple_las_path):
    """
    Test that we can create extra bytes with spaces in their name
    and that they can be accessed using __getitem__ ( [] )
    as de normal '.name' won't work
    """
    las = laspy.read(simple_las_path)
    las.add_extra_dim(laspy.ExtraBytesParams(name="Name With Spaces", type="int32"))

    assert np.all(las["Name With Spaces"] == 0)
    las["Name With Spaces"][:] = 789_464

    las = write_then_read_again(las)
    np.all(las["Name With Spaces"] == 789_464)


def test_conversion_keeps_eb(las_file_path_with_extra_bytes):
    """
    Test that converting point format does not lose extra bytes
    """
    original = laspy.read(las_file_path_with_extra_bytes)
    converted_las = laspy.convert(original, point_format_id=0)

    assert len(list(original.point_format.extra_dimension_names)) == 5
    assert list(converted_las.point_format.extra_dimension_names) == list(
        original.point_format.extra_dimension_names
    )
    for name in converted_las.point_format.extra_dimension_names:
        assert np.allclose(converted_las[name], original[name])

    converted_las = laspy.lib.write_then_read_again(converted_las)
    assert list(converted_las.point_format.extra_dimension_names) == list(
        original.point_format.extra_dimension_names
    )
    for name in converted_las.point_format.extra_dimension_names:
        assert np.allclose(converted_las[name], original[name])


def test_creating_bytes_with_name_too_long(simple_las_path):
    """
    Test error thrown when creating extra bytes with a name that is too long
    """
    las = laspy.read(simple_las_path)
    with pytest.raises(ValueError) as error:
        las.add_extra_dim(
            laspy.ExtraBytesParams(
                name="Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed non risus",
                type="int32",
            )
        )

    assert str(error.value) == "bytes too long (70, maximum length 32)"


def test_creating_bytes_with_description_too_long(simple_las_path):
    """
    Test error thrown when creating extra bytes with a name that is too long
    """
    las = laspy.read(simple_las_path)
    with pytest.raises(ValueError) as error:
        las.add_extra_dim(
            laspy.ExtraBytesParams(
                name="a fine name",
                type="int32",
                description="Lorem ipsum dolor sit amet, consectetur adipiscing elit."
                " Sed non risus",
            )
        )

    assert str(error.value) == "bytes too long (70, maximum length 32)"


def test_creating_extra_byte_with_invalid_type(simple_las_path):
    """
    Test the error message when creating extra bytes with invalid type
    """
    las = laspy.read(simple_las_path)
    with pytest.raises(TypeError):
        las.add_extra_dim(laspy.ExtraBytesParams("just_a_test", "i16"))


def test_cant_create_scaled_extra_bytes_without_both_offsets_and_scales():
    las = laspy.create()
    with pytest.raises(ValueError):
        las.add_extra_dim(
            laspy.ExtraBytesParams("must fail", "int64", scales=np.array([0.1]))
        )

    with pytest.raises(ValueError):
        las.add_extra_dim(
            laspy.ExtraBytesParams("must fail", "int64", offsets=np.array([0.1]))
        )


@pytest.mark.parametrize("num_elements", [1, 2, 3])
def test_cant_create_scaled_extra_bytes_with_offsets_array_smaller(num_elements):
    las = laspy.create()
    with pytest.raises(ValueError) as error:
        las.add_extra_dim(
            laspy.ExtraBytesParams(
                "must fail",
                f"{num_elements}int64",
                scales=np.array([0.1] * num_elements),
                offsets=np.array([0.0] * (num_elements - 1)),
            )
        )
    assert (
        str(error.value)
        == f"len(offsets) ({num_elements - 1}) is not the same as the number of elements ({num_elements})"
    )


@pytest.mark.parametrize("num_elements", [1, 2, 3])
def test_cant_create_scaled_extra_bytes_with_scales_array_smaller(num_elements):
    las = laspy.create()
    with pytest.raises(ValueError) as error:
        las.add_extra_dim(
            laspy.ExtraBytesParams(
                "must fail",
                f"{num_elements}int64",
                scales=np.array([0.1] * (num_elements - 1)),
                offsets=np.array([0.0] * num_elements),
            )
        )
    assert (
        str(error.value)
        == f"len(scales) ({num_elements - 1}) is not the same as the number of elements ({num_elements})"
    )


def test_handle_unregistered_extra_bytes():
    # Test that if the point size in the header is bigger
    # than the expected point size of the point format
    # and no extra bytes vlr is present, we can still read and
    # write the file

    def check_file(las):
        assert las.point_format.id == 6
        assert las.point_format.size == 34
        assert las.point_format.num_extra_bytes == 4
        assert np.all(las.x == np.array([1, 2, 3, 4]))
        assert np.all(las.y == np.array([1, 2, 3, 4]))
        assert np.all(las.z == np.array([1, 2, 3, 4]))
        assert list(las.point_format.extra_dimension_names) == ["ExtraBytes"]
        assert las.vlrs == []

    las = laspy.read(UNREGISTERED_EXTRA_BYTES_LAS)
    check_file(las)

    las = laspy.lib.write_then_read_again(las)
    check_file(las)


@pytest.mark.parametrize("num_extra_bytes", [2, 3, 4, 5])
def test_unregistered_extra_bytes_use_data_type_zero(num_extra_bytes):
    # When unregistered extra bytes get materialized into an ExtraBytesVlr
    # (e.g. via laspy.convert), the resulting struct must use data_type=0
    # with options=size_in_bytes. The deprecated array data_types 11-30
    # (e.g. 11 = uint8[2]) are rejected by strict readers like copc.js.
    from laspy.point import dims
    from laspy.vlrs.known import ExtraBytesVlr

    header = laspy.LasHeader(point_format=6, version="1.4")
    # Simulate the read-time auto-generation of "Un-registered ExtraBytes"
    header.point_format.dimensions.append(
        dims.DimensionInfo(
            name="ExtraBytes",
            kind=dims.DimensionKind.UnsignedInteger,
            num_bits=8 * num_extra_bytes,
            num_elements=num_extra_bytes,
            is_standard=False,
            description="Un-registered ExtraBytes",
        )
    )
    header._sync_extra_bytes_vlr()

    eb_vlrs = [v for v in header.vlrs if isinstance(v, ExtraBytesVlr)]
    assert len(eb_vlrs) == 1
    structs = eb_vlrs[0].extra_bytes_structs
    assert len(structs) == 1
    assert structs[0].data_type == 0, (
        f"data_type must be 0 (raw bytes) for {num_extra_bytes}-byte unregistered "
        f"extras; got {structs[0].data_type} (deprecated array type)"
    )
    assert structs[0].options == num_extra_bytes


def test_remove_standard_dimension_fails():
    """
    Test we cannot remove non-extra dimension
    """
    las = laspy.read(SIMPLE_LAS_FILE_PATH)
    for name in las.point_format.standard_dimension_names:
        with pytest.raises(laspy.LaspyException):
            las.remove_extra_dims(name)


def test_remove_all_extra_dimensions():
    """
    Test that if we remove all extra bytes of a file,
    the values of standard fields are not altered

    This also test that writing files from which we deleted extra bytes works
    """
    las = laspy.read(EXTRA_BYTES_LAS_FILE_PATH)

    extra_dimension_names = list(las.point_format.extra_dimension_names)
    assert len(extra_dimension_names) > 0, (
        "If the input file has no extra dimension, " "this test is useless"
    )

    copied_standard_values = [
        (name, np.copy(las[name])) for name in las.point_format.standard_dimension_names
    ]

    las.remove_extra_dims(extra_dimension_names)

    for name, copied_value in copied_standard_values:
        assert np.all(las[name] == copied_value)

    new_las = laspy.lib.write_then_read_again(las)
    assert new_las.point_format == las.point_format
    for name, copied_value in copied_standard_values:
        assert np.all(new_las[name] == copied_value)


def test_remove_some_extra_dimensions():
    """
    Test that if we remove some extra bytes of a file,
    the values of standard fields as well as kept extra bytes are not altered

    This also test that writing files from which we deleted extra bytes works
    """
    las = laspy.read(EXTRA_BYTES_LAS_FILE_PATH)

    extra_dimension_names = list(las.point_format.extra_dimension_names)
    assert len(extra_dimension_names) > 0, (
        "If the input file has no extra dimension, " "this test is useless"
    )

    extra_dimensions_to_keep = ["Colors", "Time"]
    dims_to_copy = (
        list(las.point_format.standard_dimension_names) + extra_dimensions_to_keep
    )
    copied_standard_values = [(name, np.copy(las[name])) for name in dims_to_copy]

    extra_dims_to_remove = [
        name
        for name in las.point_format.extra_dimension_names
        if name not in extra_dimensions_to_keep
    ]

    las.remove_extra_dims(extra_dims_to_remove)

    for name, copied_value in copied_standard_values:
        assert np.all(las[name] == copied_value)

    new_las = laspy.lib.write_then_read_again(las)
    assert new_las.point_format == las.point_format
    for name, copied_value in copied_standard_values:
        assert np.all(new_las[name] == copied_value)


def _declared_extra_byte_dims_on_disk(path):
    """Count ExtraBytes descriptors in the file's VLR block, straight off disk.

    laspy discards an ExtraBytes record that does not fit the point record when
    reading, so going through laspy cannot observe the corruption under test.
    """
    import struct

    with open(path, "rb") as fh:
        header = fh.read(375)
        header_size = struct.unpack_from("<H", header, 94)[0]
        num_vlrs = struct.unpack_from("<I", header, 100)[0]
        point_record_length = struct.unpack_from("<H", header, 105)[0]
        fh.seek(header_size)
        declared = 0
        for _ in range(num_vlrs):
            vlr_header = fh.read(54)
            if len(vlr_header) < 54:
                break
            record_length = struct.unpack_from("<H", vlr_header, 20)[0]
            user_id = vlr_header[2:18].split(b"\x00", 1)[0].decode("ascii", "replace")
            record_id = struct.unpack_from("<H", vlr_header, 18)[0]
            if (user_id, record_id) == ("LASF_Spec", 4):
                declared += record_length // 192
            fh.seek(record_length, 1)
    return declared, point_record_length


def test_writer_drops_extra_bytes_vlr_left_over_from_another_file(tmp_path, keep_a_second_dim=False):
    """An ExtraBytes record must never outlive the dimensions it describes.

    Copying VLRs across from a source file after dropping an extra dimension used
    to leave the record behind, producing a point record that advertises extra
    bytes it has no room for. PDAL and QGIS reject such a file with "Extra byte
    specification exceeds point length beyond base format length".
    """
    from copy import deepcopy

    header = laspy.LasHeader(version="1.4", point_format=6)
    header.add_extra_dim(laspy.ExtraBytesParams(name="doomed", type="u4"))
    if keep_a_second_dim:
        header.add_extra_dim(laspy.ExtraBytesParams(name="keep_me", type="f4"))
    source = laspy.LasData(header)
    source.x = np.arange(10, dtype=np.float64)
    source.y = np.zeros(10)
    source.z = np.zeros(10)
    source.doomed = np.arange(10, dtype=np.uint32)
    if keep_a_second_dim:
        source.keep_me = np.arange(10, dtype=np.float32) / 10.0
    source_path = str(tmp_path / "source.laz")
    source.write(source_path)

    las = laspy.read(source_path)
    las.remove_extra_dim("doomed")

    # carry VLRs over from the source, as metadata-preserving pipelines do
    donor = laspy.read(source_path)
    existing = {(v.user_id, v.record_id) for v in las.vlrs}
    for vlr in donor.vlrs:
        if (vlr.user_id, vlr.record_id) not in existing:
            las.vlrs.append(deepcopy(vlr))

    out_path = str(tmp_path / "out.laz")
    las.write(out_path)

    expected_dims = 1 if keep_a_second_dim else 0
    declared, point_record_length = _declared_extra_byte_dims_on_disk(out_path)
    assert declared == expected_dims
    assert point_record_length == laspy.PointFormat(6).size + (4 if keep_a_second_dim else 0)

    read_back = laspy.read(out_path)
    assert list(read_back.point_format.extra_dimension_names) == (
        ["keep_me"] if keep_a_second_dim else []
    )
    if keep_a_second_dim:
        assert np.allclose(read_back.keep_me, np.arange(10, dtype=np.float32) / 10.0)


def test_writer_leaves_a_consistent_extra_bytes_vlr_untouched(tmp_path):
    """The guard must not rebuild a record that already matches the point format.

    Rebuilding discards min/max values grown from the point data, so a file whose
    record is already correct has to round-trip byte-for-byte.
    """
    header = laspy.LasHeader(version="1.4", point_format=6)
    header.add_extra_dim(laspy.ExtraBytesParams(name="intensity_pct", type="u2"))
    las = laspy.LasData(header)
    las.x = np.arange(10, dtype=np.float64)
    las.y = np.zeros(10)
    las.z = np.zeros(10)
    las.intensity_pct = np.arange(10, dtype=np.uint16) * 7
    las.update_header()

    before = las.header.vlrs.get("ExtraBytesVlr")[0].record_data_bytes()
    out_path = str(tmp_path / "consistent.laz")
    las.write(out_path)

    after = laspy.read(out_path).header.vlrs.get("ExtraBytesVlr")[0].record_data_bytes()
    assert after == before


def test_writer_rebuilds_an_extra_bytes_vlr_that_over_declares(tmp_path):
    """A record describing more dims than the point format is rebuilt, not shipped.

    This is the shape produced by header-promotion helpers: a fresh header is
    built with a narrower point format and the source's VLRs are appended onto
    it wholesale, so the record survives describing dims that are no longer there.
    """
    donor_header = laspy.LasHeader(version="1.4", point_format=6)
    donor_header.add_extra_dim(laspy.ExtraBytesParams(name="gone", type="u4"))
    donor_header.add_extra_dim(laspy.ExtraBytesParams(name="keep_me", type="f4"))
    donor_eb_vlr = donor_header.vlrs.get("ExtraBytesVlr")[0]
    assert len(donor_eb_vlr.extra_bytes_structs) == 2

    header = laspy.LasHeader(version="1.4", point_format=6)
    header.add_extra_dim(laspy.ExtraBytesParams(name="keep_me", type="f4"))
    # in-place append bypasses the resync every other mutation path triggers
    header.vlrs.append(donor_eb_vlr)
    assert header.point_format.num_extra_bytes == 4

    las = laspy.LasData(header)
    las.x = np.arange(10, dtype=np.float64)
    las.y = np.zeros(10)
    las.z = np.zeros(10)
    las.keep_me = np.arange(10, dtype=np.float32) / 10.0

    out_path = str(tmp_path / "over_declared.laz")
    las.write(out_path)

    declared, point_record_length = _declared_extra_byte_dims_on_disk(out_path)
    assert declared == 1, "record still describes a dimension the point record lacks"
    assert point_record_length == laspy.PointFormat(6).size + 4

    read_back = laspy.read(out_path)
    assert list(read_back.point_format.extra_dimension_names) == ["keep_me"]
    assert np.allclose(read_back.keep_me, np.arange(10, dtype=np.float32) / 10.0)
