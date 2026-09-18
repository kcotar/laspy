import numpy as np
import pytest

import laspy
from laspy.lib import write_then_read_again


@pytest.mark.parametrize("target_point_format_id", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
def test_point_format_conversion_copies_field_values(file_path, target_point_format_id):
    original = laspy.read(file_path)
    converted = laspy.convert(original, point_format_id=target_point_format_id)
    converted = write_then_read_again(converted)

    converted_dimension_names = set(converted.point_format.dimension_names)
    dimension_expected_to_be_kept = [
        dim_name
        for dim_name in original.point_format.dimension_names
        if dim_name in converted_dimension_names
    ]

    for dim_name in dimension_expected_to_be_kept:
        assert np.allclose(
            converted[dim_name], original[dim_name]
        ), "{} not equal".format(dim_name)


def _waveform_source_las():
    """A point format 10 (RGB + NIR + wave packets) file advertising waveforms."""
    header = laspy.LasHeader(point_format=10, version="1.4")

    descriptor = laspy.vlrs.known.WaveformPacketVlr(record_id=100)
    descriptor.parse_record_data(bytes(laspy.vlrs.known.WaveformPacketStruct()))
    header.vlrs.append(descriptor)

    las = laspy.LasData(header)
    las.points = laspy.PackedPointRecord.zeros(5, header.point_format)
    las.red[:] = np.arange(5)
    las.nir[:] = np.arange(5)
    # after the points assignment: its update_header() zeroes the waveform offset
    las.header.global_encoding.waveform_data_packets_external = True
    las.header.start_of_waveform_data_packet_record = 1234
    las.evlrs = laspy.vlrs.vlrlist.VLRList(
        [
            laspy.vlrs.vlr.VLR(
                user_id="LASF_Spec", record_id=65535, record_data=b"\x00" * 8
            )
        ]
    )
    return las


@pytest.mark.parametrize(
    "target_point_format_id, expected_kept",
    [(8, ("red", "nir")), (7, ("red",)), (6, ())],
)
def test_conversion_out_of_waveform_format_drops_waveform_records(
    target_point_format_id, expected_kept
):
    """Dropping the wave packet dimensions must drop everything referencing them.

    Waveform descriptor VLRs, the waveform data EVLR and the two header fields
    pointing at the records are only legal for point formats 4, 5, 9 and 10.
    """
    original = _waveform_source_las()
    converted = laspy.convert(original, point_format_id=target_point_format_id)

    assert not any(
        vlr.user_id == "LASF_Spec" and 100 <= vlr.record_id <= 355
        for vlr in converted.header.vlrs
    )
    assert not any(
        evlr.user_id == "LASF_Spec" and evlr.record_id == 65535
        for evlr in converted.evlrs
    )
    assert converted.header.global_encoding.waveform_data_packets_internal is False
    assert converted.header.global_encoding.waveform_data_packets_external is False
    assert converted.header.start_of_waveform_data_packet_record == 0

    for dim_name in expected_kept:
        assert np.array_equal(converted[dim_name], original[dim_name])


def test_conversion_between_waveform_formats_keeps_waveform_records():
    """9 <-> 10 both keep the wave packet dimensions, so the records stay valid."""
    original = _waveform_source_las()
    converted = laspy.convert(original, point_format_id=9)

    assert any(
        vlr.user_id == "LASF_Spec" and vlr.record_id == 100
        for vlr in converted.header.vlrs
    )
    assert converted.header.global_encoding.waveform_data_packets_external is True
    assert converted.header.start_of_waveform_data_packet_record == 1234
