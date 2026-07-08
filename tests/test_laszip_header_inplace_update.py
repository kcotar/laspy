"""Regression tests for the laszip backend's in-place header update.

The laszip backend serialises the header + VLRs when the writer is constructed
(via the laszip C library), then has to fix up the EVLR offset/count and the
ExtraBytes min/max once the point stream has been written. It must do so by
patching the changed bytes *in place*: laszip's on-disk VLR bytes are not
guaranteed to round-trip byte-for-byte through laspy's typed VLR parsers, so
re-serialising the whole header could change ``offset_to_point_data`` and raise
``"writing header would change original offset to data"`` (or corrupt the file).
"""
import copy

import numpy as np
import pytest

import laspy
from laspy.lib import write_then_read_again
from laspy.vlrs.vlr import VLR

laszip_available = any(
    backend.name.lower() == "laszip"
    for backend in laspy.LazBackend.detect_available()
)
skip_if_no_laszip = pytest.mark.skipif(
    not laszip_available, reason="laszip backend not available"
)


def _non_round_tripping_vlr():
    # A LASF_Spec / record_id 0 record that laspy re-parses as a
    # ClassificationLookupVlr and re-serialises to a *different* length (the dict
    # de-duplicates the repeated class id). Models any vendor VLR -- such as one
    # copied from an original upload -- that does not round-trip byte-for-byte.
    return VLR(
        user_id="LASF_Spec",
        record_id=0,
        description="",
        record_data=(b"\x02" + b"ground".ljust(15, b"\x00")) * 2,
    )


def _make_las(with_evlr=True, with_offending_vlr=True):
    header = laspy.LasHeader(version="1.4", point_format=6)
    header.add_extra_dim(laspy.ExtraBytesParams(name="myfield", type=np.float32))
    if with_offending_vlr:
        header.vlrs.append(_non_round_tripping_vlr())
    las = laspy.LasData(header)
    n = 50
    las.x = np.linspace(0, 10, n)
    las.y = np.linspace(0, 5, n)
    las.z = np.linspace(0, 2, n)
    las.myfield = np.arange(n, dtype=np.float32)
    if with_evlr:
        las.evlrs = [
            VLR(user_id="myorg", record_id=1234, description="", record_data=b"\x01\x02")
        ]
    return las, n


@skip_if_no_laszip
def test_laszip_write_with_non_round_tripping_vlr_and_evlr():
    las, n = _make_las(with_evlr=True, with_offending_vlr=True)

    # Must not raise LaspyException("...change original offset to data...").
    out = write_then_read_again(las, do_compress=True, laz_backend=laspy.LazBackend.Laszip)

    assert out.header.point_count == n
    np.testing.assert_allclose(out.myfield, np.arange(n, dtype=np.float32))
    # The offending VLR and the EVLR both survived the write intact.
    assert any(v.user_id == "LASF_Spec" and v.record_id == 0 for v in out.header.vlrs)
    assert out.header.number_of_evlrs == 1


@skip_if_no_laszip
def test_laszip_extra_bytes_min_max_grown_with_non_round_tripping_vlr():
    # With a non-round-tripping VLR present, the ExtraBytes min/max must still be
    # patched in place to the real (grown) range -- not left at laszip's sentinel.
    las, n = _make_las(with_evlr=False, with_offending_vlr=True)

    out = write_then_read_again(las, do_compress=True, laz_backend=laspy.LazBackend.Laszip)

    eb = out.header.vlrs.get("ExtraBytesVlr")[0].extra_bytes_structs[0]
    assert eb.min[0] == 0.0
    assert eb.max[0] == float(n - 1)


@skip_if_no_laszip
def test_laszip_write_with_stale_extra_bytes_vlr_and_evlr():
    # The real-world trigger (flai PointCloudDownload): an extra dim is dropped via
    # remove_extra_dim, then an ExtraBytes VLR describing it is copied back in, so the
    # header carries an ExtraBytes VLR while point_size excludes those bytes. laspy
    # ignores such a VLR on write -> the on-disk VLR section is shorter than the
    # in-memory header, which used to make write_updated_header raise.
    donor = laspy.LasHeader(version="1.4", point_format=6)
    donor.add_extra_dim(laspy.ExtraBytesParams(name="point_order_id", type="u4"))
    stale_eb = copy.deepcopy(donor.vlrs.get("ExtraBytesVlr")[0])

    header = laspy.LasHeader(version="1.4", point_format=6)  # NO extra dim -> base point_size
    header.vlrs.append(stale_eb)  # stale: describes a dim the points do not have
    las = laspy.LasData(header)
    n = 50
    las.x = np.linspace(0, 10, n)
    las.y = np.linspace(0, 5, n)
    las.z = np.linspace(0, 2, n)
    las.evlrs = [VLR(user_id="o", record_id=1, description="", record_data=b"\x01\x02")]

    out = write_then_read_again(las, do_compress=True, laz_backend=laspy.LazBackend.Laszip)

    assert out.header.point_count == n
    # laspy drops the mismatched ExtraBytes VLR on write -> clean output
    assert not out.header.vlrs.get("ExtraBytesVlr")
    assert out.header.number_of_evlrs == 1


@skip_if_no_laszip
def test_laszip_clean_file_still_round_trips():
    # Control: a file without any awkward VLR keeps working through the same path.
    las, n = _make_las(with_evlr=True, with_offending_vlr=False)

    out = write_then_read_again(las, do_compress=True, laz_backend=laspy.LazBackend.Laszip)

    assert out.header.point_count == n
    np.testing.assert_allclose(out.myfield, np.arange(n, dtype=np.float32))
    eb = out.header.vlrs.get("ExtraBytesVlr")[0].extra_bytes_structs[0]
    assert eb.min[0] == 0.0 and eb.max[0] == float(n - 1)
