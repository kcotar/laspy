"""Regression test: CopcWriter must handle zero-point inputs.

Writing an empty LasData to COPC used to crash in _build_octree with
"zero-size array to reduction operation minimum which has no identity".
"""
import io

import numpy as np
import pytest

import laspy
from laspy import Bounds, CopcReader
from laspy.copcwriter import CopcWriter


@pytest.mark.skipif("not laspy.LazBackend.Lazrs.is_available()")
def test_copc_write_zero_points():
    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = [0.001, 0.001, 0.001]
    header.offsets = [500_000.0, 200_000.0, 0.0]
    header.add_extra_dim(laspy.ExtraBytesParams(name="confidence", type=np.float32))
    las = laspy.LasData(header=header)

    buf = io.BytesIO()
    CopcWriter.write(buf, las.header, las.points)
    data = buf.getvalue()

    # Standard reader
    read_back = laspy.read(io.BytesIO(data))
    assert len(read_back.points) == 0
    assert "confidence" in read_back.point_format.dimension_names

    # COPC reader: root hierarchy entry must exist, queries must return empty
    with CopcReader.open(io.BytesIO(data)) as copc:
        assert len(copc.root_page.entries) == 1
        root_entry = next(iter(copc.root_page.entries.values()))
        assert root_entry.point_count == 0
        assert len(copc.query()) == 0
        query_bounds = Bounds(np.array([0.0, 0.0]), np.array([1e9, 1e9]))
        assert len(copc.query(query_bounds)) == 0


@pytest.mark.skipif("not laspy.LazBackend.Lazrs.is_available()")
def test_copc_write_zero_points_without_gps_time():
    # Point formats 6+ always have gps_time; the empty-input path must not
    # try to reduce the empty gps_time array either.
    header = laspy.LasHeader(point_format=6, version="1.4")
    las = laspy.LasData(header=header)

    buf = io.BytesIO()
    CopcWriter.write(buf, las.header, las.points)

    read_back = laspy.read(io.BytesIO(buf.getvalue()))
    assert len(read_back.points) == 0
