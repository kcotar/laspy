import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import laspy

SIMPLE_COPC_FILE = Path(__file__).parent / "data" / "simple.copc.laz"

try:
    import requests
except ModuleNotFoundError:
    requests = None

try:
    import RangeHTTPServer
except ModuleNotFoundError:
    RangeHTTPServer = None


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(
            laspy.LazBackend.Laszip,
            marks=pytest.mark.skipif("not laspy.LazBackend.Laszip.is_available()"),
        ),
        pytest.param(
            laspy.LazBackend.Lazrs,
            marks=pytest.mark.skipif("not laspy.LazBackend.Lazrs.is_available()"),
        ),
        pytest.param(
            laspy.LazBackend.LazrsParallel,
            marks=pytest.mark.skipif("not laspy.LazBackend.Lazrs.is_available()"),
        ),
    ],
)
def test_reading_copc_file_normal_laz_file(backend):
    # COPC files are LAZ files with data arranged
    # in a special way that is still compatible with
    # standard file.
    # So we must be able to read a copc file as if it was
    # a classical LAZ

    las = laspy.read(SIMPLE_COPC_FILE, laz_backend=backend)
    assert las.header.version == "1.4"
    assert las.header.point_format == laspy.PointFormat(7)
    assert len(las) == 1065


@pytest.mark.skipif("not laspy.LazBackend.Lazrs.is_available()")
def test_querying_copc_local_file():
    with laspy.CopcReader.open(SIMPLE_COPC_FILE) as copc_reader:
        assert copc_reader.header.version == "1.4"
        assert copc_reader.header.point_format == laspy.PointFormat(7)
        points = copc_reader.query(resolution=50)
        assert len(points) == 24


@pytest.mark.skipif("not laspy.LazBackend.Lazrs.is_available()")
def test_querying_copc_local_file_with_page():
    path = SIMPLE_COPC_FILE.with_name("simple_with_page.copc.laz")
    point_count = laspy.read(path).header.point_count
    with laspy.CopcReader.open(path) as copc_reader:
        points = copc_reader.query()
        assert point_count == 1065 == len(points)


@pytest.mark.skipif("laspy.LazBackend.Lazrs.is_available()")
def test_querying_copc_local_file_proper_error_if_no_lazrs():
    with pytest.raises(laspy.errors.LazError):
        with laspy.CopcReader.open(SIMPLE_COPC_FILE) as _:
            pass


@pytest.mark.skipif("not laspy.LazBackend.Lazrs.is_available()")
def test_querying_copc_local_file_object():
    with open(SIMPLE_COPC_FILE, "rb") as fh:
        with laspy.CopcReader.open(fh) as copc_reader:
            assert copc_reader.header.version == "1.4"
            assert copc_reader.header.point_format == laspy.PointFormat(7)
            points = copc_reader.query(resolution=50)
            assert len(points) == 24


@pytest.mark.skipif("not laspy.LazBackend.Lazrs.is_available()")
def test_querying_copc_local_file_object_with_page():
    path = SIMPLE_COPC_FILE.with_name("simple_with_page.copc.laz")
    point_count = laspy.read(path).header.point_count
    with open(path, "rb") as fh:
        with laspy.CopcReader.open(fh) as copc_reader:
            points = copc_reader.query()
            assert point_count == 1065 == len(points)


@pytest.mark.skipif("laspy.LazBackend.Lazrs.is_available()")
def test_querying_copc_local_file_object_proper_error_if_no_lazrs():
    with pytest.raises(laspy.errors.LazError):
        with open(SIMPLE_COPC_FILE, "rb") as fh:
            with laspy.CopcReader.open(fh) as _:
                pass


@pytest.mark.skipif(
    not (
        laspy.LazBackend.Lazrs.is_available()
        and requests is not None
        and RangeHTTPServer is not None
    ),
    reason="neither lazrs, nor requests, nor RangeHTTPServer are installed",
)
def test_copc_over_http():
    server_proc = subprocess.Popen(
        [sys.executable, "-m", "RangeHTTPServer"], cwd=str(Path(__file__).parent)
    )

    with laspy.CopcReader.open(
        "http://localhost:8000/data/simple.copc.laz"
    ) as copc_reader:
        assert copc_reader.header.version == "1.4"
        assert copc_reader.header.point_format == laspy.PointFormat(7)
        points = copc_reader.query(resolution=50)
        assert len(points) == 24

    server_proc.terminate()




@pytest.mark.skipif("not laspy.LazBackend.Lazrs.is_available()")
def test_copc_roundtrip(tmp_path):
    """Read a COPC file, write it back as COPC, read again, verify all points present."""
    las = laspy.read(SIMPLE_COPC_FILE)
    output_path = str(tmp_path / "roundtrip.copc.laz")

    las.write(output_path)

    with laspy.CopcReader.open(output_path) as reader:
        points = reader.query()
        assert len(points) == 1065


@pytest.mark.skipif("not laspy.LazBackend.Lazrs.is_available()")
def test_copc_roundtrip_coordinates_match(tmp_path):
    """Verify that point coordinates survive the COPC roundtrip."""
    las = laspy.read(SIMPLE_COPC_FILE)
    output_path = str(tmp_path / "coords.copc.laz")

    las.write(output_path)

    las2 = laspy.read(output_path)
    # All points should be present (order may differ due to octree reordering)
    assert len(las2) == len(las)
    # Sorted coordinates should match
    orig_xyz = np.sort(np.column_stack([las.X, las.Y, las.Z]), axis=0)
    new_xyz = np.sort(np.column_stack([las2.X, las2.Y, las2.Z]), axis=0)
    np.testing.assert_array_equal(orig_xyz, new_xyz)


@pytest.mark.skipif("not laspy.LazBackend.Lazrs.is_available()")
def test_copc_spatial_query(tmp_path):
    """Write COPC with small target to create multiple octree levels, then query."""
    las = laspy.read(SIMPLE_COPC_FILE)
    output_path = str(tmp_path / "spatial.copc.laz")
    # Use small target to force octree subdivision
    laspy.CopcWriter.write(output_path, las.header, las.points, target_points_per_node=100)

    with laspy.CopcReader.open(output_path) as reader:
        # Query full extent
        all_points = reader.query()
        assert len(all_points) == 1065

        # Query with coarse resolution should return fewer points (LOD filtering)
        # The data spans ~4600m so spacing is ~1159m; use a large resolution value
        low_res_points = reader.query(resolution=5000)
        assert len(low_res_points) < 1065
        assert len(low_res_points) > 0


@pytest.mark.skipif("not laspy.LazBackend.Lazrs.is_available()")
def test_copc_invalid_point_format():
    """COPC requires point format 6, 7, or 8."""
    header = laspy.LasHeader(point_format=0, version="1.4")
    points = laspy.PackedPointRecord.zeros(10, header.point_format)

    with pytest.raises(laspy.LaspyException, match="point format"):
        laspy.CopcWriter.write(io.BytesIO(), header, points)


@pytest.mark.skipif("not laspy.LazBackend.Lazrs.is_available()")
def test_copc_small_file(tmp_path):
    """Write a very small COPC file (fewer points than target_points_per_node)."""
    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = [0.001, 0.001, 0.001]
    header.offsets = [0.0, 0.0, 0.0]
    points = laspy.PackedPointRecord.zeros(10, header.point_format)
    points["X"] = np.arange(10, dtype=np.int32)
    points["Y"] = np.arange(10, dtype=np.int32)
    points["Z"] = np.arange(10, dtype=np.int32)

    output_path = str(tmp_path / "small.copc.laz")
    laspy.CopcWriter.write(output_path, header, points)

    with laspy.CopcReader.open(output_path) as reader:
        result = reader.query()
        assert len(result) == 10


@pytest.mark.skipif("not laspy.LazBackend.Lazrs.is_available()")
def test_copc_info_vlr_position(tmp_path):
    """CopcInfoVlr should be the first VLR at offset 375."""
    las = laspy.read(SIMPLE_COPC_FILE)
    output_path = str(tmp_path / "vlr_pos.copc.laz")
    las.write(output_path)

    with open(output_path, "rb") as f:
        # LAS 1.4 header is 375 bytes
        f.seek(375)
        # VLR header: 2 (reserved) + 16 (user_id) + 2 (record_id) + 2 (length) + 32 (description)
        f.read(2)  # reserved
        user_id = f.read(16).rstrip(b"\x00").decode("ascii")
        record_id = int.from_bytes(f.read(2), "little")
        assert user_id == "copc"
        assert record_id == 1


@pytest.mark.skipif("not laspy.LazBackend.Lazrs.is_available()")
def test_copc_gps_time_bounds(tmp_path):
    """GPS time min/max in CopcInfoVlr should match actual data."""
    las = laspy.read(SIMPLE_COPC_FILE)
    output_path = str(tmp_path / "gps.copc.laz")
    las.write(output_path)

    with laspy.CopcReader.open(output_path) as reader:
        copc_info = reader.copc_info
        all_points = reader.query()
        if "gps_time" in all_points.point_format.dimension_names:
            gps = np.asarray(all_points["gps_time"])
            assert copc_info.gps_min <= gps.min()
            assert copc_info.gps_max >= gps.max()


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(
            laspy.LazBackend.Lazrs,
            marks=pytest.mark.skipif(
                not laspy.LazBackend.Lazrs.is_available(), reason="lazrs not installed"
            ),
        ),
        pytest.param(
            laspy.LazBackend.LazrsParallel,
            marks=pytest.mark.skipif(
                not laspy.LazBackend.Lazrs.is_available(), reason="lazrs not installed"
            ),
        ),
    ],
)
def test_copc_selective_decompression(backend):
    # We only decompress X,Y, return number, number of returns, classification
    selection = laspy.DecompressionSelection.base().decompress_classification()

    with laspy.CopcReader.open(SIMPLE_COPC_FILE) as copc_reader:
        fully_decompressed_points = copc_reader.query()

    with laspy.CopcReader.open(
        SIMPLE_COPC_FILE, decompression_selection=selection
    ) as copc_reader:
        partially_decompressed_points = copc_reader.query()

    print(
        np.sum(
            fully_decompressed_points.classification
            == partially_decompressed_points.classification
        ),
        len(fully_decompressed_points),
    )

    assert np.all(fully_decompressed_points.X == partially_decompressed_points.X)
    assert np.all(fully_decompressed_points.Y == partially_decompressed_points.Y)
    assert np.all(
        fully_decompressed_points.return_number
        == partially_decompressed_points.return_number
    )
    assert np.all(
        fully_decompressed_points.number_of_returns
        == partially_decompressed_points.number_of_returns
    )
    # assert np.all(
    #     fully_decompressed_points.classification == partially_decompressed_points.classification
    # )

    # Since COPC uses variable chunk size its easier to test that
    # values are all not eq between fully and partially decompressed
    assert np.any(
        partially_decompressed_points.point_source_id
        != fully_decompressed_points.point_source_id
    )
    assert np.any(partially_decompressed_points.Z != fully_decompressed_points.Z)


@pytest.mark.skipif("not laspy.LazBackend.Lazrs.is_available()")
@pytest.mark.parametrize(
    "copc_file",
    [SIMPLE_COPC_FILE, SIMPLE_COPC_FILE.with_name("simple_with_page.copc.laz")],
)
def test_copc_edit_rewrite_reuses_octree(copc_file, monkeypatch):
    # Editing attributes of a COPC file and re-writing it as COPC
    # must reuse the source octree structure instead of rebuilding it.
    import laspy.copcwriter as copcwriter

    def fail_build(*args, **kwargs):
        raise AssertionError("octree should not be rebuilt for attribute edits")

    monkeypatch.setattr(copcwriter, "_build_octree", fail_build)

    las = laspy.read(copc_file)
    original_x = np.array(las.X)
    las.classification[:] = 7

    output = io.BytesIO()
    laspy.CopcWriter.write(output, las.header, las.points)
    data = output.getvalue()

    # Data round-trips: same points in the same order, only classification changed
    back = laspy.read(io.BytesIO(data))
    assert np.array_equal(np.array(back.X), original_x)
    assert np.all(np.array(back.classification) == 7)

    # The rewritten file is a valid COPC with the same hierarchy
    with laspy.CopcReader.open(copc_file) as source_reader:
        source_entries = {
            (k.level, k.x, k.y, k.z): e.point_count
            for k, e in source_reader.root_page.entries.items()
            if e.point_count != -1
        }
    with laspy.CopcReader(io.BytesIO(data)) as reader:
        assert len(reader.query()) == len(las)
        rewritten_entries = {
            (k.level, k.x, k.y, k.z): e.point_count
            for k, e in reader.root_page.entries.items()
        }
    # rewritten hierarchy is a single page containing at least all
    # point-bearing entries of the source root page
    for key, count in source_entries.items():
        assert rewritten_entries[key] == count


@pytest.mark.skipif("not laspy.LazBackend.Lazrs.is_available()")
def test_copc_edit_rewrite_rebuilds_when_points_moved(monkeypatch):
    # Coordinate edits that move points outside their octree nodes
    # must fall back to a full octree rebuild.
    import laspy.copcwriter as copcwriter

    build_calls = []
    original_build = copcwriter._build_octree

    def counting_build(*args, **kwargs):
        build_calls.append(1)
        return original_build(*args, **kwargs)

    monkeypatch.setattr(copcwriter, "_build_octree", counting_build)

    las = laspy.read(SIMPLE_COPC_FILE)
    span = int(
        (las.header.maxs[0] - las.header.mins[0]) / las.header.scales[0]
    )
    las.X = np.asarray(las.X) + 4 * span

    output = io.BytesIO()
    laspy.CopcWriter.write(output, las.header, las.points)
    assert build_calls, "moving points must trigger an octree rebuild"

    with laspy.CopcReader(io.BytesIO(output.getvalue())) as reader:
        assert len(reader.query()) == len(las)


@pytest.mark.skipif("not laspy.LazBackend.Lazrs.is_available()")
def test_copc_edit_rewrite_reuse_with_added_extra_dim(monkeypatch):
    # Adding an extra dimension keeps point count and order, so the
    # source octree must still be reused and the new dim round-trips.
    import laspy.copcwriter as copcwriter

    def fail_build(*args, **kwargs):
        raise AssertionError("octree should not be rebuilt when adding a dim")

    monkeypatch.setattr(copcwriter, "_build_octree", fail_build)

    las = laspy.read(SIMPLE_COPC_FILE)
    las.add_extra_dim(laspy.ExtraBytesParams(name="instance_id", type="i4"))
    ids = np.arange(len(las), dtype=np.int32)
    las.instance_id = ids

    output = io.BytesIO()
    laspy.CopcWriter.write(output, las.header, las.points)

    back = laspy.read(io.BytesIO(output.getvalue()))
    assert "instance_id" in back.point_format.extra_dimension_names
    assert np.array_equal(np.asarray(back.instance_id), ids)


@pytest.mark.skipif("not laspy.LazBackend.Lazrs.is_available()")
def test_copc_edit_rewrite_rebuilds_on_count_preserving_reorder(monkeypatch):
    # Reordering points keeps the point count, so only the per-chunk
    # containment check can detect that the source octree is no longer
    # valid. It must trigger a rebuild (on a multi-chunk file).
    import laspy.copcwriter as copcwriter

    build_calls = []
    original_build = copcwriter._build_octree

    def counting_build(*args, **kwargs):
        build_calls.append(1)
        return original_build(*args, **kwargs)

    monkeypatch.setattr(copcwriter, "_build_octree", counting_build)

    las = laspy.read(SIMPLE_COPC_FILE)
    order = np.argsort(np.asarray(las.gps_time), kind="stable")
    assert not np.array_equal(order, np.arange(len(las)))
    reordered = las[order]

    output = io.BytesIO()
    laspy.CopcWriter.write(output, reordered.header, reordered.points)
    assert build_calls, "count-preserving reorder must trigger a rebuild"

    with laspy.CopcReader(io.BytesIO(output.getvalue())) as reader:
        assert len(reader.query()) == len(las)


@pytest.mark.skipif("not laspy.LazBackend.Lazrs.is_available()")
def test_copc_parallel_and_sequential_writes_are_equivalent(monkeypatch):
    import laspy.copcwriter as copcwriter

    assert copcwriter._HAS_PARALLEL_CHUNKS, "lazrs is expected to support compress_chunks"

    las = laspy.read(SIMPLE_COPC_FILE)
    las.classification[:] = 3

    parallel_out = io.BytesIO()
    laspy.CopcWriter.write(parallel_out, las.header, las.points)

    monkeypatch.setattr(copcwriter, "_HAS_PARALLEL_CHUNKS", False)
    las = laspy.read(SIMPLE_COPC_FILE)
    las.classification[:] = 3
    sequential_out = io.BytesIO()
    laspy.CopcWriter.write(sequential_out, las.header, las.points)

    # Same compressor, same chunk boundaries -> byte-identical files
    assert parallel_out.getvalue() == sequential_out.getvalue()

    with laspy.CopcReader(io.BytesIO(parallel_out.getvalue())) as reader:
        points = reader.query()
        assert len(points) == len(las)
        assert np.all(np.asarray(points.classification) == 3)
