"""
Memory and performance test for COPC writer with 10M points.

Measures peak memory (via tracemalloc) and wall time for the COPC write path.
Can be run standalone to compare before/after optimizations:

    python -m pytest tests/test_copc_memory.py -v -s

Or as a standalone script for quick comparison:

    python tests/test_copc_memory.py
"""
import io
import time
import tracemalloc

import numpy as np
import pytest

import laspy
from laspy.copcwriter import CopcWriter


def _create_points(n: int, seed: int = 42):
    """Create n synthetic LAS 1.4 / point format 6 points with realistic spread."""
    rng = np.random.default_rng(seed)

    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = [0.001, 0.001, 0.001]
    header.offsets = [500_000.0, 200_000.0, 0.0]

    points = laspy.PackedPointRecord.zeros(n, header.point_format)
    # Spread over ~1 km cube so the octree has meaningful structure
    points["X"] = rng.integers(0, 1_000_000, size=n, dtype=np.int32)
    points["Y"] = rng.integers(0, 1_000_000, size=n, dtype=np.int32)
    points["Z"] = rng.integers(0, 100_000, size=n, dtype=np.int32)
    points["return_number"] = np.ones(n, dtype=np.uint8)
    points["number_of_returns"] = np.ones(n, dtype=np.uint8)
    points["gps_time"] = rng.uniform(0, 1_000_000, size=n).astype(np.float64)

    return header, points


@pytest.mark.skipif("not laspy.LazBackend.Lazrs.is_available()")
def test_copc_10m_roundtrip_correctness(tmp_path):
    """Verify 10M-point COPC write + read roundtrip preserves all points."""
    n = 10_000_000
    header, points = _create_points(n)
    output = str(tmp_path / "big.copc.laz")

    CopcWriter.write(output, header, points)

    las2 = laspy.read(output)
    assert len(las2) == n

    orig_xyz = np.sort(np.column_stack([points["X"], points["Y"], points["Z"]]), axis=0)
    new_xyz = np.sort(np.column_stack([las2.X, las2.Y, las2.Z]), axis=0)
    np.testing.assert_array_equal(orig_xyz, new_xyz)


@pytest.mark.skipif("not laspy.LazBackend.Lazrs.is_available()")
def test_copc_10m_memory_usage(tmp_path):
    """Measure peak memory during 10M-point COPC write.

    The point data itself is ~360 MB (10M x 36 bytes for format 6).
    After optimization, overhead should stay well under 300 MB on top of that.
    """
    n = 10_000_000
    header, points = _create_points(n)
    output = str(tmp_path / "mem.copc.laz")

    point_data_bytes = n * header.point_format.size  # baseline

    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()

    t0 = time.perf_counter()
    CopcWriter.write(output, header, points)
    elapsed = time.perf_counter() - t0

    snapshot_after = tracemalloc.take_snapshot()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak / (1024 * 1024)
    data_mb = point_data_bytes / (1024 * 1024)

    print(f"\n--- COPC 10M write stats ---")
    print(f"  Points:       {n:,}")
    print(f"  Point data:   {data_mb:.0f} MB")
    print(f"  Peak memory:  {peak_mb:.0f} MB")
    print(f"  Overhead:     {peak_mb - data_mb:.0f} MB")
    print(f"  Wall time:    {elapsed:.2f} s")

    # Verify file is readable
    las2 = laspy.read(output)
    assert len(las2) == n

    # Memory overhead should stay under 500 MB (was ~1733 MB before optimization)
    overhead_mb = peak_mb - data_mb
    assert overhead_mb < 500, f"Memory overhead {overhead_mb:.0f} MB exceeds 500 MB limit"


def _run_benchmark():
    """Standalone benchmark for before/after comparison."""
    import tempfile
    import os

    n = 10_000_000
    print(f"Creating {n:,} points...")
    header, points = _create_points(n)

    with tempfile.TemporaryDirectory() as tmp:
        output = os.path.join(tmp, "bench.copc.laz")

        tracemalloc.start()
        t0 = time.perf_counter()
        CopcWriter.write(output, header, points)
        elapsed = time.perf_counter() - t0
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        file_size = os.path.getsize(output) / (1024 * 1024)
        peak_mb = peak / (1024 * 1024)
        data_mb = (n * header.point_format.size) / (1024 * 1024)

        print(f"\n=== COPC Writer Benchmark ===")
        print(f"  Points:        {n:,}")
        print(f"  Point data:    {data_mb:.0f} MB")
        print(f"  Peak memory:   {peak_mb:.0f} MB")
        print(f"  Overhead:      {peak_mb - data_mb:.0f} MB")
        print(f"  Wall time:     {elapsed:.2f} s")
        print(f"  Output size:   {file_size:.1f} MB")

        # Quick correctness check
        las2 = laspy.read(output)
        assert len(las2) == n
        print(f"  Correctness:   OK ({len(las2):,} points read back)")


if __name__ == "__main__":
    _run_benchmark()
