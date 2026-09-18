"""Reading a file whose point data ends before the header's declared count.

Upstream laspy logged ``Could only read N of the requested M points`` and returned a
short record, so a caller streaming a truncated file processed a partial dataset without
ever being told. This fork raises instead, with an opt-out for repair tooling.
"""

import pickle

import numpy as np
import pytest

import laspy
from laspy.errors import TruncatedPointDataError

DECLARED = 5_000
KEPT = 3_000


def _write(path, n=DECLARED):
    header = laspy.LasHeader(version="1.4", point_format=6)
    header.scales = [0.001] * 3
    header.offsets = [0.0] * 3
    las = laspy.LasData(header)
    rng = np.random.default_rng(0)
    las.x = rng.uniform(0, 100, n)
    las.y = rng.uniform(0, 100, n)
    las.z = rng.uniform(0, 10, n)
    las.write(str(path))
    return las


@pytest.fixture
def truncated(tmp_path):
    """A LAS whose header promises DECLARED points but whose data stops after KEPT."""
    path = tmp_path / "truncated.las"
    _write(path)
    with laspy.open(str(path)) as reader:
        offset = reader.header.offset_to_point_data
        point_size = reader.header.point_format.size
    with open(path, "r+b") as fh:
        fh.truncate(offset + KEPT * point_size)
    return path


@pytest.fixture
def intact(tmp_path):
    path = tmp_path / "intact.las"
    _write(path)
    return path


def test_intact_file_reads_without_complaint(intact):
    with laspy.open(str(intact)) as reader:
        assert len(reader.read_points(DECLARED)) == DECLARED


def test_read_points_raises_on_truncated_file(truncated):
    with laspy.open(str(truncated)) as reader:
        with pytest.raises(TruncatedPointDataError) as excinfo:
            reader.read_points(DECLARED)
    assert excinfo.value.declared_point_count == DECLARED
    assert excinfo.value.points_present == KEPT


def test_read_all_raises_on_truncated_file(truncated):
    with laspy.open(str(truncated)) as reader:
        with pytest.raises(TruncatedPointDataError):
            reader.read()


def test_chunk_iterator_raises_on_truncated_file(truncated):
    """The streaming path is how the pipeline reads big files, so it must raise too."""
    with laspy.open(str(truncated)) as reader:
        with pytest.raises(TruncatedPointDataError):
            for _ in reader.chunk_iterator(1_000):
                pass


def test_opt_out_restores_the_permissive_behaviour(truncated, caplog):
    with laspy.open(str(truncated), strict_point_count=False) as reader:
        points = reader.read_points(DECLARED)
    assert len(points) == KEPT
    assert "Could only read" in caplog.text


def test_points_read_advances_by_what_was_actually_read(truncated):
    """Charging the full request would leave points_read overstating the file."""
    with laspy.open(str(truncated), strict_point_count=False) as reader:
        reader.read_points(DECLARED)
        assert reader.points_read == KEPT


def test_error_survives_a_pickle_round_trip():
    """Exceptions cross Ray boundaries by pickle; a signature that demanded the extra
    arguments would arrive as an unserializable stand-in instead of this error."""
    original = TruncatedPointDataError(
        "boom", declared_point_count=DECLARED, points_present=KEPT
    )
    restored = pickle.loads(pickle.dumps(original))
    assert isinstance(restored, TruncatedPointDataError)
    assert str(restored) == "boom"
    assert restored.declared_point_count == DECLARED
    assert restored.points_present == KEPT


def test_error_is_a_laspy_exception():
    """So callers catching LaspyException broadly still handle it."""
    assert issubclass(TruncatedPointDataError, laspy.LaspyException)
