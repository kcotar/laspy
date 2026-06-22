import ctypes
import io
from typing import Any, BinaryIO, Optional

import numpy as np

from .._pointappender import IPointAppender
from .._pointreader import IPointReader
from .._pointwriter import IPointWriter
from ..errors import LaspyException
from ..header import LasHeader
from ..point.record import PackedPointRecord
from .lazbackend import ILazBackend
from .selection import DecompressionSelection

try:
    import laszip
except ModuleNotFoundError:
    laszip = None


class LaszipBackend(ILazBackend):
    def is_available(self) -> bool:
        return laszip is not None

    @property
    def supports_append(self) -> bool:
        return False

    def create_appender(self, dest: BinaryIO, header: LasHeader) -> IPointAppender:
        raise LaspyException("Laszip backend does not support appending")

    def create_reader(
        self,
        source: Any,
        header: LasHeader,
        decompression_selection: Optional[DecompressionSelection] = None,
    ) -> IPointReader:
        if decompression_selection is None:
            decompression_selection = DecompressionSelection.all()
        return LaszipPointReader(
            source,
            header,
            decompression_selection=decompression_selection,
        )

    def create_writer(
        self,
        dest: Any,
        header: "LasHeader",
    ) -> IPointWriter:
        return LaszipPointWriter(dest, header)


class LaszipPointReader(IPointReader):
    """Implementation for the laszip backend"""

    def __init__(
        self,
        source: BinaryIO,
        header: LasHeader,
        decompression_selection: DecompressionSelection,
    ) -> None:
        self._source = source
        self._source.seek(0)
        selection = decompression_selection.to_laszip()
        self.unzipper = laszip.LasUnZipper(source, selection)
        unzipper_header = self.unzipper.header
        assert unzipper_header.point_data_format == header.point_format.id
        assert unzipper_header.point_data_record_length == header.point_format.size
        self.point_size = header.point_format.size

    @property
    def source(self):
        return self._source

    def read_n_points(self, n: int) -> bytearray:
        points_data = bytearray(n * self.point_size)
        self.unzipper.decompress_into(points_data)
        return points_data

    def seek(self, point_index: int) -> None:
        self.unzipper.seek(point_index)

    def close(self) -> None:
        self.source.close()


class LaszipPointWriter(IPointWriter):
    """
    Compressed point writer using laszip backend
    """

    def __init__(self, dest: BinaryIO, header: LasHeader) -> None:
        self.dest = dest
        header.set_compressed(False)
        with io.BytesIO() as tmp:
            header.write_to(tmp)
            header_bytes = tmp.getvalue()

        self.zipper = laszip.LasZipper(self.dest, header_bytes)
        zipper_header = self.zipper.header
        assert zipper_header.point_data_format == header.point_format.id
        assert zipper_header.point_data_record_length == header.point_format.size

        header.set_compressed(True)

    @property
    def destination(self) -> BinaryIO:
        return self.dest

    def write_points(self, points: PackedPointRecord) -> None:
        points_bytes = np.frombuffer(points.array, np.uint8)
        self.zipper.compress(points_bytes)

    def done(self) -> None:
        self.zipper.done()

    def write_initial_header_and_vlrs(
        self, header: LasHeader, encoding_errors: str
    ) -> None:
        # Do nothing as creating the laszip zipper writes the header and vlrs
        pass

    @staticmethod
    def _copy_extra_bytes_min_max(src_vlr, dst_vlr) -> None:
        # Copy the data-derived min/max (and the no_data / options metadata that
        # goes with them) from the in-memory, already-grown ExtraBytes structs
        # onto the structs parsed from disk. Matched by name so it is robust to
        # ordering differences; the byte layout of the special-property slots is
        # identical between the two, so a raw memmove is safe and size-preserving.
        src_by_name = {st.format_name(): st for st in src_vlr.extra_bytes_structs}
        for dst_st in dst_vlr.extra_bytes_structs:
            src_st = src_by_name.get(dst_st.format_name())
            if src_st is None:
                continue
            dst_st.options = src_st.options
            for slot in ("_min", "_max", "_no_data"):
                ctypes.memmove(
                    ctypes.addressof(getattr(dst_st, slot)),
                    ctypes.addressof(getattr(src_st, slot)),
                    ctypes.sizeof(getattr(dst_st, slot)),
                )

    def write_updated_header(self, header: LasHeader, encoding_errors: str) -> None:
        # The laszip zipper serialised the header + VLRs when this writer was
        # constructed -- before the point stream let ``grow()`` compute the real
        # ExtraBytes min/max -- so the on-disk ExtraBytes VLR still holds the
        # ``partial_reset`` sentinels (which viewers read as bogus values such as
        # -0.01 / 0). Unlike the lazrs backends (which re-emit the whole header on
        # update), this writer only patched EVLR offsets. Re-read the on-disk
        # header (it also carries the LASzip VLR the zipper added, absent from
        # ``header``) and copy over the grown ExtraBytes min/max as well, then
        # rewrite it at the same size.
        in_mem_eb = header.vlrs.get("ExtraBytesVlr")

        if header.number_of_evlrs == 0 and not in_mem_eb:
            return

        self.dest.seek(0, io.SEEK_SET)
        file_header = LasHeader.read_from(self.dest)
        end_of_header_pos = self.dest.tell()

        if header.number_of_evlrs != 0:
            file_header.number_of_evlrs = header.number_of_evlrs
            file_header.start_of_first_evlr = header.start_of_first_evlr

        if in_mem_eb:
            file_eb = file_header.vlrs.get("ExtraBytesVlr")
            if file_eb:
                self._copy_extra_bytes_min_max(in_mem_eb[0], file_eb[0])

        self.dest.seek(0, io.SEEK_SET)
        file_header.write_to(
            self.dest, ensure_same_size=True, encoding_errors=encoding_errors
        )
        assert self.dest.tell() == end_of_header_pos
