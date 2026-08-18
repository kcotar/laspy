import ctypes
import io
from typing import Any, BinaryIO, Optional

import numpy as np

from .._pointappender import IPointAppender
from .._pointreader import IPointReader
from .._pointwriter import IPointWriter
from ..errors import LaspyException
from ..header import LAS_HEADERS_SIZE, LasHeader
from ..point.record import PackedPointRecord
from .lazbackend import ILazBackend
from .selection import DecompressionSelection

try:
    import laszip
except ModuleNotFoundError:
    laszip = None

# Size of a (non-extended) VLR header on disk:
# 2 (reserved) + 16 (user_id) + 2 (record_id) + 2 (record_length) + 32 (description)
VLR_HEADER_SIZE = 54


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

    def _find_vlr_on_disk(self, header: "LasHeader", user_id: str, record_id: int):
        """Locate a VLR's record-data offset and on-disk length by walking the VLR
        headers laszip actually wrote.

        Reading the on-disk ``record_length`` fields (rather than trusting laspy's
        re-serialised VLR sizes) keeps this immune to VLRs that do not round-trip
        byte-for-byte through laspy's typed parsers.
        """
        pos = LAS_HEADERS_SIZE[str(header.version)] + len(header.extra_header_bytes)
        for _ in range(len(header.vlrs)):
            self.dest.seek(pos, io.SEEK_SET)
            vlr_header = self.dest.read(VLR_HEADER_SIZE)
            if len(vlr_header) < VLR_HEADER_SIZE:
                break
            disk_user_id = vlr_header[2:18].split(b"\x00", 1)[0].decode(
                "ascii", "replace"
            )
            disk_record_id = int.from_bytes(vlr_header[18:20], "little", signed=False)
            record_length = int.from_bytes(vlr_header[20:22], "little", signed=False)
            data_offset = pos + VLR_HEADER_SIZE
            if disk_user_id == user_id and disk_record_id == record_id:
                return data_offset, record_length
            pos = data_offset + record_length
        return None

    def write_updated_header(self, header: LasHeader, encoding_errors: str) -> None:
        # The laszip zipper serialised the header + VLRs when this writer was
        # constructed, so two things still need fixing up on disk: the EVLR
        # offset/count (the EVLRs are appended only after the point stream), and the
        # ExtraBytes VLR min/max (laszip wrote the ``partial_reset`` sentinels --
        # bogus values such as -0.01 / 0 -- before the point stream let ``grow()``
        # compute the real range).
        #
        # We patch ONLY those bytes, in place. We deliberately do NOT re-serialise
        # the whole header (as the lazrs backends do): laszip's on-disk VLR bytes
        # are not guaranteed to round-trip byte-for-byte through laspy's typed VLR
        # parsers -- a LASF_Spec classification-lookup record, or any vendor VLR
        # laspy normalises on parse, can re-serialise to a different length. That
        # would change offset_to_point_data and raise "writing header would change
        # original offset to data" (or, worse, silently corrupt the file). Patching
        # bytes in place leaves every other VLR -- and the data offset -- untouched.
        in_mem_eb = header.vlrs.get("ExtraBytesVlr")

        if header.number_of_evlrs == 0 and not in_mem_eb:
            return

        if header.number_of_evlrs != 0:
            # In a 1.4 header the EVLR bookkeeping immediately follows the 1.3
            # header block: start_of_first_evlr (8 bytes) then number_of_evlrs (4).
            self.dest.seek(LAS_HEADERS_SIZE["1.3"], io.SEEK_SET)
            self.dest.write(
                header.start_of_first_evlr.to_bytes(8, "little", signed=False)
            )
            self.dest.write(
                header.number_of_evlrs.to_bytes(4, "little", signed=False)
            )

        if in_mem_eb:
            self.dest.seek(0, io.SEEK_SET)
            file_header = LasHeader.read_from(self.dest)
            file_eb = file_header.vlrs.get("ExtraBytesVlr")
            if file_eb:
                self._copy_extra_bytes_min_max(in_mem_eb[0], file_eb[0])
                located = self._find_vlr_on_disk(
                    file_header, file_eb[0].user_id, file_eb[0].record_id
                )
                if located is not None:
                    data_offset, on_disk_len = located
                    new_record_data = file_eb[0].record_data_bytes()
                    # Size-preserving by construction; guard so an unexpected
                    # mismatch degrades to stale min/max rather than a broken file.
                    if len(new_record_data) == on_disk_len:
                        self.dest.seek(data_offset, io.SEEK_SET)
                        self.dest.write(new_record_data)

        self.dest.seek(0, io.SEEK_END)
