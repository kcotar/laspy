import io
import logging
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from math import floor
from typing import BinaryIO, List, Optional, Union

import numpy as np

from .copc import CopcHierarchyVlr, CopcInfoVlr, Entry, VoxelKey
from .errors import LaspyException
from .header import LasHeader
from .point.record import PackedPointRecord
from .vlrs.known import LasZipVlr
from .vlrs.vlrlist import VLRList

try:
    import lazrs
except ModuleNotFoundError:
    lazrs = None

logger = logging.getLogger(__name__)

_MORTON_BITS = 21                        # bits per dimension in Morton code
_MORTON_MAX  = (1 << _MORTON_BITS) - 1  # 2097151


def _spread_bits(v: np.ndarray) -> np.ndarray:
    """Spread 21-bit integers to every 3rd bit position for 63-bit Morton encoding."""
    v = v.astype(np.uint64) & np.uint64(0x1FFFFF)
    v = (v | (v << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
    v = (v | (v << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
    v = (v | (v << np.uint64(8)))  & np.uint64(0x100F00F00F00F00F)
    v = (v | (v << np.uint64(4)))  & np.uint64(0x10C30C30C30C30C3)
    v = (v | (v << np.uint64(2)))  & np.uint64(0x1249249249249249)
    return v


def _morton_codes(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    root_min: np.ndarray,
    side: float,
) -> np.ndarray:
    """Compute 63-bit Morton (Z-order) codes for arrays of 3D coordinates.

    Quantises each axis to _MORTON_BITS bits then interleaves the bits.
    All points are assumed to lie within [root_min, root_min + side].
    """
    scale = side / _MORTON_MAX
    qx = np.clip(((x - root_min[0]) / scale), 0, _MORTON_MAX).astype(np.uint64)
    qy = np.clip(((y - root_min[1]) / scale), 0, _MORTON_MAX).astype(np.uint64)
    qz = np.clip(((z - root_min[2]) / scale), 0, _MORTON_MAX).astype(np.uint64)
    return _spread_bits(qx) | (_spread_bits(qy) << np.uint64(1)) | (_spread_bits(qz) << np.uint64(2))


@dataclass
class OctreeChunk:
    key: VoxelKey
    point_indices: np.ndarray


@dataclass
class OctreeResult:
    chunks: List[OctreeChunk]
    intermediate_keys: List[VoxelKey]  # Internal nodes that were subdivided
    center: np.ndarray
    halfsize: float
    spacing: float


def _build_octree(
    points: PackedPointRecord,
    header: LasHeader,
    target_points_per_node: int = 100_000,
    max_depth: Optional[int] = None,
) -> OctreeResult:
    x = np.asarray(points["X"], dtype=np.float64)
    y = np.asarray(points["Y"], dtype=np.float64)
    z = np.asarray(points["Z"], dtype=np.float64)

    # Convert to actual coordinates for octree bounds
    actual_x = x * header.scales[0] + header.offsets[0]
    actual_y = y * header.scales[1] + header.offsets[1]
    actual_z = z * header.scales[2] + header.offsets[2]

    mins = np.array(
        [actual_x.min(), actual_y.min(), actual_z.min()], dtype=np.float64
    )
    maxs = np.array(
        [actual_x.max(), actual_y.max(), actual_z.max()], dtype=np.float64
    )

    halfsize = float(np.max(maxs - mins)) / 2.0
    center = mins + halfsize

    if halfsize == 0.0:
        halfsize = 1.0

    n_points = len(points)
    if max_depth is None:
        # Safety cap only — the adaptive leaf criterion below drives actual depth.
        # Matches PDAL: keep splitting until every leaf has <= target_points_per_node.
        max_depth = 20

    # Adaptive cell count: scales with dataset size, matching PDAL's behaviour.
    # floor(cbrt(N) * 2) gives ~147 for 411k points; minimum 128 for small files.
    cell_count = max(128, floor(n_points ** (1.0 / 3.0) * 2))

    root_min = center - halfsize

    # Pre-sort all points once by Morton (Z-order) code.
    # This replaces the per-node random shuffle: the BFS passes Morton-ordered
    # index arrays to children, so np.unique's first-occurrence selection is
    # deterministic and spatially coherent — matching PDAL's behaviour without
    # any per-node sort.  Morton order is preserved through octant splits
    # because Z-order is locality-preserving: all points of a child octant
    # appear contiguously (and in Z-order) within the parent's sorted range.
    codes = _morton_codes(actual_x, actual_y, actual_z, root_min, 2.0 * halfsize)
    all_indices = np.argsort(codes).astype(np.intp)

    # BFS octree construction with LOD subsampling.
    root_key = VoxelKey.from_values(0, 0, 0, 0)
    chunks = []
    queue = deque()
    queue.append((root_key, all_indices))
    max_depth_used = 0

    while queue:
        key, indices = queue.popleft()

        if len(indices) == 0:
            continue

        if len(indices) <= target_points_per_node or key.level >= max_depth:
            # Leaf node: remaining points fit within target — store all without further splitting.
            chunks.append(OctreeChunk(key=key, point_indices=indices))
            if key.level > max_depth_used:
                max_depth_used = key.level
            continue

        # Internal node: voxel-occupancy subsampling.
        # indices is Morton-ordered, so np.unique keeps the Morton-first point
        # per cell — deterministic and spatially representative, no shuffle needed.
        node_side = (2.0 * halfsize) / (2 ** key.level)
        node_min = root_min + np.array([key.x, key.y, key.z], dtype=np.float64) * node_side
        cell_w = node_side / cell_count

        gx = np.clip(((actual_x[indices] - node_min[0]) / cell_w).astype(np.int32), 0, cell_count - 1)
        gy = np.clip(((actual_y[indices] - node_min[1]) / cell_w).astype(np.int32), 0, cell_count - 1)
        gz = np.clip(((actual_z[indices] - node_min[2]) / cell_w).astype(np.int32), 0, cell_count - 1)
        grid_keys = (gx.astype(np.int64) * cell_count + gy.astype(np.int64)) * cell_count + gz.astype(np.int64)

        _, first_occ = np.unique(grid_keys, return_index=True)

        keep_mask = np.zeros(len(indices), dtype=bool)
        keep_mask[first_occ] = True

        node_indices = indices[first_occ]
        remaining_indices = indices[~keep_mask]  # Morton order preserved

        chunks.append(OctreeChunk(key=key, point_indices=node_indices))
        if key.level > max_depth_used:
            max_depth_used = key.level

        # Partition remaining (Morton-ordered) into 8 children by octant.
        # Each child's subset is also Morton-ordered — no re-sort needed.
        node_center = node_min + node_side / 2.0
        px = actual_x[remaining_indices]
        py = actual_y[remaining_indices]
        pz = actual_z[remaining_indices]

        octant = (
            (px >= node_center[0]).astype(np.uint8)
            | ((py >= node_center[1]).astype(np.uint8) << 1)
            | ((pz >= node_center[2]).astype(np.uint8) << 2)
        )

        for direction in range(8):
            child_indices = remaining_indices[octant == direction]
            if len(child_indices) > 0:
                queue.append((key.child(direction), child_indices))

    chunks.sort(key=lambda c: (c.key.level, c.key.x, c.key.y, c.key.z))

    spacing = (2.0 * halfsize) / cell_count

    return OctreeResult(
        chunks=chunks,
        intermediate_keys=[],
        center=center,
        halfsize=halfsize,
        spacing=spacing,
    )


class CopcWriter:
    """Writes point cloud data as a COPC (Cloud Optimized Point Cloud) file.

    COPC files are LAZ 1.4 files with points organized in an octree structure,
    enabling efficient spatial queries and HTTP range requests.
    """

    @staticmethod
    def write(
        destination: Union[str, BinaryIO],
        header: LasHeader,
        points: PackedPointRecord,
        target_points_per_node: int = 100_000,
        max_depth: Optional[int] = None,
    ) -> None:
        """Write points as a COPC file.

        Parameters
        ----------
        destination : str or file object
            Path or writable binary stream.
        header : LasHeader
            Header to use. Must be version 1.4 with point format 6, 7, or 8.
        points : PackedPointRecord
            The points to write.
        target_points_per_node : int
            Target number of points per octree leaf node.
        max_depth : int, optional
            Maximum octree depth. Auto-computed if None.
        """
        if lazrs is None:
            raise LaspyException("COPC writing requires the 'lazrs' package")

        if header.point_format.id not in (6, 7, 8):
            raise LaspyException(
                f"COPC requires point format 6, 7, or 8, got {header.point_format.id}"
            )

        if header.version.minor < 4:
            raise LaspyException(
                f"COPC requires LAS version 1.4, got {header.version}"
            )

        if isinstance(destination, str):
            with open(destination, "wb+") as f:
                CopcWriter._write_copc(f, header, points, target_points_per_node, max_depth)
        else:
            CopcWriter._write_copc(destination, header, points, target_points_per_node, max_depth)

    @staticmethod
    def _write_copc(
        dest: BinaryIO,
        header: LasHeader,
        points: PackedPointRecord,
        target_points_per_node: int,
        max_depth: Optional[int],
    ) -> None:
        header = deepcopy(header)
        header.are_points_compressed = True
        header.point_count = len(points)
        # CopcWriter stores hierarchy via CopcInfoVlr, not as LAS EVLRs.
        # Clear these fields to prevent readers from seeking to a stale/zero offset.
        header.start_of_first_evlr = 0
        header.number_of_evlrs = 0

        # Update bounds
        if len(points) > 0:
            x = np.asarray(points["X"], dtype=np.float64)
            y = np.asarray(points["Y"], dtype=np.float64)
            z = np.asarray(points["Z"], dtype=np.float64)
            header.maxs = [
                float(x.max()) * header.scales[0] + header.offsets[0],
                float(y.max()) * header.scales[1] + header.offsets[1],
                float(z.max()) * header.scales[2] + header.offsets[2],
            ]
            header.mins = [
                float(x.min()) * header.scales[0] + header.offsets[0],
                float(y.min()) * header.scales[1] + header.offsets[1],
                float(z.min()) * header.scales[2] + header.offsets[2],
            ]

            # Update number_of_points_by_return
            header.number_of_points_by_return = [0] * len(header.number_of_points_by_return)
            return_numbers = np.asarray(points["return_number"])
            for rn in range(1, min(16, int(return_numbers.max()) + 1)):
                header.number_of_points_by_return[rn - 1] = int(np.sum(return_numbers == rn))
        else:
            header.maxs = [0.0, 0.0, 0.0]
            header.mins = [0.0, 0.0, 0.0]

        # Build octree
        octree = _build_octree(points, header, target_points_per_node, max_depth)

        # Create CopcInfoVlr with placeholder hierarchy offset
        copc_info = CopcInfoVlr()
        copc_info.center[:] = octree.center
        copc_info.halfsize = octree.halfsize
        copc_info.spacing = octree.spacing
        copc_info.hierarchy_root_offset = 0
        copc_info.hierarchy_root_size = 0

        # Set GPS time bounds if available
        if "gps_time" in points.point_format.dimension_names:
            gps = np.asarray(points["gps_time"])
            copc_info.gps_min = float(gps.min())
            copc_info.gps_max = float(gps.max())

        # Remove any existing COPC or LasZip VLRs
        vlrs_to_remove = []
        for i, vlr in enumerate(header.vlrs):
            if isinstance(vlr, (CopcInfoVlr, LasZipVlr)):
                vlrs_to_remove.append(i)
            elif hasattr(vlr, 'user_id') and vlr.user_id == "copc":
                vlrs_to_remove.append(i)
        for i in reversed(vlrs_to_remove):
            header.vlrs.pop(i)

        # Create LAZ VLR for variable-size chunks
        laz_vlr = lazrs.LazVlr.new_for_compression(
            header.point_format.id,
            header.point_format.num_extra_bytes,
            use_variable_size_chunks=True,
        )
        laszip_vlr = LasZipVlr(laz_vlr.record_data())

        # Insert CopcInfoVlr first, then LasZipVlr
        header.vlrs.insert(0, copc_info)
        header.vlrs.append(laszip_vlr)

        # Write header + VLRs
        header.write_to(dest)
        offset_to_point_data = header.offset_to_point_data

        # Write compressed chunks
        compressor = lazrs.LasZipCompressor(dest, laz_vlr)
        # After compressor init, it writes 8-byte chunk table offset placeholder
        # First chunk starts at offset_to_point_data + 8

        chunk_start = offset_to_point_data + 8
        entries = []

        point_bytes_array = np.frombuffer(points.array, dtype=np.uint8)
        point_size = header.point_format.size

        for chunk in octree.chunks:
            idx = chunk.point_indices
            n_pts = len(idx)

            # Extract point bytes for this chunk
            # Build byte indices for all points in this chunk
            byte_offsets = (idx * point_size).reshape(-1, 1) + np.arange(point_size, dtype=np.intp)
            chunk_bytes = point_bytes_array[byte_offsets.ravel()]

            compressor.compress_many(chunk_bytes)
            compressor.finish_current_chunk()

            chunk_end = dest.tell()
            byte_size = chunk_end - chunk_start

            entry = Entry()
            entry.key = chunk.key
            entry.offset = chunk_start
            entry.byte_size = byte_size
            entry.point_count = n_pts
            entries.append(entry)

            chunk_start = chunk_end

        compressor.done()

        # Build hierarchy EVLR data (all entries concatenated)
        entry_bytes = b"".join(e.to_bytes() for e in entries)

        hierarchy_vlr = CopcHierarchyVlr()
        hierarchy_vlr.data = entry_bytes

        # Write hierarchy as a proper COPC EVLR at end of file.
        # hierarchy_root_offset must point to the EVLR *data* (after the
        # 60-byte EVLR header), matching the COPC spec and PDAL's convention.
        _EVLR_HEADER_SIZE = 60
        evlr_start = dest.tell()
        hierarchy_root_offset = evlr_start + _EVLR_HEADER_SIZE
        hierarchy_root_size = len(entry_bytes)

        evlr_list = VLRList([hierarchy_vlr])
        evlr_list.write_to(dest, as_extended=True)

        # Update CopcInfoVlr with actual hierarchy location
        copc_info.hierarchy_root_offset = hierarchy_root_offset
        copc_info.hierarchy_root_size = hierarchy_root_size

        # Update header EVLR bookkeeping fields
        header.start_of_first_evlr = evlr_start
        header.number_of_evlrs = 1

        # Rewrite header with updated CopcInfoVlr and EVLR fields
        dest.seek(0)
        header.write_to(dest, ensure_same_size=True)
