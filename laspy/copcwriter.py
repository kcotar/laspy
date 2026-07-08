import io
import logging
import math
from collections import deque
from copy import deepcopy
from dataclasses import dataclass

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

# Grid resolution calibrated to match PDAL/untwine LOD distribution.
# PDAL (bottom-up) uses base_spacing = side/128, grid_cell_width = spacing/sqrt(3).
_PDAL_CELL_COUNT = math.ceil(128 * math.sqrt(3) / 1.5)  # 148


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
    n_points = len(points)
    sx, sy, sz = header.scales
    ox, oy, oz = header.offsets

    if n_points == 0:
        # Empty input: still emit the mandatory "0-0-0-0" root node so the
        # hierarchy is valid COPC; bounds degenerate to a unit cube at origin.
        return OctreeResult(
            chunks=[
                OctreeChunk(
                    key=VoxelKey.from_values(0, 0, 0, 0),
                    point_indices=np.empty(0, dtype=np.int32),
                )
            ],
            intermediate_keys=[],
            center=np.zeros(3, dtype=np.float64),
            halfsize=1.0,
            spacing=2.0 / _PDAL_CELL_COUNT,
        )

    # --- Phase 1: Contiguous int32 copies for fast sequential access ---
    # Used for bounds, morton codes, then freed before argsort to reduce peak.
    int_x = np.ascontiguousarray(points["X"])
    int_y = np.ascontiguousarray(points["Y"])
    int_z = np.ascontiguousarray(points["Z"])

    mins = np.array(
        [float(int_x.min()) * sx + ox, float(int_y.min()) * sy + oy, float(int_z.min()) * sz + oz],
        dtype=np.float64,
    )
    maxs = np.array(
        [float(int_x.max()) * sx + ox, float(int_y.max()) * sy + oy, float(int_z.max()) * sz + oz],
        dtype=np.float64,
    )

    halfsize = float(np.max(maxs - mins)) / 2.0
    center = mins + halfsize
    if halfsize == 0.0:
        halfsize = 1.0

    if max_depth is None:
        max_depth = 20

    # Our top-down approach needs a finer non-root grid (~222 cells) to capture
    # enough points at each intermediate level, matching PDAL's output.
    root_cell_count = _PDAL_CELL_COUNT
    nonroot_cell_count = math.ceil(128 * math.sqrt(3))       # 222
    root_min = center - halfsize
    side = 2.0 * halfsize

    # --- Phase 2: Morton codes in 1M-point chunks ---
    # Limits float64 temporaries to ~24 MB per chunk instead of ~240 MB for
    # the full dataset.  int_x/y/z provide fast contiguous access.
    codes = np.empty(n_points, dtype=np.uint64)
    _CHUNK = 1_000_000
    for start in range(0, n_points, _CHUNK):
        end = min(start + _CHUNK, n_points)
        sl = slice(start, end)
        ax = int_x[sl].astype(np.float64) * sx + ox
        ay = int_y[sl].astype(np.float64) * sy + oy
        az = int_z[sl].astype(np.float64) * sz + oz
        codes[sl] = _morton_codes(ax, ay, az, root_min, side)

    # --- Phase 3: Sort — free int32 coords first to reduce argsort peak ---
    del int_x, int_y, int_z
    idx_dtype = np.int32 if n_points <= np.iinfo(np.int32).max else np.intp
    all_indices = np.argsort(codes).astype(idx_dtype)
    del codes

    # --- Phase 4: Re-create contiguous int32 coords for BFS ---
    int_x = np.ascontiguousarray(points["X"])
    int_y = np.ascontiguousarray(points["Y"])
    int_z = np.ascontiguousarray(points["Z"])

    # Use int32 grid_keys when cell_count^3 fits, avoiding int64 conversions.
    max_cc = max(root_cell_count, nonroot_cell_count)
    use_int32_grid = (max_cc ** 3) < np.iinfo(np.int32).max

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

        # At max_depth, store all remaining points as a leaf.
        if key.level >= max_depth:
            chunks.append(OctreeChunk(key=key, point_indices=indices))
            if key.level > max_depth_used:
                max_depth_used = key.level
            continue

        # Voxel-occupancy subsampling: always subsample at every level
        # (matching PDAL/untwine behaviour) to produce an even LOD distribution.
        # indices is Morton-ordered, so np.unique keeps the Morton-first point
        # per cell — deterministic and spatially representative, no shuffle needed.
        cell_count = root_cell_count if key.level == 0 else nonroot_cell_count
        node_side = side / (2 ** key.level)
        node_min = root_min + np.array([key.x, key.y, key.z], dtype=np.float64) * node_side
        cell_w = node_side / cell_count

        # Build grid_keys incrementally, one axis at a time, freeing each
        # grid-cell array before computing the next.  This avoids having
        # gx + gy + gz + grid_keys all alive simultaneously.
        node_min_ix = (node_min[0] - ox) / sx
        cs_x = sx / cell_w
        gx = np.clip(((int_x[indices] - node_min_ix) * cs_x).astype(np.int32), 0, cell_count - 1)
        if use_int32_grid:
            cc32 = np.int32(cell_count)
            grid_keys = gx * cc32
        else:
            grid_keys = gx.astype(np.int64) * cell_count
        del gx

        node_min_iy = (node_min[1] - oy) / sy
        cs_y = sy / cell_w
        gy = np.clip(((int_y[indices] - node_min_iy) * cs_y).astype(np.int32), 0, cell_count - 1)
        grid_keys += gy if use_int32_grid else gy.astype(np.int64)
        del gy
        grid_keys *= cc32 if use_int32_grid else cell_count

        node_min_iz = (node_min[2] - oz) / sz
        cs_z = sz / cell_w
        gz = np.clip(((int_z[indices] - node_min_iz) * cs_z).astype(np.int32), 0, cell_count - 1)
        grid_keys += gz if use_int32_grid else gz.astype(np.int64)
        del gz

        _, first_occ = np.unique(grid_keys, return_index=True)
        del grid_keys

        keep_mask = np.zeros(len(indices), dtype=bool)
        keep_mask[first_occ] = True

        node_indices = indices[first_occ]
        remaining_indices = indices[~keep_mask]  # Morton order preserved
        del keep_mask

        chunks.append(OctreeChunk(key=key, point_indices=node_indices))
        if key.level > max_depth_used:
            max_depth_used = key.level

        # If all points fit in unique cells, nothing left for children.
        if len(remaining_indices) == 0:
            continue

        # Partition remaining (Morton-ordered) into 8 children by octant.
        # Each child's subset is also Morton-ordered — no re-sort needed.
        # Compare in integer coordinate space to avoid float64 allocation.
        node_center_ix = (node_min[0] + node_side / 2.0 - ox) / sx
        node_center_iy = (node_min[1] + node_side / 2.0 - oy) / sy
        node_center_iz = (node_min[2] + node_side / 2.0 - oz) / sz

        octant = (
            (int_x[remaining_indices] >= node_center_ix).astype(np.uint8)
            | ((int_y[remaining_indices] >= node_center_iy).astype(np.uint8) << 1)
            | ((int_z[remaining_indices] >= node_center_iz).astype(np.uint8) << 2)
        )

        # When remaining points are few enough, make children leaf nodes
        # directly (no further subdivision).  Otherwise enqueue for BFS.
        make_leaves = len(remaining_indices) <= target_points_per_node

        for direction in range(8):
            child_indices = remaining_indices[octant == direction]
            if len(child_indices) > 0:
                child_key = key.child(direction)
                if make_leaves:
                    chunks.append(OctreeChunk(key=child_key, point_indices=child_indices))
                    if child_key.level > max_depth_used:
                        max_depth_used = child_key.level
                else:
                    queue.append((child_key, child_indices))

    chunks.sort(key=lambda c: (c.key.level, c.key.x, c.key.y, c.key.z))
    spacing = side / _PDAL_CELL_COUNT

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

        # Update bounds (avoid float64 array allocation — use scalar min/max)
        if len(points) > 0:
            header.maxs = [
                float(np.max(points["X"])) * header.scales[0] + header.offsets[0],
                float(np.max(points["Y"])) * header.scales[1] + header.offsets[1],
                float(np.max(points["Z"])) * header.scales[2] + header.offsets[2],
            ]
            header.mins = [
                float(np.min(points["X"])) * header.scales[0] + header.offsets[0],
                float(np.min(points["Y"])) * header.scales[1] + header.offsets[1],
                float(np.min(points["Z"])) * header.scales[2] + header.offsets[2],
            ]

            # Update number_of_points_by_return
            header.number_of_points_by_return = [0] * len(header.number_of_points_by_return)
            return_numbers = np.asarray(points["return_number"])
            for rn in range(1, min(16, int(return_numbers.max()) + 1)):
                header.number_of_points_by_return[rn - 1] = int(np.sum(return_numbers == rn))

            # Update extra bytes min/max from actual point data
            eb_vlrs = header.vlrs.get("ExtraBytesVlr")
            if eb_vlrs:
                for eb_vlr in eb_vlrs:
                    eb_vlr.partial_reset()
                    eb_vlr.grow(points)
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
        if "gps_time" in points.point_format.dimension_names and len(points) > 0:
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

        point_size = header.point_format.size
        # View point data as 2D array (n_points, point_size) — zero-copy reshape.
        # Fancy indexing then copies only selected rows, avoiding the large
        # byte_offsets matrix (which was n_pts × point_size × 8 bytes).
        point_records_2d = np.frombuffer(points.array, dtype=np.uint8).reshape(-1, point_size)

        last_idx = len(octree.chunks) - 1
        for i, chunk in enumerate(octree.chunks):
            idx = chunk.point_indices
            n_pts = len(idx)

            # Extract point bytes: copies only n_pts × point_size uint8 bytes.
            chunk_bytes = point_records_2d[idx].ravel()

            compressor.compress_many(chunk_bytes)
            # Skip finish_current_chunk on the final chunk: done() finalizes it.
            # Calling both produces a phantom empty chunk in the LAZ chunk table
            # (n_chunks = real + 1), which breaks LASzip C++ readers (CloudCompare).
            if i != last_idx:
                compressor.finish_current_chunk()

            # Free chunk indices — no longer needed after compression.
            chunk.point_indices = None

            entry = Entry()
            entry.key = chunk.key
            entry.offset = chunk_start
            entry.point_count = n_pts
            # byte_size for non-last chunks: end position - start. Last chunk
            # is backfilled after done() from the chunk table offset.
            if i != last_idx:
                chunk_end = dest.tell()
                entry.byte_size = chunk_end - chunk_start
                chunk_start = chunk_end
            else:
                entry.byte_size = 0
            entries.append(entry)

        compressor.done()

        # Last chunk ends where the chunk table starts. lazrs wrote the chunk
        # table offset into the 8-byte placeholder at offset_to_point_data.
        if entries:
            saved_pos = dest.tell()
            dest.seek(offset_to_point_data)
            chunk_table_offset = int.from_bytes(dest.read(8), "little", signed=True)
            dest.seek(saved_pos)
            entries[-1].byte_size = chunk_table_offset - entries[-1].offset

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
