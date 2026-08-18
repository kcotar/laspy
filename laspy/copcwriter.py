import io
import logging
import math
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field

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

# Parallel per-chunk compression needs lazrs APIs added in recent versions.
_HAS_PARALLEL_CHUNKS = lazrs is not None and hasattr(
    getattr(lazrs, "ParLasZipCompressor", None), "compress_chunks"
) and hasattr(lazrs, "read_chunk_table")

# Chunks compressed per compress_chunks call. Bounds transient memory to
# ~one batch of compressed chunks while keeping all cores busy.
_PARALLEL_CHUNK_BATCH = 32

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
    # Either an index array (freshly built octree) or a slice of the
    # point array (structure reused from a source COPC file).
    point_indices: Union[np.ndarray, slice]


@dataclass
class OctreeResult:
    chunks: List[OctreeChunk]
    intermediate_keys: List[VoxelKey]  # Internal nodes that were subdivided
    center: np.ndarray
    halfsize: float
    spacing: float
    # Hierarchy entries with point_count == 0, preserved when reusing
    # the structure of a source COPC file.
    zero_keys: List[VoxelKey] = field(default_factory=list)


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


def _chunks_still_contained(
    points: PackedPointRecord,
    header: LasHeader,
    copc_info: CopcInfoVlr,
    entries: List[Entry],
    starts: np.ndarray,
) -> bool:
    """Check that every chunk's points still lie inside its node's cube.

    Attribute-only edits always pass. Coordinate edits that moved points
    out of their octree nodes make this return False, in which case the
    octree must be rebuilt.
    """
    side = 2.0 * float(copc_info.halfsize)
    root_min = np.asarray(copc_info.center, dtype=np.float64) - copc_info.halfsize

    levels = np.array([e.key.level for e in entries], dtype=np.float64)
    keys = np.array([[e.key.x, e.key.y, e.key.z] for e in entries], dtype=np.float64)
    node_side = side / np.exp2(levels)

    seg = starts[:-1].astype(np.intp)
    for axis, dim in enumerate(("X", "Y", "Z")):
        arr = np.ascontiguousarray(points[dim])
        chunk_min = np.minimum.reduceat(arr, seg).astype(np.float64)
        chunk_max = np.maximum.reduceat(arr, seg).astype(np.float64)

        scale = header.scales[axis]
        offset = header.offsets[axis]
        node_min = (root_min[axis] + keys[:, axis] * node_side - offset) / scale
        node_max = node_min + node_side / scale
        # Tolerance of one integer unit, plus slack for float rounding.
        eps = 1.0 + node_side * 1e-9 / scale
        if np.any(chunk_min < node_min - eps) or np.any(chunk_max > node_max + eps):
            return False
    return True


def _structure_from_header(
    header: LasHeader, points: PackedPointRecord
) -> Optional[OctreeResult]:
    """Recover the octree layout of the source COPC file from its header
    (CopcInfoVlr + CopcHierarchyVlr), so that an edited file can be written
    without rebuilding the octree.

    When a COPC file is read sequentially (laspy.read), points are
    decompressed chunk after chunk in file order and each LAZ chunk is one
    octree node. Hierarchy entries sorted by chunk offset therefore give
    each node its slice of the point array.

    Returns None when the structure cannot be reused: the header does not
    come from a COPC file, the hierarchy is inconsistent with the points,
    or points were moved outside their nodes.
    """
    n_points = len(points)
    if n_points == 0:
        return None

    copc_info = next((v for v in header.vlrs if isinstance(v, CopcInfoVlr)), None)
    if copc_info is None or copc_info.halfsize <= 0.0 or copc_info.spacing <= 0.0:
        return None

    hierarchy = None
    for vlr_list in (header.evlrs, header.vlrs):
        if vlr_list is None:
            continue
        hierarchy = next(
            (v for v in vlr_list if isinstance(v, CopcHierarchyVlr)), None
        )
        if hierarchy is not None:
            break
    if hierarchy is None:
        return None

    # The hierarchy EVLR payload is a concatenation of pages, and pages are
    # flat arrays of entries, so the whole payload can be parsed as entries.
    # Entries with point_count == -1 only reference a child page (whose
    # entries are in the same payload) and are skipped.
    data = hierarchy.data
    entry_size = VoxelKey.unpacker.size + Entry.unpacker.size
    if not data or len(data) % entry_size != 0:
        return None

    point_entries = []
    zero_keys = []
    seen_keys = set()
    for i in range(0, len(data), entry_size):
        entry = Entry.from_bytes(data[i : i + entry_size])
        if entry.point_count == -1:
            continue
        if entry.key in seen_keys:
            return None
        seen_keys.add(entry.key)
        if entry.point_count == 0:
            zero_keys.append(entry.key)
        else:
            point_entries.append(entry)

    if sum(e.point_count for e in point_entries) != n_points:
        return None

    point_entries.sort(key=lambda e: e.offset)
    counts = np.array([e.point_count for e in point_entries], dtype=np.int64)
    starts = np.concatenate(([0], np.cumsum(counts)))

    if not _chunks_still_contained(points, header, copc_info, point_entries, starts):
        return None

    chunks = [
        OctreeChunk(
            key=entry.key,
            point_indices=slice(int(starts[i]), int(starts[i + 1])),
        )
        for i, entry in enumerate(point_entries)
    ]

    return OctreeResult(
        chunks=chunks,
        intermediate_keys=[],
        center=np.array(copc_info.center, dtype=np.float64),
        halfsize=float(copc_info.halfsize),
        spacing=float(copc_info.spacing),
        zero_keys=zero_keys,
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
        reuse_structure: bool = True,
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
        reuse_structure : bool
            When the header comes from a COPC file (e.g. laspy.read of a
            .copc.laz) and the points still fit the source octree, reuse
            that octree instead of rebuilding it. Attribute edits keep the
            structure valid; coordinate edits that break it automatically
            fall back to a full rebuild.
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
                CopcWriter._write_copc(
                    f, header, points, target_points_per_node, max_depth, reuse_structure
                )
        else:
            CopcWriter._write_copc(
                destination, header, points, target_points_per_node, max_depth, reuse_structure
            )

    @staticmethod
    def _compress_chunks_parallel(
        dest: BinaryIO,
        laz_vlr,
        chunks: List[OctreeChunk],
        point_records_2d: np.ndarray,
        offset_to_point_data: int,
    ) -> List[Entry]:
        """Compress octree chunks on all cores via ParLasZipCompressor.

        Each buffer passed to compress_chunks becomes exactly one LAZ chunk;
        chunks are fed in batches so at most one batch of compressed chunks
        is held in memory. Per-chunk byte sizes are recovered from the chunk
        table that done() writes.
        """
        compressor = lazrs.ParLasZipCompressor(dest, laz_vlr)
        # After compressor init, it writes 8-byte chunk table offset placeholder
        # First chunk starts at offset_to_point_data + 8
        for start in range(0, len(chunks), _PARALLEL_CHUNK_BATCH):
            batch = chunks[start : start + _PARALLEL_CHUNK_BATCH]
            # Slices (reuse path) stay zero-copy views; index arrays
            # (rebuild path) copy only this batch's rows.
            compressor.compress_chunks(
                [point_records_2d[c.point_indices].ravel() for c in batch]
            )
            for chunk in batch:
                chunk.point_indices = None
        compressor.done()
        end_of_data = dest.tell()

        dest.seek(offset_to_point_data)
        chunk_table = lazrs.read_chunk_table(dest, laz_vlr)
        dest.seek(end_of_data)
        if chunk_table is None or len(chunk_table) != len(chunks):
            raise LaspyException(
                "Chunk table inconsistent after parallel COPC compression "
                f"({None if chunk_table is None else len(chunk_table)} entries "
                f"for {len(chunks)} chunks)"
            )

        entries = []
        chunk_start = offset_to_point_data + 8
        for chunk, (point_count, byte_size) in zip(chunks, chunk_table):
            entry = Entry()
            entry.key = chunk.key
            entry.offset = chunk_start
            entry.byte_size = int(byte_size)
            entry.point_count = int(point_count)
            entries.append(entry)
            chunk_start += int(byte_size)
        return entries

    @staticmethod
    def _compress_chunks_sequential(
        dest: BinaryIO,
        laz_vlr,
        chunks: List[OctreeChunk],
        point_records_2d: np.ndarray,
        offset_to_point_data: int,
    ) -> List[Entry]:
        """Single-threaded fallback for lazrs versions without compress_chunks
        (also handles the 0-point file, whose single empty chunk the parallel
        API is not exercised with)."""
        compressor = lazrs.LasZipCompressor(dest, laz_vlr)
        # After compressor init, it writes 8-byte chunk table offset placeholder
        # First chunk starts at offset_to_point_data + 8

        chunk_start = offset_to_point_data + 8
        entries = []

        last_idx = len(chunks) - 1
        for i, chunk in enumerate(chunks):
            idx = chunk.point_indices
            n_pts = (idx.stop - idx.start) if isinstance(idx, slice) else len(idx)

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
        return entries

    @staticmethod
    def _write_copc(
        dest: BinaryIO,
        header: LasHeader,
        points: PackedPointRecord,
        target_points_per_node: int,
        max_depth: Optional[int],
        reuse_structure: bool = True,
    ) -> None:
        header = deepcopy(header)
        header.are_points_compressed = True
        header.point_count = len(points)
        # Same guard as LasWriter, and it must run before the min/max grow pass
        # below reads the record back.
        header._prune_overlong_extra_bytes_vlr()
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

        # Reuse the octree structure of the source COPC file when possible,
        # otherwise build a new octree.
        octree = None
        if reuse_structure:
            octree = _structure_from_header(header, points)
        if octree is not None:
            logger.info(
                "Reusing COPC octree structure from source (%d chunks)",
                len(octree.chunks),
            )
        else:
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
        point_size = header.point_format.size
        # View point data as 2D array (n_points, point_size) — zero-copy reshape.
        # Fancy indexing then copies only selected rows, avoiding the large
        # byte_offsets matrix (which was n_pts × point_size × 8 bytes).
        point_records_2d = np.frombuffer(points.array, dtype=np.uint8).reshape(-1, point_size)

        if _HAS_PARALLEL_CHUNKS and len(points) > 0:
            entries = CopcWriter._compress_chunks_parallel(
                dest, laz_vlr, octree.chunks, point_records_2d, offset_to_point_data
            )
        else:
            entries = CopcWriter._compress_chunks_sequential(
                dest, laz_vlr, octree.chunks, point_records_2d, offset_to_point_data
            )

        # Preserve empty-node entries when reusing a source hierarchy
        for key in octree.zero_keys:
            entry = Entry()
            entry.key = key
            entries.append(entry)

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
