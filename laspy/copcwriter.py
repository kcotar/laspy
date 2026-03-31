import io
import logging
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from math import ceil, log2
from typing import BinaryIO, List, Optional, Union

import numpy as np

from .copc import CopcInfoVlr, Entry, VoxelKey
from .errors import LaspyException
from .header import LasHeader
from .point.record import PackedPointRecord
from .vlrs.known import LasZipVlr

try:
    import lazrs
except ModuleNotFoundError:
    lazrs = None

logger = logging.getLogger(__name__)


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

    center = (mins + maxs) / 2.0
    halfsize = float(np.max(maxs - mins)) / 2.0

    if halfsize == 0.0:
        halfsize = 1.0

    n_points = len(points)
    if max_depth is None:
        if n_points <= target_points_per_node:
            max_depth = 0
        else:
            max_depth = min(20, max(1, int(ceil(log2(n_points / target_points_per_node) / 3)) + 1))

    # BFS octree construction with LOD subsampling.
    # At each internal node, keep a subsample of points for that LOD level
    # and push the rest to child nodes.
    root_key = VoxelKey.from_values(0, 0, 0, 0)
    all_indices = np.arange(n_points, dtype=np.intp)

    chunks = []
    queue = deque()
    queue.append((root_key, all_indices))

    # Root bounds: center ± halfsize (cube)
    root_mins = center - halfsize
    root_maxs = center + halfsize

    max_depth_used = 0

    while queue:
        key, indices = queue.popleft()

        if len(indices) <= target_points_per_node or key.level >= max_depth:
            # Leaf node: store all remaining points
            chunks.append(OctreeChunk(key=key, point_indices=indices))
            if key.level > max_depth_used:
                max_depth_used = key.level
            continue

        # Internal node: subsample points for this LOD level.
        # Keep every 8th point at this level, push the rest to children.
        # This gives a uniform subsampling that works well for LOD.
        np.random.seed(key.level * 1000 + key.x * 100 + key.y * 10 + key.z)
        perm = np.random.permutation(len(indices))
        n_keep = max(1, len(indices) // 8)
        keep_mask = np.zeros(len(indices), dtype=bool)
        keep_mask[perm[:n_keep]] = True

        node_indices = indices[keep_mask]
        remaining_indices = indices[~keep_mask]

        # Store the subsampled points at this node
        chunks.append(OctreeChunk(key=key, point_indices=node_indices))

        # Subdivide remaining points into 8 children
        side_size = (root_maxs[0] - root_mins[0]) / (2 ** key.level)
        node_mins = root_mins + np.array([key.x, key.y, key.z], dtype=np.float64) * side_size
        node_center = node_mins + side_size / 2.0

        px = actual_x[remaining_indices]
        py = actual_y[remaining_indices]
        pz = actual_z[remaining_indices]

        octant = (
            (px >= node_center[0]).astype(np.uint8)
            | ((py >= node_center[1]).astype(np.uint8) << 1)
            | ((pz >= node_center[2]).astype(np.uint8) << 2)
        )

        for direction in range(8):
            mask = octant == direction
            child_indices = remaining_indices[mask]
            if len(child_indices) > 0:
                child_key = key.child(direction)
                queue.append((child_key, child_indices))

    # Sort chunks by level (breadth-first order)
    chunks.sort(key=lambda c: (c.key.level, c.key.x, c.key.y, c.key.z))

    spacing = (halfsize * 2.0) / (2 ** max_depth_used) if max_depth_used > 0 else halfsize * 2.0

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

        # Write hierarchy page
        hierarchy_offset = dest.tell()
        for entry in entries:
            dest.write(entry.to_bytes())
        hierarchy_size = dest.tell() - hierarchy_offset

        # Update CopcInfoVlr with actual hierarchy location
        copc_info.hierarchy_root_offset = hierarchy_offset
        copc_info.hierarchy_root_size = hierarchy_size

        # Rewrite header with updated CopcInfoVlr
        dest.seek(0)
        header.write_to(dest, ensure_same_size=True)
