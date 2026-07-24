"""Geometry helpers shared by all racing problems.

Three groups of functions:
- Spline interpolation (`interpolate_curves`): turn sparse control points
  into the dense closed Kochanek-Bartels curve that simulation, scoring,
  and rendering all run on.
- Self-intersection counting (`count_self_intersections`,
  `count_track_area_intersections`): the geometry soundness checks behind
  quality()'s geom_score.  Both use a uniform grid to prune segment pairs,
  so they stay near-linear in the number of segments.
- `lowest_turn_seam_index`: picks the smoothest vertex of a closed loop so
  the start/finish seam never sits in the middle of a corner.
"""
import functools
from typing import Dict, List, Set, Tuple

import numpy as np


def _points_as_tuples(points):
    """Convert an (N, 2) array to a tuple of (x, y) tuples.

    functools.lru_cache needs hashable arguments; arrays are not hashable,
    nested tuples are."""
    rows = []
    for p in np.asarray(points):
        rows.append(tuple(p))
    return tuple(rows)


def lowest_turn_seam_index(points) -> int:
    """Index of the vertex with the smallest local turn angle.

    Used to place the seam of a closed loop at its smoothest spot, so the
    start/end joint does not sit in the middle of a corner.
    """
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    best_i = 0
    best_ang = float('inf')
    for i in range(n):
        v1 = pts[i] - pts[(i - 1) % n]
        v2 = pts[(i + 1) % n] - pts[i]
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        d = float(np.dot(v1, v2) / (n1 * n2))
        d = max(-1.0, min(1.0, d))
        ang = float(np.arccos(d))
        if ang < best_ang:
            best_ang = ang
            best_i = i
    return best_i


@functools.lru_cache(maxsize=128)
def interpolate_curves_closed_cached(points_tuple, samples_per_segment=20):
    """Closed (periodic) Kochanek-Bartels spline polyline.

    Returns an explicitly closed polyline (last point equals first).
    """
    points = np.array(points_tuple)
    n = len(points)
    if n < 2:
        return points

    # Pad for wrap-around tangents.
    pad = np.vstack([points[-1], points, points[0], points[1] if n > 1 else points[0]])
    dim = points.shape[1]
    total_segments = n  # includes last->first
    total_points = total_segments * samples_per_segment + 1
    curve_points = np.empty((total_points, dim), dtype=points.dtype)

    idx = 0
    # Cubic Hermite basis functions, evaluated once for every sample position
    # t in [0, 1).  Reshaped to columns so that (basis column) * (point row)
    # gives one curve sample per row.
    t_vals = np.linspace(0, 1, samples_per_segment, endpoint=False)
    t2 = t_vals ** 2
    t3 = t2 * t_vals
    h1 = (2 * t3 - 3 * t2 + 1).reshape(-1, 1)
    h2 = (-2 * t3 + 3 * t2).reshape(-1, 1)
    h3 = (t3 - 2 * t2 + t_vals).reshape(-1, 1)
    h4 = (t3 - t2).reshape(-1, 1)

    for i in range(1, n + 1):
        p0, p1, p2, p3 = pad[i - 1], pad[i], pad[i + 1], pad[i + 2]
        # Neutral spline parameters (Catmull-Rom-style tangents).
        m1 = 0.5 * (p2 - p0)
        m2 = 0.5 * (p3 - p1)
        pts = h1 * p1 + h2 * p2 + h3 * m1 + h4 * m2
        curve_points[idx:idx + samples_per_segment] = pts
        idx += samples_per_segment

    curve_points[-1] = points[0]

    # Place the seam at a low-curvature location to avoid a visible kink.
    poly = curve_points[:-1]
    if len(poly) >= 4:
        best_i = lowest_turn_seam_index(poly)
        if best_i != 0:
            poly = np.vstack([poly[best_i:], poly[:best_i]])
            curve_points[:-1] = poly
            curve_points[-1] = poly[0]

    return curve_points


def interpolate_curves(points, samples_per_segment=20):
    """Interpolate control points into a Kochanek-Bartels spline polyline.

    Racing tracks are closed circuits, so the spline is always periodic."""
    pts = np.asarray(points)
    if len(pts) >= 3:
        # If the loop is already explicitly closed (last==first), drop the
        # duplicate control point for interpolation and re-close at the end.
        if np.allclose(pts[0], pts[-1], atol=1e-9, rtol=0.0):
            pts = pts[:-1]

    return interpolate_curves_closed_cached(_points_as_tuples(pts), samples_per_segment)


@functools.lru_cache(maxsize=128)
def count_self_intersections_cached(points_tuple):
    points = np.array(points_tuple)
    return _count_self_intersections_grid(points)


def count_self_intersections(points):
    points_tuple = _points_as_tuples(points)
    return count_self_intersections_cached(points_tuple)


def _compute_offset_edges(points: np.ndarray, track_width: float):
    """Compute left/right offset polylines like the renderer does.

    This approximates the track polygon used for rendering and is a good proxy
    for detecting width-based self-intersections.
    """
    curve = np.asarray(points, dtype=float)
    n = len(curve)
    if n < 2:
        return np.zeros((0, 2), dtype=float), np.zeros((0, 2), dtype=float)

    # If explicitly closed (last==first), compute offsets on unique points and
    # then re-close. This avoids degenerate direction vectors at the seam.
    has_dup_close = False
    base = curve
    if n >= 3 and np.allclose(curve[0], curve[-1], atol=1e-9, rtol=0.0):
        has_dup_close = True
        base = curve[:-1]

    m = len(base)
    if m < 2:
        return np.zeros((0, 2), dtype=float), np.zeros((0, 2), dtype=float)

    half_width = 0.5 * float(track_width)
    left_edge = np.zeros((m, 2), dtype=float)
    right_edge = np.zeros((m, 2), dtype=float)

    for j in range(m):
        if m >= 3:
            prev_i = (j - 1) % m
            next_i = (j + 1) % m
            dir_prev = base[j] - base[prev_i]
            dir_next = base[next_i] - base[j]
        else:
            # Degenerate two-point "track": treat it as a straight segment.
            if j == 0:
                dir_prev = base[1] - base[0]
            else:
                dir_prev = base[j] - base[j - 1]
            if j == m - 1:
                dir_next = base[j] - base[j - 1]
            else:
                dir_next = base[j + 1] - base[j]

        avg_dir = dir_prev + dir_next
        norm = float(np.linalg.norm(avg_dir))
        if norm < 1e-12:
            perp = np.array([0.0, 0.0], dtype=float)
        else:
            perp = np.array([-avg_dir[1], avg_dir[0]], dtype=float) / norm

        left_edge[j] = base[j] + perp * half_width
        right_edge[j] = base[j] - perp * half_width

    if has_dup_close:
        left_edge = np.vstack([left_edge, left_edge[0]])
        right_edge = np.vstack([right_edge, right_edge[0]])

    return left_edge, right_edge


def _count_polyline_crossings_grid(
    a_points: np.ndarray,
    b_points: np.ndarray,
    *,
    min_index_gap: int = 0,
) -> int:
    """Count segment intersections between two open polylines.

    Args:
        a_points: (Na,2) points for polyline A
        b_points: (Nb,2) points for polyline B
        min_index_gap: if both polylines represent the *same parameterization*
            (e.g. left/right track boundaries sampled at the same indices),
            skip checking segment pairs (i,j) with |i-j| <= min_index_gap.
            This avoids counting inevitable local proximity as "intersection".
    """
    a_points = np.asarray(a_points, dtype=float)
    b_points = np.asarray(b_points, dtype=float)
    if len(a_points) < 2 or len(b_points) < 2:
        return 0

    a0 = a_points[:-1]
    a1 = a_points[1:]
    b0 = b_points[:-1]
    b1 = b_points[1:]

    ma = len(a0)
    mb = len(b0)

    a_min = np.minimum(a0, a1)
    a_max = np.maximum(a0, a1)
    b_min = np.minimum(b0, b1)
    b_max = np.maximum(b0, b1)

    # Grid cell size based on typical segment length
    a_len = np.linalg.norm(a1 - a0, axis=1)
    b_len = np.linalg.norm(b1 - b0, axis=1)
    lens = np.concatenate([a_len[a_len > 1e-12], b_len[b_len > 1e-12]])
    if lens.size > 0:
        base = float(np.median(lens))
        cell_size = max(1e-6, base * 2.0)
    else:
        cell_size = 1.0
    # Same degenerate-curve guard as _count_self_intersections_grid: never
    # let one segment span more than ~256 cells of the overall extent.
    hi = np.maximum(np.max(a_points, axis=0), np.max(b_points, axis=0))
    lo = np.minimum(np.min(a_points, axis=0), np.min(b_points, axis=0))
    span = float(np.max(hi - lo))
    cell_size = max(cell_size, span / 256.0, 1e-6)

    origin = np.minimum(np.min(a_points, axis=0), np.min(b_points, axis=0))
    eps = max(1e-12, cell_size * 1e-9)

    def _cells(seg_min, seg_max):
        gx0 = np.floor((seg_min[:, 0] - origin[0] - eps) / cell_size).astype(np.int64)
        gx1 = np.floor((seg_max[:, 0] - origin[0] + eps) / cell_size).astype(np.int64)
        gy0 = np.floor((seg_min[:, 1] - origin[1] - eps) / cell_size).astype(np.int64)
        gy1 = np.floor((seg_max[:, 1] - origin[1] + eps) / cell_size).astype(np.int64)
        return gx0, gx1, gy0, gy1

    agx0, agx1, agy0, agy1 = _cells(a_min, a_max)
    bgx0, bgx1, bgy0, bgy1 = _cells(b_min, b_max)

    cell_to_a: Dict[Tuple[int, int], List[int]] = {}
    for i in range(ma):
        for gx in range(int(agx0[i]), int(agx1[i]) + 1):
            for gy in range(int(agy0[i]), int(agy1[i]) + 1):
                key = (gx, gy)
                if key in cell_to_a:
                    cell_to_a[key].append(i)
                else:
                    cell_to_a[key] = [i]

    count = 0
    for j in range(mb):
        for gx in range(int(bgx0[j]), int(bgx1[j]) + 1):
            for gy in range(int(bgy0[j]), int(bgy1[j]) + 1):
                key = (gx, gy)
                a_idxs = cell_to_a.get(key)
                if not a_idxs:
                    continue
                for i in a_idxs:
                    if min_index_gap > 0:
                        d = abs(int(i) - int(j))
                        if ma == mb:
                            # Same parameterization on a loop: measure the
                            # index gap around the seam as well.
                            d = min(d, ma - d)
                        if d <= int(min_index_gap):
                            continue
                    # bbox rejection
                    if a_max[i, 0] < b_min[j, 0] or b_max[j, 0] < a_min[i, 0]:
                        continue
                    if a_max[i, 1] < b_min[j, 1] or b_max[j, 1] < a_min[i, 1]:
                        continue
                    eps_i = max(1e-9, cell_size * 1e-6)
                    if _segments_intersect_inclusive(a0[i], a1[i], b0[j], b1[j], eps=eps_i):
                        count += 1
    return count


def count_track_area_intersections(
    points: np.ndarray,
    track_width: float,
    *,
    min_cross_index_gap: int = 2,
) -> int:
    """Count intersections in the rendered track area (accounts for width).

    We approximate the road boundaries as left/right offset polylines and flag:
    - left edge self-intersections
    - right edge self-intersections
    - left/right edge crossovers
    """
    left_edge, right_edge = _compute_offset_edges(points, track_width=float(track_width))
    left_self = count_self_intersections(left_edge)
    if left_self > 0:
        return int(left_self)
    right_self = count_self_intersections(right_edge)
    if right_self > 0:
        return int(right_self)
    # When comparing left vs right boundaries, we only care about crossings
    # between *non-adjacent* parts of the track. Local closeness is expected.
    cross = _count_polyline_crossings_grid(
        left_edge,
        right_edge,
        min_index_gap=int(min_cross_index_gap),
    )
    return int(cross)


def _segments_intersect_inclusive(a1, a2, b1, b2, eps: float = 1e-9) -> bool:
    """Inclusive segment intersection test.

    Counts proper crossings *and* endpoint/collinear touches (within eps).
    This is intentionally stricter than the classic CCW-only test so that
    self-intersecting or self-touching polylines are rejected.
    """

    def orient(p, q, r) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def on_segment(p, q, r) -> bool:
        # r is on segment pq (inclusive), assuming collinear within eps.
        return (
            min(p[0], q[0]) - eps <= r[0] <= max(p[0], q[0]) + eps
            and min(p[1], q[1]) - eps <= r[1] <= max(p[1], q[1]) + eps
        )

    o1 = orient(a1, a2, b1)
    o2 = orient(a1, a2, b2)
    o3 = orient(b1, b2, a1)
    o4 = orient(b1, b2, a2)

    # Proper intersection
    if (o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps):
        if (o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps):
            return True

    # Collinear / endpoint touches
    if abs(o1) <= eps and on_segment(a1, a2, b1):
        return True
    if abs(o2) <= eps and on_segment(a1, a2, b2):
        return True
    if abs(o3) <= eps and on_segment(b1, b2, a1):
        return True
    if abs(o4) <= eps and on_segment(b1, b2, a2):
        return True

    return False


def _count_self_intersections_grid(points: np.ndarray) -> int:
    """Count self-intersections using a grid to prune segment pairs."""

    points = np.asarray(points)
    n = len(points)
    if n < 4:
        return 0

    seg_start = points[:-1]
    seg_end = points[1:]
    m = n - 1

    seg_min = np.minimum(seg_start, seg_end)
    seg_max = np.maximum(seg_start, seg_end)

    seg_vec = seg_end - seg_start
    seg_len = np.linalg.norm(seg_vec, axis=1)
    nonzero = seg_len > 1e-12
    if np.any(nonzero):
        base = float(np.median(seg_len[nonzero]))
        cell_size = max(1e-9, base * 2.0)
    else:
        cell_size = 1.0
    # Floor the cell size against the overall extent: on degenerate curves
    # (nearly identical points) the median segment length approaches zero
    # and a single normal-length segment would otherwise span billions of
    # grid cells, exhausting memory.
    span = float(np.max(np.max(points, axis=0) - np.min(points, axis=0)))
    cell_size = max(cell_size, span / 256.0, 1e-9)

    origin = np.min(points, axis=0)
    eps = max(1e-12, cell_size * 1e-9)

    gx0 = np.floor((seg_min[:, 0] - origin[0] - eps) / cell_size).astype(np.int64)
    gx1 = np.floor((seg_max[:, 0] - origin[0] + eps) / cell_size).astype(np.int64)
    gy0 = np.floor((seg_min[:, 1] - origin[1] - eps) / cell_size).astype(np.int64)
    gy1 = np.floor((seg_max[:, 1] - origin[1] + eps) / cell_size).astype(np.int64)

    cell_to_segments: Dict[Tuple[int, int], List[int]] = {}
    for i in range(m):
        for gx in range(int(gx0[i]), int(gx1[i]) + 1):
            for gy in range(int(gy0[i]), int(gy1[i]) + 1):
                key = (gx, gy)
                if key in cell_to_segments:
                    cell_to_segments[key].append(i)
                else:
                    cell_to_segments[key] = [i]

    candidate_pairs: Set[Tuple[int, int]] = set()
    for segs in cell_to_segments.values():
        if len(segs) < 2:
            continue
        segs_sorted = sorted(segs)
        for a_idx in range(len(segs_sorted) - 1):
            i = segs_sorted[a_idx]
            for b_idx in range(a_idx + 1, len(segs_sorted)):
                j = segs_sorted[b_idx]
                if j - i <= 1:
                    continue
                if i == 0 and j == (m - 1):
                    # First and last segments share the loop seam endpoint.
                    continue
                candidate_pairs.add((i, j))

    count = 0
    for i, j in candidate_pairs:
        if seg_max[i, 0] < seg_min[j, 0] or seg_max[j, 0] < seg_min[i, 0]:
            continue
        if seg_max[i, 1] < seg_min[j, 1] or seg_max[j, 1] < seg_min[i, 1]:
            continue

        # Use an inclusive intersection test so self-touching is rejected too.
        # eps is scaled to the cell size used by the grid to remain robust.
        eps_i = max(1e-9, cell_size * 1e-6)
        if _segments_intersect_inclusive(seg_start[i], seg_end[i], seg_start[j], seg_end[j], eps=eps_i):
            count += 1

    return count
