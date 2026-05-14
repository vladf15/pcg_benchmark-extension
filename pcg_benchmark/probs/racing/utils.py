import functools
from typing import Dict, List, Set, Tuple

import numpy as np


@functools.lru_cache(maxsize=128)
def interpolate_curves_cached(points_tuple, samples_per_segment=20):
    points = np.array(points_tuple)
    n = len(points)
    if n < 2:
        return points
    points = np.vstack([points[0], points, points[-1]])
    dim = points.shape[1]
    total_segments = n - 1
    total_points = total_segments * samples_per_segment + 1
    curve_points = np.empty((total_points, dim), dtype=points.dtype)
    idx = 0
    t_vals = np.linspace(0, 1, samples_per_segment, endpoint=False)
    t2 = t_vals ** 2
    t3 = t2 * t_vals
    h1 = 2 * t3 - 3 * t2 + 1
    h2 = -2 * t3 + 3 * t2
    h3 = t3 - 2 * t2 + t_vals
    h4 = t3 - t2
    for i in range(1, n):
        p0, p1, p2, p3 = points[i-1], points[i], points[i+1], points[i+2]
        # The benchmark uses neutral spline parameters, which reduces to
        # Catmull-Rom-style tangents:
        #   m1 = 0.5 * (p2 - p0)
        #   m2 = 0.5 * (p3 - p1)
        m1 = 0.5 * (p2 - p0)
        m2 = 0.5 * (p3 - p1)
        pts = (
            h1[:, None] * p1 +
            h2[:, None] * p2 +
            h3[:, None] * m1 +
            h4[:, None] * m2
        )
        curve_points[idx:idx+samples_per_segment] = pts
        idx += samples_per_segment
    curve_points[-1] = points[-1]
    return curve_points


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
    t_vals = np.linspace(0, 1, samples_per_segment, endpoint=False)
    t2 = t_vals ** 2
    t3 = t2 * t_vals
    h1 = 2 * t3 - 3 * t2 + 1
    h2 = -2 * t3 + 3 * t2
    h3 = t3 - 2 * t2 + t_vals
    h4 = t3 - t2

    for i in range(1, n + 1):
        p0, p1, p2, p3 = pad[i - 1], pad[i], pad[i + 1], pad[i + 2]
        # Neutral spline parameters (Catmull-Rom-style tangents).
        m1 = 0.5 * (p2 - p0)
        m2 = 0.5 * (p3 - p1)
        pts = (
            h1[:, None] * p1 +
            h2[:, None] * p2 +
            h3[:, None] * m1 +
            h4[:, None] * m2
        )
        curve_points[idx:idx + samples_per_segment] = pts
        idx += samples_per_segment

    curve_points[-1] = points[0]

    # Place the seam at a low-curvature location to avoid a visible kink.
    poly = curve_points[:-1]
    m = len(poly)
    if m >= 4:
        best_i = 0
        best_ang = float('inf')
        for i in range(m):
            p_prev = poly[(i - 1) % m]
            p = poly[i]
            p_next = poly[(i + 1) % m]
            v1 = p - p_prev
            v2 = p_next - p
            n1 = float(np.linalg.norm(v1))
            n2 = float(np.linalg.norm(v2))
            if n1 < 1e-6 or n2 < 1e-6:
                continue
            d = float(np.dot(v1, v2) / (n1 * n2))
            d = -1.0 if d < -1.0 else (1.0 if d > 1.0 else d)
            ang = float(np.arccos(d))
            if ang < best_ang:
                best_ang = ang
                best_i = i
        if best_i != 0:
            poly = np.vstack([poly[best_i:], poly[:best_i]])
            curve_points[:-1] = poly
            curve_points[-1] = poly[0]

    return curve_points


def interpolate_curves(points, samples_per_segment=20, *, closed: bool = False):
    """Interpolate control points into a Kochanek-Bartels spline polyline."""
    pts = np.asarray(points)
    if closed and len(pts) >= 3:
        # If the loop is already explicitly closed (last==first), drop the
        # duplicate control point for interpolation and re-close at the end.
        if np.allclose(pts[0], pts[-1], atol=1e-9, rtol=0.0):
            pts = pts[:-1]

    points_tuple = tuple(map(tuple, np.asarray(pts)))
    if closed:
        return interpolate_curves_closed_cached(points_tuple, samples_per_segment)
    return interpolate_curves_cached(points_tuple, samples_per_segment)


@functools.lru_cache(maxsize=128)
def count_self_intersections_cached(points_tuple, closed: bool):
    points = np.array(points_tuple)
    return _count_self_intersections_grid(points, closed=bool(closed))


def count_self_intersections(points, *, closed: bool = False):
    points_tuple = tuple(map(tuple, np.asarray(points)))
    return count_self_intersections_cached(points_tuple, bool(closed))


def _segment_to_segment_distance_sq(p1, p2, p3, p4) -> float:
    """Return squared minimum distance between 2D segments p1-p2 and p3-p4."""

    def point_to_segment_distance_sq(p, a, b) -> float:
        abx = float(b[0] - a[0])
        aby = float(b[1] - a[1])
        apx = float(p[0] - a[0])
        apy = float(p[1] - a[1])
        ab2 = abx * abx + aby * aby
        if ab2 < 1e-18:
            return apx * apx + apy * apy
        t = (apx * abx + apy * aby) / ab2
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        cx = float(a[0]) + t * abx
        cy = float(a[1]) + t * aby
        dx = float(p[0]) - cx
        dy = float(p[1]) - cy
        return dx * dx + dy * dy

    d1 = point_to_segment_distance_sq(p1, p3, p4)
    d2 = point_to_segment_distance_sq(p2, p3, p4)
    d3 = point_to_segment_distance_sq(p3, p1, p2)
    d4 = point_to_segment_distance_sq(p4, p1, p2)
    return min(d1, d2, d3, d4)


@functools.lru_cache(maxsize=128)
def count_track_width_overlaps_cached(points_tuple, track_width: float, min_arclen_gap: float):
    pts = np.asarray(points_tuple, dtype=float)
    return _count_track_width_overlaps_grid(pts, float(track_width), float(min_arclen_gap))


def count_track_width_overlaps(points, track_width: float, min_arclen_gap: float | None = None) -> int:
    """Count self-overlaps of the *track area* implied by a centerline + width.

    Even if the centerline polyline does not intersect, the track with width can
    overlap when two far-apart parts of the centerline come within < track_width.

    Args:
        points: (N,2) polyline points (typically interpolated curve_points)
        track_width: full width of the road
        min_arclen_gap: ignore segment pairs that are too close along the path
            (prevents false positives from local neighbors in a densely-sampled polyline).

    Returns:
        Number of overlapping non-local segment pairs.
    """
    pts = np.asarray(points, dtype=float)
    if min_arclen_gap is None:
        # Ignore local neighbors along the path (dense sampling would otherwise trip this).
        min_arclen_gap = max(float(track_width) * 2.0, 1.0)
    points_tuple = tuple(map(tuple, pts))
    return int(count_track_width_overlaps_cached(points_tuple, float(track_width), float(min_arclen_gap)))


def _count_segment_vs_polyline(
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    poly_points: np.ndarray,
    *,
    eps: float,
    skip_first_segment: bool = False,
    skip_last_segment: bool = False,
) -> int:
    poly_points = np.asarray(poly_points, dtype=float)
    if len(poly_points) < 2:
        return 0

    start = poly_points[:-1]
    end = poly_points[1:]
    count = 0
    for i in range(len(start)):
        if skip_first_segment and i == 0:
            continue
        if skip_last_segment and i == (len(start) - 1):
            continue
        if _segments_intersect_inclusive(seg_a, seg_b, start[i], end[i], eps=eps):
            count += 1
    return count


def _count_track_endcap_intersections(left_edge: np.ndarray, right_edge: np.ndarray) -> int:
    """Count intersections caused by the *end caps* of the rendered road polygon.

    The renderer fills a polygon formed by `left_edge + right_edge[::-1]`.
    This implicitly adds two cap segments:
    - end cap: left_edge[-1] -> right_edge[-1]
    - start cap: right_edge[0] -> left_edge[0]

    Intersections involving these cap segments were previously not checked,
    which can lead to visible self-intersections and sharp artifacts.
    """
    left_edge = np.asarray(left_edge, dtype=float)
    right_edge = np.asarray(right_edge, dtype=float)
    if len(left_edge) < 2 or len(right_edge) < 2:
        return 0

    lens = []
    for edge in (left_edge, right_edge):
        seg_len = np.linalg.norm(edge[1:] - edge[:-1], axis=1)
        seg_len = seg_len[seg_len > 1e-12]
        if seg_len.size:
            lens.append(float(np.median(seg_len)))
    base = float(np.median(lens)) if lens else 1.0
    eps = max(1e-9, base * 1e-6)

    cap_end_a = left_edge[-1]
    cap_end_b = right_edge[-1]
    cap_start_a = right_edge[0]
    cap_start_b = left_edge[0]

    count = 0

    # Skip adjacent boundary segments (shared endpoints).
    count += _count_segment_vs_polyline(cap_end_a, cap_end_b, left_edge, eps=eps, skip_last_segment=True)
    count += _count_segment_vs_polyline(cap_end_a, cap_end_b, right_edge, eps=eps, skip_last_segment=True)
    count += _count_segment_vs_polyline(cap_start_a, cap_start_b, left_edge, eps=eps, skip_first_segment=True)
    count += _count_segment_vs_polyline(cap_start_a, cap_start_b, right_edge, eps=eps, skip_first_segment=True)

    if _segments_intersect_inclusive(cap_start_a, cap_start_b, cap_end_a, cap_end_b, eps=eps):
        count += 1

    return int(count)


def _compute_offset_edges(points: np.ndarray, track_width: float, *, closed: bool = False):
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
    if closed and n >= 3 and np.allclose(curve[0], curve[-1], atol=1e-9, rtol=0.0):
        has_dup_close = True
        base = curve[:-1]

    m = len(base)
    if m < 2:
        return np.zeros((0, 2), dtype=float), np.zeros((0, 2), dtype=float)

    half_width = 0.5 * float(track_width)
    left_edge = np.zeros((m, 2), dtype=float)
    right_edge = np.zeros((m, 2), dtype=float)

    for j in range(m):
        if closed and m >= 3:
            prev_i = (j - 1) % m
            next_i = (j + 1) % m
            dir_prev = base[j] - base[prev_i]
            dir_next = base[next_i] - base[j]
        else:
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
    closed: bool = False,
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
                        if closed and ma == mb:
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
    closed: bool = False,
) -> int:
    """Count intersections in the rendered track area (accounts for width).

    We approximate the road boundaries as left/right offset polylines and flag:
    - left edge self-intersections
    - right edge self-intersections
    - left/right edge crossovers
    """
    left_edge, right_edge = _compute_offset_edges(points, track_width=float(track_width), closed=bool(closed))
    left_self = count_self_intersections(left_edge, closed=bool(closed))
    if left_self > 0:
        return int(left_self)
    right_self = count_self_intersections(right_edge, closed=bool(closed))
    if right_self > 0:
        return int(right_self)
    # When comparing left vs right boundaries, we only care about crossings
    # between *non-adjacent* parts of the track. Local closeness is expected.
    cross = _count_polyline_crossings_grid(
        left_edge,
        right_edge,
        min_index_gap=int(min_cross_index_gap),
        closed=bool(closed),
    )
    if cross > 0:
        return int(cross)

    if closed:
        # Closed loops have no start/end caps.
        return 0

    # Also validate the implicit start/end caps of the rendered polygon.
    cap_inters = _count_track_endcap_intersections(left_edge, right_edge)
    return int(cap_inters)


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


def _count_self_intersections_grid(points: np.ndarray, *, closed: bool = False) -> int:
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
                if closed and i == 0 and j == (m - 1):
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


def _count_track_width_overlaps_grid(points: np.ndarray, track_width: float, min_arclen_gap: float) -> int:
    """Count non-local segment pairs whose distance is <= track_width.

    This approximates overlap of a constant-width track as the union of capsules
    around the centerline segments. Two capsules overlap when the centerlines are
    within track_width.
    """
    points = np.asarray(points, dtype=float)
    n = len(points)
    if n < 4:
        return 0

    seg_start = points[:-1]
    seg_end = points[1:]
    m = n - 1

    seg_vec = seg_end - seg_start
    seg_len = np.linalg.norm(seg_vec, axis=1)
    # Skip a *number of segments* proportional to sampling resolution.

    # Expanded bounding boxes (by radius=track_width/2) for grid pruning.
    radius = 0.5 * float(track_width)
    seg_min = np.minimum(seg_start, seg_end) - radius
    seg_max = np.maximum(seg_start, seg_end) + radius

    nonzero = seg_len > 1e-12
    if np.any(nonzero):
        seg_len_nz = seg_len[nonzero]
        median_seg_len = float(np.median(seg_len_nz))
        # Use a lower percentile as reference so the local skip remains
        # conservative even if some segments are much shorter than typical.
        seg_len_ref = float(np.percentile(seg_len_nz, 25))
        cell_size = max(1e-6, max(median_seg_len * 2.0, float(track_width)))
    else:
        seg_len_ref = 1.0
        median_seg_len = 1.0
        cell_size = max(1.0, float(track_width))

    local_index_gap = int(np.ceil(float(min_arclen_gap) / max(1e-12, seg_len_ref)))
    local_index_gap = max(local_index_gap, 2)

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
                candidate_pairs.add((i, j))

    thresh2 = float(track_width) * float(track_width)
    count = 0
    for i, j in candidate_pairs:
        # Ignore local neighbors along the path (dense sampling would otherwise trip this).
        # This is expressed in *index space* so it scales naturally with the
        # sampling resolution.
        if (j - i) <= local_index_gap:
            continue

        # Early bbox rejection (expanded bboxes already include radius, but keep fast checks).
        if seg_max[i, 0] < seg_min[j, 0] or seg_max[j, 0] < seg_min[i, 0]:
            continue
        if seg_max[i, 1] < seg_min[j, 1] or seg_max[j, 1] < seg_min[i, 1]:
            continue

        d2 = _segment_to_segment_distance_sq(seg_start[i], seg_end[i], seg_start[j], seg_end[j])
        if d2 <= thresh2:
            count += 1

    return count
