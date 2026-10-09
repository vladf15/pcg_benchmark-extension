"""Geometry helpers shared by all racing problems.

Three groups of functions:
- Spline interpolation (`interpolate_curves`): turn sparse control points
  into the dense closed centripetal Catmull-Rom curve that simulation, scoring,
  and rendering all run on.
- Self-intersection counting (`count_self_intersections`,
  `count_track_area_intersections`): the geometry soundness checks behind
  the self-overlap gate in quality().  Both use a uniform grid to prune segment pairs,
  so they stay near-linear in the number of segments.
- Turning functions (`turning_function`, `turning_distance`): the shape
  comparison behind diversity().
"""
import functools
from typing import Dict, List, Set, Tuple

import numpy as np


def _points_as_tuples(points):
    """Convert an (N, 2) array to a tuple of (x, y) tuples.

    functools.lru_cache needs hashable arguments; arrays are not hashable,
    nested tuples are."""
    return tuple(tuple(p) for p in np.asarray(points))


def turning_function(curve, samples=128):
    """Turning function of a closed curve (Arkin et al. 1991).

    The heading of the curve as a function of arc length, with arc length
    normalised to [0, 1) and the heading unwrapped, so it rises by 2*pi per
    lap.  Returns (headings at `samples` evenly spaced arc positions,
    winding number), or None for a curve with fewer than three distinct
    points.  The function is a step function between the curve's vertices,
    as in the paper.

    The loop is traced anticlockwise.  A decoder's choice of direction is not
    part of the layout, and left unoriented a track against itself traced
    backwards measured a distance of 1.07 (in the pi units diversity() uses),
    more than any pair of distinct reference circuits oriented the same way
    (max 0.96).  Oriented, it measures 0.024.

    128 samples: on the 24 reference circuits the p10, median and p90
    pairwise distances are 0.293 / 0.382 / 0.474 at 128, 0.293 / 0.382 /
    0.475 at 256 and 0.295 / 0.387 / 0.477 at 64, and a comparison costs
    0.10 ms at 128 against 0.32 ms at 256."""
    c = np.asarray(curve, dtype=float).reshape(-1, 2)
    if len(c) > 1 and np.allclose(c[0], c[-1]):
        c = c[:-1]
    seg = np.roll(c, -1, axis=0) - c
    seg_len = np.linalg.norm(seg, axis=1)
    keep = seg_len > 1e-9
    if np.count_nonzero(keep) < 3:
        return None
    seg, seg_len = seg[keep], seg_len[keep]
    heading = np.arctan2(seg[:, 1], seg[:, 0])
    turn = np.angle(np.exp(1j * (np.roll(heading, -1) - heading)))
    winding = int(round(float(np.sum(turn)) / (2.0 * np.pi)))
    if winding < 0:
        return turning_function(c[::-1], samples)
    theta = heading[0] + np.concatenate([[0.0], np.cumsum(turn[:-1])])
    start = np.concatenate([[0.0], np.cumsum(seg_len)[:-1]]) / float(np.sum(seg_len))
    at = np.arange(samples) / float(samples)
    return theta[np.searchsorted(start, at, side="right") - 1], winding


def turning_distance(tf1, tf2):
    """Arkin et al.'s (1991) L2 distance between two turning functions, in
    radians: the root-mean-square heading difference, minimised over the
    start point of one curve (every cyclic shift of the samples) and over a
    rotation (a constant heading offset, whose optimum is the mean
    difference, so the residual is the standard deviation).  The distance
    is unchanged by translating, rotating or scaling either curve."""
    h1, w1 = tf1
    h2, w2 = tf2
    n = len(h1)
    idx = np.arange(n)[:, None] + np.arange(n)[None, :]
    # Shifting past the end of the lap continues into the next lap, which
    # sits 2*pi*winding higher.
    diff = h1[idx % n] + 2.0 * np.pi * w1 * (idx >= n) - h2[None, :]
    var = np.mean(diff * diff, axis=1) - np.mean(diff, axis=1) ** 2
    return float(np.sqrt(max(float(np.min(var)), 0.0)))


@functools.lru_cache(maxsize=128)
def interpolate_curves_closed_cached(points_tuple, samples_per_segment=20):
    """Closed (periodic) centripetal Catmull-Rom spline polyline.

    Each segment p1 -> p2 is a cubic Hermite curve.  Its end tangents are the
    Catmull-Rom ones with the knot spacing of each chord set to the square
    root of the chord length (centripetal parameterization, Yuksel, Schaefer
    and Keyser 2011).  With equal spacing (uniform Catmull-Rom) a segment
    next to a short chord or a sharp control point can loop or form a cusp;
    centripetal spacing rules both out within a segment.  Measured on 40
    random racing genomes (seed 21): the centreline crosses itself on 25
    tracks with uniform spacing, 13 with centripetal and 19 with chordal
    (spacing = chord length).

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
        # Knot spacing of the three chords: the square root of each length.
        d0, d1, d2 = (max(float(np.linalg.norm(b - a)) ** 0.5, 1e-9)
                      for a, b in ((p0, p1), (p1, p2), (p2, p3)))
        # Catmull-Rom tangents at p1 and p2 for that spacing, scaled to this
        # segment.  With d0 = d1 = d2 they reduce to 0.5 (p2 - p0), 0.5 (p3 - p1).
        m1 = ((p1 - p0) / d0 - (p2 - p0) / (d0 + d1) + (p2 - p1) / d1) * d1
        m2 = ((p2 - p1) / d1 - (p3 - p1) / (d1 + d2) + (p3 - p2) / d2) * d1
        pts = h1 * p1 + h2 * p2 + h3 * m1 + h4 * m2
        curve_points[idx:idx + samples_per_segment] = pts
        idx += samples_per_segment

    curve_points[-1] = points[0]

    return curve_points


def interpolate_curves(points, samples_per_segment=20):
    """Interpolate control points into a centripetal Catmull-Rom polyline.

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


def compute_offset_edges(points: np.ndarray, track_width: float):
    """Push a centerline sideways by half the track width, both ways.

    Each point moves along the perpendicular of the average of its two
    neighbouring segment directions, so the road keeps a constant width through
    corners.  count_track_area_intersections runs this in metres to find
    width-based self-intersections; RacingProblem._track_edges runs it in
    pixels to draw the road.
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
                cell_to_a.setdefault((gx, gy), []).append(i)

    eps_i = max(1e-9, cell_size * 1e-6)
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
                    if _segments_intersect_inclusive(a0[i], a1[i], b0[j], b1[j], eps=eps_i):
                        count += 1
    return count


def local_radius(points: np.ndarray) -> np.ndarray:
    """Radius of the circle through each sample and its two neighbours, on a
    closed curve (first point repeated at the end or not)."""
    c = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(c) > 1 and np.allclose(c[0], c[-1], atol=1e-9, rtol=0.0):
        c = c[:-1]
    if len(c) < 3:
        return np.full(len(c), np.inf)
    a, b = np.roll(c, 1, axis=0), np.roll(c, -1, axis=0)
    cross = np.abs((c - a)[:, 0] * (b - a)[:, 1] - (c - a)[:, 1] * (b - a)[:, 0])
    return (np.linalg.norm(c - a, axis=1) * np.linalg.norm(b - c, axis=1)
            * np.linalg.norm(b - a, axis=1) / np.maximum(2.0 * cross, 1e-12))


def fold_count(points: np.ndarray, track_width: float) -> int:
    """Samples of a closed centreline where the road's inner edge folds: the
    local radius (local_radius) is under half the track width.

    Where a curve's radius of curvature is under the offset distance d, the
    offset curve at d passes a cusp and runs back on itself (Farouki and
    Neff 1990, Analytic properties of plane offset curves, Computer Aided
    Geometric Design 7: the offset is irregular where the curvature is
    1/d), so the inner road edge overlaps itself there.

    Why the radius and not the drawn edges: compute_offset_edges moves each
    sample along the bisector of its two segments by half the width, and
    that polyline only runs backwards once the turn at one 5 m sample passes
    2 asin(5 / width), 77 degrees at 16 m, against 36 degrees for the fold
    itself (a 5 m step turning 36 degrees is an 8 m radius).  Measured on the
    seed-1 runs' final tracks (TORCS arc radius, the same quantity at the
    5 m step): 124 of 200 spline and 38 of 100 Voronoi tracks had a sample
    under 8 m, 109 and 14 of them at quality 1.0, and the edge crossing
    tests below caught 7 and 1 of them; no tile track had one.  Of the 24
    reference circuits only Shanghai has one, and it already failed on that
    fold.  The radius is the one _overlap_spot uses to place a repair."""
    return int(np.count_nonzero(local_radius(points) < 0.5 * float(track_width)))


def count_track_area_intersections(
    points: np.ndarray,
    track_width: float,
    *,
    min_cross_index_gap: int = 2,
) -> int:
    """Count intersections in the rendered track area (accounts for width).

    We approximate the road boundaries as left/right offset polylines and flag:
    - samples where the inner edge folds (fold_count)
    - left edge self-intersections
    - right edge self-intersections
    - left/right edge crossovers
    """
    folds = fold_count(points, track_width)
    if folds > 0:
        return folds
    left_edge, right_edge = compute_offset_edges(points, track_width=float(track_width))
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
                cell_to_segments.setdefault((gx, gy), []).append(i)

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

    # Inclusive intersection test, so self-touching is rejected too.  eps is
    # scaled to the grid's cell size to stay robust across track scales.
    eps_i = max(1e-9, cell_size * 1e-6)
    count = 0
    for i, j in candidate_pairs:
        if seg_max[i, 0] < seg_min[j, 0] or seg_max[j, 0] < seg_min[i, 0]:
            continue
        if seg_max[i, 1] < seg_min[j, 1] or seg_max[j, 1] < seg_min[i, 1]:
            continue
        if _segments_intersect_inclusive(seg_start[i], seg_end[i], seg_start[j], seg_end[j], eps=eps_i):
            count += 1

    return count
