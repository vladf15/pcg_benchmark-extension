import functools
import itertools
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np


@functools.lru_cache(maxsize=128)
def get_racing_line_targets_cached(curve_points_tuple, track_width, entry_frac=0.2, exit_frac=0.8):
    curve_points = np.array(curve_points_tuple)
    n = len(curve_points)
    if n < 2:
        return []
    half_width = track_width / 2.0
    targets = []
    for i in range(n - 1):
        p_start = curve_points[i]
        p_end = curve_points[i + 1]
        direction = p_end - p_start
        norm = np.linalg.norm(direction)
        if norm == 0:
            perp = np.array([0, 0])
        else:
            perp = np.array([-direction[1], direction[0]]) / norm
        entry_pt = p_start + direction * entry_frac
        apex_pt = p_start + direction * 0.5
        exit_pt = p_start + direction * exit_frac
        entry = {
            'left': entry_pt + perp * half_width,
            'center': entry_pt,
            'right': entry_pt - perp * half_width
        }
        apex = {
            'left': apex_pt + perp * half_width,
            'center': apex_pt,
            'right': apex_pt - perp * half_width
        }
        exit = {
            'left': exit_pt + perp * half_width,
            'center': exit_pt,
            'right': exit_pt - perp * half_width
        }
        targets.append({'entry': entry, 'apex': apex, 'exit': exit})
    return targets


def get_racing_line_targets(curve_points, track_width, entry_frac=0.2, exit_frac=0.8):
    """Return entry/apex/exit target points for each segment."""
    curve_points_tuple = tuple(map(tuple, np.asarray(curve_points)))
    return get_racing_line_targets_cached(curve_points_tuple, track_width, entry_frac, exit_frac)

@functools.lru_cache(maxsize=128)
def interpolate_curves_cached(points_tuple, samples_per_segment=20, tension_tuple=None, bias_tuple=None, continuity=0.0):
    points = np.array(points_tuple)
    n = len(points)
    if n < 2:
        return points
    if tension_tuple is None:
        tension = np.zeros(n)
    else:
        tension = np.array(tension_tuple)
    if bias_tuple is None:
        bias = np.zeros(n)
    else:
        bias = np.array(bias_tuple)
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
        seg_tension = tension[i-1] if i-1 < len(tension) else 0.0
        seg_bias = bias[i-1] if i-1 < len(bias) else 0.0
        dt1 = ((1-seg_tension)*(1+seg_bias)*(1+continuity))/2
        dt2 = ((1-seg_tension)*(1-seg_bias)*(1-continuity))/2
        m1 = dt1*(p1-p0) + dt2*(p2-p1)
        dt3 = ((1-seg_tension)*(1+seg_bias)*(1-continuity))/2
        dt4 = ((1-seg_tension)*(1-seg_bias)*(1+continuity))/2
        m2 = dt3*(p2-p1) + dt4*(p3-p2)
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

def interpolate_curves(points, samples_per_segment=20, tension=None, bias=None):
    """
    Interpolate control points into a Kochanek-Bartels spline polyline.
    Continuity is always locked to 0.0 for smooth transitions.
    Tension and bias parameters provide all necessary variety.
    """
    points_tuple = tuple(map(tuple, np.asarray(points)))
    tension_tuple = tuple(tension) if tension is not None else None
    bias_tuple = tuple(bias) if bias is not None else None
    return interpolate_curves_cached(points_tuple, samples_per_segment, tension_tuple, bias_tuple, continuity=0.0)


@functools.lru_cache(maxsize=128)
def count_self_intersections_cached(points_tuple):
    points = np.array(points_tuple)
    return _count_self_intersections_grid(points)

def count_self_intersections(points):
    points_tuple = tuple(map(tuple, np.asarray(points)))
    return count_self_intersections_cached(points_tuple)


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
