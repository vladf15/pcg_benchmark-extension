import itertools

def get_racing_line_targets(curve_points, track_width, entry_frac=0.2, exit_frac=0.8):
    """
    For each segment, return entry, apex, and exit points with left, right, and center offsets.
    Returns: list of dicts per segment: { 'entry': {'left': pt, 'center': pt, 'right': pt}, ... }
    """
    curve_points = np.asarray(curve_points)
    n = len(curve_points)
    if n < 2:
        return []
    half_width = track_width / 2.0
    targets = []
    for i in range(n-1):
        p_start = curve_points[i]
        p_end = curve_points[i+1]
        direction = p_end - p_start
        norm = np.linalg.norm(direction)
        if norm == 0:
            perp = np.array([0, 0])
        else:
            perp = np.array([-direction[1], direction[0]]) / norm
        # Entry, apex, exit fractions
        entry_pt = p_start + direction * entry_frac
        apex_pt = p_start + direction * 0.5
        exit_pt = p_start + direction * exit_frac
        # Offset each by left/right/center
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
import numpy as np

def interpolate_curves(points, samples_per_segment=20, tension=None, bias=None, continuity=0.0):
    """
    Kochanek-Bartels spline with per-segment tension and bias.
    tension, bias: arrays/lists of length n (segments), or scalar for global value.
    """
    points = np.asarray(points)
    n = len(points)
    if n < 2:
        return points
    # If scalar, broadcast to all segments
    if tension is None:
        tension = np.zeros(n)
    elif np.isscalar(tension):
        tension = np.full(n, tension)
    else:
        tension = np.asarray(tension)
    if bias is None:
        bias = np.zeros(n)
    elif np.isscalar(bias):
        bias = np.full(n, bias)
    else:
        bias = np.asarray(bias)
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
        # Calculate tangents using per-segment Tension, Bias, Continuity
        dt1 = ((1-seg_tension)*(1+seg_bias)*(1+continuity))/2
        dt2 = ((1-seg_tension)*(1-seg_bias)*(1-continuity))/2
        m1 = dt1*(p1-p0) + dt2*(p2-p1)
        dt3 = ((1-seg_tension)*(1+seg_bias)*(1-continuity))/2
        dt4 = ((1-seg_tension)*(1-seg_bias)*(1+continuity))/2
        m2 = dt3*(p2-p1) + dt4*(p3-p2)
        # Vectorized computation for all t in t_vals
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


def count_self_intersections(points):
    # Check all pairs of non-adjacent segments for intersection
    def segments_intersect(a1, a2, b1, b2):
        def ccw(p1, p2, p3):
            return (p3[1]-p1[1]) * (p2[0]-p1[0]) > (p2[1]-p1[1]) * (p3[0]-p1[0])
        return (ccw(a1, b1, b2) != ccw(a2, b1, b2)) and (ccw(a1, a2, b1) != ccw(a1, a2, b2))
    n = len(points)
    count = 0
    for i, j in itertools.combinations(range(n-1), 2):
        # Skip adjacent or overlapping segments
        if abs(i-j) <= 1:
            continue
        if segments_intersect(points[i], points[i+1], points[j], points[j+1]):
            count += 1
    return count