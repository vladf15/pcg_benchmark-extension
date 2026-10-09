import math
import os

from .engine import CarPhysicsEngine
from .agent import SteeringAgent
from .rl_agent import RLAgent, load_policy
from . import torcs
from pcg_benchmark.probs import Problem
from pcg_benchmark.spaces import ArraySpace, FloatSpace, IntegerSpace, DictionarySpace
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pcg_benchmark.probs.racing.utils import (
    interpolate_curves,
    count_self_intersections,
    count_track_area_intersections,
    compute_offset_edges,
    turning_function,
    turning_distance,
)
from pcg_benchmark.probs.utils import get_range_reward
from collections import OrderedDict, deque

PX_PER_M = 5.0


# Trained policies live in the sibling model_training package, which is not
# importable as a module, so the path is resolved from this file.
_RL_RUNS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "model_training", "runs"))
# The policy the quality simulation drives with (see the driver comment in
# __init__).  view_track reads it too, so the viewer shows the driver that
# produced the saved scores.
DEFAULT_RL_POLICY = os.path.join(_RL_RUNS_DIR, "run20", "selected", "policy.zip")


class _BoundedCache(OrderedDict):
    """Memoization dict with LRU eviction.

    A long search run evaluates tens of thousands of unique genomes; an
    unbounded cache would keep every dense curve and trajectory ever
    scored and slowly exhaust RAM over an overnight experiment."""

    def __init__(self, maxsize):
        super().__init__()
        self._maxsize = int(maxsize)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.move_to_end(key)
        while len(self) > self._maxsize:
            self.popitem(last=False)

    def __reduce__(self):
        """Rebuild through the real constructor when pickled.

        OrderedDict's reducer calls cls() with no arguments, which cannot
        supply maxsize, so a problem holding one of these fails to unpickle.
        """
        return (self.__class__, (self._maxsize,), None, None, iter(self.items()))



def _rotated_rect(center_x, center_y, forward, right, half_len, half_wid):
    fx, fy = forward; rx, ry = right
    return [
        (center_x + fx * half_len + rx * half_wid, center_y + fy * half_len + ry * half_wid),
        (center_x + fx * half_len - rx * half_wid, center_y + fy * half_len - ry * half_wid),
        (center_x - fx * half_len - rx * half_wid, center_y - fy * half_len - ry * half_wid),
        (center_x - fx * half_len + rx * half_wid, center_y - fy * half_len + ry * half_wid),
    ]


class RacingProblem(Problem):
    """Benchmark problem for generating and evaluating 2D racetrack control points."""

    _render_desc = 'Rendering racing frames'

    # Cache bounds.  Info dicts hold dense curves (tens of kB), summaries are
    # tiny, trajectories are largest but only rendering and evaluate() use
    # them.  The other three memo pure functions one evaluation calls several
    # times: _make_curve, the spline repair in _normalize_track_points, and
    # the terms dict quality() and controlability() share (which must hold
    # more than a generation of 100).  512 curves of ~1000 samples are 8 MB.
    _INFO_CACHE_MAX = 1024
    _SIM_SUMMARY_CACHE_MAX = 8192
    _TRAJECTORY_CACHE_MAX = 32
    _CURVE_CACHE_MAX = 512
    _NORMALIZE_CACHE_MAX = 512
    _TERMS_CACHE_MAX = 256

    def _get_info_cache_key(self, track_points):
        """Dictionary key for a set of track points: a tuple of row tuples
        (arrays cannot be dict keys, tuples can).  A row is (x, y), or
        (x, y, straight share) for a free-spline genome, whose curve depends
        on all three."""
        arr = np.asarray(track_points)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 2)
        return tuple(tuple(row) for row in arr)

    def _extract_content(self, content):
        if content is None:
            return self._default_track_points
        if isinstance(content, dict):
            pts = content.get("track_points", self._default_track_points)
            if "straight_frac" in content:
                # One row per control point: x, y and the straight share of
                # the edge leaving it.  The row keeps its gene through 2-opt
                # and overlap repair, which reorder and drop points;
                # _build_curve reads the third column.
                return np.column_stack([np.asarray(pts, dtype=float).reshape(-1, 2),
                                        np.asarray(content["straight_frac"], dtype=float).reshape(-1)])
            return pts
        return content

    # Splined in genome order, free-form control points cross themselves, so
    # they are reordered into a non-crossing tour first.  Radial and the
    # constructive decoders already return a valid loop and set this False.
    _untangle_control_points = True

    @staticmethod
    def _untangle_2opt(pts, max_passes=60):
        """Reorder a closed polygon into a non-crossing tour by 2-opt: while
        two edges cross, reverse the span between them.  Each reversal
        shortens the tour, so it ends simple.  On 520 random genomes (seeds
        1-13) the most passes needed was 4.

        A simple polygon still need not give a simple road: the spline bows
        outside it, centrelines closer than the road width overlap without
        crossing, and a vertex that doubles back folds the inner edge.
        _drop_overlap_points handles all three."""
        pts = np.asarray(pts, dtype=float).copy()
        n = len(pts)
        if n < 4:
            return pts

        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

        def crosses(a, b, c, d):
            return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)

        for _ in range(max_passes):
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n):
                    if i == 0 and j == n - 1:
                        continue  # adjacent across the closing edge
                    if crosses(pts[i], pts[i + 1], pts[j], pts[(j + 1) % n]):
                        pts[i + 1:j + 1] = pts[i + 1:j + 1][::-1]
                        improved = True
            if not improved:
                break
        return pts

    def _drop_overlap_points(self, pts):
        """Remove control points until the spline passes the self-overlap test.

        A control point where the polygon doubles back forces the spline
        through a turn of a few metres.  On 40 random genomes (seed 21) every
        track failed the gate, the tightest spot a median 4.7 m from a
        control point turning 171 degrees; 13% of control points turn more
        than 150 degrees.  So the point nearest the failure is removed and
        the polygon untangled again, down to six points.  On the same 40
        genomes (16 points, fold test on, 12 m road) none fails afterwards,
        a mean of 11.7 points remain, and the median lap is 4063 m.

        Simpler options, measured with the edge crossing tests alone, left
        tracks failing: dropping every point turning over 135 degrees 12 of
        40; smoothing wherever tighter than 8 m 31 of 40 (moving the road
        40 m); an approximating cubic B-spline 35 of 40, lap 4221 m.
        """
        # Tested as _normalize_track_points returns it, not reordered again:
        # _straighten and the resampling start at vertex 0, and on 1 of 40
        # genomes a curve that passed from one start vertex failed from another.
        return self._repair_overlaps(pts, self._make_curve, self._untangle_2opt)

    def _repair_overlaps(self, pts, build_curve, reorder=None):
        """While the curve build_curve(pts) fails the shared self-overlap
        test, remove the point nearest the failure (_overlap_spot), and apply
        `reorder` to what is left, down to six points.  racing-v0 repairs its
        control points with it (_drop_overlap_points, with 2-opt as the
        reorder) and racingvoronoi its cell rim (_drop_tight_corners, no
        reorder).  Six points is the floor: a curve that still fails there
        keeps its overlap and scores 0."""
        pts = np.asarray(pts, dtype=float)
        width = float(self._track_width)
        while len(pts) > 6:
            curve = build_curve(pts)
            if (count_self_intersections(curve)
                    + count_track_area_intersections(curve, track_width=width) == 0):
                break
            k = int(np.argmin(np.linalg.norm(pts[:, :2] - self._overlap_spot(curve), axis=1)))
            pts = np.delete(pts, k, axis=0)
            if reorder is not None:
                pts = reorder(pts)
        return pts

    def _overlap_spot(self, curve):
        """Where a curve that fails the self-overlap test fails it: the
        tightest sample when its radius is under half the track width (the
        inner road edge folds there), otherwise the midpoint of the closest
        pair of samples more than half a turn of that radius plus one track
        width apart along the loop."""
        c = np.asarray(curve, dtype=float)
        if np.allclose(c[0], c[-1]):
            c = c[:-1]
        a, b = np.roll(c, 1, axis=0), np.roll(c, -1, axis=0)
        cross = np.abs((c - a)[:, 0] * (b - a)[:, 1] - (c - a)[:, 1] * (b - a)[:, 0])
        radius = (np.linalg.norm(c - a, axis=1) * np.linalg.norm(b - c, axis=1)
                  * np.linalg.norm(b - a, axis=1) / np.maximum(2.0 * cross, 1e-12))
        width = float(self._track_width)
        if radius.min() < 0.5 * width:
            return c[int(np.argmin(radius))]
        # Closest pair more than min_sep samples apart along the loop, from
        # each sample's k nearest neighbours; k doubles until every sample
        # has one far enough along the loop, so memory is n x k rather than
        # the n x n distance matrix (a 7000 m lap has n = 1400).
        from scipy.spatial import cKDTree
        n = len(c)
        min_sep = (np.pi * 0.5 * width + width) / self._curve_step()
        tree = cKDTree(c)
        k = min(n, 32)
        while True:
            d, j = tree.query(c, k=k)
            sep = np.abs(j - np.arange(n)[:, None])
            far = np.minimum(sep, n - sep) > min_sep
            if far.any(axis=1).all() or k >= n:
                break
            k = min(n, 2 * k)
        d = np.where(far, d, np.inf)
        i = int(np.argmin(d.min(axis=1)))
        return 0.5 * (c[i] + c[j[i, int(np.argmin(d[i]))]])

    def _normalize_track_points(self, track_points):
        if track_points is None:
            track_points = self._default_track_points
        elif isinstance(track_points, dict):
            track_points = track_points.get("track_points", self._default_track_points)
        track_points = np.asarray(track_points, dtype=float)
        if track_points.ndim == 1:
            track_points = track_points.reshape(-1, 2)
        # A third column, when present, is the straight-share gene (see
        # _extract_content); geometry tests read only x and y.

        if len(track_points) >= 4:
            # Only a genome's own control points are reordered and repaired.
            # A track handed in as geometry (a reference circuit, a saved
            # decode) is already a road and both steps would damage it: 2-opt
            # would reverse Suzuka's figure-of-eight crossover (a bridge), and
            # the repair would delete points of any circuit failing the gate.
            # Such a circuit stays a failure: the gate is the same for every track.
            if self._untangle_control_points and len(track_points) == self.num_points:
                # Memoised, since info, evaluate, reset and the simulation
                # summary all run it; the output is stored under its own key
                # too, so normalising a repaired genome again is a lookup.
                key = (track_points.shape, track_points.tobytes())
                fixed = self._normalize_cache.get(key)
                if fixed is None:
                    fixed = self._drop_overlap_points(self._untangle_2opt(track_points))
                    fixed = np.asarray(fixed, dtype=float)
                    self._normalize_cache[key] = fixed
                    self._normalize_cache[(fixed.shape, fixed.tobytes())] = fixed
                track_points = fixed.copy()
        return track_points

    @staticmethod
    def _curve_turn_profile(curve_points):
        """(signed turn at each interior sample, arc weight of each sample).

        On a closed curve (last point repeating the first) the first segment
        is appended once more, so ang[i] is the turn at vertex i + 1 and the
        last entry the turn at vertex 0, the start line.  Without it that turn
        is dropped, which is wrong when _place_start puts the line at a
        corner exit (a longest straight under _START_TO_CORNER_M)."""
        cp = np.asarray(curve_points, dtype=float)
        if len(cp) < 3:
            return np.zeros(0), np.zeros(0)
        if np.allclose(cp[0], cp[-1], atol=1e-9):
            cp = np.vstack([cp, cp[1:2]])
        d = cp[1:] - cp[:-1]
        n = np.linalg.norm(d, axis=1, keepdims=True)
        ok = n[:, 0] > 1e-3
        u = np.zeros_like(d)
        u[ok] = d[ok] / n[ok]
        v1, v2 = u[:-1], u[1:]
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        dots = np.clip(v1[:, 0] * v2[:, 0] + v1[:, 1] * v2[:, 1], -1.0, 1.0)
        seg = n[:, 0]
        return np.arctan2(cross, dots), 0.5 * (seg[:-1] + seg[1:])

    def _find_corners(self, signed_ang, balance_w, curv_deg_per_m=None,
                      min_turn_deg=None, spans=False):
        """Corners as a designer counts them: consecutive same-direction
        curved samples accumulated until a sustained straight or a change of
        direction, keeping every run that swept at least corner_min_turn_deg
        (or min_turn_deg when given).  Returns (turns_rad, arclens_m), one
        entry per corner, and with spans=True also each corner's (start, end)
        in metres along the lap.

        Thresholding the ACCUMULATED turn keeps the count independent of the
        sampling.  A per-vertex 20 degree test at the 5 m step needs a radius
        under 14.3 m and reports 0 turns for Suzuka, Brands Hatch, Budapest,
        Oschersleben and Sao Paulo.

        A corner closes only after a straight longer than corner_gap_max_m:
        the tile decoders sample an arc as turn jumps between straight
        samples, so closing on one straight sample would split every arc into
        sub-threshold pieces."""
        P = self._QUALITY_PARAMS
        min_turn = np.deg2rad(P["corner_min_turn_deg"] if min_turn_deg is None else min_turn_deg)
        # A sample counts as straight below a curvature line times the curve
        # step, one rule that holds at any step (the corner-core line by
        # default, 4.0 degrees at 5 m).
        if curv_deg_per_m is None:
            curv_deg_per_m = P["straight_curv_deg_per_m"]
        thresh = np.deg2rad(float(curv_deg_per_m) * self._curve_step())
        gap_max = float(P["corner_gap_max_m"])
        turns, arclens, bounds = [], [], []
        acc, run_len, cur_sign, gap_len = 0.0, 0.0, 0, 0.0
        s_pos, start = 0.0, 0.0

        def _close():
            nonlocal acc, run_len, cur_sign, gap_len
            if abs(acc) >= min_turn:
                turns.append(abs(acc))
                arclens.append(run_len)
                bounds.append((start, start + run_len))
            acc, run_len, cur_sign, gap_len = 0.0, 0.0, 0, 0.0

        for i, a in enumerate(signed_ang):
            w = float(balance_w[i])
            s_pos += w
            if abs(a) < thresh:
                if cur_sign == 0:
                    continue  # not inside a corner yet: plain straight
                gap_len += w
                run_len += w
                if gap_len > gap_max:
                    run_len -= gap_len  # don't count the trailing straight
                    _close()
                continue
            sgn = 1 if a > 0 else -1
            if sgn != cur_sign and cur_sign != 0:
                _close()
            if cur_sign == 0:
                start = s_pos - w
            acc += a
            run_len += w
            cur_sign = sgn
            gap_len = 0.0  # a curved sample resumes the corner
        _close()
        if spans:
            return turns, arclens, bounds
        return turns, arclens

    def _fia_corners(self, signed_ang, balance_w):
        """Corners as FIA Appendix O (2026) Art. 7.7 defines them for the start
        rule, "a change of direction of at least 45 degrees, with a radius of
        less than 300 m", and the longest stretch of lap between two of them.
        Returns (count, longest_gap_m).

        The walker runs on the FIA 300 m line with a 45 degree minimum and
        drops corners with a mean radius of 300 m or more.  The gap is
        measured on the centreline: Art. 7.2's racing line needs a driver."""
        P = self._QUALITY_PARAMS
        turns, arcs, bounds = self._find_corners(
            signed_ang, balance_w, P["fia_corner_curv_deg_per_m"],
            min_turn_deg=45.0, spans=True)
        keep = [b for t, a, b in zip(turns, arcs, bounds) if a / max(t, 1e-9) < 300.0]
        lap = float(np.sum(balance_w))
        if not keep:
            return 0, lap
        # Corners come in lap order; the last gap wraps through the start.
        gaps = [max(0.0, keep[k + 1][0] - keep[k][1]) for k in range(len(keep) - 1)]
        gaps.append(max(0.0, keep[0][0] + lap - keep[-1][1]))
        return len(keep), float(max(gaps))

    # Corner classes of the census (24 circuits, 360 corners on the FIA 300 m
    # line): hairpin (150 degrees or more), then by mean radius slow (< 44 m),
    # medium (44-75), fast (75-150), sweeper (>= 150), shares 0.13 / 0.29 /
    # 0.24 / 0.29 / 0.08.  Edges and the hairpin test are modelling choices;
    # at the 1.11 g limit the edges are 79, 103 and 146 km/h.
    _CORNER_CLASS_EDGES_M = (44.0, 75.0, 150.0)
    _HAIRPIN_MIN_DEG = 150.0

    def _corner_class_counts(self, turns, arcs):
        """(slow, fast) corner counts: slow = hairpins + slow corners, fast =
        fast corners + sweepers, from the corners turn_count counts (the
        _find_corners output on the FIA line)."""
        slow = fast = 0
        r_slow, r_fast, _ = self._CORNER_CLASS_EDGES_M
        for t, a in zip(turns, arcs):
            if np.rad2deg(t) >= self._HAIRPIN_MIN_DEG:
                slow += 1
                continue
            r = a / max(t, 1e-9)
            if r < r_slow:
                slow += 1
            elif r >= r_fast:
                fast += 1
        return slow, fast

    # Quasi-steady-state lap simulation (Siegler, Deakin & Crolla 2000): the
    # cornering limit at each point, then a forward pass limited by power and
    # grip and a backward pass limited by braking, in a friction ellipse.  No
    # driver, so its shares do not move when the policy is retrained.  Car
    # figures are engine.py's; the grip limits are physics_tests.py
    # measurements of it (1.11 g cornering; 1.20 g braking, the 32.7 m stop
    # from 100 km/h).  Why not the RL lap: the policy tops out at 47 of
    # 85.5 m/s, so its throttle trace describes the policy.  Why not a
    # transient simulation: that is the engine's own, and it needs a driver.
    _QSS_LAT_G = 1.11
    _QSS_BRAKE_G = 1.20
    # Curvature averaged over 5 samples (25 m) before the limit is taken: a
    # racing line cuts the centreline, and a GPS centreline is noisy at 5 m.
    # A modelling choice.
    _QSS_SMOOTH = 5

    # A braking stop (stops_20) sheds at least 20 m/s, a modelling choice
    # that leaves out lifts before fast corners.  On the 24 circuits it counts
    # 5-11 stops (median 7), inside Brembo's 3 (Silverstone, 2026) to 15
    # (Monaco, 2025) braking events per F1 lap.
    _QSS_STOP_MPS = 20.0

    def _lap_simulation(self, curve_points):
        """Quasi-steady-state lap measures: the share of lap time at full
        throttle (the samples where the car accelerates rather than brakes or
        holds the cornering limit, full throttle at top speed included), the
        share at the cornering limit, and the number of braking stops that
        shed _QSS_STOP_MPS or more.  Returns a dict of the three."""
        ang, w = self._curve_turn_profile(curve_points)
        n = len(ang)
        if n < 3:
            return {"throttle_share": 0.0, "limit_share": 0.0, "stops_20": 0}
        k = np.abs(ang) / np.maximum(w, 1e-9)
        m = self._QSS_SMOOTH
        k = np.convolve(np.r_[k[-m:], k, k[:m]], np.ones(m) / m, "same")[m:-m]
        car = getattr(self, "_qss_car", None)
        if car is None:   # read once: building an engine also builds its tyre tables
            e = CarPhysicsEngine(start_position=np.zeros(2), start_angle=0.0)
            car = self._qss_car = (e.mass, e.max_power, e.driveline_eff, e.rho_air,
                                   e.cd_a, e.cl_a, e.c_rr, float(e.max_speed))
        mass, power, eta, rho, cd_a, cl_a, c_rr, vmax = car
        g = 9.81
        drag = 0.5 * rho * cd_a
        down = 0.5 * rho * cl_a / (mass * g)              # grip gain per (m/s)^2
        roll = c_rr * g
        ay0, ab0 = self._QSS_LAT_G * g, self._QSS_BRAKE_G * g
        # cornering limit with downforce: v^2 k = ay0 (1 + down v^2)
        vlim = np.full(n, vmax)
        denom = k - ay0 * down
        ok = denom > 1e-12
        vlim[ok] = np.minimum(vmax, np.sqrt(ay0 / denom[ok]))
        s0 = int(np.argmin(vlim))                  # start at the slowest point
        order = np.r_[np.arange(s0, n), np.arange(0, s0)]
        kk, ww, vl = k[order].tolist(), w[order].tolist(), vlim[order].tolist()
        vf = [0.0] * n
        vf[0] = vl[0]
        for i in range(n - 1):
            v = vf[i]
            grip = 1.0 + down * v * v
            rem = max(0.0, 1.0 - (v * v * kk[i] / (ay0 * grip)) ** 2) ** 0.5
            a = (min(power * eta / max(v, 1.0), ay0 * grip * mass * rem) - drag * v * v) / mass - roll
            vf[i + 1] = min(vl[i + 1], math.sqrt(max(v * v + 2.0 * a * ww[i], 0.0)))
        vb = list(vf)
        for i in range(n - 2, -1, -1):
            v = vb[i + 1]
            grip = 1.0 + down * v * v
            rem = max(0.0, 1.0 - (v * v * kk[i + 1] / (ay0 * grip)) ** 2) ** 0.5
            a = ab0 * grip * rem + drag * v * v / mass + roll
            vb[i] = min(vb[i], math.sqrt(v * v + 2.0 * a * ww[i]))
        t_all = t_thr = t_lim = 0.0
        stops, run_from = 0, None
        for i in range(n):
            dt = ww[i] / max(vb[i], 0.5)
            t_all += dt
            braking = vb[i] < vf[i] - 1e-6
            at_limit = vb[i] >= vl[i] - 1e-3 and vl[i] < vmax - 1e-3
            if not braking and not at_limit:
                t_thr += dt
            if at_limit and not braking:
                t_lim += dt
            # The lap starts at its slowest point, which is not inside a
            # braking run, so no run wraps past the end.
            if braking and run_from is None:
                run_from = vb[i - 1] if i > 0 else vb[i]
            elif not braking and run_from is not None:
                stops += (run_from - vb[i]) >= self._QSS_STOP_MPS
                run_from = None
        return {"throttle_share": t_thr / max(t_all, 1e-9),
                "limit_share": t_lim / max(t_all, 1e-9), "stops_20": int(stops)}

    # Typicality (Ritchie 2007, p. 73: "To what extent is the produced item
    # an example of the artefact class in question?"), measured as the
    # Mahalanobis (1936) distance of five features from the reference
    # circuits, the 22 TUMFTM circuits that pass the validity gates and the
    # length rule.  The features are fixed in advance, one for each of
    # Togelius, De Nardi & Lucas's (2006) factors that a layout carries,
    # plus the layout itself, and all are measured without the driver:
    #   throttle_share      factor 1, sensation of speed: share of the
    #                       quasi-steady-state lap at full throttle
    #                       (_lap_simulation; 45-80% in published F1 figures)
    #   limit_share         factor 2, challenge: share of that lap at the
    #                       cornering limit
    #   curvature_entropy   factor 4, variety: how spread the lap's curvature
    #                       is (Loiacono et al. 2011)
    #   fia_corners_per_km  how densely corners come, as FIA Appendix O Art.
    #                       7.7 defines a corner
    #   compactness         the layout: 4 pi area / perimeter^2 of the
    #                       centreline; circuits fold onto compact plots
    # Factor 3 (neither impossible nor trivial) is the driven-lap stage;
    # factor 5 (drift) has no measure.  Features are standardised by the
    # reference mean and sd; their covariance gets a 0.1 ridge.  Full marks
    # up to the largest leave-one-out distance of a reference circuit (Monza
    # 4.80; then Oschersleben 3.60, Melbourne 3.57), so every circuit passes
    # when held out; zero at twice that.  calibrate_typicality.py prints it.
    #
    # Why fixed in advance: features picked for separating generated tracks
    # from circuits would fit the yardstick to what it compares.  Why five:
    # with 22 circuits a larger covariance is unstable, and five covers each
    # factor once.  Why not a band per feature: each set from 22 samples,
    # and tracks unlike any circuit in combination still pass.
    #
    # Validation (calibration/, run20 at 12 m).  specificity.py: implausible
    # tracks built from the circuits (half or double size, stretched 2.5
    # times, a polygon) and 40 smooth random loops all fail; 1 of 22 passes
    # with a 6 m, 150 m-wavelength ripple and 7 of 22 rounded into
    # 3-5-corner blobs.  robustness.py (20 random genomes per representation,
    # seed 21; share at quality 1.0 spline 0.60, tile 0.20, tilediag 0.40,
    # the other three 0): the ranking holds (Kendall tau 1.0) under ridge
    # 0.05 or 0.5, the second-largest leave-one-out distance, Hotelling's
    # (1931) T^2 region at 95% or 99%, dropping any feature but limit_share,
    # a sixth feature (turning per km), and off-road limits of 0.005 or 0.02.
    # Dropping limit_share lets hex and Voronoi tracks through (tau 0.48); a
    # minimum of 8 FIA corners rejects most spline tracks (tau 0.83).  The 95%
    # region passes 2 of the 44 rippled and blob tracks against 8 here, with
    # all 22 circuits still passing.
    _TYPICALITY_FEATURES = ("throttle_share", "limit_share", "curvature_entropy",
                            "fia_corners_per_km", "compactness")
    _TYPICALITY = {
        "mean": (0.669349, 0.0490351, 0.307654, 2.56142, 0.251404),
        "sd": (0.0255822, 0.00888109, 0.05335, 0.514379, 0.0833846),
        "inv_cov": ((2.84972, 0.807717, 1.2595, 0.266129, -0.499595),
                    (0.807717, 2.52656, -0.70023, -0.614452, 0.362862),
                    (1.2595, -0.70023, 3.22285, -1.02197, 0.360158),
                    (0.266129, -0.614452, -1.02197, 2.07803, -0.71669),
                    (-0.499595, 0.362862, 0.360158, -0.71669, 1.37373)),
        "threshold": 4.801,
    }

    def _typicality_distance(self, features):
        """Mahalanobis distance of a feature vector (in _TYPICALITY_FEATURES
        order) from the reference circuits, in their standardised units."""
        T = self._TYPICALITY
        z = (np.asarray(features, dtype=float) - np.asarray(T["mean"])) / np.asarray(T["sd"])
        return float(np.sqrt(max(0.0, z @ np.asarray(T["inv_cov"]) @ z)))

    def _straight_windows(self, signed_ang, balance_w):
        """Straights: runs of the lap whose heading changes by no more than
        straight_heading_tol_deg over their whole length.

        Works on the loop walked twice, so a straight can run through the
        seam.  Returns (lens, left): for each sample r of the doubled loop,
        left[r] starts the longest straight ending at r and lens[r] is its
        length in metres, capped at one lap.

        The tolerance bounds the heading change over the whole run, so it
        does not depend on the curve step.  A per-sample curvature test fails
        both ways: no circuit sample has exactly zero curvature, and the
        0.8 deg/m corner-core line passes a 300 m sweeper as straight.  Why
        not compare only first and last heading: an S-bend would pass.  Why
        not a sagitta test: a second tolerance, and the heading range already
        bounds it (1 degree over 500 m allows 4.4 m of drift)."""
        P = self._QUALITY_PARAMS
        n = len(signed_ang)
        # heading[j] is the direction of curve segment j relative to segment 0.
        # Sample i turns between segments i and i + 1, so a run of samples
        # l..r spans segment headings l..r + 1.
        heading = np.concatenate([[0.0], np.rad2deg(np.cumsum(np.concatenate([signed_ang, signed_ang])))])
        cum = np.concatenate([[0.0], np.cumsum(np.concatenate([balance_w, balance_w]))])
        tol = float(P["straight_heading_tol_deg"])
        left = np.zeros(2 * n, dtype=int)
        hi, lo = deque([0]), deque([0])  # segment indices of the running max and min heading
        start = 0
        for r in range(2 * n):
            j = r + 1
            while hi and heading[hi[-1]] <= heading[j]:
                hi.pop()
            hi.append(j)
            while lo and heading[lo[-1]] >= heading[j]:
                lo.pop()
            lo.append(j)
            while heading[hi[0]] - heading[lo[0]] > tol or r - start + 1 > n:
                start += 1
                if hi[0] < start:
                    hi.popleft()
                if lo[0] < start:
                    lo.popleft()
            left[r] = start
        # A sample that turns by more than the tolerance on its own leaves
        # start at r + 1: an empty run of length 0.
        lens = cum[np.arange(2 * n) + 1] - cum[np.minimum(left, np.arange(2 * n) + 1)]
        return lens, left

    def _straight_mask(self, straight_lens, straight_left):
        """Which samples lie inside a straight at least corner_gap_max_m long.

        That is the gap the corner walker bridges, so a shorter straight is
        part of its corner; with no minimum every 5 m sample of a gentle curve
        would be a straight.  Marked on the doubled loop and folded back, so a
        straight through the seam counts once."""
        n = len(straight_lens) // 2
        keep = straight_lens >= float(self._QUALITY_PARAMS["corner_gap_max_m"])
        marks = np.zeros(2 * n + 1)
        np.add.at(marks, straight_left[keep], 1)
        np.add.at(marks, np.flatnonzero(keep) + 1, -1)
        covered = np.cumsum(marks)[:2 * n] > 0
        return covered[:n] | covered[n:]

    def _straight_fraction(self, balance_w, straight_lens, straight_left):
        """Share of the lap's arc length inside straights (_straight_mask)."""
        covered = self._straight_mask(straight_lens, straight_left)
        return float(np.sum(balance_w[covered]) / max(float(np.sum(balance_w)), 1e-9))

    def _between_fraction(self, local_curv, balance_w, straight_lens, straight_left):
        """Share of the lap's arc length that is neither straight nor corner:
        the gentle curve the supervisor's "clear corner/straight separation"
        asks a track not to have much of.

        Straight is _straight_mask; corner is curvature at or above the FIA
        300 m line (Art. 7.7), averaged over corner_gap_max_m either side so a
        flat sample inside a corner is not in-between.  On the 24 circuits:
        p10 0.009, median 0.057, p90 0.096.  Why not the straight share alone:
        it cannot tell corners from 2 degree bends.  Why not a per-corner
        transition length: the FIA text does not say where a corner begins,
        and Kmonicek et al. (2019) note concept design uses no transitions."""
        P = self._QUALITY_PARAMS
        straight = self._straight_mask(straight_lens, straight_left)
        k = max(1, int(round(float(P["corner_gap_max_m"]) / self._curve_step())))
        window = np.ones(2 * k + 1) / (2 * k + 1)
        smooth = np.convolve(np.concatenate([local_curv[-k:], local_curv, local_curv[:k]]),
                             window, "valid")
        corner = (smooth >= float(P["fia_corner_curv_deg_per_m"])) & ~straight
        between = ~straight & ~corner
        return float(np.sum(balance_w[between]) / max(float(np.sum(balance_w)), 1e-9))

    def _make_curve(self, track_points):
        """The dense closed centreline simulation, scoring and rendering use:
        _build_curve, started at the start line (_place_start).  Memoised on
        the points (every _build_curve is a pure function); the cached array
        is read-only, so an in-place change raises rather than leaks."""
        pts = np.asarray(track_points, dtype=float)
        key = (pts.shape, pts.tobytes())
        curve = self._curve_cache.get(key)
        if curve is None:
            curve = np.asarray(self._place_start(self._build_curve(pts)), dtype=float)
            curve.setflags(write=False)
            self._curve_cache[key] = curve
        return curve

    # Start line to the end of its straight.  FIA Appendix O (2026) Art. 7.7:
    # "preferably at least 250 m between the start line and the first
    # corner"; Kmonicek et al. (2019, Table 2): 111-1030 m, median 320 m.
    # 250 m is a user decision, for the citable figure.
    _START_TO_CORNER_M = 250.0

    def _place_start(self, curve):
        """Rotate a closed curve to start on its longest straight
        (_straight_windows), _START_TO_CORNER_M before it ends, or at its
        first sample when it is shorter; the rest lies behind the line, where
        a grid stands, as real start lines are set.  It only re-indexes the
        loop, by one rule for every representation.  Why not on the
        representation's own polygon: that lands on a straight only by chance
        and differs between them.  Why not interpolate to exactly 250 m: a
        5 m sample is within one step."""
        c = np.asarray(curve, dtype=float).reshape(-1, 2)
        p = c[:-1] if len(c) > 1 and np.allclose(c[0], c[-1], atol=1e-9) else c
        n = len(p)
        if n < 4:
            return c
        # On the closed curve ang[i] is the turn at vertex i + 1 (see
        # _curve_turn_profile), so a straight whose samples run l..r covers
        # the segments from vertex l to vertex r + 2.
        ang, w = self._curve_turn_profile(np.vstack([p, p[:1]]))
        lens, left = self._straight_windows(ang, w)
        r = int(np.argmax(lens))
        seg = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
        v, ahead = r + 2, 0.0
        while v > left[r] and ahead < self._START_TO_CORNER_M:
            v -= 1
            ahead += seg[v % n]
        start = v % n
        out = np.vstack([p[start:], p[:start]])
        return np.vstack([out, out[:1]])

    def _build_curve(self, track_points):
        """The representation's closed centreline, before the start line.

        Here a spline through control points that are not road geometry;
        decoders whose waypoints are the road (Voronoi rims, tile arcs)
        override it and only re-space, as a spline would add wobble.  Every
        result leaves at one arc-length step, so measurement resolution does
        not depend on the representation."""
        pts = np.asarray(track_points, dtype=float)
        if pts.ndim == 2 and pts.shape[1] == 3:
            pts = self._edge_handles(pts)
        pts = pts.reshape(-1, 2)
        spline = interpolate_curves(pts, samples_per_segment=self._spline_samples(pts))
        return self._resample_uniform(self._straighten(spline), step=self._curve_step())

    # Upper end of the straight-share gene: a straight takes up to 80% of its
    # edge, the corners at either end at least 10% each.  Median lap shares
    # straight / corner / in-between on 20 random genomes (seed 21, before
    # repair), against 0.63 / 0.34 / 0.06 on the circuits:
    #
    #   gene range   straight  corner  in-between
    #   0 to 0.6         0.53    0.28        0.17
    #   0 to 0.8         0.62    0.25        0.12
    #
    # 0.8 is a user decision.  Through the full decode (40 genomes) the
    # shares are 0.65 / 0.26 / 0.07 against 0.53 / 0.26 / 0.20 for a spline
    # through the vertices; median longest straight 575 m against 395 m.
    _STRAIGHT_FRAC_MAX = 0.8

    def _edge_handles(self, rows):
        """Catmull-Rom control points for a polygon whose rows are x, y and
        the share s of the edge leaving that vertex that is straight.

        A Catmull-Rom segment p1 -> p2 is straight only when p0..p3 are
        collinear.  So an edge a -> b whose straight is at least
        corner_gap_max_m gets four points: the straight runs from
        a + u(b - a) to b - u(b - a), u = (1 - s) / 2, with one more point
        halfway between each end and its vertex.  The middle is then exact
        (2.6e-13 m off the chord, against 0.87 m with two points) and the
        corner turns between edges; the vertices are not on the curve.  A
        shorter straight gets points at 1/3 and 2/3 instead, so its corners
        run together (the walker would read it as corner anyway).  The
        halfway and 1/3 placements are modelling choices.

        Why not four points at fixed shares of every edge: 0.70 / 0.23 /
        0.06 on the same genomes, but every edge then has a straight, so no
        two corners can follow each other.  Why not Voronoi-style arcs: no
        longer a spline; clothoids would need a transition length of their
        own (Kmonicek et al. 2019 use none)."""
        rows = np.asarray(rows, dtype=float)
        a = rows[:, :2]
        edge = np.roll(a, -1, axis=0) - a
        length = np.linalg.norm(edge, axis=1)
        s = np.clip(rows[:, 2], 0.0, self._STRAIGHT_FRAC_MAX)
        gap = float(self._QUALITY_PARAMS["corner_gap_max_m"])
        out = []
        for i in range(len(a)):
            u = 0.5 * (1.0 - s[i])
            if s[i] * length[i] >= gap:
                out.extend(a[i] + f * edge[i] for f in (0.5 * u, u, 1.0 - u, 1.0 - 0.5 * u))
            else:
                out.extend(a[i] + f * edge[i] for f in (1.0 / 3.0, 2.0 / 3.0))
        return np.asarray(out)

    # A spline stretch whose heading stays within _STRAIGHTEN_TOL_DEG and
    # whose mean radius (length over heading range) is at least
    # _STRAIGHTEN_MIN_RADIUS_M becomes the chord between its ends: a
    # Catmull-Rom curve is never exactly straight, so without this a spline
    # track has no straight as the constructive ones do.  5 degrees and
    # 750 m are a user decision.  Medians over 20 random genomes (16 points):
    #
    #   rule             longest straight  start straight  straight share  max move
    #   off                        220 m            95 m            0.38     0.0 m
    #   2 deg / 750 m              295 m           123 m            0.51     1.5 m
    #   5 deg / 400 m              370 m           225 m            0.65     5.2 m
    #   5 deg / 750 m              372 m           225 m            0.52     5.2 m
    #   5 deg / 1200 m             370 m           225 m            0.42     5.2 m
    #
    # The tolerance finds the straights (at 2 degrees the longest is 77 m
    # shorter); the radius floor sets how much counts as straight (0.65 at
    # 400 m, 0.42 at 1200 m) without moving the road further.  On the 25
    # circuits the median longest straight goes 650 to 710 m, the tightest
    # corner 19.3 to 19.0 m, the road moving at most 4.7 m.
    #
    # The table predates the straight gene (_edge_handles).  With it, on 40
    # random genomes (seed 21), the median longest straight is 580 m against
    # 488 m without straightening, the share 0.65 against 0.59, at the same
    # median lap.  It also runs on the circuits typicality is fitted on.
    _STRAIGHTEN_TOL_DEG = 5.0
    _STRAIGHTEN_MIN_RADIUS_M = 750.0

    def _straighten(self, curve):
        """Replace near-straight stretches of a closed dense curve with chords.

        Walks the loop once from its sharpest vertex (so no stretch is cut
        at the walk's start), growing each stretch while its heading range
        stays within _STRAIGHTEN_TOL_DEG; one with a mean radius of at least
        _STRAIGHTEN_MIN_RADIUS_M keeps only its end vertices.  The start of
        the returned closed curve does not matter (_place_start sets it)."""
        c = np.asarray(curve, dtype=float).reshape(-1, 2)
        p = c[:-1] if len(c) > 1 and np.allclose(c[0], c[-1], atol=1e-9) else c
        m = len(p)
        if m < 4:
            return c
        seg = np.roll(p, -1, axis=0) - p                    # segment j runs p[j] -> p[j+1]
        seg_len = np.linalg.norm(seg, axis=1)
        direction = np.arctan2(seg[:, 1], seg[:, 0])
        turn = np.angle(np.exp(1j * (np.roll(direction, -1) - direction)))  # at vertex j + 1
        first = (int(np.argmax(np.abs(turn))) + 1) % m      # segment leaving the sharpest vertex
        order = (first + np.arange(m)) % m
        heading = np.rad2deg(np.concatenate([[0.0], np.cumsum(turn[order][:-1])]))
        tol = float(self._STRAIGHTEN_TOL_DEG)
        keep = np.ones(m, dtype=bool)
        a = 0
        while a < m:
            b, top, bottom = a, heading[a], heading[a]
            while (b + 1 < m and max(top, heading[b + 1]) - min(bottom, heading[b + 1]) <= tol):
                b += 1
                top, bottom = max(top, heading[b]), min(bottom, heading[b])
            length = float(np.sum(seg_len[order[a:b + 1]]))
            if b > a and length >= self._STRAIGHTEN_MIN_RADIUS_M * np.deg2rad(top - bottom):
                keep[order[a + 1:b + 1]] = False            # vertices inside the stretch
            a = b + 1
        out = p[keep]
        return np.vstack([out, out[:1]])

    # Arc-length spacing of every representation's curve.  5.0 m is the
    # spacing of the 24 reference circuits (per-file mean and median 5.00,
    # segments 4.33-5.41), which every calibration runs through this code, so
    # generated tracks are measured at their references' resolution.  An
    # absolute figure, since the anchor is the database, not the road.  The
    # chord cut stays small: 0.55 m at the 6 m fold radius of a 12 m road,
    # 0.08 m at the 38 m tightest hex tile.  One number for all, or a
    # difference would be one of measurement resolution.
    _CURVE_STEP_M = 5.0

    def _curve_step(self) -> float:
        """Arc-length spacing of every representation's curve, in metres."""
        return float(self._CURVE_STEP_M)

    def _spline_samples(self, control_points: np.ndarray) -> int:
        """Samples per spline segment, so the spline's chords are already
        shorter than the resample step (a coarse spline would cut corners no
        re-spacing restores).  Aims at half a step, since the curve bows
        outside its control polygon; clamped to 8-256 for degenerate genomes.
        One count for every segment: the long chords are straights, and on 40
        random genomes (seed 21) the curve lies at most 0.050 m from the same
        spline at 256 per segment (longest raw chord 27.2 m), so a
        per-segment count would change nothing measurable."""
        pts = np.asarray(control_points, dtype=float).reshape(-1, 2)
        n_seg = max(1, len(pts))
        if n_seg < 2:
            return 8
        closed = np.vstack([pts, pts[:1]])
        perimeter = float(np.sum(np.linalg.norm(np.diff(closed, axis=0), axis=1)))
        target = 2.0 * perimeter / (n_seg * max(self._curve_step(), 1e-9))
        return int(np.clip(np.ceil(target), 8, 256))

    def _arc_samples(self, arc_length: float) -> int:
        """Points along one constructive corner arc (tiles), half a curve
        step apart as in _spline_samples.  Longer chords would stay as
        straight pieces with kinks inside every corner, read as varied
        corners: on the 20 best tracks of a racingtile GA generation, 6
        points per quarter arc (18 m chords) against 60 gave curvature
        entropy 0.453 against 0.322 (racingtilehex 0.452 against 0.213)."""
        return max(1, int(np.ceil(2.0 * float(arc_length) / self._curve_step())))

    @staticmethod
    def _resample_uniform(points: np.ndarray, *, step: float) -> np.ndarray:
        """Resample a closed polyline to uniform arc-length spacing.  Points
        land on the polyline, so the shape is kept and only re-spaced."""
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        if len(pts) < 3:
            return pts
        if not np.allclose(pts[0], pts[-1], atol=1e-9, rtol=0.0):
            pts = np.vstack([pts, pts[:1]])
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(arc[-1])
        if total <= 1e-9:
            return pts
        count = max(3, int(round(total / max(float(step), 1e-9))))
        targets = np.linspace(0.0, total, count, endpoint=False)
        out = np.column_stack([np.interp(targets, arc, pts[:, 0]),
                               np.interp(targets, arc, pts[:, 1])])
        return np.vstack([out, out[:1]])

    def _set_track_cache(self, track_points):
        """Normalized track points.  The finish target is not set here:
        _setup_car sets it to the curve's first sample, the start line, since
        control point 0 is not on the curve (_edge_handles, _place_start)."""
        return self._normalize_track_points(track_points)

    @staticmethod
    def _closed_curve_length(points) -> float:
        """Arc length of a closed polyline, closing segment included (a
        repeated first point is dropped so that edge is not counted twice)."""
        p = np.asarray(points, dtype=float).reshape(-1, 2)
        if len(p) < 2:
            return 0.0
        if np.allclose(p[0], p[-1], atol=1e-9):
            p = p[:-1]
        if len(p) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(np.vstack([p, p[:1]]), axis=0), axis=1)))

    def _get_cached_simulation_summary(self, track_points, max_steps=None, *, curve_points=None):
        """Drive the track once and cache (steps, finished, end_xy,
        speed_entropy, offroad_frac); curve_points, when given, skip the
        curve build."""
        if max_steps is None:
            max_steps = self._default_max_steps
        track_points = self._normalize_track_points(track_points)
        if curve_points is not None:
            curve_points = np.asarray(curve_points, dtype=float)
            key = (self._get_info_cache_key(track_points), max_steps,
                   self._driver, ('curve', curve_points.shape[0]))
        else:
            key = (self._get_info_cache_key(track_points), max_steps,
                   self._driver)
        cached = self._simulation_summary_cache.get(key)
        if cached is not None:
            return cached
        if self._driver == "torcs":
            summary = self._torcs_summary(track_points, curve_points)
            self._simulation_summary_cache[key] = summary
            return summary

        if curve_points is not None:
            if len(curve_points) < 2:
                start_angle = 0.0
            else:
                start_angle = float(np.arctan2(
                    curve_points[1][1] - curve_points[0][1],
                    curve_points[1][0] - curve_points[0][0],
                ))
            self._curve_points = curve_points
            state = self._setup_car(curve_points, start_angle)
        else:
            state = self.reset(track_points=track_points)

        self._agent.reset()
        done = False
        steps = 0
        margin = float(self._track_width) * 0.5 + 2.0
        x_min = margin
        x_max = float(self._width) - margin
        y_min = margin
        y_max = float(self._height) - margin

        speeds = []
        # Off-road: car centre more than half a track width from the
        # centreline (the on-road signal; leaving the map is the oob gate).
        half_width = 0.5 * float(self._track_width)
        offroad_steps = 0
        while not done and steps < max_steps:
            action = self._agent.act(state)
            state, _, done, _ = self.step(action, track_points=None)
            x, y = float(state[0]), float(state[1])
            if x < x_min or x > x_max or y < y_min or y > y_max:
                break
            # Two projection passes: TrackGeometry.project searches 60
            # segments around the hint, and the second pass re-centres that
            # window.  One pass raises mean off-road from 0.091 to 0.139 on
            # racing (12 genomes, seed 4242).
            seg_idx, _proj = self._agent._find_projection(state[:2], self._agent.current_idx)
            if abs(self._agent._signed_lateral_offset(state[:2], seg_idx)) > half_width:
                offroad_steps += 1
            speeds.append(float(state[3]))
            steps += 1

        end_xy = state[:2].copy() if state is not None else None
        steps_len = steps + 1
        offroad_frac = offroad_steps / max(steps, 1)
        finished = (
            end_xy is not None
            and len(track_points) > 0
            and steps_len < max_steps
            and self._is_finished(end_xy, steps_len=steps_len)
        )
        speed_entropy = self._speed_profile_entropy(speeds)
        summary = (steps_len, finished, end_xy, speed_entropy, offroad_frac)
        self._simulation_summary_cache[key] = summary
        return summary

    def _torcs_summary(self, track_points, curve_points=None):
        """The simulation summary from a TORCS race (torcs.run: berniw, two
        laps, race mode).  TORCS records laps, total and best lap time,
        corner-cutting penalty and damage (raceresults.cpp), not the car's
        position, so:
          finished       both laps completed
          end_xy         the start point (completion 1 if finished, else 0)
          steps          the standing lap (total less best lap, penalty
                         included) in 0.1 s steps, a lap time with either driver
          speed_entropy  0.0 (no speed trace; reported, not scored)
          offroad_frac   0.0 with no damage (never hit the wall past the
                         10 m of grass), 1.0 otherwise
        A corner cut does not fail the lap (user decision, 2026-10-09): TORCS
        charges it in time (raceengine.cpp ReRaceRules: speed * dt * depth /
        inner radius per step once 0.7 car widths inside the inner edge,
        about the metres saved).  On 2700 saved tracks: 0.002-0.004 s on 3.
        The no-damage rule is set as offroad_full_frac is, by the circuits:
        berniw races all 22 without damage at 12 m (calibration/torcs_circuits.py).
        Not graded: TORCS reports no off-road time."""
        if curve_points is None:
            curve_points = self._make_curve(track_points)
        r = torcs.run(curve_points, width=self._track_width)
        start = np.asarray(curve_points[0], dtype=float)
        clean = r["damage"] == 0.0
        if r["finished"]:
            steps = int(round((r["time"] - r["best_lap"]) / 0.1))
        else:
            steps = int(self._default_max_steps)
        return (steps, bool(r["finished"]), start, 0.0, 0.0 if clean else 1.0)

    def _trajectory_offroad_frac(self, trajectory):
        """Off-road share of a trajectory given to info() (instead of the
        cached simulation): steps more than half a width from the centreline."""
        if trajectory is None or len(trajectory) == 0 or len(self._curve_points) < 2:
            return 1.0
        agent = getattr(self, '_agent', None)
        if agent is None:
            return 1.0     # nothing to measure with: the worst case, never full marks
        half_width = 0.5 * float(self._track_width)
        idx = 0
        offroad = 0
        for row in trajectory:
            pos = np.asarray(row[:2], dtype=float)
            seg_idx, _ = agent._find_projection(pos, idx)
            idx = seg_idx
            if abs(agent._signed_lateral_offset(pos, seg_idx)) > half_width:
                offroad += 1
        return offroad / max(len(trajectory), 1)

    def _speed_profile_entropy(self, speeds):
        """Normalised entropy of the driven speed profile (Loiacono, Cardamone
        and Lanzi 2011): 0 for a lap at one speed, 1 for equal time in every
        band.  Each step is one equal-time sample."""
        P = self._QUALITY_PARAMS
        n_bins = int(P["speed_bins"])
        if len(speeds) == 0 or n_bins < 2:
            return 0.0
        clipped = np.clip(np.asarray(speeds, dtype=float), 0.0, float(P["speed_max"]) - 1e-9)
        hist, _ = np.histogram(clipped, bins=n_bins, range=(0.0, float(P["speed_max"])))
        total = float(np.sum(hist))
        if total <= 0.0:
            return 0.0
        p = hist[hist > 0] / total
        entropy = float(-np.sum(p * np.log(p)))
        return entropy / float(np.log(n_bins))

    def _get_cached_trajectory(self, track_points, max_steps=None):
        if max_steps is None:
            max_steps = self._default_max_steps
        track_points = self._normalize_track_points(track_points)
        key = (self._get_info_cache_key(track_points), max_steps, self._driver)   # a driver drives its own line
        if key in self._trajectory_cache:
            return self._trajectory_cache[key]
        traj = self.evaluate({"track_points": track_points}, max_steps=max_steps)
        self._trajectory_cache[key] = traj
        return traj

    def clear_caches(self):
        self._curve_cache = _BoundedCache(self._CURVE_CACHE_MAX)
        self._normalize_cache = _BoundedCache(self._NORMALIZE_CACHE_MAX)
        self._terms_cache = _BoundedCache(self._TERMS_CACHE_MAX)
        self._info_cache = _BoundedCache(self._INFO_CACHE_MAX)
        self._trajectory_cache = _BoundedCache(self._TRAJECTORY_CACHE_MAX)
        self._simulation_summary_cache = _BoundedCache(self._SIM_SUMMARY_CACHE_MAX)

    def __init__(self, num_points=None, **kwargs):
        Problem.__init__(self, **kwargs)
        self._track_width = float(kwargs.get("track_width", 12.0))
        # Map 1500 m: the 24 circuits' bounding boxes span 813-2171 m (median
        # 1349), and at 1500 m every representation lands inside the length
        # band.  All geometry scales off _width, so they grow together.
        #
        # Track width 12 m (user decision, 2026-10-09): FIA Appendix O (2026)
        # Art. 7.3's minimum for a new permanent circuit, and the median of the
        # TUMFTM circuits' mean widths (9.2-15.9 m, median 11.8).  Why not
        # Art. 7.3's 15 m grid width: wider than every circuit, so off-road
        # is more lenient and the fold gate stricter than on real roads
        # (Shanghai's Turn 2 folds a 16 m road, not a 12 m one).  Why not each
        # circuit's own width: one width keeps the task the same for every
        # representation.  The driver is trained at it (track_loader.TRACK_WIDTH).
        self._width = float(kwargs.get("width", 1500.0))
        self._height = float(kwargs.get("height", 1500.0))
        self._diversity = float(kwargs.get("diversity", 0.4))
        self._default_max_steps = kwargs.get("max_steps", 7000)
        self._skip_render = kwargs.get("skip_render", False)

        # The driver the simulation stage scores with.  "rl", the default:
        # the PPO policy from model_training/, since the terms should report
        # how a driver that picks its own line handles the track (a user
        # decision).  "steering": the path follower in agent.py (pure
        # pursuit, Coulter 1992, with a Stanley term, Hoffmann et al. 2007),
        # for comparison and as the fallback.  "torcs": TORCS's berniw robot
        # (torcs.py, _torcs_summary), chosen by the kwarg or the
        # PCG_RACING_DRIVER variable, as run_batch.py does.
        #
        # The policy is run20 at 9.6M steps (runs/run20/selected): run16's
        # policy fine-tuned for 10M steps on the 12 m road (train.py, 12
        # environments, seed 20), keeping the checkpoint with the least
        # off-road time on the 24 circuits (CircuitSelection, every 600k
        # steps; the acceptance target, user: almost no off-road on the
        # TUMFTM circuits).  The circuits include its 19 training tracks, so
        # the generated tracks are the independent test.  20 random genomes
        # per representation (seed 21), tracks with any step off the road:
        #
        #                       circuits                     generated tracks off-road
        #   driver at 12 m   off-road  share mean/max  speed  spline tile hex voronoi  worst
        #   run20 at 9.6M    11 of 24  0.0020 / 0.009  30.1     14    17   18    17   0.042
        #   run16 at 6.0M    24 of 24  0.0146 / 0.047  32.2     20    20   20    20   0.098
        #
        # (speed in m/s; "worst" is the largest share of one generated lap;
        # calibration/driver_table.py.)  Its checkpoints range from 11 to 21
        # circuits off-road, as run16's did (4-12 at 16 m), so the pick is
        # partly luck of the noise.  run16 at 16 m and its fine-tuning runs
        # (run17-run19) are in thesis/CODE_EXPLAINED.md section 6.  Changing
        # the driver changes every simulation term, so results scored with
        # one driver are not comparable with another's.
        self._driver = str(kwargs.get("driver", os.environ.get("PCG_RACING_DRIVER", "rl")))
        if self._driver not in ("rl", "steering", "torcs"):
            raise ValueError("racing: driver must be rl, steering or torcs, not %r" % self._driver)
        if self._driver == "torcs" and not torcs.available():
            raise FileNotFoundError("racing: driver torcs needs wtorcs.exe in %s (set TORCS_DIR)"
                                    % torcs.TORCS_DIR)
        self._rl_policy_path = kwargs.get("rl_policy_path", DEFAULT_RL_POLICY)
        # Observation scaling belongs to the checkpoint: None keeps rl_agent's
        # 85.5 m/s and 30 degrees (the engine's limits, run13 onwards); a
        # legacy-engine checkpoint such as run10 needs 83.0 and 33.
        self._rl_obs_max_speed = kwargs.get("rl_obs_max_speed")
        self._rl_obs_max_steering = kwargs.get("rl_obs_max_steering")
        self._rl_policy = None     # loaded once on first use

        # A lap counts only after 60 steps (6 s), so a car still on the line
        # cannot finish; the fastest lap of the shortest band length (2000 m
        # at 85.5 m/s) takes 234 steps, so it never binds.  A modelling choice.
        self._lap_finish_min_steps = int(kwargs.get("lap_finish_min_steps", 60))
        # Mid-lap checkpoint, as in real lap timing: a car on the start line
        # projects onto the last segment as readily as the first and reads
        # progress near 1.0 (a car that had driven 0 m of a 4283 m lap once
        # scored as finished).  The 10% band is far wider than one step of
        # travel (8.55 m at top speed), so it cannot be jumped.
        self._lap_checkpoint_frac = float(kwargs.get("lap_checkpoint_frac", 0.5))
        self._lap_checkpoint_tol = float(kwargs.get("lap_checkpoint_tol", 0.1))
        self._lap_checkpoint_seen = False
        self._steps_since_reset = 0
        self._curve_points = []

        self.clear_caches()
        self._final_target = None

        if num_points is None:
            # The one knob that moves lap length without moving the build
            # box; 16 is a user decision.  20 random genomes (seed 21):
            #
            #   points  mean Q  median lap  laps < 3500 m  points kept  turns  off-road
            #       14   0.850      3785 m        7             10        8     0.002
            #       16   0.927      3960 m        5             11        9     0.010
            #       18   0.897      4141 m        5             13       10     0.007
            #       20   0.946      4189 m        5             14      9.5     0.007
            #
            # All four finish 20 of 20; above 14 the quality differences are
            # within what 20 genomes resolve.
            num_points = kwargs.get("num_points", 16)
        self.num_points = num_points

        if "track_points" in kwargs:
            self._default_track_points = kwargs["track_points"]
        else:
            x0, y0 = 100, 100
            x1, y1 = 400, 400
            self._default_track_points = [
                [
                    int(x0 + (x1 - x0) * i / (self.num_points - 1)),
                    int(y0 + (y1 - y0) * i / (self.num_points - 1))
                ]
                for i in range(self.num_points)
            ]

        # Control points cover the whole build box, no inset: the spline runs
        # through points on the polygon's edges, inside its hull at the
        # vertices.  40 random genomes (seed 21), road edge past the box:
        #
        #   inset (share of box side)  past the box  median lap  laps < 3500 m
        #   0.032                          -34.9 m      3584 m          0.47
        #   0                                6.6 m      3798 m          0.30
        #
        # 6.6 m past the box is still 93 m inside the map.
        x0, _y0, x1, _y1 = self._build_box()
        self._content_space = DictionarySpace({
            "track_points": ArraySpace((self.num_points, 2), FloatSpace(x0, x1)),
            # Share of each control point's outgoing edge that is an exact
            # straight; see _edge_handles.
            "straight_frac": ArraySpace((self.num_points,), FloatSpace(0.0, self._STRAIGHT_FRAC_MAX)),
        })
        self._rebuild_control_space()

    # Build box side, centred: 1300 m of the 1500 m map, a user decision.
    # The square tile grid does not read it: its road runs on cell centres
    # 1.5 cells in from each edge of 11, a 1091 m square.
    _BUILD_BOX_FRAC = 1300.0 / 1500.0

    def _build_box(self):
        """(x0, y0, x1, y1): the region every representation places track in.

        One rectangle for all, so a comparison is of representations, not of
        map handed out: the build AREA is equal, the lap LENGTH is the
        representation's own and scored by the length rule.  A border is
        needed (a point on the wall puts half the road outside): hex sizes
        its grid to the box, Voronoi keeps only cells inside it, radial caps
        its spokes at half of it, the spline samples inside it, and the
        square grid makes its outer ring grass."""
        inset = 0.5 * (1.0 - self._BUILD_BOX_FRAC) * float(min(self._width, self._height))
        return inset, inset, float(self._width) - inset, float(self._height) - inset

    # Control targets: each quantity's mean over the 19 reference circuits
    # that hosted an F1 World Championship race in 2019-2025 (Austin,
    # Budapest, Catalunya, Hockenheim, Melbourne, Mexico City, Montreal,
    # Monza, Nuerburgring, Sakhir, Sao Paulo, Shanghai, Silverstone, Sochi,
    # Spa, Spielberg, Suzuka, Yas Marina, Zandvoort), measured by
    # _quality_terms, to two significant figures.  A user decision: a
    # controlled track is close to an average current F1 circuit.
    #
    #   control             mean     sd    min    max   target  full marks
    #   length (m)          5149    759   4295   6998    5100   5050-5150
    #   longest straight     803    238    460   1255     800   795-805
    #   FIA corners         12.8    2.4      8     17      13   13 exactly
    #   slow corners         6.5    2.9      2     13       7   7 exactly
    #   fast corners         5.6    3.1      1     13       6   6 exactly
    #   start straight (m)  1074    216    705   1480    1100   1050-1150
    #   tightest radius     19.9    5.6    9.8   31.7      20   19.5-20.5
    #   curvature entropy  0.299  0.048  0.187  0.381    0.30   0.295-0.305
    #   full-throttle     0.673  0.025  0.637  0.752    0.67   0.665-0.675
    #   in-between share   0.057  0.032  0.003  0.111   0.057   0.0565-0.0575
    #
    # FIA corners and the start straight (the longest stretch between two)
    # use Art. 7.7's corner (_fia_corners).  Slow = hairpins and corners
    # under 44 m, fast = 75 m or wider (_corner_class_counts): the mix of
    # corner speeds, as smb asks for a mix of enemies, jumps and coins.
    # Full-throttle share (_lap_simulation) separates power circuits such as
    # Monza (0.75 here; 76% in Kmonicek et al. 2019, 80% in Pirelli 2025)
    # from twisty ones.  All are measured without the driver.
    #
    # Random genomes hitting each target (80 per representation, seeds 21
    # and 22; spline / tile / hex / tilediag / hexdiag / voronoi): slow
    # corners 3 / 3 / 0 / 2 / 7 / 11, fast 4 / 8 / 7 / 8 / 8 / 2, FIA corners
    # 0 / 0 / 0 / 4 / 7 / 5, start straight 11 / 0 / 0 / 10 / 1 / 0,
    # full-throttle 5 / 0 / 0 / 3 / 5 / 0; mean control score 0.68 / 0.48 /
    # 0.40 / 0.63 / 0.61 / 0.50 against the circuits' 0.82; nothing meets all
    # ten (calibration/control_targets.py).  Kept out (calibration/
    # qctl_cand.py): hairpins and medium corners, which random tracks hit
    # often (8-21 and 5-20 of 80); direction, which Voronoi never varies;
    # footprint, which depends on the map size.
    #
    # Full marks only when the value rounds to the target at its precision
    # (user decision), so the tolerance is half the last significant digit
    # (_CONTROL_STEPS / 2).  Why not 0: continuous values never hit it.  Why
    # not smb's 10% plateau: for length that is 4400-5800 m, 11 of 24
    # circuits.  Why not published figures: only length and straight have
    # them.  Why not targets drawn from a range, as upstream does: that asks
    # for every track a range allows; one target measures closeness to an
    # average F1 circuit.
    _CONTROL_TARGETS = {
        "length":            5100.0,
        "longest_straight":   800.0,
        "fia_corners":         13,
        "slow_corners":         7,
        "fast_corners":         6,
        "start_straight":    1100.0,
        "tightest_radius":     20.0,
        "curvature_entropy":    0.30,
        "throttle_share":       0.67,
        "between":              0.057,
    }
    # The last significant digit of each target in _CONTROL_TARGETS.
    _CONTROL_STEPS = {
        "length":            100.0,
        "longest_straight":   10.0,
        "fia_corners":         1,
        "slow_corners":        1,
        "fast_corners":        1,
        "start_straight":    100.0,
        "tightest_radius":     1.0,
        "curvature_entropy":    0.01,
        "throttle_share":       0.01,
        "between":              0.001,
    }
    _INTEGER_CONTROLS = ("fia_corners", "slow_corners", "fast_corners")

    def _rebuild_control_space(self):
        """Build the control space: one fixed target per control quantity.

        FloatSpace needs min < max, so a target v is [v, nextafter(v)), which
        samples v; IntegerSpace(v, v + 1) for the corner counts.  No control
        is a validity condition (Khalifa et al. 2025: controls "are designer
        choices ... rather than a playability constraint").
        self._control_bands: per control, its terms key, the domain ends where
        its score reaches 0, and the tolerance.

        Domain ends: a quality rule's zero point where one bounds the
        quantity, so a value quality rejects earns no control credit (length
        2000 / 14000 m, longest straight to 4000 m); otherwise the outer Tukey
        fence (quartile -/+ 3 IQR) of the 19 F1 circuits, out to their
        extreme if beyond, clipped and rounded outward: FIA corners 0 / 25,
        slow 0 / 19, fast 0 / 18, start straight 0 / 2400 m, curvature
        entropy 0.07 / 0.53, full-throttle 0.58 / 0.76, in-between 0 / 0.18.
        Subclasses changing _QUALITY_PARAMS after __init__ must call it again."""
        P = self._QUALITY_PARAMS
        bands = {
            #  control key         terms key                   domain ends (score 0)
            "length":            ("length_m",                 (P["length_zero_lo"], P["length_zero_hi"])),
            "longest_straight":  ("longest_straight_m",       (0.0, P["straight_max_zero_m"])),
            "fia_corners":       ("fia_corner_count",         (0.0, 25.0)),
            "slow_corners":      ("slow_corners",             (0.0, 19.0)),
            "fast_corners":      ("fast_corners",             (0.0, 18.0)),
            "start_straight":    ("fia_start_straight_m",     (0.0, 2400.0)),
            "tightest_radius":   ("tightest_corner_radius_m", (0.0, P["control_radius_zero_hi_m"])),
            "curvature_entropy": ("curvature_entropy",        (0.07, 0.53)),
            "throttle_share":    ("throttle_share",           (0.58, 0.76)),
            "between":           ("between_frac",             (0.0, 0.18)),
        }
        self._control_bands = {k: (key, dom, 0.5 * float(self._CONTROL_STEPS[k]))
                               for k, (key, dom) in bands.items()}
        T = self._CONTROL_TARGETS
        space = {}
        for k in self._control_bands:
            if k in self._INTEGER_CONTROLS:
                # Upper bound excluded by IntegerSpace, so target + 1.
                space[k] = IntegerSpace(int(T[k]), int(T[k]) + 1)
            else:
                space[k] = FloatSpace(float(T[k]), float(np.nextafter(float(T[k]), np.inf)))
        self._control_space = DictionarySpace(space)

    def _make_agent(self, curve_points):
        """Build the driver named by `driver`.  A failed policy load falls back
        to SteeringAgent and says so (not silently, not by raising).  With
        "torcs" the scores come from TORCS and this policy only drives the
        trajectories evaluate() returns for renders."""
        if self._driver in ("rl", "torcs"):
            try:
                if self._rl_policy is None:
                    self._rl_policy = load_policy(self._rl_policy_path)
                return RLAgent(curve_points, track_width=self._track_width,
                               model=self._rl_policy,
                               obs_max_speed=self._rl_obs_max_speed,
                               obs_max_steering=self._rl_obs_max_steering)
            except Exception as e:
                print("racing: could not load policy %s (%s); "
                      "falling back to SteeringAgent"
                      % (self._rl_policy_path, e))
                if self._driver == "rl":
                    self._driver = "steering"
        # It plans its speed profile from the engine's own limits.
        return SteeringAgent(curve_points, track_width=self._track_width,
                             engine=self._engine)

    def _setup_car(self, curve_points, start_angle):
        """Place the car and agent on a curve: create the physics engine and
        agent on first use, re-aim them afterwards.  Returns the reset state."""
        if not hasattr(self, '_engine') or self._engine is None:
            self._engine = CarPhysicsEngine(start_position=curve_points[0], start_angle=start_angle)
        else:
            self._engine.start_position = np.asarray(curve_points[0], dtype=float)
            self._engine.start_angle = start_angle
        if not hasattr(self, '_agent') or self._agent is None:
            self._agent = self._make_agent(curve_points)
        else:
            self._agent.curve_points = curve_points

        if len(curve_points) > 0:
            self._final_target = np.asarray(curve_points[0], dtype=float)
        self._steps_since_reset = 0
        self._lap_checkpoint_seen = False
        return self._engine.reset()

    def reset(self, track_points=None):
        track_points = self._set_track_cache(track_points)
        self._curve_points = self._make_curve(track_points)

        if len(self._curve_points) < 2:
            start_angle = 0.0
        else:
            dx = float(self._curve_points[1][0] - self._curve_points[0][0])
            dy = float(self._curve_points[1][1] - self._curve_points[0][1])
            start_angle = float(np.arctan2(dy, dx))

        return self._setup_car(self._curve_points, start_angle)

    def _get_progress_fraction(self) -> float:
        """Fraction of the lap the agent has reached, in [0, 1].  Not a
        segment index: the two drivers index different polylines (RLAgent's
        TrackGeometry ring at 3 m has ~10x the points), and an index compared
        against the benchmark's segment count passes the progress test on
        every step."""
        agent = getattr(self, '_agent', None)
        if agent is None:
            return 0.0
        return float(getattr(agent, 'progress_fraction', 0.0) or 0.0)

    def _is_finished(self, end_xy, *, steps_len: int) -> bool:
        if end_xy is None or self._final_target is None:
            return False
        if steps_len < self._lap_finish_min_steps or not self._lap_checkpoint_seen:
            return False
        dist = float(np.linalg.norm(np.asarray(end_xy, dtype=float) - np.asarray(self._final_target, dtype=float)))
        if dist < max(2.0, 0.2 * self._track_width):
            return True

        # Within the last two segments of the lap, accept a looser radius: the
        # car can stop just short of the line having driven the whole circuit.
        nseg = max(1, len(self._curve_points) - 1)
        if self._get_progress_fraction() >= 1.0 - 2.0 / nseg:
            return dist < max(18.0, 0.9 * self._track_width)

        return False

    def step(self, action, track_points=None):
        if track_points is not None:
            self._set_track_cache(track_points)

        state = self._engine.step(action)
        self._steps_since_reset += 1
        if abs(self._get_progress_fraction() - self._lap_checkpoint_frac) <= self._lap_checkpoint_tol:
            self._lap_checkpoint_seen = True
        x, y = state[0], state[1]
        final_target = self._final_target
        if final_target is None:
            dist_to_final = float('inf')
        else:
            dist_to_final = np.hypot(final_target[0] - x, final_target[1] - y)
        done = self._is_finished(state[:2], steps_len=self._steps_since_reset)
        reward = -dist_to_final
        info = {"progress": self._get_progress_fraction()}
        return state, reward, done, info

    def evaluate(self, content=None, max_steps=None):
        """Simulate the agent driving the track; return the list of car states."""
        if max_steps is None:
            max_steps = self._default_max_steps
        track_points = self._extract_content(content)
        track_points = self._set_track_cache(track_points)
        state = self.reset(track_points=track_points)
        self._agent.reset()
        done = False
        steps = 0
        trajectory = [state.copy()]
        while not done and steps < max_steps:
            action = self._agent.act(state)
            state, reward, done, info = self.step(action, track_points=None)
            trajectory.append(state.copy())
            steps += 1
        return trajectory

    def info(self, content, trajectory=None, use_cache=True):
        track_points = self._extract_content(content)
        track_points = self._normalize_track_points(track_points)

        if len(track_points) < 2:
            return {
                'num_points': self.num_points,
                'total_length': 0.0,
                'steps': 0,
                'finished': False,
                'speed_entropy': 0.0,
                'offroad_frac': 1.0,
                'track_points': track_points,
                'trajectory_end': track_points[0, :2] if len(track_points) > 0 else None,
                'curve_points': track_points[:, :2],
            }

        # The driver is part of the key: the dict carries simulation results.
        cache_key = (self._get_info_cache_key(track_points), self._driver)
        if use_cache and cache_key in self._info_cache and trajectory is None:
            return self._info_cache[cache_key]

        curve_points = self._make_curve(track_points)
        # Once here (0.2 ms) rather than per pair in diversity() (10,000 pairs
        # per generation of 100).
        tf = turning_function(curve_points)
        # Measured on the driven curve, as a circuit's quoted length is; the
        # control polyline would under-report the spline by ~9%.
        total_length = self._closed_curve_length(curve_points)
        if trajectory is None:
            steps, finished, end_xy, speed_entropy, offroad_frac = self._get_cached_simulation_summary(track_points, curve_points=curve_points)
            trajectory_end = end_xy
        else:
            steps = len(trajectory)
            trajectory_end = trajectory[-1][:2] if len(trajectory) > 0 else None
            finished = steps < self._default_max_steps and self._is_finished(trajectory_end, steps_len=steps)
            # Trajectory rows are engine states [x, y, heading, speed, steer].
            traj_speeds = [float(s[3]) for s in trajectory if len(s) > 3]
            speed_entropy = self._speed_profile_entropy(traj_speeds)
            offroad_frac = self._trajectory_offroad_frac(trajectory)

        info_dict = {
            "num_points": self.num_points,
            "total_length": total_length,
            "turning_function": tf,
            "steps": steps,
            "finished": finished,
            "speed_entropy": speed_entropy,
            "offroad_frac": offroad_frac,
            "track_points": track_points,
            "trajectory_end": trajectory_end,
            "curve_points": curve_points,
        }
        if use_cache:
            self._info_cache[cache_key] = info_dict
        return info_dict

    # ── Quality ──────────────────────────────────────────────────────────
    # Could this be a real circuit.  Quality 1 is "feasible content"
    # (Khalifa et al. 2025); closeness to a requested value is
    # controlability's.  Two hard gates (off the map, road overlapping
    # itself), then three stages, each counting once the ones before are 1.0
    # (quality()), each with one source for full marks:
    #
    #   1. rules       FIA Appendix O (2026) Supplement 2, Arts. 7.2 and 7.7
    #   2. driven lap  the driver finishes, on the road as much as on the
    #                  reference circuits
    #   3. typicality  as close to the circuits, on five features fixed in
    #                  advance, as each circuit is to the others
    #
    # Zero points below full marks only shape the gradient: where a
    # regulation gives none, a lower limit ramps from 0 and an upper one to
    # twice the limit.  Why not a band per term (turn count, entropy, shares,
    # each from Tukey fences on 22 circuits): each a separate choice of where
    # circuits end, and a track can pass every band yet be unlike any
    # circuit in combination; typicality judges them jointly.  Why no more
    # rules: Art. 7.1, "the shape of the course in plan is not subject to
    # restrictions"; Art. 7.3's 12 m width the road meets by construction.
    # Not a rule: the FIM (2024) Art. 4.2 minimum of 10 turns, which
    # Spielberg, Brands Hatch and Norisring fail; corner density is a
    # typicality feature instead (user decision, 2026-10-08).
    #
    # Validity is Prasetya and Maulidevi's (2016) closed, continuous,
    # non-intersecting; the first two hold by construction, so only
    # non-intersection is checked, as a hard gate.  Circuits are scored
    # through info() and _quality_terms on a 2600 m map so Spa fits (map
    # size feeds only the oob check).

    _QUALITY_PARAMS = {
        # ── Stage 1: rules ────────────────────────────────────────────────
        # Lap length (m) on the driven curve.  Full marks from Supplement 2's
        # 3.5 km minimum for F1, sports car and GT races to Art. 7.2's 7 km
        # recommended maximum.  Zero at Supplement 2's 2 km "minimum length
        # for ... any international competition" and at 14 km (twice the
        # limit).  Circuits: min 2294 (Norisring), p50 4608, max 6998 m.
        "length_zero_lo":    2000.0,
        "min_length":        3500.0,
        "max_length":        7000.0,
        "length_zero_hi":   14000.0,
        # Art. 7.2: "The maximum permitted length for straight sections of
        # track is 2km" (the longest straight); zero at 4 km.  Circuits: max 1255 m.
        "straight_max_m":       2000.0,
        "straight_max_zero_m":  4000.0,
        # Art. 7.7: "preferably at least 250m between the start line and the
        # first corner" (a corner: at least 45 degrees, radius under 300 m).
        # Met when some stretch between two such corners is 250 m long
        # (fia_start_straight_m); zero at 0.  Circuits: min 705 m.
        "start_to_corner_m":     250.0,
        # Self-overlap gate: any centreline crossing, road overlap or fold
        # sets quality to 0.
        "angles_on_curve":     True,   # geometry checks on dense curve vs control points
        "geom_area_check":     True,   # also count track-area overlap intersections

        # ── Geometry primitives shared by several shape terms ─────────────
        # The corner-core line: above it a sample is in the tight part of a
        # corner, so the tightest radius (a control) averages the part that
        # sets the corner's speed.  0.8 deg/m is a 72 m radius, held at
        # sqrt(1.3 * 9.81 * 72) = 30.3 m/s, two thirds of the driver's 44-47 m/s top
        # speed on the circuits.  Why not the FIA 300 m line: its average
        # takes in entry and exit and reads the corner wider than its apex.
        # A modelling choice.
        "straight_curv_deg_per_m": 0.8,
        # A straight changes heading by no more than this over its length
        # (_straight_windows); 1 degree is a user decision.  Median longest
        # straight, 24 circuits and 12 random genomes per representation
        # (first three rows on the spline before _straighten):
        #
        #   definition                  circuits  racing  tile  hex  voronoi
        #   curvature < 0.8 deg/m         1067     1142    420   645    405
        #   curvature exactly 0              0        0    400   243    197
        #   heading within 1 degree        652      275    410   255    210
        #   same, after _straighten        702      445    405   250    270
        #
        # 0.8 deg/m counts sweepers as straight; an exact test finds nothing on
        # the raw spline or the noisy circuits.  The heading tolerance keeps
        # constructive straights and finds the circuits' (0.5 and 2 degrees
        # give a raw median of 582 and 679 m).  Circuits are straightened too.
        "straight_heading_tol_deg": 1.0,
        # A corner: accumulated same-direction turn of at least this.  Turn
        # count on the FIA line against Kmonicek et al.'s (2019, Table 2)
        # official counts for 8 database circuits (Yas Marina, Sakhir,
        # Catalunya, Monza, Sao Paulo, Sochi, Spa, Suzuka), error in turns:
        #
        #   minimum turn     10     15     20     25     30     45
        #   mean error     +2.2   +0.6   -0.1   -0.9   -1.4   -3.5
        #   MAE             2.5    1.1    1.1    1.1    1.6    3.5
        #
        # 20 degrees is the sweep's centre.  At Art. 7.7's 45 degrees (a
        # definition for the start rule) the count misses 3.5 turns; on the
        # 0.8 deg/m core line it misses 5.4-7.0 at every minimum, as a 120 m
        # sweeper never reaches it.
        "corner_min_turn_deg":   20.0,
        # Straight gap bridged inside one corner, so short near-straight
        # pieces do not split it.  Against 5 m: tiles count the same corners,
        # 6 circuits gain 0-2 corners each at 5 m, Voronoi 0-2 on 8 genomes.
        # 20 m is far below any real straight, so corners are not merged
        # across one.
        "corner_gap_max_m":      20.0,
        # Art. 7.7's 300 m radius, 0.191 deg/m (turn count, in-between share).
        "fia_corner_curv_deg_per_m": 0.191,

        # Tightest corner radius is not a quality term: no published minimum
        # (FIA Appendix O, Kmonicek et al. 2019), the fold gate already
        # rejects radii under half the road (6 m), and at the 5 m step the
        # walker cannot resolve less than about 9.4 m (90 degree corners of 5,
        # 8, 12, 20, 30 m measure 9.4, 12.6, 15.8, 22.4, 30.4 m), so a term
        # would act only in that gap, on a number with no source.  The car's
        # own limit, its 11.2 m turning circle (engine.py), is 5.6 m.  It is
        # a control; its upper domain end is the circuits' outer Tukey fence
        # (tightest corners 13.9 / 16.9 / 24.0 / 36.5 m min / p25 / p75 / max,
        # fence 45.4, rounded up to 46 m).
        "control_radius_zero_hi_m":   46.0,

        # ── Curvature entropy (a typicality feature) ──────────────────────
        # How spread the lap's curvature is (Loiacono, Cardamone & Lanzi 2011).
        # 8 bins over 0-5.0 deg/m: the top is an 11.5 m radius, below every
        # circuit's tightest (13.9 m), so the top bin holds corners tighter
        # than any.  Modelling choices.  They (Sec. VI, preprint) use 16 bins
        # over all TORCS tracks' range and maximise the raw entropy; here a
        # fixed range, unsigned curvature, entropy over log(bins), judged by
        # typicality.
        "curvature_bins":             8,
        "curvature_max_deg_per_m":  5.0,

        # ── Speed entropy (info only, not scored) ─────────────────────────
        # Loiacono et al.'s speed half, in info() for analysis.  Not scored:
        # it describes the driver as much as the track.  8 bins over 0-55 m/s,
        # which clips no sample on the circuits.
        "speed_bins":              8,
        "speed_max":            55.0,

        # ── Stage 2: the driven lap ───────────────────────────────────────
        # Share of driven steps off the road (car centre past the road edge).
        # Full marks up to 0.01: the default driver's worst lap on the 24
        # circuits (0.0095, Melbourne; run20 at 12 m), rounded up to the next
        # 0.01, so a track it leaves no more than a real circuit counts as
        # drivable.  Measured again whenever the policy changes.  Zero at
        # 0.20, for the gradient only.
        "offroad_full_frac":         0.01,
        "onroad_max_frac":           0.20,
    }

    # Stage 1 terms, one per regulation, scored as their mean.
    _RULE_TERMS = ("length_score", "straight_max_score", "start_to_corner_score")

    def _quality_terms(self, info):
        """The quality terms (each in [0, 1]) and the raw values behind them,
        or None for an unusable track.  Memoised on the info object (the memo
        holds a reference, so the id stays unique) for quality() and
        controlability(); not stored in the saved info, so rescoring a saved
        info gives today's terms."""
        memo = self._terms_cache.get(id(info))
        if memo is not None and memo[0] is info:
            return memo[1]
        terms = self._compute_quality_terms(info)
        self._terms_cache[id(info)] = (info, terms)
        return terms

    def _compute_quality_terms(self, info):
        P = self._QUALITY_PARAMS
        track_points = info.get('track_points', None)
        if track_points is None:
            return None
        points = np.asarray(track_points, dtype=float)
        if len(points) == 0:
            return None

        curve_points = info.get('curve_points', None)
        if curve_points is None:
            curve_points = self._make_curve(points)
        curve_points = np.asarray(curve_points, dtype=float)

        # Out-of-bounds: 1.0 inside the map margin, falling to 0 at four track
        # widths of violation (as sokoban's heuristic term).
        margin = float(self._track_width) * 0.5 + 2.0
        oob_violation = 0.0
        if len(curve_points) > 0:
            oob_violation = max(
                0.0,
                margin - float(np.min(curve_points[:, 0])),
                float(np.max(curve_points[:, 0])) - (self._width - margin),
                margin - float(np.min(curve_points[:, 1])),
                float(np.max(curve_points[:, 1])) - (self._height - margin),
            )
        oob_score = float(get_range_reward(
            oob_violation, 0.0, 0.0, 0.0, 4.0 * float(self._track_width)))

        # Completion: the arc position the car reached (distance to the goal
        # is useless on a loop, where start == goal).
        finished = info.get('finished', False)
        trajectory_end = info.get('trajectory_end', None)
        if trajectory_end is None:
            trajectory_end = points[-1, :2]
        trajectory_end = np.asarray(trajectory_end, dtype=float)
        if finished:
            completion = 1.0
        elif len(curve_points) > 1:
            nearest = int(np.argmin(np.linalg.norm(curve_points - trajectory_end, axis=1)))
            completion = nearest / len(curve_points)
        else:
            completion = 0.0

        if len(curve_points) > 1:
            total_length = self._closed_curve_length(curve_points)
            length_score = float(get_range_reward(
                total_length, P["length_zero_lo"], P["min_length"],
                P["max_length"], P["length_zero_hi"]))
        else:
            length_score = 0.0

        # Self-overlap: centreline crossings, plus road overlaps and folds
        # with geom_area_check.
        geom_pts = curve_points if P["angles_on_curve"] else points[:, :2]
        geom_violations = int(count_self_intersections(geom_pts))
        if P["geom_area_check"]:
            geom_violations += int(count_track_area_intersections(
                curve_points,
                track_width=float(self._track_width),
                min_cross_index_gap=2,
            ))

        # Straights, curvature and corners, all on the dense curve.
        cp_d     = curve_points[1:] - curve_points[:-1]
        cp_len   = np.linalg.norm(cp_d, axis=1)
        cp_total = float(np.sum(cp_len))
        if cp_total > 1e-6 and len(curve_points) >= 3:
            signed_ang, balance_w = self._curve_turn_profile(curve_points)
            local_curv = np.rad2deg(np.abs(signed_ang)) / np.maximum(balance_w, 1e-9)

            straight_lens, straight_left = self._straight_windows(signed_ang, balance_w)
            straight_frac = self._straight_fraction(balance_w, straight_lens, straight_left)
            between_frac = self._between_fraction(
                local_curv, balance_w, straight_lens, straight_left)

            longest_straight_len = float(np.max(straight_lens))
            straight_max_score = float(get_range_reward(
                longest_straight_len, -1.0, 0.0, P["straight_max_m"], P["straight_max_zero_m"]))

            # Curvature entropy, binned by ARC LENGTH so sampling density
            # does not change it.
            n_cbins = int(P["curvature_bins"])
            c_max = float(P["curvature_max_deg_per_m"])
            c_clipped = np.clip(local_curv, 0.0, c_max - 1e-9)
            c_hist, _ = np.histogram(c_clipped, bins=n_cbins,
                                     range=(0.0, c_max), weights=balance_w)
            c_total = float(np.sum(c_hist))
            if c_total > 0.0 and n_cbins > 1:
                c_p = c_hist[c_hist > 0] / c_total
                curvature_entropy = float(-np.sum(c_p * np.log(c_p)) / np.log(n_cbins))
            else:
                curvature_entropy = 0.0
            fia_turns, fia_arcs = self._find_corners(
                signed_ang, balance_w, P["fia_corner_curv_deg_per_m"])
            turn_count = len(fia_turns)

            fia_corner_count, fia_start_straight = self._fia_corners(signed_ang, balance_w)
            start_to_corner_score = float(get_range_reward(
                fia_start_straight, 0.0, P["start_to_corner_m"], 1e12, 1e12 + 1.0))
            # Control quantities, not quality terms (_CONTROL_TARGETS).
            slow_corners, fast_corners = self._corner_class_counts(fia_turns, fia_arcs)
            lap_sim = self._lap_simulation(curve_points)
            throttle_share = lap_sim["throttle_share"]
            # Typicality features (_TYPICALITY).
            poly = curve_points[:-1] if np.allclose(curve_points[0], curve_points[-1]) else curve_points
            area = 0.5 * abs(float(np.sum(poly[:, 0] * np.roll(poly[:, 1], -1)
                                           - np.roll(poly[:, 0], -1) * poly[:, 1])))
            compactness = 4.0 * np.pi * area / max(cp_total, 1e-9) ** 2
            turning_per_km = float(np.sum(np.abs(signed_ang))) / (2.0 * np.pi) / max(cp_total / 1000.0, 1e-9)
            fia_corners_per_km = fia_corner_count / max(cp_total / 1000.0, 1e-9)
            typicality_distance = self._typicality_distance(
                (throttle_share, lap_sim["limit_share"], curvature_entropy,
                 fia_corners_per_km, compactness))
            thr = self._TYPICALITY["threshold"]
            typicality_score = float(get_range_reward(
                typicality_distance, -1.0, 0.0, thr, 2.0 * thr))

            # Tightest corner radius (a control): the tightest SUSTAINED corner
            # on the corner-core line (arc over swept angle), not a
            # one-sample spike, so sampling density does not change it.
            tightest_radius_m = 1e12  # no corner: unbounded radius
            for turn, alen in zip(*self._find_corners(signed_ang, balance_w)):
                if float(turn) > 1e-6:
                    tightest_radius_m = min(tightest_radius_m, alen / float(turn))
        else:
            straight_frac = 0.0
            between_frac = 1.0
            longest_straight_len = 0.0
            straight_max_score = 0.0
            curvature_entropy = 0.0
            turn_count = 0
            tightest_radius_m = 0.0
            fia_corner_count, fia_start_straight = 0, 0.0
            start_to_corner_score = 0.0
            fia_corners_per_km = 0.0
            slow_corners, fast_corners = 0, 0
            throttle_share = 0.0
            lap_sim = {"throttle_share": 0.0, "limit_share": 0.0, "stops_20": 0}
            compactness = turning_per_km = 0.0
            typicality_distance, typicality_score = float("inf"), 0.0

        speed_entropy = float(info.get('speed_entropy', 0.0))

        offroad_frac = float(info.get('offroad_frac', 1.0))
        on_road_score = float(get_range_reward(
            offroad_frac, -1.0, 0.0, P["offroad_full_frac"], P["onroad_max_frac"]))

        terms = {
            "oob_score":                 oob_score,
            # Stage 1: rules (_RULE_TERMS)
            "length_score":              length_score,
            "straight_max_score":        straight_max_score,
            "start_to_corner_score":     start_to_corner_score,
            # Stage 2: the driven lap
            "completion":                completion,
            "on_road_score":             on_road_score,
            # Stage 3: typicality
            "typicality_score":          typicality_score,
            # Raw values (not scored; read by controlability and calibration).
            "length_m":                  float(self._closed_curve_length(curve_points))
                                         if len(curve_points) > 1 else 0.0,
            "longest_straight_m":        float(longest_straight_len),
            "straight_frac":             float(straight_frac),
            "between_frac":              float(between_frac),
            "curvature_entropy":         float(curvature_entropy),
            "turn_count":                int(turn_count),
            "tightest_corner_radius_m":  float(tightest_radius_m),
            "fia_corner_count":          int(fia_corner_count),
            "fia_start_straight_m":      float(fia_start_straight),
            "fia_corners_per_km":        float(fia_corners_per_km),
            "slow_corners":              int(slow_corners),
            "fast_corners":              int(fast_corners),
            "throttle_share":            float(throttle_share),
            "limit_share":               float(lap_sim["limit_share"]),
            "stops_20":                  int(lap_sim["stops_20"]),
            "compactness":               float(compactness),
            "turning_per_km":            float(turning_per_km),
            "typicality_distance":       float(typicality_distance),
            "speed_entropy":             speed_entropy,
            "offroad_frac":              offroad_frac,
            "self_overlaps":             int(geom_violations),
        }
        return terms

    def quality(self, info):
        terms = self._quality_terms(info)
        if terms is None:
            return 0.0

        # Off the map: quality 0.  Why not graded: any value above 0 ranks an
        # infeasible track above a feasible one that scores less, against
        # Deb's (2000) rule (as stated by Hellwig and Beyer 2018, Sec. 2;
        # Deb's paper not opened).  No gradient is needed: every
        # representation builds inside _build_box, and no random genome
        # (20 each, seed 21) leaves the map.
        if terms["oob_score"] < 0.999:
            return 0.0

        # Road overlapping itself: quality 0 (a user decision; the search
        # gets no gradient out of an overlap).  Random genomes still
        # overlapping here: racing 0 of 520 (after repair), radial 30 of 30
        # (no repair step), tile, hex and Voronoi 0 of 40.
        if terms["self_overlaps"] > 0:
            return 0.0

        # Staged, as the benchmark's zelda, isaac and mdungeons: the mean of
        # the stage scores, a stage counting only once those before it are
        # 1.0.  Why not the product: it scores a track breaking a rule the
        # same as one also failing everything after; the staged mean ranks
        # the partly good track higher.  Rules first: they need only the
        # geometry, and a track breaking one is no circuit however it drives.
        stages = (
            float(np.mean([terms[k] for k in self._RULE_TERMS])),                # 1. FIA rules
            float(np.mean((terms["completion"], terms["on_road_score"]))),        # 2. driven lap
            terms["typicality_score"],                                            # 3. typicality
        )
        # A stage passes at 0.999; passing all three scores exactly 1.0, the
        # benchmark's feasibility line (pcg_env.quality).
        score = 0.0
        for value in stages:
            if value < 0.999:
                return (score + value) / len(stages)
            score += 1.0
        return 1.0

    def diversity(self, info1, info2):
        """How different two layouts are: Arkin et al.'s (1991) turning-
        function distance (L2 between heading-against-lap-fraction curves,
        minimised over rotation and start point; invariant to position,
        rotation and scale) in units of pi, ramped to full marks at
        self._diversity.  It compares the sequence of corners and straights
        a driver meets, as zelda and mdungeons compare solution paths.
        Length is left out (it is a control; a scaled copy is the same
        layout); a mirror image counts as different (0.331 for a circuit and
        its mirror).  Why not an occupancy grid: a shifted copy counts as new
        and small footprints look alike.  Why not speed profiles: they depend
        on the driver.  The benchmark's 0.4 (8 of its 14 problems) fits: the
        276 circuit pairs run p10 0.293, median 0.382, p90 0.474.  Random
        pairs (seed 7) median: racing 0.313, Voronoi 0.342, tile 0.468, hex
        0.478."""
        tf1 = info1.get("turning_function")
        tf2 = info2.get("turning_function")
        if tf1 is None or tf2 is None:
            return 0.0
        d = min(turning_distance(tf1, tf2) / np.pi, 1.0)
        return get_range_reward(d, 0, self._diversity, 1.0)

    def controlability(self, info, control):
        """The mean over the targets of get_range_reward(value, domain_lo,
        target - tol, target + tol, domain_hi), every upstream problem's form,
        with the domain ends of _rebuild_control_space.  Slopes are uneven
        off-centre (5100 m target: 1000 m short scores 0.69, 1000 m long
        0.77).  Why not a symmetric trapezoid: not the benchmark's form, and
        its ends could credit a value quality scores 0.  Why not one distance
        over all ten: one large miss could be paid for by the rest; here a
        missed target costs at most 1/10.  No control depends on the driver."""
        terms = self._quality_terms(info)
        if terms is None:
            return 0.0
        scores = []
        for k, (key, (dom_lo, dom_hi), tol) in self._control_bands.items():
            c = float(control[k])
            scores.append(float(get_range_reward(
                float(terms[key]), min(dom_lo, c - tol - 1e-9), c - tol,
                c + tol, max(dom_hi, c + tol + 1e-9))))
        return float(np.mean(scores))

    def _draw_bg_overlay(self, bg_draw, scale):
        """Hook: draw extra background detail (e.g. the voronoi grid) between
        the road surface and the centerline.  Base problem draws nothing."""
        return

    def _render_track_bg(self, img_w, img_h, left_edge_f, right_edge_f, scaled_curve,
                         grass_color, edge_color, road_color, centerline_color, scale=1.0):
        bg      = Image.new("RGB", (img_w, img_h), grass_color)
        bg_draw = ImageDraw.Draw(bg)
        left_edge  = [(int(round(x)), int(round(y))) for x, y in left_edge_f]
        right_edge = [(int(round(x)), int(round(y))) for x, y in right_edge_f]
        if len(left_edge)  > 1: bg_draw.line(left_edge,  fill=edge_color, width=4)
        if len(right_edge) > 1: bg_draw.line(right_edge, fill=edge_color, width=4)
        # One quad per pair of neighbouring edge points fills the road surface.
        for j in range(len(left_edge) - 1):
            bg_draw.polygon(
                [left_edge[j], left_edge[j + 1], right_edge[j + 1], right_edge[j]],
                fill=road_color,
            )
        self._draw_bg_overlay(bg_draw, scale)
        if len(scaled_curve) > 1:
            bg_draw.line(scaled_curve, fill=centerline_color, width=2)
        return bg

    # Frames: _track_edges, _render_track_bg (background once), _draw_car,
    # _draw_lookahead, _draw_hud, _iter_render_frames (replays the
    # trajectory, one frame at a time), render (collects them).  Draw helpers
    # record the pixels they touch in bbox_xs/bbox_ys, so fast mode restores
    # only that dirty rectangle per frame.

    def _track_edges(self, curve_px, half_width):
        """Road-edge polylines in pixels (utils.compute_offset_edges) as the
        tuple lists PIL draws."""
        left, right = compute_offset_edges(curve_px, track_width=2.0 * half_width)
        return ([(float(x), float(y)) for x, y in left],
                [(float(x), float(y)) for x, y in right])

    @staticmethod
    def _load_hud_font(font_size):
        for font_path in ("DejaVuSans.ttf", r"C:\Windows\Fonts\segoeui.ttf", "arial.ttf"):
            try:
                return ImageFont.truetype(font_path, font_size)
            except Exception:
                pass
        return ImageFont.load_default()

    @staticmethod
    def _draw_car(draw, state, scale, render_scale, bbox_xs, bbox_ys):
        """Draw the car (four wheels, red body, blue heading line).
        Returns the car center (car_x, car_y) in pixels."""
        car_length = 5.0 * scale
        car_width = 2.0 * scale
        car_x = float(state[0]) * scale
        car_y = float(state[1]) * scale
        angle = state[2] if len(state) > 2 else 0.0
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        forward = (float(cos_a), float(sin_a))
        right = (float(-sin_a), float(cos_a))

        # Wheels first, so the body is drawn on top of them.
        wheel_base = car_length * 0.34    # wheel center offset, along the car
        wheel_track = car_width * 0.44    # wheel center offset, across the car
        half_wheel_len = car_length * 0.20 / 2.0
        half_wheel_wid = car_width * 0.22 / 2.0
        for f_sign in (1.0, -1.0):
            for r_sign in (1.0, -1.0):
                wx = car_x + forward[0] * wheel_base * f_sign + right[0] * wheel_track * r_sign
                wy = car_y + forward[1] * wheel_base * f_sign + right[1] * wheel_track * r_sign
                wheel = _rotated_rect(wx, wy, forward, right, half_wheel_len, half_wheel_wid)
                draw.polygon(wheel, fill=(25, 25, 25), outline=(0, 0, 0))
                bbox_xs.extend(p[0] for p in wheel)
                bbox_ys.extend(p[1] for p in wheel)

        half_len = car_length / 2.0
        half_wid = car_width / 2.0
        body = _rotated_rect(car_x, car_y, forward, right, half_len, half_wid)
        draw.polygon(body, fill=(255, 0, 0), outline=(0, 0, 0))

        front_x = car_x + cos_a * half_len
        front_y = car_y + sin_a * half_len
        line_w = max(1, int(round(3.0 * float(render_scale))))
        draw.line([(car_x, car_y), (front_x, front_y)], fill=(0, 0, 255), width=line_w)

        bbox_xs.extend([p[0] for p in body] + [car_x, front_x])
        bbox_ys.extend([p[1] for p in body] + [car_y, front_y])
        return car_x, car_y

    @staticmethod
    def _draw_lookahead(draw, car_xy, lookahead, scale, render_scale, bbox_xs, bbox_ys):
        """Draw the agent's aim point: a cyan circle plus a line from the car."""
        la_x = float(lookahead[0]) * scale
        la_y = float(lookahead[1]) * scale
        radius = max(2, int(round(6.0 * float(render_scale))))
        circle_w = max(1, int(round(3.0 * float(render_scale))))
        line_w = max(1, int(round(2.0 * float(render_scale))))
        draw.ellipse(
            [(la_x - radius, la_y - radius), (la_x + radius, la_y + radius)],
            outline=(0, 255, 255), width=circle_w,
        )
        draw.line([car_xy, (la_x, la_y)], fill=(0, 200, 200), width=line_w)
        bbox_xs.extend([la_x - radius, la_x + radius, car_xy[0], la_x])
        bbox_ys.extend([la_y - radius, la_y + radius, car_xy[1], la_y])

    @staticmethod
    def _draw_hud(draw, lines, font, font_size, render_scale, bbox_xs, bbox_ys):
        """Draw the telemetry box (white rectangle with one line per entry)."""
        pad = max(2, int(round(6.0 * float(render_scale))))
        x0, y0 = pad, pad
        line_h = max(1, int(round(float(font_size) * 1.25)))
        max_w = 0
        for txt in lines:
            try:
                bbox = draw.textbbox((0, 0), txt, font=font)
                w = int(bbox[2] - bbox[0])
            except Exception:
                w = int(len(txt) * float(font_size) * 0.6)
            if w > max_w:
                max_w = w
        box_w = max_w + pad * 2
        box_h = pad * 2 + line_h * len(lines)

        outline_w = max(1, int(round(1.0 * float(render_scale))))
        draw.rectangle([(x0, y0), (x0 + box_w, y0 + box_h)], fill=(255, 255, 255), outline=(0, 0, 0), width=outline_w)
        ty = y0 + pad
        for txt in lines:
            draw.text((x0 + pad, ty), txt, fill=(0, 0, 0), font=font)
            ty += line_h

        bbox_xs.extend([x0, x0 + box_w])
        bbox_ys.extend([y0, y0 + box_h])

    def _iter_render_frames(
        self,
        content=None,
        frame_sampling=2,
        skip=None,
        *,
        fast=True,
        show_hud=True,
        progress=True,
        progress_desc=None,
        render_scale=1.0,
    ):
        """Yield frames one at a time (memory-safe for gif writers),
        replaying the cached trajectory over a background drawn once.  A HUD
        copy of the driver runs alongside to recover its action and
        lookahead, so agent.act runs on every step, sampled frames or not."""
        if skip is None:
            skip = self._skip_render
        if skip:
            return

        track_points = self._extract_content(content)
        trajectory = self._get_cached_trajectory(track_points)

        try:
            frame_sampling = max(1, int(frame_sampling))
        except Exception:
            frame_sampling = 5
        try:
            render_scale = float(render_scale)
        except Exception:
            render_scale = 1.0
        if render_scale <= 1e-9:
            render_scale = 1.0

        # ── Static background ──────────────────────────────────────────
        track_points_np = self._normalize_track_points(track_points)
        curve_np = np.asarray(self._make_curve(track_points_np), dtype=float)
        scale = float(PX_PER_M) * float(render_scale)
        img_w = int(round(float(self._width) * scale))
        img_h = int(round(float(self._height) * scale))
        curve_px = curve_np * scale
        scaled_curve = [(int(round(x)), int(round(y))) for (x, y) in curve_px]
        half_width = float(self._track_width) * scale * 0.5
        left_edge, right_edge = self._track_edges(curve_px, half_width)
        track_background = self._render_track_bg(
            img_w, img_h, left_edge, right_edge, scaled_curve,
            grass_color=(34, 139, 34), edge_color=(10, 10, 10),
            road_color=(215, 215, 215), centerline_color=(120, 120, 120),
            scale=scale,
        )

        # ── HUD setup ──────────────────────────────────────────────────
        agent = None
        font = None
        if show_hud:
            engine = getattr(self, '_engine', None)
            agent = self._make_agent(curve_np)
            agent.reset()
            dt = float(getattr(engine, 'time_step', 0.1) or 0.1)
            mass = float(getattr(engine, 'mass', 1350.0) or 1350.0)
            font_size = max(10, int(round(24.0 * float(render_scale))))
            font = self._load_hud_font(font_size)

        # ── Frame loop ─────────────────────────────────────────────────
        reuse_canvas = bool(fast)
        img = track_background.copy() if reuse_canvas else None
        prev_bbox = None
        dirty_pad_px = max(2, int(round(12.0 * float(render_scale))))

        steps = trajectory
        if progress:
            try:
                from tqdm import tqdm  # type: ignore
                desc = progress_desc if progress_desc is not None else self._render_desc
                steps = tqdm(trajectory, total=len(trajectory), desc=str(desc), leave=False, dynamic_ncols=True)
            except Exception:
                pass

        prev_angle = None
        for i, state in enumerate(steps):
            angle = state[2] if len(state) > 2 else 0.0

            # The HUD agent must see every step, or its internal progress
            # tracking would fall behind the replayed car.
            action = None
            lookahead = None
            yaw_rate = 0.0
            if show_hud and agent is not None:
                try:
                    action = agent.act(state)
                except Exception:
                    action = {'steering': 0.0, 'throttle': 0.0}
                lookahead = getattr(agent, 'last_lookahead_point', None)
                if prev_angle is not None:
                    da = (float(angle) - float(prev_angle) + np.pi) % (2.0 * np.pi) - np.pi
                    yaw_rate = float(da / dt)
                prev_angle = float(angle)

            if i % frame_sampling != 0:
                continue

            # Fresh canvas: either restore the dirty rectangle of the
            # previous frame (fast) or copy the whole background (safe).
            if reuse_canvas:
                if prev_bbox is not None:
                    img.paste(track_background.crop(prev_bbox), prev_bbox)
            else:
                img = track_background.copy()
            draw = ImageDraw.Draw(img)

            bbox_xs, bbox_ys = [], []
            car_xy = self._draw_car(draw, state, scale, render_scale, bbox_xs, bbox_ys)

            if lookahead is not None:
                try:
                    self._draw_lookahead(draw, car_xy, lookahead, scale, render_scale, bbox_xs, bbox_ys)
                except Exception:
                    pass

            if show_hud and font is not None:
                v = float(state[3]) if len(state) > 3 else 0.0
                steer_angle_deg = float(state[4]) * (180.0 / np.pi) if len(state) > 4 else 0.0
                steer_cmd = float(action.get('steering', 0.0)) if isinstance(action, dict) else 0.0
                throttle_cmd = float(action.get('throttle', 0.0)) if isinstance(action, dict) else 0.0
                a_lat = float(v * yaw_rate)
                fy = float(mass * a_lat)
                lines = [
                    f"v: {v:5.1f} m/s  ({v * 3.6:5.0f} km/h)",
                    f"steer cmd: {steer_cmd:+.2f}   steer ang: {steer_angle_deg:+5.1f} deg",
                    f"throttle: {throttle_cmd:+.2f}",
                    f"yaw rate: {yaw_rate:+6.2f} rad/s   a_lat: {a_lat:+6.2f} m/s^2",
                    f"Fy est: {fy / 1000.0:+7.2f} kN",
                ]
                self._draw_hud(draw, lines, font, font_size, render_scale, bbox_xs, bbox_ys)

            if reuse_canvas:
                prev_bbox = (
                    int(max(0, min(bbox_xs) - dirty_pad_px)),
                    int(max(0, min(bbox_ys) - dirty_pad_px)),
                    int(min(img_w, max(bbox_xs) + dirty_pad_px)),
                    int(min(img_h, max(bbox_ys) + dirty_pad_px)),
                )
                yield img.copy()   # img is reused next frame, so hand out a copy
            else:
                yield img

    def render(
        self,
        content=None,
        frame_sampling=2,
        skip=None,
        *,
        fast=True,
        show_hud=True,
        progress=True,
        progress_desc=None,
        render_scale=1.0,
    ):
        return list(self._iter_render_frames(
            content=content,
            frame_sampling=frame_sampling,
            skip=skip,
            fast=fast,
            show_hud=show_hud,
            progress=progress,
            progress_desc=progress_desc,
            render_scale=render_scale,
        ))
