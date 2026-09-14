import os

from .engine import CarPhysicsEngine
from .agent import SteeringAgent
from .rl_agent import RLAgent, load_policy
from pcg_benchmark.probs import Problem
from pcg_benchmark.spaces import ArraySpace, FloatSpace, IntegerSpace, DictionarySpace
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pcg_benchmark.probs.racing.utils import (
    interpolate_curves,
    count_self_intersections,
    count_track_area_intersections,
    lowest_turn_seam_index,
    compute_offset_edges,
)
from pcg_benchmark.probs.utils import get_range_reward
from collections import OrderedDict

PX_PER_M = 5.0


# Trained policies live in the sibling model_training package, which is not
# importable as a module, so the path is resolved from this file.
_RL_RUNS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "model_training", "runs"))


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



def _conjunctive_mean(values, weights=None):
    """Weighted geometric mean: the aggregation for criteria that must ALL hold.

    An arithmetic mean lets a term satisfied everywhere pay for one that is
    not.  Measured on 200 random genomes across the five representations, 12 of
    the 14 terms scored above 0.95 more than half the time (oob 100%, braking
    zone 94%, completion 94%, speed entropy 94%, extent 92%, time 92%), and the
    arithmetic mean reported mean quality 0.69-0.84 for content no designer
    would accept.  A geometric mean makes every term necessary, which is what a
    conjunction of design criteria is.

    Values are floored at 0.02 rather than allowed to reach zero.  A term a
    representation cannot satisfy at all would otherwise flatten the whole
    score and leave the search no gradient, which is the failure the additive
    form was avoiding.

    Why not simpler: a plain minimum is the strictest conjunction, but it
    reports only the worst term and gives no credit for improving any other.
    Why not more complex: a soft minimum with a temperature adds a constant
    that would itself need calibrating.
    """
    vals = np.clip(np.asarray(values, dtype=float), 0.02, 1.0)
    if weights is None:
        return float(np.exp(np.mean(np.log(vals))))
    w = np.asarray(weights, dtype=float)
    return float(np.exp(np.sum(w * np.log(vals)) / np.sum(w)))


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

    # Cache bounds: info dicts hold dense curves (tens of KB each), summary
    # tuples are tiny, trajectories are the largest (full state history) but
    # only rendering and explicit evaluate() calls need them.
    _INFO_CACHE_MAX = 1024
    _SIM_SUMMARY_CACHE_MAX = 8192
    _TRAJECTORY_CACHE_MAX = 32

    def _get_info_cache_key(self, track_points):
        """Dictionary key for a set of track points: a tuple of (x, y) tuples
        (arrays cannot be dict keys, tuples can)."""
        arr = np.asarray(track_points).reshape(-1, 2)
        return tuple((row[0], row[1]) for row in arr)

    def _extract_content(self, content):
        if content is None:
            return self._default_track_points
        if isinstance(content, dict):
            return content.get("track_points", self._default_track_points)
        return content

    # Free-form racing splines its control points in genome order, producing a
    # self-crossing "ball of yarn"; reordering into a non-crossing tour first
    # fixes that.  Only this class needs it.  Radial visits points in angle
    # order and the constructive decoders return points already tracing a valid
    # loop, so reordering would break the loop; those subclasses set it False.
    _untangle_control_points = True

    @staticmethod
    def _untangle_2opt(pts, max_passes=60):
        """Reorder a closed polygon's vertices into a non-self-intersecting tour
        via 2-opt: while any two edges cross, reverse the span between them.
        Each reversal removes a crossing and strictly shortens the tour, so this
        always terminates at a simple polygon (standard convex-hull-racetrack
        method; see the procedural-racetrack literature)."""
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

    def _normalize_track_points(self, track_points):
        if track_points is None:
            track_points = self._default_track_points
        elif isinstance(track_points, dict):
            track_points = track_points.get("track_points", self._default_track_points)
        track_points = np.asarray(track_points, dtype=float)
        if track_points.ndim == 1:
            track_points = track_points.reshape(-1, 2)

        if len(track_points) >= 4:
            # Untangle first (free-form racing only), so the seam is chosen on
            # the final non-crossing loop.
            if self._untangle_control_points:
                track_points = self._untangle_2opt(track_points)
            # Rotate seam to the smoothest vertex (smallest local turn angle).
            best_i = lowest_turn_seam_index(track_points)
            if best_i != 0:
                track_points = np.vstack([track_points[best_i:], track_points[:best_i]])
        return track_points

    @staticmethod
    def _curve_turn_profile(curve_points):
        """(signed turn at each interior sample, arc weight of each sample)."""
        cp = np.asarray(curve_points, dtype=float)
        if len(cp) < 3:
            return np.zeros(0), np.zeros(0)
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

    def _find_corners(self, signed_ang, balance_w):
        """Corners as a designer counts them: consecutive same-direction
        curved samples accumulated until a sustained straight or a change of
        direction, keeping every run that swept at least corner_min_turn_deg.
        Returns (turns_rad, arclens_m), one entry per corner.

        Thresholding on the ACCUMULATED turn rather than on one sample's angle
        is what makes the count invariant to sampling density.  A per-vertex
        count is not: at the 5.0 m curve step a vertex only passes a 20 degree
        test when the corner radius is under 14.3 m, and the 24 reference
        circuits sit at a 14.28 m 10th percentile, so a per-vertex count
        reports 0 turns for Suzuka, Brands Hatch, Budapest, Oschersleben and
        Sao Paulo.

        A corner closes only after a straight run longer than
        corner_gap_max_m, not on the first straight sample: the tile and hex
        decoders sample an arc as large turn jumps separated by straight
        densification samples, so resetting on a single straight sample would
        split every arc into sub-threshold pieces and find no corners at all.
        Splines are unaffected, their real straights far exceed the
        tolerance."""
        P = self._QUALITY_PARAMS
        min_turn = np.deg2rad(P["corner_min_turn_deg"])
        # Per-sample turn angle below which a sample counts as straight.
        # Derived from the same flat-out curvature threshold the
        # straight-balance term uses, times the curve step, so it states one
        # curvature rule rather than a second number that would have to be
        # re-tuned whenever the step changes.  At 5.0 m that is 4.0 degrees.
        thresh = np.deg2rad(
            float(P["straight_curv_deg_per_m"]) * self._curve_step())
        gap_max = float(P["corner_gap_max_m"])
        turns, arclens = [], []
        acc, run_len, cur_sign, gap_len = 0.0, 0.0, 0, 0.0

        def _close():
            nonlocal acc, run_len, cur_sign, gap_len
            if abs(acc) >= min_turn:
                turns.append(abs(acc))
                arclens.append(run_len)
            acc, run_len, cur_sign, gap_len = 0.0, 0.0, 0, 0.0

        for i, a in enumerate(signed_ang):
            w = float(balance_w[i])
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
            acc += a
            run_len += w
            cur_sign = sgn
            gap_len = 0.0  # a curved sample resumes the corner
        _close()
        return turns, arclens

    def _make_curve(self, track_points):
        """Dense polyline used for simulation, scoring, and rendering.

        Each representation builds its centreline the way its geometry
        requires: the base problem interpolates a spline through control
        points that are not themselves road geometry, while the decoders
        whose waypoints already ARE the road (voronoi polygons, tile arcs)
        override this and only re-space what they produced, because a spline
        through them would add wobble the design does not have.

        Whatever the shape, the result leaves here at one arc-length spacing,
        so how finely a track is measured is not a property of which
        representation produced it."""
        pts = np.asarray(track_points, dtype=float).reshape(-1, 2)
        return self._resample_uniform(
            interpolate_curves(pts, samples_per_segment=self._spline_samples(pts)),
            step=self._curve_step())

    # Arc-length spacing of every representation's curve, in metres.
    #
    # 5.0 m is the spacing of the 25 real circuits in
    # model_training/racetrack-database-master, measured across the files:
    # mean 5.00, median 5.00, min 4.58, max 5.25.  Every quality band in
    # _QUALITY_PARAMS was calibrated by running those circuits through this
    # same code, so generated content is measured at the resolution its
    # reference values were measured at.
    #
    # An absolute figure rather than a fraction of track width, because the
    # anchor is the reference database and not the road.  Corner-cutting from
    # the re-spacing stays small: at the 10 m minimum corner radius the
    # quality function enforces, a 5.0 m chord cuts 0.31 m at mid-chord
    # against a 16 m track width, and at the 38 m radius of the tightest hex
    # tile it cuts 0.08 m.
    #
    # One number for all four: a difference between them would be a
    # difference in how finely each representation is measured.
    _CURVE_STEP_M = 5.0

    def _curve_step(self) -> float:
        """Arc-length spacing of every representation's curve, in metres."""
        return float(self._CURVE_STEP_M)

    def _spline_samples(self, control_points: np.ndarray) -> int:
        """Samples per spline segment, chosen so the spline's own chords are
        already shorter than the resample step.  The uniform pass then only
        re-spaces points; if the spline were sampled coarsely it would cut
        corners first and no re-spacing could put them back.

        Aims at half a step per chord: the control polygon underestimates the
        spline's arc length, since the curve bows outside its own hull, so
        targeting a full step would leave chords longer than one.  Clamped at
        256 because a degenerate genome can put two control points 1000 m
        apart, and at 8 so a tiny loop still gets a curve."""
        pts = np.asarray(control_points, dtype=float).reshape(-1, 2)
        n_seg = max(1, len(pts))
        if n_seg < 2:
            return 8
        closed = np.vstack([pts, pts[:1]])
        perimeter = float(np.sum(np.linalg.norm(np.diff(closed, axis=0), axis=1)))
        target = 2.0 * perimeter / (n_seg * max(self._curve_step(), 1e-9))
        return int(np.clip(np.ceil(target), 8, 256))

    @staticmethod
    def _resample_uniform(points: np.ndarray, *, step: float) -> np.ndarray:
        """Resample a closed polyline to uniform arc-length spacing.

        Points land on the polyline, so this re-spaces a shape without
        changing it: a tile corner stays the six-chord approximation of its
        arc that the decoder built, it just stops being measured at a
        different density from every other representation."""
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
        track_points = self._normalize_track_points(track_points)
        if len(track_points) == 0:
            self._final_target = None
        else:
            # Tracks are closed loops: the lap finishes back at the start.
            self._final_target = track_points[0]
        return track_points

    @staticmethod
    def _closed_curve_length(points) -> float:
        """Arc length of a closed polyline, including the closing segment.

        Some representations hand back a curve whose last point repeats the
        first; that duplicate is dropped before closing so its edge is not
        counted twice."""
        p = np.asarray(points, dtype=float).reshape(-1, 2)
        if len(p) < 2:
            return 0.0
        if np.allclose(p[0], p[-1], atol=1e-9):
            p = p[:-1]
        if len(p) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(np.vstack([p, p[:1]]), axis=0), axis=1)))

    @staticmethod
    def _compute_turn_angles(points: np.ndarray) -> np.ndarray:
        """Return valid interior turn angles (radians) for a polyline."""
        diffs = points[1:] - points[:-1]
        if len(diffs) < 2:
            return np.array([], dtype=float)
        v1 = diffs[:-1]
        v2 = diffs[1:]
        v1_norm = np.linalg.norm(v1, axis=1, keepdims=True)
        v2_norm = np.linalg.norm(v2, axis=1, keepdims=True)
        valid = (v1_norm[:, 0] > 1e-3) & (v2_norm[:, 0] > 1e-3)
        v1_unit = np.zeros_like(v1)
        v2_unit = np.zeros_like(v2)
        v1_unit[valid] = v1[valid] / v1_norm[valid]
        v2_unit[valid] = v2[valid] / v2_norm[valid]
        # Dot product of consecutive unit directions (x1*x2 + y1*y2 per row),
        # clipped into arccos's valid input range.
        dots = v1_unit[:, 0] * v2_unit[:, 0] + v1_unit[:, 1] * v2_unit[:, 1]
        dots = np.clip(dots, -1.0, 1.0)
        return np.arccos(dots)[valid]

    def _get_cached_simulation_summary(self, track_points, max_steps=None, *, curve_points=None):
        """Simulate a policy and cache summary stats.

        If curve_points are provided they are used directly, skipping re-interpolation.
        """
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
        # Steps where the car is off the road (more than half a track width
        # from the centerline).  This is the on-road signal, distinct from the
        # geometry staying inside the map, which is a hard validity requirement
        # handled in info()/quality().
        half_width = 0.5 * float(self._track_width)
        offroad_steps = 0
        while not done and steps < max_steps:
            action = self._agent.act(state)
            state, _, done, _ = self.step(action, track_points=None)
            x, y = float(state[0]), float(state[1])
            if x < x_min or x > x_max or y < y_min or y > y_max:
                break
            # Two passes, not one.  TrackGeometry.project searches a
            # 60-segment window around the hint, so the second call re-centres
            # that window on the first pass's answer and can reach a nearer
            # segment the first window's edge cut off.  Collapsing them to a
            # single projection raises mean off-road fraction from 0.091 to
            # 0.139 on racing and 0.086 to 0.137 on racingradial, measured over
            # 12 genomes at seed 4242.
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

    def _trajectory_offroad_frac(self, trajectory):
        """Fraction of a supplied trajectory's steps where the car is off the
        road (> half a track width from the centerline).  Used when info() is
        given an explicit trajectory instead of running the cached sim."""
        if trajectory is None or len(trajectory) == 0 or len(self._curve_points) < 2:
            return 1.0
        agent = getattr(self, '_agent', None)
        if agent is None:
            return 0.0
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
        """Normalized entropy of the driven speed profile (Loiacono,
        Cardamone and Lanzi 2011): how evenly the lap's time is spread over
        the speed range.  0 = the whole lap at one speed, 1 = equal time in
        every speed band (fast straights AND slow corners both exist).
        Each simulation step is one equal-time sample."""
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
        # The driver is part of the key: the same track driven by a different
        # agent is a different trajectory.
        key = (self._get_info_cache_key(track_points), max_steps, self._driver)
        if key in self._trajectory_cache:
            return self._trajectory_cache[key]
        traj = self.evaluate({"track_points": track_points}, max_steps=max_steps)
        self._trajectory_cache[key] = traj
        return traj

    def clear_caches(self):
        self._info_cache = _BoundedCache(self._INFO_CACHE_MAX)
        self._trajectory_cache = _BoundedCache(self._TRAJECTORY_CACHE_MAX)
        self._simulation_summary_cache = _BoundedCache(self._SIM_SUMMARY_CACHE_MAX)

    def __init__(self, num_points=None, **kwargs):
        Problem.__init__(self, **kwargs)
        self._track_width = float(kwargs.get("track_width", 16.0))
        # Sized from the 25 real circuits, whose bounding-box spans run
        # 813-2171 m (median 1442): 1500 m clears the median real span and puts
        # every representation inside the length band rather than below it.
        # All geometry scales off _width (grid cell size, radial radii), so the
        # five representations grow together.  16 m track width matches the
        # 12-15 m of real circuits on multi-km layouts.
        self._width = float(kwargs.get("width", 1500.0))
        self._height = float(kwargs.get("height", 1500.0))
        self._diversity = float(kwargs.get("diversity", 0.4))
        self._default_max_steps = kwargs.get("max_steps", 7000)
        self._skip_render = kwargs.get("skip_render", False)

        # Which driver the simulation stage scores with.  "rl" loads the PPO
        # policy trained in model_training/ and is the default: the simulation
        # terms are meant to measure whether a track is driveable by an agent
        # that learned to drive, and a learned driver reports that directly.
        # "steering" selects the analytical pure-pursuit follower, kept for
        # comparison runs and as the fallback when the policy cannot load.
        #
        # The policy defaults to run12_stuckfix, which trains on this same
        # engine and measures best on generated tracks.  Over 12 genomes per
        # representation at seed 4242, against run10 (trained on the legacy engine):
        # racing 10/12 laps at 0.091 off-road against 9/12 at 0.372, radial
        # 11/12 at 0.086 against 7/12 at 0.354, and 1.08 s/eval against 1.59.
        # It also beats the analytical driver on both (9/12 at 0.226, 10/12 at
        # 0.093).  final.zip rather than best/: best/ is the highest-eval
        # snapshot on the held-out real circuits, final.zip is end-of-training,
        # and the workload here is generated tracks.
        self._driver = str(kwargs.get("driver", "rl"))
        self._rl_policy_path = kwargs.get(
            "rl_policy_path",
            os.path.join(_RL_RUNS_DIR, "run12_stuckfix", "final.zip"))
        # Observation scaling belongs to the checkpoint, not to the car: a
        # policy reads a different divisor as a different speed or steering
        # angle.  None keeps rl_agent's defaults, which match run10.
        self._rl_obs_max_speed = kwargs.get("rl_obs_max_speed")
        self._rl_obs_max_steering = kwargs.get("rl_obs_max_steering")
        # Loaded once on first use and reused: deserializing the checkpoint
        # costs far more than a forward pass, and every genome in a run is
        # driven by the same policy.
        self._rl_policy = None

        self._lap_finish_min_steps = int(kwargs.get("lap_finish_min_steps", 60))
        # Mid-lap checkpoint, the way lap timing works on a real circuit: the
        # lap only counts if the car was seen at the far side of the track.
        # A single "progress past 25%" threshold cannot do this job, because
        # the start line is also the finish line: a car sitting on it projects
        # onto the LAST segment as readily as the first, reads a progress
        # fraction near 1.0, and passes.  That is how a car which had driven
        # 0.0 m of a 4283 m track was scored as having completed the lap.
        # The tolerance only has to be wider than one step of travel: at the
        # engine's 85.5 m/s top speed and 0.1 s control step that is 8.55 m,
        # 0.2% of a 4283 m lap, so a 10% band cannot be jumped over.
        self._lap_checkpoint_frac = float(kwargs.get("lap_checkpoint_frac", 0.5))
        self._lap_checkpoint_tol = float(kwargs.get("lap_checkpoint_tol", 0.1))
        self._lap_checkpoint_seen = False
        self._steps_since_reset = 0
        self._curve_points = []

        self.clear_caches()
        self._final_target = None

        if num_points is None:
            # Resolution knob, and the only one that moves lap length without
            # moving the build box: inside the shared box a tour of 10/14/20
            # points runs 3695/4474/5780 m while the reach stays at 1070-1111.
            # 14 sits nearest the 4650 m median of the 25 real circuits.
            num_points = kwargs.get("num_points", 14)
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

        # Control points are sampled inside the shared build box, inset by the
        # bow allowance so the CURVE fills the box rather than overflowing it.
        x0, _y0, x1, _y1 = self._build_box()
        bow = self._SPLINE_BOW_FRAC * (x1 - x0)
        self._content_space = DictionarySpace({
            "track_points": ArraySpace((self.num_points, 2), FloatSpace(x0 + bow, x1 - bow)),
        })
        self._rebuild_control_space()

    # A Catmull-Rom curve bows outside the hull of its control points, so
    # sampling points right up to the build box would put the road past it.
    # Measured over 40 random genomes: 3.2% of the box side makes the curve's
    # reach 99.9% of the box.  The radial representation needs no such
    # allowance, since its spokes are short enough that the curve barely bows.
    _SPLINE_BOW_FRAC = 0.032

    # Fraction of the map reserved as border on each side.  The square tile
    # grid is the binding case: its road runs along interior cell centres, so
    # it reaches 1.5 cells in from each edge out of GRID_W = 11 columns, and it
    # cannot be given more room without changing the grid.
    _BUILD_INSET_FRAC = 1.5 / 11.0

    def _build_box(self):
        """(x0, y0, x1, y1): the region every representation may place track in.

        One rectangle shared by all five, so a comparison between them is a
        comparison of representations rather than of how much map each was
        handed.  Deriving each representation's own placement bound from this
        keeps the build AREA equal; the lap LENGTH each produces inside it is
        a property of the representation and is scored by the shared length
        band, not equalised here.

        Every representation reserves a border, and it has to: a control point
        on the wall already puts half the track width outside, and the spline
        bows outward between control points (median 24 m, worst 89 m past the
        control hull).  Tile and hex force their outer ring to grass, voronoi
        drops its boundary cells, and the spline representations inset their
        sampling range.  This is the single number behind all of those.
        """
        inset = self._BUILD_INSET_FRAC * float(min(self._width, self._height))
        return inset, inset, float(self._width) - inset, float(self._height) - inset

    def _rebuild_control_space(self):
        """Build the control space from this representation's quality band.

        Controlability only measures something when the target is reachable, so
        the length request is drawn from min_length..max_length: control and
        quality then ask for the same thing.

        This covers the range targets are drawn FROM, not the range
        controlability() scores against, which still uses width*16 for length
        (see there).

        Subclasses that adjust _QUALITY_PARAMS after super().__init__() must
        call this again.
        """
        P = self._QUALITY_PARAMS
        self._control_space = DictionarySpace({
            "length":    FloatSpace(float(P["min_length"]), float(P["max_length"])),
            "num_turns": IntegerSpace(1, self._MAX_TURNS),
        })

    # Upper end of the turn-count target, shared by all five representations.
    # It has to be a property of racetracks rather than of a genome: num_points
    # is a per-representation resolution knob (14 for the free spline, 20 for
    # the radial star, 15 for the constructive decoders), so deriving the
    # ceiling from num_points asked each representation for a different thing
    # and made controlability scores incomparable.  Measured by _find_corners,
    # the quantity num_turns actually reports, the 24 reference circuits run
    # 4 to 17 with a median of 10, so 24 spans the real range with headroom.
    _MAX_TURNS = 24

    def _make_agent(self, curve_points):
        """Build the driver named by the `driver` setting.

        A failed policy load falls back to SteeringAgent and says so, rather
        than raising: a missing checkpoint should not make the whole benchmark
        unimportable, and a silent fallback would report simulation scores from
        a different driver than the caller asked for.
        """
        if self._driver == "rl":
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
                self._driver = "steering"
        # The agent reads the car's mass, grip and drivetrain limits off the
        # engine, so its speed profile plans for the car that will drive it.
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
        """Fraction of the lap the agent has reached, in [0, 1].

        Not a segment index.  The two drivers index different polylines:
        SteeringAgent walks the benchmark's own curve, RLAgent walks a ring
        TrackGeometry resamples at 3 m, about 10x as many points on a typical
        track.  Comparing RLAgent's index against the benchmark's segment count
        made the lap-progress requirement pass on every step, so a car that had
        driven 0.0 m of a 4283 m track was scored as having completed the lap:
        it sits on the start line, so the distance-to-finish test passes, and
        the progress test was the only thing standing against it.
        """
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
                'avg_length': 0.0,
                'max_length': 0.0,
                'min_length': 0.0,
                'avg_turn': 0.0,
                'max_turn': 0.0,
                'min_turn': 0.0,
                'num_turns': 0,
                'steps': 0,
                'finished': False,
                'speed_entropy': 0.0,
                'offroad_frac': 1.0,
                'track_points': track_points,
                'trajectory_end': track_points[0] if len(track_points) > 0 else None,
                'curve_points': track_points,
            }

        # The driver is part of the key: this dict carries the simulation
        # results, so the same track scored with a different agent is a
        # different entry rather than a stale hit.
        cache_key = (self._get_info_cache_key(track_points), self._driver)
        if use_cache and cache_key in self._info_cache and trajectory is None:
            return self._info_cache[cache_key]

        # Control-point spacing statistics.  These describe the genome's own
        # polyline, not the track, so they stay on track_points.
        diffs = track_points[1:] - track_points[:-1]
        segment_lengths = np.linalg.norm(diffs, axis=1)
        avg_length = float(np.mean(segment_lengths))
        max_length = float(np.max(segment_lengths))
        min_length = float(np.min(segment_lengths))

        turn_angles = self._compute_turn_angles(track_points)
        avg_turn = float(np.mean(turn_angles)) if turn_angles.size > 0 else 0.0
        max_turn = float(np.max(turn_angles)) if turn_angles.size > 0 else 0.0
        min_turn = float(np.min(turn_angles)) if turn_angles.size > 0 else 0.0

        curve_points = self._make_curve(track_points)

        # Turns as a designer counts them, from the same walker the corner
        # terms use, so every representation reports the same quantity.
        # Counting control-point vertices instead made the number a property
        # of the genome's resolution rather than of the track: the free spline
        # carries 14 control points and so could never report more than 14
        # turns, while the voronoi polygon routinely reported over 30, against
        # a control range of 1..24 shared by both.
        num_turns = len(self._find_corners(
            *self._curve_turn_profile(curve_points))[0])

        # Lap length is measured on the DRIVEN curve, closed, so it means the
        # same thing for every representation and the same thing a real
        # circuit's quoted length means.  Measuring the control polyline
        # instead would under-report the spline representations by ~9%, since
        # their curve bows away from its control points, while the constructive
        # decoders would be unaffected because their curve IS their polyline.
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
            "avg_length": avg_length,
            "max_length": max_length,
            "min_length": min_length,
            "avg_turn": avg_turn,
            "max_turn": max_turn,
            "min_turn": min_turn,
            "num_turns": num_turns,
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
    # Three stages, each scoring only once the previous is satisfied (benchmark
    # house style; cf. zelda, sokoban, loderunnertile):
    #   1. soundness  - usable geometry: in bounds, no self-intersection, length
    #   2. shape      - the layout reads like a designed circuit
    #   3. simulation - the driving agent completes a lap at a decent pace
    #
    # Inside a stage the terms are grouped by what they measure, using the five
    # factors Togelius, De Nardi & Lucas (2006) give as the sources of fun in a
    # racing game.  Factor to term:
    #
    #   1. feeling of speed         straight_balance, start_straight,
    #                               braking_zone   (geometry)
    #                               time_score     (driven lap)
    #   2. challenge                min_radius, hairpin
    #   3. challenge adjusted to    no term of its own.  The two challenge
    #      the driver               thresholds are set from this car: 1.2 deg/m
    #                               is the radius its grip runs out at at its
    #                               25 m/s top speed, 10 m the radius it can
    #                               still turn.  Change the car and they move.
    #   4. variation of challenge   curvature_entropy (geometry)
    #                               speed_entropy     (driven lap)
    #                               Loiacono, Cardamone & Lanzi (2011) score
    #                               both halves of that pair for the same
    #                               reason.
    #   5. drift and slip in        NOT SCORED.  The engine runs a combined-slip
    #      corners                  tyre model but exposes slip only inside
    #                               _integrate, so no term reads it.  This is
    #                               the factor the quality function misses.
    #
    # Validity is Prasetya and Maulidevi's (2016) closed, continuous,
    # non-intersecting.  Closure and continuity hold by construction in all five
    # representations, so only non-intersection can be violated and only it is
    # scored (geom_score).  It is graded rather than binary, which departs from
    # that paper: the free-form spline's random content self-intersects about 38
    # times, and a binary gate would floor it with no gradient to climb.
    #
    # extent_score sits under no factor.  It is a plausibility floor: without it
    # two parallel straights one tile apart max out the speed group.
    #
    # Subclasses tune thresholds by overriding _QUALITY_PARAMS.
    #
    # Reference values cited below are measured from the 25 real circuits in
    # model_training/racetrack-database-master (Indianapolis excluded: it is an
    # oval and registers as all-straight with no corners).

    _QUALITY_PARAMS = {
        # ── Stage 1: soundness ────────────────────────────────────────────
        # Lap length (m), measured on the driven curve.  The band is the middle
        # 80% of the 25 real circuits, which run 2296-7000 with a 4650 median;
        # the tails are one oval-ish sprint (Norisring, 2296) and one outlier
        # (Spa, 7000), so scoring against the full range would make the term
        # pass everything.  Every representation is tuned to generate into this
        # band, with medians of 3861-5015, so the term discriminates on design
        # rather than on which representation produced the track.
        # FIA Appendix O (2026) Supplement 2 sets 3.5 km as the minimum for F1,
        # sports and GT circuits and Art. 7.2 recommends no more than 7 km, so
        # the band also sits inside the sanctioned range.
        "min_length":        3900.0,
        "max_length":        5900.0,
        # Self-intersection, the one validity condition that can be violated.
        "angles_on_curve":     True,   # geometry checks on dense curve vs control points
        "geom_area_check":     True,   # also count track-area overlap intersections
        "geom_max_violations":   20,   # geometry score reaches 0 at this many intersections

        # ── Geometry primitives shared by several shape terms ─────────────
        # Curvature below this counts as straight.  0.8 deg/m is a 72 m radius,
        # which at the 1.3 tyre friction coefficient the engine uses is the
        # radius the car can hold at 30 m/s, above its 25 m/s top speed.  So
        # "straight" here means FLAT OUT, not geometrically straight.  FIA
        # Appendix O (2026) Art. 7.7 calls a change of direction a corner below
        # a 300 m radius, an 18x stricter line; the two must not be mixed, and
        # the start_straight and braking_zone targets below are set against this
        # flat-out definition rather than the FIA one.
        "straight_curv_deg_per_m": 0.8,
        # Corners: accumulated same-direction turns >= corner_min_turn_deg count
        # as one corner.
        "corner_min_turn_deg":   30.0,
        # Straight gap bridged inside one corner.  It sits just above the
        # coarsest arc-sampling step (tile 18 m, hex 13 m on a 120 deg corner
        # and 20 m on a 60 deg one) and far below any real straight, so
        # arc-sampled corners stitch together without merging two corners
        # across a genuine straight.
        "corner_gap_max_m":      20.0,

        # ── Stage 2, speed group (Togelius factor 1) ──────────────────────
        # Straight balance: fraction of lap arc length that is straight.
        # Scale-free, so no representation is locked out by its edge scale.
        # Real circuits measure 0.20-0.40 (median 0.31), i.e. below this band.
        "straight_frac_min":    0.10,
        "straight_frac_lo":     0.30,
        "straight_frac_hi":     0.75,
        "straight_frac_max":    0.95,
        # Start straight: arc length of the straight run containing curve index
        # 0 (the seam is pre-rotated to the flattest vertex).  Shared by every
        # representation, per the one-quality-function-for-all rule.  FIA
        # Appendix O (2026) Art. 7.7 requires at least 250 m between the start
        # line and the first corner, but measures it against its own R < 300 m
        # corner definition, not the flat-out one used here.
        # Both bands come from the 25 real circuits measured through this same
        # code, as Tukey fences: full marks across the interquartile range,
        # falling to zero at 1.5 IQR beyond each quartile.  A one-sided
        # "longer is always better" band is what made these free.  Under it a
        # start straight of 50 m scored the same as a real circuit's 552 m
        # median, and a track that was one enormous straight scored 1.0 on the
        # braking zone; 94% of random genomes scored above 0.95 on it.
        # Zero below 50 m, rising linearly to full marks at 200 m and holding
        # there.  A DESIGN CHOICE, not a measurement: the 25 real circuits run
        # p25 349 m, p50 552 m, p75 897 m through this same code, so 200 m is
        # deliberately below what a real circuit has.  The target is "there is
        # a real straight off the line", not "the straight is as long as
        # Monza's".  The band is open above 200 m because a longer start
        # straight is not a defect; a track that is ONE enormous straight is
        # caught by braking_zone, which is two-sided.
        "start_straight_zero_m":   50.0,
        "start_straight_lo_m":    200.0,
        # Braking zone: longest straight anywhere on the lap, the overtaking
        # spot a designed circuit has and a random loop lacks.
        # Real longest straight: p25 744, p50 838, p75 1028.
        "braking_zone_zero_m":    319.0,
        "braking_zone_lo_m":      744.0,
        "braking_zone_hi_m":     1028.0,
        "braking_zone_max_m":    1453.0,

        # ── Stage 2, challenge group (Togelius factors 2 and 3) ───────────
        # Hairpin: one accumulated same-direction turn of hairpin_lo_deg or
        # more that is genuinely tight (1.2 deg/m = radius <= 48 m, the radius
        # at which the car's grip runs out at its 25 m/s top speed), not a wide
        # sweep.  Nothing rewards having a hairpin: the count feeds only the
        # excess penalty below and the hairpin_count diagnostic.
        "hairpin_lo_deg":            150.0,
        "hairpin_min_curv_deg_per_m":  1.2,
        # A hairpin is a signature corner some circuits have and some do not:
        # measured on the 24 reference circuits the count runs 0-3 with a
        # median of 0, never a string of them.
        "hairpin_max_count":         2.0,
        "hairpin_count_zero":        5.0,
        # Minimum corner radius: 10 m is the F1 floor (Monaco hairpin; the car
        # is 5 m long), below ~6 m a corner is undrivable.  Real circuits sit at
        # 11-37 m.  Only free-form spline hairpins come near it; the
        # constructive decoders are far wider.
        "min_corner_radius_zero_m":    6.0,
        "min_corner_radius_lo_m":     10.0,

        # ── Stage 2, variation group (Togelius factor 4) ──────────────────
        # Curvature entropy: how varied the corners are, measured on the
        # geometry alone.  Loiacono, Cardamone & Lanzi (2011) score a track on
        # the entropy of TWO profiles, curvature and speed, and argue that this
        # variety is what makes a track interesting.  Only the speed half was
        # scored here before, so this restores the pair the paper proposes.
        #
        # It also does not saturate the way a threshold term does.  Asking "is
        # there a 200 m straight" is a yes/no question every representation
        # eventually answers yes to; asking "how spread out are the corner
        # radii" has no ceiling that good content bumps into.
        #
        # 8 bins matches the speed entropy term.  The 5.0 deg/m top of the
        # range is an 11.5 m radius, the tightest corner the 25 real circuits
        # contain and just inside the 10 m floor min_radius_score enforces, so
        # the last bin means "as tight as a real circuit ever gets".
        "curvature_bins":             8,
        "curvature_max_deg_per_m":  5.0,
        # Full marks at the variety of the most varied real circuits.  Measured
        # on those 25 circuits through this same code: median 0.336, p90 0.402,
        # max 0.422.  Generated content currently runs 0.175-0.433, so the
        # target is reachable without being free.
        "curvature_entropy_lo":     0.40,

        # ── Stage 2, layout group (no Togelius factor) ────────────────────
        # Footprint: bounding-box aspect ratio and span relative to the map.
        "extent_aspect_zero":   0.15,
        "extent_aspect_lo":     0.55,
        "extent_span_zero":     0.10,
        "extent_span_lo":       0.45,

        # ── Stage 3: simulation ───────────────────────────────────────────
        # Lap time as average lap speed (m/s): Togelius factor 1 measured from
        # the driven lap.  The agent tops out near 25, so vmax 30 keeps the
        # "too fast" branch off corner-cutting laps.  A modelling choice, not a
        # published figure.
        "time_vmax":           30.0,
        "time_vmin":            8.0,
        # Speed profile entropy (Loiacono, Cardamone & Lanzi 2011): the driven
        # half of the entropy pair, Togelius factor 4 from the driver's side.
        "speed_bins":              8,
        "speed_max":            25.0,  # agent's target top speed (m/s)
        # Two-sided, same Tukey fences on the 25 real circuits: their driven
        # speed entropy runs p25 0.574, p50 0.606, p75 0.658.  The old
        # one-sided 0.55 floor put 94% of random genomes above 0.95.
        "speed_entropy_zero":   0.448,
        "speed_entropy_lo":     0.574,
        "speed_entropy_hi":     0.658,
        "speed_entropy_max":    0.784,
        # Fraction of driven steps the car may spend off the road before this
        # term hits zero.  A designed track can be driven while staying on it.
        "onroad_max_frac":           0.20,
    }

    # Shape terms, grouped by the factor each one measures.  All are computed on
    # the dense curve, so no term depends on how many vertices a representation
    # happens to use.
    #
    # The per-term weights are the flat weights this grouping replaces, and each
    # group's weight is the sum of its members', so the grouping moves no score.
    # The weights came from measured discrimination in a GA run: start_straight
    # and braking_zone separated designed content from random content, so they
    # carry 2.0; straight_balance and extent are a floor rather than an
    # objective, so they carry 0.5.  No paper sets them.
    _SHAPE_GROUPS = {
        # Factor 1: how much of the lap is taken flat out, and whether there is
        # one straight long enough to brake hard for.
        "speed": {
            "start_straight_score":    2.0,   # straight at the start line
            "braking_zone_score":      2.0,   # longest straight on the lap
            "straight_balance_score":  0.5,   # straight share of the lap (scale-free)
        },
        # Factors 2 and 3: corners tight enough to matter, with both thresholds
        # set from what this car can drive.
        "challenge": {
            "min_radius_score":        1.0,   # no corner tighter than the car can turn
            "hairpin_score":    1.0,   # not more than 2 hairpins
        },
        # Factor 4, geometry half.  Held at 1.0 rather than the 2.0 of
        # start_straight and braking_zone, which earned that weight from a GA
        # run; this term has not been through one.
        "variation": {
            "curvature_entropy_score": 1.0,   # spread of corner radii (Loiacono et al. 2011)
        },
        # No factor: a plausibility floor on the footprint.
        "layout": {
            "extent_score":            0.5,   # compact footprint spread over the map
        },
    }

    def _quality_terms(self, info):
        """Compute the quality terms shared by all racing variants.
        Returns a dict of terms, each in [0, 1], or None when the info does
        not describe a usable track."""
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

        # Out-of-bounds: 1.0 only when the whole curve stays inside the map
        # margin, fading linearly to 0 as the worst violation approaches four
        # track widths (trapezoid with the plateau pinned at zero violation,
        # like sokoban's heuristic term).
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

        # Completion: how far around the track the agent got.
        # For closed loops distance-to-goal is degenerate (start == goal, so a
        # car that crashes at the spawn point would score 1.0); use the
        # arc-position of the trajectory end along the curve instead.
        finished = info.get('finished', False)
        trajectory_end = info.get('trajectory_end', None)
        if trajectory_end is None:
            trajectory_end = points[-1]
        trajectory_end = np.asarray(trajectory_end, dtype=float)
        if finished:
            completion = 1.0
        elif len(curve_points) > 1:
            nearest = int(np.argmin(np.linalg.norm(curve_points - trajectory_end, axis=1)))
            completion = nearest / len(curve_points)
        else:
            dist_to_goal = float(np.linalg.norm(points[-1] - trajectory_end))
            curve_total_length = float(np.sum(np.linalg.norm(curve_points[1:] - curve_points[:-1], axis=1))) if len(curve_points) > 1 else 1.0
            completion = 1.0 - min(dist_to_goal / (curve_total_length + 1e-6), 1.0)

        # Lap length in the preferred range, measured on the driven curve so
        # every representation is scored on the same quantity (see info()).
        if len(curve_points) > 1:
            total_length = self._closed_curve_length(curve_points)
            length_score = float(get_range_reward(
                total_length, 0.0, P["min_length"], P["max_length"], 4.0 * P["max_length"]))
        else:
            length_score = 0.0

        # Self-intersection: 1.0 only at zero violations, fading linearly to 0
        # at geom_max_violations (trapezoid with the plateau pinned at zero).
        geom_pts = curve_points if P["angles_on_curve"] else points
        geom_violations = int(count_self_intersections(geom_pts))
        if P["geom_area_check"]:
            geom_violations += int(count_track_area_intersections(
                curve_points,
                track_width=float(self._track_width),
                min_cross_index_gap=2,
            ))
        geom_score = float(get_range_reward(
            geom_violations, 0, 0, 0, P["geom_max_violations"]))

        # Straight balance, curvature profile, and corner structure — all measured on the
        # dense curve so every representation is judged by the same yardstick.
        cp_d     = curve_points[1:] - curve_points[:-1]
        cp_len   = np.linalg.norm(cp_d, axis=1)
        cp_total = float(np.sum(cp_len))
        if cp_total > 1e-6 and len(curve_points) >= 3:
            cn = np.linalg.norm(cp_d, axis=1, keepdims=True)
            ok = cn[:, 0] > 1e-3
            cu = np.zeros_like(cp_d)
            cu[ok] = cp_d[ok] / cn[ok]
            cv1, cv2 = cu[:-1], cu[1:]
            cross = cv1[:, 0] * cv2[:, 1] - cv1[:, 1] * cv2[:, 0]
            dots  = cv1[:, 0] * cv2[:, 0] + cv1[:, 1] * cv2[:, 1]
            dots  = np.clip(dots, -1.0, 1.0)
            signed_ang = np.arctan2(cross, dots)  # signed turn at each interior point
            cp_ang = np.abs(signed_ang)

            # Straight balance: how much of the lap is straight, by arc
            # length.  Local curvature (turn angle per metre) below the
            # threshold counts as straight, so the measure is independent of
            # sampling density and of the representation's absolute scale.
            balance_w = 0.5 * (cp_len[:-1] + cp_len[1:])
            local_curv = np.rad2deg(cp_ang) / np.maximum(balance_w, 1e-9)
            straight_frac = float(
                np.sum(balance_w[local_curv < P["straight_curv_deg_per_m"]])
                / max(float(np.sum(balance_w)), 1e-9))
            straight_balance_score = float(get_range_reward(
                straight_frac, P["straight_frac_min"],
                P["straight_frac_lo"], P["straight_frac_hi"],
                P["straight_frac_max"],
            ))

            # Curvature entropy: bin the same local curvature by ARC LENGTH,
            # not by sample count, so a representation that samples its curve
            # densely does not score differently from one that samples it
            # coarsely.  This is the curvature half of Loiacono et al.'s pair;
            # speed_entropy_score below is the other half.
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
            curvature_entropy_score = float(get_range_reward(
                curvature_entropy, 0.0, P["curvature_entropy_lo"], 1.0, 1.0))

            corner_turns, corner_arclens = self._find_corners(
                signed_ang, balance_w)

            # Start straight: the contiguous straight run containing curve
            # index 0, where the car spawns.  Walking forward from the start and
            # backward from the end lets the run extend through the seam.  A lap
            # with no corner counts as one full-length straight; other terms
            # reject such loops.
            straight_mask = local_curv < P["straight_curv_deg_per_m"]
            n_ang = len(signed_ang)
            start_straight_len = 0.0
            k = 0
            while k < n_ang and straight_mask[k]:
                start_straight_len += float(balance_w[k])
                k += 1
            if k < n_ang:
                j = n_ang - 1
                while j > k and straight_mask[j]:
                    start_straight_len += float(balance_w[j])
                    j -= 1
            start_straight_score = float(get_range_reward(
                start_straight_len, P["start_straight_zero_m"],
                P["start_straight_lo_m"], 1e12, 1e12))

            # Braking zone: the longest contiguous straight anywhere on the lap,
            # where a car builds speed and brakes hard for a corner.  Graded
            # rather than gated, so voronoi (straights pinned to its diagram
            # edge scale) earns partial credit instead of being locked out.
            longest_straight_len = 0.0
            run = 0.0
            for i in range(n_ang):
                if straight_mask[i]:
                    run += float(balance_w[i])
                    if run > longest_straight_len:
                        longest_straight_len = run
                else:
                    run = 0.0
            braking_zone_score = float(get_range_reward(
                longest_straight_len, P["braking_zone_zero_m"],
                P["braking_zone_lo_m"], P["braking_zone_hi_m"],
                P["braking_zone_max_m"]))

            # Hairpins: corner runs that are both large and genuinely tight.  A
            # hairpin is a signature corner, and real circuits have 0-3 (median
            # 1), never a string of them, so only the count is scored.
            hairpin_count = 0
            for turn, alen in zip(corner_turns, corner_arclens):
                turn_deg = float(np.rad2deg(turn))
                if alen > 1e-6 and turn_deg / alen >= P["hairpin_min_curv_deg_per_m"]:
                    if turn_deg >= P["hairpin_lo_deg"]:
                        hairpin_count += 1
            # Having no hairpin is not a fault.  Measured through this same
            # code on the 24 reference circuits (Indianapolis excluded),
            # hairpin_count is 0 at the 10th percentile, 0 at the 50th and 2 at
            # the 90th, and 12 of the 24 have none at all: Monza, Spa,
            # Silverstone, Catalunya, Melbourne, Mexico City, Austin, Sakhir,
            # Sao Paulo, Sochi, Spielberg and Moscow Raceway.  150 degrees
            # accumulated at 1.2 deg/m is a genuine hairpin and most circuits
            # do not have one.
            #
            # So the band plateaus over 0..hairpin_max_count and only falls
            # away above it: a string of hairpins is not a circuit, but their
            # absence is ordinary.  Requiring one instead scored those twelve
            # circuits 0.000 here, which through the conjunctive mean of the
            # challenge group put Monza and Spa among the worst tracks the
            # function can see.
            hairpin_score = float(get_range_reward(
                hairpin_count, 0.0, 0.0,
                float(P["hairpin_max_count"]), float(P["hairpin_count_zero"])))

            # Minimum corner radius: the spline representations can place two
            # control points close together and produce a 1-3 m corner no car
            # can drive.  Scoring the tightest SUSTAINED corner (each run's
            # average curvature to radius) rather than a single-sample spike
            # keeps this invariant to sampling density, so it does not fire on
            # coarse tile/hex arc sampling.  One scale-free rule for all
            # representations; the plateau sits below every constructive
            # decoder's fixed radius, so only impossible spline hairpins lose.
            tightest_radius_m = 1e12  # no corner -> unbounded radius -> full score
            for turn, alen in zip(corner_turns, corner_arclens):
                turn_deg = float(np.rad2deg(turn))
                if turn_deg > 1e-6:
                    # radius = arc_length / angle(radians); alen and turn are the
                    # corner's total arc length and total swept angle.
                    r = alen / float(turn)
                    tightest_radius_m = min(tightest_radius_m, r)
            min_radius_score = float(get_range_reward(
                tightest_radius_m,
                P["min_corner_radius_zero_m"],
                P["min_corner_radius_lo_m"], 1e12, 1e12))
        else:
            straight_balance_score = 0.0
            curvature_entropy = 0.0
            curvature_entropy_score = 0.0
            start_straight_score = 0.0
            braking_zone_score = 0.0
            hairpin_score = 0.0
            hairpin_count = 0
            start_straight_len = 0.0
            longest_straight_len = 0.0
            tightest_radius_m = 0.0
            min_radius_score = 0.0

        # Footprint: a designed circuit occupies a compact region of the map
        # rather than a thin ribbon (two parallel straights one tile apart can
        # otherwise max out the straight-balance term).
        if len(curve_points) > 1:
            bbox_w = float(np.max(curve_points[:, 0]) - np.min(curve_points[:, 0]))
            bbox_h = float(np.max(curve_points[:, 1]) - np.min(curve_points[:, 1]))
            longer = max(bbox_w, bbox_h, 1e-6)
            aspect = min(bbox_w, bbox_h) / longer
            span = longer / max(float(min(self._width, self._height)), 1e-6)
            aspect_band = float(get_range_reward(aspect, P["extent_aspect_zero"], P["extent_aspect_lo"], 1.0, 1.0))
            span_band   = float(get_range_reward(span,   P["extent_span_zero"],   P["extent_span_lo"],   1.0, 1.0))
            extent_score = aspect_band * span_band
        else:
            extent_score = 0.0

        # Lap time — expressed as an average-speed band over the actual track
        # length so it is representation-independent (a step-per-control-point
        # budget punishes representations that happen to use many vertices).
        steps = info.get('steps', 0)
        dt = float(getattr(getattr(self, '_engine', None), 'time_step', 0.1) or 0.1)
        track_len = max(float(cp_total), 1.0)
        min_steps = track_len / (P["time_vmax"] * dt)
        max_steps_for_scoring = track_len / (P["time_vmin"] * dt)
        if not finished:
            time_score = 0.0
        elif steps < min_steps:
            time_score = max(0.0, 1.0 - (min_steps - steps) / min_steps)
        elif steps > max_steps_for_scoring:
            time_score = max(0.0, 1.0 - (steps - max_steps_for_scoring) / max_steps_for_scoring)
        else:
            time_score = 1.0

        # Speed profile entropy (Loiacono, Cardamone & Lanzi 2011): rewards a
        # lap whose driving alternates fast and slow phases.  Like time_score
        # it only counts once the agent actually finishes the lap.
        if finished:
            speed_entropy_score = float(get_range_reward(
                float(info.get('speed_entropy', 0.0)),
                P["speed_entropy_zero"], P["speed_entropy_lo"],
                P["speed_entropy_hi"], P["speed_entropy_max"]))
        else:
            speed_entropy_score = 0.0

        # On-road: penalise time the CAR spends off the road (on the grass).
        # Full marks when the car is on the road the whole lap, a mild penalty
        # for a small excursion, and zero once it is off-road for more than
        # onroad_max_frac of the lap.  A designed track can be driven while
        # staying on the road; a bad one forces the car wide.
        offroad_frac = float(info.get('offroad_frac', 1.0))
        # Plateau at offroad_frac 0 (fully on-road -> 1.0), ramping DOWN the
        # right slope to 0 at onroad_max_frac (0.20).  get_range_reward with
        # plat_low = plat_high = 0 makes [min, 0] the full plateau, then the
        # down slope reaches 0 at max_value.
        on_road_score = float(get_range_reward(
            offroad_frac, -1.0, 0.0, 0.0, P["onroad_max_frac"]))

        terms = {
            # Stage 1: soundness
            "oob_score":                 oob_score,
            "geom_score":                geom_score,   # validity (Prasetya 2016)
            "length_score":              length_score,
            # Stage 2: shape, in _SHAPE_GROUPS order
            "start_straight_score":      start_straight_score,     # speed
            "braking_zone_score":        braking_zone_score,       # speed
            "straight_balance_score":    straight_balance_score,   # speed
            "min_radius_score":          min_radius_score,         # challenge
            "hairpin_score":      hairpin_score,     # challenge
            "curvature_entropy_score":   curvature_entropy_score,  # variation
            "extent_score":              extent_score,             # layout
            # Stage 3: simulation, the same qualities read off the driven lap
            "completion":                completion,           # validity
            "on_road_score":             on_road_score,        # validity
            "time_score":                time_score,           # factor 1
            "speed_entropy_score":       speed_entropy_score,  # factor 4
            # Raw values behind the structural terms.  Not scored, and free:
            # every one is already computed for the term above it.  Calibration
            # scripts and threshold audits read them.
            "start_straight_len_m":      float(start_straight_len),
            "longest_straight_m":        float(longest_straight_len),
            "curvature_entropy":         float(curvature_entropy),
            "hairpin_count":             int(hairpin_count),
            "tightest_corner_radius_m":  float(tightest_radius_m),
            "offroad_frac":              offroad_frac,
        }

        # One score per shape group: the weighted mean of that group's terms.
        # quality() weights the groups by the sum of their members' weights, so
        # this reproduces the flat weighted mean over all seven terms.  Exposed
        # because a per-factor axis is what the expressive-range plots need.
        for group, weights in self._SHAPE_GROUPS.items():
            keys = sorted(weights)
            terms["shape_%s_score" % group] = _conjunctive_mean(
                [terms[k] for k in keys], [weights[k] for k in keys])
        return terms

    def quality(self, info):
        terms = self._quality_terms(info)
        if terms is None:
            return 0.0

        # Hard requirement: a track whose geometry leaves the map is not a
        # racetrack.  Quality collapses, scaled by how far in-bounds it is so
        # the search keeps a gradient back toward validity.
        #
        # Self-intersection is deliberately NOT hard, only graded in soundness
        # below.  Gating on it would floor the free-form spline at ~0.075 with
        # no gradient, since its random content self-intersects ~38 times, and
        # the GA would have no way to evolve out of crossings.
        if terms["oob_score"] < 0.999:
            return 0.15 * terms["oob_score"]

        # Stage 1: soundness — self-intersection (graded) and sensible length.
        soundness = _conjunctive_mean(
            (terms["geom_score"], terms["length_score"]))

        # Stage 2: shape — the weighted mean of the four factor groups (see
        # _SHAPE_GROUPS).  A group's weight is the sum of its terms' weights, so
        # speed carries 4.5 of the 8.0: start_straight and braking_zone reward
        # features a designed circuit has and a random loop lacks.
        gw = {g: sum(w.values()) for g, w in self._SHAPE_GROUPS.items()}
        groups = sorted(gw)
        shape = _conjunctive_mean([terms["shape_%s_score" % g] for g in groups],
                                  [gw[g] for g in groups])

        # Stage 3: simulation — the same qualities measured from the driven lap
        # instead of the geometry.  completion and on_road are validity (can the
        # track be driven, and driven on the road; a random loop that forces the
        # car wide scores low here), time_score is Togelius factor 1 from the
        # driver's side, speed_entropy factor 4 and the speed half of Loiacono
        # et al.'s pair.  Equal-weighted: none of the four has been through a
        # discrimination sweep of the kind that set the shape weights.
        sim = _conjunctive_mean(
            (terms["completion"], terms["time_score"],
             terms["speed_entropy_score"], terms["on_road_score"]))

        # The three stages multiply.  The previous form banked
        # 0.25 * soundness additively, and soundness averages 0.91-0.95 on
        # random content, so a merely valid loop collected about a quarter of
        # the score before any design merit.  Quality 1.0 now requires every
        # term at 1.0: reachable in principle, hard in practice.  Each stage
        # still scales the next, so the gradient stays continuous.
        return soundness * shape * sim

    def _track_to_grid(self, curve_points: np.ndarray, grid_size: int = 20) -> np.ndarray:
        pts = np.asarray(curve_points, dtype=float)
        if len(pts) == 0:
            return np.zeros(grid_size * grid_size, dtype=float)
        xi = np.clip((pts[:, 0] / self._width  * grid_size).astype(int), 0, grid_size - 1)
        yi = np.clip((pts[:, 1] / self._height * grid_size).astype(int), 0, grid_size - 1)
        grid = np.zeros((grid_size, grid_size), dtype=float)
        grid[yi, xi] = 1.0
        return grid.ravel()

    def diversity(self, info1, info2):
        grid_size = 20
        _cp1 = info1.get('curve_points')
        _cp2 = info2.get('curve_points')
        g1 = self._track_to_grid(np.asarray(_cp1 if _cp1 is not None else [], dtype=float), grid_size)
        g2 = self._track_to_grid(np.asarray(_cp2 if _cp2 is not None else [], dtype=float), grid_size)
        spatial_frac = float(np.abs(g1 - g2).sum()) / (grid_size * grid_size)

        a1 = float(info1.get('avg_turn', 0.0))
        a2 = float(info2.get('avg_turn', 0.0))
        angle_frac = min(abs(a1 - a2) / np.pi, 1.0)

        # The 0.4 plateau is out of reach for four of the five representations.
        # spatial_frac is the symmetric difference of two 20x20 occupancy
        # grids, and a track only occupies ~15-25% of the grid, so two totally
        # unrelated tracks still overlap on all the empty cells.  Measured mean
        # blended distance: racing 0.27, radial 0.18, voronoi/tile/hex 0.13,
        # i.e. diversity saturates at 0.67/0.45/0.32 respectively rather than
        # at 1.0.  That systematically rates the free-form spline as twice as
        # diverse as the grid representations, which is an artifact of the
        # measure, not a property of the content.
        blended = 0.7 * spatial_frac + 0.3 * angle_frac
        return get_range_reward(blended, 0, self._diversity, 1.0)

    def controlability(self, info, control):
        """How close the content came to what was asked for.

        Both terms are trapezoids centred on the request: full marks inside
        the tolerance, falling to zero three tolerances out on either side.
        One rule, and symmetric, so missing by a given amount costs the same
        whichever way it was missed.

        Anchoring the ends at 0 and at an absolute ceiling instead made the
        two slopes wildly uneven and paid for overshoot.  Against a 6000 m
        target on a 0 to 24000 m envelope, 2250 m scored 0.44 while 9750 m
        scored 0.83, so a search maximising controlability was pushed toward
        longer tracks whatever the target was."""
        length_err = self._width * 0.6
        l_score = get_range_reward(
            info.get('total_length', 0.0),
            control['length'] - 3.0 * length_err,
            control['length'] - length_err,
            control['length'] + length_err,
            control['length'] + 3.0 * length_err,
        )
        turns_err = 2
        t_score = get_range_reward(
            info.get('num_turns', 0),
            control['num_turns'] - 3 * turns_err,
            control['num_turns'] - turns_err,
            control['num_turns'] + turns_err,
            control['num_turns'] + 3 * turns_err,
        )
        return (l_score + t_score) / 2.0

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

    # The frame pipeline, one method per stage:
    #   _track_edges        offset the centerline into left/right road edges
    #   _render_track_bg    draw the static background once (road + overlay)
    #   _draw_car           wheels, body, heading line
    #   _draw_lookahead     the agent's aim point (cyan circle + line)
    #   _draw_hud           telemetry text box in the top-left corner
    #   _iter_render_frames replay the trajectory, yield one frame at a time
    #   render              collect the generator into a list
    #
    # Every draw helper appends the pixel coordinates it touched to
    # bbox_xs/bbox_ys.  In fast mode each new frame only restores the small
    # background rectangle the previous frame drew over (a "dirty rectangle"),
    # instead of copying the whole background image again.

    def _track_edges(self, curve_px, half_width):
        """Left/right road-edge polylines in pixels, as lists of (x, y) tuples.

        The offsetting itself is utils.compute_offset_edges, the same routine
        the geometry soundness check runs in metres; this wrapper only feeds it
        pixel coordinates and converts the result to the tuple lists PIL draws
        from."""
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
        """Yield rendered frames one at a time (memory-safe for gif writers).

        Replays the cached trajectory over a background drawn once.  A HUD
        agent runs alongside the replay purely to recover what the real agent
        was doing at each step (its action and lookahead point), which is why
        agent.act runs for every trajectory step even when the frame itself
        is skipped by frame_sampling."""
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
            agent = SteeringAgent(curve_np, track_width=float(self._track_width),
                                  engine=engine)
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
