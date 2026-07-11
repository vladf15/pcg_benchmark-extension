from .engine import CarPhysicsEngine
from .agent import SteeringAgent
from pcg_benchmark.probs import Problem
from pcg_benchmark.spaces import ArraySpace, FloatSpace, IntegerSpace, DictionarySpace
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pcg_benchmark.probs.racing.utils import (
    interpolate_curves,
    count_self_intersections,
    count_track_area_intersections,
    lowest_turn_seam_index,
)
from pcg_benchmark.probs.utils import get_range_reward

PX_PER_M = 5.0



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

    def _normalize_track_points(self, track_points):
        if track_points is None:
            track_points = self._default_track_points
        elif isinstance(track_points, dict):
            track_points = track_points.get("track_points", self._default_track_points)
        track_points = np.asarray(track_points, dtype=float)
        if track_points.ndim == 1:
            track_points = track_points.reshape(-1, 2)

        if len(track_points) >= 4:
            # Rotate seam to the smoothest vertex (smallest local turn angle).
            best_i = lowest_turn_seam_index(track_points)
            if best_i != 0:
                track_points = np.vstack([track_points[best_i:], track_points[:best_i]])
        return track_points

    def _make_curve(self, track_points):
        """Dense polyline used for simulation, scoring, and rendering.

        The base problem interpolates a smooth spline through the control
        points; subclasses whose tracks are already polygons (voronoi)
        override this with plain densification instead."""
        return interpolate_curves(track_points, samples_per_segment=10)

    def _set_track_cache(self, track_points):
        track_points = self._normalize_track_points(track_points)
        if len(track_points) == 0:
            self._final_target = None
        else:
            # Tracks are closed loops: the lap finishes back at the start.
            self._final_target = track_points[0]
        return track_points

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
            key = (self._get_info_cache_key(track_points), max_steps, ('curve', curve_points.shape[0]))
        else:
            key = (self._get_info_cache_key(track_points), max_steps)
        cached = self._simulation_summary_cache.get(key)
        if cached is not None:
            return cached

        if curve_points is not None:
            # Note: this path aims the car along the first control-polyline
            # segment, while reset() aims it along the first curve segment.
            # A historical difference, kept because changing it would change
            # every recorded simulation result.
            if len(track_points) < 2:
                start_angle = 0.0
            else:
                start_angle = float(np.arctan2(
                    track_points[1][1] - track_points[0][1],
                    track_points[1][0] - track_points[0][0],
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
        while not done and steps < max_steps:
            action = self._agent.act(state)
            state, _, done, _ = self.step(action, track_points=None)
            x, y = float(state[0]), float(state[1])
            if x < x_min or x > x_max or y < y_min or y > y_max:
                break
            speeds.append(float(state[3]))
            steps += 1

        end_xy = state[:2].copy() if state is not None else None
        steps_len = steps + 1
        finished = (
            end_xy is not None
            and len(track_points) > 0
            and steps_len < max_steps
            and self._is_finished(end_xy, steps_len=steps_len)
        )
        speed_entropy = self._speed_profile_entropy(speeds)
        summary = (steps_len, finished, end_xy, speed_entropy)
        self._simulation_summary_cache[key] = summary
        return summary

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
        key = (self._get_info_cache_key(track_points), max_steps)
        if key in self._trajectory_cache:
            return self._trajectory_cache[key]
        traj = self.evaluate({"track_points": track_points}, max_steps=max_steps)
        self._trajectory_cache[key] = traj
        return traj

    def clear_caches(self):
        self._info_cache = {}
        self._trajectory_cache = {}
        self._simulation_summary_cache = {}

    def __init__(self, num_points=None, **kwargs):
        Problem.__init__(self, **kwargs)
        self._track_width = float(kwargs.get("track_width", 16.0))
        self._width = float(kwargs.get("width", 500.0))
        self._height = float(kwargs.get("height", 500.0))
        self._diversity = float(kwargs.get("diversity", 0.4))
        self._default_max_steps = kwargs.get("max_steps", 4000)
        self._skip_render = kwargs.get("skip_render", False)

        self._lap_finish_min_steps = int(kwargs.get("lap_finish_min_steps", 60))
        self._lap_finish_min_progress_frac = float(kwargs.get("lap_finish_min_progress_frac", 0.25))
        self._steps_since_reset = 0
        self._curve_points = []

        self._info_cache = {}
        self._trajectory_cache = {}
        self._simulation_summary_cache = {}
        self._final_target = None

        if num_points is None:
            num_points = kwargs.get("num_points", 20)
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

        self._content_space = DictionarySpace({
            "track_points": ArraySpace((self.num_points, 2), FloatSpace(0, min(self._width, self._height))),
        })
        self._control_space = DictionarySpace({
            "length":    FloatSpace(500.0, self._width * 16.0),
            "num_turns": IntegerSpace(1, self.num_points * 2),
        })

    def _setup_car(self, curve_points, start_angle):
        """Place the car and agent on a curve: create the physics engine and
        agent on first use, re-aim them afterwards.  Returns the reset state."""
        if not hasattr(self, '_engine') or self._engine is None:
            self._engine = CarPhysicsEngine(start_position=curve_points[0], start_angle=start_angle)
        else:
            self._engine.start_position = np.asarray(curve_points[0], dtype=float)
            self._engine.start_angle = start_angle
        if not hasattr(self, '_agent') or self._agent is None:
            self._agent = SteeringAgent(curve_points, track_width=self._track_width)
        else:
            self._agent.curve_points = curve_points

        if len(curve_points) > 0:
            self._final_target = np.asarray(curve_points[0], dtype=float)
        self._steps_since_reset = 0
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

    def _get_progress_index(self) -> int:
        agent = getattr(self, '_agent', None)
        if agent is None:
            return 0
        if hasattr(agent, 'current_idx'):
            return int(getattr(agent, 'current_idx') or 0)
        if hasattr(agent, 'current_segment_idx'):
            return int(getattr(agent, 'current_segment_idx') or 0)
        return 0

    def _is_finished(self, end_xy, *, steps_len: int) -> bool:
        if end_xy is None or self._final_target is None:
            return False
        dist = float(np.linalg.norm(np.asarray(end_xy, dtype=float) - np.asarray(self._final_target, dtype=float)))
        final_threshold = max(2.0, 0.2 * self._track_width)
        nseg = max(1, len(self._curve_points) - 1)
        min_progress_idx = int(max(1, self._lap_finish_min_progress_frac * nseg))

        if dist < final_threshold:
            return steps_len >= self._lap_finish_min_steps and self._get_progress_index() >= min_progress_idx

        progress_idx = self._get_progress_index()
        if progress_idx >= max(0, nseg - 2):
            near_threshold = max(18.0, 0.9 * self._track_width)
            if dist < near_threshold:
                return steps_len >= self._lap_finish_min_steps and progress_idx >= min_progress_idx

        return False

    def step(self, action, track_points=None):
        if track_points is not None:
            self._set_track_cache(track_points)

        state = self._engine.step(action)
        self._steps_since_reset += 1
        x, y = state[0], state[1]
        final_target = self._final_target
        if final_target is None:
            dist_to_final = float('inf')
        else:
            dist_to_final = np.hypot(final_target[0] - x, final_target[1] - y)
        done = self._is_finished(state[:2], steps_len=self._steps_since_reset)
        reward = -dist_to_final
        info = {"waypoint": self._get_progress_index()}
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
                'track_points': track_points,
                'trajectory_end': track_points[0] if len(track_points) > 0 else None,
                'curve_points': track_points,
            }

        cache_key = self._get_info_cache_key(track_points)
        if use_cache and cache_key in self._info_cache and trajectory is None:
            return self._info_cache[cache_key]

        diffs = track_points[1:] - track_points[:-1]
        segment_lengths = np.linalg.norm(diffs, axis=1)
        total_length = float(np.sum(segment_lengths))
        avg_length = float(np.mean(segment_lengths))
        max_length = float(np.max(segment_lengths))
        min_length = float(np.min(segment_lengths))

        turn_angles = self._compute_turn_angles(track_points)
        avg_turn = float(np.mean(turn_angles)) if turn_angles.size > 0 else 0.0
        max_turn = float(np.max(turn_angles)) if turn_angles.size > 0 else 0.0
        min_turn = float(np.min(turn_angles)) if turn_angles.size > 0 else 0.0
        num_turns = int(np.sum(turn_angles > np.deg2rad(20))) if turn_angles.size > 0 else 0

        curve_points = self._make_curve(track_points)
        if trajectory is None:
            steps, finished, end_xy, speed_entropy = self._get_cached_simulation_summary(track_points, curve_points=curve_points)
            trajectory_end = end_xy
        else:
            steps = len(trajectory)
            trajectory_end = trajectory[-1][:2] if len(trajectory) > 0 else None
            finished = steps < self._default_max_steps and self._is_finished(trajectory_end, steps_len=steps)
            # Trajectory rows are engine states [x, y, heading, speed, steer].
            traj_speeds = [float(s[3]) for s in trajectory if len(s) > 3]
            speed_entropy = self._speed_profile_entropy(traj_speeds)

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
            "track_points": track_points,
            "trajectory_end": trajectory_end,
            "curve_points": curve_points,
        }
        if use_cache:
            self._info_cache[cache_key] = info_dict
        return info_dict

    # ── Quality ────────────────────────────────────────────────────────
    # All racing variants share the same staged quality structure, following
    # the benchmark house style (see zelda, sokoban, loderunnertile): quality
    # is the plain average of three stages, and a later stage only starts
    # scoring once every earlier stage is fully satisfied:
    #
    #   1. soundness  — the geometry is a usable racetrack (in bounds, no
    #                   self-intersection, sensible total length),
    #   2. shape      — the layout reads like a designed circuit (straights,
    #                   rhythm, corner structure, footprint),
    #   3. simulation — the driving agent completes a lap at a decent pace.
    #
    # This mirrors zelda (playability only counts once player/key/door all
    # exist) and loderunnertile (a four-stage chain).  Random content
    # therefore tops out around 1/3 unless it happens to be fully sound, and
    # quality 1.0 requires all three stages to be perfect.
    # Subclasses tune behaviour by overriding _QUALITY_PARAMS (thresholds).

    _QUALITY_PARAMS = {
        "min_length":        1500.0,   # total track length plateau (metres)
        "max_length":        6500.0,
        "angles_on_curve":     True,   # geometry checks on dense curve vs control points
        "geom_area_check":     True,   # also count track-area overlap intersections
        "geom_max_violations":   20,   # geometry score reaches 0 at this many intersections
        # Lap-time band as average speed over the lap (m/s).  The agent tops out
        # around 25 m/s, so vmax=30 keeps the "too fast" branch from firing on
        # corner-cutting; vmin is the slowest acceptable average lap speed.
        "time_vmax":           30.0,
        "time_vmin":            8.0,
        "straight_thresh_deg":  4.0,   # per-sample angle below this = straight section
        "straight_ideal_frac":  0.20,  # longest-straight arc fraction for score 1.0
        # Curvature profile entropy (Loiacono, Cardamone & Lanzi 2011): the
        # per-sample turn angles of the dense curve, binned, arc-length
        # weighted.  High entropy = a real mix of straights, sweepers and
        # tight corners; 0 = all-straight or constant-radius circle.
        "curvature_bins":         16,
        "curvature_bin_max_deg": 24.0, # angles above this land in the top bin
        # Plateau calibrated on random content (2026-07-09): random splined
        # tracks sit at 0.6-0.8, random voronoi tops out at ~0.63, and a
        # plain oval lands well below.  0.45 keeps the term a guard against
        # degenerate shapes without punishing designed circuits (whose long
        # straights concentrate mass in the first bin).
        "curvature_entropy_lo":  0.45, # normalized-entropy plateau lower edge
        # Speed profile entropy (same paper): how evenly the lap's time is
        # spread over the speed range, measured from the driving agent.
        "speed_bins":              8,
        "speed_max":            25.0,  # agent's target top speed (m/s)
        # Calibrated on random content (2026-07-09): finished random laps
        # measure 0.6-0.99 (corner-heavy tracks force constant speed changes),
        # so 0.55 guards against monotone-speed laps while staying reachable
        # for every representation (voronoi max observed: 0.67).
        "speed_entropy_lo":     0.55,  # normalized-entropy plateau lower edge
        # Rhythm: how many distinct straights (each >= rhythm_min_frac of the
        # lap) a designed circuit should have.  Plateau [lo, hi], zero at max.
        "rhythm_min_frac":      0.08,
        "rhythm_lo":               3,
        "rhythm_hi":               5,
        "rhythm_max":              8,
        # Corners: accumulated same-direction turns >= corner_min_turn count as
        # one corner.  Density is corners per km of lap (real circuits sit
        # around 3-5/km); variety is the std of corner angles in degrees
        # (hairpins mixed with sweepers, not all-identical 90s).
        "corner_min_turn_deg":   30.0,
        "corner_density_lo":      2.0,
        "corner_density_hi":      6.0,
        "corner_density_max":    12.0,
        "corner_variety_lo_deg": 25.0,
        "corner_variety_hi_deg": 60.0,
        "corner_variety_max_deg": 120.0,
        # Footprint: bounding-box aspect ratio (thin ribbon loops score low)
        # and span of the longer side relative to the map.
        "extent_aspect_zero":   0.15,
        "extent_aspect_lo":     0.55,
        "extent_span_zero":     0.10,
        "extent_span_lo":       0.45,
    }

    # The shape-stage terms, averaged with equal weight (benchmark house
    # style).  All are computed on the dense curve (or its total length), so
    # they stay comparable across representations — none depend on how
    # densely a representation samples its polyline.
    _SHAPE_TERMS = (
        "straight_score",             # one real main straight
        "rhythm_score",               # several significant straights
        "corner_density_score",       # corner economy, not a zigzag
        "corner_variety_score",       # mix of tight and sweeping corners
        "curvature_entropy_score",    # diverse curvature profile (Loiacono et al. 2011)
        "extent_score",               # compact footprint spread over the map
    )

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

        # Total control-polyline length in the preferred range
        if len(points) > 1:
            total_length = float(np.sum(np.linalg.norm(points[1:] - points[:-1], axis=1)))
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

        # Straight sections, rhythm, and corner structure — all measured on the
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
            n_cs   = len(cp_len)
            seg_a  = np.zeros(n_cs)
            seg_a[:-1] = np.maximum(seg_a[:-1], cp_ang)  # end-angle of each segment
            seg_a[1:]  = np.maximum(seg_a[1:],  cp_ang)  # start-angle of each segment
            is_str   = seg_a < np.deg2rad(P["straight_thresh_deg"])
            # Find runs of consecutive straight samples: diff of the 0/1 mask
            # is +1 where a run starts and -1 where it ends.
            changes  = np.diff(is_str.astype(int), prepend=0, append=0)
            r_starts = np.where(changes == 1)[0]
            r_ends   = np.where(changes == -1)[0]
            run_lens_list = []
            for s, e in zip(r_starts, r_ends):
                run_lens_list.append(float(np.sum(cp_len[s:e])))
            run_lens = np.array(run_lens_list)
            max_s    = float(run_lens.max()) if len(run_lens) else 0.0
            straight_score = min(1.0, max_s / (cp_total * max(1e-9, P["straight_ideal_frac"])))

            # Curvature profile entropy (Loiacono, Cardamone & Lanzi 2011):
            # bin the per-sample turn angles (arc-length weighted so the
            # measure is independent of how densely a representation samples
            # its polyline) and reward an even spread of curvatures.
            n_cbins = int(P["curvature_bins"])
            ang_deg = np.rad2deg(cp_ang)
            ang_deg = np.clip(ang_deg, 0.0, float(P["curvature_bin_max_deg"]) - 1e-9)
            sample_w = 0.5 * (cp_len[:-1] + cp_len[1:])  # angle i sits between segments i and i+1
            hist, _ = np.histogram(ang_deg, bins=n_cbins,
                                   range=(0.0, float(P["curvature_bin_max_deg"])),
                                   weights=sample_w)
            total_w = float(np.sum(hist))
            if total_w > 0.0 and n_cbins >= 2:
                pbin = hist[hist > 0] / total_w
                curv_entropy = float(-np.sum(pbin * np.log(pbin))) / float(np.log(n_cbins))
            else:
                curv_entropy = 0.0
            curvature_entropy_score = float(get_range_reward(
                curv_entropy, 0.0, P["curvature_entropy_lo"], 1.0, 1.0))

            # Rhythm: a designed circuit alternates corners with several
            # significant straights, not just one main straight.
            n_significant = int(np.sum(run_lens >= P["rhythm_min_frac"] * cp_total)) if len(run_lens) else 0
            rhythm_score = float(get_range_reward(
                n_significant, 0, P["rhythm_lo"], P["rhythm_hi"], P["rhythm_max"],
            ))

            # Corners: consecutive curved samples with the same turn direction,
            # accumulated until a straight sample or a direction change.  Only
            # accumulated turns above corner_min_turn count as real corners —
            # this makes the count invariant to how densely a representation
            # samples its polyline.
            min_turn = np.deg2rad(P["corner_min_turn_deg"])
            thresh   = np.deg2rad(P["straight_thresh_deg"])
            corner_turns = []
            acc, cur_sign = 0.0, 0
            for a in signed_ang:
                if abs(a) < thresh:
                    if abs(acc) >= min_turn:
                        corner_turns.append(abs(acc))
                    acc, cur_sign = 0.0, 0
                    continue
                sgn = 1 if a > 0 else -1
                if sgn != cur_sign and cur_sign != 0:
                    if abs(acc) >= min_turn:
                        corner_turns.append(abs(acc))
                    acc = 0.0
                acc += a
                cur_sign = sgn
            if abs(acc) >= min_turn:
                corner_turns.append(abs(acc))

            corners_per_km = len(corner_turns) / max(cp_total / 1000.0, 1e-6)
            corner_density_score = float(get_range_reward(
                corners_per_km, 0.0,
                P["corner_density_lo"], P["corner_density_hi"], P["corner_density_max"],
            ))
            if len(corner_turns) >= 2:
                corner_std_deg = float(np.rad2deg(np.std(np.asarray(corner_turns))))
                corner_variety_score = float(get_range_reward(
                    corner_std_deg, 0.0,
                    P["corner_variety_lo_deg"], P["corner_variety_hi_deg"],
                    P["corner_variety_max_deg"],
                ))
            else:
                corner_variety_score = 0.0
        else:
            straight_score = 0.0
            curvature_entropy_score = 0.0
            rhythm_score = 0.0
            corner_density_score = 0.0
            corner_variety_score = 0.0

        # Footprint: a designed circuit occupies a compact region of the map
        # rather than a thin ribbon (two parallel straights one tile apart can
        # otherwise max out the straight/rhythm terms).
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
                0.0, P["speed_entropy_lo"], 1.0, 1.0))
        else:
            speed_entropy_score = 0.0

        return {
            # Soundness stage
            "oob_score":                 oob_score,
            "geom_score":                geom_score,
            "length_score":              length_score,
            # Shape stage (see _SHAPE_TERMS)
            "straight_score":            straight_score,
            "rhythm_score":              rhythm_score,
            "corner_density_score":      corner_density_score,
            "corner_variety_score":      corner_variety_score,
            "curvature_entropy_score":   curvature_entropy_score,
            "extent_score":              extent_score,
            # Simulation stage
            "completion":                completion,
            "time_score":                time_score,
            "speed_entropy_score":       speed_entropy_score,
        }

    def quality(self, info):
        terms = self._quality_terms(info)
        if terms is None:
            return 0.0

        # Stage 1: the geometry must be a usable racetrack.
        soundness = (terms["oob_score"] + terms["geom_score"] + terms["length_score"]) / 3.0

        # Stage 2: layout quality, unlocked only once the geometry is fully
        # sound (benchmark style: zelda only scores playability once player,
        # key and door all exist).
        shape = 0.0
        if soundness >= 1.0:
            shape_terms = []
            for term_name in self._SHAPE_TERMS:
                shape_terms.append(terms[term_name])
            shape = sum(shape_terms) / len(shape_terms)

        # Stage 3: driving quality, unlocked only once the layout is fully
        # right (loderunnertile-style stage chain).  completion is continuous
        # arc progress around the lap; time_score and speed_entropy_score are
        # already 0 unless the agent actually finished.
        sim = 0.0
        if shape >= 1.0:
            sim = (terms["completion"] + terms["time_score"] + terms["speed_entropy_score"]) / 3.0

        return (soundness + shape + sim) / 3.0

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

        blended = 0.7 * spatial_frac + 0.3 * angle_frac
        return get_range_reward(blended, 0, self._diversity, 1.0)

    def controlability(self, info, control):
        length_err = self._width * 0.6
        l_score = get_range_reward(
            info.get('total_length', 0.0), 0,
            control['length'] - length_err,
            control['length'] + length_err,
            self._width * 16.0,
        )
        turns_err = 2
        t_score = get_range_reward(
            info.get('num_turns', 0), 0,
            control['num_turns'] - turns_err,
            control['num_turns'] + turns_err,
            self.num_points * 2,
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
        for j in range(len(left_edge_f) - 1):
            lj = (int(round(left_edge_f[j][0])),      int(round(left_edge_f[j][1])))
            lk = (int(round(left_edge_f[j+1][0])),    int(round(left_edge_f[j+1][1])))
            rj = (int(round(right_edge_f[j][0])),     int(round(right_edge_f[j][1])))
            rk = (int(round(right_edge_f[j+1][0])),   int(round(right_edge_f[j+1][1])))
            bg_draw.polygon([lj, lk, rk, rj], fill=road_color)
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
        """Return the left/right road-edge polylines (pixel coordinates).

        Each curve point is pushed sideways by half_width along the
        perpendicular of the average of its two neighboring segment
        directions, so the road keeps a constant width through corners.
        (utils._compute_offset_edges does the same offsetting in metres for
        the geometry soundness check; this one works in pixels.)"""
        left_edge, right_edge = [], []
        if len(curve_px) < 2:
            return left_edge, right_edge

        # A closed curve may repeat its first point at the end; offsetting
        # that duplicate directly would use the wrong neighbors, so drop it
        # and re-append the first offset point at the end instead.
        has_dup_close = (
            len(curve_px) >= 3
            and np.allclose(curve_px[0], curve_px[-1], atol=1e-9, rtol=0.0)
        )
        base = curve_px[:-1] if has_dup_close else curve_px
        m = len(base)
        for j in range(m):
            if m >= 3:
                dir_prev = base[j] - base[(j - 1) % m]
                dir_next = base[(j + 1) % m] - base[j]
            else:
                # Degenerate two-point "track": treat it as a straight segment.
                dir_prev = base[1] - base[0] if j == 0 else base[j] - base[j - 1]
                dir_next = base[j] - base[j - 1] if j == m - 1 else base[j + 1] - base[j]

            avg_dir = dir_prev + dir_next
            norm = float(np.linalg.norm(avg_dir))
            if norm > 0.0:
                perp = np.array([-avg_dir[1], avg_dir[0]], dtype=float) / norm
            else:
                perp = np.zeros(2)

            left = base[j] + perp * half_width
            right = base[j] - perp * half_width
            left_edge.append((float(left[0]), float(left[1])))
            right_edge.append((float(right[0]), float(right[1])))

        if has_dup_close and left_edge:
            left_edge.append(left_edge[0])
            right_edge.append(right_edge[0])
        return left_edge, right_edge

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
        dt = 0.1
        mass = 1350.0
        font_size = 24
        if show_hud:
            agent = SteeringAgent(curve_np, track_width=float(self._track_width))
            agent.reset()
            engine = getattr(self, '_engine', None)
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
                x0 = int(max(0, min(bbox_xs) - dirty_pad_px))
                y0 = int(max(0, min(bbox_ys) - dirty_pad_px))
                x1 = int(min(img_w, max(bbox_xs) + dirty_pad_px))
                y1 = int(min(img_h, max(bbox_ys) + dirty_pad_px))
                prev_bbox = (x0, y0, x1, y1)

            if reuse_canvas:
                # img is reused next frame, so hand out a copy.
                yield img.copy()
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
