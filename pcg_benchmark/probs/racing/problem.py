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
from collections import OrderedDict

PX_PER_M = 5.0


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

    # Free-form racing splines its control points in genome order, which
    # crosses itself all over a large map (the "ball of yarn").  Reordering the
    # points into a non-self-crossing tour before splining fixes this.  Only the
    # base free-form racing needs it: radial already visits its points in angle
    # order, and the constructive decoders (tile, hex, voronoi) return points
    # that already trace a valid loop in order, so reordering them would BREAK
    # the loop.  Those subclasses set this False.
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

    def _make_curve(self, track_points):
        """Dense polyline used for simulation, scoring, and rendering.

        The base problem interpolates a smooth spline through the control
        points; subclasses whose decoded points already ARE the exact road
        geometry (voronoi polygons, tile arcs) override this with plain
        densification instead, because a spline would only add artifacts."""
        return interpolate_curves(track_points, samples_per_segment=10)

    @staticmethod
    def _densify_polyline(points: np.ndarray, *, max_step: float) -> np.ndarray:
        """Subdivide a closed polygon's edges so no segment exceeds max_step."""
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        if len(points) < 2:
            return points
        out = [points[0]]
        n = len(points)
        for i in range(n):
            a = points[i]
            b = points[(i + 1) % n]
            d = float(np.linalg.norm(b - a))
            if d <= 1e-12:
                continue
            steps = max(1, int(np.ceil(d / max(float(max_step), 1e-9))))
            for s in range(1, steps + 1):
                t = s / steps
                out.append(a * (1.0 - t) + b * t)
        out_np = np.asarray(out, dtype=float)
        if len(out_np) >= 2 and not np.allclose(out_np[0], out_np[-1], atol=1e-9):
            out_np = np.vstack([out_np, out_np[0]])
        return out_np

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
        # Count the steps where the CAR is off the road (more than half a track
        # width from the centerline, i.e. driving on the grass).  A good track
        # can be driven while staying on the road; a bad one forces the car
        # wide onto the grass.  This is the on-road / car-out-of-bounds signal
        # (distinct from the geometry staying inside the map, which is a hard
        # validity requirement handled in info()/quality()).
        half_width = 0.5 * float(self._track_width)
        offroad_steps = 0
        while not done and steps < max_steps:
            action = self._agent.act(state)
            state, _, done, _ = self.step(action, track_points=None)
            x, y = float(state[0]), float(state[1])
            if x < x_min or x > x_max or y < y_min or y > y_max:
                break
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
        key = (self._get_info_cache_key(track_points), max_steps)
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
        # Map is 750x750 m (1.5x the original 500) so corners land at realistic
        # racing radii: real slow hairpins are ~15-30 m and the F1-tightest
        # (Monaco) is ~10 m, but at 500 m the fixed-grid corners were down at
        # ~11-21 m and the splines produced sub-car-length (~1-8 m) hairpins.
        # Everything geometric scales off _width (grid cell size, radial radii,
        # control-space length), so all five representations grow together.
        # Track width stays 16 m on purpose: real circuits are 12-15 m wide on
        # multi-km layouts, so a thinner road on a bigger map is more authentic.
        self._width = float(kwargs.get("width", 750.0))
        self._height = float(kwargs.get("height", 750.0))
        self._diversity = float(kwargs.get("diversity", 0.4))
        self._default_max_steps = kwargs.get("max_steps", 4000)
        self._skip_render = kwargs.get("skip_render", False)

        self._lap_finish_min_steps = int(kwargs.get("lap_finish_min_steps", 60))
        self._lap_finish_min_progress_frac = float(kwargs.get("lap_finish_min_progress_frac", 0.25))
        self._steps_since_reset = 0
        self._curve_points = []

        self.clear_caches()
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
                'offroad_frac': 1.0,
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

    # ── Quality ────────────────────────────────────────────────────────
    # All racing variants share the same staged quality structure, following
    # the benchmark house style (see zelda, sokoban, loderunnertile): quality
    # is the plain average of three stages, and a later stage only starts
    # scoring once every earlier stage is fully satisfied:
    #
    #   1. soundness  — the geometry is a usable racetrack (in bounds, no
    #                   self-intersection, sensible total length),
    #   2. shape      — the layout reads like a designed circuit (balance of
    #                   straights and corners, corner variety, curvature
    #                   profile, footprint),
    #   3. simulation — the driving agent completes a lap at a decent pace.
    #
    # This mirrors zelda (playability only counts once player/key/door all
    # exist) and loderunnertile (a four-stage chain).  Random content
    # therefore tops out around 1/3 unless it happens to be fully sound, and
    # quality 1.0 requires all three stages to be perfect.
    # Subclasses tune behaviour by overriding _QUALITY_PARAMS (thresholds).

    _QUALITY_PARAMS = {
        # Length band scaled 1.5x with the 750 m map (was 1500-6500 on 500 m),
        # so tracks keep the same relative length / complexity at the new scale.
        "min_length":        2250.0,   # total track length plateau (metres)
        "max_length":        9750.0,
        "angles_on_curve":     True,   # geometry checks on dense curve vs control points
        "geom_area_check":     True,   # also count track-area overlap intersections
        "geom_max_violations":   20,   # geometry score reaches 0 at this many intersections
        # Lap-time band as average speed over the lap (m/s).  The agent tops out
        # around 25 m/s, so vmax=30 keeps the "too fast" branch from firing on
        # corner-cutting; vmin is the slowest acceptable average lap speed.
        "time_vmax":           30.0,
        "time_vmin":            8.0,
        "straight_thresh_deg":  4.0,   # per-sample angle below this = straight section
        # Straight balance: fraction of the lap's arc length that is straight
        # (local curvature below straight_curv_deg_per_m).  Scale-free, so no
        # representation is structurally locked out the way absolute
        # corners-per-km and longest-straight-in-metres bands locked out
        # voronoi (edge scale ~33 m) and tile.  Band calibrated 2026-07-12:
        # voronoi reaches 0.25-0.39, tile 0.29-0.50, radial 0.62-0.79, random
        # splines sit too straight at ~0.88 (fade region, the search must add
        # corners); a zigzag (~0) and a near-featureless loop (>0.95) score 0.
        "straight_curv_deg_per_m": 0.8,  # below this curvature = straight (radius ~72 m)
        "straight_frac_min":    0.10,
        "straight_frac_lo":     0.30,
        "straight_frac_hi":     0.75,
        "straight_frac_max":    0.95,
        # Curvature profile entropy (Loiacono, Cardamone & Lanzi 2011): the
        # per-sample turn angles of the dense curve, binned, arc-length
        # weighted.  High entropy = a real mix of straights, sweepers and
        # tight corners; 0 = all-straight or constant-radius circle.
        "curvature_bins":         16,
        "curvature_bin_max_deg": 24.0, # angles above this land in the top bin
        # Plateau recalibrated 2026-07-12 after the tile true-geometry decode:
        # tile's constant-radius corners occupy few curvature bins, capping its
        # entropy at 0.44-0.49 on random content (elites with more straights
        # sit lower still), while splines and voronoi reach 0.7-0.8.  0.35
        # keeps the term a pure guard against ovals and featureless loops
        # (~0.1-0.2) that every representation can satisfy.
        "curvature_entropy_lo":  0.35, # normalized-entropy plateau lower edge
        # Speed profile entropy (same paper): how evenly the lap's time is
        # spread over the speed range, measured from the driving agent.
        "speed_bins":              8,
        "speed_max":            25.0,  # agent's target top speed (m/s)
        # Calibrated on random content (2026-07-09): finished random laps
        # measure 0.6-0.99 (corner-heavy tracks force constant speed changes),
        # so 0.55 guards against monotone-speed laps while staying reachable
        # for every representation (voronoi max observed: 0.67).
        "speed_entropy_lo":     0.55,  # normalized-entropy plateau lower edge
        # Corners: accumulated same-direction turns >= corner_min_turn count as
        # one corner.  Variety is the std of corner angles in degrees
        # (hairpins mixed with sweepers, not all-identical 90s).
        "corner_min_turn_deg":   30.0,
        # Max straight gap (arc length, m) bridged inside a single corner before
        # the corner is closed.  Sized above the arc-sampling densification step
        # (constructive decoders leave ~one straight sample, a few metres, per
        # corner-arc jump) but far below any real straight, so it stitches
        # arc-sampled corners together without merging genuinely separate
        # corners across a real straight.
        "corner_gap_max_m":      20.0,
        "corner_variety_lo_deg": 25.0,
        "corner_variety_hi_deg": 60.0,
        "corner_variety_max_deg": 120.0,
        # Footprint: bounding-box aspect ratio (thin ribbon loops score low)
        # and span of the longer side relative to the map.
        "extent_aspect_zero":   0.15,
        "extent_aspect_lo":     0.55,
        "extent_span_zero":     0.10,
        "extent_span_lo":       0.45,
        # Start straight: a designed circuit opens with a straight section at
        # the start line (grid + acceleration zone).  The seam is already
        # rotated to the flattest vertex, so this measures the arc length of the
        # contiguous straight run containing curve index 0.  SHARED by every
        # representation (2026-07-22, user rule: one quality function for all):
        # >= 50 m scores full, 20-50 m ramps up, < 20 m scores nothing.  50 m is
        # reachable by every representation (voronoi, the tightest, reaches it in
        # ~29% of random genomes and its GA can push to ~99 m), so it is a real
        # discriminator without locking any representation out.  The per-rep
        # overrides (old 250/375/105) are REMOVED.
        "start_straight_zero_m": 20.0,
        "start_straight_lo_m":   50.0,
        # Braking zone: the longest straight anywhere on the lap.  Graded from
        # 40 m (nothing below) to 200 m (full).  A good circuit has one long
        # straight for a braking/overtaking zone (design literature); a random
        # loop of short segments has none.  Voronoi tops out ~100 m (edges
        # pinned to the diagram scale), so it earns partial credit here and can
        # never max the term - the honest reason its random content should not
        # score as high as a representation that can build a real straight.
        "braking_zone_zero_m":   40.0,
        "braking_zone_lo_m":    200.0,
        # Hairpin (added 2026-07-19): at least one accumulated same-direction
        # corner of hairpin_lo_deg or more whose average curvature is at least
        # hairpin_min_curv_deg_per_m (i.e. the turn is actually tight, not a
        # 150-degree sweep of 200 m radius).  1.2 deg/m = radius <= ~48 m.
        "hairpin_lo_deg":            150.0,
        "hairpin_min_curv_deg_per_m":  1.2,
        # Too many hairpins (2026-07-22): a hairpin is a signature feature
        # corner; famous circuits have 0-2 (Silverstone/Monza 0, Spa/Suzuka/
        # Monaco exactly 1), never a string of them.  0-2 tight large turns
        # score full, then ramp down to 0 at 5 (an all-hairpin loop is not a
        # circuit).
        "hairpin_max_count":         2.0,
        "hairpin_count_zero":        5.0,
        # Minimum corner radius (added 2026-07-22): the tightest sustained
        # corner must be at least the plateau to score full marks, ramping to
        # zero at the "zero" edge.  10 m is the F1 floor (Monaco hairpin, the
        # tightest real racing corner; the car is 5 m long).  Below ~6 m the
        # corner is undrivable, so that is the zero edge.  Scale-free and shared
        # by all representations; both edges sit below every constructive
        # decoder's fixed corner radius (tile ~31 m, hex 60deg ~17 m at the
        # 750 m map scale), so only the free-form spline hairpins are penalised.
        "min_corner_radius_zero_m":    6.0,
        "min_corner_radius_lo_m":     10.0,
        # On-road (2026-07-23): fraction of driven steps the car may spend off
        # the road before the on-road term hits zero.  Full marks at 0% (car on
        # the road the whole lap), mild penalty for a small excursion, zero at
        # 20%+ off-road.  A designed track can be driven on-road; a bad one
        # forces the car wide onto the grass (random racing ~65% off-road,
        # voronoi ~25%, tile ~0%).
        "onroad_max_frac":           0.20,
    }

    # The shape-stage terms.  All are computed on the dense curve (or its total
    # length), so they stay comparable across representations — none depend on
    # how densely a representation samples its polyline.  corner_variety,
    # curvature_entropy and hairpin were REMOVED from scoring 2026-07-23 (passed
    # by ~99% of random content, and mutually redundant); they remain in the
    # terms dict as diagnostics only.
    _SHAPE_TERMS = (
        "straight_balance_score",     # balance of straights vs corners (scale-free)
        "extent_score",               # compact footprint spread over the map
        "start_straight_score",       # straight section at the start line
        "braking_zone_score",         # at least one long straight (braking/overtaking)
        "hairpin_excess_score",       # not more than 2 hairpins
        "min_radius_score",           # no corner tighter than a car can drive
    )

    # Shape terms are weighted, not equal-averaged.  The terms that genuinely
    # separate a designed circuit from a random loop are start_straight (a
    # proper start line) and braking_zone (at least one long straight for
    # braking/overtaking); they carry the most weight so the search has a real
    # gradient to climb and random content, which lacks them, scores low.
    # straight_balance and extent are a lighter floor.  This is a lever that
    # lowers random-content scores WITHOUT locking any representation out: it
    # rewards features (a long straight, a start straight) that a random loop
    # lacks, rather than punishing a representation for its inherent geometry.
    _SHAPE_WEIGHTS = {
        "straight_balance_score":  0.5,
        "extent_score":            0.5,
        "min_radius_score":        1.0,
        "hairpin_excess_score":    1.0,
        "start_straight_score":    2.0,
        "braking_zone_score":      2.0,
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

            # Corners: consecutive curved samples with the same turn direction,
            # accumulated until a SUSTAINED straight or a direction change.  Only
            # accumulated turns above corner_min_turn count as real corners —
            # this makes the count invariant to how densely a representation
            # samples its polyline.
            #
            # A corner is closed only after a straight run longer than
            # corner_gap_max_m, not on the first straight sample.  This matters
            # for the constructive decoders (tile, hex): their corner arcs are
            # sampled as a few large turn jumps with straight densification
            # samples between them (e.g. 20,0,20,0,20 for one hex corner), so
            # resetting on a single straight sample would split every arc into
            # sub-threshold pieces and detect no corners at all.  Bridging short
            # straight gaps stitches the arc back into one corner.  Splines are
            # unaffected: their corners are already continuous runs and real
            # straights are far longer than the gap tolerance.
            min_turn = np.deg2rad(P["corner_min_turn_deg"])
            thresh   = np.deg2rad(P["straight_thresh_deg"])
            gap_max  = float(P["corner_gap_max_m"])
            corner_turns = []
            corner_arclens = []   # arc length of each corner run (for hairpin tightness)
            acc, run_len, cur_sign = 0.0, 0.0, 0
            gap_len = 0.0         # arc length of the current straight gap inside a corner

            def _close_corner():
                nonlocal acc, run_len, cur_sign, gap_len
                if abs(acc) >= min_turn:
                    corner_turns.append(abs(acc))
                    corner_arclens.append(run_len)
                acc, run_len, cur_sign, gap_len = 0.0, 0.0, 0, 0.0

            for i, a in enumerate(signed_ang):
                w = float(balance_w[i])
                if abs(a) < thresh:
                    if cur_sign == 0:
                        continue  # not inside a corner yet: plain straight
                    # Inside a corner: tolerate a short straight gap (arc-sampled
                    # decoders), but a long straight run ends the corner.
                    gap_len += w
                    run_len += w
                    if gap_len > gap_max:
                        run_len -= gap_len  # don't count the trailing straight
                        _close_corner()
                    continue
                sgn = 1 if a > 0 else -1
                if sgn != cur_sign and cur_sign != 0:
                    _close_corner()
                acc += a
                run_len += w
                cur_sign = sgn
                gap_len = 0.0  # a curved sample resumes the corner
            _close_corner()

            if len(corner_turns) >= 2:
                corner_std_deg = float(np.rad2deg(np.std(np.asarray(corner_turns))))
                corner_variety_score = float(get_range_reward(
                    corner_std_deg, 0.0,
                    P["corner_variety_lo_deg"], P["corner_variety_hi_deg"],
                    P["corner_variety_max_deg"],
                ))
            else:
                corner_variety_score = 0.0

            # Start straight: the contiguous straight run containing curve
            # index 0 (the seam, already rotated to the flattest vertex, and
            # the exact spot where the car spawns).  Walk forward from the
            # start and backward from the end of the angle array so the run
            # may extend through the seam; a lap with no corner at all counts
            # as one full-length straight (other terms reject such loops).
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

            # Braking zone: the longest contiguous straight ANYWHERE on the lap.
            # A designed circuit has at least one long straight that lets the car
            # build speed and then brake hard for a corner (the overtaking spot);
            # a random loop made of short segments has none.  Measured on random
            # content this separates good from random (a good circuit reaches
            # 500 m+, random voronoi tops out ~100 m).  GRADED, not a gate, and
            # starting at 40 m so voronoi (whose straights are pinned to the
            # ~50 m diagram edge scale and cannot chain long) still earns partial
            # credit and is never locked out: it just cannot max this one term,
            # which is the honest reason its random content should not score as
            # high as a rep that can build a real braking zone.
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
                P["braking_zone_lo_m"], 1e12, 1e12))

            # Hairpin: the largest accumulated same-direction turn among the
            # corner runs that are actually tight (average curvature at or
            # above the threshold).  Graded from 0 so the search has a
            # gradient toward tighter, longer corners.  Also count how many
            # such large tight turns there are: a hairpin is a signature FEATURE
            # corner, and real circuits have 0-2 of them (Silverstone/Monza 0,
            # Spa/Suzuka/Monaco exactly 1), never a string of them.
            best_hairpin_deg = 0.0
            hairpin_count = 0
            for turn, alen in zip(corner_turns, corner_arclens):
                turn_deg = float(np.rad2deg(turn))
                if alen > 1e-6 and turn_deg / alen >= P["hairpin_min_curv_deg_per_m"]:
                    best_hairpin_deg = max(best_hairpin_deg, turn_deg)
                    if turn_deg >= P["hairpin_lo_deg"]:
                        hairpin_count += 1
            hairpin_score = float(get_range_reward(
                best_hairpin_deg, 0.0,
                P["hairpin_lo_deg"], 1e12, 1e12))
            # Too many hairpins: full marks for 0..hairpin_max_count (2) tight
            # large turns, then ramps down to 0 at hairpin_count_zero.  A track
            # that is a string of hairpins is not a circuit.  The plateau is
            # [-1, max_count] so counts 0/1/2 all score 1.0 and the DOWN slope
            # penalises 3+ toward zero at the zero edge.
            hairpin_excess_score = float(get_range_reward(
                hairpin_count, -1.0, 0.0,
                float(P["hairpin_max_count"]), float(P["hairpin_count_zero"])))

            # Minimum corner radius: a real car cannot take a corner tighter
            # than roughly the F1 floor (Monaco hairpin ~10 m; the car itself is
            # 5 m long).  The free-form spline representations (racing, radial)
            # can otherwise place two control points close together and make a
            # corner of 1-3 m radius that no car can physically drive.  We
            # score the tightest SUSTAINED corner (each corner run's average
            # curvature -> its radius), not a single-sample spike, so the term
            # is invariant to how densely a representation samples its polyline
            # and does not fire on the coarse tile/hex arc sampling.  This is a
            # single scale-free rule applied identically to all representations;
            # the plateau sits below every constructive decoder's fixed corner
            # radius (tile ~31 m, hex 60deg ~17 m), so it only penalises the
            # impossible spline hairpins and never locks out a representation.
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
            curvature_entropy_score = 0.0
            corner_variety_score = 0.0
            start_straight_score = 0.0
            braking_zone_score = 0.0
            hairpin_score = 0.0
            hairpin_excess_score = 0.0
            hairpin_count = 0
            start_straight_len = 0.0
            longest_straight_len = 0.0
            best_hairpin_deg = 0.0
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
                0.0, P["speed_entropy_lo"], 1.0, 1.0))
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

        return {
            # Soundness stage
            "oob_score":                 oob_score,
            "geom_score":                geom_score,
            "length_score":              length_score,
            # Shape stage (see _SHAPE_TERMS)
            "straight_balance_score":    straight_balance_score,
            "extent_score":              extent_score,
            "start_straight_score":      start_straight_score,
            "braking_zone_score":        braking_zone_score,
            "hairpin_excess_score":      hairpin_excess_score,
            "min_radius_score":          min_radius_score,
            # Diagnostics only — NOT scored (2026-07-23): corner_variety,
            # curvature_entropy and hairpin are passed by ~99% of random content
            # for every representation, so they only inflated the score without
            # discriminating; corner_variety/curvature_entropy also duplicate
            # each other, and hairpin duplicates hairpin_excess.  Kept here so
            # the thesis can still report the (literature-standard) curvature
            # entropy value, but they no longer contribute to quality.
            "corner_variety_score":      corner_variety_score,
            "curvature_entropy_score":   curvature_entropy_score,
            "hairpin_score":             hairpin_score,
            # Raw diagnostics for the structural terms (not scored directly;
            # used by calibration scripts and threshold audits)
            "start_straight_len_m":      float(start_straight_len),
            "longest_straight_m":        float(longest_straight_len),
            "hairpin_best_turn_deg":     float(best_hairpin_deg),
            "hairpin_count":             int(hairpin_count),
            "tightest_corner_radius_m":  float(tightest_radius_m),
            "offroad_frac":              offroad_frac,
            # Simulation stage
            "completion":                completion,
            "time_score":                time_score,
            "speed_entropy_score":       speed_entropy_score,
            "on_road_score":             on_road_score,
        }

    # Stage weights.  Soundness is nearly free for the constructive decoders
    # (their loops are drivable by construction), so it carries the least
    # weight; the discriminating quality lives in shape and simulation.  The
    # three sum to 1 so quality stays in [0, 1].
    _STAGE_WEIGHTS = (0.25, 0.35, 0.40)  # soundness, shape, sim

    def quality(self, info):
        terms = self._quality_terms(info)
        if terms is None:
            return 0.0

        # HARD REQUIREMENT (2026-07-24): a track whose GEOMETRY leaves the map is
        # invalid — it is not a racetrack at all.  oob_score is the
        # geometry-in-bounds check; when it is not satisfied, quality collapses
        # (scaled by how far in-bounds it is, so the search still has a small
        # gradient back toward a valid track).
        #
        # Self-intersection (geom_score) is NOT a hard requirement: it is a
        # graded part of soundness below.  Making it hard was a regression this
        # summer — it floored the free-form racing spline at ~0.075 with no
        # gradient (its random content self-intersects ~38 times), so the GA
        # could no longer evolve its way out of crossings the way it used to.
        # Grading it restores that climb; the 2-opt untangle in
        # _normalize_track_points additionally starts racing much closer to a
        # simple loop.
        if terms["oob_score"] < 0.999:
            return 0.15 * terms["oob_score"]

        # Stage 1: soundness — self-intersection (graded) and sensible length.
        soundness = (terms["geom_score"] + terms["length_score"]) / 2.0

        # Stage 2: shape — layout quality, weighted so the discriminating terms
        # (start straight, braking zone) dominate (see _SHAPE_WEIGHTS).
        W = self._SHAPE_WEIGHTS
        w_total = sum(W.values())
        shape = sum(W[name] * terms[name] for name in self._SHAPE_TERMS) / w_total

        # Stage 3: simulation — how well the track actually drives.  on_road
        # penalises time the car spends off the road (on the grass); a random
        # loop that forces the car wide scores low here.
        sim = (terms["completion"] + terms["time_score"]
               + terms["speed_entropy_score"] + terms["on_road_score"]) / 4.0

        # SOFT gating between the soft stages: each smoothly SCALES the next so
        # quality is a continuous gradient (no 1/3 staircase jumps), which gives
        # the GA a climb to follow (Woodruff, "Fitness by Design"; PCG-fitness
        # literature: a good fitness is gradual, not binary).  A valid track
        # starts from the base once the hard requirements are met; the shape and
        # sim stages, which random content largely fails, carry the weight so
        # random tracks stay low and only a genuinely good layout climbs high.
        w_s, w_shape, w_sim = self._STAGE_WEIGHTS
        return (w_s * soundness
                + w_shape * shape * soundness
                + w_sim * sim * shape * soundness)

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
