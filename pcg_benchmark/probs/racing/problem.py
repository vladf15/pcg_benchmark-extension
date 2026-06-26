from .engine import CarPhysicsEngine
from .agent import SteeringAgent
from pcg_benchmark.probs import Problem
from pcg_benchmark.spaces import ArraySpace, FloatSpace, IntegerSpace, DictionarySpace
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont
from pcg_benchmark.probs.racing.utils import (
    interpolate_curves,
    count_self_intersections,
    count_track_area_intersections,
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

    def _get_info_cache_key(self, track_points):
        arr = np.asarray(track_points)
        return tuple(map(tuple, arr.reshape(-1, 2)))

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

        if self._closed_loop and len(track_points) >= 4:
            pts = track_points
            n = len(pts)
            best_i = 0
            best_angle = float('inf')
            # Rotate seam to the smoothest vertex (smallest local turn angle).
            for i in range(n):
                p_prev = pts[(i - 1) % n]
                p = pts[i]
                p_next = pts[(i + 1) % n]
                v1 = p - p_prev
                v2 = p_next - p
                n1 = float(np.linalg.norm(v1))
                n2 = float(np.linalg.norm(v2))
                if n1 < 1e-6 or n2 < 1e-6:
                    continue
                d = float(np.dot(v1, v2) / (n1 * n2))
                d = max(-1.0, min(1.0, d))
                ang = float(np.arccos(d))
                if ang < best_angle:
                    best_angle = ang
                    best_i = i
            if best_i != 0:
                track_points = np.vstack([pts[best_i:], pts[:best_i]])
        return track_points

    def _set_track_cache(self, track_points):
        track_points = self._normalize_track_points(track_points)
        self._track_points_np = track_points
        if len(track_points) == 0:
            self._final_target = None
        elif self._closed_loop:
            self._final_target = track_points[0]
        else:
            self._final_target = track_points[-1]
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
        dots = np.clip(np.einsum('ij,ij->i', v1_unit, v2_unit), -1.0, 1.0)
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
            if len(track_points) < 2:
                start_angle = 0.0
            else:
                start_angle = float(np.arctan2(
                    track_points[1][1] - track_points[0][1],
                    track_points[1][0] - track_points[0][0],
                ))
            self._curve_points = curve_points
            if not hasattr(self, '_engine') or self._engine is None:
                self._engine = CarPhysicsEngine(start_position=curve_points[0], start_angle=start_angle)
            else:
                self._engine.start_position = np.asarray(curve_points[0], dtype=float)
                self._engine.start_angle = start_angle
            if not hasattr(self, '_agent') or self._agent is None:
                self._agent = SteeringAgent(curve_points, track_width=self._track_width, enable_wander=self._enable_wander)
            else:
                self._agent.curve_points = curve_points
            if self._closed_loop and len(curve_points) > 0:
                self._final_target = np.asarray(curve_points[0], dtype=float)
            self._steps_since_reset = 0
            state = self._engine.reset()
            self._car_state = state
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

        while not done and steps < max_steps:
            action = self._agent.act(state)
            state, _, done, _ = self.step(action, track_points=None)
            x, y = float(state[0]), float(state[1])
            if x < x_min or x > x_max or y < y_min or y > y_max:
                break
            steps += 1

        end_xy = state[:2].copy() if state is not None else None
        steps_len = steps + 1
        finished = (
            end_xy is not None
            and len(track_points) > 0
            and steps_len < max_steps
            and self._is_finished(end_xy, steps_len=steps_len)
        )
        summary = (steps_len, finished, end_xy)
        self._simulation_summary_cache[key] = summary
        return summary

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

        self._closed_loop = bool(kwargs.get("closed_loop", True))
        self._lap_finish_min_steps = int(kwargs.get("lap_finish_min_steps", 60))
        self._lap_finish_min_progress_frac = float(kwargs.get("lap_finish_min_progress_frac", 0.25))
        self._steps_since_reset = 0
        self._curve_points = []

        self._enable_wander = kwargs.get("enable_wander", False)

        self._info_cache = {}
        self._trajectory_cache = {}
        self._simulation_summary_cache = {}
        self._track_points_np = None
        self._final_target = None

        if num_points is None:
            num_points = kwargs.get("num_points", 10)
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

        self._car_state = np.array([self._default_track_points[0][0], self._default_track_points[0][1], 0.0, 0.0, 0.0])

        self._content_space = DictionarySpace({
            "track_points": ArraySpace((self.num_points, 2), FloatSpace(0, min(self._width, self._height))),
        })
        self._control_space = DictionarySpace({
            "length":    FloatSpace(500.0, self._width * 16.0),
            "num_turns": IntegerSpace(1, self.num_points * 2),
        })

    def reset(self, track_points=None):
        track_points = self._set_track_cache(track_points)
        self._curve_points = interpolate_curves(
            track_points,
            samples_per_segment=10,
            closed=self._closed_loop,
        )

        if len(self._curve_points) < 2:
            start_angle = 0.0
        else:
            dx = float(self._curve_points[1][0] - self._curve_points[0][0])
            dy = float(self._curve_points[1][1] - self._curve_points[0][1])
            start_angle = float(np.arctan2(dy, dx))

        if self._closed_loop and len(self._curve_points) > 0:
            self._final_target = np.asarray(self._curve_points[0], dtype=float)

        self._steps_since_reset = 0
        if not hasattr(self, '_engine') or self._engine is None:
            self._engine = CarPhysicsEngine(start_position=self._curve_points[0], start_angle=start_angle)
        else:
            self._engine.start_position = np.asarray(self._curve_points[0], dtype=float)
            self._engine.start_angle = start_angle
        if not hasattr(self, '_agent') or self._agent is None:
            self._agent = SteeringAgent(
                self._curve_points,
                track_width=self._track_width,
                enable_wander=self._enable_wander,
            )
        else:
            self._agent.curve_points = self._curve_points
        state = self._engine.reset()
        self._car_state = state
        return state

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
            if not self._closed_loop:
                return True
            return steps_len >= self._lap_finish_min_steps and self._get_progress_index() >= min_progress_idx

        progress_idx = self._get_progress_index()
        if progress_idx >= max(0, nseg - 2):
            near_threshold = max(18.0, 0.9 * self._track_width)
            if dist < near_threshold:
                if not self._closed_loop:
                    return True
                return steps_len >= self._lap_finish_min_steps and progress_idx >= min_progress_idx

        return False

    def step(self, action, track_points=None):
        if track_points is not None:
            self._set_track_cache(track_points)

        state = self._engine.step(action)
        self._car_state = state
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

    def evaluate(self, content=None, max_steps=None, profile=False):
        if max_steps is None:
            max_steps = self._default_max_steps
        track_points = self._extract_content(content)
        track_points = self._set_track_cache(track_points)
        state = self.reset(track_points=track_points)
        self._agent.reset()
        done = False
        steps = 0
        trajectory = [state.copy()]
        t0 = time.time() if profile else None
        step_time = 0.0
        act_time = 0.0
        while not done and steps < max_steps:
            if profile:
                t1 = time.time()
            action = self._agent.act(state)
            if profile:
                act_time += time.time() - t1
                t2 = time.time()
            state, reward, done, info = self.step(action, track_points=None)
            if profile:
                step_time += time.time() - t2
            trajectory.append(state.copy())
            steps += 1
        if profile:
            total = time.time() - t0
            print(f"[PROFILE] evaluate: {total:.3f}s, agent.act: {act_time:.3f}s, step: {step_time:.3f}s, steps: {steps}")
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

        curve_points = interpolate_curves(track_points, samples_per_segment=10, closed=self._closed_loop)
        if trajectory is None:
            steps, finished, end_xy = self._get_cached_simulation_summary(track_points, curve_points=curve_points)
            trajectory_end = end_xy
        else:
            steps = len(trajectory)
            trajectory_end = trajectory[-1][:2] if len(trajectory) > 0 else None
            finished = steps < self._default_max_steps and self._is_finished(trajectory_end, steps_len=steps)

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
            "track_points": track_points,
            "trajectory_end": trajectory_end,
            "curve_points": curve_points,
        }
        if use_cache:
            self._info_cache[cache_key] = info_dict
        return info_dict

    def _compute_spatial_clustering_penalty(self, track_points, min_segment_gap=3, threshold_multiplier=1.1):
        points = np.asarray(track_points, dtype=float)
        n = len(points)

        if n < min_segment_gap + 1:
            return 1.0

        threshold = self._track_width * threshold_multiplier
        num_violations = 0
        total_violation_severity = 0.0

        for i in range(n):
            for j in range(i + min_segment_gap, n):
                spatial_distance = np.linalg.norm(points[i] - points[j])
                if spatial_distance < threshold:
                    num_violations += 1
                    severity = 1.0 - (spatial_distance / threshold)
                    total_violation_severity += severity

        if num_violations == 0:
            return 1.0

        avg_severity = total_violation_severity / num_violations
        return max(0.0, 1.0 - avg_severity * 0.8)

    def _segment_to_segment_distance(self, p1, p2, p3, p4):
        def point_to_segment_distance(p, a, b):
            ab = b - a
            ap = p - a
            ab_sq = np.dot(ab, ab)
            if ab_sq < 1e-12:
                return np.linalg.norm(ap)
            t = np.clip(np.dot(ap, ab) / ab_sq, 0.0, 1.0)
            return np.linalg.norm(p - (a + t * ab))

        return min(
            point_to_segment_distance(p1, p3, p4),
            point_to_segment_distance(p2, p3, p4),
            point_to_segment_distance(p3, p1, p2),
            point_to_segment_distance(p4, p1, p2),
        )

    def _compute_segment_proximity_penalty(self, curve_points, min_distance=None):
        if min_distance is None:
            min_distance = self._track_width * 2.0

        curve_points = np.asarray(curve_points, dtype=float)
        n = len(curve_points)

        if n < 5:
            return 1.0

        min_ratio = 1.0

        for i in range(n - 1):
            seg_i_min = np.min(curve_points[i:i+2], axis=0)
            seg_i_max = np.max(curve_points[i:i+2], axis=0)

            for j in range(i + 3, n - 1):
                seg_j_min = np.min(curve_points[j:j+2], axis=0)
                seg_j_max = np.max(curve_points[j:j+2], axis=0)

                bbox_gap = max(
                    max(seg_i_min[0] - seg_j_max[0], seg_j_min[0] - seg_i_max[0]),
                    max(seg_i_min[1] - seg_j_max[1], seg_j_min[1] - seg_i_max[1])
                )
                if bbox_gap > min_distance:
                    continue

                dist = self._segment_to_segment_distance(
                    curve_points[i], curve_points[i + 1],
                    curve_points[j], curve_points[j + 1]
                )

                if dist < min_distance:
                    ratio = float(dist / (min_distance + 1e-12))
                    if ratio < min_ratio:
                        min_ratio = ratio
                        if min_ratio < 0.15:
                            break

            if min_ratio < 0.15:
                break

        if min_ratio >= 1.0:
            return 1.0

        return float(max(0.0, min_ratio) ** 8)

    # ── Quality ────────────────────────────────────────────────────────
    # All racing variants share the same quality structure, following the
    # benchmark house style (see zelda/sokoban/ddave): an equally-weighted
    # average of track-shape statistics, plus a simulation group that only
    # counts once the geometry is sound (in bounds, no extreme turns, no
    # self-intersection) — analogous to zelda only scoring playability once
    # player/key/door exist.  Subclasses tune behaviour by overriding
    # _QUALITY_PARAMS (thresholds) and _QUALITY_STAT_TERMS (which terms make
    # up the stats group), and add terms via _extra_quality_terms().

    _QUALITY_PARAMS = {
        "min_curvature_deg":    8.0,   # below this avg curvature → linear ramp penalty
        "ideal_curvature_deg": 18.0,   # gaussian target for avg curvature
        "ideal_variety_deg":   10.0,   # gaussian target for curvature std
        "sharp_turn_deg":      50.0,   # angle that counts as a sharp turn
        "sharp_turn_allowance":   2,   # sharp turns tolerated before penalty
        "sharp_turn_window":    3.0,   # extra sharp turns until score hits zero
        "harsh_turn_deg":      85.0,   # single turn above this slashes sharp score (None = off)
        "max_turn_deg":       120.0,   # soft cap on the single largest turn
        "max_turn_soft_deg":   12.0,   # exp falloff width beyond the cap
        "min_length":        1500.0,   # total track length scoring range (metres)
        "max_length":        6500.0,
        "seg_min_pref":       180.0,   # preferred control-segment length range (None = off)
        "seg_max_pref":       520.0,
        "angles_on_curve":     True,   # angles/geometry on dense curve vs control points
        "geom_area_check":     True,   # also count track-area overlap intersections
        "geom_severity":        1.5,
        "max_steps_per_point":  200,
        "straight_thresh_deg":  4.0,   # per-sample angle below this = straight section
        "straight_ideal_frac":  0.20,  # longest-straight arc fraction for score 1.0
        "curve_min_frac":       0.20,  # min curved fraction for non-zero diversity score
        "curve_ideal_frac":     0.50,  # upper bound of ideal curved fraction
        "sim_gate_min_straight": 0.0,  # minimum straight_score to unlock simulation bonus
    }

    # Five terms that measure track quality across all three racing representations.
    # Validity/penalty terms (oob, geom, curvature_penalty, etc.) are handled
    # exclusively by the simulation gate in quality(), not averaged into stats,
    # so the formula is meaningful and comparable across racing, tile, and voronoi.
    _QUALITY_STAT_TERMS = (
        "curvature_score", "variety_score",
        "length_score", "straight_score", "curvature_diversity_score",
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
            curve_points = interpolate_curves(points, samples_per_segment=10, closed=self._closed_loop)
        curve_points = np.asarray(curve_points, dtype=float)

        # Out-of-bounds (multiplicative)
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
        oob_penalty = float(np.exp(-oob_violation / max(1e-6, float(self._track_width) * 0.5)))

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
        elif self._closed_loop and len(curve_points) > 1:
            nearest = int(np.argmin(np.linalg.norm(curve_points - trajectory_end, axis=1)))
            completion = nearest / len(curve_points)
        else:
            dist_to_goal = float(np.linalg.norm(points[-1] - trajectory_end))
            curve_total_length = float(np.sum(np.linalg.norm(curve_points[1:] - curve_points[:-1], axis=1))) if len(curve_points) > 1 else 1.0
            completion = 1.0 - min(dist_to_goal / (curve_total_length + 1e-6), 1.0)

        # Curvature terms — variants measure on dense curve or raw vertices
        angle_pts = curve_points if P["angles_on_curve"] else points
        turn_angles = self._compute_turn_angles(angle_pts)
        avg_curvature = float(np.mean(turn_angles)) if turn_angles.size > 0 else 0.0
        curvature_std = float(np.std(turn_angles)) if turn_angles.size > 0 else 0.0

        min_curvature = np.deg2rad(P["min_curvature_deg"])
        curvature_penalty = (avg_curvature / min_curvature) if avg_curvature < min_curvature else 1.0

        ideal_curvature = np.deg2rad(P["ideal_curvature_deg"])
        curvature_score = float(np.exp(-((avg_curvature - ideal_curvature) ** 2) / (2 * (ideal_curvature / 2) ** 2))) if avg_curvature > 0 else 0.0

        ideal_variety = np.deg2rad(P["ideal_variety_deg"])
        variety_score = float(np.exp(-((curvature_std - ideal_variety) ** 2) / (2 * (ideal_variety / 2) ** 2))) if avg_curvature > 0 else 0.0

        # Sharp and extreme turns
        sharp_turns = int(np.sum(turn_angles > np.deg2rad(P["sharp_turn_deg"])))
        allowance = P["sharp_turn_allowance"]
        sharp_turn_penalty = 1.0 - min((sharp_turns - allowance) / P["sharp_turn_window"], 1.0) if sharp_turns > allowance else 1.0
        max_turn = float(np.max(turn_angles)) if turn_angles.size > 0 else 0.0
        if P["harsh_turn_deg"] is not None and max_turn > np.deg2rad(P["harsh_turn_deg"]):
            sharp_turn_penalty *= 0.25
        excess_turn = max(0.0, max_turn - np.deg2rad(P["max_turn_deg"]))
        turn_penalty = float(np.exp(-excess_turn / max(1e-6, np.deg2rad(P["max_turn_soft_deg"]))))

        # Control-segment statistics
        if len(points) > 1:
            seg_lens = np.linalg.norm(points[1:] - points[:-1], axis=1)
            mean_length = float(np.mean(seg_lens))
            std_length = float(np.std(seg_lens))
            variance_penalty = float(np.exp(-std_length / (mean_length + 1e-6)))

            total_length = float(np.sum(seg_lens))
            if total_length < P["min_length"]:
                length_score = total_length / P["min_length"]
            elif total_length > P["max_length"]:
                length_score = P["max_length"] / total_length
            else:
                length_score = 1.0

            if P["seg_min_pref"] is not None:
                out_of_range = np.sum((seg_lens < P["seg_min_pref"]) | (seg_lens > P["seg_max_pref"]))
                spacing_penalty = 1.0 - min(out_of_range / max(1, len(seg_lens)), 1.0)
            else:
                spacing_penalty = 1.0
        else:
            variance_penalty = 1.0
            spacing_penalty = 1.0
            length_score = 0.0

        # Self-intersection (multiplicative)
        geom_pts = curve_points if P["angles_on_curve"] else points
        geom_violations = int(count_self_intersections(geom_pts, closed=self._closed_loop))
        if P["geom_area_check"]:
            geom_violations += int(count_track_area_intersections(
                curve_points,
                track_width=float(self._track_width),
                min_cross_index_gap=2,
                closed=self._closed_loop,
            ))
        geom_penalty = float(np.exp(-P["geom_severity"] * float(min(geom_violations, 50))))

        # Straight section and curvature diversity (always on dense curve, all variants)
        cp_d     = curve_points[1:] - curve_points[:-1]
        cp_len   = np.linalg.norm(cp_d, axis=1)
        cp_total = float(np.sum(cp_len))
        if cp_total > 1e-6 and len(curve_points) >= 3:
            cv1 = cp_d[:-1]; cv2 = cp_d[1:]
            cn1 = np.linalg.norm(cv1, axis=1, keepdims=True)
            cn2 = np.linalg.norm(cv2, axis=1, keepdims=True)
            ok  = (cn1[:, 0] > 1e-3) & (cn2[:, 0] > 1e-3)
            dt  = np.ones(len(cv1))
            if ok.any():
                dt[ok] = np.clip(
                    np.einsum('ij,ij->i', cv1[ok] / cn1[ok], cv2[ok] / cn2[ok]),
                    -1.0, 1.0,
                )
            cp_ang = np.arccos(dt)   # turn angle at each interior curve point
            n_cs   = len(cp_len)
            seg_a  = np.zeros(n_cs)
            seg_a[:-1] = np.maximum(seg_a[:-1], cp_ang)  # end-angle of each segment
            seg_a[1:]  = np.maximum(seg_a[1:],  cp_ang)  # start-angle of each segment
            is_str   = seg_a < np.deg2rad(P["straight_thresh_deg"])
            changes  = np.diff(is_str.astype(int), prepend=0, append=0)
            r_starts = np.where(changes == 1)[0]
            r_ends   = np.where(changes == -1)[0]
            max_s    = max((float(np.sum(cp_len[s:e])) for s, e in zip(r_starts, r_ends)), default=0.0)
            straight_score = min(1.0, max_s / (cp_total * max(1e-9, P["straight_ideal_frac"])))
            curved_frac    = float(np.sum(cp_len[~is_str])) / cp_total
            curvature_diversity_score = get_range_reward(
                curved_frac, 0.0, P["curve_min_frac"], P["curve_ideal_frac"], 1.0
            )
        else:
            straight_score = 0.0
            curvature_diversity_score = 0.0

        clustering_penalty = self._compute_spatial_clustering_penalty(points, min_segment_gap=3, threshold_multiplier=1.1)
        proximity_penalty = self._compute_segment_proximity_penalty(geom_pts, min_distance=self._track_width * 2.0)

        # Lap time
        steps = info.get('steps', 0)
        n_segments = max(len(points) - 1, 1)
        min_steps = 30 * n_segments
        max_steps_for_scoring = P["max_steps_per_point"] * n_segments
        if not finished:
            time_score = 0.0
        elif steps < min_steps:
            time_score = max(0.0, 1.0 - (min_steps - steps) / min_steps)
        elif steps > max_steps_for_scoring:
            time_score = max(0.0, 1.0 - (steps - max_steps_for_scoring) / max_steps_for_scoring)
        else:
            time_score = 1.0

        return {
            "completion":         completion,
            "curvature_score":    curvature_score,
            "curvature_penalty":  curvature_penalty,
            "variety_score":      variety_score,
            "sharp_turn_penalty": sharp_turn_penalty,
            "variance_penalty":   variance_penalty,
            "spacing_penalty":    spacing_penalty,
            "length_score":       length_score,
            "clustering_penalty":        clustering_penalty,
            "proximity_penalty":         proximity_penalty,
            "oob_penalty":               oob_penalty,
            "turn_penalty":              turn_penalty,
            "geom_penalty":              geom_penalty,
            "straight_score":            straight_score,
            "curvature_diversity_score": curvature_diversity_score,
            "time_score":                time_score,
        }

    def _extra_quality_terms(self, info):
        """Hook for subclasses to add problem-specific quality terms."""
        return {}

    def quality(self, info):
        terms = self._quality_terms(info)
        if terms is None:
            return 0.0
        terms.update(self._extra_quality_terms(info))

        # Track-shape statistics: five quality terms, equally weighted
        stats = sum(terms[k] for k in self._QUALITY_STAT_TERMS) / len(self._QUALITY_STAT_TERMS)

        # Simulation bonus only counts once the track is geometrically sound
        # and meets the minimum straight-section requirement (parameterised so
        # tile can set a stricter threshold while racing/voronoi use 0.0).
        added = 0.0
        P = self._QUALITY_PARAMS
        if (terms["oob_penalty"] >= 1.0
                and terms["geom_penalty"] >= 1.0
                and terms["straight_score"] >= P["sim_gate_min_straight"]):
            added = (terms["completion"] + terms["time_score"]) / 2.0

        return (stats + added) / 2.0

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

    def _render_track_bg(self, img_w, img_h, left_edge_f, right_edge_f, scaled_curve,
                         grass_color, edge_color, road_color, centerline_color):
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
        if len(scaled_curve) > 1:
            bg_draw.line(scaled_curve, fill=centerline_color, width=2)
        return bg

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
        if skip is None:
            skip = self._skip_render
        if skip:
            return []

        track_points = self._extract_content(content)
        trajectory = self._get_cached_trajectory(track_points)

        try:
            frame_sampling = int(frame_sampling)
        except Exception:
            frame_sampling = 5
        if frame_sampling < 1:
            frame_sampling = 1

        track_points_np = self._normalize_track_points(track_points)
        curve_np = np.asarray(
            interpolate_curves(track_points_np, samples_per_segment=10, closed=self._closed_loop),
            dtype=float,
        )

        try:
            render_scale = float(render_scale)
        except Exception:
            render_scale = 1.0
        if render_scale <= 1e-9:
            render_scale = 1.0

        scale = float(PX_PER_M) * float(render_scale)
        img_w = int(round(float(self._width) * scale))
        img_h = int(round(float(self._height) * scale))
        curve_px = curve_np * scale

        grass_color = (34, 139, 34)
        edge_color = (10, 10, 10)
        road_color = (215, 215, 215)
        centerline_color = (120, 120, 120)

        frames = []
        car_length = 5.0 * scale
        car_width = 2.0 * scale

        scaled_curve = [(int(round(x)), int(round(y))) for (x, y) in curve_px]
        half_width = float(self._track_width) * scale * 0.5

        left_edge_f = []
        right_edge_f = []
        n = len(curve_px)
        if n >= 2:
            has_dup_close = self._closed_loop and n >= 3 and np.allclose(curve_px[0], curve_px[-1], atol=1e-9, rtol=0.0)
            base = curve_px[:-1] if has_dup_close else curve_px
            m = len(base)
            for j in range(m):
                if self._closed_loop and m >= 3:
                    prev_i = (j - 1) % m
                    next_i = (j + 1) % m
                    dir_prev = base[j] - base[prev_i]
                    dir_next = base[next_i] - base[j]
                else:
                    dir_prev = base[1] - base[0] if j == 0 else base[j] - base[j - 1]
                    dir_next = base[j] - base[j - 1] if j == m - 1 else base[j + 1] - base[j]

                avg_dir = dir_prev + dir_next
                norm = float(np.linalg.norm(avg_dir))
                perp = np.array([-avg_dir[1], avg_dir[0]], dtype=float) / norm if norm > 0.0 else np.zeros(2)

                left = base[j] + perp * half_width
                right = base[j] - perp * half_width
                left_edge_f.append((float(left[0]), float(left[1])))
                right_edge_f.append((float(right[0]), float(right[1])))

            if has_dup_close and left_edge_f:
                left_edge_f.append(left_edge_f[0])
                right_edge_f.append(right_edge_f[0])

        track_background = self._render_track_bg(
            img_w, img_h, left_edge_f, right_edge_f, scaled_curve,
            grass_color, edge_color, road_color, centerline_color,
        )

        agent = None
        font = None
        dt = 0.1
        mass = 1350.0
        font_size = 24
        if show_hud:
            agent = SteeringAgent(
                curve_np,
                track_width=float(self._track_width),
                enable_wander=bool(self._enable_wander),
            )
            agent.reset()

            try:
                dt = float(getattr(getattr(self, '_engine', None), 'time_step', 0.1))
            except Exception:
                dt = 0.1
            if dt <= 1e-9:
                dt = 0.1

            try:
                mass = float(getattr(getattr(self, '_engine', None), 'mass', 1350.0))
            except Exception:
                mass = 1350.0

            base_font_size = 24
            font_size = max(10, int(round(float(base_font_size) * float(render_scale))))
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", font_size)
            except Exception:
                try:
                    font = ImageFont.truetype(r"C:\\Windows\\Fonts\\segoeui.ttf", font_size)
                except Exception:
                    try:
                        font = ImageFont.truetype("arial.ttf", font_size)
                    except Exception:
                        font = ImageFont.load_default()

        reuse_canvas = bool(fast)
        img = None
        prev_bbox = None
        if reuse_canvas:
            img = track_background.copy()
        dirty_pad_px = max(2, int(round(12.0 * float(render_scale))))

        iterator = enumerate(trajectory)
        if progress:
            try:
                from tqdm import tqdm  # type: ignore
                desc = progress_desc if progress_desc is not None else 'Rendering racing frames'
                iterator = enumerate(
                    tqdm(
                        trajectory,
                        total=len(trajectory),
                        desc=str(desc),
                        leave=False,
                        dynamic_ncols=True,
                    )
                )
            except Exception:
                iterator = enumerate(trajectory)

        prev_angle = None
        for i, state in iterator:
            angle = state[2] if len(state) > 2 else 0.0

            action = None
            lookahead = None
            yaw_rate = 0.0
            if show_hud and agent is not None:
                try:
                    action = agent.act(state)
                except Exception:
                    action = {'steering': 0.0, 'throttle': 0.0}
                lookahead = getattr(agent, 'last_lookahead_point', None)

                if prev_angle is None:
                    yaw_rate = 0.0
                else:
                    da = (float(angle) - float(prev_angle) + np.pi) % (2.0 * np.pi) - np.pi
                    yaw_rate = float(da / dt)
                prev_angle = float(angle)

            if i % frame_sampling != 0:
                continue

            if reuse_canvas:
                if prev_bbox is not None:
                    try:
                        img.paste(track_background.crop(prev_bbox), prev_bbox)
                    except Exception:
                        pass
            else:
                img = track_background.copy()

            draw = ImageDraw.Draw(img)

            car_x = float(state[0]) * scale
            car_y = float(state[1]) * scale
            angle = state[2] if len(state) > 2 else 0.0
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            forward = (float(cos_a), float(sin_a))
            right = (float(-sin_a), float(cos_a))

            dx = car_length / 2.0
            dy = car_width / 2.0

            wheel_len = car_length * 0.20
            wheel_wid = car_width * 0.22
            half_wheel_len = wheel_len / 2.0
            half_wheel_wid = wheel_wid / 2.0
            wheel_base = car_length * 0.34
            wheel_track = car_width * 0.44
            wheel_centers = [
                (car_x + forward[0] * wheel_base + right[0] * wheel_track, car_y + forward[1] * wheel_base + right[1] * wheel_track),
                (car_x + forward[0] * wheel_base - right[0] * wheel_track, car_y + forward[1] * wheel_base - right[1] * wheel_track),
                (car_x - forward[0] * wheel_base + right[0] * wheel_track, car_y - forward[1] * wheel_base + right[1] * wheel_track),
                (car_x - forward[0] * wheel_base - right[0] * wheel_track, car_y - forward[1] * wheel_base - right[1] * wheel_track),
            ]
            wheel_pts = []
            for wx, wy in wheel_centers:
                wheel = _rotated_rect(wx, wy, forward, right, half_wheel_len, half_wheel_wid)
                draw.polygon(wheel, fill=(25, 25, 25), outline=(0, 0, 0))
                wheel_pts.extend(wheel)

            corners = [
                (car_x + cos_a * dx - sin_a * dy, car_y + sin_a * dx + cos_a * dy),
                (car_x + cos_a * dx + sin_a * dy, car_y + sin_a * dx - cos_a * dy),
                (car_x - cos_a * dx + sin_a * dy, car_y - sin_a * dx - cos_a * dy),
                (car_x - cos_a * dx - sin_a * dy, car_y - sin_a * dx + cos_a * dy),
            ]
            draw.polygon(corners, fill=(255, 0, 0), outline=(0, 0, 0))
            front_x = car_x + cos_a * dx
            front_y = car_y + sin_a * dx
            car_line_w = max(1, int(round(3.0 * float(render_scale))))
            draw.line([(car_x, car_y), (front_x, front_y)], fill=(0, 0, 255), width=car_line_w)

            bbox_xs = [p[0] for p in corners] + [car_x, front_x]
            bbox_ys = [p[1] for p in corners] + [car_y, front_y]
            if wheel_pts:
                bbox_xs.extend([p[0] for p in wheel_pts])
                bbox_ys.extend([p[1] for p in wheel_pts])

            if lookahead is not None:
                try:
                    la_x = float(lookahead[0]) * scale
                    la_y = float(lookahead[1]) * scale
                    rr = max(2, int(round(6.0 * float(render_scale))))
                    la_w = max(1, int(round(3.0 * float(render_scale))))
                    la_line_w = max(1, int(round(2.0 * float(render_scale))))
                    draw.ellipse([(la_x - rr, la_y - rr), (la_x + rr, la_y + rr)], outline=(0, 255, 255), width=la_w)
                    draw.line([(car_x, car_y), (la_x, la_y)], fill=(0, 200, 200), width=la_line_w)
                    bbox_xs.extend([la_x - rr, la_x + rr, car_x, la_x])
                    bbox_ys.extend([la_y - rr, la_y + rr, car_y, la_y])
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

                pad = max(2, int(round(6.0 * float(render_scale))))
                x0, y0 = pad, pad
                line_h = max(1, int(round(float(font_size) * 1.25)))
                max_w = 0
                for txt in lines:
                    w = 0
                    try:
                        bbox = draw.textbbox((0, 0), txt, font=font)
                        w = int(bbox[2] - bbox[0])
                    except Exception:
                        try:
                            w, _h = draw.textsize(txt, font=font)
                            w = int(w)
                        except Exception:
                            w = int(len(txt) * float(font_size) * 0.6)
                    if w > max_w:
                        max_w = w
                box_w = max_w + pad * 2
                box_h = pad * 2 + line_h * len(lines)

                hud_outline_w = max(1, int(round(1.0 * float(render_scale))))
                draw.rectangle([(x0, y0), (x0 + box_w, y0 + box_h)], fill=(255, 255, 255), outline=(0, 0, 0), width=hud_outline_w)
                ty = y0 + pad
                for txt in lines:
                    draw.text((x0 + pad, ty), txt, fill=(0, 0, 0), font=font)
                    ty += line_h

                bbox_xs.extend([x0, x0 + box_w])
                bbox_ys.extend([y0, y0 + box_h])

            if reuse_canvas:
                try:
                    x0 = int(max(0, min(bbox_xs) - dirty_pad_px))
                    y0 = int(max(0, min(bbox_ys) - dirty_pad_px))
                    x1 = int(min(img_w, max(bbox_xs) + dirty_pad_px))
                    y1 = int(min(img_h, max(bbox_ys) + dirty_pad_px))
                    prev_bbox = (x0, y0, x1, y1)
                except Exception:
                    prev_bbox = None

            frames.append(img.copy() if reuse_canvas else img)

        return frames
