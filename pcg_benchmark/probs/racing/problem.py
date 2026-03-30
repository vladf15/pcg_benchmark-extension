from .engine import CarPhysicsEngine
from .agent import SimpleAgent, SteeringAgent
from pcg_benchmark.probs import Problem
from pcg_benchmark.spaces import ArraySpace, IntegerSpace, FloatSpace, DictionarySpace
import numpy as np
import json
import time
from PIL import Image, ImageDraw
from pcg_benchmark.probs.racing.utils import interpolate_curves, count_self_intersections
import os
import itertools
from functools import wraps

class RacingProblem(Problem):
    """Benchmark problem for generating and evaluating 2D racetrack control points."""

    def _profile(func):
        """Record wall-clock durations for a wrapped method."""
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            start = time.time()
            result = func(self, *args, **kwargs)
            elapsed = time.time() - start
            if not hasattr(self, '_profile_data'):
                self._profile_data = {}
            if func.__name__ not in self._profile_data:
                self._profile_data[func.__name__] = []
            self._profile_data[func.__name__].append(elapsed)
            return result
        return wrapper

    def _get_info_cache_key(self, track_points, tension=None, bias=None):
        """Create a hashable cache key from points and spline parameters."""
        arr = np.asarray(track_points)
        points_key = tuple(map(tuple, arr.reshape(-1, arr.shape[-1])))
        tension_key = None
        bias_key = None
        if tension is not None:
            tension_arr = np.asarray(tension, dtype=float).reshape(-1)
            tension_key = tuple(map(float, tension_arr.tolist()))
        if bias is not None:
            bias_arr = np.asarray(bias, dtype=float).reshape(-1)
            bias_key = tuple(map(float, bias_arr.tolist()))
        return (points_key, tension_key, bias_key)

    def _extract_content(self, content):
        """Parse content into (track_points, tension, bias)."""
        if content is None:
            return self._default_track_points, None, None
        if isinstance(content, dict):
            return (
                content.get("track_points", self._default_track_points),
                content.get("tension", None),
                content.get("bias", None),
            )
        return content, None, None

    def _normalize_track_points(self, track_points):
        """Return an (N,2) float ndarray for track points."""
        if track_points is None:
            track_points = self._default_track_points
        elif isinstance(track_points, dict):
            track_points = track_points.get("track_points", self._default_track_points)
        track_points = np.asarray(track_points, dtype=float)
        if track_points.ndim == 1:
            track_points = track_points.reshape(-1, 2)
        return track_points

    def _set_track_cache(self, track_points):
        """Cache normalized track points and final target."""
        track_points = self._normalize_track_points(track_points)
        self._track_points_np = track_points
        self._final_target = track_points[-1] if len(track_points) > 0 else None
        return track_points

    def _get_cached_simulation_summary(self, track_points, tension=None, bias=None, max_steps=None):
        """Simulate a policy and cache only summary stats."""
        if max_steps is None:
            max_steps = self._default_max_steps
        track_points = self._normalize_track_points(track_points)
        key = (self._get_info_cache_key(track_points, tension=tension, bias=bias), max_steps)
        cached = self._simulation_summary_cache.get(key)
        if cached is not None:
            return cached

        # Reuse precomputed curve points when available (info() already builds them).
        state = self.reset(track_points=track_points, tension=tension, bias=bias)
        self._agent.reset()
        done = False
        steps = 0
        # Cached simulation is used for feature computation (not for agent training).
        # Add a conservative early-out if the car goes clearly out-of-bounds to
        # avoid spending max_steps on hopeless trajectories.
        margin = float(self._track_width) * 0.5 + 2.0
        x_min = margin
        x_max = float(self._width - 1) - margin
        y_min = margin
        y_max = float(self._height - 1) - margin

        while not done and steps < max_steps:
            action = self._agent.act(state)
            state, _, done, _ = self.step(action, track_points=None)
            x, y = float(state[0]), float(state[1])
            if x < x_min or x > x_max or y < y_min or y > y_max:
                break
            steps += 1

        end_xy = state[:2].copy() if state is not None else None
        steps_len = steps + 1
        finished = False
        if end_xy is not None and len(track_points) > 0:
            finished = (steps_len < max_steps) and (np.linalg.norm(track_points[-1] - end_xy) < 10.0)

        summary = (steps_len, finished, end_xy)
        self._simulation_summary_cache[key] = summary
        return summary

    def _get_cached_simulation_summary_with_curve(self, track_points, curve_points, max_steps=None):
        """Simulate a policy using already-interpolated curve_points and cache summary stats."""
        if max_steps is None:
            max_steps = self._default_max_steps
        track_points = self._normalize_track_points(track_points)
        curve_points = np.asarray(curve_points, dtype=float)
        key = (self._get_info_cache_key(track_points, tension=None, bias=None), max_steps, ('curve', curve_points.shape[0]))
        cached = self._simulation_summary_cache.get(key)
        if cached is not None:
            return cached

        # Manual reset without recomputing interpolation.
        if len(track_points) < 2:
            start_angle = 0.0
        else:
            x0, y0 = track_points[0]
            x1, y1 = track_points[1]
            start_angle = float(np.arctan2(y1 - y0, x1 - x0))

        self._curve_points = curve_points
        if not hasattr(self, '_engine') or self._engine is None:
            self._engine = CarPhysicsEngine(start_position=self._curve_points[0], start_angle=start_angle)
        else:
            self._engine.start_position = np.asarray(self._curve_points[0], dtype=float)
            self._engine.start_angle = start_angle
        if not hasattr(self, '_agent') or self._agent is None:
            if self._agent_type == "steering":
                self._agent = SteeringAgent(self._curve_points, track_width=self._track_width,
                                           enable_wander=self._enable_wander)
            else:
                self._agent = SimpleAgent(track_points)
        else:
            self._agent.curve_points = self._curve_points

        state = self._engine.reset()
        self._car_state = state
        self._agent.reset()

        done = False
        steps = 0
        margin = float(self._track_width) * 0.5 + 2.0
        x_min = margin
        x_max = float(self._width - 1) - margin
        y_min = margin
        y_max = float(self._height - 1) - margin

        while not done and steps < max_steps:
            action = self._agent.act(state)
            state, _, done, _ = self.step(action, track_points=None)
            x, y = float(state[0]), float(state[1])
            if x < x_min or x > x_max or y < y_min or y > y_max:
                break
            steps += 1

        end_xy = state[:2].copy() if state is not None else None
        steps_len = steps + 1
        finished = False
        if end_xy is not None and len(track_points) > 0:
            finished = (steps_len < max_steps) and (np.linalg.norm(track_points[-1] - end_xy) < 10.0)

        summary = (steps_len, finished, end_xy)
        self._simulation_summary_cache[key] = summary
        return summary

    def _get_cached_trajectory(self, track_points, tension=None, bias=None, max_steps=None):
        """Roll out a policy and cache the full trajectory."""
        if max_steps is None:
            max_steps = self._default_max_steps
        track_points = self._normalize_track_points(track_points)
        key = (self._get_info_cache_key(track_points, tension=tension, bias=bias), max_steps)
        if key in self._trajectory_cache:
            return self._trajectory_cache[key]
        traj = self.evaluate({"track_points": track_points, "tension": tension, "bias": bias}, max_steps=max_steps)
        self._trajectory_cache[key] = traj
        return traj

    def get_profile_data(self):
        """Return profiling data collected by `_profile`."""
        return getattr(self, '_profile_data', {})

    def clear_caches(self):
        """Clear cached info and trajectories."""
        self._info_cache = {}
        self._trajectory_cache = {}
        self._simulation_summary_cache = {}

    def __init__(self, num_points=None, **kwargs):
        Problem.__init__(self, **kwargs)
        self._track_width = kwargs.get("track_width", 50)
        self._width = kwargs.get("width", 800)
        self._height = kwargs.get("height", 600)
        self._default_max_steps = kwargs.get("max_steps", 4000)
        self._enable_profiling = kwargs.get("enable_profiling", False)
        self._skip_render = kwargs.get("skip_render", False)
        
        # Agent configuration
        self._agent_type = kwargs.get("agent_type", "steering")  # "simple" or "steering"
        self._enable_wander = kwargs.get("enable_wander", False)
        
        self._info_cache = {}
        self._trajectory_cache = {}
        self._simulation_summary_cache = {}
        self._track_points_np = None
        self._final_target = None

        if num_points is None:
            num_points = kwargs.get("num_points", 8)
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
            "track_points": ArraySpace((self.num_points, 2), FloatSpace(0, min(self._width, self._height) - 20)),
            "tension": ArraySpace((self.num_points-1,), FloatSpace(-1, 1)),
            "bias": ArraySpace((self.num_points-1,), FloatSpace(-1, 1)),
        })
        self._control_space = DictionarySpace({
            "steering": FloatSpace(-1, 1),
            "throttle": FloatSpace(-1, 1),
        })
        
    def reset(self, track_points=None, tension=None, bias=None):
        """Initialize engine/agent state for a new instance."""
        track_points = self._set_track_cache(track_points)
        if len(track_points) < 2:
            start_direction = (1.0, 0.0)
        else:
            x0, y0 = track_points[0]
            x1, y1 = track_points[1]
            start_direction = (x1 - x0, y1 - y0)
        start_angle = np.arctan2(start_direction[1], start_direction[0])
        self._curve_points = interpolate_curves(track_points, samples_per_segment=10, tension=tension, bias=bias)
        if not hasattr(self, '_engine') or self._engine is None:
            self._engine = CarPhysicsEngine(start_position=self._curve_points[0], start_angle=start_angle)
        else:
            self._engine.start_position = np.asarray(self._curve_points[0], dtype=float)
            self._engine.start_angle = start_angle
        if not hasattr(self, '_agent') or self._agent is None:
            # Create agent based on configured type
            if self._agent_type == "steering":
                self._agent = SteeringAgent(self._curve_points, track_width=self._track_width,
                                           enable_wander=self._enable_wander)
            else:
                self._agent = SimpleAgent(track_points)
        else:
            self._agent.curve_points = self._curve_points
        state = self._engine.reset()
        self._car_state = state
        return state

    def step(self, action, track_points=None):
        """Advance the car state by one step and return (state, reward, done, info)."""
        if track_points is not None:
            self._set_track_cache(track_points)

        state = self._engine.step(action)
        self._car_state = state
        x, y = state[0], state[1]
        final_target = self._final_target
        if final_target is None:
            dist_to_final = float('inf')
        else:
            dist_to_final = np.hypot(final_target[0] - x, final_target[1] - y)
        final_threshold = 10.0
        done = (final_target is not None) and (dist_to_final < final_threshold)
        reward = -dist_to_final
        info = {"waypoint": getattr(self._agent, "current_curve_idx", None)}
        return state, reward, done, info

    def evaluate(self, content=None, max_steps=None, profile=False):
        """Simulate agent trajectory through the racetrack and return a list of states."""
        if max_steps is None:
            max_steps = self._default_max_steps
        track_points, tension, bias = self._extract_content(content)
        track_points = self._set_track_cache(track_points)
        state = self.reset(track_points=track_points, tension=tension, bias=bias)
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
        """Return features for an instance of a track and the agent simulation metrics."""
        track_points, tension, bias = self._extract_content(content)
        track_points = self._normalize_track_points(track_points)
        
        # Handle degenerate tracks with < 2 points
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
                'steps': 0,
                'finished': False,
                'track_points': track_points,
                'trajectory_end': track_points[0] if len(track_points) > 0 else None,
                'curve_points': track_points,
                'tension': None if tension is None else np.asarray(tension, dtype=float),
                'bias': None if bias is None else np.asarray(bias, dtype=float),
            }
        
        cache_key = self._get_info_cache_key(track_points, tension=tension, bias=bias)
        if use_cache and cache_key in self._info_cache and trajectory is None:
            return self._info_cache[cache_key]
        
        num_points = self.num_points
        diffs = track_points[1:] - track_points[:-1]
        segment_lengths = np.linalg.norm(diffs, axis=1)
        total_length = np.sum(segment_lengths)
        avg_length = np.mean(segment_lengths)
        max_length = np.max(segment_lengths)
        min_length = np.min(segment_lengths)

        v1 = diffs[:-1]
        v2 = diffs[1:]
        v1_norm = np.linalg.norm(v1, axis=1, keepdims=True)
        v2_norm = np.linalg.norm(v2, axis=1, keepdims=True)
        valid = (v1_norm[:, 0] > 1e-3) & (v2_norm[:, 0] > 1e-3)
        v1_unit = np.zeros_like(v1)
        v2_unit = np.zeros_like(v2)
        v1_unit[valid] = v1[valid] / v1_norm[valid]
        v2_unit[valid] = v2[valid] / v2_norm[valid]
        dots = np.einsum('ij,ij->i', v1_unit, v2_unit)
        dots = np.clip(dots, -1.0, 1.0)
        turn_angles = np.arccos(dots)
        turn_angles = turn_angles[valid]
        avg_turn = np.mean(turn_angles) if turn_angles.size > 0 else 0.0
        max_turn = np.max(turn_angles) if turn_angles.size > 0 else 0.0
        min_turn = np.min(turn_angles) if turn_angles.size > 0 else 0.0
        if trajectory is None:
            # Build curve once, then reuse it for both simulation and returned info.
            curve_points = interpolate_curves(track_points, samples_per_segment=10, tension=tension, bias=bias)
            steps, finished, end_xy = self._get_cached_simulation_summary_with_curve(track_points, curve_points)
            trajectory_end = end_xy
        else:
            steps = len(trajectory)
            trajectory_end = trajectory[-1][:2] if len(trajectory) > 0 else None
            finished = steps < self._default_max_steps and trajectory_end is not None and np.linalg.norm(track_points[-1] - trajectory_end) < 10.0

        if trajectory is not None:
            curve_points = interpolate_curves(track_points, samples_per_segment=10, tension=tension, bias=bias)

        info_dict = {
            "num_points": num_points,
            "total_length": total_length,
            "avg_length": avg_length,
            "max_length": max_length,
            "min_length": min_length,
            "avg_turn": avg_turn,
            "max_turn": max_turn,
            "min_turn": min_turn,
            "steps": steps,
            "finished": finished,
            "track_points": track_points,
            "trajectory_end": trajectory_end,
            "curve_points": curve_points,
            "tension": None if tension is None else np.asarray(tension, dtype=float),
            "bias": None if bias is None else np.asarray(bias, dtype=float),
        }
        if use_cache:
            self._info_cache[cache_key] = info_dict
        return info_dict


    def _compute_spatial_clustering_penalty(self, track_points, min_segment_gap=3, threshold_multiplier=1.1):
        """
        Penalize control points that are far apart in sequence but close in space.
        This prevents self-intersecting tracks by discouraging loops and crossovers.
        
        Args:
            track_points: (N, 2) array of control points
            min_segment_gap: minimum segment separation to check (e.g., 3 means points[i] vs points[i+3:])
            threshold_multiplier: multiplier for track_width to determine distance threshold
        
        Returns:
            penalty value in [0, 1], where 1.0 = no penalty, 0.0 = severe violations
        """
        points = np.asarray(track_points, dtype=float)
        n = len(points)
        
        if n < min_segment_gap + 1:
            return 1.0  # Not enough points to form violations
        
        threshold = self._track_width * threshold_multiplier
        num_violations = 0
        total_violation_severity = 0.0
        
        for i in range(n):
            for j in range(i + min_segment_gap, n):
                spatial_distance = np.linalg.norm(points[i] - points[j])
                if spatial_distance < threshold:
                    num_violations += 1
                    # Severity: how much it violates the threshold (0 to 1)
                    severity = 1.0 - (spatial_distance / threshold)
                    total_violation_severity += severity
        
        if num_violations == 0:
            return 1.0
        
        # Penalize based on average violation severity
        avg_severity = total_violation_severity / num_violations
        penalty = max(0.0, 1.0 - avg_severity * 0.8)
        return penalty

    def _segment_to_segment_distance(self, p1, p2, p3, p4):
        """
        Compute minimum distance between two line segments.
        Segment 1: p1->p2, Segment 2: p3->p4
        """
        def point_to_segment_distance(p, a, b):
            """Distance from point p to line segment a->b"""
            ab = b - a
            ap = p - a
            ab_sq = np.dot(ab, ab)
            if ab_sq < 1e-12:
                return np.linalg.norm(ap)
            t = np.dot(ap, ab) / ab_sq
            t = np.clip(t, 0.0, 1.0)
            closest = a + t * ab
            return np.linalg.norm(p - closest)
        
        # Min distance is the minimum of all point-to-segment distances
        d1 = min(point_to_segment_distance(p1, p3, p4), 
                 point_to_segment_distance(p2, p3, p4))
        d2 = min(point_to_segment_distance(p3, p1, p2), 
                 point_to_segment_distance(p4, p1, p2))
        return min(d1, d2)

    def _compute_segment_proximity_penalty(self, curve_points, min_distance=None):
        """
        Penalize when non-adjacent curve segments come too close.
        Prevents later track sections from overlapping earlier segments.
        
        Args:
            curve_points: (N, 2) array of interpolated curve points
            min_distance: minimum required distance between non-adjacent segments
                         Defaults to 1.5 * track_width
        
        Returns:
            penalty value in [0, 1], where 1.0 = no penalty, 0.0 = severe violations
        """
        if min_distance is None:
            min_distance = self._track_width * 1.5
        
        curve_points = np.asarray(curve_points, dtype=float)
        n = len(curve_points)
        
        if n < 5:
            return 1.0  # Not enough points to form multiple segments
        
        violations = 0
        total_severity = 0.0
        max_violations_to_check = 50  # Early termination after finding many violations
        
        # Check all non-adjacent segment pairs
        for i in range(n - 1):
            # Quick spatial bounds check: skip if segments are obviously far apart
            seg_i_min = np.min(curve_points[i:i+2], axis=0)
            seg_i_max = np.max(curve_points[i:i+2], axis=0)
            
            # Start checking j segments that are at least 3 steps away
            for j in range(i + 3, n - 1):
                seg_j_min = np.min(curve_points[j:j+2], axis=0)
                seg_j_max = np.max(curve_points[j:j+2], axis=0)
                
                # Skip if bounding boxes are far apart (quick rejection)
                bbox_gap = max(
                    max(seg_i_min[0] - seg_j_max[0], seg_j_min[0] - seg_i_max[0]),
                    max(seg_i_min[1] - seg_j_max[1], seg_j_min[1] - seg_i_max[1])
                )
                if bbox_gap > min_distance:
                    continue  # Too far, skip this pair
                
                dist = self._segment_to_segment_distance(
                    curve_points[i], curve_points[i + 1],
                    curve_points[j], curve_points[j + 1]
                )
                
                if dist < min_distance:
                    violations += 1
                    # Severity: how much it violates the threshold (normalized to [0, 1])
                    severity = 1.0 - (dist / min_distance)
                    total_severity += severity
                    
                    # Early termination: if we found many violations, stop checking
                    if violations >= max_violations_to_check:
                        break
            
            if violations >= max_violations_to_check:
                break
        
        if violations == 0:
            return 1.0
        
        # Penalize based on average violation severity
        avg_severity = total_severity / violations
        penalty = max(0.0, 1.0 - avg_severity * 1.0)
        return penalty

    def quality(self, info):
        """Compute a quality score for a track, returning 0.0 for invalid geometry."""
        track_points = info.get('track_points', None)
        if track_points is None:
            return 0.0

        steps = info.get('steps', 4000)
        finished = info.get('finished', False)
        points = np.asarray(track_points, dtype=float)
        curve_points = info.get('curve_points', None)
        if curve_points is None:
            curve_points = interpolate_curves(points, samples_per_segment=10, tension=info.get('tension', None), bias=info.get('bias', None))
        curve_points = np.asarray(curve_points, dtype=float)

        margin = float(self._track_width) * 0.5 + 2.0
        if len(curve_points) > 0:
            if (
                np.min(curve_points[:, 0]) < margin or
                np.max(curve_points[:, 0]) > (self._width - 1 - margin) or
                np.min(curve_points[:, 1]) < margin or
                np.max(curve_points[:, 1]) > (self._height - 1 - margin)
            ):
                return 0.0
        trajectory_end = info.get('trajectory_end', None)
        if trajectory_end is None:
            trajectory_end = points[-1] if len(points) > 0 else None
        if trajectory_end is None:
            return 0.0
        final_point = points[-1] if len(points) > 0 else None
        if final_point is None:
            return 0.0
        dist_to_goal = np.linalg.norm(final_point - trajectory_end)
        control_total_length = np.sum(np.linalg.norm(points[1:] - points[:-1], axis=1)) if len(points) > 1 else 1.0
        curve_total_length = np.sum(np.linalg.norm(curve_points[1:] - curve_points[:-1], axis=1)) if len(curve_points) > 1 else control_total_length
        completion_pct = 1.0 - min(dist_to_goal / (curve_total_length + 1e-6), 1.0)
        if finished:
            completion_pct = 1.0

        diffs = curve_points[1:] - curve_points[:-1]
        v1 = diffs[:-1]
        v2 = diffs[1:]
        v1_norm = np.linalg.norm(v1, axis=1, keepdims=True)
        v2_norm = np.linalg.norm(v2, axis=1, keepdims=True)
        valid = (v1_norm[:, 0] > 1e-3) & (v2_norm[:, 0] > 1e-3)
        v1_unit = np.zeros_like(v1)
        v2_unit = np.zeros_like(v2)
        v1_unit[valid] = v1[valid] / v1_norm[valid]
        v2_unit[valid] = v2[valid] / v2_norm[valid]
        dots = np.einsum('ij,ij->i', v1_unit, v2_unit)
        dots = np.clip(dots, -1.0, 1.0)
        turn_angles = np.arccos(dots)
        turn_angles = turn_angles[valid]
        avg_curvature = np.mean(turn_angles) if turn_angles.size > 0 else 0.0
        curvature_variety = np.std(turn_angles) if turn_angles.size > 0 else 0.0

        if len(points) > 1:
            control_segment_lengths = np.linalg.norm(points[1:] - points[:-1], axis=1)
            mean_length = np.mean(control_segment_lengths)
            std_length = np.std(control_segment_lengths)

            total_length = float(np.sum(control_segment_lengths))
            min_len = 1500.0
            max_len = 6500.0
            if total_length < min_len:
                length_score = total_length / min_len
            elif total_length > max_len:
                length_score = max_len / total_length
            else:
                length_score = 1.0

            variance_penalty = np.exp(-std_length / (mean_length + 1e-6))

            preferred_min = 180.0
            preferred_max = 520.0
            out_of_range = np.sum((control_segment_lengths < preferred_min) | (control_segment_lengths > preferred_max))
            out_of_range_penalty = 1.0 - min(out_of_range / max(1, len(control_segment_lengths)), 1.0)
        else:
            variance_penalty = 1.0
            out_of_range_penalty = 1.0
            length_score = 0.0

        min_curvature = np.deg2rad(8)
        if avg_curvature < min_curvature:
            curvature_penalty = (avg_curvature / min_curvature)
        else:
            curvature_penalty = 1.0

        ideal_curvature = np.deg2rad(18)
        curvature_score = np.exp(-((avg_curvature - ideal_curvature) ** 2) / (2 * (ideal_curvature/2) ** 2)) if avg_curvature > 0 else 0.0
        sharp_turns = np.sum(np.array(turn_angles) > np.deg2rad(50))
        sharp_turn_penalty = 1.0 - min((sharp_turns - 2) / 3.0, 1.0) if sharp_turns > 2 else 1.0
        max_turn = float(np.max(turn_angles)) if turn_angles.size > 0 else 0.0
        if max_turn > np.deg2rad(120):
            return 0.0
        if max_turn > np.deg2rad(85):
            sharp_turn_penalty *= 0.25
        ideal_variety = np.deg2rad(10)
        variety_score = np.exp(-((curvature_variety - ideal_variety) ** 2) / (2 * (ideal_variety/2) ** 2)) if avg_curvature > 0 else 0.0

        intersections = count_self_intersections(curve_points)
        if intersections > 0:
            return 0.0

        intersection_score = 1.0
        
        # Compute spatial clustering penalty to prevent overlapping track geometry
        spatial_clustering_penalty = self._compute_spatial_clustering_penalty(points, min_segment_gap=3, threshold_multiplier=1.1)
        
        # Compute segment proximity penalty to prevent later segments from overlapping earlier ones
        segment_proximity_penalty = self._compute_segment_proximity_penalty(curve_points, min_distance=self._track_width * 3.0)

        min_steps = 30 * (len(points) - 1)
        max_steps_for_scoring = 200 * (len(points) - 1)
        steps = info.get('steps', 0)
        if not finished:
            time_score = 0.0
        elif steps < min_steps:
            time_score = max(0.0, 1.0 - (min_steps - steps) / min_steps)
        elif steps > max_steps_for_scoring:
            time_score = max(0.0, 1.0 - (steps - max_steps_for_scoring) / max_steps_for_scoring)
        else:
            time_score = 1.0

        w_completion = 0.20
        w_curvature = 0.12
        w_variety = 0.04
        w_curvature_penalty = 0.10
        w_sharp_turn_penalty = 0.10
        w_variance_penalty = 0.10
        w_out_of_range_penalty = 0.10
        w_length_score = 0.08
        w_intersection = 0.04
        w_spatial_clustering = 0.01
        w_segment_proximity = 0.10
        w_time = 0.01
        quality = (
            w_completion * completion_pct +
            w_curvature * curvature_score +
            w_variety * variety_score +
            w_curvature_penalty * curvature_penalty +
            w_sharp_turn_penalty * sharp_turn_penalty +
            w_variance_penalty * variance_penalty +
            w_out_of_range_penalty * out_of_range_penalty +
            w_length_score * length_score +
            w_intersection * intersection_score +
            w_spatial_clustering * spatial_clustering_penalty +
            w_segment_proximity * segment_proximity_penalty +
            w_time * time_score
        )
        return quality

    def diversity(self, info1, info2):
        return 0.0

    def controlability(self, info, control):
        return 1.0

    def render(self, content=None, frame_sampling=5, skip=None):
        """Return a list of PIL frames displaying the trajectory of the car."""
        if skip is None:
            skip = self._skip_render
        if skip:
            return []

        track_points, tension, bias = self._extract_content(content)
        trajectory = self._get_cached_trajectory(track_points, tension=tension, bias=bias)

        if trajectory is not None and len(trajectory) > 0:
            adaptive_sampling = max(1, int(len(trajectory) // 250))
            frame_sampling = max(int(frame_sampling), adaptive_sampling)

        track_points_np = self._normalize_track_points(track_points)
        curve_points = interpolate_curves(track_points_np, samples_per_segment=10, tension=tension, bias=bias)
        curve_np = np.asarray(curve_points, dtype=float)

        grass_color = (34, 139, 34)
        edge_color = (10, 10, 10)
        road_color = (215, 215, 215)
        centerline_color = (120, 120, 120)

        frames = []
        car_length = 30
        car_width = 16

        scaled_curve = [(int(round(x)), int(round(y))) for (x, y) in curve_np]
        half_width = float(self._track_width) * 0.5

        def _segment_intersection_point(a1, a2, b1, b2):
            ax1, ay1 = a1
            ax2, ay2 = a2
            bx1, by1 = b1
            bx2, by2 = b2
            dax = ax2 - ax1
            day = ay2 - ay1
            dbx = bx2 - bx1
            dby = by2 - by1
            denom = dax * dby - day * dbx
            if abs(denom) < 1e-12:
                return None
            dx = bx1 - ax1
            dy = by1 - ay1
            t = (dx * dby - dy * dbx) / denom
            u = (dx * day - dy * dax) / denom
            if t <= 1e-9 or t >= 1.0 - 1e-9 or u <= 1e-9 or u >= 1.0 - 1e-9:
                return None
            return (ax1 + t * dax, ay1 + t * day)

        def _trim_self_intersections(polyline, max_passes=6):
            pts = [(float(x), float(y)) for (x, y) in polyline]
            if len(pts) < 4:
                return pts
            for _ in range(max_passes):
                m = len(pts) - 1
                found = False
                for i in range(m - 2):
                    a1 = pts[i]
                    a2 = pts[i + 1]
                    for j in range(i + 2, m):
                        if j - i <= 1:
                            continue
                        b1 = pts[j]
                        b2 = pts[j + 1]
                        p = _segment_intersection_point(a1, a2, b1, b2)
                        if p is None:
                            continue
                        pts = pts[: i + 1] + [p] + pts[j + 1 :]
                        found = True
                        break
                    if found:
                        break
                if not found:
                    break
            return pts

        left_edge_f = []
        right_edge_f = []
        n = len(curve_np)
        if n >= 2:
            for j in range(n):
                if j == 0:
                    dir_prev = curve_np[1] - curve_np[0]
                else:
                    dir_prev = curve_np[j] - curve_np[j - 1]
                if j == n - 1:
                    dir_next = curve_np[j] - curve_np[j - 1]
                else:
                    dir_next = curve_np[j + 1] - curve_np[j]

                avg_dir = dir_prev + dir_next
                norm = float(np.linalg.norm(avg_dir))
                if norm == 0.0:
                    perp = np.array([0.0, 0.0], dtype=float)
                else:
                    perp = np.array([-avg_dir[1], avg_dir[0]], dtype=float) / norm

                left = curve_np[j] + perp * half_width
                right = curve_np[j] - perp * half_width
                left_edge_f.append((float(left[0]), float(left[1])))
                right_edge_f.append((float(right[0]), float(right[1])))

            left_edge_f = _trim_self_intersections(left_edge_f)
            right_edge_f = _trim_self_intersections(right_edge_f)
            left_edge = [(int(round(x)), int(round(y))) for (x, y) in left_edge_f]
            right_edge = [(int(round(x)), int(round(y))) for (x, y) in right_edge_f]

        track_polygon = left_edge + right_edge[::-1]

        track_background = Image.new("RGB", (self._width, self._height), grass_color)
        bg_draw = ImageDraw.Draw(track_background)
        if len(track_polygon) >= 3:
            bg_draw.polygon(track_polygon, fill=road_color)

        edge_line_width = 2
        if len(left_edge) > 1:
            bg_draw.line(left_edge, fill=edge_color, width=edge_line_width)
        if len(right_edge) > 1:
            bg_draw.line(right_edge, fill=edge_color, width=edge_line_width)

        if len(scaled_curve) > 1:
            bg_draw.line(scaled_curve, fill=centerline_color, width=2)

        def _rotated_rect(center_x, center_y, forward, right, half_len, half_wid):
            fx, fy = forward
            rx, ry = right
            return [
                (center_x + fx * half_len + rx * half_wid, center_y + fy * half_len + ry * half_wid),
                (center_x + fx * half_len - rx * half_wid, center_y + fy * half_len - ry * half_wid),
                (center_x - fx * half_len - rx * half_wid, center_y - fy * half_len - ry * half_wid),
                (center_x - fx * half_len + rx * half_wid, center_y - fy * half_len + ry * half_wid),
            ]

        for i, state in enumerate(trajectory):
            if i % frame_sampling != 0:
                continue
            img = track_background.copy()
            draw = ImageDraw.Draw(img)

            car_x, car_y = state[0], state[1]
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
            for wx, wy in wheel_centers:
                wheel = _rotated_rect(wx, wy, forward, right, half_wheel_len, half_wheel_wid)
                draw.polygon(wheel, fill=(25, 25, 25), outline=(0, 0, 0))

            corners = [
                (car_x + cos_a * dx - sin_a * dy, car_y + sin_a * dx + cos_a * dy),
                (car_x + cos_a * dx + sin_a * dy, car_y + sin_a * dx - cos_a * dy),
                (car_x - cos_a * dx + sin_a * dy, car_y - sin_a * dx - cos_a * dy),
                (car_x - cos_a * dx - sin_a * dy, car_y - sin_a * dx + cos_a * dy),
            ]
            draw.polygon(corners, fill=(255, 0, 0), outline=(0, 0, 0))
            front_x = car_x + cos_a * dx
            front_y = car_y + sin_a * dx
            draw.line([(car_x, car_y), (front_x, front_y)], fill=(0, 0, 255), width=3)
            frames.append(img)

        return frames



