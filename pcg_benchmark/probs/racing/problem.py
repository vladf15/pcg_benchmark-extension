from .engine import CarPhysicsEngine
from .agent import SimpleAgent
from pcg_benchmark.probs import Problem
from pcg_benchmark.spaces import ArraySpace, IntegerSpace, FloatSpace, DictionarySpace
import numpy as np
import json
from PIL import Image, ImageDraw
from pcg_benchmark.probs.racing.utils import interpolate_curves, count_self_intersections
import os
import itertools

class RacingProblem(Problem):

    def __init__(self, num_points=None, **kwargs):
        Problem.__init__(self, **kwargs)
        self._track_width = kwargs.get("track_width", 50)
        self._width = kwargs.get("width", 800)
        self._height = kwargs.get("height", 600)

        # Determine number of points
        if num_points is None:
            num_points = kwargs.get("num_points", 6)
        self.num_points = num_points

        # Default track is a straight line from (100, 100) to (400, 400)
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

        # Always initialize _car_state to a valid state
        self._car_state = np.array([self._default_track_points[0][0], self._default_track_points[0][1], 0.0, 0.0, 0.0])

        # Content space: (N,2) track points, (N-1,) tension, (N-1,) bias
        self._content_space = DictionarySpace({
            "track_points": ArraySpace((self.num_points, 2), FloatSpace(0, min(self._width, self._height) - 20)),
            "tension": ArraySpace((self.num_points-1,), FloatSpace(-1, 1)),
            "bias": ArraySpace((self.num_points-1,), FloatSpace(-1, 1)),
        })
        self._control_space = DictionarySpace({
            "steering": FloatSpace(-1, 1),
            "throttle": FloatSpace(-1, 1),
        })
        
    def reset(self, track_points=None):
        # Accept either array or dictionary, keep as tuples/lists for indexing
        if track_points is None:
            track_points = self._default_track_points
        elif isinstance(track_points, dict):
            track_points = track_points.get("track_points", self._default_track_points)
        # Ensure all points are tuples
        track_points = np.array(track_points)
        if len(track_points) < 2:
            start_direction = (1.0, 0.0)
        else:
            x0, y0 = track_points[0]
            x1, y1 = track_points[1]
            start_direction = (x1 - x0, y1 - y0)
        start_angle = np.arctan2(start_direction[1], start_direction[0])
        # For math, convert to array and validate shape
        track_points_np = np.array(track_points)
        if track_points_np.ndim == 1:
            track_points_np = track_points_np.reshape(-1, 2)
        self._curve_points = interpolate_curves(track_points_np, samples_per_segment=10)
        if not hasattr(self, '_engine') or self._engine is None:
            self._engine = CarPhysicsEngine(start_position=self._curve_points[0], start_angle=start_angle)
        else:
            self._engine.start_position = self._curve_points[0]
            self._engine.start_angle = start_angle
        if not hasattr(self, '_agent') or self._agent is None:
            self._agent = SimpleAgent(self._curve_points)
        else:
            self._agent.curve_points = self._curve_points
        state = self._engine.reset()
        self._car_state = state
        return state

    def step(self, action, track_points=None):
        # Use cached curve points
        state = self._engine.step(action)
        self._car_state = state
        # Use last track point for termination
        if track_points is None:
            track_points = self._default_track_points
        elif isinstance(track_points, dict):
            track_points = track_points.get("track_points", self._default_track_points)
        track_points = np.array(track_points)
        x, y = state[0], state[1]
        final_idx = len(track_points) - 1
        final_target = track_points[final_idx]
        dist_to_final = np.hypot(final_target[0] - x, final_target[1] - y)
        final_threshold = 10.0
        done = dist_to_final < final_threshold
        reward = -dist_to_final
        info = {"waypoint": getattr(self._agent, "current_curve_idx", None)}
        return state, reward, done, info

    def evaluate(self, content=None, max_steps=10000):
        # Accept either array or dictionary
        if content is None:
            track_points = self._default_track_points
        elif isinstance(content, dict):
            track_points = content.get("track_points", self._default_track_points)
        else:
            track_points = content
        track_points = np.array(track_points)
        state = self.reset(track_points=track_points)
        self._agent.reset()
        done = False
        steps = 0
        trajectory = [state.copy()]
        while not done and steps < max_steps:
            action = self._agent.act(state)
            state, reward, done, info = self.step(action, track_points=track_points)
            trajectory.append(state.copy())
            steps += 1
        return trajectory

    def info(self, content):
        # Accept either array or dictionary
        if content is not None and isinstance(content, dict):
            track_points = content.get("track_points", self._default_track_points)
        elif content is not None:
            track_points = content
        else:
            track_points = self._default_track_points
        track_points = np.array(track_points)
        if track_points.ndim == 1:
            track_points = track_points.reshape(-1, 2)
        num_points = self.num_points
        segment_lengths = [np.linalg.norm(track_points[i+1] - track_points[i]) for i in range(num_points-1)]
        total_length = sum(segment_lengths)
        avg_length = np.mean(segment_lengths)
        max_length = np.max(segment_lengths)
        min_length = np.min(segment_lengths)
        
        turn_angles = []
        for i in range(1, num_points-1):
            v1 = track_points[i] - track_points[i-1]
            v2 = track_points[i+1] - track_points[i]
            if np.linalg.norm(v1) > 1e-3 and np.linalg.norm(v2) > 1e-3:
                v1 /= np.linalg.norm(v1)
                v2 /= np.linalg.norm(v2)
                angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
                turn_angles.append(angle)
        avg_turn = np.mean(turn_angles) if turn_angles else 0.0
        max_turn = np.max(turn_angles) if turn_angles else 0.0
        min_turn = np.min(turn_angles) if turn_angles else 0.0
        #trajectory calculated by running agent on track
        trajectory = self.evaluate(content=track_points)
        steps = len(trajectory)
        finished = steps < 10000 and np.linalg.norm(track_points[-1] - trajectory[-1][:2]) < 10.0
        return {
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
        }


    def quality(self, info):
        # Penalize self-intersections
        track_points = info.get('track_points', None)
        if track_points is None:
            return 0.0
        intersections = count_self_intersections(track_points)
        intersection_penalty = 1.0 / (1.0 + intersections)
        # Reward equal mix of turns and straights
        turn_angles = []
        points = np.array(track_points)
        for i in range(1, len(points)-1):
            v1 = points[i] - points[i-1]
            v2 = points[i+1] - points[i]
            if np.linalg.norm(v1) > 1e-3 and np.linalg.norm(v2) > 1e-3:
                v1 /= np.linalg.norm(v1)
                v2 /= np.linalg.norm(v2)
                angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
                turn_angles.append(angle)
        if not turn_angles:
            return intersection_penalty * 0.5  # Only penalty if no turns
        # Define a threshold for 'straight' vs 'turn' (e.g., 15 degrees)
        straight_thresh = np.deg2rad(15)
        num_turns = sum(a > straight_thresh for a in turn_angles)
        num_straights = sum(a <= straight_thresh for a in turn_angles)
        # Reward is highest when turns and straights are balanced
        total = num_turns + num_straights
        if total == 0:
            mix_score = 0.0
        else:
            mix_score = 1.0 - abs(num_turns - num_straights) / total
        # Combine: intersection penalty and mix reward
        return intersection_penalty * mix_score

    def diversity(self, info1, info2):
        return 0.0

    def controlability(self, info, control):
        return 1.0

    def render(self, content=None, frame_sampling=5):
        # Accept either array or dictionary
        if content is not None and isinstance(content, dict):
            track_points = content.get("track_points", self._default_track_points)
        elif content is not None:
            track_points = np.array(content)
        else:
            track_points = self._default_track_points
        trajectory = self.evaluate(content=track_points)
        # Interpolate curve for rendering
        track_points_np = np.array(track_points)
        if track_points_np.ndim == 1:
            track_points_np = track_points_np.reshape(-1, 2)
        curve_points = interpolate_curves(track_points_np, samples_per_segment=10)
        frames = []
        car_length = 30
        car_width = 16
        for i, state in enumerate(trajectory):
            if i % frame_sampling != 0:
                continue
            img = Image.new("RGB", (self._width, self._height), (255, 255, 255))
            draw = ImageDraw.Draw(img)
            # Draw the center curve
            scaled_curve = [(int(x), int(y)) for (x, y) in curve_points]
            # Compute left/right edge curves using bisector for smoothness
            left_edge = []
            right_edge = []
            half_width = self._track_width / 2.0
            n = len(curve_points)
            for j in range(n):
                # Get previous and next direction
                if j == 0:
                    dir_prev = curve_points[1] - curve_points[0]
                else:
                    dir_prev = curve_points[j] - curve_points[j-1]
                if j == n-1:
                    dir_next = curve_points[j] - curve_points[j-1]
                else:
                    dir_next = curve_points[j+1] - curve_points[j]
                # Average direction
                avg_dir = dir_prev + dir_next
                norm = np.linalg.norm(avg_dir)
                if norm == 0:
                    perp = np.array([0, 0])
                else:
                    perp = np.array([-avg_dir[1], avg_dir[0]]) / norm
                left = curve_points[j] + perp * half_width
                right = curve_points[j] - perp * half_width
                left_edge.append((int(left[0]), int(left[1])))
                right_edge.append((int(right[0]), int(right[1])))
            # Draw filled track area (polygon)
            track_polygon = left_edge + right_edge[::-1]
            draw.polygon(track_polygon, fill=(220, 220, 220))
            # Draw edges
            if len(left_edge) > 1:
                draw.line(left_edge, fill=(0, 0, 0), width=2)
            if len(right_edge) > 1:
                draw.line(right_edge, fill=(0, 0, 0), width=2)
            # Draw centerline
            if len(scaled_curve) > 1:
                draw.line(scaled_curve, fill=(100, 100, 100), width=2)
            # Draw car
            car_x, car_y = state[0], state[1]
            if len(state) > 2:
                angle = state[2]
            else:
                angle = 0.0
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            dx = car_length / 2.0
            dy = car_width / 2.0
            corners = [
                (car_x + cos_a * dx - sin_a * dy, car_y + sin_a * dx + cos_a * dy),  # front right
                (car_x + cos_a * dx + sin_a * dy, car_y + sin_a * dx - cos_a * dy),  # front left
                (car_x - cos_a * dx + sin_a * dy, car_y - sin_a * dx - cos_a * dy),  # rear left
                (car_x - cos_a * dx - sin_a * dy, car_y - sin_a * dx + cos_a * dy),  # rear right
            ]
            draw.polygon(corners, fill=(255, 0, 0), outline=(0, 0, 0))
            front_x = car_x + cos_a * dx
            front_y = car_y + sin_a * dx
            draw.line([(car_x, car_y), (front_x, front_y)], fill=(0, 0, 255), width=3)
            frames.append(img)
        return frames



