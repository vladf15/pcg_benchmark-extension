from .engine import CarPhysicsEngine
from .agent import SimpleAgent
from pcg_benchmark.probs import Problem
from pcg_benchmark.spaces import ArraySpace, IntegerSpace, FloatSpace, DictionarySpace
import numpy as np
import json
from PIL import Image, ImageDraw
import os

class RacingProblem(Problem):
    def __init__(self, **kwargs):
        Problem.__init__(self, **kwargs)
        self._default_track_points = kwargs.get("track_points", [(200,200), (220, 250), (230, 250), (250, 260), (300, 300), (320, 310)])
        self._track_width = kwargs.get("track_width", 50)
        self._width = kwargs.get("width", 800)
        self._height = kwargs.get("height", 600)

        # Always initialize _car_state to a valid state
        self._car_state = np.array([self._default_track_points[0][0], self._default_track_points[0][1], 0.0, 0.0, 0.0])

        # Content space: algorithms will generate (N,2) array of track points
        self._content_space = ArraySpace((len(self._default_track_points), 2), FloatSpace(0, min(self._width, self._height) - 20))
        self._control_space = DictionarySpace({
            "steering": FloatSpace(-1, 1),
            "throttle": FloatSpace(-1, 1),
            
        })
        
    def reset(self, track_points=None):
        # Use provided track_points or fallback to default
        if track_points is None:
            track_points = self._default_track_points
        self._engine = CarPhysicsEngine(start_position=track_points[0], start_angle=0.0)
        self._agent = SimpleAgent(track_points)
        state = self._engine.reset()
        self._car_state = state
        return state

    def step(self, action, track_points=None):
        # Use provided track_points or fallback to default
        if track_points is None:
            track_points = self._default_track_points
        state = self._engine.step(action)
        self._car_state = state
        x, y = state[0], state[1]
        # Check if car is close enough to the final waypoint
        final_idx = len(track_points) - 1
        final_target = track_points[final_idx]
        dist_to_final = np.hypot(final_target[0] - x, final_target[1] - y)
        # You may adjust this threshold as needed
        final_threshold = 10.0
        done = dist_to_final < final_threshold
        reward = -dist_to_final
        info = {"waypoint": self._agent.current_waypoint}
        return state, reward, done, info

    def evaluate(self, content=None, max_steps=10000):
        # content is the track_points array (N,2)
        if content is None:
            track_points = self._default_track_points
        else:
            track_points = np.array(content)
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
        return {"car_state": self._car_state}

    def quality(self, info):
        return 1.0

    def diversity(self, info1, info2):
        return 0.0

    def controlability(self, info, control):
        return 1.0

    def render(self, content=None, frame_sampling=5):
        """
        Returns a list of images (frames) of the agent driving around the track, with the car as a rotated rectangle.
        :param content: the track points to use
        :param frame_sampling: sample every Nth frame (default 1 = all frames)
        :return: list of PIL Image objects
        """
        if content is None:
            track_points = self._default_track_points
        else:
            track_points = np.array(content)
        trajectory = self.evaluate(content=track_points)
        frames = []
        car_length = 30
        car_width = 16
        for i, state in enumerate(trajectory):
            if i % frame_sampling != 0:
                continue
            img = Image.new("RGB", (self._width, self._height), (255, 255, 255))
            draw = ImageDraw.Draw(img)
            scaled_points = [(
                int(x),
                int(y)
            ) for (x, y) in track_points]
            if len(scaled_points) > 1:
                draw.line(scaled_points, fill=(0, 0, 0), width=self._track_width)
            draw.line(scaled_points, fill=(100, 100, 100), width=2)
            car_x, car_y = state[0], state[1]
            if len(state) > 2:
                angle = state[2]
            else:
                angle = 0.0
            # Calculate rectangle corners
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
            # Optionally, draw a line to indicate front
            front_x = car_x + cos_a * dx
            front_y = car_y + sin_a * dx
            draw.line([(car_x, car_y), (front_x, front_y)], fill=(0, 0, 255), width=3)
            frames.append(img)
        return frames

    

