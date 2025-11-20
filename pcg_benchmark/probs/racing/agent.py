import numpy as np

class SimpleAgent:
    def __init__(self, track_points):
        self.track_points = track_points
        self.current_waypoint = 1

    def reset(self):
        self.current_waypoint = 1

    def act(self, car_state):
        x, y, angle, velocity, steering_angle = car_state
        if self.current_waypoint >= len(self.track_points):
            return {'steering': 0.0, 'throttle': 0.0}
        target = self.track_points[self.current_waypoint]
        dx = target[0] - x
        dy = target[1] - y
        distance = np.hypot(dx, dy)
        if distance < 10.0:
            self.current_waypoint += 1
        angle_to_target = np.arctan2(dy, dx)
        steering = angle_to_target - angle
        steering = (steering + np.pi) % (2 * np.pi) - np.pi
        return {'steering': steering, 'throttle': 1.0}