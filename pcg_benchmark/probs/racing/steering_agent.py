import math
import numpy as np

class SteeringAgent:
    """SEEK/AVOID steering controller that follows a centerline and pushes away from walls."""

    def __init__(self, path_points, track_width=50, wall_avoid_dist=30.0, avoid_strength=1.5, max_speed=20.0, max_force=5.0, arrive_radius=10.0):
        self.path_points = np.array(path_points)
        self.track_width = track_width
        self.wall_avoid_dist = wall_avoid_dist
        self.avoid_strength = avoid_strength
        self.max_speed = max_speed
        self.max_force = max_force
        self.arrive_radius = arrive_radius
        self._pos = np.zeros(2, dtype=float)
        self._vel = np.zeros(2, dtype=float)
        self._desired_vel = np.zeros(2, dtype=float)
        self._avoid_force = np.zeros(2, dtype=float)
        self._precompute_path_vectors()
        self.reset()

    def _precompute_path_vectors(self):
        """Precompute unit tangents and normals for the path."""
        n = len(self.path_points)
        self._tangents = np.zeros((n, 2), dtype=float)
        self._normals = np.zeros((n, 2), dtype=float)
        for i in range(n):
            if i < n - 1:
                tangent = self.path_points[i + 1] - self.path_points[i]
            else:
                tangent = np.zeros(2, dtype=float)
            norm_tan = np.linalg.norm(tangent)
            if norm_tan > 1e-6:
                tangent = tangent / norm_tan
                normal = np.array([-tangent[1], tangent[0]])
            else:
                tangent = np.zeros(2, dtype=float)
                normal = np.zeros(2, dtype=float)
            self._tangents[i] = tangent
            self._normals[i] = normal

    def reset(self):
        """Reset progress along the path."""
        self.current_idx = 0

    def act(self, car_state):
        """Compute steering/throttle using a desired-velocity formulation."""
        x, y, angle, velocity, steering_angle = car_state
        self._pos[0] = x
        self._pos[1] = y
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        self._vel[0] = cos_a * velocity
        self._vel[1] = sin_a * velocity
        if self.current_idx >= len(self.path_points):
            return {'steering': 0.0, 'throttle': 0.0}
        target = self.path_points[self.current_idx]
        to_target = target - self._pos
        dist = math.hypot(float(to_target[0]), float(to_target[1]))
        if dist < self.arrive_radius:
            self.current_idx += 1
            desired_speed = self.max_speed * (dist / self.arrive_radius)
        else:
            desired_speed = self.max_speed
        if dist > 1e-6:
            self._desired_vel[:] = to_target / (dist + 1e-6) * desired_speed
        else:
            self._desired_vel[:] = 0.0

        normal = self._normals[self.current_idx]
        offset = float((self._pos[0] - target[0]) * normal[0] + (self._pos[1] - target[1]) * normal[1])
        self._avoid_force[:] = 0.0
        if abs(offset) > self.track_width/2 - self.wall_avoid_dist:
            direction = -np.sign(offset)
            self._avoid_force[:] = normal * direction * self.avoid_strength * (self.track_width/2 - abs(offset)) / self.wall_avoid_dist
        steer = self._desired_vel + self._avoid_force - self._vel
        steer_norm = math.hypot(float(steer[0]), float(steer[1]))
        if steer_norm > self.max_force:
            steer = steer / steer_norm * self.max_force
        vxs = float((self._vel + steer)[0])
        vys = float((self._vel + steer)[1])
        desired_angle = math.atan2(vys, vxs)
        steering = desired_angle - angle
        steering = (steering + np.pi) % (2 * np.pi) - np.pi
        desired_speed_now = math.hypot(float(self._desired_vel[0]), float(self._desired_vel[1]))
        throttle = (desired_speed_now - float(velocity)) / self.max_speed
        throttle = -1.0 if throttle < -1.0 else 1.0 if throttle > 1.0 else throttle
        return {'steering': steering, 'throttle': throttle}
