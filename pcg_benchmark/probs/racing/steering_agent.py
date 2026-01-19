import numpy as np

class SteeringAgent:
    """
    Agent implementing SEEK and AVOID steering behaviors.
    Seeks the center of the racing line and avoids track walls.
    """
    def __init__(self, path_points, track_width=50, wall_avoid_dist=30.0, avoid_strength=1.5, max_speed=20.0, max_force=5.0, arrive_radius=10.0):
        self.path_points = np.array(path_points)
        self.track_width = track_width
        self.wall_avoid_dist = wall_avoid_dist
        self.avoid_strength = avoid_strength
        self.max_speed = max_speed
        self.max_force = max_force
        self.arrive_radius = arrive_radius
        self.reset()

    def reset(self):
        self.current_idx = 0

    def act(self, car_state):
        x, y, angle, velocity, steering_angle = car_state
        pos = np.array([x, y])
        vel = np.array([np.cos(angle), np.sin(angle)]) * velocity
        # --- SEEK center of racing line ---
        if self.current_idx >= len(self.path_points):
            return {'steering': 0.0, 'throttle': 0.0}
        target = self.path_points[self.current_idx]
        to_target = target - pos
        dist = np.linalg.norm(to_target)
        # Arrive: slow down as we approach
        if dist < self.arrive_radius:
            self.current_idx += 1
            desired_speed = self.max_speed * (dist / self.arrive_radius)
        else:
            desired_speed = self.max_speed
        desired_vel = to_target / (dist + 1e-6) * desired_speed if dist > 1e-6 else np.zeros(2)

        # --- AVOID track walls ---
        # Estimate local tangent and normal to the path
        if self.current_idx < len(self.path_points) - 1:
            next_pt = self.path_points[self.current_idx + 1]
        else:
            next_pt = target
        tangent = next_pt - target
        norm_tan = np.linalg.norm(tangent)
        if norm_tan > 1e-6:
            tangent /= norm_tan
            normal = np.array([-tangent[1], tangent[0]])
        else:
            normal = np.array([0.0, 0.0])
        # Project car position onto normal from centerline
        offset = np.dot(pos - target, normal)
        # If close to wall, steer away
        avoid_force = np.zeros(2)
        if abs(offset) > self.track_width/2 - self.wall_avoid_dist:
            direction = -np.sign(offset)  # steer toward center
            avoid_force = normal * direction * self.avoid_strength * (self.track_width/2 - abs(offset)) / self.wall_avoid_dist
        # Combine seek and avoid
        steer = desired_vel + avoid_force - vel
        steer_norm = np.linalg.norm(steer)
        if steer_norm > self.max_force:
            steer = steer / steer_norm * self.max_force
        # Convert steering force to steering angle command
        desired_angle = np.arctan2((vel + steer)[1], (vel + steer)[0])
        steering = desired_angle - angle
        steering = (steering + np.pi) % (2 * np.pi) - np.pi
        # Throttle: proportional to how much we want to speed up
        throttle = np.clip((np.linalg.norm(desired_vel) - velocity) / self.max_speed, -1.0, 1.0)
        return {'steering': steering, 'throttle': throttle}
