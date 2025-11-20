# A simple 2D car physics engine with friction and inertia
import numpy as np

class CarPhysicsEngine:
    def __init__(self, start_position, start_angle=0.0, time_step=0.1, friction_coef=0.1, max_steering=np.pi/4, max_speed=10.0, max_throttle=1.0, max_brake=-5.0, steering_rate=np.pi/2, length=2.5, lateral_friction=2.0):
        self.time_step = time_step
        self.friction_coef = friction_coef
        self.max_steering = max_steering
        self.max_throttle = max_throttle
        self.max_brake = max_brake  # negative value, m/s^2
        self.max_speed = max_speed
        self.length = length  # wheelbase 
        self.steering_rate = steering_rate  # max steering change per second
        self.lateral_friction = lateral_friction  # higher = more grip
        self.start_position = start_position
        self.start_angle = start_angle
        self.reset()

    def reset(self):
        self.position = np.array(self.start_position, dtype=float)
        self.angle = self.start_angle
        self.velocity = np.zeros(2, dtype=float)  # 2d vector for velocity [vx, vy]
        self.steering_angle = 0.0
        return self.get_state()

    def step(self, action):
        # action: dict with 'steering' (radians), 'throttle' (acceleration, can be negative for braking)
        target_steering = np.clip(action.get('steering', 0.0), -self.max_steering, self.max_steering)
        # Throttle input is normalized between [-1, 1] and then scaled by max_throttle/max_brake
        norm_throttle = np.clip(action.get('throttle', 0.0), -1.0, 1.0)
        if norm_throttle >= 0:
            throttle = norm_throttle * self.max_throttle
        else:
            throttle = norm_throttle * -self.max_brake  # max_brake is negative, so -max_brake is positive

        # Limit steering rate (how fast the wheels can turn)
        steering_diff = target_steering - self.steering_angle
        max_steering_change = self.steering_rate * self.time_step
        steering_change = np.clip(steering_diff, -max_steering_change, max_steering_change)
        self.steering_angle += steering_change
        self.steering_angle = np.clip(self.steering_angle, -self.max_steering, self.max_steering)

        # Get car's local velocity (forward, lateral)
        cos_a = np.cos(self.angle)
        sin_a = np.sin(self.angle)
        rot = np.array([[cos_a, sin_a], [-sin_a, cos_a]])  # world to local
        v_local = np.dot(rot, self.velocity)
        v_forward = v_local[0]
        v_lateral = v_local[1]

        # Apply throttle/brake to forward velocity
        # Acceleration decreases as car nears top speed
        accel_scale = max(0.0, 1.0 - v_forward / self.max_speed) if throttle > 0 else 1.0
        effective_accel = throttle * accel_scale
        acceleration = effective_accel - self.friction_coef * v_forward
        v_forward += acceleration * self.time_step
        v_forward = np.clip(v_forward, 0.0, self.max_speed)

        # Lateral friction (simulate grip/sliding)
        max_lateral_acc = self.lateral_friction * 9.81
        if abs(self.steering_angle) > 1e-4 and v_forward > 0.1:
            turning_radius = self.length / np.tan(self.steering_angle)
            desired_lateral = v_forward ** 2 / abs(turning_radius)
            if abs(desired_lateral) > max_lateral_acc:
                # Too much force: reduce lateral velocity (simulate sliding)
                v_lateral *= max_lateral_acc / abs(desired_lateral)
        # Apply friction to lateral velocity
        v_lateral *= np.exp(-self.lateral_friction * self.time_step)

        # Convert local velocity back to world frame
        v_local = np.array([v_forward, v_lateral])
        inv_rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])  # local to world
        self.velocity = inv_rot @ v_local

        # Update heading based on forward velocity and steering
        if abs(self.steering_angle) > 1e-4:
            turning_radius = self.length / np.tan(self.steering_angle)
            angular_velocity = v_forward / turning_radius
        else:
            angular_velocity = 0.0
        self.angle += angular_velocity * self.time_step

        # Update position
        self.position += self.velocity * self.time_step

        return self.get_state()

    def get_state(self):
        # Returns [x, y, angle, speed, steering_angle]
        speed = np.linalg.norm(self.velocity)
        return np.array([self.position[0], self.position[1], self.angle, speed, self.steering_angle])