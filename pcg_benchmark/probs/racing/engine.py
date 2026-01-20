import math
import numpy as np

class CarPhysicsEngine:
    """2D kinematic-ish car model with steering, acceleration, and lateral slip."""

    def _tire_force(self, stiffness, slip_angle):
        """Return a saturated lateral tire force for a slip angle."""
        slip_abs = abs(slip_angle)
        if slip_abs > self.max_slip:
            slip_angle = self.max_slip if slip_angle >= 0.0 else -self.max_slip
        return -stiffness * slip_angle
    
    def __init__(
        self,
        start_position,
        start_angle=0.0,
        time_step=0.1,
        friction_coef=0.1,
        max_steering=np.pi / 4,
        max_speed=60.0,
        max_throttle=5.0,
        max_brake=-10.0,
        steering_rate=np.pi / 2,
        length=2.5,
        lateral_friction=2.0,
    ):
        self.time_step = time_step
        self.friction_coef = friction_coef
        self.max_steering = max_steering
        self.max_throttle = max_throttle
        self.max_brake = max_brake
        self.max_speed = max_speed
        self.length = length
        self.steering_rate = steering_rate
        self.lateral_friction = lateral_friction
        self.start_position = np.array(start_position, dtype=float)
        self.start_angle = start_angle

        self.tire_stiffness_front = 8000.0
        self.tire_stiffness_rear = 8000.0
        self.mass = 1200.0
        self.g = 9.81
        self.max_slip = np.deg2rad(15)
        self.max_tire_force = self.mass * self.g * 0.7
        self.car_state = np.zeros(5, dtype=float)
        self.reset()

    def reset(self):
        """Reset state to the configured start pose."""
        self.position = self.start_position.copy()
        self.angle = self.start_angle
        self.velocity = np.zeros(2, dtype=float)
        self.steering_angle = 0.0
        return self.get_state()

    def step(self, action):
        """Advance the physics by one time step using a control dict."""
        def _clamp(x, lo, hi):
            return lo if x < lo else hi if x > hi else x

        target_steering = _clamp(float(action.get('steering', 0.0)), -self.max_steering, self.max_steering)
        norm_throttle = _clamp(float(action.get('throttle', 0.0)), -1.0, 1.0)
        throttle = norm_throttle * self.max_throttle if norm_throttle >= 0.0 else norm_throttle * -self.max_brake

        time_step = self.time_step
        max_steering = self.max_steering

        steering_diff = target_steering - self.steering_angle
        max_steering_change = self.steering_rate * time_step
        self.steering_angle += _clamp(steering_diff, -max_steering_change, max_steering_change)
        self.steering_angle = _clamp(self.steering_angle, -max_steering, max_steering)

        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)

        v_forward = cos_a * self.velocity[0] + sin_a * self.velocity[1]
        v_lateral = -sin_a * self.velocity[0] + cos_a * self.velocity[1]

        accel_scale = max(0.0, 1.0 - v_forward / self.max_speed) if throttle > 0 else 1.0
        effective_accel = throttle * accel_scale
        acceleration = effective_accel - self.friction_coef * v_forward
        v_forward += acceleration * time_step
        v_forward = _clamp(v_forward, 0.0, self.max_speed)

        if abs(v_forward) > 0.1:
            beta = math.atan2(v_lateral, abs(v_forward))
            if abs(self.steering_angle) > 1e-4:
                turning_radius = self.length / math.tan(self.steering_angle)
                yaw_rate = v_forward / turning_radius
            else:
                yaw_rate = 0.0
            steering_effect = 1.0 / (1.0 + 0.03 * v_forward)
            slip_angle_front = beta + self.steering_angle * steering_effect - self.length * 0.5 * yaw_rate / max(abs(v_forward), 1e-3)
            slip_angle_rear = beta - self.length * 0.5 * yaw_rate / max(abs(v_forward), 1e-3)
        else:
            slip_angle_front = 0.0
            slip_angle_rear = 0.0

        F_yf = self._tire_force(self.tire_stiffness_front, slip_angle_front)
        F_yr = self._tire_force(self.tire_stiffness_rear, slip_angle_rear)
        F_yf = _clamp(F_yf, -self.max_tire_force, self.max_tire_force)
        F_yr = _clamp(F_yr, -self.max_tire_force, self.max_tire_force)
        v_lateral += (F_yf + F_yr) / self.mass * time_step
        v_lateral *= math.exp(-self.lateral_friction * 0.5 * time_step)

        self.velocity[0] = cos_a * v_forward - sin_a * v_lateral
        self.velocity[1] = sin_a * v_forward + cos_a * v_lateral

        if abs(self.steering_angle) > 1e-4:
            turning_radius = self.length / math.tan(self.steering_angle)
            angular_velocity = v_forward / turning_radius
        else:
            angular_velocity = 0.0
        self.angle += angular_velocity * time_step

        self.position += self.velocity * time_step
        return self.get_state()

    def get_state(self):
        """Return the current state as a preallocated ndarray."""
        self.car_state[0] = self.position[0]
        self.car_state[1] = self.position[1]
        self.car_state[2] = self.angle
        self.car_state[3] = math.hypot(float(self.velocity[0]), float(self.velocity[1]))
        self.car_state[4] = self.steering_angle
        return self.car_state