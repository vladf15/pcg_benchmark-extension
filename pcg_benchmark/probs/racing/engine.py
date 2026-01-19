# A simple 2D car physics engine with friction and inertia
import numpy as np

class CarPhysicsEngine:
    def __init__(self, start_position, start_angle=0.0, time_step=0.1, friction_coef=0.1, max_steering=np.pi/4, max_speed=60.0, max_throttle=5.0, max_brake=-10.0, steering_rate=np.pi/2, length=2.5, lateral_friction=2.0):
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
        norm_throttle = np.clip(action.get('throttle', 0.0), -1.0, 1.0)
        throttle = norm_throttle * self.max_throttle if norm_throttle >= 0 else norm_throttle * -self.max_brake

        # Limit steering rate
        steering_diff = target_steering - self.steering_angle
        max_steering_change = self.steering_rate * self.time_step
        self.steering_angle += np.clip(steering_diff, -max_steering_change, max_steering_change)
        self.steering_angle = np.clip(self.steering_angle, -self.max_steering, self.max_steering)

        # Precompute trigonometric values
        cos_a = np.cos(self.angle)
        sin_a = np.sin(self.angle)

        # Transform velocity to local frame
        v_forward = cos_a * self.velocity[0] + sin_a * self.velocity[1]
        v_lateral = -sin_a * self.velocity[0] + cos_a * self.velocity[1]

        #throttle/brake is applied to forward velocity
        accel_scale = max(0.0, 1.0 - v_forward / self.max_speed) if throttle > 0 else 1.0
        effective_accel = throttle * accel_scale
        acceleration = effective_accel - self.friction_coef * v_forward
        v_forward += acceleration * self.time_step
        v_forward = np.clip(v_forward, 0.0, self.max_speed)

        # --- Realistic tire grip and understeer ---
        # Parameters for tire model
        tire_stiffness_front = 8000.0  # N/rad
        tire_stiffness_rear = 8000.0   # N/rad
        mass = 1200.0  # kg
        g = 9.81
        # Nonlinear tire force model (saturates at high slip angles)
        def tire_force(stiffness, slip_angle):
            # Saturate force for large slip angles (drifting)
            max_slip = np.deg2rad(15)
            force = -stiffness * slip_angle
            if abs(slip_angle) > max_slip:
                force = -stiffness * max_slip * np.sign(slip_angle)
            return force

        # Calculate slip angles
        if abs(v_forward) > 0.1:
            beta = np.arctan2(v_lateral, abs(v_forward))  # body slip angle
            # Calculate yaw rate (omega)
            if abs(self.steering_angle) > 1e-4:
                turning_radius = self.length / np.tan(self.steering_angle)
                yaw_rate = v_forward / turning_radius
            else:
                yaw_rate = 0.0
            # Steering effectiveness drops at high speed
            steering_effect = 1.0 / (1.0 + 0.03 * v_forward)
            slip_angle_front = beta + self.steering_angle * steering_effect - self.length * 0.5 * yaw_rate / max(abs(v_forward), 1e-3)
            slip_angle_rear = beta - self.length * 0.5 * yaw_rate / max(abs(v_forward), 1e-3)
        else:
            slip_angle_front = 0.0
            slip_angle_rear = 0.0
        # Lateral tire forces (nonlinear model)
        F_yf = tire_force(tire_stiffness_front, slip_angle_front)
        F_yr = tire_force(tire_stiffness_rear, slip_angle_rear)
        # Limit tire force to friction circle
        max_tire_force = mass * g * 0.7  # 0.7 = friction coefficient
        F_yf = np.clip(F_yf, -max_tire_force, max_tire_force)
        F_yr = np.clip(F_yr, -max_tire_force, max_tire_force)
        # Update lateral velocity and yaw rate
        v_lateral += (F_yf + F_yr) / mass * self.time_step
        # More sliding: reduce lateral friction
        v_lateral *= np.exp(-self.lateral_friction * 0.5 * self.time_step)

        # Transform velocity back to world frame
        self.velocity[0] = cos_a * v_forward - sin_a * v_lateral
        self.velocity[1] = sin_a * v_forward + cos_a * v_lateral

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