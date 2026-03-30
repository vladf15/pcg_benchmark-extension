import math
import numpy as np

class CarPhysicsEngine:
    """2D bicycle-model car physics with a Pacejka LUT.

    State returned by `get_state()` is `[x, y, angle, speed, steering_angle]`.
    Control input to `step()` is a dict with normalized keys:
    - `steering` in [-1, 1]
    - `throttle` in [-1, 1] (positive drive, negative brake)
    """

    def _build_pacejka_lut(self):
        """Precompute Pacejka (magic formula) coefficients over a fixed grid."""
        B_lat, C_lat, D_lat, E_lat = 1.6, 1.70, 1.0, 0.97
        B_lon, C_lon, D_lon, E_lon = 1.9, 1.95, 1.0, 0.97

        slip_angles = np.linspace(-np.deg2rad(25), np.deg2rad(25), 101)
        self.lut_slip_angles = slip_angles
        self._lut_sa_min = float(slip_angles[0])
        self._lut_sa_max = float(slip_angles[-1])
        self._lut_sa_step = float(slip_angles[1] - slip_angles[0])
        x_lat = B_lat * slip_angles
        self.lut_lateral = D_lat * np.sin(C_lat * np.arctan(x_lat - E_lat * (x_lat - np.arctan(x_lat))))
        
        slip_ratios = np.linspace(-1.0, 1.0, 101)
        self.lut_slip_ratios = slip_ratios
        self._lut_sr_min = float(slip_ratios[0])
        self._lut_sr_max = float(slip_ratios[-1])
        self._lut_sr_step = float(slip_ratios[1] - slip_ratios[0])
        x_lon = B_lon * slip_ratios
        self.lut_longitudinal = D_lon * np.sin(C_lon * np.arctan(x_lon - E_lon * (x_lon - np.arctan(x_lon))))

        self._lut_sa_inv_step = 1.0 / self._lut_sa_step
        self._lut_sr_inv_step = 1.0 / self._lut_sr_step

    def _lut_interp_uniform(self, y, x, x_min, x_max, inv_step):
        """Fast scalar interpolation for a uniformly spaced LUT grid."""
        if x <= x_min:
            return float(y[0])
        if x >= x_max:
            return float(y[-1])
        f = (x - x_min) * inv_step
        i = int(f)
        t = f - i
        y0 = float(y[i])
        y1 = float(y[i + 1])
        return y0 + (y1 - y0) * t

    def _get_lateral_force_coeff(self, slip_angle):
        """Return normalized lateral force coefficient for a slip angle."""
        return self._lut_interp_uniform(
            self.lut_lateral,
            float(slip_angle),
            self._lut_sa_min,
            self._lut_sa_max,
            self._lut_sa_inv_step,
        )

    def _get_longitudinal_force_coeff(self, slip_ratio):
        """Return normalized longitudinal force coefficient for a slip ratio."""
        return self._lut_interp_uniform(
            self.lut_longitudinal,
            float(slip_ratio),
            self._lut_sr_min,
            self._lut_sr_max,
            self._lut_sr_inv_step,
        )

    @staticmethod
    def _sign(x: float) -> float:
        """Sign helper returning -1, 0, or +1."""
        return -1.0 if x < 0.0 else (1.0 if x > 0.0 else 0.0)
    
    def __init__(
        self,
        start_position,
        start_angle=0.0,
        time_step=0.1,
        friction_coef=0.0,
        max_steering=np.deg2rad(28.0),
        max_speed=55.0,
        max_throttle=1.0,
        max_brake=-1.0,
        steering_rate=np.deg2rad(180.0),
        length=5.0,
        lateral_friction=0.55,
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

        # Vehicle properties
        self.mass = 1350.0
        self.lf = self.length * 0.55
        self.lr = self.length * 0.45
        self.inertia_z = self.mass * (self.length ** 2) * 0.34
        self.c_rr = 0.016
        self.rho_air = 1.225
        self.cd_a = 1.35

        # Forces in Newtons. Throttle is normalized in [-1, 1].
        self.max_drive_force = 6000.0
        self.max_brake_force = 12000.0

        self._throttle_state = 0.0
        self.throttle_slew_rate = 2.2

        self.tire_mu = 1.55

        self.max_slip_angle = np.deg2rad(25.0)
        self.max_slip_ratio = 1.0

        self.g = 9.81
        self.normal_load = self.mass * self.g
        self.load_front = self.normal_load * 0.55  # 55% front, 45% rear typical
        self.load_rear = self.normal_load * 0.45

        self.max_tire_force_front = self.load_front * self.tire_mu
        self.max_tire_force_rear = self.load_rear * self.tire_mu
        
        self.car_state = np.zeros(5, dtype=float)
        self._build_pacejka_lut()
        self.reset()

    def reset(self):
        """Reset state to the configured start pose."""
        self.position = self.start_position.copy()
        self.angle = self.start_angle
        self.velocity = np.zeros(2, dtype=float)
        self.yaw_rate = 0.0
        self.steering_angle = 0.0
        self._wheel_omega = 0.0
        self._throttle_state = 0.0
        return self.get_state()

    def step(self, action):
        """Advance the physics by one time step using a control dict."""
        time_step = self.time_step
        cos_a0 = math.cos(self.angle)
        sin_a0 = math.sin(self.angle)
        v_forward0 = cos_a0 * self.velocity[0] + sin_a0 * self.velocity[1]
        v0 = abs(float(v_forward0))
        max_steer_low = math.radians(80.0)
        max_steer_high = float(self.max_steering)
        # Blend over ~0..14 m/s from low-speed lock to high-speed limit.
        t_lock = v0 / 14.0
        if t_lock < 0.0:
            t_lock = 0.0
        elif t_lock > 1.0:
            t_lock = 1.0
        max_steering = max_steer_low * (1.0 - t_lock) + max_steer_high * t_lock

        steering_input = float(action.get('steering', 0.0))
        if steering_input < -1.0:
            steering_input = -1.0
        elif steering_input > 1.0:
            steering_input = 1.0
        target_steering = steering_input * max_steering

        throttle_cmd = float(action.get('throttle', 0.0))
        if throttle_cmd < -1.0:
            throttle_cmd = -1.0
        elif throttle_cmd > 1.0:
            throttle_cmd = 1.0

        throttle_input = throttle_cmd
        dth = throttle_input - float(self._throttle_state)
        max_dth = float(self.throttle_slew_rate) * time_step
        if dth < -max_dth:
            dth = -max_dth
        elif dth > max_dth:
            dth = max_dth
        throttle_input = float(self._throttle_state) + dth
        if throttle_input < -1.0:
            throttle_input = -1.0
        elif throttle_input > 1.0:
            throttle_input = 1.0
        self._throttle_state = throttle_input

        steering_diff = target_steering - self.steering_angle
        max_steering_change = self.steering_rate * time_step
        if steering_diff < -max_steering_change:
            steering_diff = -max_steering_change
        elif steering_diff > max_steering_change:
            steering_diff = max_steering_change
        self.steering_angle += steering_diff
        if self.steering_angle < -max_steering:
            self.steering_angle = -max_steering
        elif self.steering_angle > max_steering:
            self.steering_angle = max_steering

        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)

        v_forward = cos_a * self.velocity[0] + sin_a * self.velocity[1]
        v_lateral = -sin_a * self.velocity[0] + cos_a * self.velocity[1]

        yaw_rate = float(self.yaw_rate)

        speed_abs = abs(v_forward)
        rolling_force = self.c_rr * self.mass * self.g
        drag_force = 0.5 * self.rho_air * self.cd_a * (speed_abs ** 2)
        resist_force = (rolling_force + drag_force) * (1.0 if v_forward >= 0.0 else -1.0)

        if throttle_input >= 0.0:
            drive_force = throttle_input * self.max_drive_force
            brake_force = 0.0
        else:
            drive_force = 0.0
            brake_force = (-throttle_input) * self.max_brake_force

        Fx_req = drive_force - brake_force

        v = abs(v_forward)
        t = (v - 6.0) / 24.0
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        steering_gain = (1.75 * (1.0 - t)) + (0.45 * t)
        delta = self.steering_angle * steering_gain

        vxf = v_forward
        vyf = v_lateral + self.lf * yaw_rate
        vxr = v_forward
        vyr = v_lateral - self.lr * yaw_rate

        vx_eps = 0.6
        vxf_safe = self._sign(vxf) * max(abs(vxf), vx_eps)
        vxr_safe = self._sign(vxr) * max(abs(vxr), vx_eps)

        slip_angle_front = delta - math.atan2(vyf, vxf_safe)
        slip_angle_rear = -math.atan2(vyr, vxr_safe)

        msa = self.max_slip_angle
        if slip_angle_front < -msa:
            slip_angle_front = -msa
        elif slip_angle_front > msa:
            slip_angle_front = msa
        if slip_angle_rear < -msa:
            slip_angle_rear = -msa
        elif slip_angle_rear > msa:
            slip_angle_rear = msa

        lat_coeff_f = self._get_lateral_force_coeff(slip_angle_front)
        lat_coeff_r = self._get_lateral_force_coeff(slip_angle_rear)

        Fy0_f = float(lat_coeff_f) * self.max_tire_force_front
        Fy0_r = float(lat_coeff_r) * self.max_tire_force_rear

        v_lat_scale = v / (v + 1.0)
        Fy0_f *= v_lat_scale
        Fy0_r *= v_lat_scale

        Fx_cap_total = (self.max_tire_force_front + self.max_tire_force_rear)
        wheel_omega = float(getattr(self, '_wheel_omega', 0.0))
        omega_rate = (Fx_req / max(self.mass, 1.0)) * 0.6
        wheel_omega += omega_rate * time_step
        if wheel_omega < 0.0:
            wheel_omega = 0.0

        v_ref = 2.0
        denom = speed_abs if speed_abs > v_ref else v_ref
        kappa = (wheel_omega - v_forward) / denom
        if kappa < -self.max_slip_ratio:
            kappa = -self.max_slip_ratio
        elif kappa > self.max_slip_ratio:
            kappa = self.max_slip_ratio

        lon_coeff = self._get_longitudinal_force_coeff(kappa)
        Fx0_total = float(lon_coeff) * Fx_cap_total
        self._wheel_omega = wheel_omega

        Fx0_f = Fx0_total * (self.load_front / self.normal_load)
        Fx0_r = Fx0_total * (self.load_rear / self.normal_load)

        muFzf = self.max_tire_force_front
        muFzr = self.max_tire_force_rear

        Fy_f = Fy0_f
        Fy_r = Fy0_r

        if muFzf > 1e-6:
            mag2 = Fx0_f * Fx0_f + Fy_f * Fy_f
            lim2 = muFzf * muFzf
            if mag2 > lim2:
                s = muFzf / math.sqrt(mag2)
                Fx0_f *= s
                Fy_f *= s
        if muFzr > 1e-6:
            mag2 = Fx0_r * Fx0_r + Fy_r * Fy_r
            lim2 = muFzr * muFzr
            if mag2 > lim2:
                s = muFzr / math.sqrt(mag2)
                Fx0_r *= s
                Fy_r *= s

        c = math.cos(delta)
        s = math.sin(delta)
        Fx_f = Fx0_f * c - Fy_f * s
        Fy_f_b = Fx0_f * s + Fy_f * c

        Fx_r = Fx0_r
        Fy_r_b = Fy_r

        Fx_body = Fx_f + Fx_r - resist_force
        Fy_body = Fy_f_b + Fy_r_b

        ax = Fx_body / self.mass + yaw_rate * v_lateral
        ay = Fy_body / self.mass - yaw_rate * v_forward
        v_forward += ax * time_step
        v_lateral += ay * time_step

        mz = (self.lf * Fy_f_b) - (self.lr * Fy_r_b)
        yaw_rate += (mz / max(self.inertia_z, 1.0)) * time_step

        if v_forward < 0.0:
            v_forward = 0.0
        elif v_forward > self.max_speed:
            v_forward = self.max_speed

        if self.lateral_friction > 0.0:
            damp = math.exp(-self.lateral_friction * time_step)
            v_lateral *= damp
            yaw_rate *= damp

        self.yaw_rate = yaw_rate

        self.velocity[0] = cos_a * v_forward - sin_a * v_lateral
        self.velocity[1] = sin_a * v_forward + cos_a * v_lateral

        self.angle += yaw_rate * time_step

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