"""Simcade car physics v2 ("2D Forza" feel) for the driving model.

Standalone upgrade of the benchmark's CarPhysicsEngine (which stays frozen,
since the GA quality function depends on it). Same external interface:
`step({'steering': s, 'throttle': t})` with both in [-1, 1], `get_state()`
returning [x, y, angle, speed, steering_angle], plus position / velocity /
angle / yaw_rate / steering_angle attributes.

Structure follows TORCS SimuV2 where it affects handling:
- 5 internal substeps of 20 ms per 100 ms control step (semi-implicit Euler)
- dynamic bicycle model; ONE magic-formula curve evaluated on the COMBINED
  slip magnitude, force split back by slip direction. The friction circle
  is therefore implicit: a tire braking hard has little slip left pointing
  sideways, so it cannot also corner hard. No hand-coded clamp, and no
  traction-control priority, so power really can break the rear out.
- load-sensitive grip (mu falls as vertical load rises), plus longitudinal
  and lateral weight transfer. Load sensitivity is what makes transfer a
  BALANCE change instead of a zero-sum move between axles.
- tire relaxation (first-order lag, 0.3 m relaxation length, integrated
  exactly) which is the physical stabilizer replacing v1's hacks
- aerodynamic drag AND downforce; top speed is drag-limited (~300 km/h),
  not a hard clamp
- torque curve through an auto-shifting 6-speed; acceleration has gears

What is deliberately simplified (the "simcade" part):
- no wheel-spin state: the pedal demands a force fraction (of the
  drivetrain in the current gear, or of the brake hardware) and the tire
  converts it back into slip. Over-demand pushes the tire past its curve
  peak, losing force AND cornering grip, so wheelspin and lockup have
  their consequences without the stiff wheel integrator (which is what
  historically exploded at this dt).
- two axles, not four wheels: no differential, no per-wheel lockup
- no suspension, roll, or pitch degrees of freedom (2D top-down)

Human input limits (no superhuman inputs):
- steering: fixed 33 deg lock, rate-limited to 70 deg/s at the road wheels
  (lock-to-lock takes ~0.94 s, matching quick human hands through a normal
  steering ratio)
- pedals: slew 6.0/s (full throttle to full brake takes ~0.33 s)
- the 10 Hz control interval itself is the reaction-time floor

Removed v1 driver aids: the speed-blended steering lock (75->30 deg), the
hidden steering gain schedule (1.05->0.45), the lateral-priority traction
clamp, the strong artificial yaw damping (0.6 -> 0.1 tire-scrub level), the
lateral velocity scrub, and the v/(v+1) lateral force fade. The agent has to
stabilize the car itself.
"""

import math

import numpy as np


def _pacejka(x, B, C, E):
    """Magic-formula shape, D folded into the caller (peak value 1.0)."""
    xb = B * x
    return math.sin(C * math.atan(xb - E * (xb - math.atan(xb))))


def _sign(x):
    return -1.0 if x < 0.0 else (1.0 if x > 0.0 else 0.0)


_PEAK_SLIP = math.radians(10.0)      # slip at which the tire curve peaks



class CarPhysicsEngineV2:

    def __init__(self, start_position, start_angle=0.0, time_step=0.1,
                 substeps=5):
        self.time_step = float(time_step)      # control interval (agent rate)
        self.substeps = int(substeps)
        self._dt = self.time_step / self.substeps

        # Body: same rounded 911 as v1 (1500 kg, 3.0 m wheelbase, 40/60).
        self.mass = 1500.0
        self.length = 3.0
        self.lf = 1.8                          # cg to front axle (rear-heavy)
        self.lr = 1.2
        self.inertia_z = self.mass * (self.length ** 2) * 0.34
        self.h_cg = 0.45                       # cg height, sets load transfer
        self.g = 9.81
        self.load_front_static = self.mass * self.g * 0.40
        self.load_rear_static = self.mass * self.g * 0.60
        self.tire_mu = 1.3

        # Tires: ONE magic-formula curve driven by the COMBINED slip
        # magnitude, split into longitudinal and lateral by direction
        # (TORCS SimuV2 does the same). The friction circle then falls out
        # of the tire model instead of being imposed by hand: at the limit
        # the force saturates in whatever direction the slip points, so
        # braking really does eat cornering grip with no extra clamp.
        # Peaks near 10 deg of slip angle, falls to ~0.89 of peak by 30 deg,
        # so slides cost grip but stay catchable (Forza-style).
        self._tire_BCE = (12.0, 1.6, 0.6)
        self.max_slip_angle = math.radians(30.0)
        self.relax_len = 0.3                   # tire relaxation length (m)
        self.slide_grip = 0.92                 # kinetic/static friction ratio
        # Load sensitivity (TORCS: grip falls as vertical load rises). This
        # is what makes weight transfer change the car's BALANCE rather than
        # just move a linear grip budget between the axles: the loaded axle
        # gains less than the unloaded axle loses, so total grip drops in
        # hard braking and hard cornering. mu = mu0 * (1 - k * (Fz/Fz0 - 1)),
        # clamped so it can never go negative or exceed mu0 at zero load.
        self.load_sens = 0.10
        self._Fz_ref_f = self.load_front_static
        self._Fz_ref_r = self.load_rear_static

        # Resistance and aero. Downforce adds grip with v^2; drag sets the
        # top speed (~84 m/s in 6th) instead of a hard clamp.
        self.c_rr = 0.015
        self.rho_air = 1.225
        self.cd_a = 0.65
        self.cl_a = 0.75
        self.aero_front_share = 0.40           # keeps aero balance neutral

        # Drivetrain: torque curve -> auto 6-speed -> rear wheels.
        self.wheel_radius = 0.33
        self.driveline_eff = 0.90
        self.gear_ratios = (13.4, 8.9, 6.6, 5.2, 4.2, 3.2)   # incl. final
        self.shift_up_rpm = 7400.0
        self.shift_down_rpm = 3600.0
        self.limiter_rpm = 8000.0
        self._torque_rpm = np.array([0.0, 1000.0, 3000.0, 6500.0, 8000.0])
        self._torque_nm = np.array([180.0, 250.0, 450.0, 450.0, 330.0])
        self.max_brake_force = 16000.0         # ~1.1 g, 100-0 in ~35 m

        # Human input limits (see module docstring).
        self.max_steering = math.radians(33.0)
        self.steering_rate = math.radians(70.0)
        self.throttle_slew_rate = 6.0

        # Track width, for lateral load transfer. In a two-axle model the
        # left/right split is not simulated directly, but transferring load
        # across the track still matters through LOAD SENSITIVITY: the
        # outside tire gains less grip than the inside one loses, so hard
        # cornering costs total grip. Modelled as an effective grip loss on
        # each axle proportional to its share of lateral acceleration.
        self.track_width_m = 1.6
        self.yaw_scrub = 0.1                   # small contact-patch scrub;
        # NOT the v1 stabilizer (that was 0.6 and did the driver's job).

        # Nominal top speed for observation normalization (actual top speed
        # is drag-limited near this value).
        self.max_speed = 83.0

        self.start_position = np.array(start_position, dtype=float)
        self.start_angle = float(start_angle)
        self.car_state = np.zeros(5, dtype=float)
        self.reset()

    # ── lifecycle ─────────────────────────────────────────────────────────

    def reset(self):
        self.position = self.start_position.copy()
        self.angle = float(self.start_angle)
        self.velocity = np.zeros(2, dtype=float)
        self.yaw_rate = 0.0
        self.steering_angle = 0.0
        self.gear = 0
        self.rpm = 0.0
        self._pedal = 0.0
        self._alpha_f = 0.0                    # relaxed slip angles
        self._alpha_r = 0.0
        self._ax_prev = 0.0                    # for longitudinal transfer
        self._ay_prev = 0.0                    # for lateral transfer
        # (Fz_f, Fz_r, Fx_f, Fx_r, Fy_f, Fy_r, Fy_total) from the last
        # substep, for diagnostics. Set here too so it can be read before
        # the first step.
        self.debug_forces = (self.load_front_static, self.load_rear_static,
                             0.0, 0.0, 0.0, 0.0, 0.0)
        return self.get_state()

    def get_state(self):
        self.car_state[0] = self.position[0]
        self.car_state[1] = self.position[1]
        self.car_state[2] = self.angle
        self.car_state[3] = math.hypot(float(self.velocity[0]),
                                       float(self.velocity[1]))
        self.car_state[4] = self.steering_angle
        return self.car_state

    # ── stepping ──────────────────────────────────────────────────────────

    def step(self, action):
        steer_cmd = min(max(float(action.get('steering', 0.0)), -1.0), 1.0)
        pedal_cmd = min(max(float(action.get('throttle', 0.0)), -1.0), 1.0)
        target_steer = steer_cmd * self.max_steering
        for _ in range(self.substeps):
            self._substep(target_steer, pedal_cmd)
        return self.get_state()

    def _grip(self, Fz, Fz_ref, ay=0.0):
        """Maximum tire force at vertical load Fz, with load sensitivity.

        Real tires lose grip coefficient as they are pressed harder, so
        doubling the load does not double the force. TORCS models this
        explicitly; without it, weight transfer is a zero-sum move between
        axles and the car has no balance to manage.

        `ay` is the current lateral acceleration. A two-axle model has no
        separate left and right tires, but transferring load across the
        track still costs grip for the same reason: the outside tire gains
        less than the inside one loses.
        """
        mu = self.tire_mu * (1.0 - self.load_sens * (Fz / Fz_ref - 1.0))
        x = min(abs(ay) * self.h_cg / (self.g * self.track_width_m), 1.0)
        return max(mu, 0.35 * self.tire_mu) * Fz * (1.0 - self.load_sens * x * x)

    def _axle_force(self, alpha, Fx_demand, mu_Fz):
        """Combined-slip tire force for one axle (TORCS SimuV2 structure).

        `alpha` is the (relaxed) slip angle and `Fx_demand` the longitudinal
        force the pedal is asking for. The longitudinal demand is converted
        back into an equivalent slip, combined with the slip angle as a
        vector, and ONE magic-formula curve evaluated on its magnitude. The
        force is then split by the slip direction, so:

        - a tire braking at its limit has almost no lateral force left
          (locked-up understeer)
        - a driven tire past its grip loses cornering force (power
          oversteer), with no separate friction-circle clamp
        - beyond the curve peak the force FALLS, which is what makes a
          breakaway snap rather than fade

        Returns (Fx, Fy). Fy is positive for a positive slip angle.
        """
        if mu_Fz <= 1e-6:
            return 0.0, 0.0
        frac = min(max(Fx_demand / mu_Fz, -1.6), 1.6)
        if abs(frac) < 1e-9:
            sx = 0.0
        else:
            # Longitudinal slip needed to actually DELIVER the demand,
            # given the slip angle already present. In TORCS the wheel spins
            # up until its slip produces the demanded drive torque; with no
            # wheel state, solving for that slip directly is the equivalent.
            # Skipping this step (using the free-rolling slip) silently
            # threw the demand away whenever the car was cornering: at 5 deg
            # of slip angle a 25%-of-grip throttle request delivered only
            # 14%, because adding a small kappa to a large alpha barely
            # rotates the combined slip vector.
            sx = self._slip_for_combined(abs(frac), abs(alpha))
            sx = math.copysign(sx, frac)
        sy = alpha
        s = math.hypot(sx, sy)
        if s < 1e-9:
            return 0.0, 0.0
        f = _pacejka(s, *self._tire_BCE) * mu_Fz
        # Past the curve peak the tire is sliding: apply kinetic friction so
        # the transition has a step, not a smooth fade.
        if s > math.radians(12.0):
            f *= self.slide_grip
        return f * sx / s, f * sy / s

    def _slip_for_combined(self, want_fx, alpha):
        """Longitudinal slip that delivers `want_fx` (as a fraction of the
        tire's peak force) alongside an existing slip angle `alpha`.

        Solved by bisection on the combined-slip force, which is monotone in
        sx up to the point where the tire saturates. If even unlimited slip
        cannot deliver the demand, the tire is over-driven and the largest
        searched slip is returned, which is a spinning (or locked) wheel.
        """
        def fx_at(sx):
            s = math.hypot(sx, alpha)
            if s < 1e-9:
                return 0.0
            f = _pacejka(s, *self._tire_BCE)
            if s > math.radians(12.0):
                f *= self.slide_grip
            return f * sx / s

        lo, hi = 0.0, _PEAK_SLIP * 3.0
        if fx_at(hi) < want_fx:
            return hi                      # over-demanded: wheel spins/locks
        # 12 iterations resolve the slip to ~0.01 degrees, far finer than
        # the force differences the car can feel, and half the cost of the
        # 24 the first version used (this runs twice per axle per substep).
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            if fx_at(mid) < want_fx:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _drive_force(self, v_forward):
        """Available drive force at the rear wheels: torque curve through
        the auto gearbox. rpm follows ground speed (no wheel-spin state)."""
        omega_wheel = max(v_forward, 0.0) / self.wheel_radius
        rpm = omega_wheel * self.gear_ratios[self.gear] * 60.0 / (2.0 * math.pi)
        # Auto shifts with hysteresis; after an upshift rpm lands ~4900,
        # well above the 3600 downshift point, so it cannot hunt.
        while self.gear < len(self.gear_ratios) - 1 and rpm > self.shift_up_rpm:
            self.gear += 1
            rpm = omega_wheel * self.gear_ratios[self.gear] * 60.0 / (2.0 * math.pi)
        while self.gear > 0 and rpm < self.shift_down_rpm:
            self.gear -= 1
            rpm = omega_wheel * self.gear_ratios[self.gear] * 60.0 / (2.0 * math.pi)
        self.rpm = rpm
        if rpm >= self.limiter_rpm:
            return 0.0                          # rev limiter
        torque = float(np.interp(rpm, self._torque_rpm, self._torque_nm))
        return (torque * self.gear_ratios[self.gear] * self.driveline_eff
                / self.wheel_radius)

    def _substep(self, target_steer, pedal_cmd):
        dt = self._dt

        # ── human-limited inputs ──────────────────────────────────────────
        d = target_steer - self.steering_angle
        max_d = self.steering_rate * dt
        self.steering_angle += min(max(d, -max_d), max_d)

        dp = pedal_cmd - self._pedal
        max_dp = self.throttle_slew_rate * dt
        self._pedal += min(max(dp, -max_dp), max_dp)

        # ── body-frame state ──────────────────────────────────────────────
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)
        v_fwd = cos_a * self.velocity[0] + sin_a * self.velocity[1]
        v_lat = -sin_a * self.velocity[0] + cos_a * self.velocity[1]
        yaw = self.yaw_rate
        v = math.hypot(v_fwd, v_lat)

        # ── axle loads: static + downforce + longitudinal transfer ────────
        downforce = 0.5 * self.rho_air * self.cl_a * v * v
        dFz = (self.h_cg / self.length) * self.mass * self._ax_prev
        Fz_f = self.load_front_static + downforce * self.aero_front_share - dFz
        Fz_r = self.load_rear_static + downforce * (1.0 - self.aero_front_share) + dFz
        # A wheel cannot push up on the road; keep a floor so extreme
        # (clamped) accelerations cannot produce negative grip.
        Fz_f = max(Fz_f, 0.15 * self.load_front_static)
        Fz_r = max(Fz_r, 0.15 * self.load_rear_static)
        # Grip limits, reduced by load sensitivity and by the lateral
        # transfer the current cornering is causing.
        muFz_f = self._grip(Fz_f, self._Fz_ref_f, self._ay_prev)
        muFz_r = self._grip(Fz_r, self._Fz_ref_r, self._ay_prev)

        # ── longitudinal demand from the pedal ────────────────────────────
        # Throttle asks for a share of the drive force available in this
        # gear (rear axle only, RWD); brake asks for a share of the brake
        # hardware, split front/rear by current load. These are DEMANDS: the
        # tire model below decides what is actually delivered, and an
        # over-demand slides the tire instead of being clipped here.
        drive_avail = self._drive_force(v_fwd)  # also updates gear/rpm
        if self._pedal >= 0.0:
            Fx_demand_f, Fx_demand_r = 0.0, self._pedal * drive_avail
        elif v_fwd <= 0.1:
            Fx_demand_f = Fx_demand_r = 0.0     # brakes hold, never reverse
        else:
            total = self._pedal * self.max_brake_force   # negative
            share_f = Fz_f / (Fz_f + Fz_r)
            Fx_demand_f, Fx_demand_r = total * share_f, total * (1.0 - share_f)

        # ── slip angles with exact-exponential relaxation ─────────────────
        delta = self.steering_angle             # no hidden gain schedule
        vx_safe = _sign(v_fwd) * max(abs(v_fwd), 0.6) if v_fwd != 0.0 else 0.6
        a_f_raw = delta - math.atan2(v_lat + self.lf * yaw, vx_safe)
        a_r_raw = -math.atan2(v_lat - self.lr * yaw, vx_safe)
        msa = self.max_slip_angle
        a_f_raw = min(max(a_f_raw, -msa), msa)
        a_r_raw = min(max(a_r_raw, -msa), msa)
        # First-order tire lag integrated exactly: unconditionally stable,
        # fast at speed (tau = 15 ms at 20 m/s), slow near standstill,
        # which is precisely what stabilizes low speed without fake scrub.
        k = 1.0 - math.exp(-max(v, 0.5) * dt / self.relax_len)
        self._alpha_f += (a_f_raw - self._alpha_f) * k
        self._alpha_r += (a_r_raw - self._alpha_r) * k

        # ── combined-slip tire forces (see _axle_force) ───────────────────
        Fx_f, Fy_f = self._axle_force(self._alpha_f, Fx_demand_f, muFz_f)
        Fx_r, Fy_r = self._axle_force(self._alpha_r, Fx_demand_r, muFz_r)

        # ── sum forces (front axle rotated by the steer angle) ────────────
        cd, sd = math.cos(delta), math.sin(delta)
        Fx_body = (Fx_f * cd - Fy_f * sd) + Fx_r
        Fy_f_b = Fx_f * sd + Fy_f * cd
        Fy_body = Fy_f_b + Fy_r

        speed_abs = abs(v_fwd)
        resist = (self.c_rr * self.mass * self.g
                  + 0.5 * self.rho_air * self.cd_a * speed_abs * speed_abs)
        Fx_body -= resist * (1.0 if v_fwd >= 0.0 else -1.0)

        # ── integrate (semi-implicit: velocities first, then pose) ────────
        # NO Coriolis terms here: velocity is stored in the WORLD frame and
        # re-decomposed with the updated heading at the next substep, so the
        # frame rotation already applies the kinematic coupling. Adding
        # yaw*v terms on top (as the v1 engine does) double-counts it and
        # silently wastes about half of the cornering force.
        ax = Fx_body / self.mass
        ay = Fy_body / self.mass
        v_fwd += ax * dt
        v_lat += ay * dt

        mz = self.lf * Fy_f_b - self.lr * Fy_r
        yaw += (mz / self.inertia_z) * dt
        yaw *= math.exp(-self.yaw_scrub * dt)

        # Safety rails well outside the physical envelope (drag limits real
        # top speed to ~84 m/s; these only stop numerical blowups).
        v_fwd = min(max(v_fwd, 0.0), 90.0)
        v_lat = min(max(v_lat, -30.0), 30.0)
        yaw = min(max(yaw, -6.0), 6.0)

        self._ax_prev = Fx_body / self.mass     # proper accel for transfer
        self._ay_prev = Fy_body / self.mass
        # Last-substep force breakdown, for physics_tests and debugging.
        self.debug_forces = (Fz_f, Fz_r, Fx_f, Fx_r, Fy_f, Fy_r, Fy_body)
        self.yaw_rate = yaw
        self.velocity[0] = cos_a * v_fwd - sin_a * v_lat
        self.velocity[1] = sin_a * v_fwd + cos_a * v_lat
        self.angle = (self.angle + yaw * dt + math.pi) % (2.0 * math.pi) - math.pi
        self.position += self.velocity * dt
