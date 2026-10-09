import math
import numpy as np

class CarPhysicsEngine:
    """2D single-track (bicycle) car physics with a Pacejka LUT.

    State returned by `get_state()` is `[x, y, angle, speed, steering_angle]`,
    where speed is measured along the car.  Control input to `step()` is a dict
    with normalized keys:
    - `steering` in [-1, 1]
    - `throttle` in [-1, 1] (positive drive, negative brake)

    Every force comes from a physical mechanism: Pacejka tyre curves on
    load-sensitive, transferred axle loads, a torque curve through an auto
    gearbox, brakes, aerodynamic drag and downforce, and tyre relaxation.
    No steering assist, artificial damping, slip clamp or speed clamp: yaw is
    resisted only by the slip it creates, a tyre can pass its peak and break
    away, and top speed is power against drag.  Combined slip follows TORCS:
    one magic formula on the slip vector's magnitude, split along its
    direction, so the friction circle holds by construction.

    Two simplifications, documented where they occur: the pedal demands a
    FORCE and the slip delivering it is solved for (wheelspin and lockup have
    consequences, not dynamics); and two axles, not four wheels (no
    differential; lateral transfer enters only as the grip it costs).
    """

    # Magic-formula shape, TORCS' defaults for a tire (Wymann et al., TORCS;
    # source src/modules/simu/simuv2/wheel.cpp, read in the jeremybennett/torcs
    # mirror of the SourceForge tree: Ca 30, RFactor 0.8, EFactor 0.7, lines
    # 47-49), combined as
    #     C = 2 - asin(RFactor) * 2 / pi,   B = Ca / C,   E = EFactor
    # (lines 88-90), on the slip magnitude capped at 1.5 (line 221).  It peaks
    # at slip 0.18 (10.2 degrees), where a real tyre does.  Why not softer: a
    # curve peaking at 26.1 degrees gives 0.39 of its peak at the 2 degrees a
    # car corners on (0.74 here), and the car understeered to 5.4 m/s^2
    # against the 12.8 mu implies.
    _MF_RFACTOR = 0.8
    _MF_EFACTOR = 0.7
    _MF_CA = 30.0
    # Load sensitivity, also TORCS': grip per newton falls with load,
    #     mu(Fz) = mu * (lfMin + (lfMax - lfMin) * exp(lfK * Fz / opLoad)),
    # so load transfer changes the balance and the outside tyre cannot just
    # take over from the inside one.  lfMin 0.8, lfMax 1.6, operating load
    # 1.2 x static: TORCS' defaults (wheel.cpp lines 50-52, 228).
    _LF_MIN = 0.8
    _LF_MAX = 1.6
    _OP_LOAD_FACTOR = 1.2        # operating load, as a multiple of static load

    def _build_pacejka_lut(self):
        """Precompute the combined-slip magic formula over a fixed grid: one
        curve on the slip MAGNITUDE (as TORCS), not two forces clamped after."""
        self._mf_C = 2.0 - math.asin(self._MF_RFACTOR) * 2.0 / math.pi
        self._mf_B = self._MF_CA / self._mf_C
        self._mf_E = self._MF_EFACTOR
        self._lf_k = math.log((1.0 - self._LF_MIN) / (self._LF_MAX - self._LF_MIN))

        # TORCS caps the slip magnitude at 1.5; past it the curve is flat.
        slips = np.linspace(0.0, 1.5, 151)
        self._lut_s_min = float(slips[0])
        self._lut_s_max = float(slips[-1])
        self._lut_s_inv_step = 1.0 / float(slips[1] - slips[0])
        bx = self._mf_B * slips
        self.lut_combined = np.sin(
            self._mf_C * np.arctan(bx * (1.0 - self._mf_E) + self._mf_E * np.arctan(bx)))

        # Per lateral slip sy, the longitudinal slip where F(|s|) sx / |s|
        # peaks: the force is monotone in sx only up to there, so the search
        # for the slip meeting a demand stops there.  On the LUT's grid, so
        # it matches the engine's force.
        sx = np.linspace(0.0, 1.5, 1501)
        self._peak_sx = np.empty_like(slips)
        for i, sy in enumerate(slips):
            mag = np.minimum(np.hypot(sx, sy), 1.5)
            fx = np.interp(mag, slips, self.lut_combined) * sx / np.maximum(np.hypot(sx, sy), 1e-12)
            self._peak_sx[i] = sx[int(np.argmax(fx))]

    def _get_combined_force_coeff(self, slip_magnitude):
        """Normalized tire force for a combined slip magnitude."""
        return self._lut_interp_uniform(
            self.lut_combined, float(slip_magnitude),
            self._lut_s_min, self._lut_s_max, self._lut_s_inv_step,
        )

    def _grip(self, load, static_load, mu_nominal=None):
        """Peak tyre force at this vertical load (N); doubling the load does
        not double it, which makes weight transfer change the BALANCE."""
        # Defaults so physics_tests can measure the load curve on its own.
        if mu_nominal is None:
            mu_nominal = self.tire_mu
        op_load = max(self._OP_LOAD_FACTOR * static_load, 1e-6)
        mu = mu_nominal * (self._LF_MIN + (self._LF_MAX - self._LF_MIN)
                           * math.exp(self._lf_k * load / op_load))
        return mu * load

    def _axle_grip(self, load, transfer, static_load, mu_nominal):
        """Peak force of an axle's two tires, in newtons, with `transfer`
        newtons moved from the inside tire to the outside one.  A tire cannot
        pull on the road, so once the inside tire is unloaded it lifts and
        the outside one carries the whole axle."""
        half, half_static = 0.5 * load, 0.5 * static_load
        shift = min(transfer, half)
        return (self._grip(half + shift, half_static, mu_nominal)
                + self._grip(half - shift, half_static, mu_nominal))

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

    def __init__(
        self,
        start_position,
        start_angle=0.0,
        time_step=0.1,
        # 30 deg road-wheel lock (passenger cars 30-40, performance cars at the
        # low end): with the 2.45 m wheelbase a 4.24 m radius, the 911's real
        # 11.2 m kerb-to-kerb circle.  steering=1 is this angle at every speed,
        # as the Simulated Car Racing interface requires.
        max_steering=np.deg2rad(30.0),
        max_speed=85.5,      # observation scale only; top speed is power against drag
        # 60 deg/s at the road wheels: through the 992's 15.0-12.25:1 ratio
        # (Porsche 2020) 900 deg/s at the wheel, a fast evasive input (FMVSS
        # 126's sine-with-dwell, 49 CFR 571.126 S7.9, peaks at 2 pi 0.7 270 =
        # 1190 deg/s).  A modelling choice; it binds (lock takes 5 steps).
        steering_rate=np.deg2rad(60.0),
        length=2.45,                        # 992 wheelbase (m)
        # 5 substeps: 50 Hz physics under a 10 Hz controller.  Over 40 genomes
        # (all representations) mean quality 0.7613 / 0.7719 / 0.7720 / 0.7669
        # and laps 36 / 38 / 38 / 37 at 30 / 50 / 100 / 200 Hz, for 0.257 /
        # 0.300 / 0.454 / 0.787 s per evaluation: 50 Hz is the cheapest rate
        # that scores as every rate above it.  (TORCS's 500 Hz serves rendering.)
        physics_substeps=5,
    ):
        self.time_step = time_step
        self.physics_substeps = int(physics_substeps)
        self.max_steering = max_steering
        self.max_speed = max_speed
        self.length = length                # wheelbase (m)
        self.steering_rate = steering_rate
        self.start_position = np.array(start_position, dtype=float)
        self.start_angle = start_angle

        # Car: Porsche 911 Carrera S (992.1, PDK), from Porsche's 2020 US
        # technical data ("Porsche 2020") unless marked: wheelbase 2450 mm,
        # rear track 1557 mm, 308 km/h, turning circle 11.2 m, tyres 245/35
        # ZR20 and 305/30 ZR21, 1534 kg (the 3382 lb curb weight, the same
        # document the drivetrain is calibrated to).  40/60 front/rear is a
        # MODELLING CHOICE (not published).
        self.mass = 1534.0
        self.lf = self.length * 0.60        # cg sits nearer the rear axle
        self.lr = self.length * 0.40
        # Yaw inertia from a dynamic index Izz / (m lf lr) of 1 (near 1 for
        # most cars; Milliken and Milliken 1995, not re-read): 2210 kg m^2.
        # ESTIMATED (not published).  NHTSA's database (Heydinger et al.
        # 1999) gives 0.854-1.243, median 1.095, over 94 passenger cars.
        self.inertia_z = self.mass * self.lf * self.lr
        # Cg height, the lever arm of load transfer.  DERIVED from the car's
        # 1300 mm height (Porsche 2020): cg over roof height averages 0.388 in
        # NHTSA's database (Heydinger et al. 1999; 38 cars with a driver, sd
        # 0.012, range 0.365-0.415, independent of class and mass, Fig. 4), so
        # 0.50 m (0.47-0.54).  Caveat: those cars are 1.33-1.50 m high, the
        # 911 just below them.  Why not the database's lowest car (0.489 m):
        # it ignores the 911 being 30 mm lower than any.  A 10% error is 10%
        # in load transfer.  Accepted cost (user decision): from 0.48 m up the
        # car no longer oversteers under power in physics_tests (2.9 degrees
        # body slip against 6.9).  calibration/nhtsa_cgroof.py, oversteer_cg.py.
        self.h_cg = 0.50
        # Rolling resistance.  ESTIMATED, and high: the EU tyre label
        # (Regulation (EU) 2020/740, Annex I Part A) runs from class A (6.5
        # N/kN) to E (10.6 and above); 0.015 is 15 N/kN, 229 N against 2686 N
        # of drag at top speed, so it barely moves the car.
        self.c_rr = 0.015
        self.rho_air = 1.225                # ISA sea-level air density, kg/m^3
        # Drag area.  Cd 0.31 is published (Porsche 2020); frontal area is
        # not, so cd_a is a CALIBRATION, fitted with driveline_eff to the
        # published gears and figures: 0.56 / 0.57 / 0.58 m^2 give 310.8 /
        # 309.1 / 307.4 km/h, 0.58 the published 191 mph.  It implies 1.87 m^2,
        # 0.78 of the car's width-by-height box, and carries the rolling and
        # downforce estimates with it.  calibration/porsche_fit.py.
        self.cd_a = 0.58
        # Downforce.  ESTIMATED (not published; a wingless 911 makes little):
        # 165 N (1.1% of weight) at 30 m/s, 700 N at 61.7 m/s.  Split like the
        # weight so it does not shift the balance with speed (a MODELLING
        # CHOICE).
        self.cl_a = 0.30
        self.aero_front_share = 0.40

        # ── Drivetrain: torque curve through an auto gearbox ──────────────
        # The published engine, not a constant force: 530 N.m from 2300 to
        # 5000 rpm, 331 kW (443 hp, 450 PS) at 6500, maximum engine speed
        # 7500 rpm (Porsche 2020).  The point at 6500 rpm (486 N.m) follows
        # from the power; those at 0, 1500 and 7500 rpm (300, 450, 400 N.m)
        # are ESTIMATED, since the curve outside the plateau is not
        # published.  The shift points (7200 up, 3200 down) are a MODELLING
        # CHOICE: 7200 rpm sits under the 7500 rpm limiter, with hysteresis so
        # the gearbox cannot hunt (an upshift on the published ratios drops
        # to 4700-5800 rpm).  Upshifting at 6800 / 7200 / 7500 rpm gives a
        # quarter mile of 11.80 / 11.71 / 11.67 s.  Torque reaches the road
        # through whichever gear is engaged, so acceleration falls away with
        # speed and steps at each shift, the way a real car does.
        self._torque_rpm = np.array([0.0, 1500.0, 2300.0, 5000.0, 6500.0, 7500.0])
        self._torque_nm  = np.array([300.0, 450.0, 530.0, 530.0, 486.0, 400.0])
        self.limiter_rpm  = 7500.0
        self.shift_up_rpm = 7200.0
        self.shift_down_rpm = 3200.0
        # Clutch-slip engine speed off the line: 3000 rpm models launch
        # control, against Porsche's launch figures of 3.3 s to 60 mph and an
        # 11.7 s quarter mile (model: 3.24, 11.71 s).  A check, not a fit: at 0
        # the model gives 3.41 and 11.88 s against the published 3.5 and 11.9 s
        # without launch control.  A CALIBRATION.
        self.launch_rpm = 3000.0
        # Driveline efficiency.  A CALIBRATION with cd_a: 0.73 gives the 11.7 s
        # quarter mile (0.70 / 0.72 / 0.74 / 0.76 / 0.80 / 0.90 give 11.88 /
        # 11.75 / 11.63 / 11.52 / 11.50 / 11.16 s at cd_a 0.54).  Below
        # friction alone because it also stands in for rotating inertia; why
        # not model that: its per-gear figures are not published, while this
        # is pinned by a published time.  calibration/porsche_fit.py.
        self.driveline_eff = 0.73
        # Unloaded radius of the 305/30 ZR21 rear tyre: 21 x 25.4 / 2 + 0.30 x
        # 305 = 358 mm.  DERIVED.
        self.wheel_radius = 0.358
        # Overall ratios: the PDK's 4.89 ... 0.61 through the 3.39 axle and 0.92
        # rear constant (Porsche 2020), 4.89 x 3.39 x 0.92 = 15.25 and so on.
        # PUBLISHED.  Top speed comes in sixth at 6670 rpm, drag limited.
        self.gear_ratios = (15.25, 9.89, 6.71, 4.87, 3.68, 2.93, 2.37, 1.90)
        # 18.7 kN = 1.24 g, a 32 m stop from 100 km/h.  ESTIMATED (no 100-0
        # figure for the 992 found); just under the tyre limit (19.6 kN), so
        # grip binds, not the brakes.  physics_tests measures 32.7 m.
        self.max_brake_force = 18700.0

        # Peak drive force and power, unused by the engine: a planner's rough
        # capability figures.
        self.max_drive_force = (float(np.max(self._torque_nm))
                                * self.gear_ratios[0] * self.driveline_eff
                                / self.wheel_radius)
        self.max_power = 331000.0           # 331 kW (450 PS), 992 Carrera S

        self._throttle_state = 0.0
        # Pedal travel: 0 to full in 0.17 s, so a controller cannot swap full
        # brake and full throttle within one step.  ESTIMATED; brake the same.
        self.throttle_slew_rate = 6.0

        # Peak friction, a MODELLING CHOICE: 1.3 is typical of a warm
        # performance road tyre on dry asphalt and sets the 1.11 g cornering
        # (physics_tests); every grip number scales with it.
        self.tire_mu = 1.3

        # Staggered tyres as on the real car (245 front, 305 rear).  With the
        # cg 60% rearward and equal tyres the rear runs out of grip first and
        # the car spins under trail braking; the wider rear gives a positive
        # understeer gradient instead of an artificial yaw damper.  Grip is
        # proportional to width, normalised so overall grip stays tire_mu.
        # A steady corner needs Fy_f / Fy_r = lr / lf = 0.667, so the front
        # must saturate first (mu_f Fz_f / mu_r Fz_r < 0.667); equal tyres give
        # exactly 0.667 and the car held a 1.4 Hz yaw oscillation; these
        # widths give 0.536 and it settles.
        self.tire_width_front = 0.245
        self.tire_width_rear = 0.305
        width_mean = 0.5 * (self.tire_width_front + self.tire_width_rear)
        self.tire_mu_front = self.tire_mu * self.tire_width_front / width_mean
        self.tire_mu_rear = self.tire_mu * self.tire_width_rear / width_mean

        # Tyre relaxation length: lateral force lags a slip change by sigma/v,
        # the turn-in delay, and it damps the yaw transient.  ESTIMATED,
        # inside the measured range: Alcazar Vargas et al. (2022) measured
        # 0.3-0.9 m on a passenger tyre, their model 0.5-0.6 m at 10 m/s for
        # these loads.  It also freezes the slip at zero speed instead of a
        # singularity.
        self.relaxation_length = 0.5

        self.track_width = 1.55             # lateral transfer lever: 992 rear track (Porsche 2020)

        # No slip clamp or yaw damper: a tyre peaking where a real one does
        # (10 degrees) damps yaw itself; only a too-soft tyre would need them.

        self.g = 9.81
        self.normal_load = self.mass * self.g
        # Static axle loads from lf and lr, so the split cannot drift from them.
        self.load_front_static = self.normal_load * (self.lr / self.length)
        self.load_rear_static = self.normal_load * (self.lf / self.length)

        self.car_state = np.zeros(7, dtype=float)
        self._build_pacejka_lut()
        self.reset()

    def reset(self):
        """Reset state to the configured start pose."""
        self.position = self.start_position.copy()
        self.angle = self.start_angle
        self.velocity = np.zeros(2, dtype=float)
        self.yaw_rate = 0.0
        self.steering_angle = 0.0
        self._throttle_state = 0.0
        # Lagged slip angles, load-transfer accelerations and gearbox at rest.
        self._slip_angle_front = 0.0
        self._slip_angle_rear = 0.0
        self._ax_body = 0.0
        self._ay_body = 0.0
        self.gear = 0
        self.rpm = 0.0
        self.debug_forces = (self.load_front_static, self.load_rear_static,
                             0.0, 0.0, 0.0, 0.0, 0.0)
        return self.get_state()

    def step(self, action):
        """Advance one CONTROL step, integrating physics_substeps times with
        the action held (as TORCS separates its 0.002 s physics from 0.02 s
        control).  Against a 500 Hz reference on a chicane, one substep
        carries 7.9 m of RMS position error, ten carry 1.0 m; five is set on
        scoring against cost (__init__)."""
        n = max(1, int(self.physics_substeps))
        dt = self.time_step / n
        state = None
        for _ in range(n):
            state = self._integrate(action, dt)
        return state

    def _drive_force(self, v_forward):
        """Drive force at the rear wheels (N).  Engine speed follows road
        speed through the engaged gear (no wheel-spin state); the gearbox
        shifts with hysteresis; torque comes off the published curve, so
        acceleration falls with speed and steps at each shift."""
        omega_wheel = max(v_forward, 0.0) / self.wheel_radius
        top = len(self.gear_ratios) - 1

        def rpm_at(gear):
            return omega_wheel * self.gear_ratios[gear] * 60.0 / (2.0 * math.pi)

        rpm = rpm_at(self.gear)
        if self.gear == 0 and rpm < self.launch_rpm:
            rpm = self.launch_rpm       # clutch still slipping off the line
        while self.gear < top and rpm > self.shift_up_rpm:
            self.gear += 1
            rpm = rpm_at(self.gear)
        while self.gear > 0 and rpm < self.shift_down_rpm:
            self.gear -= 1
            rpm = rpm_at(self.gear)
        self.rpm = rpm
        if rpm >= self.limiter_rpm:
            return 0.0                      # on the limiter
        torque = float(np.interp(rpm, self._torque_rpm, self._torque_nm))
        return (torque * self.gear_ratios[self.gear] * self.driveline_eff
                / self.wheel_radius)

    def _slip_for_force(self, force_fraction, slip_angle, braking=False):
        """Longitudinal slip that delivers `force_fraction` of the tire's peak
        alongside an existing `slip_angle`.

        The pedal asks for FORCE: a real wheel spins up until its slip makes
        the engine's torque, and with no wheel state solving for that slip is
        the equivalent (a fixed slip ratio barely rotates the slip vector
        mid-corner and delivers a fraction of the request).

        Bisected up to the slip where the force peaks (_peak_sx), where it is
        monotone; over [0, 1] any demand above F(1) = 0.90 of peak reads as
        unreachable and locks the wheel, as it did under trail braking.

        A demand above the peak: under power the wheel spins (slip 1); under
        braking ABS (standard on the 992) caps slip at the straight-line peak
        (0.18), since ABS regulates slip, not force; holding the combined
        peak (0.38-1.5 with slip angle) would slide past lock.  A modelling
        choice.  Why not model the ABS cycle: its 4-20 Hz modulation is beyond
        a 10 Hz controller, and it holds the tyre around this slip.
        """
        lateral = math.sin(slip_angle)       # the lateral slip axle_forces uses

        def fx_at(sx):
            mag = math.hypot(sx, lateral)
            if mag < 1e-9:
                return 0.0
            return self._get_combined_force_coeff(min(mag, 1.5)) * sx / mag

        hi = self._lut_interp_uniform(self._peak_sx, abs(lateral), self._lut_s_min,
                                      self._lut_s_max, self._lut_s_inv_step)
        if braking:
            hi = min(hi, float(self._peak_sx[0]))
        if fx_at(hi) < force_fraction:
            return hi if braking else 1.0
        lo = 0.0
        for _ in range(12):                 # resolves slip to ~0.01 degrees
            mid = 0.5 * (lo + hi)
            if fx_at(mid) < force_fraction:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _integrate(self, action, time_step):
        """Advance the physics by `time_step` seconds using a control dict.

        Stages: read and rate-limit the inputs, compute per-axle slip and
        tire forces (Pacejka + friction circle), then integrate the body.
        """

        # ── Inputs: steering lock, pedal slew, steering rate limit ────────
        # A speed-independent lock: steering=1 is always 30 deg, as the
        # Simulated Car Racing interface fixes it to one angle (Loiacono,
        # Cardamone and Lanzi 2013, Table 3: "corresponds to an angle of
        # 0.366519 rad"), so the command means one thing at every speed.
        max_steering = float(self.max_steering)

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

        # ── Body-frame velocities and resistance forces ───────────────────
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)

        v_forward = cos_a * self.velocity[0] + sin_a * self.velocity[1]
        v_lateral = -sin_a * self.velocity[0] + cos_a * self.velocity[1]

        yaw_rate = float(self.yaw_rate)

        speed_abs = abs(v_forward)
        rolling_force = self.c_rr * self.mass * self.g
        drag_force = 0.5 * self.rho_air * self.cd_a * (speed_abs ** 2)
        resist_force = (rolling_force + drag_force) * (1.0 if v_forward >= 0.0 else -1.0)

        v = abs(v_forward)
        delta = self.steering_angle           # the tyres get the commanded angle unaltered

        # ── Load transfer ─────────────────────────────────────────────────
        # Longitudinal: dFz = m ax h_cg / L, with ax from the previous substep
        # (the usual explicit treatment; a simultaneous solve needs an inner
        # iteration for a correction 0.02 s late).  Loads floor at zero: a
        # tyre can be unloaded, not pulled into the road.
        downforce = 0.5 * self.rho_air * self.cl_a * v * v
        dFz = self.mass * float(self._ax_body) * self.h_cg / self.length
        Fz_f = self.load_front_static + downforce * self.aero_front_share - dFz
        Fz_r = self.load_rear_static + downforce * (1.0 - self.aero_front_share) + dFz
        if Fz_f < 0.0:
            Fz_f = 0.0
        if Fz_r < 0.0:
            Fz_r = 0.0

        # Lateral: with load sensitivity the outside tyre gains less than the
        # inside one loses, so cornering shrinks an axle's grip.  Cornering at
        # ay moves m ay h_cg / track of load outward (the total; roll centres
        # only split it); each axle is evaluated as two tyres through _grip,
        # so the transfer costs what TORCS' curve says.  Split between axles
        # by static load, a modelling choice (the real split follows roll
        # stiffness, not published), which keeps the stagger's balance.  Why
        # not a fixed grip-loss coefficient: a free number.  Why not four
        # wheels (TORCS): needs roll stiffnesses and per-wheel state for a
        # loss this reproduces per axle.
        transfer = self.mass * abs(float(self._ay_body)) * self.h_cg / self.track_width
        front_share = self.load_front_static / self.normal_load
        muFzf = self._axle_grip(Fz_f, transfer * front_share,
                                self.load_front_static, self.tire_mu_front)
        muFzr = self._axle_grip(Fz_r, transfer * (1.0 - front_share),
                                self.load_rear_static, self.tire_mu_rear)

        # ── Lateral tire forces from per-axle slip angles ─────────────────
        vxf = v_forward
        vyf = v_lateral + self.lf * yaw_rate
        vxr = v_forward
        vyr = v_lateral - self.lr * yaw_rate

        # Steady-state slip angles (atan2 needs no floor at zero speed).
        slip_ss_front = delta - math.atan2(vyf, vxf)
        slip_ss_rear = -math.atan2(vyr, vxr)

        # Tyre relaxation, time constant sigma/v, as the exact solution of the
        # first-order lag over a substep (stable at any step).  At a
        # standstill the factor is zero and the slip angles hold, keeping the
        # model finite at v = 0 without a clamp.
        relax = 1.0 - math.exp(-v * time_step / max(self.relaxation_length, 1e-9))
        self._slip_angle_front += (slip_ss_front - self._slip_angle_front) * relax
        self._slip_angle_rear += (slip_ss_rear - self._slip_angle_rear) * relax
        slip_angle_front = self._slip_angle_front
        slip_angle_rear = self._slip_angle_rear

        # ── Longitudinal slip demand ──────────────────────────────────────
        # The pedal demands a force; _slip_for_force solves for its slip.
        # TORCS derives slip from wheel spin, which needs per-wheel rotation;
        # without it drive force cannot persist after a lift and response is
        # the throttle slew alone, the one place this is simpler than TORCS on
        # purpose.  Rear-wheel drive; braking shared by the transferred loads.
        drive_available = self._drive_force(v_forward)   # also shifts gears
        if throttle_input >= 0.0:
            demand_f, demand_r = 0.0, throttle_input * drive_available
        elif v_forward <= 0.1:
            demand_f = demand_r = 0.0   # standing still: brakes hold
        else:
            total = throttle_input * self.max_brake_force   # negative
            # Split by GRIP, not load, so both axles reach their limit
            # together (by load the narrower front would lock first).
            share_f = muFzf / max(muFzf + muFzr, 1e-6)
            demand_f = total * share_f
            demand_r = total * (1.0 - share_f)

        # Turn each force demand into the slip that would deliver it.
        def slip_for(demand, mu_load, slip_angle):
            if mu_load <= 1e-6 or abs(demand) < 1e-9:
                return 0.0
            fraction = min(abs(demand) / mu_load, 1.0)
            return math.copysign(self._slip_for_force(fraction, abs(slip_angle), demand < 0.0),
                                 demand)

        kappa_f = slip_for(demand_f, muFzf, slip_angle_front)
        kappa_r = slip_for(demand_r, muFzr, slip_angle_rear)

        # ── Combined-slip tyre forces, one axle at a time ─────────────────
        # TORCS' formulation: the slip VECTOR (longitudinal slip, sin(slip
        # angle)), one magic formula on its magnitude, the force split back
        # along it, so the friction circle holds with no clamp.
        def axle_forces(slip_angle, slip_ratio, mu_load):
            lateral_slip = math.sin(slip_angle)
            magnitude = math.hypot(slip_ratio, lateral_slip)
            if magnitude < 1e-9:
                return 0.0, 0.0
            capped = min(magnitude, 1.5)
            force = self._get_combined_force_coeff(capped) * mu_load
            return force * slip_ratio / magnitude, force * lateral_slip / magnitude

        Fx0_f, Fy_f = axle_forces(slip_angle_front, kappa_f, muFzf)
        Fx0_r, Fy_r = axle_forces(slip_angle_rear, kappa_r, muFzr)

        # No ceiling here: torque curve and brakes bound the demand, the tyre
        # decides what reaches the road (an over-demand is a spinning or
        # locked wheel losing grip).
        # Front-axle forces from wheel frame into body frame.
        c = math.cos(delta)
        s = math.sin(delta)
        Fx_f = Fx0_f * c - Fy_f * s
        Fy_f_b = Fx0_f * s + Fy_f * c

        Fx_r = Fx0_r
        Fy_r_b = Fy_r

        Fx_body = Fx_f + Fx_r - resist_force
        Fy_body = Fy_f_b + Fy_r_b

        # ── Integrate the body (velocities, yaw, pose) ────────────────────
        # No Coriolis terms (v_x_dot = Fx/m + r v_y ...): those apply when the
        # body-frame velocity is the state.  Here velocity is stored in world
        # axes and projected in and out each substep, which already accounts
        # for the frame turning; with the terms the path curves at twice the
        # yaw rate and cornering halves (5.81 against 11.63 m/s^2).
        ax = Fx_body / self.mass
        ay = Fy_body / self.mass
        v_forward += ax * time_step
        v_lateral += ay * time_step

        mz = (self.lf * Fy_f_b) - (self.lr * Fy_r_b)
        yaw_rate += (mz / max(self.inertia_z, 1.0)) * time_step

        if v_forward < 0.0:                   # the only guard: no reversing; top speed is not clamped
            v_forward = 0.0
        self.yaw_rate = yaw_rate              # damped only by the tyres' slip angles

        # Body-frame accelerations (drag included) for the next substep's
        # load transfer.
        self._ax_body = Fx_body / self.mass
        self._ay_body = Fy_body / self.mass
        # Last substep's force breakdown, read by the physics harness.
        self.debug_forces = (Fz_f, Fz_r, Fx_f, Fx_r, Fy_f, Fy_r, Fy_body)

        self.velocity[0] = cos_a * v_forward - sin_a * v_lateral
        self.velocity[1] = sin_a * v_forward + cos_a * v_lateral

        self.angle += yaw_rate * time_step
        self.angle = (self.angle + math.pi) % (2.0 * math.pi) - math.pi

        self.position += self.velocity * time_step
        return self.get_state()

    def get_state(self):
        """The state [x, y, angle, forward speed, steering, lateral speed,
        yaw rate] in one buffer rewritten each call (copy to keep it).
        Element 3 is speed ALONG the car, what a speed profile plans; 5 and 6
        are what a learned driver reads beyond a path follower."""
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)
        vx = float(self.velocity[0])
        vy = float(self.velocity[1])
        self.car_state[0] = self.position[0]
        self.car_state[1] = self.position[1]
        self.car_state[2] = self.angle
        self.car_state[3] = cos_a * vx + sin_a * vy
        self.car_state[4] = self.steering_angle
        self.car_state[5] = -sin_a * vx + cos_a * vy
        self.car_state[6] = self.yaw_rate
        return self.car_state
