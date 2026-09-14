import math
import numpy as np

class CarPhysicsEngine:
    """2D single-track (bicycle) car physics with a Pacejka LUT.

    State returned by `get_state()` is `[x, y, angle, speed, steering_angle]`,
    where speed is measured along the car.  Control input to `step()` is a dict
    with normalized keys:
    - `steering` in [-1, 1]
    - `throttle` in [-1, 1] (positive drive, negative brake)

    Every force in the model comes from a physical mechanism: Pacejka tire
    curves on load-sensitive, transferred axle loads, a torque curve through an
    auto gearbox, brake hardware, aerodynamic drag and downforce, and tire
    relaxation lag.  There is no speed-scheduled steering assist, no artificial
    yaw or sideslip damping, no slip-angle clamp and no top-speed clamp; a yaw
    rate is resisted only by the slip angles it creates, the tire is free to
    run past its grip peak and break away, and top speed falls out of the rev
    limiter in top gear against drag.

    Combined slip follows TORCS: ONE magic formula evaluated on the magnitude
    of the slip vector, split back along its direction, so the friction circle
    holds by construction rather than being imposed by a clamp.

    Two deliberate simplifications remain, both documented where they occur.
    The pedal demands a FORCE and the slip that delivers it is solved for,
    rather than a wheel spinning up under torque, so wheelspin and lockup have
    their consequences but no dynamics of their own.  And the car has two axles
    rather than four wheels, so there is no differential and no true left/right
    load split; lateral transfer enters only as the grip it costs.
    """

    # Magic-formula shape, from TORCS' defaults for a tire (simuv2 wheel.cpp):
    # Ca 30, RFactor 0.8, EFactor 0.7, combined as
    #     C = 2 - asin(RFactor) * 2 / pi,   B = Ca / C,   E = EFactor.
    # The shape matters more than it looks.  The curve this replaces peaked at
    # 26.1 degrees of slip, so at the 2 degrees a car actually corners on it
    # produced 0.39 of its peak force where this one produces 0.74, and a car
    # on tires half as stiff as they should be understeers until it cannot
    # reach its own grip: measured sustained cornering was 5.4 m/s^2 against
    # the 12.8 that mu implies.  This curve peaks at 10.0 degrees, which is
    # where a real tire peaks.
    _MF_RFACTOR = 0.8
    _MF_EFACTOR = 0.7
    _MF_CA = 30.0
    # Load sensitivity, also TORCS': grip per newton falls as a tire is loaded,
    # so mu runs from lf_max at zero load down towards lf_min when heavily
    # loaded, passing through the nominal value at the operating load.
    #     mu(Fz) = mu * (lfMin + (lfMax - lfMin) * exp(lfK * Fz / opLoad))
    # This is what makes load transfer change the car's balance rather than
    # just move numbers around, and it is why the outside tire in a corner
    # cannot simply take over from the inside one.
    _LF_MIN = 0.8
    _LF_MAX = 1.6
    _OP_LOAD_FACTOR = 1.2        # operating load, as a multiple of static load

    def _build_pacejka_lut(self):
        """Precompute the combined-slip magic formula over a fixed grid.

        One curve, not two.  TORCS evaluates a single magic formula on the
        MAGNITUDE of the combined slip vector and then splits the result along
        that vector's direction, which keeps the total force inside the
        friction circle by construction instead of computing two forces
        independently and clamping them afterwards.
        """
        self._mf_C = 2.0 - math.asin(self._MF_RFACTOR) * 2.0 / math.pi
        self._mf_B = self._MF_CA / self._mf_C
        self._mf_E = self._MF_EFACTOR
        self._lf_k = math.log((1.0 - self._LF_MIN) / (self._LF_MAX - self._LF_MIN))

        # Slip magnitude is non-negative and TORCS caps it at 1.5, past which
        # the tire is fully sliding and the curve is flat anyway.
        slips = np.linspace(0.0, 1.5, 151)
        self._lut_s_min = float(slips[0])
        self._lut_s_max = float(slips[-1])
        self._lut_s_inv_step = 1.0 / float(slips[1] - slips[0])
        bx = self._mf_B * slips
        self.lut_combined = np.sin(
            self._mf_C * np.arctan(bx * (1.0 - self._mf_E) + self._mf_E * np.arctan(bx)))

    def _get_combined_force_coeff(self, slip_magnitude):
        """Normalized tire force for a combined slip magnitude."""
        return self._lut_interp_uniform(
            self.lut_combined, float(slip_magnitude),
            self._lut_s_min, self._lut_s_max, self._lut_s_inv_step,
        )

    def _grip(self, load, static_load, mu_nominal=None):
        """Peak tire force available at this vertical load, in newtons.

        Grip per newton falls as a tire is pressed harder, so doubling the
        load does not double the force.  That is what makes weight transfer a
        change in the car's BALANCE rather than a zero-sum move of a fixed
        grip budget between the axles.
        """
        # physics_tests.test_load_sensitivity calls this with two arguments to
        # measure the load-sensitivity curve on its own, so the nominal mu has
        # to default.
        if mu_nominal is None:
            mu_nominal = self.tire_mu
        op_load = max(self._OP_LOAD_FACTOR * static_load, 1e-6)
        mu = mu_nominal * (self._LF_MIN + (self._LF_MAX - self._LF_MIN)
                           * math.exp(self._lf_k * load / op_load))
        return mu * load

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
        # 30 deg road-wheel lock: passenger cars run 30-40 deg and performance
        # cars sit at the low end for high-speed stability.  With the 2.45 m
        # wheelbase this gives a 4.24 m minimum turn radius (11.2 m kerb-to-kerb
        # circle, the real 911 figure).  steering=1 maps to this angle at every
        # speed and the tires receive exactly it, so the command means one thing
        # throughout, as the Simulated Car Racing interface requires.
        max_steering=np.deg2rad(30.0),
        # Nominal top speed, kept only as a scale for observation
        # normalisation.  It does NOT clamp anything: the real top speed comes
        # out of the rev limiter in top gear against aerodynamic drag.
        max_speed=85.5,                     # nominal only, see below
        # 60 deg/s at the road wheels.  Through a sports-car steering ratio of
        # ~15:1 that is 900 deg/s at the steering wheel, already a violent
        # input (a sharp evasive input measures ~140 deg/s there).  Slow enough
        # that the limit actually binds: reaching full lock takes 5 control
        # steps, so the controller cannot slam lock-to-lock between them.
        steering_rate=np.deg2rad(60.0),
        length=2.45,                        # 992 wheelbase (m)
        # Integration substeps per control step: 5 gives 50 Hz physics under a
        # 10 Hz controller.  Chosen by measurement, not by taste.  Scoring is
        # flat across rates and cost is not: over 40 genomes on all five
        # representations, mean quality is 0.7613 / 0.7719 / 0.7720 / 0.7669
        # at 30 / 50 / 100 / 200 Hz, with 36 / 38 / 38 / 37 laps finished, for
        # 0.257 / 0.300 / 0.454 / 0.787 seconds per evaluation.  50 Hz is the
        # cheapest rate that scores the same as every rate above it; 30 Hz is
        # the first that does not.  TORCS runs 500 Hz because it renders to a
        # human, which is a different requirement from scoring a lap.
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

        # Car spec: Porsche 911 Carrera S (992) on track tires.  Figures below
        # are the published car, not round numbers: 1555 kg kerb (manual),
        # 2.45 m wheelbase, 40/60 front/rear (the real car is ~39/61), 308 km/h.
        # With the 30 deg lock the bicycle-model radius is 4.24 m, an 11.2 m
        # kerb-to-kerb circle over a 1.55 m track, which is the published
        # turning circle.
        self.mass = 1555.0
        self.lf = self.length * 0.60        # cg sits nearer the rear axle
        self.lr = self.length * 0.40
        # Yaw inertia from the dynamic-index-1 relation Izz = m * lf * lr,
        # giving ~2240 kg m^2, inside the 1700-2500 range measured for sports
        # cars.
        self.inertia_z = self.mass * self.lf * self.lr
        # Centre-of-gravity height, the lever arm that turns longitudinal
        # acceleration into load transfer between the axles.  ESTIMATED, not
        # published: manufacturers do not quote cg height, and 0.45-0.50 m is
        # the usual range for a low sports car.  Load transfer scales linearly
        # with it, so a 10% error here is a 10% error in how much the car
        # pitches its grip about under braking and power.
        self.h_cg = 0.47
        self.c_rr = 0.015
        self.rho_air = 1.225
        self.cd_a = 0.60                    # Cd 0.29 x frontal area ~2.07 m^2
        # Downforce, which adds grip with v^2.  ESTIMATED: lift and downforce
        # coefficients are not published for road cars, and a 911 without a
        # wing makes little either way, so this is deliberately small next to
        # cd_a.  It only matters near the top of the speed range, which the
        # benchmark never reaches (the agent plans to 25 m/s), so it is close
        # to inert here and is present for completeness rather than effect.
        # Split slightly rearward to match where the car carries its weight,
        # so the aero does not shift the balance with speed.
        self.cl_a = 0.30
        self.aero_front_share = 0.40

        # ── Drivetrain: torque curve through an auto gearbox ──────────────
        # The published engine, not a constant force: 530 N.m from 2300 to
        # 5000 rpm and 331 kW (450 PS) at 6500.  Torque reaches the road
        # through whichever gear is engaged, so acceleration falls away with
        # speed and steps at each shift, the way a real car does.
        self._torque_rpm = np.array([0.0, 1500.0, 2300.0, 5000.0, 6500.0, 7500.0])
        self._torque_nm  = np.array([300.0, 450.0, 530.0, 530.0, 486.0, 400.0])
        self.limiter_rpm  = 7500.0
        self.shift_up_rpm = 6800.0
        self.shift_down_rpm = 3200.0
        # Engine speed the clutch holds while it is still slipping, which is
        # how a car launches: below this the wheels are turning too slowly to
        # spin the engine, and without it the model would launch on idle
        # torque and take 4.1 s to 100 km/h instead of the published 3.5.
        self.launch_rpm = 3000.0
        self.driveline_eff = 0.90
        self.wheel_radius = 0.358           # 305/30 R21 rear, rolling radius
        # Overall ratios (gearbox x final drive).  These are the first six of
        # the 992's eight-speed PDK (whose overall ratios run 18.9, 11.4, 7.5,
        # 5.7, 4.4, 3.4, 2.8, 2.2), shortened about 3% so top gear reaches the
        # published 308 km/h exactly at the limiter.  The seventh and eighth
        # are overdrive ratios for economy and are above the car's top speed,
        # so dropping them changes nothing that is simulated here.  First is short
        # on purpose: a 450 hp rear-drive car is TRACTION limited off the
        # line, not force limited, so a tall first would quietly make the car
        # unable to spin its rear wheels at all.
        self.gear_ratios = (18.40, 11.00, 7.40, 5.54, 4.40, 3.29)
        # 100-0 km/h in ~32 m = 1.23 g = 18.7 kN.  Sits just under the tire
        # limit (mu 1.3 gives 19.8 kN), so grip and not the brakes is the
        # binding constraint, as on the real car.
        self.max_brake_force = 18700.0

        # Peak drive force (first gear, peak torque) and peak power.  The
        # engine does not use these: they exist so a planner can ask what the
        # car is roughly capable of without simulating the gearbox.
        self.max_drive_force = (float(np.max(self._torque_nm))
                                * self.gear_ratios[0] * self.driveline_eff
                                / self.wheel_radius)
        self.max_power = 331000.0           # 331 kW (450 PS), 992 Carrera S

        self._throttle_state = 0.0
        self.throttle_slew_rate = 6.0

        # Peak friction coefficient.  A MODELLING CHOICE, not a measurement:
        # 1.3 is typical of a warm performance road tire on dry asphalt, and it
        # is what sets the car's 1.17 g cornering.  Every grip number in the
        # model scales with it.
        self.tire_mu = 1.3

        # Staggered tires, front narrower than rear, as fitted to the real car
        # (245/35 R20 front, 305/30 R21 rear).  This is not a detail: with the
        # cg 60% rearward and the same tire at both ends, the rear axle runs
        # out of grip before the front and the car oversteers into a spin the
        # moment it is asked to brake and turn together.  A wider rear tire is
        # how the real 911 is made stable, and it is what gives this model a
        # positive understeer gradient instead of an artificial yaw damper.
        #
        # Grip is taken proportional to contact width, normalised about the
        # mean so the car's overall grip stays at tire_mu and only the balance
        # between the axles moves.
        #
        # The size of the stagger decides whether the car is stable at all.
        # Equilibrium in a steady corner needs Fy_f / Fy_r = lr / lf = 0.667,
        # so the front must run out of grip first, which needs
        # mu_f*Fz_f / mu_r*Fz_r below 0.667.  The static load split alone gives
        # exactly 0.667, i.e. neutral with no margin, and the car then rings at
        # its limit instead of settling: measured, it held a 1.4 Hz yaw
        # oscillation indefinitely and never converged.  These widths give
        # 0.536 and the car settles.
        self.tire_width_front = 0.245
        self.tire_width_rear = 0.305
        width_mean = 0.5 * (self.tire_width_front + self.tire_width_rear)
        self.tire_mu_front = self.tire_mu * self.tire_width_front / width_mean
        self.tire_mu_rear = self.tire_mu * self.tire_width_rear / width_mean

        # Tire relaxation length: the distance the wheel rolls before lateral
        # force reaches 63% of its steady-state value.  Lateral force therefore
        # lags a slip-angle change by sigma/v seconds, which is the physical
        # mechanism behind the delay a driver feels on turn-in, and it is what
        # damps the yaw transient.  0.5 m sits in the usual passenger-car range.
        #
        # It also removes the low-speed singularity without a guard: the lag
        # constant sigma/v grows without bound as the car slows, so the slip
        # angle freezes instead of diverging when forward speed reaches zero.
        self.relaxation_length = 0.5

        # Axle track, the lever lateral load transfer acts over.
        self.track_width = 1.55             # 992 rear track (m)

        # No slip-angle clamp and no artificial yaw damper.  Both were needed
        # only while the tire was too soft: on the previous curve, which peaked
        # at 26 degrees of slip, the car could not reach its own grip and had
        # to be held straight by hand.  A tire that peaks where a real one does
        # generates its restoring moment early enough that the tires damp yaw
        # themselves, which is what they do on a real car.

        self.g = 9.81
        self.normal_load = self.mass * self.g
        # Static axle loads follow from where the cg sits, so the 40/60 split
        # is a consequence of lf and lr rather than a second, independent
        # figure that could drift out of step with them.
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
        # Lagged slip angles (tire relaxation), the accelerations that drive
        # load transfer, and the gearbox, all at rest.
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
        """Advance one CONTROL step, integrating the physics in substeps.

        The control rate and the integration rate are separate concerns, which
        is how TORCS is arranged too: it integrates at 0.002 s while the
        controller acts every 0.02 s.  Here the agent still acts every
        time_step, but the body is integrated physics_substeps times at
        time_step/physics_substeps with the action held constant.

        The substep count is set by convergence, not by taste.  Measured
        against a 500 Hz reference on the same chicane: one substep (10 Hz)
        carries 7.9 m of RMS position error on a road whose half-width is 8 m,
        a full track width of error from the timestep alone, while 10 substeps
        (100 Hz) carry 1.0 m for roughly twice the physics cost.
        """
        n = max(1, int(self.physics_substeps))
        dt = self.time_step / n
        state = None
        for _ in range(n):
            state = self._integrate(action, dt)
        return state

    def _drive_force(self, v_forward):
        """Drive force available at the rear wheels right now, in newtons.

        Engine speed follows road speed through the engaged gear, since there
        is no wheel-spin state, and the gearbox shifts on its own with
        hysteresis so it cannot hunt between two ratios.  Torque comes off the
        published curve, which is why acceleration falls away with speed and
        steps at every shift instead of being one flat number.
        """
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

    def _slip_for_force(self, force_fraction, slip_angle):
        """Longitudinal slip that delivers `force_fraction` of the tire's peak
        alongside an existing `slip_angle`.

        The pedal asks for FORCE, not slip.  On a real car the wheel spins up
        until its slip produces the torque the engine is making; with no wheel
        state, solving for that slip is the equivalent step, and skipping it
        quietly loses the request whenever the car is cornering.  Adding a
        small longitudinal slip to a large slip angle barely rotates the
        combined slip vector, so a straight fixed slip ratio delivers only a
        fraction of what was asked for mid-corner.

        Bisected because the combined force is monotone in slip up to
        saturation.  If even the largest slip searched cannot meet the demand,
        the tire is over-driven and that slip is returned, which is a spinning
        or locked wheel.
        """
        def fx_at(sx):
            mag = math.hypot(sx, slip_angle)
            if mag < 1e-9:
                return 0.0
            return self._get_combined_force_coeff(min(mag, 1.5)) * sx / mag

        hi = 1.0
        if fx_at(hi) < force_fraction:
            return hi
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
        # The lock clamped here is speed-independent, so steering=1 always maps
        # to the same 30 deg command.  The Simulated Car Racing Championship
        # interface (Loiacono et al., arXiv:1304.1672) fixes steering=1 to one
        # constant angle for the same reason: the command has to mean one thing.
        # The tires receive exactly this angle, so the agent's division by the
        # same figure is exact at every speed.
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
        # The tires receive the commanded angle unaltered.
        delta = self.steering_angle

        # ── Load transfer ─────────────────────────────────────────────────
        # Accelerating pitches load onto the rear axle and braking onto the
        # front, by dFz = m * ax * h_cg / L about the cg.  ax is taken from the
        # previous substep, which is the usual explicit treatment: solving it
        # simultaneously with the tire forces it feeds would need an inner
        # iteration for a correction that is already small at 100 Hz.
        #
        # Loads are floored at zero because a tire can be unloaded but cannot
        # be pulled into the road; that is the physical limit of the model, not
        # a tuning guard.
        downforce = 0.5 * self.rho_air * self.cl_a * v * v
        dFz = self.mass * float(self._ax_body) * self.h_cg / self.length
        Fz_f = self.load_front_static + downforce * self.aero_front_share - dFz
        Fz_r = self.load_rear_static + downforce * (1.0 - self.aero_front_share) + dFz
        if Fz_f < 0.0:
            Fz_f = 0.0
        if Fz_r < 0.0:
            Fz_r = 0.0

        # Grip per newton falls as a tire is loaded, so the axle that load
        # transfer pushes down does not gain grip in proportion.  This is what
        # makes braking into a corner shift the balance toward the front and
        # power-on shift it toward the rear, rather than leaving the car's
        # behaviour the same everywhere.

        # Lateral load transfer.  A two-axle model has no separate left and
        # right tire to move load between, but the cost of moving it is real
        # and follows from the same load sensitivity: the outside tire gains
        # less grip than the inside one loses, so hard cornering shrinks an
        # axle's total grip.  Modelled as that net loss, from how much lateral
        # acceleration the car is pulling against how far it can lean on its
        # track width before a wheel lifts.
        lateral_shift = min(abs(self._ay_body) * self.h_cg
                            / (self.g * self.track_width), 1.0)
        # The 0.25 is a CHOSEN coefficient, not a measured one: it sets how
        # much of the load-sensitivity swing a full lateral transfer costs.
        # The shape (quadratic in the transfer) follows from grip falling off
        # with load, but the magnitude is a modelling choice, and it is the
        # least defensible number in this file.
        lateral_grip_loss = 1.0 - (self._LF_MAX - self._LF_MIN) * 0.25 * lateral_shift ** 2
        muFzf = self._grip(Fz_f, self.load_front_static, self.tire_mu_front) * lateral_grip_loss
        muFzr = self._grip(Fz_r, self.load_rear_static, self.tire_mu_rear) * lateral_grip_loss

        # ── Lateral tire forces from per-axle slip angles ─────────────────
        vxf = v_forward
        vyf = v_lateral + self.lf * yaw_rate
        vxr = v_forward
        vyr = v_lateral - self.lr * yaw_rate

        # Steady-state slip angles in each axle frame.  atan2 is defined at
        # zero forward speed, so no floor on the divisor is needed.
        slip_ss_front = delta - math.atan2(vyf, vxf)
        slip_ss_rear = -math.atan2(vyr, vxr)

        # Tire relaxation: lateral force cannot appear instantly, it builds as
        # the tire rolls, with time constant sigma/v.  The exponential form is
        # the exact solution of the first-order lag over one substep, so it is
        # stable for any timestep instead of only for dt < sigma/v.
        #
        # At a standstill the factor is zero and the slip angles hold their
        # last value, which is what keeps the model finite at v = 0 without a
        # clamp: the ill-conditioned steady-state value is computed but never
        # blended in.
        relax = 1.0 - math.exp(-v * time_step / max(self.relaxation_length, 1e-9))
        self._slip_angle_front += (slip_ss_front - self._slip_angle_front) * relax
        self._slip_angle_rear += (slip_ss_rear - self._slip_angle_rear) * relax
        slip_angle_front = self._slip_angle_front
        slip_angle_rear = self._slip_angle_rear

        # ── Longitudinal slip demand ──────────────────────────────────────
        # The pedal demands a slip ratio directly.  TORCS derives it from wheel
        # spin, sx = (v_tangent - omega * r) / |v_tangent|, which needs a
        # rotational state per wheel; without one the pedal stands in for it,
        # so drive force cannot persist after a lift and response time is set
        # by the throttle slew rate alone.  This is the one place the model
        # stays simpler than TORCS on purpose.
        # Rear-wheel drive: traction through the rear axle only, braking
        # shared by the TRANSFERRED loads so the front takes a larger share the
        # harder the car brakes, as it does on the road.
        drive_available = self._drive_force(v_forward)   # also shifts gears
        if throttle_input >= 0.0:
            demand_f, demand_r = 0.0, throttle_input * drive_available
        elif v_forward <= 0.1:
            demand_f = demand_r = 0.0   # standing still: brakes hold
        else:
            total = throttle_input * self.max_brake_force   # negative
            # Split by each axle's GRIP, not its load.  A brake system is
            # proportioned so both axles reach their limit together, and with
            # a narrower tire at the front, splitting by load alone would lock
            # the front first and lengthen the stop.
            share_f = muFzf / max(muFzf + muFzr, 1e-6)
            demand_f = total * share_f
            demand_r = total * (1.0 - share_f)

        # Turn each force demand into the slip that would deliver it.
        def slip_for(demand, mu_load, slip_angle):
            if mu_load <= 1e-6 or abs(demand) < 1e-9:
                return 0.0
            fraction = min(abs(demand) / mu_load, 1.0)
            return math.copysign(self._slip_for_force(fraction, abs(slip_angle)), demand)

        kappa_f = slip_for(demand_f, muFzf, slip_angle_front)
        kappa_r = slip_for(demand_r, muFzr, slip_angle_rear)

        # ── Combined-slip tire forces, one axle at a time ─────────────────
        # TORCS' formulation: build the slip VECTOR from longitudinal slip and
        # sin(slip angle), run one magic formula on its magnitude, and split
        # the resulting force back along the vector.  The friction circle then
        # holds by construction, so there is no clamp to apply afterwards and
        # no choice to make about which of the two components to sacrifice.
        # Asking for more of one automatically leaves less of the other.
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

        # No drivetrain ceiling applied here any more: the torque curve and
        # the brake hardware already bounded the DEMAND above, and the tire
        # decides how much of it reaches the road.  An over-demand shows up as
        # a spinning or locked wheel losing grip, which is what it is, rather
        # than as a number being clipped.

        # Rotate front-axle tire forces from wheel frame into body frame.
        c = math.cos(delta)
        s = math.sin(delta)
        Fx_f = Fx0_f * c - Fy_f * s
        Fy_f_b = Fx0_f * s + Fy_f * c

        Fx_r = Fx0_r
        Fy_r_b = Fy_r

        Fx_body = Fx_f + Fx_r - resist_force
        Fy_body = Fy_f_b + Fy_r_b

        # ── Integrate the body (velocities, yaw, pose) ────────────────────
        # No Coriolis terms here, deliberately.  The body-frame equations
        #     v_x_dot = Fx/m + r*v_y,   v_y_dot = Fy/m - r*v_x
        # apply when the VELOCITY ITSELF is the body-frame state being carried
        # forward.  Here it is not: self.velocity is stored in world axes, and
        # every substep projects it into the body frame, integrates, and
        # projects it back through the heading of that same substep.  That
        # round trip already accounts for the frame turning, so adding the
        # terms as well rotates the velocity vector twice per step.
        #
        # The effect was not subtle.  The car's path curved at twice its yaw
        # rate, sideslip grew until the tires balanced the surplus, and its
        # sustained cornering came out at exactly half of what the tire forces
        # supported: 5.81 m/s^2 measured against 11.63 m/s^2 of tire force.
        ax = Fx_body / self.mass
        ay = Fy_body / self.mass
        v_forward += ax * time_step
        v_lateral += ay * time_step

        mz = (self.lf * Fy_f_b) - (self.lr * Fy_r_b)
        yaw_rate += (mz / max(self.inertia_z, 1.0)) * time_step

        # Only the reverse guard remains.  Top speed is not clamped: the rev
        # limiter in top gear and aerodynamic drag settle it between them,
        # which is what sets a real car's top speed.
        if v_forward < 0.0:
            v_forward = 0.0

        # Nothing damps sideways velocity or yaw artificially.  Both are
        # resisted only by the tires: a yaw rate builds slip angles at both
        # axles, those slip angles generate the restoring moment, and the
        # relaxation lag sets how quickly it arrives.
        self.yaw_rate = yaw_rate

        # Longitudinal acceleration in the body frame, kept for the next
        # substep's load transfer.  Fx_body already carries drag and rolling
        # resistance, so this is what the chassis actually feels.
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
        """Return the current state as a preallocated ndarray.

        Callers that keep the value across steps must copy it: this is one
        buffer, rewritten in place on every call.

        Element 3 is speed ALONG THE CAR, not the magnitude of the velocity
        vector.  The two differ while the car is sliding, and the agent's speed
        profile plans forward progress, so forward speed is the quantity it
        needs.

        Elements 5 and 6 are lateral speed and yaw rate, the two body-frame
        quantities a learned driver reads that a path follower does not.
        Appending them keeps indices 0-4 at their existing meaning, so every
        caller that slices the first five stays correct.
        """
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
