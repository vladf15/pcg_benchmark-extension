import math

import numpy as np


class SteeringAgent:
    """Path-following agent: pure-pursuit steering with a cross-track trim,
    and a speed target taken from a precomputed speed profile.

    The speed profile is the standard racing-line approach, computed once per
    track in three steps:

      1. Corner limit: at every path point the lateral grip bound gives a
         maximum cornering speed  v = sqrt(ay_max / curvature).
      2. Backward pass: driving toward a corner, speed may exceed the corner
         limit only by what braking can shed over the remaining distance
         (v^2 <= v_next^2 + 2 * a_brake * ds), so the car brakes *before*
         corners instead of reacting inside them.
      3. Forward pass: speed may rise out of a corner only as fast as the car
         accelerates (v^2 <= v_prev^2 + 2 * a_accel * ds), so corner exits ramp
         up smoothly instead of snapping to full throttle.

    What makes the profile match the car rather than a set of hand-picked
    constants is that a_brake and a_accel are not constants: grip is one
    circle, so the share left for accelerating or braking depends on how much
    the corner is already using, and both are also limited by the drivetrain,
    by engine power at speed, and by drag.  See _available_ax.

    act() then steers at a lookahead point and tracks the profile, which keeps
    the per-step work small and the behavior explainable.  Steering output is
    normalized to [-1, 1] for the engine.
    """

    def __init__(self, curve_points, track_width=12.0, max_speed=25.0, engine=None):
        """Create a steering agent for the given path.

        `engine` is an optional CarPhysicsEngine whose mass, grip and
        drivetrain limits the speed profile is built from, so the plan is made
        for the car that will actually drive it.  Without one the defaults
        below describe the same 992 Carrera S.
        """
        self.track_width = float(track_width)
        self.max_speed = float(max_speed)

        self.last_lookahead_point = None

        # ── Car parameters, read from the engine when one is supplied ──
        g = 9.81
        self.wheelbase       = float(getattr(engine, 'length', 2.45))
        mass                 = float(getattr(engine, 'mass', 1555.0))
        tire_mu              = float(getattr(engine, 'tire_mu', 1.3))
        tire_mu_rear         = float(getattr(engine, 'tire_mu_rear', tire_mu))
        rear_load_share      = float(getattr(engine, 'lf', 1.47)) /                                float(getattr(engine, 'length', 2.45))
        max_drive_force      = float(getattr(engine, 'max_drive_force', 12700.0))
        max_brake_force      = float(getattr(engine, 'max_brake_force', 18700.0))
        self._max_power      = float(getattr(engine, 'max_power', 331000.0))
        self._mass           = mass
        # Resistance the car always carries, subtracted from the acceleration
        # it can plan on and added to what braking achieves.
        self._drag_k         = 0.5 * float(getattr(engine, 'rho_air', 1.225)) \
                                   * float(getattr(engine, 'cd_a', 0.60)) / mass
        self._roll_a         = float(getattr(engine, 'c_rr', 0.015)) * g

        # Fraction of the limit the plan aims for.  Not a fudge: a plan that
        # asks for 100% of grip leaves the controller nothing to correct with,
        # and racing bots (TORCS, Speed Dreams) plan corner speed at 80-90% for
        # exactly that reason.  One factor for all three axes, so the margin is
        # the same whichever direction the car is loaded in.
        self.grip_utilisation = 0.9

        # Tire limits.  Cornering and braking use every tire, so their ceiling
        # is the whole car's grip.  ACCELERATING does not: this is a rear-drive
        # car, and only the rear axle can push, so its ceiling is that axle's
        # share of the weight against its own tire.  Planning acceleration
        # against the whole car's grip asks for around 11.5 m/s^2 where the
        # rear can deliver about 7.6, and the car simply spins its wheels on
        # every corner exit instead of following the plan.
        self.ay_max          = self.grip_utilisation * tire_mu * g
        self.ax_max_brake_tires = self.ay_max
        self.ax_max_accel_tires = self.grip_utilisation * tire_mu_rear * rear_load_share * g
        # Machine limits, separate from the tires exactly as TUM's
        # trajectory_planning_helpers separates ggv (tire) from ax_max_machines
        # (drivetrain): the tires say how much force the road will take, the
        # drivetrain says how much the car can produce.
        self.ax_max_drive  = self.grip_utilisation * max_drive_force / mass
        self.ax_max_brake  = self.grip_utilisation * max_brake_force / mass
        # Friction-ellipse exponent.  1 is a diamond (linear trade), 2 a true
        # ellipse; TUM's calc_vel_profile exposes the same parameter over the
        # same range and 1.4 sits where measured tire data usually falls.
        self.friction_exponent = 1.4
        self.min_speed     = 4.0    # never plan slower than this

        # Steering law.  Lookahead grows with speed so the aim point stays
        # roughly a fixed time ahead of the car.
        self.lookahead_base = 8.0    # metres at standstill
        self.lookahead_gain = 0.35   # extra metres per m/s of speed
        self.lookahead_max  = 30.0
        # Cross-track gain, tuned on generated tracks rather than on circles.
        # A sweep over constant-radius circles prefers 1.6, which more than
        # halves the tracking error there, and it is much worse where it
        # matters: on real content it drops the two spline representations
        # from 6 of 8 laps finished to 1 and 0.  Circles have no corner tight
        # enough to punish an aggressive correction, and generated tracks are
        # full of them, so the circle result does not transfer.
        self.stanley_k      = 0.35
        self.stanley_v0     = 1.5
        self.yaw_damp_gain  = 0.05   # seconds; rate feedback, see act()
        # Divisor that turns a wanted wheel angle into the [-1, 1] command.  It
        # equals the engine's steering lock, and the engine now applies that
        # angle unaltered, so the command is exact at every speed.
        self.nominal_max_steer_rad = float(getattr(engine, 'max_steering',
                                                   math.radians(30.0)))
        # Below this heading error (radians) on a near-centered car, hold
        # straight instead of chasing tiny errors (deadzone stops twitching).
        self.steer_deadzone_rad = math.radians(0.6)

        # Slow down when the car drifts outside the usable corridor
        # (half width minus a margin).
        self.corridor_margin = max(1.0, 0.18 * self.track_width)

        # Speed tracking: full throttle / full brake at these speed errors
        # (m/s).  Braking is stiffer than accelerating so the car does not
        # lag behind a falling profile and enter corners too fast.
        #
        # The throttle side is deliberately gentle.  First gear is short
        # enough to overwhelm the rear tires, so flooring the pedal out of a
        # slow corner spins them; a driver squeezes instead.  Swept over all
        # five representations, laps finished run 34, 35, 35, 38, 37 at a
        # deficit of 5, 8, 12, 18, 25 m/s, and off-road time falls from 0.104
        # to 0.029 across the same range.
        self.accel_deficit_full = 18.0
        self.brake_excess_full  = 2.5
        self.react_time         = 0.3

        # Projection window (path indices): search a window around the last
        # known segment so progress stays monotonic.
        self.search_ahead = 55
        self.search_back = 15
        self.max_index_advance = 12

        self.current_idx = 0
        self._prev_angle = None      # for the yaw-rate estimate in act()
        self._dt = 0.1               # engine control interval
        self.curve_points = curve_points  # setter precomputes everything

    # ── Path geometry and speed profile (once per track) ─────────────────

    def _precompute_path_geometry(self):
        """Precompute segment geometry and the speed profile for the path."""
        pts = np.asarray(self.path_points, dtype=float)
        n = len(pts)
        if n < 2:
            self._seg_a = np.zeros((0, 2))
            self._seg_ab = np.zeros((0, 2))
            self._seg_ab2 = np.zeros((0,))
            self._seg_len = np.zeros((0,))
            self._seg_norm = np.zeros((0, 2))
            self._seg_s0 = np.zeros((0,))
            self._cum_s = np.zeros((1,))
            self._speed_profile = np.zeros((0,))
            self._path_curvature = np.zeros((0,))
            return

        seg = pts[1:] - pts[:-1]
        seg_len = np.linalg.norm(seg, axis=1)
        seg_tan = np.zeros_like(seg)
        nonzero = seg_len > 1e-9
        # Unit tangent per segment: divide each (x, y) by the segment length.
        # reshape(-1, 1) turns the lengths into a column so numpy divides the
        # x and y of each row by that row's length.
        seg_tan[nonzero] = seg[nonzero] / seg_len[nonzero].reshape(-1, 1)

        self._seg_a = pts[:-1]
        self._seg_ab = seg
        # Squared length of each segment (x*x + y*y per row).
        self._seg_ab2 = seg[:, 0] * seg[:, 0] + seg[:, 1] * seg[:, 1]
        self._seg_len = seg_len
        self._seg_norm = np.stack([-seg_tan[:, 1], seg_tan[:, 0]], axis=1)

        # Arc length at the start of each segment / at each vertex.
        self._seg_s0 = np.concatenate(([0.0], np.cumsum(seg_len[:-1])))
        self._cum_s = np.concatenate(([0.0], np.cumsum(seg_len)))

        self._closed = bool(n >= 4 and np.allclose(pts[0], pts[-1], atol=1e-6))
        self._speed_profile = self._compute_speed_profile(pts, seg_tan, seg_len)

    def _available_ax(self, speed, curvature, braking):
        """Longitudinal acceleration still available at `speed` on a corner of
        `curvature`, in m/s^2 and always positive.

        Lateral and longitudinal grip come out of one friction circle, so
        whatever the corner is already using is not available for accelerating
        or braking.  The trade follows the friction ellipse

            ax_avail = ax_max * (1 - (ay_used / ay_max)^n)^(1/n)

        which is the form TUM's calc_vel_profile uses, with n = 1 a linear
        trade and n = 2 a true ellipse.  Whichever of the tires and the
        drivetrain binds first wins, and drag and rolling resistance are then
        applied with their real sign: they fight acceleration and help braking.
        """
        ay_used = speed * speed * abs(curvature)
        ratio = ay_used / self.ay_max
        if ratio >= 1.0:
            ax_tires = 0.0
        else:
            n = self.friction_exponent
            ax_ceiling = (self.ax_max_brake_tires if braking
                          else self.ax_max_accel_tires)
            ax_tires = ax_ceiling * math.pow(1.0 - math.pow(ratio, n), 1.0 / n)

        if braking:
            ax_machine = self.ax_max_brake
        else:
            ax_machine = self.ax_max_drive
            if speed > 1e-3:
                # Above the crossover the engine cannot deliver peak force any
                # more and power sets the ceiling.
                ax_power = self.grip_utilisation * self._max_power / (self._mass * speed)
                if ax_power < ax_machine:
                    ax_machine = ax_power

        ax = min(ax_tires, ax_machine)
        ax_resist = self._drag_k * speed * speed + self._roll_a
        ax = ax + ax_resist if braking else ax - ax_resist
        return max(ax, 0.0)

    def _compute_speed_profile(self, pts, seg_tan, seg_len):
        """Per-vertex speed limits: corner limit, then brake and accel passes,
        with longitudinal capability reduced by the lateral demand at each
        point (see _available_ax)."""
        n = len(pts)
        m = len(seg_len)  # = n - 1 segments

        # Curvature at each interior vertex: turn angle between the two
        # adjacent segments divided by the local arc length.  Kept, because the
        # passes below need to know how loaded the tires are at every point.
        curvature = np.zeros(n, dtype=float)
        for i in range(1, m):
            t0, t1 = seg_tan[i - 1], seg_tan[i]
            dot = float(np.clip(np.dot(t0, t1), -1.0, 1.0))
            angle = math.acos(dot)
            ds = 0.5 * float(seg_len[i - 1] + seg_len[i])
            if ds > 1e-9 and angle > 1e-9:
                curvature[i] = angle / ds
        if self._closed and m >= 2:
            # The seam vertex (0 == n-1) also has a turn angle.
            t0, t1 = seg_tan[-1], seg_tan[0]
            dot = float(np.clip(np.dot(t0, t1), -1.0, 1.0))
            angle = math.acos(dot)
            ds = 0.5 * float(seg_len[-1] + seg_len[0])
            if ds > 1e-9 and angle > 1e-9:
                curvature[0] = angle / ds
                curvature[-1] = curvature[0]

        v_corner = np.full(n, self.max_speed, dtype=float)
        nonzero = curvature > 1e-12
        v_corner[nonzero] = np.sqrt(self.ay_max / curvature[nonzero])
        profile = np.clip(v_corner, self.min_speed, self.max_speed)

        # Backward pass: entering point i at profile[i] must allow braking down
        # to profile[i+1] over segment i, at the deceleration actually left
        # over once point i+1's corner has taken its share of the grip.  For
        # closed loops run the pass twice so the constraint crosses the seam.
        rounds = 2 if self._closed else 1
        for _ in range(rounds):
            for i in range(m - 1, -1, -1):
                a = self._available_ax(profile[i + 1], curvature[i + 1], braking=True)
                v_allowed = math.sqrt(profile[i + 1] ** 2 + 2.0 * a * float(seg_len[i]))
                if profile[i] > v_allowed:
                    profile[i] = v_allowed
            if self._closed:
                profile[-1] = profile[0] = min(profile[0], profile[-1])

        # Forward pass: speed builds only as fast as the grip left over at
        # point i allows, so a corner exit ramps up as the wheel unwinds.
        for _ in range(rounds):
            for i in range(m):
                a = self._available_ax(profile[i], curvature[i], braking=False)
                v_reachable = math.sqrt(profile[i] ** 2 + 2.0 * a * float(seg_len[i]))
                if profile[i + 1] > v_reachable:
                    profile[i + 1] = v_reachable
            if self._closed:
                profile[-1] = profile[0] = min(profile[0], profile[-1])

        self._path_curvature = curvature
        return profile

    def reset(self):
        """Reset agent to start of path."""
        self.current_idx = 0
        self.last_lookahead_point = None
        self._prev_angle = None

    # ── Path queries ──────────────────────────────────────────────────────

    def _find_projection(self, point, start_idx):
        """Project `point` onto a window of the polyline around `start_idx`.

        Returns `(seg_idx, proj_point)` for the closest segment in the window.
        """
        nseg = len(self.path_points) - 1
        if nseg <= 0:
            return 0, np.asarray(point, dtype=float)

        i0 = int(max(0, min(start_idx - self.search_back, nseg - 1)))
        i1 = int(min(nseg - 1, max(i0, start_idx) + self.search_ahead))
        sl = slice(i0, i1 + 1)

        p = np.asarray(point, dtype=float)
        a = self._seg_a[sl]
        ab = self._seg_ab[sl]
        ab2 = self._seg_ab2[sl]

        # Fraction t of the way along each segment where the point projects,
        # clamped to [0, 1] so the projection stays on the segment.
        t = (p - a)[:, 0] * ab[:, 0] + (p - a)[:, 1] * ab[:, 1]
        t = np.clip(t / np.where(ab2 > 1e-12, ab2, 1.0), 0.0, 1.0)
        proj = a + ab * t.reshape(-1, 1)
        diff = p - proj
        d2 = diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1]  # squared distances

        j = int(np.argmin(d2))
        return i0 + j, proj[j]

    def _point_at_distance_ahead(self, seg_idx, from_point, distance):
        """Return the point `distance` metres further along the polyline."""
        pts = self.path_points
        nseg = len(pts) - 1
        if nseg <= 0:
            return np.asarray(from_point, dtype=float)

        i = int(max(0, min(seg_idx, nseg - 1)))
        s_here = self._arc_position(i, from_point)
        s_target = s_here + float(distance)

        total_len = float(self._cum_s[-1])
        if s_target >= total_len:
            if self._closed and total_len > 1e-9:
                s_target = s_target % total_len
            else:
                return pts[-1].copy()

        j = int(np.searchsorted(self._cum_s, s_target, side='right') - 1)
        j = int(max(0, min(j, nseg - 1)))
        ds = s_target - float(self._seg_s0[j])
        seg_len = float(self._seg_len[j])
        if seg_len < 1e-9:
            return pts[j].copy()
        return self._seg_a[j] + self._seg_ab[j] * (ds / seg_len)

    def _arc_position(self, seg_idx, point):
        """Arc length of `point` projected onto segment `seg_idx`."""
        a = self._seg_a[seg_idx]
        ab = self._seg_ab[seg_idx]
        ab2 = float(self._seg_ab2[seg_idx])
        t = 0.0
        if ab2 > 1e-12:
            t = float(np.clip(np.dot(np.asarray(point, dtype=float) - a, ab) / ab2, 0.0, 1.0))
        return float(self._seg_s0[seg_idx] + t * self._seg_len[seg_idx])

    def _signed_lateral_offset(self, point, seg_idx):
        """Signed offset from the centerline at `seg_idx` (positive = left)."""
        nseg = len(self.path_points) - 1
        if nseg <= 0:
            return 0.0
        i = int(max(0, min(seg_idx, nseg - 1)))
        return float(np.dot(np.asarray(point, dtype=float) - self._seg_a[i], self._seg_norm[i]))

    @property
    def progress_fraction(self):
        """Fraction of the lap reached, in [0, 1].

        A fraction rather than a segment index because the two drivers index
        different polylines: this agent walks the benchmark's own curve, while
        RLAgent walks a ring TrackGeometry resamples at a fixed 3 m spacing,
        about 10x as many points on a typical track.  RacingProblem compares
        progress against a fraction of the lap, so it has to be handed one.
        """
        nseg = max(1, len(self.path_points) - 1)
        return float(min(max(int(self.current_idx), 0), nseg)) / nseg

    @property
    def curve_points(self):
        return self.path_points

    @curve_points.setter
    def curve_points(self, points):
        self.path_points = np.array(points, dtype=float)
        self._closed = False
        self._precompute_path_geometry()

    # ── Control ───────────────────────────────────────────────────────────

    def act(self, car_state):
        """Return an action dict `{steering, throttle}` for the current state.

        `car_state` is `[x, y, angle, speed, steering_angle, ...]`.  Later
        elements carry body-frame velocity for the learned driver; a path
        follower plans from the centerline and does not read them.
        """
        x, y, angle, speed = car_state[0], car_state[1], car_state[2], car_state[3]
        if len(self.path_points) < 2:
            return {'steering': 0.0, 'throttle': 0.0}
        pos = np.array([float(x), float(y)])
        spd = float(speed)

        # Track progress: project onto the path near the last known segment,
        # never jumping backward past the window or too far forward at once.
        seg_idx, proj = self._find_projection(pos, start_idx=self.current_idx)
        cur = int(self.current_idx)
        if seg_idx < cur:
            seg_idx = max(seg_idx, cur - self.search_back)
        else:
            seg_idx = min(seg_idx, cur + self.max_index_advance)
        self.current_idx = int(seg_idx)

        # ── Steering: pure pursuit + cross-track trim + rate feedback ──
        # Aiming from the car (not from its projection) means that when the
        # car is pushed off the road, the geometry itself points back toward
        # the track, so recovery needs no special case.
        # lookahead_base already equals the smallest useful lookahead, so
        # only the upper end needs clamping.
        lookahead = self.lookahead_base + self.lookahead_gain * spd
        lookahead = min(lookahead, self.lookahead_max)
        look_pt = self._point_at_distance_ahead(self.current_idx, proj, lookahead)
        self.last_lookahead_point = (float(look_pt[0]), float(look_pt[1]))

        to_aim = look_pt - pos
        heading_err = math.atan2(to_aim[1], to_aim[0]) - float(angle)
        heading_err = (heading_err + math.pi) % (2.0 * math.pi) - math.pi

        # Pure pursuit: the wheel angle that puts the car on the circular arc
        # through its current pose and the aim point,
        #     delta = atan(2 * L * sin(eta) / ld),
        # with ld the true distance to the aim point rather than the requested
        # lookahead, since on a curve the two differ.  This is the geometry the
        # car actually needs; using the heading error raw asks for 1.6x to 6x
        # more than that and only stays stable if something downstream divides
        # it back out.
        #
        # Past a quarter turn the geometry stops being usable: sin(eta) falls
        # back toward zero as the aim point swings behind the car, so the arc
        # solution asks for LESS steering the more wrong the car is pointed,
        # and at 180 degrees it asks for none at all.  A car that overshoots a
        # corner too tight to make would then drive away in a straight line.
        # Outside the quarter turn the answer is simply full lock toward the
        # aim point, which is the standard treatment of pursuit's blind spot.
        ld = float(np.linalg.norm(to_aim))
        if ld <= 1e-6:
            delta_pursuit = 0.0
        elif abs(heading_err) > 0.5 * math.pi:
            delta_pursuit = math.copysign(self.nominal_max_steer_rad, heading_err)
        else:
            delta_pursuit = math.atan2(2.0 * self.wheelbase * math.sin(heading_err), ld)

        # Path curvature at the car, used below to tell the rate feedback what
        # yaw rate the corner actually calls for.
        curvature_ref = 0.0
        if len(self._path_curvature) > 0:
            curvature_ref = float(self._path_curvature[
                min(self.current_idx, len(self._path_curvature) - 1)])
        lat_off = self._signed_lateral_offset(pos, self.current_idx)
        stanley = math.atan2(self.stanley_k * (-lat_off), spd + self.stanley_v0)

        # Rate feedback.  The pursuit arc and the cross-track trim are both
        # proportional terms, and on their own they ring: the car corrects, the
        # yaw it built carries it past the line, and it corrects back.
        #
        # What is damped is the yaw rate ERROR against the rate the path itself
        # calls for, v * curvature, not the yaw rate outright.  Damping the raw
        # rate would fight steady cornering, where a large yaw rate is exactly
        # what the corner needs; damping the error only resists rotating faster
        # or slower than the corner asks for.
        yaw_rate_measured = 0.0
        if self._prev_angle is not None:
            d = (float(angle) - self._prev_angle + math.pi) % (2.0 * math.pi) - math.pi
            yaw_rate_measured = d / self._dt
        self._prev_angle = float(angle)
        yaw_rate_ref = math.copysign(spd * curvature_ref, delta_pursuit)
        delta_damp = -self.yaw_damp_gain * (yaw_rate_measured - yaw_rate_ref)

        steer_rad = delta_pursuit + stanley + delta_damp

        # Deadzone: on a near-straight aim, hold the wheel still rather than
        # chasing sub-degree errors (removes idle twitching).
        if abs(heading_err) < self.steer_deadzone_rad and abs(lat_off) < 0.5:
            steer_rad = 0.0

        steering = steer_rad / self.nominal_max_steer_rad
        steering = min(max(steering, -1.0), 1.0)

        # ── Throttle: track the speed profile over a short horizon ──
        # Taking the minimum over now / react_time / 2*react_time ahead makes
        # braking start early enough that the proportional controller does
        # not lag behind a falling profile.
        s_here = self._arc_position(self.current_idx, proj)
        total_len = float(self._cum_s[-1])
        target = self.max_speed
        for dt_ahead in (0.0, self.react_time, 2.0 * self.react_time):
            s = s_here + spd * dt_ahead
            if self._closed and total_len > 1e-9:
                s = s % total_len
            target = min(target, float(np.interp(s, self._cum_s, self._speed_profile)))

        # Off the corridor the plan no longer applies: slow down instead.
        half_width = 0.5 * self.track_width
        corridor = max(1.0, half_width - self.corridor_margin)
        abs_off = abs(lat_off)
        if abs_off > corridor:
            over = min((abs_off - corridor) / max(half_width - corridor, 1e-6), 1.0)
            target = max(self.min_speed, target * (1.0 - 0.8 * over))

        err = target - spd
        if err >= 0.0:
            throttle = err / self.accel_deficit_full
        else:
            throttle = err / self.brake_excess_full
        throttle = min(max(throttle, -1.0), 1.0)

        return {'steering': float(steering), 'throttle': float(throttle)}
