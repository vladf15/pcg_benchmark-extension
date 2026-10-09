import math

import numpy as np


class SteeringAgent:
    """Path follower: pure-pursuit steering with a cross-track trim, tracking
    a speed profile computed once per track, the standard racing-line way:
    the corner limit v = sqrt(ay_max / curvature) at every point, a backward
    pass (v^2 <= v_next^2 + 2 a_brake ds, so it brakes before corners) and a
    forward pass (v^2 <= v_prev^2 + 2 a_accel ds).  a_brake and a_accel come
    from the car, not constants: the grip a corner uses is not available
    for braking or accelerating, and drivetrain, power and drag limit them
    too (_available_ax).  Steering is normalised to [-1, 1]."""

    def __init__(self, curve_points, track_width=12.0, max_speed=None, engine=None):
        """`engine`: the CarPhysicsEngine whose limits the plan is built from
        (defaults describe the same 992).  `max_speed` caps the plan, by
        default the car's own 85.5 m/s (a 25 m/s cap held it to 24.1 m/s on
        the circuits, against 61.7 uncapped)."""
        self.track_width = float(track_width)
        if max_speed is None:
            max_speed = getattr(engine, 'max_speed', 85.5)
        self.max_speed = float(max_speed)

        self.last_lookahead_point = None

        # ── Car parameters, read from the engine when one is supplied ──
        g = 9.81
        self.wheelbase       = float(getattr(engine, 'length', 2.45))
        mass                 = float(getattr(engine, 'mass', 1534.0))
        tire_mu              = float(getattr(engine, 'tire_mu', 1.3))
        tire_mu_rear         = float(getattr(engine, 'tire_mu_rear', tire_mu))
        rear_load_share      = (float(getattr(engine, 'lf', 1.47))
                                / float(getattr(engine, 'length', 2.45)))
        max_drive_force      = float(getattr(engine, 'max_drive_force', 12700.0))
        max_brake_force      = float(getattr(engine, 'max_brake_force', 18700.0))
        self._max_power      = float(getattr(engine, 'max_power', 331000.0))
        self._mass           = mass
        # Resistance: against acceleration, with braking.
        self._drag_k         = 0.5 * float(getattr(engine, 'rho_air', 1.225)) \
                                   * float(getattr(engine, 'cd_a', 0.58)) / mass
        self._roll_a         = float(getattr(engine, 'c_rr', 0.015)) * g

        # "Swept" below: the 24 circuits plus 20 random genomes per
        # representation (racing, tile, hex, voronoi; seed 21, seed 7 for the
        # final settings), 104 laps, one constant at a time, on the 16 m road
        # (edge at 8 m): laps finished, tracks with any off-road step, the
        # car's largest offset from the centreline, mean lap speed.  Not
        # circles: they punish no aggressive correction, and gains that halve
        # circle error measured far worse on generated tracks.
        # calibration/driver_sweep.py.

        # Fraction of the limit the plan aims for (100% leaves nothing to
        # correct with), one factor on every axis.  Swept at 1.12 g cornering
        # (0.47 m cg), every other setting as below:
        #
        #   utilisation   laps     tracks off-road   worst offset   circuit speed
        #   0.80          103/104        5           838 m            29.4 m/s
        #   0.77          104/104        0           2.6 m (2.1)      28.9 m/s
        #   0.75          104/104        0           1.8 m (1.7)      28.6 m/s
        #   0.72          104/104        0           1.6 m            28.1 m/s
        #
        # (seed 7 in brackets.)  At 0.80 a car leaves the map on a hex track.
        # 0.75 over 0.77 (1% slower) keeps more margin, since the search makes
        # harder corners than random genomes.
        self.grip_utilisation = 0.75

        # Tyre limits: cornering and braking use every tyre; accelerating only
        # the driven rear axle (planning on the whole car asks 11.5 m/s^2 of a
        # rear that gives 7.6, and the wheels spin on every exit).
        self.ay_max          = self.grip_utilisation * tire_mu * g
        self.ax_max_brake_tires = self.ay_max
        self.ax_max_accel_tires = self.grip_utilisation * tire_mu_rear * rear_load_share * g
        # Machine limits, separate from the tyres as TUM's
        # trajectory_planning_helpers separates ggv from ax_max_machines.
        self.ax_max_drive  = self.grip_utilisation * max_drive_force / mass
        self.ax_max_brake  = self.grip_utilisation * max_brake_force / mass
        # Friction-ellipse exponent (1 linear, 2 an ellipse; the same parameter
        # as TUM's calc_vel_profile, Heilmeier et al. 2020).  Swept: 1.0 costs
        # 0.4 m/s, 2.0 takes a track off the road (12.0 m); 1.4 neither.
        self.friction_exponent = 1.4

        # Lookahead grows with speed (a roughly fixed time ahead).  Swept: a
        # 5 m base takes a track off the road, 12 m doubles the offset; gains
        # 0.5 / 0.35 / 0.25 / 0.15 m per m/s give a median worst offset of 1.9
        # / 1.5 / 1.3 / 1.0 m, and 0.1 or 0 the same as 0.15, the smallest that
        # still grows with speed.  No cap needed (17 m at 62 m/s).
        self.lookahead_base = 8.0    # metres at standstill
        self.lookahead_gain = 0.15   # extra metres per m/s of speed
        # Cross-track gain.  Circles prefer 1.6 (half their error), which drops
        # the spline representations from 6 of 8 laps to 1 and 0.  At full top
        # speed 0.2 raises the worst offset to 4.1 m, 0.5 takes a track off.
        self.stanley_k      = 0.35
        # Keeps the Stanley term finite at a standstill; 0.5, 1.5 and 3 m/s
        # measure the same (a modelling choice).
        self.stanley_v0     = 1.5
        # Yaw-rate feedback gain (s, see act()).  Without it 7 tracks leave the
        # road; 0.05 / 0.1 / 0.15 / 0.2 s give a worst offset of 3.8 / 1.8 /
        # 2.6 / 9.8 m.
        self.yaw_damp_gain  = 0.1
        # The engine's lock, which it applies unaltered: the command is exact.
        self.nominal_max_steer_rad = float(getattr(engine, 'max_steering',
                                                   math.radians(30.0)))

        # Full throttle / brake at these speed errors (m/s), proportional in
        # between.  Mean circuit lap speed:
        #
        #   full throttle at   18    12     8     5     3     2     1
        #   circuit speed    26.7  27.6  28.3  29.0  29.6  30.0  30.0 m/s
        #
        # Clean down to 2, at 1 a car leaves the map; 3 keeps one clean
        # setting from the failure.  Full brake at 2.5 / 4 / 6: the first two
        # clean, 6 takes 4 tracks off; 2.5 likewise keeps a margin.
        self.accel_deficit_full = 3.0
        self.brake_excess_full  = 2.5

        # Projection window (segments) around the last one, so progress is
        # monotonic.  From the geometry: a step covers at most 8.6 m (under
        # two 5 m segments), so 12 per step and 55 ahead never bind; 15 back
        # covers a car sliding back after a spin.
        self.search_ahead = 55
        self.search_back = 15
        self.max_index_advance = 12

        self.current_idx = 0
        self._prev_angle = None      # for the yaw-rate estimate in act()
        self._dt = float(getattr(engine, 'time_step', 0.1))
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
        """Longitudinal acceleration left at `speed` on `curvature` (m/s^2,
        >= 0): ax_max (1 - (ay_used / ay_max)^n)^(1/n), TUM's calc_vel_profile
        form, the tighter of tyres and drivetrain, then drag and rolling
        resistance with their real sign."""
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

        # Curvature at each interior vertex: turn angle over the local arc.
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
        profile = np.minimum(v_corner, self.max_speed)

        # Backward pass: point i must allow braking to profile[i+1] over
        # segment i, at the deceleration left by i+1's corner; twice on a
        # closed loop so the constraint crosses the seam.
        rounds = 2 if self._closed else 1
        for _ in range(rounds):
            for i in range(m - 1, -1, -1):
                a = self._available_ax(profile[i + 1], curvature[i + 1], braking=True)
                v_allowed = math.sqrt(profile[i + 1] ** 2 + 2.0 * a * float(seg_len[i]))
                if profile[i] > v_allowed:
                    profile[i] = v_allowed
            if self._closed:
                profile[-1] = profile[0] = min(profile[0], profile[-1])

        # Forward pass: speed builds only as fast as the grip left at i allows.
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
        """Fraction of the lap reached, in [0, 1] (a fraction, since the two
        drivers index different polylines)."""
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
        """{steering, throttle} for car_state [x, y, angle, speed, ...] (the
        body-frame elements after are for the learned driver)."""
        x, y, angle, speed = car_state[0], car_state[1], car_state[2], car_state[3]
        if len(self.path_points) < 2:
            return {'steering': 0.0, 'throttle': 0.0}
        pos = np.array([float(x), float(y)])
        spd = float(speed)

        # Progress: project near the last segment, within the window.
        seg_idx, proj = self._find_projection(pos, start_idx=self.current_idx)
        cur = int(self.current_idx)
        if seg_idx < cur:
            seg_idx = max(seg_idx, cur - self.search_back)
        else:
            seg_idx = min(seg_idx, cur + self.max_index_advance)
        self.current_idx = int(seg_idx)

        # ── Steering: pure pursuit + cross-track trim + rate feedback ──
        # Aimed from the car, not its projection, so off the road the geometry
        # itself points back: recovery needs no special case.
        lookahead = self.lookahead_base + self.lookahead_gain * spd
        look_pt = self._point_at_distance_ahead(self.current_idx, proj, lookahead)
        self.last_lookahead_point = (float(look_pt[0]), float(look_pt[1]))

        to_aim = look_pt - pos
        heading_err = math.atan2(to_aim[1], to_aim[0]) - float(angle)
        heading_err = (heading_err + math.pi) % (2.0 * math.pi) - math.pi

        # Pure pursuit (Coulter 1992): delta = atan(2 L sin(eta) / ld), the arc
        # through the pose and the aim point, ld its true distance (the raw
        # heading error asks 1.6-6x more).  Past a quarter turn sin(eta) falls
        # back to zero and asks LESS steering the worse the heading, so there
        # it is full lock toward the aim point, the standard treatment.
        ld = float(np.linalg.norm(to_aim))
        if ld <= 1e-6:
            delta_pursuit = 0.0
        elif abs(heading_err) > 0.5 * math.pi:
            delta_pursuit = math.copysign(self.nominal_max_steer_rad, heading_err)
        else:
            delta_pursuit = math.atan2(2.0 * self.wheelbase * math.sin(heading_err), ld)

        curvature_ref = 0.0
        if len(self._path_curvature) > 0:
            curvature_ref = float(self._path_curvature[
                min(self.current_idx, len(self._path_curvature) - 1)])
        lat_off = self._signed_lateral_offset(pos, self.current_idx)
        # Cross-track term of the Stanley controller (Hoffmann et al. 2007,
        # Eq. 9), atan(k * e / (k_soft + v)); k_soft (stanley_v0) keeps it
        # finite at a standstill, and they found 1 m/s appropriate.
        stanley = math.atan2(self.stanley_k * (-lat_off), spd + self.stanley_v0)

        # Rate feedback, the k_d,yaw (r_meas - r_traj) term of Hoffmann et al.'s
        # Eq. 9: the two proportional terms alone ring.  It damps the yaw rate
        # ERROR against the path's v * curvature, not the raw rate, which
        # would fight steady cornering.
        yaw_rate_measured = 0.0
        if self._prev_angle is not None:
            d = (float(angle) - self._prev_angle + math.pi) % (2.0 * math.pi) - math.pi
            yaw_rate_measured = d / self._dt
        self._prev_angle = float(angle)
        yaw_rate_ref = math.copysign(spd * curvature_ref, delta_pursuit)
        delta_damp = -self.yaw_damp_gain * (yaw_rate_measured - yaw_rate_ref)

        steer_rad = delta_pursuit + stanley + delta_damp

        steering = steer_rad / self.nominal_max_steer_rad
        steering = min(max(steering, -1.0), 1.0)

        # ── Throttle: the profile at the car (it already brakes ahead) ──
        # Reading the minimum 0.1-0.5 s ahead too costs 0.4-1.5 m/s for no
        # tighter line.  No off-road slow-down or steering deadzone: neither
        # engaged on the 104 laps swept.
        s_here = self._arc_position(self.current_idx, proj)
        total_len = float(self._cum_s[-1])
        if self._closed and total_len > 1e-9:
            s_here = s_here % total_len
        target = float(np.interp(s_here, self._cum_s, self._speed_profile))

        err = target - spd
        if err >= 0.0:
            throttle = err / self.accel_deficit_full
        else:
            throttle = err / self.brake_excess_full
        throttle = min(max(throttle, -1.0), 1.0)

        return {'steering': float(steering), 'throttle': float(throttle)}
