import math

import numpy as np


class SteeringAgent:
    """Path-following agent: lookahead + Stanley steering, and a speed target
    taken from a precomputed physical speed profile.

    The speed profile is the standard approach used in racing AI and
    trajectory planning, computed once per track in three steps:

      1. Corner limit: at every path point the lateral acceleration bound
         gives a maximum cornering speed  v = sqrt(a_lat_max / curvature).
      2. Backward pass: driving toward a corner, speed may exceed the corner
         limit only by what braking can shed over the remaining distance
         (v^2 <= v_next^2 + 2 * a_brake * ds), so the car brakes *before*
         corners instead of reacting inside them.
      3. Forward pass: speed may rise out of a corner only as fast as the
         engine accelerates (v^2 <= v_prev^2 + 2 * a_accel * ds), so corner
         exits ramp up smoothly instead of snapping to full throttle.

    act() then only has to steer at a lookahead point and track the profile,
    which keeps the per-step work small and the behavior explainable.
    Steering output is normalized to [-1, 1] for the engine.
    """

    def __init__(self, curve_points, track_width=12.0, max_speed=25.0):
        """Create a steering agent for the given path."""
        self.track_width = float(track_width)
        self.max_speed = float(max_speed)

        self.last_lookahead_point = None

        # Physical limits used by the speed profile (m/s^2).  The engine's
        # tires allow ~12.8 m/s^2 (mu 1.3).  Racing-game bots (TORCS/Speed
        # Dreams) plan corner speed at 80-90% of available grip, leaving a
        # margin for control error; 11 is ~86% of grip, which is as much as
        # this steering law holds cleanly (higher runs wide on tight tile
        # corners).
        self.lat_accel_max = 11.0   # cornering
        self.brake_decel   = 6.0    # braking before corners
        self.accel_max     = 4.0    # acceleration out of corners
        self.min_speed     = 4.0    # never plan slower than this

        # Steering law.  Lookahead is left at the tuned agile values: the
        # engine's yaw damping now handles stability, so the agent does not need
        # a long (sluggish) lookahead that would run the tighter grid tracks
        # wide.  Short lookahead + engine damping keeps corners crisp AND stable.
        self.lookahead_base = 8.0    # metres at standstill
        self.lookahead_gain = 0.35   # extra metres per m/s of speed
        self.lookahead_max  = 30.0
        self.stanley_k      = 0.35   # cross-track gain (trim only; the
        self.stanley_v0     = 1.5    # lookahead aim does the main work)
        self.nominal_max_steer_rad = math.radians(28.0)
        # Damp the steering by the car's own rotation rate: when the car is
        # already turning toward the aim point, ease off a little so it stops on
        # the line rather than swinging past it.  Kept light (the engine yaw
        # damping does the heavy lifting) so it smooths the residual settle
        # wiggle without slowing corner entry.
        self.yaw_damp_gain = 0.03    # seconds; steering reduced per rad/s of yaw
        # Below this heading error (radians) on a near-centered car, hold
        # straight instead of chasing tiny errors (deadzone stops twitching).
        self.steer_deadzone_rad = math.radians(0.6)

        # Slow down when the car drifts outside the usable corridor
        # (half width minus a margin).
        self.corridor_margin = max(1.0, 0.18 * self.track_width)

        # Speed tracking: full throttle / full brake at these speed errors
        # (m/s).  Braking is stiffer than accelerating so the car does not
        # lag behind a falling profile and enter corners too fast.
        self.accel_deficit_full = 5.0
        self.brake_excess_full  = 2.5
        self.react_time         = 0.3

        # Projection window (path indices), same scheme as before: search a
        # window around the last known segment so progress stays monotonic.
        self.search_ahead = 55
        self.search_back = 15
        self.max_index_advance = 12

        self.current_idx = 0
        self._prev_angle = None      # for yaw-rate estimate in act()
        self._dt = 0.1               # engine time step; steering damping only
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

    def _compute_speed_profile(self, pts, seg_tan, seg_len):
        """Per-vertex speed limits: corner limit, then brake and accel passes."""
        n = len(pts)
        m = len(seg_len)  # = n - 1 segments

        # Curvature at each interior vertex: turn angle between the two
        # adjacent segments divided by the local arc length.
        v_corner = np.full(n, self.max_speed, dtype=float)
        for i in range(1, m):
            t0, t1 = seg_tan[i - 1], seg_tan[i]
            dot = float(np.clip(np.dot(t0, t1), -1.0, 1.0))
            angle = math.acos(dot)
            ds = 0.5 * float(seg_len[i - 1] + seg_len[i])
            if ds > 1e-9 and angle > 1e-9:
                curvature = angle / ds
                v_corner[i] = math.sqrt(self.lat_accel_max / curvature)
        if self._closed and m >= 2:
            # The seam vertex (0 == n-1) also has a turn angle.
            t0, t1 = seg_tan[-1], seg_tan[0]
            dot = float(np.clip(np.dot(t0, t1), -1.0, 1.0))
            angle = math.acos(dot)
            ds = 0.5 * float(seg_len[-1] + seg_len[0])
            if ds > 1e-9 and angle > 1e-9:
                v_corner[0] = math.sqrt(self.lat_accel_max / (angle / ds))
                v_corner[-1] = v_corner[0]

        profile = np.clip(v_corner, self.min_speed, self.max_speed)

        # Backward pass: entering point i at profile[i] must allow braking
        # down to profile[i+1] over segment i.  For closed loops run the pass
        # twice so the constraint propagates across the seam.
        rounds = 2 if self._closed else 1
        for _ in range(rounds):
            for i in range(m - 1, -1, -1):
                v_allowed = math.sqrt(profile[i + 1] ** 2 + 2.0 * self.brake_decel * float(seg_len[i]))
                if profile[i] > v_allowed:
                    profile[i] = v_allowed
            if self._closed:
                profile[-1] = profile[0] = min(profile[0], profile[-1])

        # Forward pass: speed can only build up at accel_max.
        for _ in range(rounds):
            for i in range(m):
                v_reachable = math.sqrt(profile[i] ** 2 + 2.0 * self.accel_max * float(seg_len[i]))
                if profile[i + 1] > v_reachable:
                    profile[i + 1] = v_reachable
            if self._closed:
                profile[-1] = profile[0] = min(profile[0], profile[-1])

        return profile

    def reset(self):
        """Reset agent to start of path."""
        self.current_idx = 0
        self.last_lookahead_point = None
        self._prev_angle = None      # for yaw-rate estimate in act()
        self._dt = 0.1               # engine time step; steering damping only

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

        `car_state` is `[x, y, angle, speed, steering_angle]`.
        """
        x, y, angle, speed, _steer = car_state
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

        # ── Steering: aim the car at a lookahead point + Stanley term ──
        # Aiming from the car (not from its projection) means that when the
        # car is pushed off the road, the heading error itself points back
        # toward the track, so recovery needs no special case.
        # lookahead_base already equals the smallest useful lookahead, so
        # only the upper end needs clamping.
        lookahead = self.lookahead_base + self.lookahead_gain * spd
        lookahead = min(lookahead, self.lookahead_max)
        look_pt = self._point_at_distance_ahead(self.current_idx, proj, lookahead)
        self.last_lookahead_point = (float(look_pt[0]), float(look_pt[1]))

        heading_err = math.atan2(look_pt[1] - pos[1], look_pt[0] - pos[0]) - float(angle)
        heading_err = (heading_err + math.pi) % (2.0 * math.pi) - math.pi

        lat_off = self._signed_lateral_offset(pos, self.current_idx)
        stanley = math.atan2(self.stanley_k * (-lat_off), spd + self.stanley_v0)

        # Estimate the car's turn rate from the heading change since last call,
        # then subtract it: if the car is already rotating toward the aim, we
        # need less steering.  This damping is what stops the recovery wiggle.
        yaw_rate_est = 0.0
        if self._prev_angle is not None:
            d = (float(angle) - self._prev_angle + math.pi) % (2.0 * math.pi) - math.pi
            yaw_rate_est = d / self._dt
        self._prev_angle = float(angle)

        steer_rad = heading_err + stanley - self.yaw_damp_gain * yaw_rate_est

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
