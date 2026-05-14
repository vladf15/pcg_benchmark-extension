import math

import numpy as np


class SteeringAgent:
    """Path-following agent using a Stanley-style steering law.

    Steering output is normalized to [-1, 1] for the engine.
    """

    def __init__(
        self,
        curve_points,
        track_width=12.0,
        line_strategy=None,
        max_speed=25.128261272942474,
        enable_wander=False,
    ):
        """Create a steering agent for the given path."""
        self.path_points = np.array(curve_points, dtype=float)
        self.track_width = float(track_width)
        self.max_speed = float(max_speed)
        self.enable_wander = bool(enable_wander)

        self.last_lookahead_point = None

        self.look_ahead_dist = 8.299299919527938
        self.curvature_lookahead = 55.0
        self.predict_time_base = 0.40
        self.predict_time_speed_gain = 0.012
        self.search_ahead = 55
        self.search_back = 15
        self.max_index_advance = 12

        self.corridor_margin = max(1.0, 0.18 * float(self.track_width))
        # NOTE: We intentionally avoid any default-to-center behavior.
        # Lateral position is controlled by a persistent setpoint and edge containment only.
        self.centering_gain_rad = 0.0
        self.containment_gain_rad = math.radians(79.80565380471872)

        # Persistent lateral setpoint (signed offset from centerline) that we keep unless
        # a racing-line strategy (or edge containment) changes it.
        self._lat_setpoint = 0.0
        self.lat_setpoint_hold_alpha = 0.21479116970541606
        self.lat_setpoint_turn_alpha = 0.5491085743829589
        self.lat_setpoint_edge_push = 0.65

        # Entry/apex fractions are measured relative to half-track corridor width.
        # Slightly wider outside-before-turn entry (still clamped for safety).
        self.racing_line_max_frac = 0.82
        self.racing_line_entry_frac = 1.12
        self.racing_line_apex_frac = 0.019266969270553408
        self.racing_line_enable = True

        self.nominal_max_steer_rad = math.radians(28.0)
        self.stanley_k = 0.23482655615044332
        self.stanley_v0 = 1.5
        self.curvature_ff_gain = 0.43593169971431184
        self.curvature_ff_gain_line = 0.1392239380926806

        self.lookahead_base = 7.623782135263054
        self.lookahead_speed_gain = 0.1033620201029109
        self.lookahead_low_speed_cutoff = 13.985505684631006
        self.lookahead_low_speed_scale = 0.5592245566254428
        self.lookahead_min = 7.623782135263054
        self.lookahead_max = 40.76421249709552

        self.min_speed = 2.8032007263857475
        self.turn_speed = 7.0
        self.lat_accel_max = 10.401453052397617

        self.current_idx = 0

        self._precompute_path_geometry()
    
    def _precompute_path_geometry(self):
        """Precompute per-segment geometry for fast projection/lookahead queries."""
        pts = np.asarray(self.path_points, dtype=float)
        n = len(pts)
        if n < 2:
            self._seg_vec = np.zeros((0, 2), dtype=float)
            self._seg_len = np.zeros((0,), dtype=float)
            self._seg_tan = np.zeros((0, 2), dtype=float)
            self._seg_norm = np.zeros((0, 2), dtype=float)
            self._seg_a = np.zeros((0, 2), dtype=float)
            self._seg_b = np.zeros((0, 2), dtype=float)
            self._seg_ab = np.zeros((0, 2), dtype=float)
            self._seg_ab2 = np.zeros((0,), dtype=float)
            self._scratch_t = np.zeros((0,), dtype=float)
            self._scratch_d2 = np.zeros((0,), dtype=float)
            self._seg_s0 = np.zeros((0,), dtype=float)
            self._cum_s = np.zeros((0,), dtype=float)
            return

        seg_a = pts[:-1].copy()
        seg_b = pts[1:].copy()
        seg = seg_b - seg_a
        seg_len = np.linalg.norm(seg, axis=1)
        seg_tan = np.zeros_like(seg)
        nonzero = seg_len > 1e-9
        seg_tan[nonzero] = seg[nonzero] / seg_len[nonzero, None]
        seg_norm = np.stack([-seg_tan[:, 1], seg_tan[:, 0]], axis=1)

        seg_ab2 = np.einsum('ij,ij->i', seg, seg)

        seg_s0 = np.zeros((len(seg_len),), dtype=float)
        if len(seg_len) > 0:
            seg_s0[1:] = np.cumsum(seg_len[:-1])
        cum_s = np.concatenate(([0.0], np.cumsum(seg_len)))

        self._seg_vec = seg
        self._seg_len = seg_len
        self._seg_tan = seg_tan
        self._seg_norm = seg_norm

        self._seg_a = seg_a
        self._seg_b = seg_b
        self._seg_ab = seg
        self._seg_ab2 = seg_ab2
        self._seg_s0 = seg_s0
        self._cum_s = cum_s

        m = len(seg_ab2)
        self._scratch_t = np.zeros((m,), dtype=float)
        self._scratch_d2 = np.zeros((m,), dtype=float)

        self._scratch_pos = np.zeros((2,), dtype=float)
        self._scratch_proj = np.zeros((2,), dtype=float)
        self._scratch_pred = np.zeros((2,), dtype=float)

        self._scratch_vel = np.zeros((2,), dtype=float)
    
    def reset(self):
        """Reset agent to start of path."""
        self.current_idx = 0
        self.last_lookahead_point = None
        self._lat_setpoint = 0.0

    def _find_projection(self, point, start_idx):
        """Project `point` onto a local window of the polyline.

        Returns `(seg_idx, proj_point)` where `seg_idx` is the best segment index.
        """
        nseg = len(self.path_points) - 1
        if nseg <= 0:
            return 0, np.asarray(point, dtype=float)

        i0 = int(max(0, min(start_idx - int(self.search_back), nseg - 1)))
        i1 = int(min(nseg - 1, max(i0, start_idx) + int(self.search_ahead)))
        sl = slice(i0, i1 + 1)

        p = np.asarray(point, dtype=float)

        a = self._seg_a[sl]
        ab = self._seg_ab[sl]
        ab2 = self._seg_ab2[sl]
        t = self._scratch_t[sl]
        d2 = self._scratch_d2[sl]

        # Dot-product numerator reused for projection parameter t on each segment.
        pab = self._scratch_d2[sl]
        pab[:] = p[0] * ab[:, 0] + p[1] * ab[:, 1]
        pab -= (a[:, 0] * ab[:, 0] + a[:, 1] * ab[:, 1])
        denom = np.where(ab2 > 1e-12, ab2, 1.0)
        t[:] = pab / denom
        np.clip(t, 0.0, 1.0, out=t)

        proj_x = a[:, 0] + ab[:, 0] * t
        proj_y = a[:, 1] + ab[:, 1] * t
        dx = p[0] - proj_x
        dy = p[1] - proj_y
        d2[:] = dx * dx + dy * dy

        j = int(np.argmin(d2))
        best_i = i0 + j
        out = self._scratch_proj
        out[0] = float(proj_x[j])
        out[1] = float(proj_y[j])
        return best_i, out

    def _point_at_distance_ahead(self, seg_idx, from_point, distance):
        """Return a point `distance` ahead along the polyline from `from_point`."""
        pts = self.path_points
        nseg = len(pts) - 1
        if nseg <= 0:
            return np.asarray(from_point, dtype=float)

        i = int(max(0, min(seg_idx, nseg - 1)))
        p = np.asarray(from_point, dtype=float)

        a = self._seg_a[i]
        ab = self._seg_ab[i]
        ab2 = float(self._seg_ab2[i])
        if ab2 > 1e-12:
            t = float(np.dot(p - a, ab) / ab2)
            t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        else:
            t = 0.0
        s_here = float(self._seg_s0[i] + t * self._seg_len[i])
        s_target = s_here + float(distance)

        total_len = float(self._cum_s[-1]) if len(self._cum_s) else 0.0
        if s_target >= total_len:
            return pts[-1].copy()

        j = int(np.searchsorted(self._cum_s, s_target, side='right') - 1)
        j = int(max(0, min(j, nseg - 1)))
        ds = s_target - float(self._seg_s0[j])
        seg_len = float(self._seg_len[j])
        if seg_len < 1e-9:
            return pts[j].copy()
        u = ds / seg_len
        return self._seg_a[j] + self._seg_ab[j] * u

    def _signed_lateral_offset(self, point, seg_idx):
        """Signed lateral offset from centerline at `seg_idx` (positive = left)."""
        pts = self.path_points
        nseg = len(pts) - 1
        if nseg <= 0:
            return 0.0
        i = int(max(0, min(seg_idx, nseg - 1)))
        a = pts[i]
        n = self._seg_norm[i] if len(self._seg_norm) > 0 else np.array([0.0, 0.0])
        return float(np.dot(point - a, n))
    
    @property
    def curve_points(self):
        return self.path_points
    
    @curve_points.setter
    def curve_points(self, points):
        self.path_points = np.array(points, dtype=float)
        self._precompute_path_geometry()
    
    def act(self, car_state):
        """Return an action dict `{steering, throttle}` for the current `car_state`.

        `car_state` is `[x, y, angle, speed, steering_angle]`.
        """
        x, y, angle, speed, _steering_angle = car_state
        pos = self._scratch_pos
        pos[0] = float(x)
        pos[1] = float(y)
        cos_a = math.cos(float(angle))
        sin_a = math.sin(float(angle))
        vel = self._scratch_vel
        spd = float(speed)
        vel[0] = cos_a * spd
        vel[1] = sin_a * spd

        if len(self.path_points) < 2:
            return {'steering': 0.0, 'throttle': 0.0}

        predict_t = self.predict_time_base + self.predict_time_speed_gain * float(speed)
        predicted = self._scratch_pred
        predicted[0] = pos[0] + vel[0] * predict_t
        predicted[1] = pos[1] + vel[1] * predict_t
        seg_idx, proj = self._find_projection(predicted, start_idx=self.current_idx)

        seg_idx = int(seg_idx)
        cur = int(self.current_idx)
        if seg_idx < cur:
            seg_idx = max(seg_idx, cur - int(self.search_back))
        else:
            seg_idx = min(seg_idx, cur + int(self.max_index_advance))
        self.current_idx = int(seg_idx)

        half_width = self.track_width * 0.5
        corridor = max(1.0, half_width - self.corridor_margin)
        lat_off = self._signed_lateral_offset(predicted, self.current_idx)
        abs_off = abs(lat_off)

        lookahead_speed_gain = float(self.lookahead_speed_gain)
        if spd < float(self.lookahead_low_speed_cutoff):
            # Taper speed contribution at low speed to force earlier turn-in.
            low_t = spd / float(self.lookahead_low_speed_cutoff)
            if low_t < 0.0:
                low_t = 0.0
            elif low_t > 1.0:
                low_t = 1.0
            speed_scale = float(self.lookahead_low_speed_scale) + (1.0 - float(self.lookahead_low_speed_scale)) * low_t
            lookahead = float(self.lookahead_base) + lookahead_speed_gain * spd * speed_scale
        else:
            lookahead = float(self.lookahead_base) + lookahead_speed_gain * spd
        if abs_off > corridor:
            lookahead *= 0.45
        elif abs_off > 0.85 * corridor:
            lookahead *= 0.65
        if lookahead < self.lookahead_min:
            lookahead = self.lookahead_min
        elif lookahead > self.lookahead_max:
            lookahead = self.lookahead_max
        look_pt = self._point_at_distance_ahead(self.current_idx, proj, lookahead)

        t_dx = float(look_pt[0] - proj[0])
        t_dy = float(look_pt[1] - proj[1])
        t2 = t_dx * t_dx + t_dy * t_dy
        if t2 > 1e-12:
            path_heading = math.atan2(t_dy, t_dx)
        else:
            tan = self._seg_tan[self.current_idx] if len(self._seg_tan) > 0 else (1.0, 0.0)
            path_heading = math.atan2(float(tan[1]), float(tan[0]))

        heading_err = path_heading - float(angle)
        heading_err = (heading_err + math.pi) % (2 * math.pi) - math.pi
        abs_he = abs(float(heading_err))

        near_pt = self._point_at_distance_ahead(self.current_idx, proj, max(20.0, 0.45 * lookahead))
        far_pt = self._point_at_distance_ahead(self.current_idx, proj, max(60.0, float(self.curvature_lookahead), lookahead + 35.0))
        n_dx = float(near_pt[0] - proj[0])
        n_dy = float(near_pt[1] - proj[1])
        f_dx = float(far_pt[0] - proj[0])
        f_dy = float(far_pt[1] - proj[1])
        n2 = n_dx * n_dx + n_dy * n_dy
        f2 = f_dx * f_dx + f_dy * f_dy
        if n2 > 1e-12 and f2 > 1e-12:
            invn = 1.0 / math.sqrt(n2)
            invf = 1.0 / math.sqrt(f2)
            nux, nuy = n_dx * invn, n_dy * invn
            fux, fuy = f_dx * invf, f_dy * invf
            dot = nux * fux + nuy * fuy
            if dot < -1.0:
                dot = -1.0
            elif dot > 1.0:
                dot = 1.0
            bend = math.acos(dot)
            cross = nux * fuy - nuy * fux
            bend_sign = -1.0 if cross < 0.0 else (1.0 if cross > 0.0 else 0.0)
        else:
            bend = 0.0
            bend_sign = 0.0

        bend_factor = bend / 1.1
        if bend_factor < 0.0:
            bend_factor = 0.0
        elif bend_factor > 1.0:
            bend_factor = 1.0

        # Default behavior: keep your current lateral position (no centering).
        max_lat = 0.98 * float(corridor)
        if self._lat_setpoint < -max_lat:
            self._lat_setpoint = -max_lat
        elif self._lat_setpoint > max_lat:
            self._lat_setpoint = max_lat

        # Gently track the current lateral offset to avoid stale setpoints.
        hold_alpha = float(self.lat_setpoint_hold_alpha)
        if hold_alpha < 0.0:
            hold_alpha = 0.0
        elif hold_alpha > 1.0:
            hold_alpha = 1.0
        self._lat_setpoint = (1.0 - hold_alpha) * float(self._lat_setpoint) + hold_alpha * float(lat_off)

        # Edge containment: if we're getting too close to the corridor edge, nudge the
        # setpoint back inward (still not toward center unless necessary).
        edge_soft = 0.92 * float(corridor)
        if abs_off > edge_soft and corridor > 1e-6:
            over = (abs_off - edge_soft) / (float(corridor) - edge_soft + 1e-9)
            if over < 0.0:
                over = 0.0
            elif over > 1.0:
                over = 1.0
            inward = (1.0 - over) * abs_off + over * edge_soft
            self._lat_setpoint = math.copysign(inward, float(lat_off))

        desired_lat_off = float(self._lat_setpoint)
        racing_line_enable = bool(self.racing_line_enable)
        using_racing_line = bool(racing_line_enable and bend_factor > 0.02 and bend_sign != 0.0)
        if using_racing_line:
            outside_lat = (-float(bend_sign)) * (self.racing_line_entry_frac * corridor)
            inside_lat = (float(bend_sign)) * (self.racing_line_apex_frac * corridor)

            he_entry = 0.58
            he_apex = 0.98
            if abs_he <= he_entry:
                phase_he = 0.0
            elif abs_he >= he_apex:
                phase_he = 1.0
            else:
                phase_he = (abs_he - he_entry) / (he_apex - he_entry)

            bf_entry = 0.48
            bf_apex = 0.94
            if bend_factor <= bf_entry:
                phase_bf = 0.0
            elif bend_factor >= bf_apex:
                phase_bf = 1.0
            else:
                phase_bf = (bend_factor - bf_entry) / (bf_apex - bf_entry)

            # Hairpins: don't let curvature alone force an early apex.
            if bend_factor > 0.65:
                phase_bf *= 0.65
            else:
                phase_bf *= 0.96

            # Combine heading- and curvature-based phase with the more advanced one.
            phase_t = phase_he if phase_he > phase_bf else phase_bf

            desired_lat_off = outside_lat * (1.0 - phase_t) + inside_lat * phase_t

            if bend_factor > 0.55 and spd < 14.0:
                late_entry = 0.94
                late_apex = 1.10
                if abs_he <= late_entry:
                    late_t = 0.0
                elif abs_he >= late_apex:
                    late_t = 1.0
                else:
                    late_t = (abs_he - late_entry) / (late_apex - late_entry)
                desired_lat_off = outside_lat * (1.0 - late_t) + desired_lat_off * late_t

            # Keep racing-line desire within bounds, but never collapse back to center.
            max_off = min(float(self.racing_line_max_frac) * float(corridor), 0.98 * float(corridor))
            if desired_lat_off < -max_off:
                desired_lat_off = -max_off
            elif desired_lat_off > max_off:
                desired_lat_off = max_off

            # Blend setpoint toward the racing-line request.
            turn_alpha = float(self.lat_setpoint_turn_alpha)
            if turn_alpha < 0.0:
                turn_alpha = 0.0
            elif turn_alpha > 1.0:
                turn_alpha = 1.0
            self._lat_setpoint = (1.0 - turn_alpha) * float(self._lat_setpoint) + turn_alpha * float(desired_lat_off)
            desired_lat_off = float(self._lat_setpoint)

        if using_racing_line and desired_lat_off != 0.0:
            if len(self._seg_norm) > 0:
                nrm = self._seg_norm[self.current_idx]
                look_pt = look_pt + nrm * float(desired_lat_off)

        self.last_lookahead_point = (float(look_pt[0]), float(look_pt[1]))

        cross_track_err = -float(lat_off - desired_lat_off)
        stanley = math.atan2(self.stanley_k * cross_track_err, (spd + self.stanley_v0))

        ff_gain = float(self.curvature_ff_gain_line) if using_racing_line else float(self.curvature_ff_gain)
        ff = ff_gain * float(bend_sign) * float(bend_factor)

        lat_norm = float(lat_off) / float(corridor)
        if lat_norm < -1.0:
            lat_norm_clamped = -1.0
        elif lat_norm > 1.0:
            lat_norm_clamped = 1.0
        else:
            lat_norm_clamped = lat_norm

        center_term = 0.0
        contain_term = 0.0
        if abs_off > corridor:
            outside_denom = float(half_width - corridor)
            if outside_denom < 1e-6:
                outside_weight = 1.0
            else:
                outside_weight = (abs_off - corridor) / outside_denom
                if outside_weight < 0.0:
                    outside_weight = 0.0
                elif outside_weight > 1.0:
                    outside_weight = 1.0
            contain_term = -float(self.containment_gain_rad) * lat_norm_clamped * outside_weight

        delta_cmd = heading_err + stanley + ff + contain_term
        steering = delta_cmd / (self.nominal_max_steer_rad if self.nominal_max_steer_rad > 1e-6 else 1.0)
        if steering < -1.0:
            steering = -1.0
        elif steering > 1.0:
            steering = 1.0


        speed_floor = max(1.0, 0.60 * self.min_speed)

        if abs_off <= corridor:
            desired_speed_dist = self.max_speed
        else:
            hard_brake_off = max(corridor * 2.2, corridor + 1.0)
            t = (abs_off - corridor) / (hard_brake_off - corridor)
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
            desired_speed_dist = self.max_speed * (1.0 - t) + self.min_speed * t

        ax = float(near_pt[0] - proj[0])
        ay = float(near_pt[1] - proj[1])
        cx = float(far_pt[0] - proj[0])
        cy = float(far_pt[1] - proj[1])
        bx = float(far_pt[0] - near_pt[0])
        by = float(far_pt[1] - near_pt[1])
        ab = math.hypot(ax, ay)
        bc = math.hypot(bx, by)
        ca = math.hypot(cx, cy)
        cross = abs(ax * cy - ay * cx)
        if cross <= 1e-6 or ab <= 1e-6 or bc <= 1e-6 or ca <= 1e-6:
            desired_speed_turn = float(self.max_speed)
        else:
            # Circumcircle radius from (proj, near, far) controls corner speed.
            radius = (ab * bc * ca) / (2.0 * cross)
            desired_speed_turn = math.sqrt(max(0.0, float(self.lat_accel_max)) * float(radius))
        if desired_speed_turn > float(self.max_speed):
            desired_speed_turn = float(self.max_speed)
        if desired_speed_turn < float(self.turn_speed):
            desired_speed_turn = float(self.turn_speed)

        turning_into_bend = (bend_sign != 0.0) and ((float(bend_sign) * float(heading_err)) > 0.03) and (abs_he > 0.06)
        non_hairpin = bend_factor < 0.55

        # Corner slowdown: keep it conservative on real turns, but avoid unnecessary
        # braking on very soft bends.
        if bend_factor < 0.28:
            desired_speed_turn *= (1.0 - 0.60 * float(bend_factor))
        else:
            desired_speed_turn *= (1.0 - 0.70 * float(bend_factor))
        if non_hairpin and (not turning_into_bend) and (abs_he < 0.20) and (abs_off < 0.70 * corridor):
            desired_speed_turn *= 1.03
        if desired_speed_turn < float(self.turn_speed):
            desired_speed_turn = float(self.turn_speed)

        if corridor > 1e-6 and abs_off > 0.60 * corridor:
            edge_ratio = abs_off / corridor
            t_edge = (edge_ratio - 0.60) / 0.40
            if t_edge < 0.0:
                t_edge = 0.0
            elif t_edge > 1.0:
                t_edge = 1.0
            desired_speed_turn *= (1.0 - 0.74 * t_edge)
            if desired_speed_turn < float(self.min_speed):
                desired_speed_turn = float(self.min_speed)

        desired_speed = desired_speed_dist
        if desired_speed_turn < desired_speed:
            desired_speed = desired_speed_turn
        if desired_speed < speed_floor:
            desired_speed = speed_floor

        if spd < 11.0 and abs_he > 0.25:
            # Extra slowdown for tight low-speed corners to protect exit width.
            hairpin_speed_t = (11.0 - spd) / 7.0
            if hairpin_speed_t < 0.0:
                hairpin_speed_t = 0.0
            elif hairpin_speed_t > 1.0:
                hairpin_speed_t = 1.0
            hairpin_heading_t = (abs_he - 0.25) / 0.45
            if hairpin_heading_t < 0.0:
                hairpin_heading_t = 0.0
            elif hairpin_heading_t > 1.0:
                hairpin_heading_t = 1.0
            hairpin_t = hairpin_speed_t if hairpin_speed_t > hairpin_heading_t else hairpin_heading_t
            desired_speed *= (1.0 - 0.64 * hairpin_t)
            if desired_speed < float(self.min_speed):
                desired_speed = float(self.min_speed)

        abs_steer = abs(float(steering))
        if abs_steer > 0.85:
            t_steer = (abs_steer - 0.85) / 0.15
            if t_steer < 0.0:
                t_steer = 0.0
            elif t_steer > 1.0:
                t_steer = 1.0
        else:
            t_steer = 0.0

        # Steering-based speed penalty. We keep it for safety, but reduce it on soft
        # bends when we're clearly unwinding (helps earlier exit acceleration).
        if abs_steer > 0.85:
            steer_pen = 0.35
            if (bend_factor < 0.35) and (not turning_into_bend) and abs_he < 0.22:
                steer_pen = 0.20
            if non_hairpin and (not turning_into_bend) and abs_he < 0.20:
                steer_pen = 0.18
            desired_speed *= (1.0 - steer_pen * t_steer)
            if desired_speed < float(self.min_speed):
                desired_speed = float(self.min_speed)

        # Exit recovery: once we are unwinding out of a bend, ramp target speed up
        # aggressively to start throttle application much earlier.
        exit_recover = (
            (not turning_into_bend)
            and non_hairpin
            and (abs_he < 0.18)
            and (bend_factor < 0.48)
            and (abs_steer < 0.90)
            and (abs_off < 0.75 * corridor)
        )
        if exit_recover:
            straight_t = 1.0 - (abs_he / 0.18)
            if straight_t < 0.0:
                straight_t = 0.0
            elif straight_t > 1.0:
                straight_t = 1.0
            unwind_t = 1.0 - (bend_factor / 0.48)
            if unwind_t < 0.0:
                unwind_t = 0.0
            elif unwind_t > 1.0:
                unwind_t = 1.0
            exit_t = 0.65 * straight_t + 0.35 * unwind_t
            min_exit_speed = float(self.max_speed) * (0.60 + 0.18 * exit_t)
            if desired_speed < min_exit_speed:
                desired_speed = min_exit_speed

        throttle = (desired_speed - float(speed)) / (self.max_speed if self.max_speed > 1.0 else 1.0)

        if throttle < 0.0:
            v_now = float(speed)
            v_norm = v_now / (self.max_speed if self.max_speed > 1.0 else 1.0)
            if v_norm < 0.0:
                v_norm = 0.0
            elif v_norm > 1.0:
                v_norm = 1.0
            bend_brake = float(bend_factor) ** 1.35
            brake_scale = 1.0 + 2.05 * bend_brake * v_norm
            if non_hairpin and (not turning_into_bend):
                brake_scale *= 0.88
            throttle *= brake_scale

            # Prevent "exit braking": don't amplify braking just because steering is
            # still unwinding after the apex.
            if abs_steer > 0.85 and turning_into_bend:
                throttle *= (1.0 + 1.25 * t_steer)

            # On corner exit, strongly damp residual braking so throttle resumes earlier.
            if exit_recover:
                throttle *= 0.50
                if throttle < -0.22:
                    throttle = -0.22
        if throttle < -1.0:
            throttle = -1.0
        elif throttle > 1.0:
            throttle = 1.0

        return {'steering': float(steering), 'throttle': float(throttle)}
