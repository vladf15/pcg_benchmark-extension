import math
import numpy as np
from .utils import get_racing_line_targets

class SimpleAgent:
    """Waypoint-following agent that tracks entry/apex/exit targets per segment."""

    def __init__(self, curve_points, track_width=50, line_strategy=None):
        self.curve_points = curve_points
        self.track_width = track_width
        self.current_segment_idx = 0
        self.current_phase = 'entry'
        if line_strategy is None:
            self.line_strategy = {'entry': 'center', 'apex': 'center', 'exit': 'center'}
        else:
            self.line_strategy = line_strategy
        self.targets = get_racing_line_targets(curve_points, track_width)
        self._apex_turn_factors = self._compute_apex_turn_factors()

    def _compute_apex_turn_factors(self):
        """Precompute per-segment throttle scaling based on turn angle."""
        factors = [1.0] * len(self.targets)
        for seg_idx in range(1, len(self.targets) - 1):
            prev_apex = self.targets[seg_idx - 1]['apex']['center']
            curr_apex = self.targets[seg_idx]['apex']['center']
            next_apex = self.targets[seg_idx + 1]['apex']['center']
            v1 = curr_apex - prev_apex
            v2 = next_apex - curr_apex
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 > 1e-3 and norm2 > 1e-3:
                v1 /= norm1
                v2 /= norm2
                turn_angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
                factors[seg_idx] = max(0.3, 1.0 - turn_angle/np.pi)
        return factors

    def reset(self):
        """Reset agent progress along the target sequence."""
        self.current_segment_idx = 0
        self.current_phase = 'entry'

    def act(self, car_state):
        """Compute steering/throttle toward the current phase target."""
        x, y, angle, velocity, steering_angle = car_state
        seg_idx = self.current_segment_idx
        phase = self.current_phase
        if seg_idx >= len(self.targets):
            return {'steering': 0.0, 'throttle': 0.0}
        target = self.targets[seg_idx][phase][self.line_strategy[phase]]
        dx = target[0] - x
        dy = target[1] - y
        distance = math.hypot(float(dx), float(dy))
        threshold = 12.0
        if distance < threshold:
            if phase == 'entry':
                self.current_phase = 'apex'
            elif phase == 'apex':
                self.current_phase = 'exit'
            elif phase == 'exit':
                self.current_phase = 'entry'
                self.current_segment_idx += 1
        angle_to_target = math.atan2(float(dy), float(dx))
        steering = angle_to_target - angle
        steering = (steering + np.pi) % (2 * np.pi) - np.pi
        # Normalize steering to [-1, 1] range (engine expects this)
        steering = np.clip(steering / np.pi, -1.0, 1.0)
        turn_factor = 1.0
        if phase == 'apex' and 0 <= seg_idx < len(self._apex_turn_factors):
            turn_factor = self._apex_turn_factors[seg_idx]
        throttle = turn_factor if phase == 'apex' else 1.0
        return {'steering': steering, 'throttle': throttle}


class SteeringAgent:
    """Path-following racing agent with Stanley-style steering.

    Core idea (cheap, stable, widely used):
    - Predict a short time into the future.
    - Project onto the centerline polyline (monotonic progress).
    - Choose a lookahead point with distance increasing with speed.
    - Compute steering with a Stanley controller:
        `delta = heading_error + atan(k * cross_track_error / (v + v0))`
      plus a small curvature feedforward term based on upcoming bend.

    This replaces the older Reynolds force-blending steering logic while keeping
    the same projection/indexing machinery and runtime profile.

    Main tuning knobs:
    - `lookahead_base`, `lookahead_speed_gain`: speed-dependent lookahead distance.
    - `stanley_k`: cross-track correction gain.
    - `stanley_v0`: low-speed stabilizer (prevents huge correction when v≈0).
    - `curvature_ff_gain`: small signed feedforward into upcoming bends.
    """

    def __init__(
        self,
        curve_points,
        track_width=50,
        line_strategy=None,
        max_speed=22.0,
        enable_wander=False,
    ):
        """
        Args:
            curve_points: Track control points or interpolated curve
            track_width: Width of track
            line_strategy: Dict specifying racing line preferences
            max_speed: Maximum velocity of agent (reduced for stability)
            enable_wander: Whether to add wander behavior for variety
        """
        self.path_points = np.array(curve_points, dtype=float)
        self.track_width = float(track_width)
        self.max_speed = float(max_speed)
        self.enable_wander = bool(enable_wander)

        # Prediction + projection parameters
        # Keep lookahead modest even at speed; too much lookahead makes the agent
        # "cut corners" and ignore corridor containment.
        self.look_ahead_dist = 35.0
        self.curvature_lookahead = 75.0
        self.predict_time_base = 0.75
        self.predict_time_speed_gain = 0.04
        self.search_ahead = 40
        self.search_back = 0

        # Corridor containment
        self.corridor_margin = 6.0
        # Mild centering always, strong correction only when outside corridor.
        self.centering_gain_rad = math.radians(3.0)
        self.containment_gain_rad = math.radians(18.0)

        # Racing-line bias (very lightweight): shift the lookahead target laterally.
        # Uses path normal; positive offset is to the left of the centerline.
        # Heuristic phases are inferred from bend strength + heading error.
        self.racing_line_max_frac = 0.85
        self.racing_line_entry_frac = 0.70  # outside before apex
        self.racing_line_apex_frac = 0.35   # inside near apex
        self.racing_line_enable = True

        # Stanley steering parameters
        # Steering output is normalized to [-1, 1] and interpreted by the engine.
        # Use the nominal high-speed steering limit (engine blends up at low speed).
        self.nominal_max_steer_rad = math.radians(28.0)
        self.stanley_k = 0.035
        self.stanley_v0 = 1.5
        self.curvature_ff_gain = 0.35
        # When following a biased racing line (outside->apex), do not over-rotate early.
        self.curvature_ff_gain_line = 0.22

        # Lookahead distance as a function of speed (keeps high-speed stable)
        self.lookahead_base = float(self.look_ahead_dist)
        self.lookahead_speed_gain = 1.2
        self.lookahead_min = 18.0
        self.lookahead_max = 110.0

        # Speed planning (path-aware)
        self.min_speed = 4.0
        self.turn_speed = 9.0

        # Monotonic progress along the polyline
        self.current_idx = 0

        self._precompute_path_geometry()

        # Optional profiling (disabled by default)
        self._profile_enabled = False
        self._profile_accum = {
            'calls': 0,
            'predict': 0.0,
            'project': 0.0,
            'lookahead': 0.0,
            'forces': 0.0,
            'convert': 0.0,
            'total': 0.0,
        }
    
    def _precompute_path_geometry(self):
        """Precompute segment vectors, lengths, unit tangents and normals."""
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

        # Arc-length parameterization helpers
        seg_s0 = np.zeros((len(seg_len),), dtype=float)
        if len(seg_len) > 0:
            seg_s0[1:] = np.cumsum(seg_len[:-1])
        cum_s = np.concatenate(([0.0], np.cumsum(seg_len)))

        self._seg_vec = seg
        self._seg_len = seg_len
        self._seg_tan = seg_tan
        self._seg_norm = seg_norm

        # Cache for fast projection queries
        self._seg_a = seg_a
        self._seg_b = seg_b
        self._seg_ab = seg
        self._seg_ab2 = seg_ab2
        self._seg_s0 = seg_s0
        self._cum_s = cum_s

        # Scratch arrays reused per act() call (avoid allocations)
        m = len(seg_ab2)
        self._scratch_t = np.zeros((m,), dtype=float)
        self._scratch_d2 = np.zeros((m,), dtype=float)

        # Scratch scalars/vectors reused in act()
        self._scratch_pos = np.zeros((2,), dtype=float)
        self._scratch_proj = np.zeros((2,), dtype=float)
        self._scratch_pred = np.zeros((2,), dtype=float)

        # Scratch velocity (world frame)
        self._scratch_vel = np.zeros((2,), dtype=float)
    
    def reset(self):
        """Reset agent to start of path."""
        self.current_idx = 0

    def _find_projection(self, point, start_idx):
        """Fast projection of point onto polyline near start_idx.

        Vectorized over a window of segments.
        Returns: (seg_idx, proj_point)
        """
        nseg = len(self.path_points) - 1
        if nseg <= 0:
            return 0, np.asarray(point, dtype=float)

        # Search only locally around the current progress index (monotonic path following)
        i0 = int(max(0, min(start_idx - int(self.search_back), nseg - 1)))
        i1 = int(min(nseg - 1, max(i0, start_idx) + int(self.search_ahead)))
        sl = slice(i0, i1 + 1)

        p = np.asarray(point, dtype=float)

        a = self._seg_a[sl]
        ab = self._seg_ab[sl]
        ab2 = self._seg_ab2[sl]
        t = self._scratch_t[sl]
        d2 = self._scratch_d2[sl]

        # Compute projection parameter t without forming (p-a) matrix
        # dot(p-a,ab) = dot(p,ab) - dot(a,ab)
        # numerator = dot(p,ab) - dot(a,ab)
        pab = self._scratch_d2[sl]  # reuse d2 slice as numerator scratch
        pab[:] = p[0] * ab[:, 0] + p[1] * ab[:, 1]
        pab -= (a[:, 0] * ab[:, 0] + a[:, 1] * ab[:, 1])
        denom = np.where(ab2 > 1e-12, ab2, 1.0)
        t[:] = pab / denom
        np.clip(t, 0.0, 1.0, out=t)

        # proj = a + ab*t; d2 = ||p-proj||^2 computed component-wise
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
        """Return point `distance` ahead along polyline from a point on seg_idx.

        Uses precomputed arc-length for O(log N) lookup.
        """
        pts = self.path_points
        nseg = len(pts) - 1
        if nseg <= 0:
            return np.asarray(from_point, dtype=float)

        i = int(max(0, min(seg_idx, nseg - 1)))
        p = np.asarray(from_point, dtype=float)

        # Compute current arc-length s at the given point (projected onto segment i)
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

        # Clamp to end of path
        total_len = float(self._cum_s[-1]) if len(self._cum_s) else 0.0
        if s_target >= total_len:
            return pts[-1].copy()

        # Find segment containing s_target
        j = int(np.searchsorted(self._cum_s, s_target, side='right') - 1)
        j = int(max(0, min(j, nseg - 1)))
        ds = s_target - float(self._seg_s0[j])
        seg_len = float(self._seg_len[j])
        if seg_len < 1e-9:
            return pts[j].copy()
        u = ds / seg_len
        return self._seg_a[j] + self._seg_ab[j] * u

    def _signed_lateral_offset(self, point, seg_idx):
        """Signed distance from point to local centerline using segment normal."""
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
        """Compute steering/throttle from the current car state.

        Steering is a Stanley controller with:
        - heading error to the path tangent (from projection -> lookahead point)
        - cross-track error (signed lateral offset to centerline)
        - small curvature feedforward from upcoming bend

        Throttle/brake is planned from distance-to-track plus upcoming curvature,
        with braking strengthened for sharp turns at high speed.

        Args:
            car_state: [x, y, angle, speed, steering_angle]

        Returns:
            {'steering': float in [-1, 1], 'throttle': float in [-1, 1]}
        """
        t_total0 = None
        if self._profile_enabled:
            import time
            t_total0 = time.perf_counter()

        x, y, angle, speed, steering_angle = car_state
        pos = self._scratch_pos
        pos[0] = float(x)
        pos[1] = float(y)
        cos_a = math.cos(float(angle))
        sin_a = math.sin(float(angle))
        vel = self._scratch_vel
        spd = float(speed)
        vel[0] = cos_a * spd
        vel[1] = sin_a * spd

        pts = self.path_points
        if len(pts) < 2:
            return {'steering': 0.0, 'throttle': 0.0}

        # Predict future position (Reynolds-style)
        if self._profile_enabled:
            import time
            t0 = time.perf_counter()
        predict_t = self.predict_time_base + self.predict_time_speed_gain * float(speed)
        predicted = self._scratch_pred
        predicted[0] = pos[0] + vel[0] * predict_t
        predicted[1] = pos[1] + vel[1] * predict_t
        if self._profile_enabled:
            self._profile_accum['predict'] += (time.perf_counter() - t0)

        # Find projection of predicted position onto the path (search forward only)
        if self._profile_enabled:
            import time
            t0 = time.perf_counter()
        seg_idx, proj = self._find_projection(predicted, start_idx=self.current_idx)
        if self._profile_enabled:
            self._profile_accum['project'] += (time.perf_counter() - t0)

        # Enforce monotonic progress
        if seg_idx < self.current_idx:
            seg_idx = self.current_idx
        self.current_idx = int(seg_idx)

        # Cross-track error from predicted position (used for containment + speed planning).
        half_width = self.track_width * 0.5
        corridor = max(1.0, half_width - self.corridor_margin)
        lat_off = self._signed_lateral_offset(predicted, self.current_idx)
        abs_off = abs(lat_off)

        # Choose a speed-dependent look-ahead target along the path from the projection.
        if self._profile_enabled:
            import time
            t0 = time.perf_counter()
        lookahead = self.lookahead_base + self.lookahead_speed_gain * spd
        # If we're near/outside corridor, reduce lookahead so we prioritize re-entry.
        if abs_off > corridor:
            lookahead *= 0.55
        elif abs_off > 0.85 * corridor:
            lookahead *= 0.70
        if lookahead < self.lookahead_min:
            lookahead = self.lookahead_min
        elif lookahead > self.lookahead_max:
            lookahead = self.lookahead_max
        look_pt = self._point_at_distance_ahead(self.current_idx, proj, lookahead)
        if self._profile_enabled:
            self._profile_accum['lookahead'] += (time.perf_counter() - t0)

        if self._profile_enabled:
            import time
            t0 = time.perf_counter()

        # (half_width/corridor/lat_off already computed above)

        # Path heading from projection -> lookahead point.
        t_dx = float(look_pt[0] - proj[0])
        t_dy = float(look_pt[1] - proj[1])
        t2 = t_dx * t_dx + t_dy * t_dy
        if t2 > 1e-12:
            path_heading = math.atan2(t_dy, t_dx)
        else:
            tan = self._seg_tan[self.current_idx] if len(self._seg_tan) > 0 else (1.0, 0.0)
            path_heading = math.atan2(float(tan[1]), float(tan[0]))

        # Heading error to tangent.
        heading_err = path_heading - float(angle)
        heading_err = (heading_err + math.pi) % (2 * math.pi) - math.pi

        # Upcoming bend (for feedforward + braking). Signed by cross product.
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

        # Stanley cross-track term: use predicted cross-track error for anticipation.
        # Note sign: lat_off>0 means left of centerline (by our normal), so steer right.
        desired_lat_off = 0.0
        if self.racing_line_enable and bend_factor > 0.12 and bend_sign != 0.0:
            # bend_sign>0: left turn ahead (CCW). Outside entry is to the right (negative lat).
            outside_lat = (-float(bend_sign)) * (self.racing_line_entry_frac * corridor)
            inside_lat = (float(bend_sign)) * (self.racing_line_apex_frac * corridor)

            # Use heading error magnitude as a cheap proxy for entry->apex timing.
            # Blend smoothly so we don't snap to apex too early (reduces corner cutting).
            abs_he = abs(float(heading_err))
            he_entry = 0.18
            he_apex = 0.55
            if abs_he <= he_entry:
                phase_t = 0.0
            elif abs_he >= he_apex:
                phase_t = 1.0
            else:
                phase_t = (abs_he - he_entry) / (he_apex - he_entry)

            desired_lat_off = outside_lat * (1.0 - phase_t) + inside_lat * phase_t

            max_off = float(self.racing_line_max_frac) * float(corridor)
            if desired_lat_off < -max_off:
                desired_lat_off = -max_off
            elif desired_lat_off > max_off:
                desired_lat_off = max_off

        # If we are following a biased racing line, shift the steering *target* laterally.
        # Recompute heading error against the shifted target so the agent doesn't cut in.
        if desired_lat_off != 0.0:
            if len(self._seg_norm) > 0:
                nrm = self._seg_norm[self.current_idx]
                look_pt = look_pt + nrm * float(desired_lat_off)

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

        # Cross-track error to the desired racing line (centerline if desired_lat_off==0).
        cross_track_err = -float(lat_off - desired_lat_off)
        stanley = math.atan2(self.stanley_k * cross_track_err, (spd + self.stanley_v0))

        # Small curvature feedforward: start rotating into the bend.
        ff_gain = float(self.curvature_ff_gain_line) if (self.racing_line_enable and desired_lat_off != 0.0) else float(self.curvature_ff_gain)
        ff = ff_gain * float(bend_sign) * float(bend_factor)

        # Corridor containment / centering:
        # - A small always-on centering term reduces wall scraping.
        # - A stronger correction is added only when outside the corridor.
        lat_norm = float(lat_off) / float(corridor)
        if lat_norm < -1.0:
            lat_norm_clamped = -1.0
        elif lat_norm > 1.0:
            lat_norm_clamped = 1.0
        else:
            lat_norm_clamped = lat_norm

        # Keep centering based on actual track centerline so we don't drive off-track
        # when the racing-line bias pushes toward the edges.
        center_term = -float(self.centering_gain_rad) * lat_norm
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

        delta_cmd = heading_err + stanley + ff + center_term + contain_term
        steering = delta_cmd / (self.nominal_max_steer_rad if self.nominal_max_steer_rad > 1e-6 else 1.0)
        if steering < -1.0:
            steering = -1.0
        elif steering > 1.0:
            steering = 1.0

        if self._profile_enabled:
            self._profile_accum['forces'] += (time.perf_counter() - t0)

        # ---------------- Speed planning (distance-to-track) ----------------
        # Per requirement: acceleration/braking should be driven by distance from
        # the projection to the track centerline (lat_off), not heading/angle.
        #
        # Keep a simple policy:
        # - Base target speed comes from distance-to-track (lat_off)
        # - Add proactive braking based on upcoming curvature (brake before turns)
        # - Never request 0 target speed (always keep moving, even out-of-bounds)
        # Speed floor: never stop completely.
        speed_floor = max(1.0, 0.60 * self.min_speed)

        # Distance-to-track based target.
        if abs_off <= corridor:
            desired_speed_dist = self.max_speed
        else:
            hard_brake_off = max(corridor * 3.0, corridor + 1.0)
            t = (abs_off - corridor) / (hard_brake_off - corridor)
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
            # Blend max_speed -> min_speed as we go further out.
            desired_speed_dist = self.max_speed * (1.0 - t) + self.min_speed * t

        # Map bend_factor into a speed target. Start braking early, but keep
        # straights / gentle bends fast.
        if bend_factor <= 0.10:
            turn_scale = 0.0
        elif bend_factor <= 0.25:
            turn_scale = 0.55
        else:
            turn_scale = 0.78
        desired_speed_turn = self.max_speed * (1.0 - turn_scale * bend_factor)
        if desired_speed_turn < self.turn_speed:
            desired_speed_turn = self.turn_speed

        # Final target speed respects both distance-to-track and upcoming turns.
        desired_speed = desired_speed_dist
        if desired_speed_turn < desired_speed:
            desired_speed = desired_speed_turn
        if desired_speed < speed_floor:
            desired_speed = speed_floor

        # Convert speed error into throttle/brake command in [-1, 1].
        throttle = (desired_speed - float(speed)) / (self.max_speed if self.max_speed > 1.0 else 1.0)

        # Extra pre-turn braking: stronger braking when the turn is sharper AND we're faster.
        # This biases braking earlier without relying on heading/angle caps.
        if throttle < 0.0:
            v_now = float(speed)
            v_norm = v_now / (self.max_speed if self.max_speed > 1.0 else 1.0)
            if v_norm < 0.0:
                v_norm = 0.0
            elif v_norm > 1.0:
                v_norm = 1.0
            # bend_factor is 0..1 (computed above)
            brake_scale = 1.0 + 1.6 * bend_factor * v_norm
            throttle *= brake_scale
        if throttle < -1.0:
            throttle = -1.0
        elif throttle > 1.0:
            throttle = 1.0

        if self._profile_enabled:
            self._profile_accum['convert'] += (time.perf_counter() - t0)
            self._profile_accum['calls'] += 1
            self._profile_accum['total'] += (time.perf_counter() - t_total0)

        return {'steering': steering, 'throttle': throttle}

    def get_profile_breakdown(self, reset=False):
        """Return accumulated internal timing breakdown for act()."""
        out = dict(self._profile_accum)
        if reset:
            for k in self._profile_accum:
                self._profile_accum[k] = 0.0 if k != 'calls' else 0
        return out