"""Precomputed geometry for a closed racetrack centerline.

Takes a centerline polyline (map units, already scaled by track_loader),
resamples it to uniform spacing and precomputes everything the RL
environment asks about every step:

- projection of the car position onto the centerline (with a warm-start
  index so it is a local search, not a full scan)
- arc-length position and signed lateral offset
- signed curvature at each centerline point (positive = left turn)
- curvature sampled at arc distances ahead of the car (the "what is
  coming" part of the observation)

Also provides reversed and mirrored variants for training augmentation.
"""

import numpy as np


class TrackGeometry:
    def __init__(self, points, track_width, map_size, spacing=3.0):
        pts = np.asarray(points, dtype=float)
        # Drop an explicit closing point; the ring wraps implicitly.
        if np.linalg.norm(pts[-1] - pts[0]) < 1e-6:
            pts = pts[:-1]
        self.track_width = float(track_width)
        self.half_width = 0.5 * self.track_width
        self.map_size = float(map_size)
        self.spacing = float(spacing)

        self.points = self._resample_closed(pts, self.spacing)
        self.n = len(self.points)

        nxt = np.roll(self.points, -1, axis=0)
        seg = nxt - self.points
        self.seg_len = np.linalg.norm(seg, axis=1)
        self.seg_len = np.maximum(self.seg_len, 1e-9)
        self.seg_tan = seg / self.seg_len[:, None]
        self.cum_s = np.concatenate([[0.0], np.cumsum(self.seg_len)])[:-1]
        self.length = float(np.sum(self.seg_len))
        self.curvature = self._signed_curvature()

    # ── construction helpers ──────────────────────────────────────────────

    @staticmethod
    def _resample_closed(pts, spacing):
        """Uniform arc-length resample of a closed polyline."""
        ring = np.vstack([pts, pts[0]])
        seg = np.linalg.norm(np.diff(ring, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        total = arc[-1]
        n = max(int(round(total / spacing)), 16)
        # endpoint=False keeps the ring open (no duplicated closing point)
        targets = np.linspace(0.0, total, n, endpoint=False)
        x = np.interp(targets, arc, ring[:, 0])
        y = np.interp(targets, arc, ring[:, 1])
        return np.column_stack([x, y])

    def _signed_curvature(self):
        """Menger curvature at every point, sign from the turn direction
        (positive = turning left when driving in point order)."""
        p0 = np.roll(self.points, 1, axis=0)
        p1 = self.points
        p2 = np.roll(self.points, -1, axis=0)
        a = p1 - p0
        b = p2 - p1
        c = p2 - p0
        cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        la = np.linalg.norm(a, axis=1)
        lb = np.linalg.norm(b, axis=1)
        lc = np.linalg.norm(c, axis=1)
        denom = np.maximum(la * lb * lc, 1e-9)
        return 2.0 * cross / denom

    # ── augmentation variants ─────────────────────────────────────────────

    def reversed(self):
        """Same track, driven the other way around."""
        return TrackGeometry(self.points[::-1], self.track_width,
                             self.map_size, self.spacing)

    def mirrored(self):
        """Track flipped left-right about the map center (all left turns
        become right turns)."""
        pts = self.points.copy()
        pts[:, 0] = self.map_size - pts[:, 0]
        return TrackGeometry(pts, self.track_width, self.map_size, self.spacing)

    # ── per-step queries ──────────────────────────────────────────────────

    def project(self, pos, hint_idx=None):
        """Project a position onto the centerline.

        Returns (seg_idx, s, lateral_offset, heading_error_ref) where
        s is the arc-length position, lateral_offset is signed (positive =
        left of the centerline in driving direction) and heading_error_ref
        is the tangent angle of the matched segment.

        With a hint index only a local window of segments is searched;
        without one the whole ring is scanned (used at reset).
        """
        pos = np.asarray(pos, dtype=float)

        def _match(idx):
            base = self.points[idx]
            tan = self.seg_tan[idx]
            rel = pos[None, :] - base
            t = np.clip(np.einsum("ij,ij->i", rel, tan) / self.seg_len[idx], 0.0, 1.0)
            proj = base + tan * (t * self.seg_len[idx])[:, None]
            d2 = np.einsum("ij,ij->i", pos[None, :] - proj, pos[None, :] - proj)
            k = int(np.argmin(d2))
            return idx, t, proj, d2, k

        if hint_idx is None:
            idx, t, proj, d2, k = _match(np.arange(self.n))
        else:
            # Window centered on the hint. It must be wide enough to hold a
            # full step of travel in BOTH directions (the projection can fall
            # behind as well as jump ahead), so it is symmetric.
            idx = (np.arange(hint_idx - 30, hint_idx + 30)) % self.n
            idx, t, proj, d2, k = _match(idx)
            # If the best local match still sits implausibly far from the
            # centerline, the window missed the true segment (a stale hint
            # after a fast stretch). Fall back to a full scan so a lateral
            # offset is never fabricated from the wrong segment.
            if d2[k] > (4.0 * self.track_width) ** 2:
                idx, t, proj, d2, k = _match(np.arange(self.n))

        seg_idx = int(idx[k])
        t_best = float(t[k])
        s = float(self.cum_s[seg_idx] + t_best * self.seg_len[seg_idx])
        tangent = self.seg_tan[seg_idx]
        rel_best = pos - proj[k]
        # 2D cross product of tangent x (pos - proj): positive = left side
        lateral = float(tangent[0] * rel_best[1] - tangent[1] * rel_best[0])
        heading_ref = float(np.arctan2(tangent[1], tangent[0]))
        return seg_idx, s, lateral, heading_ref

    def curvature_at_s(self, s):
        """Curvature at (wrapped) arc position s, nearest-point lookup."""
        i = int(round((s % self.length) / self.spacing)) % self.n
        return float(self.curvature[i])

    def curvature_ahead(self, s, edges):
        """Worst signed curvature in each arc band ahead of position s.

        `edges` are band boundaries in metres: band i covers
        [edges[i], edges[i+1]) ahead, and the value reported is the
        curvature with the largest magnitude (keeping its sign) among the
        centerline points inside the band. Point sampling instead would let
        a short corner hide between two sample distances; a band never
        misses one. Returns len(edges) - 1 values.
        """
        out = np.empty(len(edges) - 1)
        i0 = int(round((s % self.length) / self.spacing))
        for b in range(len(edges) - 1):
            # Offsets are relative to i0; keep them relative until the end.
            # (A previous version compared the upper offset against the
            # ABSOLUTE lower index, which made every band scan to roughly
            # twice the car's position: every band saw every corner within
            # 2*s, phantom hairpins pinned the speed target everywhere, and
            # the scripted driver crawled at ~9 m/s on every circuit.)
            lo = int(np.floor(edges[b] / self.spacing))
            hi = max(int(np.ceil(edges[b + 1] / self.spacing)), lo + 1)
            idx = np.arange(i0 + lo, i0 + hi) % self.n
            band = self.curvature[idx]
            out[b] = band[int(np.argmax(np.abs(band)))]
        return out

    def pose_at_s(self, s):
        """Position and tangent heading at arc position s (used for spawn)."""
        s = s % self.length
        i = int(np.searchsorted(self.cum_s, s, side="right") - 1)
        i = max(0, min(i, self.n - 1))
        t = (s - self.cum_s[i]) / self.seg_len[i]
        pos = self.points[i] + self.seg_tan[i] * (t * self.seg_len[i])
        heading = float(np.arctan2(self.seg_tan[i][1], self.seg_tan[i][0]))
        return pos, heading

    def delta_s(self, s_new, s_old):
        """Signed forward progress from s_old to s_new, wrap-aware."""
        return (s_new - s_old + 0.5 * self.length) % self.length - 0.5 * self.length
