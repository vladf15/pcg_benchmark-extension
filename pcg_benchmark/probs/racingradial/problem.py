from __future__ import annotations

from pcg_benchmark.probs.racing.problem import RacingProblem
from pcg_benchmark.spaces import ArraySpace, FloatSpace, DictionarySpace
import numpy as np


class RacingRadialProblem(RacingProblem):
    """Radial (polar) track representation, after Togelius, De Nardi and
    Lucas (2007), "Towards Automatic Personalised Content Creation for
    Racing Games".

    Genome = num_points polar control points around the map centre, each a
    pair of fractions in [0, 1]:

        polar_points[i] = (angle_fraction, radius_fraction)

    Decoding sorts the points by angle, then places each at

        centre + radius * (cos(angle), sin(angle))

    with angle = angle_fraction * 2*pi and radius linearly mapped into
    [radius_min, radius_max].  Because the control points are visited in
    angle order, the control polygon winds once around the centre and can
    never cross itself: validity comes from the parameterization, not from
    a repair step.  The price is expressiveness: only star-shaped tracks
    (every point visible from the centre) can be represented.

    Everything downstream of decoding (Catmull-Rom spline, quality terms,
    simulation, diversity, controlability) is inherited unchanged from
    RacingProblem, so the parameterization is the only difference between
    racing-v0 and racingradial-v0.
    """

    _render_desc = 'Rendering radial frames'

    # Points are decoded in angle order (already non-self-crossing), so the
    # base class's 2-opt untangle must not reorder them.
    _untangle_control_points = False

    def __init__(self, **kwargs):
        radius_min = kwargs.pop('radius_min', None)
        radius_max = kwargs.pop('radius_max', None)
        # A star of radii needs more spokes than a free tour of points needs
        # corners to fill the same box: at 14 points this representation makes
        # a 3220 m lap, below the shared band, where the free spline made
        # 4870 m.
        kwargs.setdefault('num_points', 20)

        super().__init__(**kwargs)

        # Radii are measured from the map centre, so the largest one that keeps
        # every control point inside the shared build box is half the box.
        # radius_min keeps points off the exact centre, where coincident points
        # would produce zero-length segments.
        x0, _y0, x1, _y1 = self._build_box()
        half_map = 0.5 * float(min(self._width, self._height))
        if radius_min is None:
            radius_min = 0.16 * half_map                  # 120 on the 1500 map
        if radius_max is None:
            radius_max = 0.5 * (x1 - x0)                  # 650 on the 1500 map
        self._radius_min = float(radius_min)
        self._radius_max = float(radius_max)
        self._center = np.array([self._width / 2.0, self._height / 2.0])

        self._content_space = DictionarySpace({
            "polar_points": ArraySpace((self.num_points, 2), FloatSpace(0.0, 1.0)),
        })
        # Control space is inherited from RacingProblem (_CONTROL_TARGETS).

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def _decode_polar_points(self, polar_points: np.ndarray) -> np.ndarray:
        """Turn the (num_points, 2) fraction genome into (x, y) track points."""
        fractions = np.asarray(polar_points, dtype=float).reshape(-1, 2)
        if len(fractions) != self.num_points:
            raise ValueError(
                f"polar_points must have shape ({self.num_points}, 2), got {fractions.shape}"
            )
        fractions = np.clip(fractions, 0.0, 1.0)

        # Angle order is what guarantees a non-self-intersecting polygon:
        # argsort gives the row order that visits the points counterclockwise.
        order = np.argsort(fractions[:, 0], kind='stable')
        ordered = fractions[order]

        angles = ordered[:, 0] * 2.0 * np.pi
        radii = self._radius_min + ordered[:, 1] * (self._radius_max - self._radius_min)

        points = np.empty((len(angles), 2), dtype=float)
        points[:, 0] = self._center[0] + radii * np.cos(angles)
        points[:, 1] = self._center[1] + radii * np.sin(angles)
        return points

    def _extract_content(self, content):
        if content is None or (isinstance(content, dict) and 'track_points' in content):
            return super()._extract_content(content)
        if isinstance(content, dict) and 'polar_points' in content:
            return self._decode_polar_points(content['polar_points'])
        return super()._extract_content(content)

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _draw_bg_overlay(self, bg_draw, scale):
        """Draw the centre point and the radius_min/radius_max circles in
        gray, so the polar coordinate frame is visible behind the track."""
        cx, cy = self._center * scale
        for radius in (self._radius_min * scale, self._radius_max * scale):
            bg_draw.ellipse(
                [int(round(cx - radius)), int(round(cy - radius)),
                 int(round(cx + radius)), int(round(cy + radius))],
                outline=(70, 70, 70), width=1,
            )
        r = max(2, int(round(2 * scale)))
        bg_draw.ellipse(
            [int(round(cx - r)), int(round(cy - r)), int(round(cx + r)), int(round(cy + r))],
            fill=(70, 70, 70),
        )
