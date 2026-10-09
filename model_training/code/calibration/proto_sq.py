"""Square tile shape candidates: corners each builds between two straights, as
the walker counts them, and the smallest clearance from the road edge to the
cell edge (8 m half width).  Units: apothems (68.18 m)."""
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
REPO = _os.path.normpath(_os.path.join(HERE, '..', '..', '..'))
import sys, warnings, numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, REPO)
from tile_catalog_fn import tile_corners
import pcg_benchmark.probs.racingtile.problem as SQ
p = SQ.RacingTileProblem()
HX = 0.5 * p._width / SQ.GRID_W
TR = SQ._TRANSITION_M / HX; STEP = 0.5 * p._curve_step() / HX


def solve2(segs_fn, start, heading, end):
    def e(a, b):
        return SQ._walk_path(segs_fn(a, b), start, heading, lambda r, t: 1)[-1]
    e0 = e(0.0, 0.0)
    a, b = np.linalg.solve(np.column_stack([e(1.0, 0.0) - e0, e(0.0, 1.0) - e0]), np.asarray(end) - e0)
    return a, b


def corner_pts(segs):
    return SQ._walk_smooth(segs, (0.0, 1.0), -90.0, TR, STEP, (1.0, 0.0))


def straight_pts(segs):
    return SQ._walk_smooth(segs, (-1.0, 0.0), 0.0, TR, STEP, (1.0, 0.0))


def clearance(pts):
    """Smallest distance from the centre line to the cell edge, minus 8 m, metres,
    away from the two ports (points within 0.2 apothems of a port skipped)."""
    m = pts * HX
    d = HX - np.max(np.abs(m), axis=1)
    ends = [pts[0], pts[-1]]
    far = np.array([min(np.linalg.norm(q - e) for e in ends) > 0.3 for q in pts])
    return (d[far].min() - 8.0) if far.any() else np.nan


def show(lab, pts):
    print("%-34s %-60s clear %.1f m" % (lab, tile_corners(p, pts * HX), clearance(pts)))


mode = sys.argv[1]
if mode == "sharp":
    for t in (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.65, 0.9, 1.0):
        show("corner tangent %.2f (r %.0f m)" % (t, t * HX), corner_pts(SQ._corner_path(SQ.CORNER, 90.0, 0, 0.4, t)))
if mode == "tighten":
    for r2 in (0.2, 0.25, 0.3, 0.35, 0.4, 0.5):
        for k in (1.5, 2.0, 2.5, 3.0):
            r1 = k * r2
            try:
                a, b = solve2(lambda a, b: [("L", a), ("A", r1, 45.0), ("A", r2, 45.0), ("L", b)], (0.0, 1.0), -90.0, (1.0, 0.0))
            except np.linalg.LinAlgError:
                continue
            if min(a, b) < 0:
                print("tighten r2 %.2f r1 %.2f: no fit (%.2f, %.2f)" % (r2, r1, a, b)); continue
            show("tighten r2 %.2f r1 %.2f (a %.2f b %.2f)" % (r2, r1, a, b), corner_pts([("L", a), ("A", r1, 45.0), ("A", r2, 45.0), ("L", b)]))
if mode == "double":
    for r in (0.2, 0.25, 0.3, 0.35, 0.4, 0.5):
        # symmetric: entry a, arc 45, straight m, arc 45, exit a
        def segs(a, m):
            return [("L", a), ("A", r, 45.0), ("L", m), ("A", r, 45.0), ("L", a)]
        a, m = solve2(segs, (0.0, 1.0), -90.0, (1.0, 0.0))
        if min(a, m) < 0:
            print("double r %.2f: no fit (%.2f, %.2f)" % (r, a, m)); continue
        show("double r %.2f (a %.2f m %.2f = %.0f m)" % (r, a, m, m * HX), corner_pts(segs(a, m)))
if mode == "kink":
    for cs in (20.0, 30.0, 40.0, 45.0, 50.0):
        for ks in (0.3, 0.4, 0.5):
            for kr in (0.4, 0.6):
                def segs(lead, mid):
                    return [("L", lead), ("A", ks, 90.0 + cs), ("L", mid), ("A", kr, -cs), ("L", 0.1)]
                lead, mid = solve2(segs, (0.0, 1.0), -90.0, (1.0, 0.0))
                if min(lead, mid) < 0:
                    print("kink counter %.0f strong %.2f second %.1f: no fit (%.2f, %.2f)" % (cs, ks, kr, lead, mid)); continue
                show("kink c%.0f s%.2f r%.1f (%.2f,%.2f)" % (cs, ks, kr, lead, mid), corner_pts(segs(lead, mid)))
if mode == "ess":
    for s in (8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 24.0):
        for frac in (1.0, 0.8, 0.6):
            R = frac * 2.0 / (4.0 * np.sin(np.deg2rad(s)))
            try:
                segs = SQ._chicane_path(s, R)
            except ValueError:
                continue
            show("ess swing %.0f R %.2f (%.0f m) off %.1f m" % (s, R, R * HX, 2 * R * (1 - np.cos(np.deg2rad(s))) * HX), straight_pts(segs))
