"""Hex, hexdiag and diag shape candidates: corners each builds between two
straights, as the walker counts them."""
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
REPO = _os.path.normpath(_os.path.join(HERE, '..', '..', '..'))
import sys, warnings, numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, REPO)
from tile_catalog_fn import tile_corners
import pcg_benchmark.probs.racingtile.problem as SQ
mode = sys.argv[1]

if mode in ("hex", "hexdiag"):
    import pcg_benchmark.probs.racingtilehex.problem as HX
    p = HX.RacingTileHexProblem()
    AP = p._hex_size() * np.sqrt(3.0) / 2.0
    n_arc = lambda radius, t: p._arc_samples(radius * AP * abs(t))
    print("apothem %.1f m" % AP)
if mode == "hex":
    for turn in (60.0, 120.0):
        for sr in (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7):
            pts = SQ._tile_points(SQ.SHARP, turn, n_arc, sharp_radius=sr)
            print("sharp %3.0f  tangent %.2f r %.1f m  %s" % (turn, sr, sr / np.tan(np.deg2rad(turn / 2)) * AP, tile_corners(p, pts * AP)))
    for turn in (60.0, 120.0):
        for ks in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            try:
                pts = SQ._tile_points(SQ.KINK, turn, n_arc, kink_strong=ks)
            except Exception as e:
                print("kink %3.0f strong %.2f: %s" % (turn, ks, e)); continue
            print("kink %3.0f strong %.2f (%.0f m)  %s" % (turn, ks, ks * AP, tile_corners(p, pts * AP)))
    for s in (10.0, 12.0, 13.0, 14.0, 16.0, 18.0, 20.0, 22.0):
        R = 2.0 / (4.0 * np.sin(np.deg2rad(s)))
        pts = SQ._tile_points(SQ.CHICANE, 0.0, n_arc, chicane=(s, R))
        print("ess swing %.0f R %.2f (%.0f m) offset %.1f m  %s" % (s, R, R * AP, 2 * R * (1 - np.cos(np.deg2rad(s))) * AP, tile_corners(p, pts * AP)))
if mode == "hexdiag":
    import pcg_benchmark.probs.racingtilehexdiag.problem as HD
    # fvkink: face 0 to corner 8; vary the arc radius below the largest that fits
    for kind, a, b in ((HD.FVKINK, 0, 8), (HD.VBEND, 6, 8), (HD.FVCORNER, 0, 7), (HD.VCORNER, 6, 7)):
        start, end = HD._port_xy(a), HD._port_xy(b)
        heading = np.rad2deg(HD._port_angle(a)) + 180.0
        turn = (np.rad2deg(HD._port_angle(b)) - heading + 180.0) % 360.0 - 180.0
        rmax = min(np.linalg.norm(start), np.linalg.norm(end)) / np.tan(np.deg2rad(abs(turn)) / 2.0)
        for f in (1.0, 0.8, 0.6, 0.5, 0.4, 0.3):
            radius = f * rmax
            def segs(x, y):
                return [("L", x), ("A", radius, turn), ("L", y)]
            def we(x, y):
                return SQ._walk_path(segs(x, y), start, heading, lambda r, t: 1)[-1]
            e0 = we(0, 0)
            x, y = np.linalg.solve(np.column_stack([we(1, 0) - e0, we(0, 1) - e0]), end - e0)
            if min(x, y) < -1e-9:
                print("%s %d-%d f %.1f: no fit" % (kind, a, b, f)); continue
            pts = SQ._walk_path(segs(max(x, 0), max(y, 0)), start, heading, n_arc)
            print("%-9s %d-%d turn %.0f f %.1f r %.0f m (straights %.2f %.2f)  %s" % (kind, a, b, turn, f, radius * AP, x, y, tile_corners(p, pts * AP)))
if mode == "diag":
    import pcg_benchmark.probs.racingtilediag.problem as D
    p = D.RacingTileDiagProblem()
    for kind, lo, hi in ((D.DENTRY, 0.5, 2.41), (D.DCORNER, 0.5, 1.41)):
        for r in np.linspace(lo, hi, 12):
            try:
                segs, start, heading, end = D._diag_segments(kind, r)
            except ValueError as e:
                print(kind, r, e); continue
            hx = 0.5 * p._width / D.GRID_W
            pts = SQ._walk_smooth(segs, start, heading, SQ._TRANSITION_M / hx, 0.5 * p._curve_step() / hx, end)
            print("%s r %.2f (%.0f m)  %s" % ("DENTRY" if kind == D.DENTRY else "DCORNER", r, r * hx, tile_corners(p, pts * hx)))
