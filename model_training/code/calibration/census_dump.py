"""Corner census of the 24 reference circuits (FIA corner line, as turn_count counts them).
Per corner: swept angle, mean radius (arc / angle), direction, and the straight gap
to the next corner.  Prints class frequencies per lap."""
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
REPO = _os.path.normpath(_os.path.join(HERE, '..', '..', '..'))
import sys, warnings, numpy as np
warnings.filterwarnings("ignore")
R = REPO
sys.path.insert(0, R); sys.path.insert(0, _os.path.join(R, "model_training", "code"))
import track_loader
from pcg_benchmark.probs.racing.problem import RacingProblem
p = RacingProblem(width=2600.0, height=2600.0)
P = p._QUALITY_PARAMS
names = [n for n in track_loader.list_tracks() if n != "IMS"]
allc = []
per = []
for n in names:
    pts = np.asarray(track_loader.load_track(n, resample_spacing=5.0)["points"], float)
    curve = p._make_curve(p._normalize_track_points(pts))
    ang, w = p._curve_turn_profile(curve)
    # signed corners: rerun walker keeping sign and end index
    thr = np.deg2rad(P["fia_corner_curv_deg_per_m"] * p._curve_step()); gapmax = P["corner_gap_max_m"]
    mt = np.deg2rad(P["corner_min_turn_deg"])
    cs = []; acc = run = gap = 0.0; sgn = 0; s = 0.0; start = 0.0
    for i, a in enumerate(ang):
        s += w[i]
        if abs(a) < thr:
            if sgn == 0: continue
            gap += w[i]; run += w[i]
            if gap > gapmax:
                if abs(acc) >= mt: cs.append((sgn, abs(acc), run - gap, s - gap))
                acc = run = gap = 0.0; sgn = 0
            continue
        g = 1 if a > 0 else -1
        if g != sgn and sgn != 0:
            if abs(acc) >= mt: cs.append((sgn, abs(acc), run, s - w[i]))
            acc = run = gap = 0.0
        if sgn == 0 or g != sgn: start = s
        acc += a; run += w[i]; sgn = g; gap = 0.0
    if abs(acc) >= mt: cs.append((sgn, abs(acc), run, s))
    L = p._closed_curve_length(curve)
    rows = []
    for k, (g, t, arc, end) in enumerate(cs):
        nxt = cs[(k + 1) % len(cs)]
        gap_next = (nxt[3] - nxt[2] - end) % L
        rows.append((g, np.rad2deg(t), arc / t, gap_next, g != nxt[0]))
    per.append((n, L, len(rows)))
    allc += rows
import json; json.dump([list(map(float,r[1:4]))+[bool(r[4])] for r in allc], open(_os.path.join(HERE, "census_corners.json"),"w"))
a = np.array([(r[1], r[2], r[3], r[4]) for r in allc])
turn, rad, gapn, flip = a.T
print("circuits %d, corners %d, per lap median %.0f" % (len(names), len(a), np.median([x[2] for x in per])))
print("radius m pctl 10/25/50/75/90: %s" % np.round(np.percentile(rad, [10, 25, 50, 75, 90]), 0))
print("turn deg pctl 10/25/50/75/90: %s" % np.round(np.percentile(turn, [10, 25, 50, 75, 90]), 0))
hair = turn >= 150
print("classes (share of corners):")
for lab, m in (("hairpin >=150 deg", hair),
               ("slow r<44 (not hairpin)", (rad < 44) & ~hair), ("medium 44-75", (rad >= 44) & (rad < 75) & ~hair),
               ("fast 75-150", (rad >= 75) & (rad < 150)), ("sweeper >=150", rad >= 150)):
    print("  %-26s %.2f  (%.1f per lap)" % (lab, m.mean(), m.sum() / len(names)))
ch = (flip > 0.5) & (gapn < 50)
print("direction flip with < 50 m straight to next corner (chicane / S pair): %.2f of corners (%.1f per lap)" % (ch.mean(), ch.sum() / len(names)))
print("turn angle classes: <60 %.2f  60-100 %.2f  100-150 %.2f  >=150 %.2f" % ((turn < 60).mean(), ((turn >= 60) & (turn < 100)).mean(), ((turn >= 100) & (turn < 150)).mean(), hair.mean()))
edges = [20, 52.5, 75, 105, 127.5, 150, 400]
lab = ["20-52", "52-75", "75-105", "105-127", "127-150", ">=150"]
cls = np.select([rad < 44, rad < 75, rad < 150], [0, 1, 2], 3)
print("angle bin   slow  medium  fast  sweeper   (share of all 360)")
for i in range(6):
    m = (turn >= edges[i]) & (turn < edges[i + 1])
    print("%-9s " % lab[i] + "  ".join("%.3f" % (m & (cls == k)).mean() for k in range(4)), " row %.3f" % m.mean())
