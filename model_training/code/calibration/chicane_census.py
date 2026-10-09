"""Chicanes on the 24 reference circuits: opposite-direction corner pairs less than 50 m apart (FIA 300 m line)."""
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
REPO = _os.path.normpath(_os.path.join(HERE, '..', '..', '..'))
import sys, warnings, numpy as np
warnings.filterwarnings("ignore")
R = REPO; sys.path.insert(0, R); sys.path.insert(0, _os.path.join(R, "model_training", "code"))
import track_loader
from pcg_benchmark.probs.racing.problem import RacingProblem
p = RacingProblem(width=2600.0, height=2600.0); FIA = p._QUALITY_PARAMS["fia_corner_curv_deg_per_m"]
rows = []
for n in [n for n in track_loader.list_tracks() if n != "IMS"]:
    pts = np.asarray(track_loader.load_track(n, resample_spacing=5.0)["points"], float)
    c = p._make_curve(p._normalize_track_points(pts))
    ang, w = p._curve_turn_profile(c)
    turns, arcs, spans = p._find_corners(ang, w, FIA, spans=True)
    s = np.r_[0, np.cumsum(w)]
    sign = [np.sign(ang[(s[:-1] >= a) & (s[:-1] < b)].sum()) for a, b in spans]
    L = s[-1]
    for k in range(len(turns)):
        j = (k + 1) % len(turns)
        gap = (spans[j][0] - spans[k][1]) % L
        if sign[k] != sign[j] and gap < 50.0:
            rows.append((n, np.rad2deg(turns[k]), arcs[k] / turns[k], np.rad2deg(turns[j]), arcs[j] / turns[j], gap))
rows.sort(key=lambda r: r[0])
for r in rows: print("%-13s %5.0f deg r %5.1f m | %5.0f deg r %5.1f m | gap %4.1f m" % r)
t = np.array([[r[1], r[3]] for r in rows]); rad = np.array([[r[2], r[4]] for r in rows]); g = np.array([r[5] for r in rows])
print("\n%d chicane pairs on %d circuits" % (len(rows), len({r[0] for r in rows})))
print("turn per corner: p10 %.0f  median %.0f  p90 %.0f deg | radius p10 %.1f median %.1f p90 %.1f m | gap median %.1f m" % (
    *np.percentile(t, [10, 50, 90]), *np.percentile(rad, [10, 50, 90]), np.median(g)))
