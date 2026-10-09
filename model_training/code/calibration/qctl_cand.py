"""Candidate control quantities: F1-mean target (2 s.f.), share of random tracks that round to it, and score spread.

Kept as the record of the candidate screen behind the controls left out of
RacingProblem._CONTROL_TARGETS.  It reads saved random genomes and their
terms (qdata_seed21.pkl, qdata_seed22.pkl, qspec_all2.pkl; 31 MB, decoded by
the earlier 31- and 70-tile vocabularies), which are not in the repository,
so it does not run as is.  control_targets.py measures the controls in use
on freshly sampled genomes."""
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
REPO = _os.path.normpath(_os.path.join(HERE, '..', '..', '..'))
import pickle, sys, warnings, math, numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, REPO)
from pcg_benchmark.probs.racing.problem import RacingProblem
from pcg_benchmark.probs.utils import get_range_reward
P = RacingProblem()
F1 = {"Austin", "Budapest", "Catalunya", "Hockenheim", "Melbourne", "MexicoCity", "Montreal", "Monza", "Nuerburgring", "Sakhir",
      "SaoPaulo", "Shanghai", "Silverstone", "Sochi", "Spa", "Spielberg", "Suzuka", "YasMarina", "Zandvoort"}
S = pickle.load(open(_os.path.join(HERE, "qspec_all2.pkl"), "rb"))
D = pickle.load(open(_os.path.join(HERE, "qdata_seed21.pkl"), "rb")) + pickle.load(open(_os.path.join(HERE, "qdata_seed22.pkl"), "rb"))
for s, r in zip(S, D):
    assert s["name"] == r["name"]
    c = r["curve"]; p = c[:-1] if np.allclose(c[0], c[-1]) else c
    area = 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
    ang, w = P._curve_turn_profile(c); thr = np.deg2rad(P._QUALITY_PARAMS["fia_corner_curv_deg_per_m"] * P._curve_step())
    turn_dir = np.sign(area)   # +1 anticlockwise
    s["clockwise"] = float(area < 0)
    # share of corner turning against the lap direction (the "other" hand), on samples tighter than the FIA line
    big = np.abs(ang) > thr
    s["counter_turn_share"] = float(np.abs(ang[big & (np.sign(ang) != turn_dir)]).sum() / max(np.abs(ang[big]).sum(), 1e-9))
    s["n_slow_all"] = s["n_hairpin"] + s["n_slow"]
    s["n_fast_all"] = s["n_fast"] + s["n_sweeper"]
    s["footprint_km2"] = float(np.ptp(p[:, 0]) * np.ptp(p[:, 1]) / 1e6)
del D
CANDS = ["n_hairpin", "n_slow_all", "n_medium", "n_fast_all", "qss_brakes_20", "qss_hard_stops", "qss_throttle_dist", "qss_vmean", "qss_lap_s",
         "fia_start_straight_t", "counter_turn_share", "footprint_km2", "clockwise", "fia_corners", "compactness"]
def round2(x):   # two significant figures; returns (target, step)
    if x == 0: return 0.0, 1.0
    e = math.floor(math.log10(abs(x))) - 1; step = 10.0 ** e
    return round(x / step) * step, step
INT = {"n_hairpin", "n_slow_all", "n_medium", "n_fast_all", "qss_brakes_20", "qss_hard_stops", "fia_corners", "clockwise"}
circ = [s for s in S if s["name"] in F1]
print("%-20s %9s %8s | %-44s | %s" % ("candidate", "F1 mean", "target", "random hitting the target (of 80): spl til hex vor", "F1 circuits hitting it"))
for k in CANDS:
    v = np.array([s[k] for s in circ if s.get(k) is not None and np.isfinite(s.get(k, np.nan))], float)
    m = v.mean()
    if k in INT: t, step = float(round(m)), 1.0
    else: t, step = round2(m)
    tol = step / 2
    hit = lambda x: x is not None and np.isfinite(x) and abs(x - t) <= tol + 1e-12
    rr = [sum(hit(s.get(k)) for s in S if s["group"] == r) for r in ("racing", "tile", "hex", "voronoi")]
    med = [np.nanmedian([s.get(k, np.nan) for s in S if s["group"] == r]) for r in ("racing", "tile", "hex", "voronoi")]
    print("%-20s %9.3f %8.3g | %3d %3d %3d %3d   (medians %s) | %d of %d" % (k, m, t, *rr, " ".join("%.3g" % x for x in med), sum(hit(x) for x in v), len(v)))
