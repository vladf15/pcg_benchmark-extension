"""Driver-independent lap (quasi-steady-state) and corner-structure candidates -> qcand4_{seed}.pkl.
    python memcap.py 1500 qcand4.py SEED"""
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
REPO = _os.path.normpath(_os.path.join(HERE, '..', '..', '..'))
import os, pickle, sys, warnings, numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, REPO)
from pcg_benchmark.probs.racing.problem import RacingProblem
from corner_classes import corners
P = RacingProblem(width=2600.0, height=2600.0); Q = P._QUALITY_PARAMS
seed = int(sys.argv[1]); G = 9.81
# Car, from engine.py and physics_tests.py (992 Carrera S)
M, PWR, ETA, CDA, RHO, CRR = 1555.0, 331000.0, 0.90, 0.60, 1.2, 0.015
AY, AXB, AXT, VMAX = 1.12 * G, 1.19 * G, 1.12 * G, 85.5

def curvature(c, smooth=5):
    ang, w = P._curve_turn_profile(c)
    k = ang / np.maximum(w, 1e-9)
    if smooth > 1:   # racing line cuts the centreline: a 25 m moving average (modelling choice)
        k = np.convolve(np.r_[k[-smooth:], k, k[:smooth]], np.ones(smooth) / smooth, "same")[smooth:-smooth]
    return k, w

def qss(c):
    """Quasi-steady-state speed profile on a closed curve.  Returns v per sample, ds, and the limiting state."""
    k, ds = curvature(c)
    n = len(k); ak = np.abs(k)
    vlim = np.minimum(VMAX, np.sqrt(AY / np.maximum(ak, 1e-9)))
    def fwd(v0, vl):
        v = np.empty(n); v[0] = v0
        for i in range(n - 1):
            ay = v[i] ** 2 * ak[i]; rem = max(0.0, 1 - (ay / AY) ** 2) ** 0.5
            drive = min(PWR * ETA / max(v[i], 1.0), AXT * M * rem)
            a = (drive - 0.5 * RHO * CDA * v[i] ** 2 - CRR * M * G) / M
            v[i + 1] = min(vl[i + 1], np.sqrt(max(v[i] ** 2 + 2 * a * ds[i], 0.0)))
        return v
    def bwd(vf):
        v = vf.copy()
        for i in range(n - 2, -1, -1):
            ay = v[i + 1] ** 2 * ak[i + 1]; rem = max(0.0, 1 - (ay / AY) ** 2) ** 0.5
            a = AXB * rem + (0.5 * RHO * CDA * v[i + 1] ** 2 + CRR * M * G) / M
            v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * a * ds[i]))
        return v
    # start the loop at its slowest point so the closed lap needs one pass
    s = int(np.argmin(vlim)); roll = lambda x: np.r_[x[s:], x[:s]]
    ak, ds, vl = roll(ak), roll(ds), roll(vlim)
    vf = fwd(vl[0], vl); v = bwd(vf)
    brake = v < vf - 1e-6
    atlim = (~brake) & (v >= vl - 1e-3) & (vl < VMAX - 1e-3)
    throttle = ~brake & ~atlim
    return v, ds, brake, throttle, atlim

def lap_measures(c):
    v, ds, brake, throttle, atlim = qss(c)
    dt = ds / np.maximum(v, 0.5); T = dt.sum(); L = ds.sum()
    m = {"qss_lap_s": float(T), "qss_vmean": float(L / T), "qss_vtop": float(v.max()), "qss_vmin": float(v.min()),
         "qss_throttle_time": float(dt[throttle].sum() / T), "qss_throttle_dist": float(ds[throttle].sum() / L),
         "qss_brake_time": float(dt[brake].sum() / T), "qss_limit_time": float(dt[atlim].sum() / T)}
    # braking events: runs of braking samples (circular)
    ev = []; n = len(v); i = 0
    idx = np.where(brake)[0]
    if len(idx):
        runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        for r in runs:
            a, b = r[0], min(r[-1] + 1, n - 1)
            ev.append((v[max(a - 1, 0)], v[b]))
    drops = np.array([p - q for p, q in ev]) if ev else np.zeros(1)
    vt = v.max()
    m["qss_brakes"] = float(len(ev)); m["qss_brakes_per_km"] = len(ev) / (L / 1000)
    m["qss_brakes_20"] = float(np.sum(drops >= 20)); m["qss_brakes_30"] = float(np.sum(drops >= 30))
    m["qss_max_drop"] = float(drops.max())
    m["qss_hard_stops"] = float(sum(p >= 0.8 * vt and q <= 0.4 * vt for p, q in ev))   # long straight into a sharp corner (Tilke 2015)
    m["qss_speed_entropy"] = float((lambda h: -(h[h > 0] * np.log(h[h > 0])).sum() / np.log(8))(np.histogram(v, 8, (0, VMAX), weights=dt)[0] / T))
    return m

def structure(c):
    m = {}
    turn, rad = corners(P, c)
    if len(turn) >= 2:
        # repeated corners: a corner with a twin within 5 degrees of turn and 5% of radius
        tw = [any(abs(turn[i] - turn[j]) < 5 and abs(rad[i] - rad[j]) / max(rad[i], 1e-9) < 0.05 for j in range(len(turn)) if j != i) for i in range(len(turn))]
        m["repeat_share"] = float(np.mean(tw))
        m["log_radius_sd"] = float(np.std(np.log(np.maximum(rad, 1.0))))
        m["turn_sd_deg"] = float(np.std(turn))
    else:
        m["repeat_share"] = 1.0; m["log_radius_sd"] = 0.0; m["turn_sd_deg"] = 0.0
    k, w = curvature(c, smooth=1); ak = np.abs(np.rad2deg(k))
    curved = ak > Q["fia_corner_curv_deg_per_m"]
    # constant-radius share: curved samples whose curvature is within 5% of both neighbours
    const = curved & (np.abs(ak - np.roll(ak, 1)) < 0.05 * ak) & (np.abs(ak - np.roll(ak, -1)) < 0.05 * ak)
    m["const_radius_share"] = float(w[const].sum() / max(w[curved].sum(), 1e-9))
    # curvature change per metre inside corners, p90 (deg/m per m)
    dk = np.abs(np.diff(np.r_[ak, ak[:1]])) / np.maximum(w, 1e-9)
    m["dk_p90"] = float(np.percentile(dk[curved], 90)) if curved.any() else 0.0
    ang, bw = P._curve_turn_profile(c)
    lens, _ = P._straight_windows(ang, bw); lens = np.sort(np.asarray(lens, float))[::-1]
    m["straights_200"] = float(np.sum(lens >= 200)); m["straights_100"] = float(np.sum(lens >= 100))
    m["second_over_first"] = float(lens[1] / lens[0]) if len(lens) > 1 and lens[0] > 0 else 0.0
    return m, turn, rad

base = pickle.load(open("qdata_seed%d.pkl" % seed, "rb"))
out = []
for r in base:
    m = {"name": r["name"], "group": r["group"], "quality": r["quality"]}
    s, turn, rad = structure(r["curve"]); m.update(s); m.update(lap_measures(r["curve"]))
    h = turn >= 150
    cls = np.where(h, 0, np.where(rad < 44, 1, np.where(rad < 75, 2, np.where(rad < 150, 3, 4)))) if len(turn) else np.zeros(0, int)
    m["class_seq"] = cls.tolist()
    out.append(m)
del base
pickle.dump(out, open("qcand4_%d.pkl" % seed, "wb"))
print("saved", len(out))
