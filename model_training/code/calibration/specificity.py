"""Specificity of the quality function: how many tracks that are not
plausible circuits it still scores 1.0.

The reference circuits measure sensitivity (real circuits pass); this
measures the other half.  Each negative family is built from the 21
reference circuits (or drawn at random) so that it keeps most of a
circuit's statistics and breaks one property a circuit designer would not
accept:

  half size        each circuit scaled by 0.5 about its centroid
  double size      scaled by 2
  stretched        scaled by 2.5 along its longest straight's direction
  rippled          a 6 m sideways ripple of 150 m wavelength along the lap
  blob             positions averaged over 600 m, which rounds every corner away
  polygon          simplified to a polygon (30 m tolerance), corners with no radius
  random loop      smooth closed loops r(t) = R (1 + sum_k a_k cos(k t + p_k)),
                   k = 2..8, scaled to 3.5-7 km (40 drawn, seed 21)

Scored on geometry with the lap assumed driven on the road (gates, rules,
typicality), so a pass means the yardstick itself accepts the shape; the
driven-lap stage could only remove more.  A 6000 m map keeps the map edge
from deciding.

    python ../memcap.py 3000 specificity.py
"""
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "model_training", "code"))
import track_loader  # noqa: E402
from pcg_benchmark.probs.racing.problem import RacingProblem  # noqa: E402

MAP = 6000.0
P = RacingProblem(width=MAP, height=MAP)


def stage(pts):
    """('pass' | 'gate' | 'rules' | 'typical', terms) for a closed point list."""
    pts = np.asarray(pts, float)
    pts = pts - pts.mean(0) + MAP / 2.0
    curve = np.asarray(P._make_curve(P._normalize_track_points(pts)))
    info = {"track_points": pts, "curve_points": curve, "finished": True, "offroad_frac": 0.0}
    t = P._quality_terms(info)
    if t["oob_score"] < 0.999 or t["self_overlaps"] > 0:
        return "gate", t
    if any(t[k] < 0.999 for k in P._RULE_TERMS):
        return "rules", t
    if t["typicality_score"] < 0.999:
        return "typical", t
    return "pass", t


def resample(pts, step=5.0):
    p = np.vstack([pts, pts[:1]])
    d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    s = np.arange(0.0, d[-1], step)
    return np.c_[np.interp(s, d, p[:, 0]), np.interp(s, d, p[:, 1])]


def longest_straight_dir(pts):
    seg = np.diff(np.vstack([pts, pts[:1]]), axis=0)
    h = np.arctan2(seg[:, 1], seg[:, 0])
    best, best_len, i, n = 0.0, 0.0, 0, len(h)
    while i < n:                     # longest run within 2 degrees of its first heading
        j = i
        while j + 1 < n and abs(np.angle(np.exp(1j * (h[j + 1] - h[i])))) < np.deg2rad(2.0):
            j += 1
        if j - i > best_len:
            best_len, best = j - i, h[i]
        i = j + 1
    return best


def rippled(pts):
    p = resample(pts)
    nrm = np.diff(np.vstack([p, p[:1]]), axis=0)
    nrm = np.c_[-nrm[:, 1], nrm[:, 0]] / np.maximum(np.hypot(*nrm.T), 1e-9)[:, None]
    s = np.arange(len(p)) * 5.0
    return p + nrm * (6.0 * np.sin(2 * np.pi * s / 150.0))[:, None]


def blob(pts, window=600.0):
    p = resample(pts)
    w = int(window / 5.0) | 1
    ext = np.vstack([p[-w:], p, p[:w]])
    k = np.ones(w) / w
    return np.c_[np.convolve(ext[:, 0], k, "same"), np.convolve(ext[:, 1], k, "same")][w:-w]


def polygon(pts, tol=30.0):
    """Douglas-Peucker simplification of the closed line, then 5 m spacing."""
    def dp(a):
        if len(a) < 3:
            return a
        e, q = a[-1] - a[0], a - a[0]
        d = np.abs(e[0] * q[:, 1] - e[1] * q[:, 0]) / max(np.hypot(*e), 1e-9)
        i = int(np.argmax(d))
        if d[i] <= tol:
            return np.vstack([a[0], a[-1]])
        return np.vstack([dp(a[:i + 1])[:-1], dp(a[i:])])
    p = resample(pts)
    far = int(np.argmax(np.hypot(*(p - p[0]).T)))
    poly = np.vstack([dp(p[:far + 1])[:-1], dp(np.vstack([p[far:], p[:1]]))[:-1]])
    return resample(poly)


def random_loops(n, seed=21):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    out = []
    for _ in range(n):
        r = np.ones_like(t)
        for k in range(2, 9):
            r += rng.uniform(0, 0.25 / k * 2) * np.cos(k * t + rng.uniform(0, 2 * np.pi))
        loop = np.c_[r * np.cos(t), r * np.sin(t)]
        length = np.sum(np.hypot(*np.diff(np.vstack([loop, loop[:1]]), axis=0).T))
        out.append(loop * rng.uniform(3500, 7000) / length)
    return out


def main():
    names = [n for n in track_loader.list_tracks() if n != "IMS"]
    circuits = {}
    for n in names:
        pts = np.asarray(track_loader.load_track(n, resample_spacing=5.0)["points"], float)
        if stage(pts)[0] == "pass":
            circuits[n] = pts
    print("reference circuits passing: %d" % len(circuits))
    fams = {
        "half size": [(p - p.mean(0)) * 0.5 for p in circuits.values()],
        "double size": [(p - p.mean(0)) * 2.0 for p in circuits.values()],
        "stretched": [],
        "rippled": [rippled(p) for p in circuits.values()],
        "blob": [blob(p) for p in circuits.values()],
        "polygon": [polygon(p) for p in circuits.values()],
        "random loop": random_loops(40),
    }
    for p in circuits.values():
        a = longest_straight_dir(p)
        u = np.array([np.cos(a), np.sin(a)])
        q = p - p.mean(0)
        along = q @ u
        fams["stretched"].append(q + np.outer(along * 1.5, u))
    print("%-13s %5s | %4s %5s %7s %4s" % ("family", "n", "gate", "rules", "typical", "PASS"))
    for fam, tracks in fams.items():
        st = [stage(t)[0] for t in tracks]
        print("%-13s %5d | %4d %5d %7d %4d" % (fam, len(st), st.count("gate"), st.count("rules"),
                                               st.count("typical"), st.count("pass")), flush=True)


if __name__ == "__main__":
    main()
