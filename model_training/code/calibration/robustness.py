"""Robustness of the quality function's choices: how the verdict on real
circuits, on random genomes of each representation and on the specificity
negatives moves when one choice is changed at a time.

Variants of the typicality stage (refitted on the reference circuits each
time) and of the driven-lap stage:
  base             ridge 0.1, threshold = largest leave-one-out distance
  ridge 0.05/0.5   another ridge on the standardised covariance
  2nd LOO          threshold = second-largest leave-one-out distance
  Hotelling 95/99  no ridge; the prediction region for one new draw from the
                   circuits' population, T^2 <= p (n+1)(n-1) / (n (n-p))
                   F_{p, n-p}(level), from Hotelling's (1931) T^2
                   distribution (it assumes roughly normal features)
  drop <feature>   the four other features
  + turning/km     a sixth feature, total turning per km
  + min corners    a rule: at least the fewest FIA corners of any reference
                   circuit (8)
  off-road x0.5 / x2    the driven-lap off-road limit (base: the
                         problem's offroad_full_frac)

Random genomes: N per representation (seed 21) with the default driver.
Ranking: the representations ordered by their share at quality 1.0;
Kendall's tau against the base ranking.

    python ../memcap.py 3000 robustness.py [N]
"""
import gc
import os
import sys
import warnings

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")
os.environ["PCG_BENCHMARK_WORKERS"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import specificity as S  # noqa: E402  (also puts the repo on sys.path)

REPS = (("racing", "pcg_benchmark.probs.racing.problem", "RacingProblem"),
        ("racingtile", "pcg_benchmark.probs.racingtile.problem", "RacingTileProblem"),
        ("racingtilehex", "pcg_benchmark.probs.racingtilehex.problem", "RacingTileHexProblem"),
        ("racingtilediag", "pcg_benchmark.probs.racingtilediag.problem", "RacingTileDiagProblem"),
        ("racingtilehexdiag", "pcg_benchmark.probs.racingtilehexdiag.problem", "RacingTileHexDiagProblem"),
        ("racingvoronoi", "pcg_benchmark.probs.racingvoronoi.problem", "RacingVoronoiProblem"))
BASE = list(S.P._TYPICALITY_FEATURES)
EXTRA = "turning_per_km"
OFF = float(S.P._QUALITY_PARAMS["offroad_full_frac"])


def record(p, info):
    """What every variant needs from one scored track."""
    t = p._quality_terms(info)
    ok = t["oob_score"] >= 0.999 and t["self_overlaps"] == 0 and all(t[k] >= 0.999 for k in p._RULE_TERMS)
    return {"rules": ok, "finished": t["completion"] >= 0.999, "offroad": t["offroad_frac"],
            "corners": t["fia_corner_count"], "x": {k: float(t[k]) for k in BASE + [EXTRA]}}


def fit(X, ridge):
    mu, sd = X.mean(0), X.std(0)
    Z = (X - mu) / sd
    return mu, sd, np.linalg.inv(np.cov(Z.T) + ridge * np.eye(X.shape[1]))


def dist(x, f):
    z = (x - f[0]) / f[1]
    return float(np.sqrt(max(0.0, z @ f[2] @ z)))


def threshold(X, ridge, rule):
    n, p = X.shape
    if rule.startswith("hotelling"):
        level = float(rule[-2:]) / 100.0
        t2 = p * (n + 1) * (n - 1) / (n * (n - p)) * stats.f.ppf(level, p, n - p)
        return float(np.sqrt(t2))
    loo = sorted(dist(X[i], fit(np.delete(X, i, 0), ridge)) for i in range(n))
    return loo[-1] if rule == "loo" else loo[-2]


def variants():
    yield "base", BASE, 0.1, "loo", OFF, 0
    yield "ridge 0.05", BASE, 0.05, "loo", OFF, 0
    yield "ridge 0.5", BASE, 0.5, "loo", OFF, 0
    yield "2nd LOO", BASE, 0.1, "loo2", OFF, 0
    yield "Hotelling 95", BASE, 0.0, "hotelling95", OFF, 0
    yield "Hotelling 99", BASE, 0.0, "hotelling99", OFF, 0
    for k in BASE:
        yield "drop " + k, [f for f in BASE if f != k], 0.1, "loo", OFF, 0
    yield "+ turning/km", BASE + [EXTRA], 0.1, "loo", OFF, 0
    yield "+ min corners", BASE, 0.1, "loo", OFF, 8
    yield "off-road %g" % (0.5 * OFF), BASE, 0.1, "loo", 0.5 * OFF, 0
    yield "off-road %g" % (2.0 * OFF), BASE, 0.1, "loo", 2.0 * OFF, 0


def passes(r, feats, f, thr, offroad, min_corners, driven=True):
    if not r["rules"] or r["corners"] < min_corners:
        return False
    if driven and (not r["finished"] or r["offroad"] > offroad + 1e-12):
        return False
    return dist(np.array([r["x"][k] for k in feats]), f) <= thr + 1e-9


def main(n):
    circ, names = [], [n_ for n_ in S.track_loader.list_tracks() if n_ != "IMS"]
    from pcg_benchmark.probs.racing.problem import RacingProblem
    pc = RacingProblem(width=2600.0, height=2600.0)
    for name in names:
        pts = np.asarray(S.track_loader.load_track(name, resample_spacing=5.0)["points"], float)
        circ.append(record(pc, pc.info({"track_points": pts})))
    ref = [r for r in circ if r["rules"]]                        # the 21 reference circuits
    groups = {}
    for name, mod, cls in REPS:
        p = getattr(__import__(mod, fromlist=[cls]), cls)()
        p._content_space.seed(21)
        groups[name] = [record(p, p.info(p._content_space.sample())) for _ in range(n)]
        del p
        gc.collect()
        print("scored", name, flush=True)
    negatives = []
    for fam in ("rippled", "blob"):
        f = getattr(S, fam)
        for name in names:
            pts = np.asarray(S.track_loader.load_track(name, resample_spacing=5.0)["points"], float)
            if S.stage(pts)[0] == "pass":
                q = f(pts)
                q = q - q.mean(0) + S.MAP / 2.0
                curve = np.asarray(S.P._make_curve(S.P._normalize_track_points(q)))
                negatives.append(record(S.P, {"track_points": q, "curve_points": curve, "finished": True,
                                              "offroad_frac": 0.0}))
    base_order = None
    short = {"racing": "splin", "racingtile": "tile", "racingtilehex": "hex", "racingtilediag": "tdiag",
             "racingtilehexdiag": "hdiag", "racingvoronoi": "voron"}
    print("\n%-26s %6s %8s | %s | %5s" % ("variant", "thr", "circuits", "  ".join("%5s" % short[g] for g in groups),
                                         "neg"))
    for label, feats, ridge, rule, off, mc in variants():
        X = np.array([[r["x"][k] for k in feats] for r in ref])
        f = fit(X, ridge)
        thr = threshold(X, ridge, rule)
        cp = sum(passes(r, feats, f, thr, off, mc) for r in circ)
        share = [np.mean([passes(r, feats, f, thr, off, mc) for r in g]) for g in groups.values()]
        neg = sum(passes(r, feats, f, thr, off, mc, driven=False) for r in negatives)
        if base_order is None:
            base_order = share
            tau = 1.0
        else:
            tau = stats.kendalltau(base_order, share).statistic
        print("%-26s %6.2f %5d/24 | %s | %2d/%d  tau %.2f" % (label, thr, cp, "  ".join("%5.2f" % s for s in share),
                                                             neg, len(negatives), tau), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
