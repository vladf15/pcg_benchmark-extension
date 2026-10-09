"""The control targets and their domain ends (RacingProblem._CONTROL_TARGETS,
_rebuild_control_space), and how often random genomes hit each target.

  circuits  mean, sd, min, max and the outer Tukey fences of every control
            quantity over the 19 reference circuits that hosted an F1 World
            Championship race in 2019-2025
  random    per representation, N random genomes (seeds 21 and 22, 40
            each): how many hit each target within its tolerance, and the
            mean control score; the circuits likewise

Geometry only: no control is measured on the driver.

    python ../memcap.py 3000 control_targets.py [N]
"""
import gc
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
os.environ["PCG_BENCHMARK_WORKERS"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "model_training", "code"))
import track_loader  # noqa: E402
from pcg_benchmark.probs.racing.problem import RacingProblem  # noqa: E402
from pcg_benchmark.probs.utils import get_range_reward  # noqa: E402

F1 = ["Austin", "Budapest", "Catalunya", "Hockenheim", "Melbourne", "MexicoCity", "Montreal", "Monza",
      "Nuerburgring", "Sakhir", "SaoPaulo", "Shanghai", "Silverstone", "Sochi", "Spa", "Spielberg", "Suzuka",
      "YasMarina", "Zandvoort"]
REPS = (("spline", "pcg_benchmark.probs.racing.problem", "RacingProblem"),
        ("tile", "pcg_benchmark.probs.racingtile.problem", "RacingTileProblem"),
        ("hex", "pcg_benchmark.probs.racingtilehex.problem", "RacingTileHexProblem"),
        ("tilediag", "pcg_benchmark.probs.racingtilediag.problem", "RacingTileDiagProblem"),
        ("hexdiag", "pcg_benchmark.probs.racingtilehexdiag.problem", "RacingTileHexDiagProblem"),
        ("voronoi", "pcg_benchmark.probs.racingvoronoi.problem", "RacingVoronoiProblem"))


def terms(p, pts):
    curve = np.asarray(p._make_curve(p._normalize_track_points(np.asarray(pts, float))))
    return p._quality_terms({"track_points": pts, "curve_points": curve, "finished": True})


def score(p, t):
    """Per-control hits and the mean control score, as controlability()."""
    B, T = p._control_bands, p._CONTROL_TARGETS
    hits, sc = {}, []
    for k, (key, (lo, hi), tol) in B.items():
        c, v = float(T[k]), float(t[key])
        hits[k] = abs(v - c) <= tol + 1e-12
        sc.append(float(get_range_reward(v, min(lo, c - tol - 1e-9), c - tol, c + tol, max(hi, c + tol + 1e-9))))
    return hits, float(np.mean(sc))


def main(n):
    p = RacingProblem(width=2600.0, height=2600.0)
    keys = [key for key, _, _ in p._control_bands.values()]
    rows = [terms(p, np.asarray(track_loader.load_track(c, resample_spacing=5.0)["points"], float)) for c in F1]
    print("F1 circuits (19):")
    for key in keys:
        v = np.array([r[key] for r in rows], float)
        q1, q3 = np.percentile(v, [25, 75])
        print("  %-26s mean %9.4f sd %8.4f min %9.4f max %9.4f | outer fences %9.4f %9.4f" % (
            key, v.mean(), v.std(ddof=1), v.min(), v.max(), q1 - 3 * (q3 - q1), q3 + 3 * (q3 - q1)))
    groups = {"circuits": [score(p, r) for r in rows]}
    for name, mod, cls in REPS:
        q = getattr(__import__(mod, fromlist=[cls]), cls)()
        res = []
        for seed in (21, 22):
            q._content_space.seed(seed)
            res += [score(q, terms(q, q._extract_content(q._content_space.sample()))) for _ in range(n)]
        groups[name] = res
        del q
        gc.collect()
    print("\nhits per target, and mean control score:")
    ctl = list(p._control_bands)
    print("%-18s " % "" + " ".join("%9s" % g[:9] for g in groups))
    for k in ctl:
        print("%-18s " % k + " ".join("%5d/%-3d" % (sum(h[k] for h, _ in g), len(g)) for g in groups.values()))
    print("%-18s " % "mean score" + " ".join("%9.3f" % np.mean([s for _, s in g]) for g in groups.values()))
    print("%-18s " % "all ten" + " ".join("%9d" % sum(all(h.values()) for h, _ in g) for g in groups.values()))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
