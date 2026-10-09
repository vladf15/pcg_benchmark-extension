"""The before/after measurement every change to the car or the quality
function is checked with (project instructions): per representation, N
random genomes at a fixed seed scored through info() and quality() with the
default driver: laps finished, mean quality, genomes at quality 1.0, where
the rest stop (gate, rules, drive, typicality), median lap length, mean
off-road share and seconds per evaluation; then the 24 reference circuits.

    python ../memcap.py 3000 validate.py [N] [seed]      (default 20, 21)
"""
import gc
import os
import sys
import time
import warnings
from collections import Counter

import numpy as np

warnings.filterwarnings("ignore")
os.environ["PCG_BENCHMARK_WORKERS"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "model_training", "code"))
import track_loader  # noqa: E402
from pcg_benchmark.probs.racing.problem import RacingProblem  # noqa: E402

REPS = (("racing", "pcg_benchmark.probs.racing.problem", "RacingProblem"),
        ("racingtile", "pcg_benchmark.probs.racingtile.problem", "RacingTileProblem"),
        ("racingtilehex", "pcg_benchmark.probs.racingtilehex.problem", "RacingTileHexProblem"),
        ("racingtilediag", "pcg_benchmark.probs.racingtilediag.problem", "RacingTileDiagProblem"),
        ("racingtilehexdiag", "pcg_benchmark.probs.racingtilehexdiag.problem", "RacingTileHexDiagProblem"),
        ("racingvoronoi", "pcg_benchmark.probs.racingvoronoi.problem", "RacingVoronoiProblem"),
        ("racingradial", "pcg_benchmark.probs.racingradial.problem", "RacingRadialProblem"))


def stop(p, info):
    t = p._quality_terms(info)
    if t is None or t["oob_score"] < 0.999 or t["self_overlaps"] > 0:
        return "gate"
    if any(t[k] < 0.999 for k in p._RULE_TERMS):
        return "rules"
    if min(t["completion"], t["on_road_score"]) < 0.999:
        return "drive"
    if t["typicality_score"] < 0.999:
        return "typical"
    return "pass"


def main(n, seed):
    print("%-18s %6s %6s %5s %8s %8s %6s  stops" % ("seed %d, N=%d" % (seed, n), "laps", "qual", "q=1",
                                                       "lap m", "offroad", "s/eval"))
    for name, mod, cls in REPS:
        p = getattr(__import__(mod, fromlist=[cls]), cls)()
        p._content_space.seed(seed)
        genomes = [p._content_space.sample() for _ in range(n)]
        t0 = time.perf_counter()
        infos = [p.info(g) for g in genomes]
        dt = (time.perf_counter() - t0) / n
        q = np.array([float(p.quality(i)) for i in infos])
        st = Counter(stop(p, i) for i in infos)
        print("%-18s %3d/%-2d %6.3f %5d %8.0f %8.4f %6.2f  %s" % (
            name, sum(bool(i["finished"]) for i in infos), n, q.mean(), int((q >= 0.999).sum()),
            np.median([i["total_length"] for i in infos]), np.mean([i.get("offroad_frac", 1.0) for i in infos]),
            dt, dict(st)), flush=True)
        del p, infos
        gc.collect()
    p = RacingProblem(width=2600.0, height=2600.0)
    names = [c for c in track_loader.list_tracks() if c != "IMS"]
    infos = [p.info({"track_points": np.asarray(track_loader.load_track(c, resample_spacing=5.0)["points"], float)})
             for c in names]
    q = np.array([p.quality(i) for i in infos])
    off = np.array([i.get("offroad_frac", 1.0) for i in infos])
    print("circuits: finished %d/24, quality 1.0 on %d, off-road on %d (mean %.4f, max %.4f)" % (
        sum(bool(i["finished"]) for i in infos), int((q >= 0.999).sum()), int((off > 0).sum()), off.mean(), off.max()))
    print("   below 1.0: " + ", ".join("%s %.3f (%s)" % (c, v, stop(p, i)) for c, v, i in zip(names, q, infos) if v < 0.999))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20, int(sys.argv[2]) if len(sys.argv) > 2 else 21)
