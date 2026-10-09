"""Voronoi weight ceiling and site count by one rule: the setting under
which the most random genomes meet every regulation (RacingProblem's
_RULE_TERMS at full marks, after both gates).  40 random genomes per seed,
geometry only (no driver).  Also reports typicality passes, the median
number of selectable cells, and median lap, longest straight and FIA start
straight.  Feeds the _WEIGHT_MAX_M2 comment in racingvoronoi/problem.py.

    python ../memcap.py 3000 voronoi_rule.py [seed ...]        (default 21 22)
"""
import gc
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
os.environ["PCG_BENCHMARK_WORKERS"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "..")))
from pcg_benchmark.probs.racingvoronoi.problem import RacingVoronoiProblem  # noqa: E402

N = 40
SETTINGS = [(c, w) for c in (30, 40, 50)
            for w in (2250.0, 10000.0, 20000.0, 40000.0, 60000.0, 80000.0, 120000.0)]


def run(cells, w_max, seed):
    p = RacingVoronoiProblem(weight_max_m2=w_max, num_cells=cells)
    sp = p._content_space
    sp.seed(seed)
    rules = typ = gates = 0
    lap, longest, start, eligible = [], [], [], []
    for _ in range(N):
        content = sp.sample()
        pts = p._extract_content(content)
        eligible.append(cells - len(p._ineligible_cells))
        curve = np.asarray(p._make_curve(p._normalize_track_points(np.asarray(pts, float))))
        t = p._quality_terms({"track_points": pts, "curve_points": curve, "finished": True})
        if t["oob_score"] < 0.999 or t["self_overlaps"] > 0:
            gates += 1
            continue
        ok = all(t[k] >= 0.999 for k in p._RULE_TERMS)
        rules += ok
        typ += ok and t["typicality_score"] >= 0.999
        lap.append(t["length_m"])
        longest.append(t["longest_straight_m"])
        start.append(t["fia_start_straight_m"])
    print("seed %d  sites %d  w_max %6.0f | gates %2d  rules %2d/%d  rules+typical %2d | "
          "selectable p50 %4.1f  lap %4.0f  longest %4.0f  start %4.0f" % (
              seed, cells, w_max, gates, rules, N, typ, np.median(eligible),
              np.median(lap), np.median(longest), np.median(start)), flush=True)
    del p
    gc.collect()


if __name__ == "__main__":
    seeds = [int(a) for a in sys.argv[1:]] or [21, 22]
    for seed in seeds:
        for cells, w_max in SETTINGS:
            run(cells, w_max, seed)
