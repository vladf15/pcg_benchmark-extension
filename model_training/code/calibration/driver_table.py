"""Driver comparison row: circuits off-road, share mean/max, mean speed; and
generated tracks (20 random genomes, seed 21) with any off-road step, per
representation, and the worst lap's share.  python driver_table.py POLICY.zip"""
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
REPO = _os.path.normpath(_os.path.join(HERE, '..', '..', '..'))
import gc, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore"); os.environ["PCG_BENCHMARK_WORKERS"] = "1"
ROOT = REPO
sys.path.insert(0, ROOT); sys.path.insert(0, _os.path.join(ROOT, "model_training", "code"))
import track_loader
from pcg_benchmark.probs.racing.problem import RacingProblem
from pcg_benchmark.probs.racingtile.problem import RacingTileProblem
from pcg_benchmark.probs.racingtilehex.problem import RacingTileHexProblem
from pcg_benchmark.probs.racingvoronoi.problem import RacingVoronoiProblem
pol = sys.argv[1]
p = RacingProblem(width=2600.0, height=2600.0, rl_policy_path=pol)
off, sp = [], []
for n in track_loader.list_tracks():
    if n == "IMS": continue
    i = p.info({"track_points": np.asarray(track_loader.load_track(n, resample_spacing=5.0)["points"], float)})
    off.append(i["offroad_frac"])
    if i["finished"]: sp.append(i["total_length"] / (i["steps"] * p._engine.time_step))
off = np.array(off); del p; gc.collect()
gen, worst = [], 0.0
for C in (RacingProblem, RacingTileProblem, RacingTileHexProblem, RacingVoronoiProblem):
    q = C(rl_policy_path=pol); s = q._content_space; s.seed(21)
    o = np.array([q.info(s.sample())["offroad_frac"] for _ in range(20)])
    gen.append(int((o > 0).sum())); worst = max(worst, float(o.max())); del q; gc.collect()
print("%s: circuits %d of 24, %.4f / %.3f, %.1f m/s | generated spline %d tile %d hex %d voronoi %d, worst %.3f"
      % (pol.split("runs")[-1], int((off > 0).sum()), off.mean(), off.max(), np.mean(sp), *gen, worst))
