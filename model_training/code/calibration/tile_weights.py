"""WFC weights of the four tile problems against the current quality
function: for each (grass, straight, curve) setting of the weight tables in
the module comments, the random genomes (40 per seed, seeds 21-23) that
pass both gates, every regulation (_RULE_TERMS) and typicality, geometry
only (the lap is assumed driven), with the median turn count.  The same rule
sets the Voronoi weight ceiling (voronoi_rule.py).

    python ../memcap.py 3000 tile_weights.py [problem ...]
    problem: square, diag, hex, hexdiag (default: all)
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
import pcg_benchmark.probs.racingtile.problem as sq  # noqa: E402
import pcg_benchmark.probs.racingtilediag.problem as dg  # noqa: E402
import pcg_benchmark.probs.racingtilehex.problem as hx  # noqa: E402
import pcg_benchmark.probs.racingtilehexdiag.problem as hd  # noqa: E402

N = 40
SEEDS = (21, 22, 23)
# The settings each module's comment tabulates; the first is the one in use.
SETTINGS = {
    "square": [(3, 32, 0.125), (3, 4, 0.125), (3, 16, 0.125), (8, 8, 0.125)],
    "diag": [(5, 8, 0.0625), (3, 4, 0.0625), (3, 4, 0.125), (3, 4, 0.25), (8, 8, 0.0625), (8, 8, 0.125)],
    "hex": [(4, 20, 0.1), (4, 10, 0.05), (4, 10, 0.1), (4, 10, 0.15), (4, 10, 0.3), (4, 40, 0.1),
            (4, 80, 0.1), (8, 20, 0.1), (8, 40, 0.1)],
    "hexdiag": [(8, 20, 0.05), (4, 10, 0.1), (4, 10, 0.05)],
}


def make(name, grass, straight, curve):
    if name == "square":
        w = {k: (grass if k == sq.GRASS else straight if k == sq.STRAIGHT else curve) for k in sq._TYPE_WEIGHTS}
        sq.RacingTileProblem._WFC_WEIGHTS = np.array([w[t] for t, _ in sq.RacingTileProblem._WFC_TILES])
        return sq.RacingTileProblem()
    if name == "diag":
        w = {k: (grass if k == dg.GRASS else straight if k in (dg.STRAIGHT, dg.DSTRAIGHT) else curve)
             for k in dg._DIAG_TYPE_WEIGHTS}
        dg.RacingTileDiagProblem._WFC_WEIGHTS = np.array([w[t] for t, _ in dg.RacingTileDiagProblem._WFC_TILES])
        return dg.RacingTileDiagProblem()
    if name == "hex":
        hx._GRASS_WEIGHT, hx._STRAIGHT_WEIGHT, hx._CURVE_WEIGHT = float(grass), float(straight), float(curve)
        hx.RacingTileHexProblem._WFC_WEIGHTS = None
        return hx.RacingTileHexProblem()
    hd._GRASS_WEIGHT, hd._STRAIGHT_WEIGHT, hd._CURVE_WEIGHT = float(grass), float(straight), float(curve)
    for k in hd._NEW_WEIGHTS:
        hd._NEW_WEIGHTS[k] = float(straight) if k == hd.VSTRAIGHT else float(curve)
    hd.RacingTileHexDiagProblem._WFC_WEIGHTS = None
    return hd.RacingTileHexDiagProblem()


def measure(p, seed):
    sp = p._content_space
    sp.seed(seed)
    passed, turns = 0, []
    for _ in range(N):
        pts = p._extract_content(sp.sample())
        curve = np.asarray(p._make_curve(p._normalize_track_points(np.asarray(pts, float))))
        t = p._quality_terms({"track_points": pts, "curve_points": curve, "finished": True,
                              "offroad_frac": 0.0})
        turns.append(t["turn_count"])
        passed += p.quality({"track_points": pts, "curve_points": curve, "finished": True,
                             "offroad_frac": 0.0}) >= 0.999
    return passed, float(np.median(turns))


if __name__ == "__main__":
    names = sys.argv[1:] or list(SETTINGS)
    for name in names:
        for grass, straight, curve in SETTINGS[name]:
            res = []
            for seed in SEEDS:
                p = make(name, grass, straight, curve)
                res.append(measure(p, seed))
                del p
                gc.collect()
            print("%-8s grass %g  straight %g  curve %g | pass %s (total %d)  turns %s" % (
                name, grass, straight, curve, " ".join("%2d" % r[0] for r in res),
                sum(r[0] for r in res), " ".join("%.0f" % r[1] for r in res)), flush=True)
