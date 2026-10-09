"""Square tiles: sweep transition length and tile weights; geometry and lap-simulation measures on 40 genomes.
    python memcap.py 1500 tile_sweep.py MODE [SEED]    MODE = transition | weights"""
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
REPO = _os.path.normpath(_os.path.join(HERE, '..', '..', '..'))
import gc, sys, warnings, numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, REPO)
sys.argv = [sys.argv[0]] + sys.argv[1:]
MODE = sys.argv[1]; SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 21
src = open(_os.path.join(HERE, "qcand4.py")).read(); src = src[:src.index("base = pickle.load")].replace("seed = int(sys.argv[1]); ", "")
exec(src)                      # qss(), lap_measures(), structure(), corners via P
from corner_classes import CENSUS
import pcg_benchmark.probs.racingtile.problem as T
SC = ("length_score", "straight_max_score", "start_to_corner_score", "typicality_score")   # the regulations and typicality
CIRC_LIMIT = 0.052

def run(label):
    p = T.RacingTileProblem(); sp = p._content_space; sp.seed(SEED)
    lim, rep, hp, allc, sc, turns, laps, shapes = [], [], [], [], [], [], [], []
    for _ in range(40):
        c = sp.sample(); raw = p._normalize_track_points(p._extract_content(c)); curve = p._make_curve(raw)
        t = p._quality_terms({"track_points": raw, "curve_points": curve})
        sc.append([t[k] for k in SC]); turns.append(t["turn_count"]); laps.append(t["length_m"])
        lim.append(lap_measures(np.asarray(curve))["qss_limit_time"])
        st, tt, rr = structure(np.asarray(curve)); rep.append(st["repeat_share"])
        allc += list(zip(tt, rr)); hp += [(a, b) for a, b in zip(tt, rr) if a >= 150]
        shapes.append(len({(round(a / 10), round(b / 5)) for a, b in zip(tt, rr)}) / max(len(tt), 1))
    tt = np.array([a for a, _ in allc]); rr = np.array([b for _, b in allc]); h = tt >= 150
    sh = np.array([h.mean(), ((rr < 44) & ~h).mean(), ((rr >= 44) & (rr < 75) & ~h).mean(), ((rr >= 75) & (rr < 150) & ~h).mean(), ((rr >= 150) & ~h).mean()])
    sc = np.array(sc)
    print("%-28s limit time p50 %.3f (circuits %.3f) | repeat %.2f | shapes/corner %.2f | hairpins %d, %d shapes | mix %s L1 %.2f | turns p50 %.0f | lap p50 %4.0f | all four %2d | longest straight ok %2d"
          % (label, np.median(lim), CIRC_LIMIT, np.median(rep), np.median(shapes), len(hp), len({(round(a / 10), round(b / 2)) for a, b in hp}),
             np.round(sh, 2).tolist(), np.abs(sh - CENSUS).sum(), np.median(turns), np.median(laps), int(np.all(sc > 0.999, 1).sum()), int((sc[:, 2] > 0.999).sum())), flush=True)
    del p; gc.collect()

if MODE == "transition":
    for ell in (0.0, 10.0, 20.0, 30.0, 40.0):
        T._TRANSITION_M = ell
        run("transition %2.0f m" % ell)
else:
    base = dict(T._TYPE_WEIGHTS)
    T._TRANSITION_M = float(sys.argv[3]) if len(sys.argv) > 3 else 40.0
    grid = [dict(straight=s_, corner=c_, chicane=0.05, others=0.0, grass=g_)
            for g_ in (3.0, 5.0) for s_ in (4.0, 8.0, 12.0) for c_ in (0.5, 1.0)]
    for g in grid:
        w = dict(base); w[T.STRAIGHT] = g["straight"]; w[T.CORNER] = g["corner"]; w[T.CHICANE] = g["chicane"]; w[T.GRASS] = g["grass"]
        for k in (T.SHARP, T.HAIRPIN, T.KINK, T.KINK_MIRROR, T.S_CHICANE): w[k] = g["others"]
        T.RacingTileProblem._WFC_WEIGHTS = np.array([w[t] for t, _ in T.RacingTileProblem._WFC_TILES])
        run("gr %.0f st %.0f co %.1f" % (g["grass"], g["straight"], g["corner"]))
