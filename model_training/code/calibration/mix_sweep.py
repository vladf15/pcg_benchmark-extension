"""Built corner mix of tile tracks at given curve weights, one line per
weight, against the census of the 24 reference circuits.
    python memcap.py 1500 mix_sweep.py REP SEED N W1,W2,...    (W = curve weight; 'cur' = as in code)
Per line: class shares (hairpin, slow, medium, fast, sweeper) and their L1
from the census; angle-bin L1; joint (angle x radius) L1; tracks with full
marks on the three regulations and typicality (SC); median turns
and lap; tracks failing the overlap or bounds gate; for diag and hexdiag the
median corner-heading share of the straights and the tracks with it in
0.25-0.75 ("mixed"); seconds per genome."""
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
REPO = _os.path.normpath(_os.path.join(HERE, '..', '..', '..'))
import gc, json, sys, time, warnings, numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, REPO)
from corner_classes import corners

REP, SEED, NG = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
WS = sys.argv[4].split(",") if len(sys.argv) > 4 else ["cur"]
EDGES = [20, 52.5, 75, 105, 127.5, 150, 1e9]
SC = ("length_score", "straight_max_score", "start_to_corner_score", "typicality_score")   # the regulations and typicality


def joint(turn, rad):
    j = np.zeros((6, 4))
    for t, r in zip(turn, rad):
        a = np.searchsorted(EDGES, t, side="right") - 1
        j[a, 0 if r < 44 else 1 if r < 75 else 2 if r < 150 else 3] += 1
    return j


cen = np.array(json.load(open(_os.path.join(HERE, "census_corners.json"))), dtype=object)
CJ = joint(cen[:, 0].astype(float), cen[:, 1].astype(float)); CJ /= CJ.sum()
CC = np.r_[CJ[5].sum(), CJ[:5].sum(0)]

import pcg_benchmark.probs.racingtile.problem as SQ
import pcg_benchmark.probs.racingtilediag.problem as D
import pcg_benchmark.probs.racingtilehex.problem as HX
import pcg_benchmark.probs.racingtilehexdiag.problem as HD


def build(w):
    if REP in ("tile", "diag"):
        if w != "cur":
            w, st, gr = (float(x) for x in (w.split(":") + ["4", "3"][len(w.split(":")) - 1:])[:3])
            SQ._TYPE_WEIGHTS[SQ.STRAIGHT] = st; D._DIAG_WEIGHTS[D.DSTRAIGHT] = st; SQ._TYPE_WEIGHTS[SQ.GRASS] = gr
            for k in SQ._CURVE_KINDS:
                SQ._TYPE_WEIGHTS[k] = w
            for k in D._DIAG_KINDS[1:]:
                D._DIAG_WEIGHTS[k] = w
            SQ.RacingTileProblem._WFC_WEIGHTS = np.array([SQ._TYPE_WEIGHTS[t] for t, _ in SQ.RacingTileProblem._WFC_TILES])
            D.RacingTileDiagProblem._WFC_WEIGHTS = np.array([D._DIAG_WEIGHTS[t] if t in D._DIAG_KINDS else SQ._TYPE_WEIGHTS[t]
                                                             for t, _ in D.RacingTileDiagProblem._WFC_TILES])
        return SQ.RacingTileProblem() if REP == "tile" else D.RacingTileDiagProblem()
    if w != "cur":
        w, st, gr = (float(x) for x in (w.split(":") + ["10", "4"][len(w.split(":")) - 1:])[:3])
        HX._CURVE_WEIGHT = w; HX._STRAIGHT_WEIGHT = st; HX._GRASS_WEIGHT = gr
        for k in list(HD._NEW_WEIGHTS):
            HD._NEW_WEIGHTS[k] = st if k == HD.VSTRAIGHT else w
    HX.RacingTileHexProblem._WFC_WEIGHTS = None
    HD.RacingTileHexDiagProblem._WFC_WEIGHTS = None
    return HX.RacingTileHexProblem() if REP == "hex" else HD.RacingTileHexDiagProblem()


for w in WS:
    p = build(w); sp = p._content_space; sp.seed(SEED)
    J = np.zeros((6, 4)); sc, turns, laps, gate, share = [], [], [], 0, []
    t0 = time.perf_counter()
    for _ in range(NG):
        c = sp.sample()
        raw = p._normalize_track_points(p._extract_content(c)); curve = np.asarray(p._make_curve(raw))
        t = p._quality_terms({"track_points": raw, "curve_points": curve})
        sc.append([t[k] > 0.999 for k in SC]); turns.append(t["turn_count"]); laps.append(t["length_m"])
        gate += int(t["self_overlaps"] > 0 or t["oob_score"] < 0.999)
        tu, ra = corners(p, curve); J += joint(tu, ra)
        if REP in ("diag", "hexdiag"):
            seg = np.diff(curve, axis=0); ln = np.linalg.norm(seg, axis=1)
            per, off = (90.0, 45.0) if REP == "diag" else (60.0, 30.0)
            hd = np.rad2deg(np.arctan2(seg[:, 1], seg[:, 0])) % per
            ax = ln[(hd < 3) | (hd > per - 3)].sum(); dg = ln[np.abs(hd - off) < 3].sum()
            share.append(dg / max(ax + dg, 1e-9))
    dt = (time.perf_counter() - t0) / NG
    Jn = J / J.sum(); cls = np.r_[Jn[5].sum(), Jn[:5].sum(0)]
    sc = np.array(sc); share = np.array(share)
    mixed = "" if not len(share) else " | diag share p50 %.2f mixed %2d" % (np.median(share), int(((share >= 0.25) & (share <= 0.75)).sum()))
    print("%-7s w %-6s | classes %s L1 %.2f | angle L1 %.2f | joint L1 %.2f | all four %2d | turns %2.0f | lap %4.0f | gate %d%s | %.2f s" % (
        REP, w, np.round(cls, 2).tolist(), np.abs(cls - CC).sum(), np.abs(Jn.sum(1) - CJ.sum(1)).sum(), np.abs(Jn - CJ).sum(),
        int(sc.all(1).sum()), np.median(turns), np.median(laps), gate, mixed, dt), flush=True)
    del p; gc.collect()
