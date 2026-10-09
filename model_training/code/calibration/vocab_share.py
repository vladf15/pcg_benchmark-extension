"""Share of each representation's curve tiles (gene values, all orientations)
per census cell (angle bin x radius class), against the census of the 24
reference circuits.  A tile goes to the cell of its largest corner, as the
walker counts it alone between two straights; square tiles are averaged over
shape genes 0, 0.25, ..., 1.
    python memcap.py 1500 vocab_share.py REP [old]"""
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
REPO = _os.path.normpath(_os.path.join(HERE, '..', '..', '..'))
import json, sys, warnings, numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, REPO)
from tile_catalog_fn import tile_corners
REP = sys.argv[1]
EDGES = [20, 52.5, 75, 105, 127.5, 150, 1e9]
ALAB = ["20-52", "52-75", "75-105", "105-127", "127-150", ">=150"]
CLAB = ["slow", "medium", "fast", "sweeper"]


def cell(t, r):
    a = int(np.searchsorted(EDGES, t, side="right") - 1)
    return a, (0 if r < 44 else 1 if r < 75 else 2 if r < 150 else 3)


cen = np.array(json.load(open(_os.path.join(HERE, "census_corners.json"))), dtype=object)
C = np.zeros((6, 4))
for t, r in zip(cen[:, 0].astype(float), cen[:, 1].astype(float)):
    C[cell(t, r)] += 1
C /= C.sum()

V = np.zeros((6, 4)); names = {}
if REP in ("tile", "diag"):
    import pcg_benchmark.probs.racingtile.problem as SQ
    if REP == "tile":
        p = SQ.RacingTileProblem()
    else:
        import pcg_benchmark.probs.racingtilediag.problem as D; p = D.RacingTileDiagProblem()
    straight = {SQ.STRAIGHT, getattr(sys.modules.get("pcg_benchmark.probs.racingtilediag.problem"), "DSTRAIGHT", -1)}
    for t, r in p._WFC_TILES:
        if t == 0 or t in straight:
            continue
        for g in np.linspace(0, 1, 5):
            cs = tile_corners(p, p._tile_polyline(t, r, g)[1])
            if cs:
                big = max(cs, key=lambda x: x[0]); V[cell(*big)] += 0.2
                names.setdefault(cell(*big), set()).add(t)
else:
    import pcg_benchmark.probs.racingtilehex.problem as HX
    if REP == "hex":
        p = HX.RacingTileHexProblem(); n = HX._N_WFC_TILES; NEW = {}
    else:
        import pcg_benchmark.probs.racingtilehexdiag.problem as HD
        p = HD.RacingTileHexDiagProblem(); n = HD._N_TILES; NEW = HD._NEW_TILES
    for t in range(1, n):
        if t in NEW:
            if NEW[t][0] == HD.VSTRAIGHT:
                continue
        elif HX._TILE_KIND[t][0] == HX.STRAIGHT:
            continue
        cs = tile_corners(p, p._tile_polyline(t)[1])
        if cs:
            big = max(cs, key=lambda x: x[0]); V[cell(*big)] += 1
V /= V.sum()
reach = V > 0
print("%s: curve-tile share per cell | census share   (cells the vocabulary reaches: census %.2f of corners)" % (REP, C[reach].sum()))
print("angle bin  " + "".join("%-15s" % c for c in CLAB))
for i in range(6):
    print("%-9s  " % ALAB[i] + "".join("%.3f | %.3f    " % (V[i, k], C[i, k]) for k in range(4)))
Cr = np.where(reach, C, 0); Cr /= Cr.sum()
print("L1 against the census over the reached cells, renormalised: %.2f;  against the whole census: %.2f" % (np.abs(V - Cr).sum(), np.abs(V - C).sum()))
print("angle bins: vocab %s census %s" % (np.round(V.sum(1), 2).tolist(), np.round(C.sum(1), 2).tolist()))
