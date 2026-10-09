"""Every tile of the four tile problems: the road edge's clearance to its own
cell boundary away from the ports (centre line 20 m or more from a port),
and whether the solved straights of the corner paths are all non-negative."""
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
REPO = _os.path.normpath(_os.path.join(HERE, '..', '..', '..'))
import sys, warnings, numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, REPO)
import pcg_benchmark.probs.racingtile.problem as SQ
import pcg_benchmark.probs.racingtilediag.problem as D
import pcg_benchmark.probs.racingtilehex.problem as HX
import pcg_benchmark.probs.racingtilehexdiag.problem as HD
HALF_W = 8.0

orig = SQ._walk_path
neg = []
def spy(segs, *a, **k):
    for sg in segs:
        if sg[0] == "L" and sg[1] < -1e-9:
            neg.append(sg[1])
    return orig(segs, *a, **k)

def clear_sq(pts, h):
    d = h - np.max(np.abs(pts), axis=1)
    return d

def report(name, rows):
    rows = sorted(rows)
    print("%-10s worst clearance %.1f m (%s)  tiles %d" % (name, rows[0][0], rows[0][1], len(rows)))
    for r in rows[:4]:
        print("     %.1f m  %s" % r)

# square and diag
for M, cls in ((SQ, SQ.RacingTileProblem), (D, D.RacingTileDiagProblem)):
    p = cls(); h = 0.5 * p._width / SQ.GRID_W; rows = []
    for t, r in p._WFC_TILES:
        if t == 0:
            continue
        for g in np.linspace(0, 1, 5):
            SQ._walk_path = spy; D._walk_path = spy
            e, pts = p._tile_polyline(t, r, g)
            SQ._walk_path = orig; D._walk_path = orig
            ends = [pts[0], pts[-1]]
            far = np.array([min(np.linalg.norm(q - x) for x in ends) >= 20.0 for q in pts])
            if far.any():
                rows.append((float(clear_sq(pts[far], h).min() - HALF_W), "%s rot %d gene %.2f" % (t, r, g)))
    report(cls.__name__, rows)
    del p
print("negative straights seen in square and diag paths:", len(neg), neg[:5])

for cls in (HX.RacingTileHexProblem, HD.RacingTileHexDiagProblem):
    p = cls(); ap = p._hex_size() * np.sqrt(3) / 2; rows = []
    normals = [np.array([np.cos(HX._FACE_ANGLE[d]), np.sin(HX._FACE_ANGLE[d])]) for d in HX._DIRS]
    n = HD._N_TILES if cls is HD.RacingTileHexDiagProblem else HX._N_WFC_TILES
    for t in range(1, n):
        e, pts = p._tile_polyline(t)
        ends = [pts[0], pts[-1]]
        far = np.array([min(np.linalg.norm(q - x) for x in ends) >= 20.0 for q in pts])
        if not far.any():
            continue
        d = ap - np.max(np.stack([pts[far] @ nn for nn in normals]), axis=0)
        rows.append((float(d.min() - HALF_W), "tile %d %s" % (t, HX._TILE_KIND.get(t) if t < HX._N_WFC_TILES else HD._NEW_TILES[t])))
    report(cls.__name__, rows)
    del p
