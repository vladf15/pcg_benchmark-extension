"""Square WFC tiles with diagonal roads (racingtilediag-v0).

The square tile representation (racingtile) with roads that may also leave a
cell through one of its four corners, so a track can run in 8 directions
instead of 4.  On 80 random square-tile genomes (seeds 21 and 22) 48% of the
lap runs within 3 degrees of the four grid directions, against 3% on the 24
reference circuits, and every hairpin turns 178-180 degrees (circuits
156-210).  A road through a cell corner joins the two cells that meet there
diagonally, which lets straights run at 45 degrees and corners turn by 45,
90 and 135 degrees between them.

Everything else is racingtile's: the 11 x 11 grid, WFC with one boost gene
per cell (Genetic-WFC, Bailly and Levieux 2023), re-encoding, the shape gene
per cell and the clothoid transitions.  racingtile and racingtilehex are
unchanged.

Ports.  A cell has 8: its faces N, E, S, W (0-3) and its corners NE, SE, SW,
NW (4-7).  A face port joins the face neighbour, a corner port the diagonal
neighbour.  A corner is shared by four cells, so WFC gets two rules on top of
face matching:
  - two diagonal neighbours both use their shared corner or both leave it
    free (the road passes through it from one to the other);
  - two face neighbours never both use a corner they share: a road through
    that corner belongs to the diagonal pair, and the side cells keep it free,
    so no two roads meet at one point.
Both are pairwise, so WFC's AC-3 propagation carries them over the 8
neighbours of each cell unchanged in form.

New tiles, in the canonical frame of racingtile (units of the half cell,
68.2 m, y north; turned in quarter turns clockwise):
  DSTRAIGHT      corner to opposite corner, SW to NE: a 45 degree straight,
                 2 sqrt 2 half cells (193 m).
  DENTRY         face to far corner, W to NE: a straight, a 45 degree arc, a
                 straight along the diagonal; DENTRY_MIRROR goes W to SE.
                 The arc radius is the shape gene, 0.6-1.9 half cells
                 (41-130 m, the fast class), the straights solved so the
                 path ends on the corner.  DENTRY_WIDE and its mirror are the
                 same at 2.12-2.41 (145-164 m, the sweeper class); 2.41 half
                 cells is the arc with no straight before it.
  DCORNER        corner to adjacent corner, SW to NW: a 90 degree turn from
                 one diagonal to the other, radius 0.5-0.85 half cells (34-58
                 m, the medium class) with equal straights; DCORNER_WIDE the
                 same at 0.95-1.35 (65-92 m, the fast class).  1.41 is the arc
                 with no straight.
The face-to-near-corner turn (135 degrees inside one cell) is left out: a
135 degree turn is already a DENTRY followed by a DCORNER, and a single-cell
version would put its arc within a half cell of the corner its road leaves
through.
"""
from __future__ import annotations

import numpy as np

from pcg_benchmark.probs.racingtile.problem import (
    E, GRASS, GRID_H, GRID_W, N, S, STRAIGHT, W, _CURVE_KINDS, _N_WFC_TILES, _OPEN_EDGES, _SHAPE_LEVELS,
    RacingTileProblem, _solve_straights, _walk_smooth)
from pcg_benchmark.probs.racingtile import problem as _square

NE, SE, SW, NW = 4, 5, 6, 7
_PORT_DELTA = {N: (-1, 0), E: (0, 1), S: (1, 0), W: (0, -1),
               NE: (-1, 1), SE: (1, 1), SW: (1, -1), NW: (-1, -1)}
_PORT_OPPOSITE = {N: S, S: N, E: W, W: E, NE: SW, SW: NE, SE: NW, NW: SE}
# For a face neighbour in direction d: the corners the two cells share, as
# (this cell's corner, the neighbour's name for the same point).
_SHARED_CORNERS = {N: ((NE, SE), (NW, SW)), S: ((SE, NE), (SW, NW)),
                   E: ((NE, NW), (SE, SW)), W: ((NW, NE), (SW, SE))}

DSTRAIGHT, DENTRY, DENTRY_MIRROR, DCORNER = 20, 21, 22, 23
DENTRY_WIDE, DENTRY_WIDE_MIRROR, DCORNER_WIDE = 24, 25, 26
_DIAG_KINDS = (DSTRAIGHT, DENTRY, DENTRY_MIRROR, DENTRY_WIDE, DENTRY_WIDE_MIRROR,
               DCORNER, DCORNER_WIDE)
# The wide kinds have their base kind's ports and path, at a larger radius.
_DIAG_BASE = {DENTRY_WIDE: DENTRY, DENTRY_WIDE_MIRROR: DENTRY_MIRROR, DCORNER_WIDE: DCORNER}


def _turn_port(port, rot):
    """A port turned `rot` quarter turns clockwise: faces cycle N-E-S-W,
    corners NE-SE-SW-NW."""
    return (port + rot) % 4 if port < 4 else 4 + (port - 4 + rot) % 4


# Ports of every (kind, rotation): racingtile's face tiles, then the new ones.
_PORTS = dict(_OPEN_EDGES)
for _rot in range(2):
    _PORTS[(DSTRAIGHT, _rot)] = frozenset(_turn_port(p, _rot) for p in (SW, NE))
for _rot in range(4):
    for _k in (DENTRY, DENTRY_WIDE):
        _PORTS[(_k, _rot)] = frozenset(_turn_port(p, _rot) for p in (W, NE))
    for _k in (DENTRY_MIRROR, DENTRY_WIDE_MIRROR):
        _PORTS[(_k, _rot)] = frozenset(_turn_port(p, _rot) for p in (W, SE))
    for _k in (DCORNER, DCORNER_WIDE):
        _PORTS[(_k, _rot)] = frozenset(_turn_port(p, _rot) for p in (SW, NW))
# Entry port at rotation 0, the port a canonical path starts from.
_ENTRY0 = {DSTRAIGHT: SW, DENTRY: W, DENTRY_MIRROR: W, DENTRY_WIDE: W, DENTRY_WIDE_MIRROR: W,
           DCORNER: SW, DCORNER_WIDE: SW}

# Canonical port positions (half cells, y north), for building paths.
_PORT_XY = {N: (0.0, 1.0), E: (1.0, 0.0), S: (0.0, -1.0), W: (-1.0, 0.0),
            NE: (1.0, 1.0), SE: (1.0, -1.0), SW: (-1.0, -1.0), NW: (-1.0, 1.0)}

# Shape gene ranges of the new tiles (arc radius in half cells), set, like
# racingtile's _SHAPE_RANGES, so the corner walker puts each kind in one
# class of the corner census; measured with the tile alone between two
# straights, transitions on, as (turn, mean radius) at the ends of the range:
#   DENTRY       0.60-1.90 (41-130 m): 42-44 degrees at 79-141 m, fast.
#   DENTRY_WIDE  2.12-2.41 (145-164 m): 43-44 degrees at 150-168 m, sweeper;
#                2.41 is the arc with no straight before it.
#   DCORNER      0.50-0.85 (34-58 m): 88-89 degrees at 51-72 m, medium.
#   DCORNER_WIDE 0.95-1.35 (65-92 m): 88-89 degrees at 78-96 m, fast; 1.35
#                keeps the road edge 5.7 m inside the cell (1.41, the arc
#                with no straight, leaves 5.6 m).
# Script: model_training/code/calibration/proto_hx.py.
_DIAG_SHAPE_RANGES = {DENTRY: (0.6, 1.9), DENTRY_MIRROR: (0.6, 1.9),
                      DENTRY_WIDE: (2.12, 2.41), DENTRY_WIDE_MIRROR: (2.12, 2.41),
                      DCORNER: (0.5, 0.85), DCORNER_WIDE: (0.95, 1.35)}

# WFC weights as in racingtile (one weight for every curve tile; grass and
# the two straights their own).  The vocabulary against the census (curve
# tiles | circuits):
#
#   angle       slow          medium        fast          sweeper
#   20-52       0.00 | 0.01   0.04 | 0.04   0.20 | 0.12   0.17 | 0.07
#   52-75       0.01 | 0.06   0    | 0.08   0    | 0.05   0    | 0.01
#   75-105      0.17 | 0.12   0.18 | 0.07   0.06 | 0.06   0    | 0
#   105-127     0.08 | 0.04   0    | 0.03   0    | 0.02   0    | 0
#   127-150     0.04 | 0.05   0    | 0.03   0    | 0.01   0    | 0
#   150 and up  0.06 | 0.05   0    | 0.05   0    | 0.03   0    | 0
#
# The reached cells hold 0.67 of the corners; over them L1 0.36 (with only
# the four diagonal kinds: 0.58, reaching 0.56).  Weights swept as in
# racingtile, plus tracks with a diagonal share of straight length of
# 0.25-0.75 ("mixed"), and passing on geometry ("pass", tile_weights.py):
#
#   grass straights curve    class L1 (s21, s22, s23)   all four      turns        mixed        pass
#     3      4      0.0625   0.17                       8             20           21           18 22 25
#     3      4      0.125    0.19                       8             28           26           24 19 23
#     3      4      0.25     0.21                       3             35           29           23 26 19
#     5      8      0.0625   0.21  0.19  0.20           13  14  10    20  22  24   25  27  21   21 24 21   (in use)
#     8      8      0.0625   0.15  0.25  0.28           10   9  14    20  21  22   23  23  18   20 21 24
#     8      8      0.125    0.17  0.19  0.19            9  12   7    19  23  24   25  28  18   19 25 22
#
# Rule: the most tracks with all four.  Every setting passes 65-68 of 120
# (the same at 12 m), within the 3-7 that separate seeds, so the pass count
# does not choose.  Class L1 is 0.15-0.28 at every setting against 0.72 for
# the four-kind vocabulary (which kept 36-38 of 40 mixed; here DENTRY is one
# of two tiles per face-to-far-corner pair, so tracks switch less, median
# diagonal share 0.37-0.49).  vocab_share.py, mix_sweep.py.
_DIAG_TYPE_WEIGHTS = {GRASS: 5.0, STRAIGHT: 8.0, DSTRAIGHT: 8.0,
                      **{k: 0.0625 for k in _CURVE_KINDS + _DIAG_KINDS[1:]}}

# racingtile's tiles, then DSTRAIGHT x2 and the six curve kinds x4: 77.
_N_DIAG_TILES = _N_WFC_TILES + 2 + 4 * (len(_DIAG_KINDS) - 1)


def _diag_segments(kind, radius):
    """Turtle segments (racingtile's _walk_path form) of a new tile at
    rotation 0, and its start, heading and end.  The two straights around the
    arc are solved so the path ends on its exit port: the end point is affine
    in them, so three walks give the linear system."""
    kind = _DIAG_BASE.get(kind, kind)
    if kind == DSTRAIGHT:
        return [("L", 2.0 * np.sqrt(2.0))], _PORT_XY[SW], 45.0, _PORT_XY[NE]
    if kind == DCORNER:
        start, heading, end, turn = _PORT_XY[SW], 45.0, _PORT_XY[NW], 90.0
    else:
        start, heading = _PORT_XY[W], 0.0
        end, turn = (_PORT_XY[NE], 45.0) if kind == DENTRY else (_PORT_XY[SE], -45.0)

    def segs(a, b):
        return [("L", a), ("A", radius, turn), ("L", b)]

    a, b = _solve_straights(segs, start, heading, end)
    if a < -1e-9 or b < -1e-9:
        raise ValueError("diagonal tile %d at radius %.2f does not fit the cell" % (kind, radius))
    return segs(max(a, 0.0), max(b, 0.0)), start, heading, end


class RacingTileDiagProblem(RacingTileProblem):
    """racingtile with 8 ports per cell; see the module docstring."""

    _WFC_TILES = RacingTileProblem._WFC_TILES + [
        (DSTRAIGHT, 0), (DSTRAIGHT, 1),
        *[(t, r) for t in _DIAG_KINDS[1:] for r in range(4)],
    ]
    _WFC_WEIGHTS = np.array([_DIAG_TYPE_WEIGHTS[t] for t, _ in _WFC_TILES])
    _WFC_COMPAT = None      # this class's own; racingtile keeps its 4-port table
    _WFC_TILE_IDX = None
    _PORTS = _PORTS

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert len(self._WFC_TILES) == _N_DIAG_TILES

    # ── WFC over 8 neighbours ─────────────────────────────────────────

    @classmethod
    def _wfc_compat(cls):
        """compat[d][i]: the tiles that may sit in direction d (a port) of tile
        i.  Face directions: the shared face agrees (both open or both
        closed), and the two cells do not both use a corner they share.
        Corner directions: both use the shared corner or neither does."""
        if cls._WFC_COMPAT is not None:
            return cls._WFC_COMPAT
        ports = [cls._PORTS[t] for t in cls._WFC_TILES]
        compat = {}
        for d in _PORT_DELTA:
            opp = _PORT_OPPOSITE[d]
            compat[d] = []
            for pi in ports:
                allowed = set()
                for j, pj in enumerate(ports):
                    if (d in pi) != (opp in pj):
                        continue
                    if d < 4 and any(a in pi and b in pj for a, b in _SHARED_CORNERS[d]):
                        continue
                    allowed.add(j)
                compat[d].append(frozenset(allowed))
        cls._WFC_COMPAT = compat
        return compat

    def _wfc_propagate(self, wave, stack, compat):
        """racingtile's AC-3 propagation over all 8 neighbours."""
        while stack:
            r, c = stack.pop()
            cur = wave[r][c]
            for d, (dr, dc) in _PORT_DELTA.items():
                nr, nc = r + dr, c + dc
                if not (0 <= nr < GRID_H and 0 <= nc < GRID_W):
                    continue
                allowed = set()
                for ti in cur:
                    allowed |= compat[d][ti]
                new = wave[nr][nc] & allowed
                if not new:
                    return False
                if new != wave[nr][nc]:
                    wave[nr][nc] = new
                    stack.append((nr, nc))
        return True

    # ── Loop tracing over 8 ports ─────────────────────────────────────

    def _build_neighbor_graph(self, types, rotations):
        """Road cells joined through a port both of them open."""
        neighbors = {}
        for r in range(GRID_H):
            for c in range(GRID_W):
                ports = self._PORTS[(int(types[r, c]), int(rotations[r, c]))]
                if not ports:
                    continue
                nbrs = []
                for p in ports:
                    dr, dc = _PORT_DELTA[p]
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < GRID_H and 0 <= nc < GRID_W and _PORT_OPPOSITE[p]
                            in self._PORTS[(int(types[nr, nc]), int(rotations[nr, nc]))]):
                        nbrs.append((nr, nc))
                neighbors[(r, c)] = nbrs
        return neighbors

    @staticmethod
    def _direction_toward(r, c, target_r, target_c):
        """The port that steps from (r, c) to the target cell."""
        for p, (dr, dc) in _PORT_DELTA.items():
            if r + dr == target_r and c + dc == target_c:
                return p
        return None

    # ── Road geometry of the new tiles ────────────────────────────────

    def _tile_polyline(self, tile_type, rotation, shape=0.5):
        """racingtile's _tile_polyline for its own tiles; the new ones are
        built from _diag_segments, with the same clothoid transitions and
        the same rotation into the map frame."""
        kind = int(tile_type)
        if kind not in _DIAG_KINDS:
            return super()._tile_polyline(tile_type, rotation, shape)
        q = round(float(shape) * _SHAPE_LEVELS) / _SHAPE_LEVELS
        key = (kind, int(rotation), q)
        if key in self._tile_polylines:
            return self._tile_polylines[key]
        hx = 0.5 * self._width / GRID_W
        hy = 0.5 * self._height / GRID_H
        lo, hi = _DIAG_SHAPE_RANGES.get(kind, (0.0, 0.0))
        segs, start, heading, end = _diag_segments(kind, lo + q * (hi - lo))
        step = 0.5 * self._curve_step() / hx
        if kind == DSTRAIGHT:
            pts = np.array([start, end], dtype=float)
        else:
            pts = _walk_smooth(segs, start, heading, _square._TRANSITION_M / hx, step, end)
        for _ in range(int(rotation)):
            pts = np.column_stack([pts[:, 1], -pts[:, 0]])
        pts = np.column_stack([pts[:, 0] * hx, -pts[:, 1] * hy])
        entry = _turn_port(_ENTRY0[kind], int(rotation))
        self._tile_polylines[key] = (entry, pts)
        return self._tile_polylines[key]
