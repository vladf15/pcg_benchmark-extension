"""Hex WFC tiles with roads through cell corners (racingtilehexdiag-v0).

racingtilehex with roads that may also leave a cell through one of its six
corners, as racingtilediag does for the square tiles.  A hex road crosses a
face along the face normal, so the hex tiles run in 6 headings (0, 60 and
120 degrees, each both ways) and turn by 60 or 120 degrees.  A road through
a corner runs along the corner's radial, 30 degrees off the face normals,
which gives 12 headings and turns of 30 and 90 degrees as well.

Everything else is racingtilehex's: the 11 x 11 odd-r grid, the cell size,
WFC with one boost gene per cell (Genetic-WFC, Bailly and Levieux 2023),
re-encoding, and ids 0-105; the weights are this module's own
(_NEW_WEIGHTS).  racingtilehex is unchanged.

Ports.  A cell has 12: its faces NE, E, SE, SW, W, NW (0-5, racingtilehex's
numbering) and its corners (6-11), corner k at -30 + 60 k degrees (y down),
between face k and face k + 1.  Three cells meet at a corner, so unlike the
square grid there is no cell diagonally across it: the line from a cell's
centre through its corner k continues along the edge between its face
neighbours k and k + 1, and reaches the far corner of that edge, which
belongs to the cell two steps out (the neighbour of neighbour k across face
k + 1).  A corner port joins those two cells, and the road between them runs
one edge length (one hex size, 79 m at the 68.4 m apothem) along the shared
edge of the two side cells, half of the road in each.  WFC gets three rules
on top of face matching:
  - the two cells a corner port joins both use their ends of the edge or
    neither does;
  - two face neighbours never both use a corner they share, so no two roads
    meet at a point;
  - when a cell uses a corner, the two side cells keep their own road at
    least _EDGE_CLEARANCE_M from the edge the corner road runs along.
All three are pairwise, between a cell and one of its 12 port neighbours, so
AC-3 propagation carries them as racingtilehex's face rule does.

New tiles, 51: one per unordered port pair (like racingtilehex's 15 plain
pairs), each the largest arc the cell holds between its two ports, as the
plain hex tiles are, and a tighter bend on each face to far corner pair:
  VSTRAIGHT   corner to opposite corner: a straight through the centre, 2
              sizes (158 m), 3 tiles.
  VBEND       corner to the corner two along: 60 degree turn, radius 2.00
              apothems (137 m), 6 tiles.
  VCORNER     corner to the next corner: 120 degree turn, radius 0.67 (46
              m), 6 tiles.
  FVKINK      face to far corner: 30 degree turn, radius 3.73 (255 m), then
              0.15 apothems straight to the corner, 12 tiles.
  FVCORNER    face to middle corner: 90 degree turn, radius 1.00 (68 m), then
              0.15 straight, 12 tiles.
  FVKINK_TIGHT  face to far corner at half FVKINK's radius, 1.87 (128 m),
              with straights either side, 12 tiles (see _RADIUS_SHARE).
Largest arc: tangent to the two ports' radials, which cross at the cell
centre, with the tangent point on the nearer port, so the radius is that
port's distance from the centre over tan(turn / 2).  The face to near
corner pair (150 degree turn) is left out, as racingtilediag leaves out its
face to near corner turn: the radius would be 0.27 apothems (18 m), under the
44 m slow corner class edge, and the road would leave through a corner of
the face it came in through.
"""
from __future__ import annotations

import numpy as np

from pcg_benchmark.probs.racingtile.problem import _solve_straights, _walk_path
from pcg_benchmark.probs.racingtilehex.problem import (
    GRID_H, GRID_W, STRAIGHT, _DIRS, _OPEN_EDGES as _HEX_OPEN_EDGES, _TILE_KIND, RacingTileHexProblem,
    _neighbor)

_SIZE = 2.0 / np.sqrt(3.0)          # centre to corner, in apothems
_CORNERS = tuple(range(6, 12))


def _port_angle(p):
    """Angle (radians, y down) from the cell centre to port p."""
    return np.deg2rad(-60.0 + 60.0 * p) if p < 6 else np.deg2rad(-30.0 + 60.0 * (p - 6))


def _port_xy(p):
    """Port p in apothems from the cell centre: a face midpoint or a corner."""
    a = _port_angle(p)
    return (1.0 if p < 6 else _SIZE) * np.array([np.cos(a), np.sin(a)])


def _port_opposite(p):
    return (p + 3) % 6 if p < 6 else 6 + (p - 6 + 3) % 6


def _port_neighbor(r, c, p):
    """The cell port p joins: the face neighbour, or for corner k the cell
    two steps out along the corner's radial."""
    if p < 6:
        return _neighbor(r, c, p)
    k = p - 6
    nr, nc = _neighbor(r, c, k)
    return _neighbor(nr, nc, (k + 1) % 6)


# For face d, the corners the cell shares with its neighbour across d, each as
# (this cell's corner port, the neighbour's face the corner road runs along).
# Corner d sits between faces d and d + 1: its road runs along the edge
# between the neighbours across d and d + 1, which is face d + 2 of the first.
# Corner d - 1 sits between faces d - 1 and d: its road runs along face d + 4
# of the neighbour across d.
_SHARED = {d: ((6 + d, (d + 2) % 6), (6 + (d - 1) % 6, (d + 4) % 6)) for d in _DIRS}

VSTRAIGHT, VBEND, VCORNER, FVKINK, FVCORNER = "vstraight", "vbend", "vcorner", "fvkink", "fvcorner"
FVKINK_TIGHT = "fvkinktight"

# Arc radius of each new kind as a share of the largest arc that fits between
# its ports (_new_tile_path); 1 unless listed.  Each kind is placed in one
# class of the corner census (racingtile _TYPE_WEIGHTS), measured with the
# tile alone between two straights as (turn, mean radius): VBEND 58 degrees
# at 138 m (fast), VCORNER 117 at 49 m (medium), FVKINK 29 at 256 m
# (sweeper), FVKINK_TIGHT at 0.5 of that arc 30 at 135 m (fast), FVCORNER 88
# at 71 m (medium).  Script: model_training/code/calibration/proto_hx.py.
_RADIUS_SHARE = {FVKINK_TIGHT: 0.5}

# id -> ports, ids 0-105 racingtilehex's; id -> (kind, entry port, exit port)
# for the new ones, entry the lower port.
_TILE_PORTS = dict(_HEX_OPEN_EDGES)
_NEW_TILES = {}


def _add(kind, a, b):
    a, b = sorted((a, b))
    _NEW_TILES[len(_TILE_PORTS)] = (kind, a, b)
    _TILE_PORTS[len(_TILE_PORTS)] = frozenset((a, b))


for _k in range(3):
    _add(VSTRAIGHT, 6 + _k, 6 + _k + 3)
for _k in range(6):
    _add(VBEND, 6 + _k, 6 + (_k + 2) % 6)
for _k in range(6):
    _add(VCORNER, 6 + _k, 6 + (_k + 1) % 6)
for _f in range(6):
    _add(FVKINK, _f, 6 + (_f + 2) % 6)
    _add(FVKINK, _f, 6 + (_f + 3) % 6)
    _add(FVCORNER, _f, 6 + (_f + 1) % 6)
    _add(FVCORNER, _f, 6 + (_f - 2) % 6)
for _f in range(6):
    _add(FVKINK_TIGHT, _f, 6 + (_f + 2) % 6)
    _add(FVKINK_TIGHT, _f, 6 + (_f + 3) % 6)
_N_TILES = len(_TILE_PORTS)
assert _N_TILES == len(_HEX_OPEN_EDGES) + 51

# WFC weights, this class's own, as in racingtilehex (one weight for every
# curve tile).  The vocabulary against the census (curve tiles | circuits):
#
#   angle       slow          medium        fast          sweeper
#   20-52       0    | 0.01   0    | 0.04   0.16 | 0.12   0.12 | 0.07
#   52-75       0.04 | 0.06   0.08 | 0.08   0.08 | 0.05   0    | 0.01
#   75-105      0.12 | 0.12   0.16 | 0.07   0    | 0.06   0    | 0
#   105-127     0    | 0.04   0.08 | 0.03   0    | 0.02   0    | 0
#   127-150     0.08 | 0.05   0    | 0.03   0    | 0.01   0    | 0
#   150 and up  0.08 | 0.05   0    | 0.05   0    | 0.03   0    | 0
#
# The reached cells hold 0.69 of the corners; over them L1 0.30 (one tile per
# port pair: 0.32, reaching 0.56).  Weights swept as in racingtile, plus
# tracks with a corner share of straight length of 0.25-0.75 ("mixed"):
#
#   grass straights curve   class L1 (s21, s22, s23)   all four      turns        mixed
#     4     10      0.1     0.38  0.35  0.37            9   7   6    24  25  26   32  32  28
#     4     10      0.05    0.34  0.37  0.36            9   4   9    24  23  22   35  23  23
#     8     20      0.05    0.28  0.30  0.34            9  20  15    15  20  16   25  25  22   (in use)
#
# Rule: the most tracks with all four (as defined in racingtile).  Through
# the current quality function's gates, rules and typicality
# (calibration/tile_weights.py, 12 m road) the three pass 0, 1 and 0 of
# 120, too few for that count to choose.  One tile per port pair at its hex
# counterpart's weight gives class L1 0.26 but joint L1 0.97 (0.57-0.64
# here), 22-26 with all four and 35-38 mixed; none here keeps 35 mixed (each
# new pair has one or two tiles against a hex pair's four to nine).
# vocab_share.py, mix_sweep.py, tile_weights.py.
_GRASS_WEIGHT = 8.0
_STRAIGHT_WEIGHT = 20.0
_CURVE_WEIGHT = 0.05
_NEW_WEIGHTS = {VSTRAIGHT: _STRAIGHT_WEIGHT,
                **{k: _CURVE_WEIGHT for k in (VBEND, VCORNER, FVKINK, FVCORNER, FVKINK_TIGHT)}}

# Minimum distance from a side cell's road centre line to the edge a corner
# road runs along: half a road width for each road (6 m each at the 12 m
# road) plus 5.7 m, the clearance racingtilehex keeps between a road edge
# and its cell edge (see _HAIRPIN_LEG there).  Here the corner road's edge
# takes the place of the cell edge.
_EDGE_CLEARANCE_EXTRA_M = 5.7


def _new_tile_path(kind, entry, exit_, n_arc):
    """Centre line (apothems, y down) of a new tile from its entry port to its
    exit port: a straight, the largest arc that fits, a straight, with the
    straights solved so the path ends on the exit port (the end point is
    affine in them)."""
    start, end = _port_xy(entry), _port_xy(exit_)
    if kind == VSTRAIGHT:
        return np.array([start, end])
    heading = np.rad2deg(_port_angle(entry)) + 180.0
    turn = (np.rad2deg(_port_angle(exit_)) - heading + 180.0) % 360.0 - 180.0
    radius = (_RADIUS_SHARE.get(kind, 1.0) * min(np.linalg.norm(start), np.linalg.norm(end))
              / np.tan(np.deg2rad(abs(turn)) / 2.0))

    def segs(a, b):
        return [("L", a), ("A", radius, turn), ("L", b)]

    a, b = _solve_straights(segs, start, heading, end)
    if a < -1e-9 or b < -1e-9:
        raise ValueError("tile %s %d-%d does not fit the cell" % (kind, entry, exit_))
    pts = _walk_path(segs(max(a, 0.0), max(b, 0.0)), start, heading, n_arc)
    pts[-1] = end
    return pts


class RacingTileHexDiagProblem(RacingTileHexProblem):
    """racingtilehex with 12 ports per cell; see the module docstring."""

    _TILE_PORTS = _TILE_PORTS
    _WFC_WEIGHTS = None     # this class's own; racingtilehex keeps its table

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._compat = None
        # Per cell, the tiles whose port neighbours are all on the grid.
        self._on_grid = [[frozenset(t for t in range(_N_TILES) if all(
            0 <= _port_neighbor(r, c, p)[0] < GRID_H and 0 <= _port_neighbor(r, c, p)[1] < GRID_W
            for p in _TILE_PORTS[t])) for c in range(GRID_W)] for r in range(GRID_H)]

    @classmethod
    def _weights(cls):
        if cls._WFC_WEIGHTS is None:
            w = np.zeros(_N_TILES)
            w[0] = _GRASS_WEIGHT
            for i in range(1, len(_HEX_OPEN_EDGES)):
                w[i] = _STRAIGHT_WEIGHT if _TILE_KIND[i][0] == STRAIGHT else _CURVE_WEIGHT
            for i, (kind, _, _) in _NEW_TILES.items():
                w[i] = _NEW_WEIGHTS[kind]
            cls._WFC_WEIGHTS = w
        return cls._WFC_WEIGHTS

    # ── WFC over 12 port neighbours ───────────────────────────────────

    def _clear_faces(self):
        """For each tile, the faces of its cell its road keeps at least the
        edge clearance from (see _EDGE_CLEARANCE_EXTRA_M).  Measured on the
        tile's centre line, resampled to 1 m, against each edge segment."""
        need = float(self._track_width) + _EDGE_CLEARANCE_EXTRA_M
        size = self._hex_size()
        corners = [size * np.array([np.cos(_port_angle(6 + k)), np.sin(_port_angle(6 + k))])
                   for k in range(6)]
        out = []
        for t in range(_N_TILES):
            if not _TILE_PORTS[t]:
                out.append(frozenset(_DIRS))
                continue
            pts = self._resample_uniform(self._tile_polyline(t)[1], step=1.0)
            clear = set()
            for f in _DIRS:
                a, b = corners[(f - 1) % 6], corners[f]      # face f runs corner f-1 to f
                u = np.clip((pts - a) @ (b - a) / ((b - a) @ (b - a)), 0.0, 1.0)
                if np.linalg.norm(pts - (a + u[:, None] * (b - a)), axis=1).min() >= need:
                    clear.add(f)
            out.append(frozenset(clear))
        return out

    def _wfc_compat(self):
        """compat[p][i]: the tiles that may sit across port p of tile i.
        Faces: the shared face agrees, and neither cell breaks the corner
        rules for the corners the two share.  Corners: both use their ends
        of the edge or neither does."""
        if self._compat is not None:
            return self._compat
        ports = [_TILE_PORTS[t] for t in range(_N_TILES)]
        clear = self._clear_faces()

        def breaks(i, j, d):
            # Tile i, with tile j across its face d.
            for corner, face_of_j in _SHARED[d]:
                if corner in ports[i] and (face_of_j not in clear[j] or
                                           _port_opposite(corner) in ports[j]):
                    return True
            return False

        compat = {}
        for p in range(12):
            opp = _port_opposite(p)
            compat[p] = []
            for i in range(_N_TILES):
                allowed = set()
                for j in range(_N_TILES):
                    if (p in ports[i]) != (opp in ports[j]):
                        continue
                    if p < 6 and (breaks(i, j, p) or breaks(j, i, opp)):
                        continue
                    allowed.add(j)
                compat[p].append(frozenset(allowed))
        self._compat = compat
        return compat

    def _wfc_propagate(self, wave, stack, compat):
        """racingtilehex's AC-3 propagation over all 12 port neighbours.  A
        corner port's partner is two steps out, so from a cell next to the
        grass border it can lie off the grid, where no neighbour rule reaches;
        the popped cell drops those tiles first.  Every such cell is popped
        in the first propagation, from the border, since its face toward the
        border closes."""
        while stack:
            r, c = stack.pop()
            cur = wave[r][c] & self._on_grid[r][c]
            if not cur:
                return False
            wave[r][c] = cur
            for p in range(12):
                nr, nc = _port_neighbor(r, c, p)
                if not (0 <= nr < GRID_H and 0 <= nc < GRID_W):
                    continue
                allowed = set()
                for ti in cur:
                    allowed |= compat[p][ti]
                new = wave[nr][nc] & allowed
                if not new:
                    return False
                if new != wave[nr][nc]:
                    wave[nr][nc] = new
                    stack.append((nr, nc))
        return True

    # ── Loop tracing over 12 ports ────────────────────────────────────

    def _build_neighbor_graph(self, tiles):
        """Road cells joined through a port both of them open."""
        neighbors = {}
        for r in range(GRID_H):
            for c in range(GRID_W):
                ports = _TILE_PORTS[int(tiles[r, c])]
                if not ports:
                    continue
                nbrs = []
                for p in ports:
                    nr, nc = _port_neighbor(r, c, p)
                    if (0 <= nr < GRID_H and 0 <= nc < GRID_W
                            and _port_opposite(p) in _TILE_PORTS[int(tiles[nr, nc])]):
                        nbrs.append((nr, nc))
                neighbors[(r, c)] = nbrs
        return neighbors

    def _direction_toward(self, r, c, target_r, target_c):
        """The port that steps from (r, c) to the target cell."""
        for p in range(12):
            if _port_neighbor(r, c, p) == (target_r, target_c):
                return p
        return None

    def _port_point(self, r, c, p):
        cx, cy = self._hex_center(r, c)
        x, y = self._hex_size() * np.sqrt(3.0) / 2.0 * _port_xy(p)
        return cx + x, cy + y

    def _edge_midpoints(self, cells):
        """One waypoint per cell: the port toward the next cell."""
        points = []
        for i, (r, c) in enumerate(cells):
            nr, nc = cells[(i + 1) % len(cells)]
            p = self._direction_toward(r, c, nr, nc)
            points.append(self._hex_center(r, c) if p is None else self._port_point(r, c, p))
        return points

    def _loop_to_track_points(self, loop, tiles):
        """racingtilehex's _loop_to_track_points, except that a tile entered
        through a corner keeps its first point: the previous tile ended at
        the other end of the edge between them, one hex size away."""
        n = len(loop)
        points = []
        for i, (r, c) in enumerate(loop):
            pr, pc = loop[(i - 1) % n]
            entry, local = self._tile_polyline(tiles[r, c])
            came = self._direction_toward(r, c, pr, pc)
            if came != entry:
                local = local[::-1]
            cx, cy = self._hex_center(r, c)
            points.extend((cx + x, cy + y) for x, y in local[(1 if came < 6 else 0):])
        return points

    # ── Road geometry of the new tiles ────────────────────────────────

    def _tile_polyline(self, tile):
        """racingtilehex's _tile_polyline for ids 0-105; the new tiles from
        _new_tile_path, in metres from the cell centre."""
        tile = int(tile)
        if tile not in _NEW_TILES:
            return super()._tile_polyline(tile)
        if tile in self._tile_polylines:
            return self._tile_polylines[tile]
        apothem = self._hex_size() * np.sqrt(3.0) / 2.0
        kind, entry, exit_ = _NEW_TILES[tile]
        pts = _new_tile_path(kind, entry, exit_,
                             lambda radius, t: self._arc_samples(radius * apothem * abs(t)))
        self._tile_polylines[tile] = (entry, apothem * pts)
        return self._tile_polylines[tile]

    def _draw_hex_road(self, draw, r, c, tile, sx, sy):
        """racingtilehex's road drawing, plus half of the edge road from each
        corner port, so the two cells a corner joins draw the edge between
        them."""
        if not _TILE_PORTS[tile]:
            return
        cx, cy = self._hex_center(r, c)
        _, local = self._tile_polyline(tile)
        pieces = [[((cx + x) * sx, (cy + y) * sy) for x, y in local]]
        h = 0.5 * self._hex_size()
        for p in _TILE_PORTS[tile]:
            if p >= 6:
                x0, y0 = self._port_point(r, c, p)
                a = _port_angle(p)
                pieces.append([(x0 * sx, y0 * sy), ((x0 + h * np.cos(a)) * sx, (y0 + h * np.sin(a)) * sy)])
        road_px = max(1, int(round(self._track_width * 0.5 * (sx + sy))))
        for poly in pieces:
            draw.line(poly, fill=(25, 25, 25), width=road_px + 4, joint="curve")
        for poly in pieces:
            draw.line(poly, fill=(210, 210, 210), width=road_px, joint="curve")
