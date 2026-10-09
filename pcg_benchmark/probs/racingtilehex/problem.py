"""Racetrack generation with Genetic-WFC on a hexagonal lattice.

Same pipeline as racingtile (see that module's header for the full
deviation list against Bailly and Levieux 2023): one boost zone per cell drives
a Simple Tiled WFC, and the placed modules are re-encoded into the chromosome
after evaluation.  Deviations 1-10 listed there apply here unchanged.

TWO FURTHER DEVIATIONS, specific to the hex lattice:

11. Six neighbours, not the paper's four (Sec. III-B); WFC does not need a
    square lattice (Sec. II-A).  This variant tests whether the lattice, not
    the algorithm, limits the shapes a constructive representation reaches.
12. Modules are enumerated, not rotated: a road joins any two of six faces,
    so all C(6,2) = 15 pairs are generated, each with variants from
    racingtile's shapes (_VARIANTS), 106 modules against the paper's 7 (more
    WFC time, their Table I); the plain pairs give 60 and 120 degree corners.
"""
from __future__ import annotations

import numpy as np
from pcg_benchmark.probs.racingtile.problem import (
    STRAIGHT, CORNER, SHARP, HAIRPIN, KINK, S_CHICANE, ESS, SWEEP, _GeneticWFCProblem,
    _exit_port, _largest_loop, _rectangle_loop, _tile_points)
from PIL import Image, ImageDraw


# ── Hex grid (pointy-top, odd-r offset storage) ───────────────────────────
# Stored as a (GRID_H, GRID_W) array like the square problem, so border,
# genome and render loop carry over; six faces make the turn menu {0, 60,
# 120} degrees.  Only 3 of 15 plain pairs are opposite faces, so 80% are
# corners (with the variants, 27 of 105 road tiles join opposite faces, 24 of
# them curved).  11x11 as racingtile, so both hand the search the same
# 121-gene genome and the lattice is the only difference.  The cost: rows sit
# 1.5 size apart against sqrt(3) size for columns, so the width binds and an
# untiled band stays along the bottom of the box (13x11 fills it to 2% but
# breaks the genome match).
GRID_H = 11
GRID_W = 11

GRASS = 0

# Six face directions, clockwise from the upper-right face of a pointy-top hex.
# NE, E, SE, SW, W, NW.  Opposite of direction d is (d + 3) % 6.
NE, E, SE, SW, W, NW = 0, 1, 2, 3, 4, 5
_DIRS = (NE, E, SE, SW, W, NW)
_OPPOSITE = {d: (d + 3) % 6 for d in _DIRS}

# odd-r offset neighbour deltas (dr, dc) depend on whether the row is even or
# odd, because odd rows are shifted half a hex to the right.  Index: [row & 1].
_DIR_DELTA = {
    NE: [(-1, 0), (-1, 1)],
    E:  [(0, 1),  (0, 1)],
    SE: [(1, 0),  (1, 1)],
    SW: [(1, -1), (1, 0)],
    W:  [(0, -1), (0, -1)],
    NW: [(-1, -1), (-1, 0)],
}

# Angle (radians) from a hex centre to each face midpoint, pointy-top layout.
# NE face sits at -60 deg (screen y grows downward), stepping +60 deg clockwise.
_FACE_ANGLE = {d: np.deg2rad(-60.0 + 60.0 * i) for i, d in enumerate(_DIRS)}


def _neighbor(r, c, d):
    dr, dc = _DIR_DELTA[d][r & 1]
    return r + dr, c + dc


# ── Tile vocabulary ───────────────────────────────────────────────────────
# Ids 1-15: one plain road tile per face pair; binary sockets as on squares
# (neighbours agree on the shared face).  Ids 16-105: variants per pair
# (_VARIANTS) with the same faces, differing only inside the cell.
# racingtile's three-arc CHICANE is not used on hexes (user decision).
_FACE_PAIRS = [
    frozenset({a, b})
    for i, a in enumerate(_DIRS)
    for b in _DIRS[i + 1:]
]
assert len(_FACE_PAIRS) == 15


def _separation(pair):
    """Faces between the two open faces the short way round: 3 is opposite
    (straight), 2 a 60 degree turn, 1 a 120 degree turn."""
    a, b = sorted(pair)
    return min(b - a, 6 - (b - a))


# Variants per face separation, (racingtile kind, mirror, value): SHARP's
# tangent length, KINK's strong radius, or the ESS / SWEEP swing; None for
# the defaults.  Each sits in one census class (_weights), measured alone
# between straights as (turn, mean radius):
#   opposite  S_CHICANE: 42 degrees at 34 m and 86 at 30 m (slow), twice
#             each.  ESS at 20 and 16 degrees: 37 degrees at 108 m and 30 at
#             134 m (fast).  SWEEP at 13: 24 degrees at 165 m (sweeper).
#   60        SHARP at 0.25, 0.40, 0.60: 60 degrees at 38 m (slow), 53 m and
#             73 m (medium).  The plain corner: 58 degrees at 119 m (fast).
#             KINK at the default 0.4: 78 degrees at 29-40 m (slow); at 0.7:
#             78 degrees at 52 m (medium).  0.8 leaves a negative straight.
#             HAIRPIN: 180 degrees at 30 m, with 60 degree swings either side.
#   120       the plain corner: 117 degrees at 44 m, on the slow/medium edge.
#             KINK: 140 degrees at 22-25 m (slow).  HAIRPIN: 180 degrees at
#             22 m, with 30 degree swings either side.
# No 120 degree tile is wider than the plain corner, the largest that fits.
# calibration/proto_hx.py.
_VARIANTS = {
    3: [(S_CHICANE, False, None), (S_CHICANE, True, None),
        (ESS, False, 20.0), (ESS, True, 20.0), (ESS, False, 16.0), (ESS, True, 16.0),
        (SWEEP, False, 13.0), (SWEEP, True, 13.0)],
    2: [(SHARP, False, 0.25), (SHARP, False, 0.40), (SHARP, False, 0.60),
        (KINK, False, None), (KINK, True, None), (KINK, False, 0.7), (KINK, True, 0.7),
        (HAIRPIN, False, None)],
    1: [(KINK, False, None), (KINK, True, None), (HAIRPIN, False, None)],
}

# WFC tile index -> frozenset of open faces, and -> (tile kind, mirror,
# value) with the kinds of racingtile.  0 = grass.
_OPEN_EDGES = {0: frozenset()}
_TILE_KIND = {0: None}
for _i, _pair in enumerate(_FACE_PAIRS, start=1):
    _OPEN_EDGES[_i] = _pair
    _TILE_KIND[_i] = (STRAIGHT if _separation(_pair) == 3 else CORNER, False, None)
for _pair in _FACE_PAIRS:
    for _kind in _VARIANTS[_separation(_pair)]:
        _TILE_KIND[len(_OPEN_EDGES)] = _kind
        _OPEN_EDGES[len(_OPEN_EDGES)] = _pair

# frozenset of two open faces -> plain WFC tile index (inverse map, for fallback).
_EDGES_TO_TILE = {pair: i for i, pair in enumerate(_FACE_PAIRS, start=1)}

_N_WFC_TILES    = len(_OPEN_EDGES)  # 106: 0 = grass, 1-15 plain pairs, 16-105 variants

# WFC weights of grass, the plain straight and every curve tile (see
# RacingTileHexProblem._weights).
_GRASS_WEIGHT = 4.0
_STRAIGHT_WEIGHT = 20.0
_CURVE_WEIGHT = 0.1

# Hairpin leg and kink strong radius per turn angle (apothems, 68.4 m), the
# two lengths a hex cell does not fit at the square values.  Swept in 0.05
# steps on the 16 m road (8 m half width), at the 64.2 m apothem of the
# 1091 m box:
#   hairpin leg: the longest leg that keeps the road edge at least 5.7 m inside
#     the cell, the square hairpin's clearance.  120 degrees: 0.95 (6.3 m;
#     1.00 leaves 3.5 m).  60 degrees: 0.40 (5.7 m; 0.45 leaves 2.5 m).  The
#     centre line scales with the apothem, the edge a half width off it, so
#     at the 68.4 m apothem and the 12 m road the clearances only grow.
#   kink: at the square's strong radius 0.4 the straight between the two
#     turns solves to -0.13 at 120 degrees, 0.35 leaves 0.003 and 0.30 leaves
#     0.14.  60 degrees keeps 0.4 (straight 0.26).  Both are in apothems, so
#     they hold at any cell size.
# Radii at the 68.4 m apothem: plain 39.5 m (120 degrees) and 118.5 m (60),
# hairpin 20.5 m, kink 20.5 m for 140 degrees and 27.4 m for 80, S chicane
# 27.4 m (_S_CHICANE_HEX); the census class of each tile is in _VARIANTS.
_HAIRPIN_LEG = {120.0: 0.95, 60.0: 0.40}
_KINK_STRONG = {120.0: 0.30, 60.0: 0.40}
# The hex S chicane: 45 degree swings (90 degree reversals) at 0.4 apothems,
# 27.4 m.  Target: the circuits' tight chicanes (opposite corner pairs under
# 50 m apart, both radii under 45 m; 12 pairs), median 92 degrees at 30 m,
# tightest 19 m (calibration/chicane_census.py).  racingtile's S (70 degree
# swings at 17.1 m) is tighter than every one.  Why not 30 m: it needs 2.13
# apothems along the cell, which has 2.
_S_CHICANE_HEX = (45.0, 0.4)


class RacingTileHexProblem(_GeneticWFCProblem):
    """Racetrack generation on a hexagonal WFC grid.

    Identical in spirit to racingtile-v0 (Genetic-WFC over a fixed lattice,
    the pipeline in racingtile's _GeneticWFCProblem), but the lattice is
    hexagonal: each cell has six faces, a road tile may connect ANY face to
    ANY other face, and corners are 60 or 120 degrees instead of a fixed 90.
    This removes the boxy right-angle look of the square tiles while keeping
    the same constructive drivability guarantee (each cell holds its own
    disjoint road piece, so tracks never self-overlap).
    """

    _SEED_TILE = _EDGES_TO_TILE[frozenset({E, W})]   # a straight

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._hex_geom = None

        # No quality parameter is overridden.  40 random genomes (seed 7): no
        # crossing, no fold (no tile radius under 20 m), closest separated
        # approach 32.4 m.  The check costs 10.3 ms per track.

    # ── Hex pixel geometry ────────────────────────────────────────────
    # A pointy-top hex of "size" (centre to vertex), placed so the INTERIOR
    # cells (the outer ring is grass) fill the build box: their centres span
    # sqrt(3) size (GRID_W - 2.5) across and 1.5 size (GRID_H - 3) down.

    def _hex_geometry(self):
        """(size, origin_x, origin_y), computed once and cached.

        Sized so the ROAD fits the build box, not just the cell centres.  A
        cell's road runs out to its face midpoints and its tile shapes bulge
        toward its vertices, so the reach of an interior cell is bounded by
        the hexagon itself: one apothem (size * sqrt(3) / 2) sideways and one
        size vertically, added at both ends of the span of centres.

        Why not bound only the centres: the road then runs one apothem past
        the box on each side.  Measured over 20 random genomes (seed 21), a
        centre-bounded grid (76.5 m apothem) puts the road at x = 40..1460 m
        against the 100..1400 m box; this one puts it at 115..1385 m.  The
        build box is what every representation shares, so a representation
        that overflows it is not compared on equal terms.  The cost: the
        smaller cell pulls the plain 120 degree corner under the 44 m
        slow/medium class edge (see the radii above _HAIRPIN_LEG)."""
        if self._hex_geom is None:
            x0, y0, x1, y1 = self._build_box()
            bw, bh = x1 - x0, y1 - y0
            # Interior centres span columns 1..GRID_W-2, with odd rows shifted
            # half a hex right, and rows 1..GRID_H-2.
            centres_w = np.sqrt(3.0) * (GRID_W - 2.5)
            centres_h = 1.5 * (GRID_H - 3)
            reach_w = centres_w + np.sqrt(3.0)   # + one apothem at each end
            reach_h = centres_h + 2.0            # + one size at each end
            # The tighter of the two axes sets the size, so the grid fits the
            # box rather than overflowing it.
            size = float(min(bw / reach_w, bh / reach_h))
            # Centre the reach in the box, then place the origin so the first
            # interior centre sits one apothem (or one size) inside that reach.
            ox = (x0 + 0.5 * (bw - size * reach_w) + size * np.sqrt(3.0) / 2.0
                  - 1.5 * np.sqrt(3.0) * size)
            oy = y0 + 0.5 * (bh - size * reach_h) + size - 2.25 * size
            self._hex_geom = (size, float(ox), float(oy))
        return self._hex_geom

    def _hex_size(self):
        return self._hex_geometry()[0]

    def _hex_center(self, r, c):
        size, ox, oy = self._hex_geometry()
        x = ox + size * np.sqrt(3.0) * (c + 0.5 * (r & 1) + 0.5)
        y = oy + size * 1.5 * (r + 0.5)
        return x, y

    def _face_midpoint(self, r, c, d):
        """Pixel midpoint of face d of cell (r, c): the point on the hex edge
        where the road crosses into the neighbour."""
        cx, cy = self._hex_center(r, c)
        size = self._hex_size()
        # Distance from centre to an edge midpoint (apothem) = size * sqrt(3)/2.
        apothem = size * np.sqrt(3.0) / 2.0
        a = _FACE_ANGLE[d]
        return cx + apothem * np.cos(a), cy + apothem * np.sin(a)

    # ── WFC compatibility (binary sockets over six directions) ────────
    _WFC_WEIGHTS = None  # populated once (grass weighted heavily)
    _WFC_COMPAT  = None

    # A legal request wins outright, as in racingtile (at a finite 1000,
    # re-decoding a WFC-written genome changed 21 of 121 genes on 1 of 20).

    @classmethod
    def _weights(cls):
        if cls._WFC_WEIGHTS is None:
            # One weight for all curve tiles, as in racingtile.  The vocabulary
            # against the corner census (curve tiles | circuits):
            #
            #   angle       slow          medium        fast          sweeper
            #   20-52       0    | 0.01   0    | 0.04   0.12 | 0.12   0.06 | 0.07
            #   52-75       0.06 | 0.06   0.12 | 0.08   0.06 | 0.05   0    | 0.01
            #   75-105      0.18 | 0.12   0.12 | 0.07   0    | 0.06   0    | 0
            #   105-127     0    | 0.04   0.06 | 0.03   0    | 0.02   0    | 0
            #   127-150     0.12 | 0.05   0    | 0.03   0    | 0.01   0    | 0
            #   150 and up  0.12 | 0.05   0    | 0.05   0    | 0.03   0    | 0
            #
            # The reached cells hold 0.69 of the corners; over them L1 0.26
            # (the plain pairs with S, SHARP, HAIRPIN and kinks: 0.32, reaching
            # 0.42).  calibration/vocab_share.py.  Weights swept as in
            # racingtile; no setting passes a genome of 120 through gates,
            # rules and typicality (tile_weights.py, 12 m road too), so that
            # count cannot choose; hex tracks stop at typicality.
            #
            #   grass straight curve   class L1 (s21, s22, s23)   all four   turns
            #     4     10     0.05    0.43                       1          33
            #     4     10     0.1     0.42                       2          38
            #     4     10     0.15    0.43                       0          45
            #     4     10     0.3     0.46                       0          48
            #     4     20     0.1     0.44  0.46  0.52           3  2  0    36 30 34   (in use)
            #     4     40     0.1     0.44                       2          36
            #     4     80     0.1     0.46  0.49  0.52           3  1  1    36 35 32
            #     8     20     0.1     0.43                       1          33
            #     8     40     0.1     0.44                       2          30
            #
            # Rule: the most tracks with all four, then the nearer mix.  No
            # setting reaches the circuits' 15 turns: the 120 degree pairs carry
            # three 140-180 degree tiles for one plain corner, and variants keep
            # straights beside their arcs.  Variants at weight 0 give L1 0.40,
            # 24 turns, 9 with all four (seed 21); variants requested only,
            # L1 0.49, 21 turns, 16 / 11 / 19.  mix_sweep.py, tile_weights.py.
            w = np.zeros(_N_WFC_TILES)
            for i in range(1, _N_WFC_TILES):
                w[i] = _STRAIGHT_WEIGHT if _TILE_KIND[i][0] == STRAIGHT else _CURVE_WEIGHT
            # Grass outweighs each road tile so WFC draws a loop through the
            # grid rather than filling it.
            w[GRASS] = _GRASS_WEIGHT
            cls._WFC_WEIGHTS = w
        return cls._WFC_WEIGHTS

    @classmethod
    def _wfc_compat(cls):
        if cls._WFC_COMPAT is not None:
            return cls._WFC_COMPAT
        compat = {}
        for d in _DIRS:
            opp = _OPPOSITE[d]
            compat[d] = []
            for i in range(_N_WFC_TILES):
                a_open = d in _OPEN_EDGES[i]
                allowed = set()
                for j in range(_N_WFC_TILES):
                    b_open = opp in _OPEN_EDGES[j]
                    if b_open == a_open:
                        allowed.add(j)
                compat[d].append(frozenset(allowed))
        cls._WFC_COMPAT = compat
        return compat

    def _wfc_propagate(self, wave, stack, compat):
        """AC-3 constraint propagation over six directions. False on contradiction."""
        while stack:
            r, c = stack.pop()
            cur = wave[r][c]
            for d in _DIRS:
                nr, nc = _neighbor(r, c, d)
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

    # ── Neighbour graph / loop machinery ──────────────────────────────

    def _build_neighbor_graph(self, tiles):
        """Map each road cell (r, c) to its mutually-connected neighbours.

        Two tiles are neighbours only when A has an open face toward B and B has
        an open face back toward A."""
        neighbors: dict = {}
        for r in range(GRID_H):
            for c in range(GRID_W):
                edges = _OPEN_EDGES[int(tiles[r, c])]
                if not edges:
                    continue
                nbrs = []
                for d in edges:
                    nr, nc = _neighbor(r, c, d)
                    if 0 <= nr < GRID_H and 0 <= nc < GRID_W:
                        if _OPPOSITE[d] in _OPEN_EDGES[int(tiles[nr, nc])]:
                            nbrs.append((nr, nc))
                neighbors[(r, c)] = nbrs
        return neighbors

    def _edge_midpoints(self, cells):
        """One waypoint per cell: the shared-face midpoint with the next cell."""
        n = len(cells)
        points = []
        for i, (r, c) in enumerate(cells):
            nr, nc = cells[(i + 1) % n]
            d = self._direction_toward(r, c, nr, nc)
            if d is None:
                cx, cy = self._hex_center(r, c)
                points.append((cx, cy))
            else:
                points.append(self._face_midpoint(r, c, d))
        return points

    def _keep_largest_component(self, tiles):
        """Keep only road tiles forming the largest single closed loop
        (racingtile's _largest_loop)."""
        largest = _largest_loop(self._build_neighbor_graph(tiles))
        if largest is None:
            return tiles
        tiles = tiles.copy()
        for r in range(GRID_H):
            for c in range(GRID_W):
                if tiles[r, c] != GRASS and (r, c) not in largest:
                    tiles[r, c] = GRASS
        return tiles

    # ── Hooks of _GeneticWFCProblem ───────────────────────────────────

    def _n_tiles(self):
        return len(self._weights())   # the vocabulary, here or in a subclass

    def _tile_weights(self):
        return self._weights()

    def _ids_to_tiles(self, ids):
        return ids

    def _tile_ids(self, tiles):
        return tiles

    def _fallback_tiles(self, rng):
        return self._ring_fallback(rng)

    def _content_track_points(self, content, tiles):
        return self._grid_to_track_points(tiles)

    def _tile_background(self, img_w, img_h, content, tiles):
        return self._make_tile_bg(img_w, img_h, tiles)

    def _ring_fallback(self, rng):
        """Deterministic closed hex ring as a last resort.

        Walk a rectangular block of cells clockwise and set each cell's tile to
        the face-pair joining its previous and next neighbour.  Because face
        directions are parity-dependent on a hex grid, the pair is computed from
        the actual step directions, so the ring is always drivable."""
        loop = _rectangle_loop(rng)

        tiles = np.zeros((GRID_H, GRID_W), dtype=int)
        n = len(loop)
        for i, (r, c) in enumerate(loop):
            pr, pc = loop[(i - 1) % n]
            nr, nc = loop[(i + 1) % n]
            from_dir = self._direction_toward(r, c, pr, pc)
            to_dir   = self._direction_toward(r, c, nr, nc)
            if from_dir is None or to_dir is None or from_dir == to_dir:
                continue
            tiles[r, c] = _EDGES_TO_TILE.get(frozenset({from_dir, to_dir}), GRASS)
        # Any cell whose neighbours were not axial-adjacent stays grass; drop
        # the whole ring to a smaller valid loop by keeping the largest one.
        return self._keep_largest_component(tiles)

    def _direction_toward(self, r, c, target_r, target_c):
        """Return the hex face direction stepping from (r,c) to the target."""
        for d in _DIRS:
            nr, nc = _neighbor(r, c, d)
            if nr == target_r and nc == target_c:
                return d
        return None

    # ── Decoding: tile grid -> track waypoints ────────────────────────

    def _tile_polyline(self, tile):
        """Road centre line of one tile, relative to its cell centre in metres.

        Returns (entry face, points) from the midpoint of the entry face (the
        lower-numbered open face) to the midpoint of the other.  Every road
        crosses a face perpendicular to it, along the radial from the hex
        centre, so consecutive tiles join without a kink.  The canonical path
        from racingtile's _tile_points is in apothem units; the linear map that
        sends its entry and exit face normals to this pair's face normals places
        it in the map's y-down frame.  Both pairs of normals meet at the same
        angle, so the map is a rotation or a reflection."""
        tile = int(tile)
        if tile in self._tile_polylines:
            return self._tile_polylines[tile]
        apothem = self._hex_size() * np.sqrt(3.0) / 2.0
        entry, other = sorted(_OPEN_EDGES[tile])
        kind, mirror, value = _TILE_KIND[tile]
        sep = _separation(_OPEN_EDGES[tile])
        turn = 180.0 - 60.0 * sep
        pts = _tile_points(kind, turn,
                           lambda radius, t: self._arc_samples(radius * apothem * abs(t)),
                           mirror=mirror, hairpin_leg=_HAIRPIN_LEG.get(turn, 0.0),
                           kink_strong=value if kind == KINK and value else _KINK_STRONG.get(turn, 0.4),
                           sharp_radius=value if kind == SHARP else 0.6,
                           ess=value if kind in (ESS, SWEEP) else None,
                           s_chicane=_S_CHICANE_HEX)
        n_in = np.array([np.cos(_FACE_ANGLE[entry]), np.sin(_FACE_ANGLE[entry])])
        n_out = np.array([np.cos(_FACE_ANGLE[other]), np.sin(_FACE_ANGLE[other])])
        if sep == 3:
            # Canonical straight axis (1, 0) points at the exit face.
            m = np.array([[n_out[0], -n_out[1]], [n_out[1], n_out[0]]])
        else:
            m = np.column_stack([n_in, n_out]) @ np.linalg.inv(
                np.column_stack([[0.0, 1.0], _exit_port(turn)]))
        self._tile_polylines[tile] = (entry, apothem * (pts @ m.T))
        return self._tile_polylines[tile]

    def _loop_to_track_points(self, loop, tiles):
        """Convert an ordered cell loop to waypoints following the road geometry.

        Each tile contributes its centre line from _tile_polyline, reversed
        when the loop enters through the other open face, without its first
        point (the previous tile already ended there)."""
        n = len(loop)
        points = []
        for i, (r, c) in enumerate(loop):
            pr, pc = loop[(i - 1) % n]
            entry, local = self._tile_polyline(tiles[r, c])
            if self._direction_toward(r, c, pr, pc) != entry:
                local = local[::-1]
            cx, cy = self._hex_center(r, c)
            points.extend((cx + x, cy + y) for x, y in local[1:])
        return points

    def _grid_to_track_points(self, tiles):
        loop = self._extract_loop(tiles)
        if loop is None:
            return None
        return self._loop_to_track_points(loop, tiles)

    # ── Hex rendering ─────────────────────────────────────────────────

    def _make_tile_bg(self, img_w, img_h, tiles):
        bg   = Image.new("RGB", (img_w, img_h), (34, 139, 34))
        draw = ImageDraw.Draw(bg)
        sx = img_w / float(self._width)
        sy = img_h / float(self._height)
        # Outline every hex cell, then draw the road pieces on top.
        for r in range(GRID_H):
            for c in range(GRID_W):
                self._draw_hex_outline(draw, r, c, sx, sy)
        for r in range(GRID_H):
            for c in range(GRID_W):
                self._draw_hex_road(draw, r, c, int(tiles[r, c]), sx, sy)
        return bg

    def _hex_corners(self, r, c):
        """The six vertices of a pointy-top hex (pixel space)."""
        cx, cy = self._hex_center(r, c)
        size = self._hex_size()
        pts = []
        for i in range(6):
            a = np.deg2rad(-90.0 + 60.0 * i)  # pointy-top: first vertex up
            pts.append((cx + size * np.cos(a), cy + size * np.sin(a)))
        return pts

    def _draw_hex_outline(self, draw, r, c, sx, sy):
        pts = [(x * sx, y * sy) for x, y in self._hex_corners(r, c)]
        draw.polygon(pts, outline=(20, 100, 20))

    def _draw_hex_road(self, draw, r, c, tile, sx, sy):
        """Draw one tile's road at the track width along its centre line, so
        the picture is the road the car drives: a dark line two pixels wider
        per side under the grey road."""
        if not _OPEN_EDGES[tile]:
            return
        cx, cy = self._hex_center(r, c)
        _, local = self._tile_polyline(tile)
        poly = [((cx + x) * sx, (cy + y) * sy) for x, y in local]
        road_px = max(1, int(round(self._track_width * 0.5 * (sx + sy))))
        draw.line(poly, fill=(25, 25, 25), width=road_px + 4, joint="curve")
        draw.line(poly, fill=(210, 210, 210), width=road_px, joint="curve")
