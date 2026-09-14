"""Racetrack generation with Genetic-WFC on a hexagonal lattice.

Same pipeline as racingtile (see that module's header for the full
deviation list against Bailly and Levieux 2023): one boost zone per cell drives
a Simple Tiled WFC, and the placed modules are re-encoded into the chromosome
after evaluation.  Deviations 1-10 listed there apply here unchanged.

TWO FURTHER DEVIATIONS, specific to the hex lattice:

11. Six neighbours instead of four.  The paper takes "into account four
    neighbors: left, right, top and bottom" with four 90-degree rotations
    (Sec. III-B).  WFC itself does not require a square lattice, and the paper
    notes the neighbour count may vary (Sec. II-A, citing [12]); this variant
    exists to test whether the lattice, rather than the algorithm, is what
    limits the shapes a constructive representation can reach.

12. The module set is enumerated rather than rotated.  On a square grid a
    module plus four rotations covers the vocabulary; on a hex grid a road
    piece joins any two of six faces, so all C(6,2) = 15 pairs are generated
    directly and rotation is implicit in which pair a module names.  This
    yields 16 modules against the paper's 7, which costs WFC time (their
    Table I) but is what gives 60 and 120 degree corners instead of only 90.
"""
from __future__ import annotations

import numpy as np
from pcg_benchmark.probs.racing.problem import RacingProblem
from pcg_benchmark.spaces import ArraySpace, IntegerSpace, DictionarySpace
from PIL import Image, ImageDraw


# ── Hex grid (pointy-top, odd-r offset storage) ───────────────────────────
# Stored as a (GRID_H, GRID_W) array exactly like the square tile problem, so
# border forcing, genome shape and the render loop carry over unchanged.  Only
# the neighbour graph differs: six faces instead of four, so the turn menu
# becomes {0, 60, 120} degrees with no 90-degree corners at all.
# 6-way connectivity is inherently corner-dense: a road tile joins any two of
# six faces, and only 3 of the 15 pairs are opposite faces, so 80% of the road
# vocabulary is a corner.
#
# 11x11 matches racingtile's grid exactly, so both representations hand the
# search the same 121-gene genome and the lattice is the only difference
# between them.  That is what makes the pair a controlled experiment rather
# than two separate representations that also happen to differ in search-space
# size.
#
# The cost, stated plainly: pointy-top hexes pack tighter vertically than
# horizontally (rows sit 1.5 * size apart against sqrt(3) * size for columns),
# so a square hex grid cannot fill a square map.  With equal rows and columns
# the WIDTH binds, the hexes grow to fill it, and the grid leaves an untiled
# band along the bottom of the build box.  A 13x11 grid fills the box in both
# axes to within 2% but costs the genome-length match; this is the other side
# of that trade.
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


# ── Tile vocabulary: every unordered pair of faces is one road tile ───────
# C(6, 2) = 15 face-pairs + grass = 16 tiles.  A tile's "open edges" are simply
# its two connected faces (or the empty set for grass).  Binary sockets: a face
# is open or closed, and two neighbouring cells are compatible when the shared
# face agrees (both open or both closed) — identical rule to the square tiles,
# just over six directions.
_FACE_PAIRS = [
    frozenset({a, b})
    for i, a in enumerate(_DIRS)
    for b in _DIRS[i + 1:]
]
assert len(_FACE_PAIRS) == 15

# WFC tile index -> frozenset of open faces.  0 = grass.
_OPEN_EDGES = {0: frozenset()}
for _i, _pair in enumerate(_FACE_PAIRS, start=1):
    _OPEN_EDGES[_i] = _pair

# frozenset of two open faces -> WFC tile index (inverse map, for fallback).
_EDGES_TO_TILE = {pair: i for i, pair in enumerate(_FACE_PAIRS, start=1)}

_N_WFC_TILES    = len(_OPEN_EDGES)  # 16: 0 = grass, 1-15 = road pairs


class _HexGridSpace(DictionarySpace):
    """Genome = one boost zone per cell (Genetic-WFC, Bailly and Levieux 2023).

    Following Sec. III-E of the paper, each gene holds the tile whose selection
    probability is boosted when WFC collapses that cell, so the genome steers
    generation without ever overriding it: a tile that constraint propagation
    has already eliminated has probability zero, and boosting zero leaves it
    zero.  Generation therefore cannot produce an adjacency violation, and no
    request ever has to be detected and undone.

    Every cell carries a boost ("one boost zone per grid cell"): gene 0 boosts
    grass, values 1-15 boost that face-pair road tile at cell (r,c).  Each gene
    maps to one cell of the layout, so crossover and mutation via contentSwap
    make small, local changes to the track.

    Re-encoding (Sec. III-E(b)) is implemented in _reencode, called from
    info(): after evaluation the genome is rewritten to the raw WFC output, so
    crossover recombines layouts that were actually built.  Decoding is a pure
    function of the genome, since every individual gets its own full WFC pass
    from the same seed, so the same genome yields the same track in any
    instance and in any order."""

    def __init__(self, problem_ref):
        super().__init__({
            "tile_prefs": ArraySpace(
                (GRID_H * GRID_W,),
                # IntegerSpace max is exclusive (isSampled uses value < max, and
                # sample() draws integers(min, max)), so 0.._N_WFC_TILES-1 are
                # the valid tile indices: 0 = grass, 1..15 = the face-pair roads.
                IntegerSpace(0, _N_WFC_TILES),
            ),
        })
        self._prob = problem_ref

    def seed(self, seed):
        """Seed the genome draw, not just the nested spaces.

        GenericSpace.seed only reaches the nested ArraySpace, and sample()
        below never consults it: the genome comes from init_content, which
        without an explicit generator builds a fresh unseeded one on every
        call.  So seeding this space used to have no effect at all and two
        identical runs drew different populations.
        """
        super().seed(seed)
        self._random = np.random.default_rng(seed)

    def sample(self):
        return self._prob.init_content(self._random)


class RacingTileHexProblem(RacingProblem):
    """Racetrack generation on a hexagonal WFC grid.

    Identical in spirit to racingtile-v0 (Genetic-WFC over a fixed lattice),
    but the lattice is hexagonal: each cell has six faces, a road tile may
    connect ANY face to ANY other face, and corners are 60 or 120 degrees
    instead of a fixed 90.  This removes the boxy right-angle look of the
    square tiles while keeping the same constructive drivability guarantee
    (each cell holds its own disjoint road piece, so tracks never self-overlap).
    """

    # Decoded points already trace a valid loop in order; the base class's
    # 2-opt untangle must not reorder them (it would break the loop).
    _untangle_control_points = False

    def __init__(self, **kwargs):
        kwargs.setdefault('num_points', 15)
        # max_steps is NOT overridden: the shared 7000-step budget applies.
        # Now that every representation generates a lap of the same length, a
        # per-representation cap would score the budget rather than the track.
        super().__init__(**kwargs)
        self._hex_geom = None
        self._content_space = _HexGridSpace(self)
        # genome bytes -> (tiles, wave_array), where wave_array is a
        # (GRID_H, GRID_W) int array of WFC tile indices.  _decode_cache_keys
        # holds the same genomes in insertion order, giving the eviction policy
        # something to pop from.
        self._decode_cache: dict = {}
        self._decode_cache_keys: list = []
        self._DECODE_CACHE_MAX = int(kwargs.get("decode_cache_max", 512))
        # genome bytes -> raw WFC output before loop pruning, used by _reencode.
        self._raw_wfc: dict = {}

        # Grid construction guarantees the track never overlaps itself, so the
        # area-overlap check is disabled.  This is the ONLY quality parameter
        # hex overrides; the length band is the shared one, since lap length is
        # tuned in the generator (see _weights) rather than by moving the
        # target.
        self._QUALITY_PARAMS = {
            **self._QUALITY_PARAMS,
            "geom_area_check": False,
        }

    # ── Hex pixel geometry ────────────────────────────────────────────
    # A pointy-top hex of "size" (centre-to-vertex), sized and positioned so
    # the INTERIOR cells fill the shared build box.  Interior is what matters:
    # the outer ring is forced to grass, so rows 1..GRID_H-2 and columns
    # 1..GRID_W-2 are the only cells a road can occupy.  Their centres span
    # sqrt(3) * size * (GRID_W - 2.5) across and 1.5 * size * (GRID_H - 3) down.

    def _hex_geometry(self):
        """(size, origin_x, origin_y), computed once and cached."""
        if self._hex_geom is None:
            x0, y0, x1, y1 = self._build_box()
            bw, bh = x1 - x0, y1 - y0
            span_w = np.sqrt(3.0) * (GRID_W - 2.5)
            span_h = 1.5 * (GRID_H - 3)
            # The tighter of the two axes sets the size, so the grid fits the
            # box rather than overflowing it.
            size = float(min(bw / span_w, bh / span_h))
            # Centre the interior reach in the box.
            ox = x0 + 0.5 * (bw - size * span_w) - 1.5 * np.sqrt(3.0) * size
            oy = y0 + 0.5 * (bh - size * span_h) - 2.25 * size
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

    # Boost-zone multiplier (Bailly and Levieux 2023, Sec. III-E use "a fixed
    # and very high boosting factor").  1000 against a grass weight of 4 makes
    # a requested tile win essentially whenever it is still legal, so the
    # genome is expressive, while a request that propagation already ruled out
    # stays impossible rather than becoming a contradiction.
    _BOOST_FACTOR = 1000.0

    @classmethod
    def _weights(cls):
        if cls._WFC_WEIGHTS is None:
            w = np.ones(_N_WFC_TILES, dtype=float)
            # Grass outweighs each road tile so WFC draws a loop through the
            # grid rather than filling it.  The ratio also sets lap length: 4.0
            # gives 4177 m against the 4650 m median of the 25 real circuits,
            # where 6.0 gives 3903 m.  Hex is less sensitive to this knob than
            # racingtile because 12 of its 15 road tiles are corners, so loops
            # turn back on themselves before they grow long.
            w[GRASS] = 4.0
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

    @staticmethod
    def _connected_components(neighbors):
        visited, components = set(), []
        for seed in neighbors:
            if seed in visited:
                continue
            comp, stack = [], [seed]
            while stack:
                cell = stack.pop()
                if cell in visited:
                    continue
                visited.add(cell)
                comp.append(cell)
                for nb in neighbors.get(cell, []):
                    if nb not in visited:
                        stack.append(nb)
            components.append(comp)
        return components

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
        """Keep only road tiles forming the largest single closed loop."""
        neighbors = self._build_neighbor_graph(tiles)
        components = self._connected_components(neighbors)

        loops = []
        for comp in components:
            is_loop = all(len(neighbors.get(cell, [])) == 2 for cell in comp)
            if is_loop:
                loops.append(comp)

        if not loops:
            return tiles

        largest = set(max(loops, key=len))
        tiles = tiles.copy()
        for r in range(GRID_H):
            for c in range(GRID_W):
                if tiles[r, c] != GRASS and (r, c) not in largest:
                    tiles[r, c] = GRASS
        return tiles

    # ── WFC runner ────────────────────────────────────────────────────

    def _run_wfc(self, wave, rng, compat, boosts=None):
        """Observe (min-entropy) / collapse (weighted) / propagate (AC-3).

        `boosts` is the genome's boost zones as a (GRID_H, GRID_W) int array,
        one per cell (Bailly and Levieux 2023, Sec. III-E): entry 0 means no
        boost, otherwise the tile index whose selection probability is
        multiplied by _BOOST_FACTOR when this cell is collapsed.  The boost
        only reweights choices that are still legal -- a tile already
        eliminated by propagation has probability zero, and scaling zero leaves
        it zero, so the genome can never force a constraint violation.

        Returns a (GRID_H, GRID_W) tile-index array or None on contradiction."""
        weights = self._weights()
        while True:
            min_e, candidates = float('inf'), []
            for r in range(GRID_H):
                for c in range(GRID_W):
                    n = len(wave[r][c])
                    if n > 1:
                        if n < min_e:
                            min_e, candidates = n, [(r, c)]
                        elif n == min_e:
                            candidates.append((r, c))
            if not candidates:
                break
            r, c = candidates[int(rng.integers(len(candidates)))]
            possible = list(wave[r][c])
            w = weights[possible].astype(float)
            if boosts is not None:
                # Scale up the requested tile's weight.  Every cell carries a
                # boost (gene 0 requests grass), so there is no "no request"
                # case.  If propagation has already ruled that tile out it is
                # absent from `possible`, so the request simply has no effect:
                # nothing to detect, nothing to roll back.
                w[np.asarray(possible) == boosts[r, c]] *= self._BOOST_FACTOR
            w = w / w.sum()
            chosen = possible[int(rng.choice(len(possible), p=w))]
            wave[r][c] = {chosen}
            if not self._wfc_propagate(wave, [(r, c)], compat):
                return None
        tiles = np.zeros((GRID_H, GRID_W), dtype=int)
        for r in range(GRID_H):
            for c in range(GRID_W):
                tiles[r, c] = next(iter(wave[r][c])) if wave[r][c] else GRASS
        return tiles

    @staticmethod
    def _copy_wave(wave):
        return [[set(cell) for cell in row] for row in wave]

    def _decode_cache_store(self, prefs_arr, tiles, wave, raw=None):
        """Insert a decoded genome into the bounded cache.

        Pure memoization: with a fixed seed and no repair path, decoding is a
        function of the genome alone, so a hit is indistinguishable from a
        recompute.

        `raw` is the WFC output BEFORE loop pruning, kept here rather than in a
        separate dict so it is evicted together with its entry and can never go
        missing while the decode is still cached (_reencode needs both)."""
        key = prefs_arr.tobytes()
        if key not in self._decode_cache:
            self._decode_cache_keys.append(prefs_arr.copy())
            while len(self._decode_cache_keys) > self._DECODE_CACHE_MAX:
                oldest = self._decode_cache_keys.pop(0)
                self._decode_cache.pop(oldest.tobytes(), None)
                self._raw_wfc.pop(oldest.tobytes(), None)
        self._decode_cache[key] = (tiles, wave)
        self._raw_wfc[key] = np.asarray(tiles if raw is None else raw).copy()

    def _decode_genome(self, tile_prefs):
        """Decode a boost-zone genome to a (GRID_H, GRID_W) tile-index array.

        This is Alg. 1 line 18 of Bailly and Levieux (2023), `l <- generate(c)`:
        one full WFC pass per individual, with the genome supplying the boost
        zones.  tile_prefs is a flat int array of length GRID_H*GRID_W holding
        the module ID to boost in each cell; value 0 boosts grass, 1-15 boost
        that face-pair road tile.  _run_wfc multiplies the requested module's
        selection probability at each collapse, so the genome biases generation
        but can never force a placement propagation has already ruled out.

        Deterministic: the same genome always decodes to the same track,
        because the first attempt always uses the same seed (the paper's "we
        use the same random generator seed every time we generate a level")."""
        tile_prefs = np.asarray(tile_prefs, dtype=int)
        if tile_prefs.size != GRID_H * GRID_W:
            raise ValueError(
                "tile_prefs must have %d entries for this %dx%d grid, got %d"
                % (GRID_H * GRID_W, GRID_H, GRID_W, tile_prefs.size))
        key = tile_prefs.tobytes()
        if key in self._decode_cache:
            return self._decode_cache[key][0]

        compat   = self._wfc_compat()
        n_tiles  = _N_WFC_TILES
        prefs_2d = tile_prefs.reshape(GRID_H, GRID_W)

        # ── Full WFC from scratch ──────────────────────────────────────────
        # The border is a hard constraint (level structure, not a genome
        # request), so it is forced here.  The genome itself never touches the
        # wave — it is passed to _run_wfc as boost zones, which only reweight
        # choices that are already legal.
        base_wave = [[set(range(n_tiles)) for _ in range(GRID_W)] for _ in range(GRID_H)]
        border = []
        for r in range(GRID_H):
            for c in range(GRID_W):
                if r == 0 or r == GRID_H - 1 or c == 0 or c == GRID_W - 1:
                    base_wave[r][c] = {GRASS}
                    border.append((r, c))
        self._wfc_propagate(base_wave, border, compat)

        # Seed a road if the genome asks for grass everywhere, else WFC has
        # nothing to build a loop from.  Use a straight-through tile at centre.
        if not np.any(prefs_2d[1:GRID_H - 1, 1:GRID_W - 1]):
            sr, sc = GRID_H // 2, GRID_W // 2
            straight = _EDGES_TO_TILE[frozenset({E, W})]
            if straight in base_wave[sr][sc]:
                base_wave[sr][sc] = {straight}
                self._wfc_propagate(base_wave, [(sr, sc)], compat)

        # Fixed seed sequence, NOT one derived from the genome.  A fixed stream
        # gives parent and child the same dice, leaving the genome as the only
        # difference between them, which is what makes the layout heritable; a
        # genome-derived seed would hand every mutant an unrelated random
        # stream and let a one-gene change redraw the whole track.  Still
        # deterministic: the same genome decodes to the same track.
        for attempt in range(100):
            rng  = np.random.default_rng(attempt * 1_000_003 + 7)
            wave = self._copy_wave(base_wave)
            result = self._run_wfc(wave, rng, compat, boosts=prefs_2d)
            if result is None:
                continue
            tiles = self._keep_largest_component(result)
            if self._extract_loop(tiles) is not None:
                # Store the RAW WFC output alongside: re-encoding must record
                # what WFC placed, not what survived pruning (see _reencode).
                self._decode_cache_store(tile_prefs, tiles, tiles.copy(), raw=result)
                return tiles

        # Total failure — deterministic hexagonal ring fallback.
        rng   = np.random.default_rng(int(np.sum(tile_prefs)) % (2**31))
        tiles = self._ring_fallback(rng)
        self._decode_cache_store(tile_prefs, tiles, tiles.copy())
        return tiles

    def init_content(self, rng=None):
        """Return a random dense boost-zone genome, one boost per cell.

        Every cell carries a boost, as in Bailly and Levieux 2023 Sec. III-E
        ("We use one boost zone per grid cell").  Gene 0 boosts grass and
        1.._N_WFC_TILES-1 boost that face-pair road tile.

        Density is what makes the genome expressive.  WFC collapses the
        most-constrained cells first, so a cell carrying no boost is usually
        decided by its neighbours long before its own gene would be consulted;
        at a sparse ~15% of cells only ~13% of requests reach the layout.
        Boosting every cell gives the genome a say wherever WFC looks, which is
        what makes offspring resemble their parents.

        Genes are drawn UNIFORMLY over the whole vocabulary, which is the
        paper's "the first population is initialized with random chromosomes"
        (Sec. III-E(c)).  Grass is one value of sixteen rather than a weighted
        majority for two reasons.  Biasing the draw toward grass starves the
        genome of road and roughly halves track length (1401 m mean at a 15%
        road rate against 2706 m uniform, on a 3273 m min_length).  It also
        keeps mutation effective, since contentSwap draws replacement genes
        from this function and a grass-heavy draw would make most mutations
        grass-onto-grass no-ops.  The preference for sparse loops lives in the
        grass-weighted WFC collapse (_weights) instead, which is where it
        belongs."""
        if rng is None:
            rng = np.random.default_rng()
        elif isinstance(rng, int):
            rng = np.random.default_rng(rng)
        n = GRID_H * GRID_W
        tile_prefs = rng.integers(0, _N_WFC_TILES, size=n).astype(int)
        return {"tile_prefs": tile_prefs}

    def _ring_fallback(self, rng):
        """Deterministic closed hex ring as a last resort.

        Walk a rectangular block of cells clockwise and set each cell's tile to
        the face-pair joining its previous and next neighbour.  Because face
        directions are parity-dependent on a hex grid, the pair is computed from
        the actual step directions, so the ring is always drivable."""
        min_dim = 3
        r0 = int(rng.integers(1, GRID_H - min_dim - 1))
        c0 = int(rng.integers(1, GRID_W - min_dim - 1))
        r1 = int(rng.integers(r0 + min_dim, min(r0 + min_dim + 5, GRID_H - 1) + 1))
        c1 = int(rng.integers(c0 + min_dim, min(c0 + min_dim + 5, GRID_W - 1) + 1))
        loop = []
        for c in range(c0, c1):
            loop.append((r0, c))
        for r in range(r0, r1):
            loop.append((r, c1))
        for c in range(c1, c0, -1):
            loop.append((r1, c))
        for r in range(r1, r0, -1):
            loop.append((r, c0))

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

    def _extract_loop(self, tiles):
        """Traverse mutually-connected road tiles to find a closed loop."""
        neighbors = self._build_neighbor_graph(tiles)
        cycle_cells = {cell for cell, nbrs in neighbors.items() if len(nbrs) == 2}
        if not cycle_cells:
            return None

        start = next(iter(cycle_cells))
        loop, prev, cur = [start], None, start
        while True:
            nxt = None
            for nb in neighbors.get(cur, []):
                if nb != prev and nb in cycle_cells:
                    nxt = nb
                    break
            if nxt is None or nxt == start:
                break
            loop.append(nxt)
            prev, cur = cur, nxt

        if len(loop) < 4 or start not in neighbors.get(loop[-1], []):
            return None
        return loop

    # How many samples each corner tile's arc contributes to the polyline.
    # Arc radius is apothem*tan(sep/2): 38 m for a 120 degree turn (adjacent
    # faces), 113 m for a 60 degree turn.  _make_curve then re-spaces the
    # polyline at 5.0 m, so the tight corner is a 6-sided approximation
    # (0.6 m of corner-cutting, against a 16 m track width).
    _ARC_SAMPLES = 6

    def _loop_to_track_points(self, loop):
        """Convert an ordered cell loop to waypoints following the road geometry.

        Each road tile connects two of its faces.  The road crosses every face
        PERPENDICULAR to that face (i.e. along the radial direction from the hex
        centre), so a straight tile (opposite faces) is a line through the
        centre, and a 60 or 120 degree tile is the unique circular arc tangent
        to both faces' perpendiculars.  That arc is sampled _ARC_SAMPLES times.

        Because every cell's road enters and leaves perpendicular to the shared
        face, the piece in the next cell continues along the exact same
        direction across that face: consecutive tiles link C1-smoothly with no
        kink, which is what makes the whole loop read as one flowing track."""
        n = len(loop)
        points = []
        for i, (r, c) in enumerate(loop):
            pr, pc = loop[(i - 1) % n]
            nr, nc = loop[(i + 1) % n]
            d_in  = self._direction_toward(r, c, pr, pc)
            d_out = self._direction_toward(r, c, nr, nc)
            if d_in is None or d_out is None:
                # Non-adjacent step (should not happen for a valid loop) — emit
                # the centre so the polyline stays continuous.
                points.append(self._hex_center(r, c))
                continue

            exit_mid = self._face_midpoint(r, c, d_out)

            # Straight tile: opposite faces -> single line through the centre,
            # emit exit only (the densifier fills the segment).
            if _OPPOSITE[d_in] == d_out:
                points.append(exit_mid)
                continue

            # Corner tile: the arc tangent to both face perpendiculars.
            arc = self._corner_arc(r, c, d_in, d_out)
            points.extend(arc)
        return points

    def _corner_arc(self, r, c, d_in, d_out):
        """Sample the circular arc that crosses faces d_in and d_out of cell
        (r, c) perpendicular to each face.

        For a regular hexagon the perpendicular to a face is the radial
        direction, so the road direction at a face midpoint is that face's
        radial.  The arc tangent to both radials has its centre at the
        intersection of the two FACE LINES (each tangent to the hexagon at its
        face midpoint); the two radii are equal by symmetry.  Entering and
        leaving along the face normals is exactly the "perpendicular exit"
        requirement, and it is what lets neighbouring cells join without a kink.

        Returns _ARC_SAMPLES points, ending at the d_out face midpoint."""
        cx, cy = self._hex_center(r, c)
        apothem = self._hex_size() * np.sqrt(3.0) / 2.0

        a_in  = float(_FACE_ANGLE[d_in])
        a_out = float(_FACE_ANGLE[d_out])
        m_in  = np.array([cx + apothem * np.cos(a_in),  cy + apothem * np.sin(a_in)])
        m_out = np.array([cx + apothem * np.cos(a_out), cy + apothem * np.sin(a_out)])

        # Each face line passes through the face midpoint with direction
        # perpendicular to that face's radial (i.e. tangent to the hexagon).
        t_in  = np.array([-np.sin(a_in),  np.cos(a_in)])
        t_out = np.array([-np.sin(a_out), np.cos(a_out)])

        # Solve m_in + s*t_in = m_out + u*t_out for the arc centre.
        A = np.array([[t_in[0], -t_out[0]], [t_in[1], -t_out[1]]])
        b = m_out - m_in
        det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        if abs(det) < 1e-9:
            # Parallel face lines (only the opposite-face straight, handled by
            # the caller) — emit a straight sample as a safe fallback.
            return [tuple(m_in + (m_out - m_in) * s / self._ARC_SAMPLES)
                    for s in range(1, self._ARC_SAMPLES + 1)]
        s = (b[0] * A[1, 1] - b[1] * A[0, 1]) / det
        centre = m_in + s * t_in

        radius = float(np.linalg.norm(centre - m_in))
        a0 = np.arctan2(m_in[1]  - centre[1], m_in[0]  - centre[0])
        a1 = np.arctan2(m_out[1] - centre[1], m_out[0] - centre[0])
        sweep = (a1 - a0 + np.pi) % (2.0 * np.pi) - np.pi  # shortest way round
        pts = []
        for k in range(1, self._ARC_SAMPLES + 1):
            a = a0 + sweep * k / self._ARC_SAMPLES
            pts.append((float(centre[0] + radius * np.cos(a)),
                        float(centre[1] + radius * np.sin(a))))
        return pts

    def _grid_to_track_points(self, tiles):
        loop = self._extract_loop(tiles)
        if loop is None:
            return None
        return self._loop_to_track_points(loop)

    def _best_effort_path(self, tiles):
        """For rendering: largest connected chain of road tiles, loop or not."""
        neighbors = self._build_neighbor_graph(tiles)
        components = self._connected_components(neighbors)
        best = max(components, key=len) if components else []
        if len(best) < 2:
            return None

        best_set = set(best)
        start = best[0]
        for cell in best:
            in_chain = [nb for nb in neighbors.get(cell, []) if nb in best_set]
            if len(in_chain) == 1:
                start = cell
                break

        path, prev, cur = [start], None, start
        while True:
            nxt = None
            for nb in neighbors.get(cur, []):
                if nb != prev and nb in best_set:
                    nxt = nb
                    break
            if nxt is None or nxt == path[0]:
                break
            path.append(nxt)
            prev, cur = cur, nxt

        if len(path) < 2:
            return None
        return self._edge_midpoints(path)

    # ── Content extraction bridge ─────────────────────────────────────

    def _genome_to_tiles(self, content):
        return self._decode_genome(np.asarray(content["tile_prefs"], dtype=int))

    def _extract_content(self, content):
        if isinstance(content, dict) and "tile_prefs" in content:
            tiles = self._genome_to_tiles(content)
            pts = self._grid_to_track_points(tiles)
            if pts is None:
                pts = self._best_effort_path(tiles)
            if pts is None:
                pts = self._default_track_points
            return np.array(pts)
        return super()._extract_content(content)

    def _make_curve(self, track_points):
        """Waypoints already trace the road (arcs + straights); only re-space
        them to the shared arc-length step."""
        return self._resample_uniform(track_points, step=self._curve_step())

    def _reencode(self, content, tiles):
        """Rewrite the genome to record the tiles WFC actually placed.

        A boost is only a request: WFC honours it when the tile is still legal
        at the moment that cell collapses, and ignores it otherwise.  Measured
        on random genomes, only ~13% of ROAD requests survive, so without this
        step most genes would describe a track that was never built, crossover
        would mix wishes rather than layouts, and offspring would share almost
        nothing with their parents.

        The paper's fix (Alg. 1 l.20, `c <- reencode(l)`) is to write the
        chosen module back into the chromosome "as if it was the chromosome's
        choice in the first place".  Every gene then describes a tile that
        really exists, so crossover recombines buildable layouts.

        The genome dict is mutated IN PLACE.  generators/search.py hands the
        chromosome's own content object to env.evaluate() without copying it,
        so writing here updates the individual the GA will breed from, which
        is exactly the paper's ordering: generate, evaluate, re-encode.

        What gets written back is the RAW WFC output, not the pruned loop.
        The paper re-encodes "the ID number of the asset that has been placed
        in the map", i.e. what the constructive algorithm chose.  Writing the
        pruned layout instead would delete every road tile that
        _keep_largest_component grassed over, and since that happens each
        generation, road could only ever leave the genome: the population
        ratchets down to tiny loops (measured: 29 road genes -> 4 in one
        round).  Keeping the raw output preserves off-loop road as material
        for later crossover.
        """
        prefs = content.get("tile_prefs", None)
        if prefs is None:
            return
        raw = self._raw_wfc.get(np.asarray(prefs, dtype=int).tobytes(), None)
        flat = np.asarray(raw if raw is not None else tiles, dtype=int).reshape(-1)
        prefs_arr = np.asarray(prefs, dtype=int)
        if flat.shape != prefs_arr.shape:
            return
        if isinstance(prefs, np.ndarray) and prefs.shape == flat.shape:
            prefs[:] = flat          # keep the GA's own array object
        else:
            content["tile_prefs"] = flat

    # ── Problem interface ─────────────────────────────────────────────

    def info(self, content, trajectory=None, use_cache=True):
        """Decode the genome to a tile loop, re-encode it, then score it.

        Content that is not a tile genome (raw track points, a bare array, or
        None) goes straight to the base problem.  That is the same test
        _extract_content applies, so the two agree on what counts as a genome,
        and every representation answers the same set of content forms.
        """
        if not (isinstance(content, dict) and "tile_prefs" in content):
            return super().info(content, trajectory=trajectory, use_cache=use_cache)
        tiles = self._genome_to_tiles(content)
        self._reencode(content, tiles)
        track_pts = self._grid_to_track_points(tiles)

        if track_pts is None or len(track_pts) < 3:
            return {
                'num_points': 0, 'total_length': 0.0,
                'avg_length': 0.0, 'max_length': 0.0, 'min_length': 0.0,
                'avg_turn': 0.0, 'max_turn': 0.0, 'min_turn': 0.0,
                'num_turns': 0, 'steps': 0, 'finished': False,
                'track_points': np.zeros((0, 2)), 'trajectory_end': None,
                'curve_points': np.zeros((0, 2)),
            }

        result = super().info(
            {"track_points": np.array(track_pts)},
            trajectory=trajectory,
            use_cache=use_cache,
        )
        return result

    # ── Hex rendering ─────────────────────────────────────────────────

    def render(self, content=None, **kwargs):
        if isinstance(content, dict) and "tile_prefs" in content:
            self._tile_render_content = content
        else:
            self._tile_render_content = None
        return super().render(content, **kwargs)

    def _render_track_bg(self, img_w, img_h, left_edge_f, right_edge_f, scaled_curve,
                         grass_color, edge_color, road_color, centerline_color, scale=1.0):
        content = getattr(self, '_tile_render_content', None)
        if content is None:
            return super()._render_track_bg(img_w, img_h, left_edge_f, right_edge_f,
                                             scaled_curve, grass_color, edge_color,
                                             road_color, centerline_color, scale=scale)
        tiles = self._genome_to_tiles(content)
        return self._make_tile_bg(img_w, img_h, tiles)

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

    def _tile_road_polyline(self, r, c, d_a, d_b):
        """The road piece inside one tile as a pixel polyline from face d_a to
        face d_b: a straight through the centre for opposite faces, otherwise
        the tangent arc (same geometry the scored centerline uses)."""
        m_a = self._face_midpoint(r, c, d_a)
        m_b = self._face_midpoint(r, c, d_b)
        if _OPPOSITE[d_a] == d_b:
            return [m_a, m_b]
        arc = self._corner_arc(r, c, d_a, d_b)  # samples from just after m_a to m_b
        return [m_a] + list(arc)

    def _draw_hex_road(self, draw, r, c, tile, sx, sy):
        edges = _OPEN_EDGES[tile]
        if not edges:
            return
        ROAD = (210, 210, 210)
        width = max(2, int(self._hex_size() * 0.5 * ((sx + sy) / 2.0)))
        d_a, d_b = tuple(edges)
        poly = [(x * sx, y * sy) for x, y in self._tile_road_polyline(r, c, d_a, d_b)]
        if len(poly) >= 2:
            draw.line(poly, fill=ROAD, width=width, joint="curve")
        # Round the caps so adjacent tiles' pieces butt together seamlessly.
        rr = max(1, width // 2)
        for px, py in (poly[0], poly[-1]):
            draw.ellipse([px - rr, py - rr, px + rr, py + rr], fill=ROAD)
