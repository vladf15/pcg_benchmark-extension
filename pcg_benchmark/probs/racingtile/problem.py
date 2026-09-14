"""Racetrack generation with Genetic-WFC (Bailly and Levieux 2023, IEEE ToG
15(1):36-45).

The pipeline is the paper's: a chromosome of one boost zone per grid cell
(Sec. III-E(a)) drives a Simple Tiled WFC (Sec. III-B), the resulting level is
evaluated, and the placed modules are written back into the chromosome
(Sec. III-E(b), Alg. 1 l.20).  Selection, crossover, mutation and elitist
reinsertion come from the benchmark's shared GA, which follows Alg. 1.

DEVIATIONS FROM THE SOURCE, and why each one is necessary here:

1.  Adjacency is derived, not learned.  The paper extracts constraints from
    manually authored sample grids (Sec. III-B).  A racetrack tile's
    compatibility is fully determined by connectivity -- a face is open or
    closed, and neighbours must agree -- so _wfc_compat computes the exact
    relation in closed form.  Learning it from examples could only approximate
    a rule we already know exactly, and there is no corpus of authored tile
    racetracks to learn from.

2.  Module weights are hand-set, not sampled frequencies.  The paper weights
    selection by how often each module appears in the sample grids.  With no
    sample grids there is nothing to count, so _WFC_WEIGHTS outweighs each road
    tile with grass, at 3.0 against 1.0 here and 4.0 on the hex lattice, whose
    denser corner vocabulary closes loops sooner.  This is the one knob
    controlling how sparse the loop is: equal weights fill the grid with road
    instead of drawing a circuit through it.

3.  Cell choice uses the possibility count, not the paper's Eq. 1.  The paper
    scores cells with H = -sum(1/N - p_i) and takes the lowest.  _run_wfc takes
    the fewest remaining possibilities, the classic WFC heuristic.  Eq. 1 as
    printed sums to zero identically for any distribution, so the intended
    form (absolute value or square) is ambiguous; rather than guess at a
    formula, this uses the standard heuristic.  The two agree whenever the
    available modules are equiprobable.

4.  No greyblock second pass.  The paper generates greyblocks and then runs a
    second WFC to swap in visual assets (Sec. III-D).  Here the tiles ARE the
    final road geometry: the benchmark scores shape and driving, not art, so
    the second pass would have nothing to substitute.

5.  Grass serves as both the air module and the border module (Sec. III-C).
    The paper uses a dedicated module for each.  Grass already has the socket
    both need, closed on every face, so separate modules would enlarge the
    vocabulary (and WFC's cost, Table I) without adding expressive power.
    The outer ring is forced to grass, which is the paper's border trick and
    also keeps the track inside the benchmark's out-of-bounds margin.

6.  Single-loop pruning (_keep_largest_component) has no counterpart in the
    paper, whose validity rule is "exactly one player spawn, else fitness
    -inf".  A racetrack needs the analogous rule, one closed circuit, but WFC's
    adjacency constraints happily produce several disjoint loops.  Setting
    fitness to -inf would leave the GA no gradient, which the paper could
    afford because spawn count is trivial to satisfy and loop count is not, so
    the largest closed loop is kept instead.

7.  Re-encoding is kept, with a different justification.  The paper's reason
    (Sec. III-E(b)) is that its selection probabilities track how many of each
    module have already been placed, so after a crossover the same chromosome
    decodes differently.  Weights here are static, so that particular
    nondeterminism cannot arise.  Re-encoding earns its place for the other
    half of the paper's argument: only ~13% of road boosts survive constraint
    propagation, so without it crossover would recombine unexpressed wishes
    rather than built layouts.

8.  An all-grass genome is seeded with one road tile.  Not in the paper, which
    has no degenerate level: air and border modules still form a playable
    space.  An all-grass grid is not a degenerate racetrack, it is no track at
    all, so WFC is given one straight at the centre to grow from.

9.  Grid is 11x11 against the paper's 15x15.  Sized for turn count, not for
    the paper's level size: 11x11 yields a median of 17.5 turns on random
    content, bracketing the ~15 average of real circuits.

DEVIATION IMPOSED BY THE HARNESS, not chosen here:

10. Crossover is uniform, not spatial.  The paper uses "a custom one point
    crossover operator that randomly chooses with equal probability to separate
    the grid horizontally or vertically" (Sec. III-E(c)), so each child
    inherits a contiguous half-grid.  The benchmark's generator calls
    contentSwap(a, b, 0.5), which swaps genes independently and scatters them,
    so a child cannot inherit a whole corner sequence intact.  Changing it
    would mean changing the shared generator, which every representation uses,
    so it is recorded here as a known departure rather than fixed locally.
"""
from __future__ import annotations

import numpy as np
from pcg_benchmark.probs.racing.problem import RacingProblem
from pcg_benchmark.spaces import ArraySpace, IntegerSpace, DictionarySpace
from PIL import Image, ImageDraw


# 11x11 grid: median 17.5 turns on random content (range 7-25 over 12 samples),
# against the ~15-turn average of famous European circuits, so the shared 1-24
# control range is reachable from both ends.  Every corner is a quarter circle
# of radius half a cell, so the radius is set by the grid rather than tunable:
# 68 m on the default 1500 m map, a medium-speed corner.
GRID_H = 11
GRID_W = 11

GRASS    = 0
STRAIGHT = 1
CORNER   = 2

N, E, S, W = 0, 1, 2, 3

_DIR_DELTA = {N: (-1, 0), E: (0, 1), S: (1, 0), W: (0, -1)}
_OPPOSITE  = {N: S, S: N, E: W, W: E}

# (tile_type, rotation) -> frozenset of open edge directions
_OPEN_EDGES = {
    **{(GRASS, r): frozenset() for r in range(4)},
    (STRAIGHT, 0): frozenset({E, W}),
    (STRAIGHT, 1): frozenset({N, S}),
    (STRAIGHT, 2): frozenset({E, W}),
    (STRAIGHT, 3): frozenset({N, S}),
    (CORNER, 0): frozenset({N, E}),
    (CORNER, 1): frozenset({E, S}),
    (CORNER, 2): frozenset({S, W}),
    (CORNER, 3): frozenset({W, N}),
}

# frozenset of two open directions -> (tile_type, rotation)
_EDGES_TO_TILE = {
    frozenset({E, W}): (STRAIGHT, 0),
    frozenset({N, S}): (STRAIGHT, 1),
    frozenset({N, E}): (CORNER, 0),
    frozenset({E, S}): (CORNER, 1),
    frozenset({S, W}): (CORNER, 2),
    frozenset({W, N}): (CORNER, 3),
}


_N_WFC_TILES    = 7     # len(_WFC_TILES): 0=grass … 6=corner_WN


class _TileGridSpace(DictionarySpace):
    """Genome = one boost zone per cell (Genetic-WFC, Bailly and Levieux 2023).

    Following Sec. III-E of the paper, each gene holds the tile whose selection
    probability is boosted when WFC collapses that cell, so the genome steers
    generation without ever overriding it: a tile that constraint propagation
    has already eliminated has probability zero, and boosting zero leaves it
    zero.  Generation therefore cannot produce an adjacency violation, and no
    request ever has to be detected and undone.

    Every cell carries a boost ("one boost zone per grid cell"): gene 0 boosts
    grass, values 1-6 boost that road tile variant at cell (r,c).  Each gene
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
                # the valid tile indices: 0 = grass, 1..6 = the road tiles.
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


class RacingTileProblem(RacingProblem):

    # Decoded points already trace a valid loop in order; the base class's
    # 2-opt untangle must not reorder them (it would break the loop).
    _untangle_control_points = False

    def __init__(self, **kwargs):
        kwargs.setdefault('num_points', 15)
        # max_steps is NOT overridden: the shared 7000-step budget applies.
        # Now that every representation generates a lap of the same length, a
        # per-representation cap would score the budget rather than the track.
        super().__init__(**kwargs)
        self._content_space = _TileGridSpace(self)
        # genome bytes -> (types, rotations, wave_array), where wave_array is a
        # (GRID_H, GRID_W) int array of WFC tile indices.  _decode_cache_keys
        # holds the same genomes in insertion order, giving the eviction
        # policy something to pop from.
        self._decode_cache: dict = {}
        self._decode_cache_keys: list = []
        # genome bytes -> raw WFC output before loop pruning, used by _reencode.
        self._raw_wfc: dict = {}
        # Bound sits well above one GA population (default 100) so a generation
        # is never evicted while it is still being evaluated.
        self._DECODE_CACHE_MAX = int(kwargs.get("decode_cache_max", 512))

        # Grid construction guarantees the track never overlaps itself (each
        # cell holds its own disjoint road piece), so the area-overlap check is
        # disabled: it only fires on artifacts of the corner-cutting waypoint
        # polyline.  This is the ONLY quality parameter tile overrides.  The
        # length band is deliberately the shared one, since lap length is
        # tuned in the generator (see _WFC_WEIGHTS) rather than by moving the
        # target, and start_straight keeps the shared 50 m figure.  Tile
        # straights are cell-quantized at 136 m on the default map, so a track
        # either clears 50 m outright or has no straight at the seam at all.
        self._QUALITY_PARAMS = {
            **self._QUALITY_PARAMS,
            "geom_area_check": False,
        }

    # ── WFC tile index constants (used internally) ────────────────────
    # Indices into _WFC_TILES: 0=grass, 1=straight_H, 2=straight_V,
    #                          3=corner_NE, 4=corner_ES, 5=corner_SW, 6=corner_WN
    _WFC_TILES = [
        (GRASS, 0),
        (STRAIGHT, 0), (STRAIGHT, 1),
        (CORNER, 0), (CORNER, 1), (CORNER, 2), (CORNER, 3),
    ]
    # Grass outweighs each road tile so WFC draws a loop through the grid
    # rather than filling it.  The ratio also sets lap length, since a denser
    # road means a longer loop: 3.0 gives 4383 m against the 4650 m median of
    # the 25 real circuits, where 6.0 gives 3577 m and 1.0 packs the grid.
    _WFC_WEIGHTS = np.array([3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    _WFC_COMPAT  = None  # populated once on first use; never changes

    # Boost-zone multiplier (Bailly and Levieux 2023, Sec. III-E use "a fixed
    # and very high boosting factor").  1000 against a grass weight of 3 makes
    # a requested tile win essentially whenever it is still legal, so the
    # genome is expressive, while a request that propagation already ruled out
    # stays impossible rather than becoming a contradiction.
    _BOOST_FACTOR = 1000.0

    @classmethod
    def _wfc_compat(cls):
        if cls._WFC_COMPAT is not None:
            return cls._WFC_COMPAT
        compat = {}
        for d in (N, E, S, W):
            opp = _OPPOSITE[d]
            compat[d] = []
            for t1, r1 in cls._WFC_TILES:
                # Tile B may sit in direction d of tile A only when their
                # touching edges agree: both open (road continues) or both
                # closed (grass meets grass).
                a_open = d in _OPEN_EDGES[(t1, r1)]
                allowed = set()
                for j, (t2, r2) in enumerate(cls._WFC_TILES):
                    b_open = opp in _OPEN_EDGES[(t2, r2)]
                    if b_open == a_open:
                        allowed.add(j)
                compat[d].append(frozenset(allowed))
        cls._WFC_COMPAT = compat
        return compat

    def _wfc_propagate(self, wave, stack, compat):
        """AC-3 constraint propagation. Returns False on contradiction."""
        while stack:
            r, c = stack.pop()
            cur = wave[r][c]
            for d, (dr, dc) in _DIR_DELTA.items():
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

    # ── Single-loop enforcement ────────────────────────────────────────

    def _build_neighbor_graph(self, types, rotations):
        """Map each road cell (r, c) to its mutually-connected neighbours.

        Two tiles are neighbours only when tile A has an open edge toward B
        *and* B has an open edge back toward A — simple grid adjacency is not
        enough.  Used by loop extraction, component filtering, and rendering."""
        neighbors: dict = {}
        for r in range(GRID_H):
            for c in range(GRID_W):
                edges = _OPEN_EDGES[(int(types[r, c]), int(rotations[r, c]))]
                if not edges:
                    continue
                nbrs = []
                for d in edges:
                    dr, dc = _DIR_DELTA[d]
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < GRID_H and 0 <= nc < GRID_W:
                        if _OPPOSITE[d] in _OPEN_EDGES[(int(types[nr, nc]), int(rotations[nr, nc]))]:
                            nbrs.append((nr, nc))
                neighbors[(r, c)] = nbrs
        return neighbors

    @staticmethod
    def _connected_components(neighbors):
        """Group the cells of a neighbour graph into connected components."""
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
        """Convert an ordered list of grid cells to pixel waypoints.

        One point per cell: the midpoint of the edge shared with the next
        cell — the exact spot where the road crosses between the two tiles."""
        n = len(cells)
        points = []
        for i, (r, c) in enumerate(cells):
            nr, nc = cells[(i + 1) % n]
            px = (c + nc + 1) * self._width  / (2 * GRID_W)
            py = (r + nr + 1) * self._height / (2 * GRID_H)
            points.append((px, py))
        return points

    def _keep_largest_component(self, types, rotations):
        """Keep only road tiles forming the largest single closed loop.

        Uses mutual-edge connectivity (tile A's open edge must match tile B's
        opposite edge) rather than simple grid adjacency.  This way two loops
        that merely touch in the grid are treated as separate candidates and
        only the biggest one survives."""
        neighbors = self._build_neighbor_graph(types, rotations)
        components = self._connected_components(neighbors)

        # A closed loop ⟺ every cell in the component has degree exactly 2
        loops = [comp for comp in components
                 if all(len(neighbors.get(cell, [])) == 2 for cell in comp)]

        if not loops:
            return types, rotations  # No valid loop; _extract_loop will reject

        largest = set(max(loops, key=len))
        types, rotations = types.copy(), rotations.copy()
        for r in range(GRID_H):
            for c in range(GRID_W):
                if types[r, c] != GRASS and (r, c) not in largest:
                    types[r, c], rotations[r, c] = GRASS, 0
        return types, rotations

    # ── WFC runner (Merrell's Model Synthesis) ────────────────────────

    def _run_wfc(self, wave, rng, compat, boosts=None):
        """Merrell's model-synthesis observe-collapse loop.

        Observe: pick the uncollapsed cell with fewest remaining possibilities,
                 break ties randomly.
        Collapse: draw a tile weighted by _WFC_WEIGHTS, multiplied by the
                  genome's boost where one applies (see `boosts`).
        Propagate: AC-3 after each collapse.

        `boosts` is the genome's boost zones as a (GRID_H, GRID_W) int array,
        one per cell (Bailly and Levieux 2023, Sec. III-E): entry 0 means no
        boost, otherwise the tile index whose selection probability is
        multiplied by _BOOST_FACTOR when this cell is collapsed. The boost only
        reweights choices that are still legal -- a tile already eliminated by
        propagation has probability zero, and scaling zero leaves it zero, so
        the genome can never force a constraint violation.

        Returns (types, rotations) or None on contradiction.
        """
        grass_i = 0
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
            w = self._WFC_WEIGHTS[possible].astype(float)
            if boosts is not None:
                # Scale up the requested tile's weight.  Every cell carries a
                # boost (gene 0 requests grass), so there is no "no request"
                # case.  If propagation has already ruled that tile out it is
                # absent from `possible`, so the request simply has no effect:
                # nothing to detect, nothing to roll back.
                w[np.asarray(possible) == boosts[r, c]] *= self._BOOST_FACTOR
            w = w / w.sum()  # normalize to probabilities
            chosen = possible[int(rng.choice(len(possible), p=w))]
            wave[r][c] = {chosen}
            if not self._wfc_propagate(wave, [(r, c)], compat):
                return None
        types     = np.zeros((GRID_H, GRID_W), dtype=int)
        rotations = np.zeros((GRID_H, GRID_W), dtype=int)
        for r in range(GRID_H):
            for c in range(GRID_W):
                ti = next(iter(wave[r][c])) if wave[r][c] else grass_i
                types[r, c], rotations[r, c] = self._WFC_TILES[ti]
        return types, rotations

    # ── Wave ↔ tile-array conversion helpers ─────────────────────────

    _WFC_TILE_IDX: dict | None = None  # (type, rotation) → WFC tile index

    @classmethod
    def _tile_to_idx(cls):
        if cls._WFC_TILE_IDX is None:
            cls._WFC_TILE_IDX = {(t, r): i for i, (t, r) in enumerate(cls._WFC_TILES)}
        return cls._WFC_TILE_IDX

    def _types_to_wave(self, types, rotations):
        """Convert (types, rotations) to a (GRID_H, GRID_W) tile-index array."""
        idx = self._tile_to_idx()
        return np.array(
            [[idx.get((int(types[r, c]), int(rotations[r, c])), 0)
              for c in range(GRID_W)]
             for r in range(GRID_H)],
            dtype=int,
        )

    @staticmethod
    def _copy_wave(wave):
        """Deep-copy a wave: a new grid where every possibility set is a copy,
        so changes to the copy never leak back into the original."""
        return [[set(cell) for cell in row] for row in wave]

    def _decode_cache_store(self, prefs_arr, types, rotations, wave, raw=None):
        """Insert a decoded genome into the cache.

        Pure memoization: with a fixed seed and no repair path, decoding is a
        function of the genome alone, so a hit is indistinguishable from a
        recompute.

        `raw` is the WFC output BEFORE loop pruning (as a tile-index array),
        kept here rather than in a separate dict so it is evicted together with
        its entry and can never go missing while the decode is still cached
        (_reencode needs both).

        The cache is bounded, oldest genome evicted first, so an overnight run
        does not grow without limit."""
        key = prefs_arr.tobytes()
        if key not in self._decode_cache:
            self._decode_cache_keys.append(prefs_arr.copy())
            while len(self._decode_cache_keys) > self._DECODE_CACHE_MAX:
                oldest = self._decode_cache_keys.pop(0)
                self._decode_cache.pop(oldest.tobytes(), None)
                self._raw_wfc.pop(oldest.tobytes(), None)
        self._decode_cache[key] = (types, rotations, wave)
        self._raw_wfc[key] = np.asarray(wave if raw is None else raw).copy()

    def _decode_genome(self, tile_prefs):
        """Decode a boost-zone genome to (types, rotations) via Genetic-WFC.

        This is Alg. 1 line 18 of Bailly and Levieux (2023), `l <- generate(c)`:
        one full WFC pass per individual, with the genome supplying the boost
        zones.  tile_prefs is a flat int array of length GRID_H*GRID_W holding
        the module ID to boost in each cell; value 0 boosts grass, 1-6 boost
        that road tile.  _run_wfc multiplies the requested module's selection
        probability at each collapse, so the genome biases generation but can
        never force a placement propagation has already ruled out.

        Deterministic: the same genome always decodes to the same track,
        because the first attempt always uses the same seed (the paper's "we
        use the same random generator seed every time we generate a level").
        Returns (types, rotations), falling back to a rectangle if every
        attempt contradicts.
        """
        tile_prefs = np.asarray(tile_prefs, dtype=int)
        if tile_prefs.size != GRID_H * GRID_W:
            raise ValueError(
                "tile_prefs must have %d entries for this %dx%d grid, got %d"
                % (GRID_H * GRID_W, GRID_H, GRID_W, tile_prefs.size))
        key = tile_prefs.tobytes()
        if key in self._decode_cache:
            t, r_, _ = self._decode_cache[key]
            return t, r_

        compat   = self._wfc_compat()
        n_tiles  = len(self._WFC_TILES)
        grass_i  = 0
        prefs_2d = tile_prefs.reshape(GRID_H, GRID_W)

        # ── Full WFC from scratch ──────────────────────────────────────────
        # Build the base wave once: the border is a hard constraint (level
        # structure, not a genome request), so it is forced here.  The genome
        # itself never touches the wave — it is passed to _run_wfc as boost
        # zones, which only reweight choices that are already legal.
        base_wave = [[set(range(n_tiles)) for _ in range(GRID_W)]
                     for _ in range(GRID_H)]
        border = []
        for r in range(GRID_H):
            for c in range(GRID_W):
                if r == 0 or r == GRID_H - 1 or c == 0 or c == GRID_W - 1:
                    base_wave[r][c] = {grass_i}
                    border.append((r, c))
        self._wfc_propagate(base_wave, border, compat)

        # A genome that asks for grass everywhere still needs a road seed,
        # otherwise WFC has nothing to build a loop from.  This is a property
        # of the all-grass genome rather than of any request, so it stays a
        # forced observation.
        if not np.any(prefs_2d[1:GRID_H - 1, 1:GRID_W - 1]):
            sr, sc = GRID_H // 2, GRID_W // 2
            straight_h = 1
            if straight_h in base_wave[sr][sc]:
                base_wave[sr][sc] = {straight_h}
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

            types, rotations = self._keep_largest_component(*result)
            if self._extract_loop(types, rotations) is not None:
                new_wave = self._types_to_wave(types, rotations)
                # Store the RAW WFC output alongside: re-encoding must record
                # what WFC placed, not what survived pruning (see _reencode).
                self._decode_cache_store(tile_prefs, types, rotations, new_wave,
                                         raw=self._types_to_wave(*result))
                return types, rotations

        # Total failure — deterministic rectangle as last resort
        rng = np.random.default_rng(int(np.sum(tile_prefs)) % (2**31))
        fb  = self._rectangular_fallback(rng)
        t   = np.asarray(fb["types"],     dtype=int)
        r_  = np.asarray(fb["rotations"], dtype=int)
        w   = self._types_to_wave(t, r_)
        self._decode_cache_store(tile_prefs, t, r_, w)
        return t, r_

    def init_content(self, rng=None):
        """Return a random dense boost-zone genome, one boost per cell.

        Every cell carries a boost, as in Bailly and Levieux 2023 Sec. III-E
        ("We use one boost zone per grid cell").  Gene 0 boosts grass and
        1.._N_WFC_TILES-1 boost that road tile.

        Density is what makes the genome expressive.  WFC collapses the
        most-constrained cells first, so a cell carrying no boost is usually
        decided by its neighbours long before its own gene would be consulted;
        at a sparse ~15% of cells only 13% of requests reach the layout.
        Boosting every cell gives the genome a say wherever WFC looks, which is
        what makes offspring resemble their parents.

        Genes are drawn UNIFORMLY over the whole vocabulary, which is the
        paper's "the first population is initialized with random chromosomes"
        (Sec. III-E(c)).  Grass is one value of seven rather than a weighted
        majority for two reasons.  Biasing the draw toward grass starves the
        genome of road and cuts track length by two thirds (1066 m mean at a
        15% road rate against 3031 m uniform, on a 3273 m min_length).  It also
        keeps mutation effective, since contentSwap draws replacement genes
        from this function and a grass-heavy draw would make most mutations
        grass-onto-grass no-ops.  The preference for sparse loops lives in the
        grass-weighted WFC collapse (_WFC_WEIGHTS) instead, which is where it
        belongs."""
        if rng is None:
            rng = np.random.default_rng()
        elif isinstance(rng, int):
            rng = np.random.default_rng(rng)
        n = GRID_H * GRID_W
        tile_prefs = rng.integers(0, _N_WFC_TILES, size=n).astype(int)
        return {"tile_prefs": tile_prefs}

    def _rectangular_fallback(self, rng):
        min_dim = 3
        r0 = int(rng.integers(1, GRID_H - min_dim - 1))
        c0 = int(rng.integers(1, GRID_W - min_dim - 1))
        r1 = int(rng.integers(r0 + min_dim, min(r0 + min_dim + 5, GRID_H - 1) + 1))
        c1 = int(rng.integers(c0 + min_dim, min(c0 + min_dim + 5, GRID_W - 1) + 1))
        # Walk the rectangle clockwise: top edge, right edge, bottom, left.
        loop = []
        for c in range(c0, c1):
            loop.append((r0, c))
        for r in range(r0, r1):
            loop.append((r, c1))
        for c in range(c1, c0, -1):
            loop.append((r1, c))
        for r in range(r1, r0, -1):
            loop.append((r, c0))

        types     = np.zeros((GRID_H, GRID_W), dtype=int)
        rotations = np.zeros((GRID_H, GRID_W), dtype=int)
        n = len(loop)
        for i, (r, c) in enumerate(loop):
            pr, pc = loop[(i - 1) % n]
            nr, nc = loop[(i + 1) % n]
            from_dir = self._direction_toward(r, c, pr, pc)
            to_dir   = self._direction_toward(r, c, nr, nc)
            t, rot = _EDGES_TO_TILE.get(frozenset({from_dir, to_dir}), (GRASS, 0))
            types[r, c], rotations[r, c] = t, rot
        return {"types": types, "rotations": rotations}

    @staticmethod
    def _direction_toward(r, c, target_r, target_c):
        """Return the compass direction that steps from (r, c) to the target cell."""
        for d, (dr, dc) in _DIR_DELTA.items():
            if r + dr == target_r and c + dc == target_c:
                return d
        return None

    # ── Decoding: tile grid -> track waypoints ────────────────────────

    def _extract_loop(self, types, rotations):
        """
        Traverse mutually-connected road tiles to find a closed loop.
        Returns ordered list of (r, c) or None if no valid cycle exists.
        """
        neighbors = self._build_neighbor_graph(types, rotations)

        # A simple cycle requires every node to have degree exactly 2
        cycle_cells = {cell for cell, nbrs in neighbors.items() if len(nbrs) == 2}
        if not cycle_cells:
            return None

        # Walk the loop: from each cell, continue to the neighbour we did not
        # just come from.  (next(iter(...)) takes an arbitrary element from
        # the set; sets cannot be indexed.)
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

    # How many samples each corner tile's quarter-circle arc contributes to
    # the track polyline.  6 samples over a 68 m-radius quarter circle is one
    # point every 15 degrees (~18 m); _make_curve then re-spaces the polyline
    # at 5.0 m, so a corner is driven as a 6-sided approximation of its arc
    # (0.6 m of corner-cutting at mid-chord, against a 16 m track width).
    _ARC_SAMPLES = 6

    def _corner_arc_center(self, r, c, rotation):
        """The cell corner both open edges touch: the arc's centre point."""
        cw = self._width / GRID_W
        ch = self._height / GRID_H
        x0, y0 = c * cw, r * ch
        return {
            0: (x0 + cw, y0),       # corner NE
            1: (x0 + cw, y0 + ch),  # corner ES
            2: (x0,      y0 + ch),  # corner SW
            3: (x0,      y0),       # corner WN
        }[int(rotation)]

    def _loop_to_track_points(self, loop, types, rotations):
        """Convert an ordered cell loop to waypoints that follow the actual
        tile road geometry.

        Straight tiles contribute their exit-edge midpoint (the road inside
        them is the exact straight between entry and exit midpoints).
        Corner tiles contribute _ARC_SAMPLES points along their true
        quarter-circle centerline (radius = half a cell, centred on the
        cell corner shared by the two open edges), ending at the exit
        midpoint.  The polyline therefore IS the drawn road, and the curve
        step only densifies it instead of fitting a spline (which bowed the
        straights and wobbled at corners because the edge-midpoint chords
        alternate 41.7 m / 29.5 m spacing)."""
        n = len(loop)
        points = []
        for i, (r, c) in enumerate(loop):
            pr, pc = loop[(i - 1) % n]
            nr, nc = loop[(i + 1) % n]
            exit_mid = (
                (c + nc + 1) * self._width / (2 * GRID_W),
                (r + nr + 1) * self._height / (2 * GRID_H),
            )
            if int(types[r, c]) != CORNER:
                points.append(exit_mid)
                continue

            entry_mid = (
                (c + pc + 1) * self._width / (2 * GRID_W),
                (r + pr + 1) * self._height / (2 * GRID_H),
            )
            cx, cy = self._corner_arc_center(r, c, rotations[r, c])
            radius = 0.5 * self._width / GRID_W
            a0 = np.arctan2(entry_mid[1] - cy, entry_mid[0] - cx)
            a1 = np.arctan2(exit_mid[1] - cy, exit_mid[0] - cx)
            sweep = (a1 - a0 + np.pi) % (2.0 * np.pi) - np.pi  # shortest way
            for s in range(1, self._ARC_SAMPLES + 1):
                a = a0 + sweep * s / self._ARC_SAMPLES
                points.append((cx + radius * np.cos(a), cy + radius * np.sin(a)))
        return points

    def _grid_to_track_points(self, types, rotations):
        """Convert tile grid to ordered pixel waypoints along the tile road."""
        loop = self._extract_loop(types, rotations)
        if loop is None:
            return None
        return self._loop_to_track_points(loop, types, rotations)

    def _best_effort_path(self, types, rotations):
        """For rendering: find the largest connected chain of road tiles even if
        it is not a closed loop. Returns ordered pixel waypoints or None."""
        neighbors = self._build_neighbor_graph(types, rotations)
        components = self._connected_components(neighbors)

        best = max(components, key=len) if components else []
        if len(best) < 2:
            return None

        # Traverse in order: start from an endpoint (degree 1) if one exists
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
    # Overriding this makes all parent methods (simulate, render, etc.)
    # work transparently with tile-grid content dicts.

    def _genome_to_tiles(self, content):
        """Decode a tile-preference genome dict to (types, rotations)."""
        return self._decode_genome(np.asarray(content["tile_prefs"], dtype=int))

    def _extract_content(self, content):
        if isinstance(content, dict) and "tile_prefs" in content:
            types, rotations = self._genome_to_tiles(content)
            pts = self._grid_to_track_points(types, rotations)
            if pts is None:
                pts = self._best_effort_path(types, rotations)
            if pts is None:
                pts = self._default_track_points
            return np.array(pts)
        return super()._extract_content(content)

    def _make_curve(self, track_points):
        """Tile waypoints already trace the exact road (straights + sampled
        arcs), so the curve is only re-spaced to the shared arc-length step;
        a spline through them would re-introduce wobble."""
        return self._resample_uniform(track_points, step=self._curve_step())

    def _reencode(self, content, types, rotations):
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
        flat = (np.asarray(raw, dtype=int) if raw is not None
                else self._types_to_wave(types, rotations)).reshape(-1)
        prefs_arr = np.asarray(prefs, dtype=int)
        if flat.shape != prefs_arr.shape:
            return
        if isinstance(prefs, np.ndarray) and prefs.shape == flat.shape:
            prefs[:] = flat          # keep the GA's own array object
        else:
            content["tile_prefs"] = flat

    def info(self, content, trajectory=None, use_cache=True):
        """Decode the genome to a tile loop, re-encode it, then score it.

        Content that is not a tile genome (raw track points, a bare array, or
        None) goes straight to the base problem.  That is the same test
        _extract_content applies, so the two agree on what counts as a genome,
        and every representation answers the same set of content forms.
        """
        if not (isinstance(content, dict) and "tile_prefs" in content):
            return super().info(content, trajectory=trajectory, use_cache=use_cache)
        types, rotations = self._genome_to_tiles(content)
        self._reencode(content, types, rotations)
        track_pts = self._grid_to_track_points(types, rotations)

        if track_pts is None or len(track_pts) < 3:
            return {
                'num_points': 0, 'total_length': 0.0,
                'avg_length': 0.0, 'max_length': 0.0, 'min_length': 0.0,
                'avg_turn': 0.0, 'max_turn': 0.0, 'min_turn': 0.0,
                'num_turns': 0, 'steps': 0, 'finished': False,
                'track_points': np.zeros((0, 2)), 'trajectory_end': None,
                'curve_points': np.zeros((0, 2)),
            }

        return super().info(
            {"track_points": np.array(track_pts)},
            trajectory=trajectory,
            use_cache=use_cache,
        )

    # ── Tile rendering ────────────────────────────────────────────────

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
        types, rotations = self._genome_to_tiles(content)
        return self._make_tile_bg(img_w, img_h, types, rotations)

    def _make_tile_bg(self, img_w, img_h, types, rotations):
        bg   = Image.new("RGB", (img_w, img_h), (34, 139, 34))
        draw = ImageDraw.Draw(bg)
        cw   = img_w / GRID_W
        ch   = img_h / GRID_H
        for r in range(GRID_H):
            for c in range(GRID_W):
                self._draw_tile_pil(draw, c * cw, r * ch, cw, ch,
                                    int(types[r, c]), int(rotations[r, c]))
        for i in range(GRID_H + 1):
            y = int(i * ch)
            draw.line([0, y, img_w, y], fill=(20, 100, 20), width=1)
        for j in range(GRID_W + 1):
            x = int(j * cw)
            draw.line([x, 0, x, img_h], fill=(20, 100, 20), width=1)
        return bg

    def _draw_tile_pil(self, draw, x0, y0, cw, ch, t, rot):
        ROAD = (210, 210, 210)
        EDGE = (25,  25,  25)
        GRASS_C = (34, 139, 34)
        draw.rectangle([x0, y0, x0 + cw - 1, y0 + ch - 1], fill=GRASS_C)
        if t == GRASS:
            return
        rw = cw * 0.5
        rh = ch * 0.5
        if t == STRAIGHT:
            if rot in (0, 2):
                ry = y0 + (ch - rh) / 2
                draw.rectangle([x0, ry, x0 + cw, ry + rh], fill=ROAD)
                draw.line([x0, ry,      x0 + cw, ry],      fill=EDGE, width=2)
                draw.line([x0, ry + rh, x0 + cw, ry + rh], fill=EDGE, width=2)
            else:
                rx = x0 + (cw - rw) / 2
                draw.rectangle([rx, y0, rx + rw, y0 + ch], fill=ROAD)
                draw.line([rx,      y0, rx,      y0 + ch], fill=EDGE, width=2)
                draw.line([rx + rw, y0, rx + rw, y0 + ch], fill=EDGE, width=2)
        else:
            _C = {0: (x0+cw, y0,    90,  180),
                  1: (x0+cw, y0+ch, 180, 270),
                  2: (x0,    y0+ch, 270, 360),
                  3: (x0,    y0,    0,   90)}
            cx, cy, s, e = _C[rot]
            r_in, r_out, steps = cw / 4, cw * 3 / 4, 24
            pts = []
            for i in range(steps + 1):
                a = np.deg2rad(s + (e - s) * i / steps)
                pts.append((cx + r_out * np.cos(a), cy + r_out * np.sin(a)))
            for i in range(steps + 1):
                a = np.deg2rad(e - (e - s) * i / steps)
                pts.append((cx + r_in * np.cos(a), cy + r_in * np.sin(a)))
            draw.polygon(pts, fill=ROAD)
            for ar in (r_in, r_out):
                draw.arc([cx - ar, cy - ar, cx + ar, cy + ar], s, e, fill=EDGE, width=2)
