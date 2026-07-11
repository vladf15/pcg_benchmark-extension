from __future__ import annotations

import zlib
import numpy as np
from pcg_benchmark.probs.racing.problem import RacingProblem
from pcg_benchmark.spaces import ArraySpace, IntegerSpace, DictionarySpace
from PIL import Image, ImageDraw


GRID_H = 12
GRID_W = 12

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
_ROAD_GENE_RATE = 0.15  # fraction of cells with an active road request in random genomes


class _TileGridSpace(DictionarySpace):
    """Genome = a partial tile map (Genetic-WFC style, Bailly & Levieux 2022).

    tile_prefs[r*GRID_W+c] == 0 means "no preference" — WFC decides the cell
    freely.  Values 1-6 request that road tile variant at cell (r,c); requests
    are stamped as hard pre-collapse observations before WFC fills the rest,
    and silently skipped if they contradict earlier stamps.  Each active gene
    therefore maps directly to one tile of the layout, so crossover and
    mutation via contentSwap make small, local changes to the track."""

    def __init__(self, problem_ref):
        super().__init__({
            "tile_prefs": ArraySpace(
                (GRID_H * GRID_W,),
                IntegerSpace(0, _N_WFC_TILES),
            ),
        })
        self._prob = problem_ref

    def sample(self):
        return self._prob.init_content()


class RacingTileProblem(RacingProblem):

    def __init__(self, **kwargs):
        kwargs.setdefault('num_points', 15)
        kwargs.setdefault('max_steps', 2000)
        super().__init__(**kwargs)
        self._content_space = _TileGridSpace(self)
        # genome bytes → (types, rotations, wave_array)
        # wave_array is a (GRID_H, GRID_W) int array of WFC tile indices;
        # stored for localised repair (Bailly & Levieux 2022).
        # _decode_cache_keys holds the same genomes as arrays (in insertion
        # order) so the nearest-genome scan is one vectorized comparison
        # instead of a Python loop over the whole cache.
        self._decode_cache: dict = {}
        self._decode_cache_keys: list = []

        # Tile tracks live on a fixed grid, so the generic length targets must
        # be expressed in cell units: waypoints are tile-edge midpoints spaced
        # ~1 cell apart, and a good loop uses 24-60 tiles.  Grid construction
        # guarantees the track never overlaps itself (each cell holds its own
        # disjoint road piece), so the area-overlap check is disabled — it
        # only fires on artifacts of the corner-cutting waypoint polyline.
        cell = float(self._width) / GRID_W
        self._QUALITY_PARAMS = {
            **self._QUALITY_PARAMS,
            "min_length":   24.0 * cell,
            "max_length":   60.0 * cell,
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
    # Grass weighted heavily so road tiles form sparse loops rather than filling the grid
    _WFC_WEIGHTS = np.array([6.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    _WFC_COMPAT  = None  # populated once on first use; never changes

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
        loops = []
        for comp in components:
            is_loop = True
            for cell in comp:
                if len(neighbors.get(cell, [])) != 2:
                    is_loop = False
                    break
            if is_loop:
                loops.append(comp)

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

    def _run_wfc(self, wave, rng, compat):
        """Merrell's model-synthesis observe-collapse loop.

        Observe: pick the uncollapsed cell with fewest remaining possibilities,
                 break ties randomly.
        Collapse: draw a tile weighted by _WFC_WEIGHTS.
        Propagate: AC-3 after each collapse.

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
            w = self._WFC_WEIGHTS[possible]
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

    def _wave_to_types(self, wave_arr):
        """Convert a (GRID_H, GRID_W) tile-index array to (types, rotations)."""
        types = np.zeros((GRID_H, GRID_W), dtype=int)
        rots  = np.zeros((GRID_H, GRID_W), dtype=int)
        for r in range(GRID_H):
            for c in range(GRID_W):
                types[r, c], rots[r, c] = self._WFC_TILES[int(wave_arr[r, c])]
        return types, rots

    def _types_to_wave(self, types, rotations):
        """Convert (types, rotations) to a (GRID_H, GRID_W) tile-index array."""
        idx      = self._tile_to_idx()
        wave_arr = np.zeros((GRID_H, GRID_W), dtype=int)
        for r in range(GRID_H):
            for c in range(GRID_W):
                wave_arr[r, c] = idx.get((int(types[r, c]), int(rotations[r, c])), 0)
        return wave_arr

    # ── Localised repair (Bailly & Levieux 2022, §3.2) ───────────────

    def _localised_repair(self, new_prefs, old_prefs, old_wave):
        """Repair a wave by only re-collapsing cells whose preferences changed.

        Start from old_wave (a valid collapsed track).  Un-collapse the cells
        whose genome preference differs, propagate constraints outward via AC-3
        from their still-collapsed neighbours, re-stamp the new preferences,
        then re-run WFC on the remaining uncollapsed cells only.

        Returns (types, rotations, new_wave) or None if repair fails (caller
        falls back to a full WFC pass).
        """
        compat  = self._wfc_compat()
        n_tiles = len(self._WFC_TILES)
        new_p2d = new_prefs.reshape(GRID_H, GRID_W)
        old_p2d = old_prefs.reshape(GRID_H, GRID_W)

        # Reconstruct set-based wave from collapsed tile indices; borders stay {0}.
        wave = []
        for r in range(GRID_H):
            row = []
            for c in range(GRID_W):
                row.append({int(old_wave[r, c])})
            wave.append(row)

        # Interior cells whose genome preference changed → need re-collapsing.
        diff_cells = []
        for r in range(1, GRID_H - 1):
            for c in range(1, GRID_W - 1):
                if int(new_p2d[r, c]) != int(old_p2d[r, c]):
                    diff_cells.append((r, c))

        if not diff_cells:
            # Nothing changed — reconstruct directly without any WFC work.
            t, ro = self._wave_to_types(old_wave)
            return t, ro, old_wave.copy()

        # Un-collapse changed cells to the full set of possibilities.
        uncollapsed = set(diff_cells)
        for r, c in diff_cells:
            wave[r][c] = set(range(n_tiles))

        # Propagate constraints inward from the still-collapsed neighbours.
        seed: list = []
        seen_seed: set = set()
        for r, c in uncollapsed:
            for dr, dc in _DIR_DELTA.values():
                nr, nc = r + dr, c + dc
                if (0 <= nr < GRID_H and 0 <= nc < GRID_W
                        and (nr, nc) not in uncollapsed
                        and (nr, nc) not in seen_seed):
                    seed.append((nr, nc))
                    seen_seed.add((nr, nc))

        if not self._wfc_propagate(wave, seed, compat):
            return None  # Contradiction → fall back to full WFC

        # Stamp new preferences for changed cells (skip on conflict).
        for r, c in diff_cells:
            pref = int(new_p2d[r, c])
            if pref == 0 or pref not in wave[r][c] or len(wave[r][c]) == 1:
                continue
            snapshot = self._copy_wave(wave)
            wave[r][c] = {pref}
            if not self._wfc_propagate(wave, [(r, c)], compat):
                wave = snapshot  # Preference contradicts context — skip it

        # Re-run WFC on the (now small) set of uncollapsed cells.
        # _run_wfc skips cells already collapsed (len == 1), so only the
        # repaired region gets new random choices.
        base_repair = wave  # keep base so we can retry with different seeds
        genome_seed = zlib.crc32(np.ascontiguousarray(new_prefs).tobytes())
        for attempt in range(20):
            wave_copy = self._copy_wave(base_repair)
            result = self._run_wfc(wave_copy, np.random.default_rng((genome_seed + attempt * 7_919) % (2**32)), compat)
            if result is None:
                continue
            types, rotations = self._keep_largest_component(*result)
            if self._extract_loop(types, rotations) is not None:
                new_wave = self._types_to_wave(types, rotations)
                return types, rotations, new_wave

        return None  # All repair attempts failed → caller uses full WFC

    @staticmethod
    def _copy_wave(wave):
        """Deep-copy a wave: a new grid where every possibility set is a copy,
        so changes to the copy never leak back into the original."""
        copied = []
        for row in wave:
            copied.append([set(cell) for cell in row])
        return copied

    def _decode_cache_store(self, prefs_arr, types, rotations, wave):
        """Insert a decoded genome into the cache (dict + key array list)."""
        self._decode_cache[prefs_arr.tobytes()] = (types, rotations, wave)
        self._decode_cache_keys.append(prefs_arr.copy())

    def _decode_genome(self, tile_prefs):
        """Decode a partial-map genome to (types, rotations) via Genetic-WFC.

        tile_prefs is a flat int array of length GRID_H*GRID_W.  Value 0 means
        "no preference"; values 1-6 request that road tile variant at the cell.
        Requests are stamped as hard pre-collapse observations in row-major
        order — each is propagated immediately and skipped (wave restored) if
        it contradicts earlier stamps.  WFC then fills the remaining cells.
        Deterministic (fixed per-attempt RNG) and cached per genome.
        Returns (types, rotations) — falls back to a rectangle on total failure.
        """
        tile_prefs = np.asarray(tile_prefs, dtype=int)
        key = tile_prefs.tobytes()
        if key in self._decode_cache:
            t, r_, _ = self._decode_cache[key]
            return t, r_

        compat   = self._wfc_compat()
        n_tiles  = len(self._WFC_TILES)
        grass_i  = 0
        prefs_2d = tile_prefs.reshape(GRID_H, GRID_W)

        # ── Localised repair (Bailly & Levieux 2022) ──────────────────────
        # Before doing a full WFC pass, check whether any cached genome is
        # "close enough" to this one.  The threshold must stay at mutation
        # scale (contentSwap at rate 0.05 on 144 cells changes ~7): two
        # *unrelated* sparse random genomes already differ in only ~40 cells,
        # so a generous threshold makes every fresh genome "repair" from the
        # first decoded track and inherit it almost verbatim, collapsing the
        # whole population onto one phenotype.  Only mutation-sized diffs get
        # the localised treatment; anything larger does a full WFC pass.
        if self._decode_cache:
            # Compare this genome against every cached genome at once:
            # np.stack piles the cached genomes into a matrix (one per row),
            # != marks each differing cell, count_nonzero counts them per row.
            diffs = np.count_nonzero(np.stack(self._decode_cache_keys) != tile_prefs, axis=1)
            j = int(np.argmin(diffs))
            if int(diffs[j]) <= 15:
                best_old_prefs = self._decode_cache_keys[j]
                best_old_wave  = self._decode_cache[best_old_prefs.tobytes()][2]
                repaired = self._localised_repair(tile_prefs, best_old_prefs, best_old_wave)
                if repaired is not None:
                    types, rotations, new_wave = repaired
                    self._decode_cache_store(tile_prefs, types, rotations, new_wave)
                    return types, rotations

        # ── Full WFC from scratch ──────────────────────────────────────────
        # Build the stamped wave once — stamping is deterministic, only the
        # WFC fill afterwards varies between attempts.
        base_wave = []
        for r in range(GRID_H):
            row = []
            for c in range(GRID_W):
                row.append(set(range(n_tiles)))
            base_wave.append(row)
        border = []
        for r in range(GRID_H):
            for c in range(GRID_W):
                if r == 0 or r == GRID_H - 1 or c == 0 or c == GRID_W - 1:
                    base_wave[r][c] = {grass_i}
                    border.append((r, c))
        self._wfc_propagate(base_wave, border, compat)

        # Stamp the genome's road requests as hard observations
        stamped = 0
        for r in range(1, GRID_H - 1):
            for c in range(1, GRID_W - 1):
                pref = int(prefs_2d[r, c])
                if pref == 0 or pref not in base_wave[r][c]:
                    continue
                if len(base_wave[r][c]) == 1:
                    stamped += 1  # already forced to the requested tile
                    continue
                snapshot = self._copy_wave(base_wave)
                base_wave[r][c] = {pref}
                if self._wfc_propagate(base_wave, [(r, c)], compat):
                    stamped += 1
                else:
                    base_wave = snapshot

        # A genome with no expressible requests still needs a road seed,
        # otherwise grass-heavy WFC tends to produce an empty grid.
        if stamped == 0:
            sr, sc = GRID_H // 2, GRID_W // 2
            straight_h = 1
            if straight_h in base_wave[sr][sc]:
                base_wave[sr][sc] = {straight_h}
                self._wfc_propagate(base_wave, [(sr, sc)], compat)

        # Seed the fill from the genome so different genomes explore different
        # layouts (a fixed seed sequence makes weakly-stamped genomes collapse
        # to near-identical tracks), while staying deterministic per genome.
        genome_seed = zlib.crc32(key)
        for attempt in range(100):
            rng  = np.random.default_rng((genome_seed + attempt * 1_000_003) % (2**32))
            wave = self._copy_wave(base_wave)

            result = self._run_wfc(wave, rng, compat)
            if result is None:
                continue

            types, rotations = self._keep_largest_component(*result)
            if self._extract_loop(types, rotations) is not None:
                new_wave = self._types_to_wave(types, rotations)
                self._decode_cache_store(tile_prefs, types, rotations, new_wave)
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
        """Return a random sparse partial-map genome.

        Only ~_ROAD_GENE_RATE of cells carry an active road request — a loop
        uses 16-40 of the 144 cells, so denser genomes just produce conflicting
        stamps that get skipped.  Sparseness also keeps contentSwap mutation
        sparse, since it swaps cells with a fresh sample from this function.
        No WFC pre-validation: _decode_genome handles failures internally."""
        if rng is None:
            rng = np.random.default_rng()
        elif isinstance(rng, int):
            rng = np.random.default_rng(rng)
        n = GRID_H * GRID_W
        active = rng.random(n) < _ROAD_GENE_RATE
        tile_prefs = np.where(active, rng.integers(1, _N_WFC_TILES, size=n), 0).astype(int)
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
        cycle_cells = set()
        for cell, nbrs in neighbors.items():
            if len(nbrs) == 2:
                cycle_cells.add(cell)
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

    def _grid_to_track_points(self, types, rotations):
        """Convert tile grid to ordered pixel waypoints (edge midpoints)."""
        loop = self._extract_loop(types, rotations)
        if loop is None:
            return None
        return self._edge_midpoints(loop)

    def _best_effort_path(self, types, rotations):
        """For rendering: find the largest connected chain of road tiles even if
        it is not a closed loop. Returns ordered pixel waypoints or None."""
        neighbors = self._build_neighbor_graph(types, rotations)
        components = self._connected_components(neighbors)

        if components:
            best = max(components, key=len)
        else:
            best = []
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

    # ── Problem interface ─────────────────────────────────────────────

    def info(self, content, trajectory=None, use_cache=True):
        types, rotations = self._genome_to_tiles(content)
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
        import math as _m
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
                a = _m.radians(s + (e - s) * i / steps)
                pts.append((cx + r_out * _m.cos(a), cy + r_out * _m.sin(a)))
            for i in range(steps + 1):
                a = _m.radians(e - (e - s) * i / steps)
                pts.append((cx + r_in * _m.cos(a), cy + r_in * _m.sin(a)))
            draw.polygon(pts, fill=ROAD)
            for ar in (r_in, r_out):
                draw.arc([cx - ar, cy - ar, cx + ar, cy + ar], s, e, fill=EDGE, width=2)
