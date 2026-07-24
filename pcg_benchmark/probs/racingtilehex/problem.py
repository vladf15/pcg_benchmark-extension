from __future__ import annotations

import zlib
import numpy as np
from pcg_benchmark.probs.racing.problem import RacingProblem
from pcg_benchmark.spaces import ArraySpace, IntegerSpace, DictionarySpace
from PIL import Image, ImageDraw


# ── Hex grid (pointy-top, odd-r offset storage) ───────────────────────────
# The map is stored as a (GRID_H, GRID_W) array, exactly like the square tile
# problem, so border forcing, the genome shape, and the render loop carry over
# unchanged.  The only difference is the *neighbour graph*: each cell has six
# faces instead of four, so a road tile can connect any face to any other face
# and the natural turn menu becomes {0 (straight), 60, 120} degrees with no
# 90-degree corners at all.
# 11x11 grid (2026-07-24): gives ~16.5 turns on random content, matching the
# ~15-turn average of famous European circuits and the other representations
# (was 12x12 ~ 24 turns - the hex 6-way connectivity makes it corner-dense).
# The 60-degree corner radius grows slightly to ~18.8 m, still realistic.
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
_ROAD_GENE_RATE = 0.15  # fraction of cells with an active road request in random genomes


class _HexGridSpace(DictionarySpace):
    """Genome = a partial hex-tile map (Genetic-WFC style, Bailly & Levieux 2022).

    tile_prefs[r*GRID_W+c] == 0 means "no preference" — WFC decides the cell
    freely.  Values 1-15 request that face-pair road tile at cell (r,c);
    requests are stamped as hard pre-collapse observations before WFC fills the
    rest, and silently skipped if they contradict earlier stamps.  Each active
    gene therefore maps directly to one tile of the layout, so crossover and
    mutation via contentSwap make small, local changes to the track."""

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

    def sample(self):
        return self._prob.init_content()


class RacingTileHexProblem(RacingProblem):
    """Racetrack generation on a hexagonal WFC grid.

    Identical in spirit to racingtile-v0 (Genetic-WFC over a fixed lattice with
    localised repair), but the lattice is hexagonal: each cell has six faces, a
    road tile may connect ANY face to ANY other face, and corners are 60 or 120
    degrees instead of a fixed 90.  This removes the boxy right-angle look of
    the square tiles while keeping the same constructive drivability guarantee
    (each cell holds its own disjoint road piece, so tracks never self-overlap).
    """

    # Decoded points already trace a valid loop in order; the base class's
    # 2-opt untangle must not reorder them (it would break the loop).
    _untangle_control_points = False

    def __init__(self, **kwargs):
        kwargs.setdefault('num_points', 15)
        kwargs.setdefault('max_steps', 2000)
        super().__init__(**kwargs)
        self._content_space = _HexGridSpace(self)
        # genome bytes -> (tiles, wave_array); wave_array is a (GRID_H, GRID_W)
        # int array of WFC tile indices, stored for localised repair.
        self._decode_cache: dict = {}
        self._decode_cache_keys: list = []
        self._DECODE_CACHE_MAX = int(kwargs.get("decode_cache_max", 512))

        # Hex tracks live on a fixed grid, so the generic length targets are
        # expressed in cell units.  A hex cell's centre-to-centre spacing is
        # sqrt(3) * size horizontally / 1.5 * size vertically; we use the
        # horizontal pitch as the nominal cell size for the length bands.
        cell = float(self._width) / GRID_W
        self._QUALITY_PARAMS = {
            **self._QUALITY_PARAMS,
            "min_length":   24.0 * cell,
            "max_length":   60.0 * cell,
            "geom_area_check": False,
        }

    # ── Hex pixel geometry ────────────────────────────────────────────
    # A pointy-top hex of "size" (centre-to-vertex).  We pick size so the grid
    # of GRID_W columns spans the playfield width, matching the square version's
    # habit of filling self._width / self._height.

    def _hex_size(self):
        # Horizontal centre spacing is sqrt(3) * size; GRID_W columns plus the
        # half-hex odd-row offset must fit in self._width.
        return float(self._width) / (np.sqrt(3.0) * (GRID_W + 0.5))

    def _hex_center(self, r, c):
        size = self._hex_size()
        x = size * np.sqrt(3.0) * (c + 0.5 * (r & 1) + 0.5)
        y = size * 1.5 * (r + 0.5)
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

    @classmethod
    def _weights(cls):
        if cls._WFC_WEIGHTS is None:
            w = np.ones(_N_WFC_TILES, dtype=float)
            w[GRASS] = 6.0  # grass heavy so road forms sparse loops, not a fill
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

    def _run_wfc(self, wave, rng, compat):
        """Observe (min-entropy) / collapse (weighted) / propagate (AC-3).

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
            w = weights[possible]
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

    # ── Localised repair (Bailly & Levieux 2022, §3.2) ───────────────

    def _localised_repair(self, new_prefs, old_prefs, old_wave):
        """Repair a wave by only re-collapsing cells whose preferences changed."""
        compat  = self._wfc_compat()
        n_tiles = _N_WFC_TILES
        new_p2d = new_prefs.reshape(GRID_H, GRID_W)
        old_p2d = old_prefs.reshape(GRID_H, GRID_W)

        wave = [[{int(old_wave[r, c])} for c in range(GRID_W)] for r in range(GRID_H)]

        diff_cells = []
        for r in range(1, GRID_H - 1):
            for c in range(1, GRID_W - 1):
                if int(new_p2d[r, c]) != int(old_p2d[r, c]):
                    diff_cells.append((r, c))

        if not diff_cells:
            return old_wave.copy(), old_wave.copy()

        uncollapsed = set(diff_cells)
        for r, c in diff_cells:
            wave[r][c] = set(range(n_tiles))

        seed: list = []
        seen_seed: set = set()
        for r, c in uncollapsed:
            for d in _DIRS:
                nr, nc = _neighbor(r, c, d)
                if (0 <= nr < GRID_H and 0 <= nc < GRID_W
                        and (nr, nc) not in uncollapsed
                        and (nr, nc) not in seen_seed):
                    seed.append((nr, nc))
                    seen_seed.add((nr, nc))

        if not self._wfc_propagate(wave, seed, compat):
            return None

        for r, c in diff_cells:
            pref = int(new_p2d[r, c])
            if pref == 0 or pref not in wave[r][c] or len(wave[r][c]) == 1:
                continue
            snapshot = self._copy_wave(wave)
            wave[r][c] = {pref}
            if not self._wfc_propagate(wave, [(r, c)], compat):
                wave = snapshot

        base_repair = wave
        genome_seed = zlib.crc32(np.ascontiguousarray(new_prefs).tobytes())
        for attempt in range(20):
            wave_copy = self._copy_wave(base_repair)
            result = self._run_wfc(wave_copy, np.random.default_rng((genome_seed + attempt * 7_919) % (2**32)), compat)
            if result is None:
                continue
            tiles = self._keep_largest_component(result)
            if self._extract_loop(tiles) is not None:
                return tiles, tiles.copy()

        return None

    @staticmethod
    def _copy_wave(wave):
        return [[set(cell) for cell in row] for row in wave]

    def _decode_cache_store(self, prefs_arr, tiles, wave):
        """Insert a decoded genome into the bounded cache (dict + key list)."""
        key = prefs_arr.tobytes()
        if key not in self._decode_cache:
            self._decode_cache_keys.append(prefs_arr.copy())
            while len(self._decode_cache_keys) > self._DECODE_CACHE_MAX:
                oldest = self._decode_cache_keys.pop(0)
                self._decode_cache.pop(oldest.tobytes(), None)
        self._decode_cache[key] = (tiles, wave)

    def _decode_genome(self, tile_prefs):
        """Decode a partial-map genome to a (GRID_H, GRID_W) tile-index array.

        tile_prefs is a flat int array of length GRID_H*GRID_W.  Value 0 means
        "no preference"; values 1-15 request that face-pair road tile.  Requests
        are stamped as hard pre-collapse observations, propagated immediately,
        skipped on conflict; WFC then fills the rest.  Deterministic per genome,
        cached, with localised repair for mutation-sized neighbours."""
        tile_prefs = np.asarray(tile_prefs, dtype=int)
        key = tile_prefs.tobytes()
        if key in self._decode_cache:
            return self._decode_cache[key][0]

        compat   = self._wfc_compat()
        n_tiles  = _N_WFC_TILES
        prefs_2d = tile_prefs.reshape(GRID_H, GRID_W)

        # ── Localised repair (mutation-sized diffs only) ──────────────────
        if self._decode_cache:
            diffs = np.count_nonzero(np.stack(self._decode_cache_keys) != tile_prefs, axis=1)
            j = int(np.argmin(diffs))
            if int(diffs[j]) <= 15:
                best_old_prefs = self._decode_cache_keys[j]
                best_old_wave  = self._decode_cache[best_old_prefs.tobytes()][1]
                repaired = self._localised_repair(tile_prefs, best_old_prefs, best_old_wave)
                if repaired is not None:
                    tiles, new_wave = repaired
                    self._decode_cache_store(tile_prefs, tiles, new_wave)
                    return tiles

        # ── Full WFC from scratch ──────────────────────────────────────────
        base_wave = [[set(range(n_tiles)) for _ in range(GRID_W)] for _ in range(GRID_H)]
        border = []
        for r in range(GRID_H):
            for c in range(GRID_W):
                if r == 0 or r == GRID_H - 1 or c == 0 or c == GRID_W - 1:
                    base_wave[r][c] = {GRASS}
                    border.append((r, c))
        self._wfc_propagate(base_wave, border, compat)

        stamped = 0
        for r in range(1, GRID_H - 1):
            for c in range(1, GRID_W - 1):
                pref = int(prefs_2d[r, c])
                if pref == 0 or pref not in base_wave[r][c]:
                    continue
                if len(base_wave[r][c]) == 1:
                    stamped += 1
                    continue
                snapshot = self._copy_wave(base_wave)
                base_wave[r][c] = {pref}
                if self._wfc_propagate(base_wave, [(r, c)], compat):
                    stamped += 1
                else:
                    base_wave = snapshot

        # Seed a road if the genome expressed nothing, else grass-heavy WFC
        # tends to leave the grid empty.  Use a straight-through tile at centre.
        if stamped == 0:
            sr, sc = GRID_H // 2, GRID_W // 2
            straight = _EDGES_TO_TILE[frozenset({E, W})]
            if straight in base_wave[sr][sc]:
                base_wave[sr][sc] = {straight}
                self._wfc_propagate(base_wave, [(sr, sc)], compat)

        genome_seed = zlib.crc32(key)
        for attempt in range(100):
            rng  = np.random.default_rng((genome_seed + attempt * 1_000_003) % (2**32))
            wave = self._copy_wave(base_wave)
            result = self._run_wfc(wave, rng, compat)
            if result is None:
                continue
            tiles = self._keep_largest_component(result)
            if self._extract_loop(tiles) is not None:
                self._decode_cache_store(tile_prefs, tiles, tiles.copy())
                return tiles

        # Total failure — deterministic hexagonal ring fallback.
        rng   = np.random.default_rng(int(np.sum(tile_prefs)) % (2**31))
        tiles = self._ring_fallback(rng)
        self._decode_cache_store(tile_prefs, tiles, tiles.copy())
        return tiles

    def init_content(self, rng=None):
        """Return a random sparse partial-map genome (~_ROAD_GENE_RATE active)."""
        if rng is None:
            rng = np.random.default_rng()
        elif isinstance(rng, int):
            rng = np.random.default_rng(rng)
        n = GRID_H * GRID_W
        active = rng.random(n) < _ROAD_GENE_RATE
        tile_prefs = np.where(active, rng.integers(1, _N_WFC_TILES, size=n), 0).astype(int)
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
    _ARC_SAMPLES = 6

    def _loop_to_track_points(self, loop, tiles):
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
        return self._loop_to_track_points(loop, tiles)

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
        """Waypoints already trace the road (arcs + straights); only densify."""
        return self._densify_polyline(
            track_points,
            max_step=float(self._track_width) * 0.35,
        )

    def _count_loop_turns(self, loop, tiles):
        """Designer-style turn count: consecutive same-way corner tiles are one
        turn.  The dense arc sampling makes per-vertex angle counting
        meaningless, so controlability uses this instead."""
        turns = 0
        prev_sign = 0
        n = len(loop)
        for i, (r, c) in enumerate(loop):
            edges = _OPEN_EDGES[int(tiles[r, c])]
            if len(edges) != 2:
                prev_sign = 0
                continue
            d_in  = self._direction_toward(r, c, *loop[(i - 1) % n])
            d_out = self._direction_toward(r, c, *loop[(i + 1) % n])
            if d_in is None or d_out is None or _OPPOSITE[d_in] == d_out:
                prev_sign = 0  # straight tile
                continue
            # Sign of the turn from the two face-midpoint bearings.
            ex, ey = self._face_midpoint(r, c, d_in)
            xx, xy = self._face_midpoint(r, c, d_out)
            cx, cy = self._hex_center(r, c)
            cross = (ex - cx) * (xy - cy) - (ey - cy) * (xx - cx)
            sign = 1 if cross > 0 else (-1 if cross < 0 else 0)
            if sign != 0 and sign != prev_sign:
                turns += 1
            prev_sign = sign
        return turns

    # ── Problem interface ─────────────────────────────────────────────

    def info(self, content, trajectory=None, use_cache=True):
        tiles = self._genome_to_tiles(content)
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
        loop = self._extract_loop(tiles)
        if loop is not None:
            result['num_turns'] = self._count_loop_turns(loop, tiles)
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
