from __future__ import annotations

import numpy as np
from pcg_benchmark.probs.racing.engine import CarPhysicsEngine
from pcg_benchmark.probs.racing.agent import SteeringAgent
from pcg_benchmark.probs.racing.problem import (
    PX_PER_M,
    RacingProblem,
    _rotated_rect,
)
from pcg_benchmark.probs.racing.utils import count_self_intersections
from pcg_benchmark.spaces import ArraySpace, FloatSpace, IntegerSpace, DictionarySpace
from pcg_benchmark.probs.utils import get_range_reward
from PIL import Image, ImageDraw, ImageFont


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


N_BLACKLIST  = 16  # number of blacklist constraint slots in the genome
_N_WFC_TILES = 7   # len(_WFC_TILES): 0=grass … 6=corner_WN
_INACTIVE_TI = _N_WFC_TILES  # sentinel: this slot carries no constraint
_WFC_SEED_MAX = 32768  # exclusive upper bound for the wfc_seed gene


class _TileGridSpace(DictionarySpace):
    """Genome = three parallel blacklist arrays + an explicit WFC seed.

    Each slot i says: at cell (bl_rows[i], bl_cols[i]) WFC must not use
    tile variant bl_tiles[i].  bl_tiles[i] == _INACTIVE_TI means unused.
    wfc_seed is an independent integer mixed into the RNG seed so that two
    chromosomes with identical blacklists but different wfc_seeds produce
    different WFC layouts.  This keeps the population varied even after the
    GA converges on a good constraint pattern.
    contentSwap on all four fields acts as both crossover and mutation."""

    def __init__(self, problem_ref):
        super().__init__({
            "bl_rows":  ArraySpace((N_BLACKLIST,), IntegerSpace(0, GRID_H)),
            "bl_cols":  ArraySpace((N_BLACKLIST,), IntegerSpace(0, GRID_W)),
            # 0-6 = specific tile variant to forbid; _N_WFC_TILES = inactive
            "bl_tiles": ArraySpace((N_BLACKLIST,), IntegerSpace(0, _N_WFC_TILES + 1)),
            # Explicit WFC seed — evolved independently of the blacklist
            "wfc_seed": ArraySpace((1,), IntegerSpace(0, _WFC_SEED_MAX)),
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
        self._decode_cache: dict = {}   # genome bytes → (types, rotations)

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

    @classmethod
    def _wfc_compat(cls):
        """Precompute compatibility table: compat[dir][tile_i] = set of tile indices
        that can appear on the other side of tile_i in that direction."""
        compat = {}
        for d in (N, E, S, W):
            opp = _OPPOSITE[d]
            compat[d] = []
            for t1, r1 in cls._WFC_TILES:
                a_open = d in _OPEN_EDGES[(t1, r1)]
                allowed = frozenset(
                    j for j, (t2, r2) in enumerate(cls._WFC_TILES)
                    if (opp in _OPEN_EDGES[(t2, r2)]) == a_open
                )
                compat[d].append(allowed)
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

    def _keep_largest_component(self, types, rotations):
        """Keep only road tiles forming the largest single closed loop.

        Uses mutual-edge connectivity (tile A's open edge must match tile B's
        opposite edge) rather than simple grid adjacency.  This way two loops
        that merely touch in the grid are treated as separate candidates and
        only the biggest one survives."""
        # Build mutual-match neighbour graph (same logic as _extract_loop)
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

        # Find connected components via mutual connectivity
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

    def _run_wfc(self, wave, rng, compat):
        """Merrell's model-synthesis observe-collapse loop.

        Observe: pick the uncollapsed cell with minimum Shannon entropy
                 (= fewest remaining possibilities), break ties randomly.
        Collapse: draw a tile weighted by _WFC_WEIGHTS.
        Propagate: AC-3 arc-consistency after each collapse.

        Because the seed is placed and propagated before this is called,
        the seed's neighbours already have the most-constrained waves and
        are therefore selected first.  Road naturally spreads outward from
        the seed without any extra steering.

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
            w = self._WFC_WEIGHTS[possible]; w = w / w.sum()
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

    def _decode_genome(self, bl_rows, bl_cols, bl_tiles, wfc_seed=0):
        """Decode a blacklist genome to (types, rotations) via deterministic WFC.

        The same genome (including wfc_seed) always produces the same tile grid.
        wfc_seed is an independent genome gene that shifts the RNG base so two
        chromosomes with identical blacklists but different seeds produce different
        WFC layouts — the primary mechanism for maintaining variety in later GA
        iterations once the blacklist constraints have converged.
        Results are cached keyed on the full genome bytes.
        Returns (types, rotations) — falls back to a rectangle on total failure.
        """
        _wfc_seed = int(np.asarray(wfc_seed).flat[0]) if hasattr(wfc_seed, '__len__') else int(wfc_seed)
        key = (np.asarray(bl_rows).tobytes()
               + np.asarray(bl_cols).tobytes()
               + np.asarray(bl_tiles).tobytes()
               + _wfc_seed.to_bytes(4, 'little'))
        if key in self._decode_cache:
            return self._decode_cache[key]

        # Deterministic seed: polynomial hash over blacklist values, then shifted
        # by wfc_seed so the gene independently steers the WFC outcome.
        vals    = np.array([*bl_rows, *bl_cols, *bl_tiles], dtype=np.int64)
        weights = np.arange(1, len(vals) + 1, dtype=np.int64) * np.int64(2654435761)
        seed    = int((np.sum(vals * weights) + np.int64(_wfc_seed) * np.int64(999983))
                      % np.int64(2**31))

        # Build blacklist dict: (r, c) → set of forbidden WFC tile indices
        blacklist: dict = {}
        for i in range(N_BLACKLIST):
            ti = int(bl_tiles[i])
            if ti >= _INACTIVE_TI:
                continue
            r, c = int(bl_rows[i]) % GRID_H, int(bl_cols[i]) % GRID_W
            blacklist.setdefault((r, c), set()).add(ti)

        compat  = self._wfc_compat()
        n_tiles = len(self._WFC_TILES)
        grass_i = 0

        for attempt in range(100):
            rng = np.random.default_rng((seed + attempt * 1_000_003) % (2**31))

            wave = [[set(range(n_tiles)) for _ in range(GRID_W)]
                    for _ in range(GRID_H)]

            # Apply blacklist constraints before propagation
            for (r, c), forbidden in blacklist.items():
                remaining = wave[r][c] - forbidden
                wave[r][c] = remaining if remaining else {grass_i}

            # Border = grass
            border = []
            for r in range(GRID_H):
                for c in range(GRID_W):
                    if r == 0 or r == GRID_H - 1 or c == 0 or c == GRID_W - 1:
                        wave[r][c] = {grass_i}
                        border.append((r, c))

            # Single straight-tile seed in the inner 80% of the grid.
            # Using a STRAIGHT tile (not a random road tile) avoids biasing
            # the WFC toward corners at the seed, which helps it grow a single
            # connected loop rather than spawning isolated corner clusters.
            sr = int(rng.integers(2, GRID_H - 2))  # rows 2-9 (well inside border)
            sc = int(rng.integers(2, GRID_W - 2))  # cols 2-9
            straight_idxs = [i for i, (t, _) in enumerate(self._WFC_TILES)
                             if t == STRAIGHT]
            seed_tile = straight_idxs[int(rng.integers(len(straight_idxs)))]
            available = wave[sr][sc] - {grass_i}
            if seed_tile in available:
                wave[sr][sc] = {seed_tile}
                border.append((sr, sc))

            if not self._wfc_propagate(wave, border, compat):
                continue

            result = self._run_wfc(wave, rng, compat)
            if result is None:
                continue

            types, rotations = self._keep_largest_component(*result)
            if self._extract_loop(types, rotations) is not None:
                self._decode_cache[key] = (types, rotations)
                return types, rotations

        # Total failure — deterministic rectangle as last resort
        rng = np.random.default_rng(seed % (2**31))
        fb  = self._rectangular_fallback(rng)
        t   = np.asarray(fb["types"],     dtype=int)
        r_  = np.asarray(fb["rotations"], dtype=int)
        self._decode_cache[key] = (t, r_)
        return t, r_

    def init_content(self, rng=None):
        """Return a random genome that decodes to a valid single-loop track.
        Genome fields: bl_rows, bl_cols, bl_tiles (blacklist constraints) plus
        wfc_seed (steers the WFC RNG independently of the constraint pattern).
        contentSwap on all four fields acts as crossover and mutation."""
        if rng is None:
            rng = np.random.default_rng()
        elif isinstance(rng, int):
            rng = np.random.default_rng(rng)

        for _ in range(200):
            bl_rows  = rng.integers(0, GRID_H,           size=N_BLACKLIST).astype(int)
            bl_cols  = rng.integers(0, GRID_W,           size=N_BLACKLIST).astype(int)
            bl_tiles = rng.integers(0, _N_WFC_TILES + 1, size=N_BLACKLIST).astype(int)
            wfc_seed = rng.integers(0, _WFC_SEED_MAX,    size=1).astype(int)
            types, rotations = self._decode_genome(bl_rows, bl_cols, bl_tiles, wfc_seed)
            if self._extract_loop(types, rotations) is not None:
                return {"bl_rows": bl_rows, "bl_cols": bl_cols,
                        "bl_tiles": bl_tiles, "wfc_seed": wfc_seed}

        # Absolute fallback: empty blacklist, fresh random seed
        bl_rows  = rng.integers(0, GRID_H, size=N_BLACKLIST).astype(int)
        bl_cols  = rng.integers(0, GRID_W, size=N_BLACKLIST).astype(int)
        bl_tiles = np.full(N_BLACKLIST, _INACTIVE_TI, dtype=int)
        wfc_seed = rng.integers(0, _WFC_SEED_MAX, size=1).astype(int)
        return {"bl_rows": bl_rows, "bl_cols": bl_cols,
                "bl_tiles": bl_tiles, "wfc_seed": wfc_seed}

    def _rectangular_fallback(self, rng):
        min_dim = 3
        r0 = int(rng.integers(1, GRID_H - min_dim - 1))
        c0 = int(rng.integers(1, GRID_W - min_dim - 1))
        r1 = int(rng.integers(r0 + min_dim, min(r0 + min_dim + 5, GRID_H - 1) + 1))
        c1 = int(rng.integers(c0 + min_dim, min(c0 + min_dim + 5, GRID_W - 1) + 1))
        loop = []
        for c in range(c0, c1):     loop.append((r0, c))
        for r in range(r0, r1):     loop.append((r,  c1))
        for c in range(c1, c0, -1): loop.append((r1, c))
        for r in range(r1, r0, -1): loop.append((r,  c0))

        types     = np.zeros((GRID_H, GRID_W), dtype=int)
        rotations = np.zeros((GRID_H, GRID_W), dtype=int)
        n = len(loop)
        for i, (r, c) in enumerate(loop):
            pr, pc = loop[(i - 1) % n]
            nr, nc = loop[(i + 1) % n]
            from_dir = next(d for d, (dr, dc) in _DIR_DELTA.items() if r + dr == pr and c + dc == pc)
            to_dir   = next(d for d, (dr, dc) in _DIR_DELTA.items() if r + dr == nr and c + dc == nc)
            t, rot = _EDGES_TO_TILE.get(frozenset({from_dir, to_dir}), (GRASS, 0))
            types[r, c], rotations[r, c] = t, rot
        return {"types": types, "rotations": rotations}

    # ── Decoding: tile grid -> track waypoints ────────────────────────

    def _extract_loop(self, types, rotations):
        """
        Traverse mutually-connected road tiles to find a closed loop.
        Returns ordered list of (r, c) or None if no valid cycle exists.
        """
        neighbors = {}
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

        # A simple cycle requires every node to have degree exactly 2
        cycle_cells = {cell for cell, nbrs in neighbors.items() if len(nbrs) == 2}
        if not cycle_cells:
            return None

        start = next(iter(cycle_cells))
        loop, prev, cur = [start], None, start
        while True:
            nxt = next(
                (nb for nb in neighbors.get(cur, []) if nb != prev and nb in cycle_cells),
                None,
            )
            if nxt is None or nxt == start:
                break
            loop.append(nxt)
            prev, cur = cur, nxt

        if len(loop) < 4 or start not in neighbors.get(loop[-1], []):
            return None
        return loop

    def _grid_to_track_points(self, types, rotations):
        """Convert tile grid to ordered pixel waypoints.
        One point per tile: the midpoint of the shared edge with the next tile.
        This gives the exact crossing points where the path transitions between tiles."""
        loop = self._extract_loop(types, rotations)
        if loop is None:
            return None
        n = len(loop)
        points = []
        for i, (r, c) in enumerate(loop):
            nr, nc = loop[(i + 1) % n]
            px = (c + nc + 1) * self._width  / (2 * GRID_W)
            py = (r + nr + 1) * self._height / (2 * GRID_H)
            points.append((px, py))
        return points


    def _best_effort_path(self, types, rotations):
        """For rendering: find the largest connected chain of road tiles even if
        it is not a closed loop. Returns ordered pixel waypoints or None."""
        # Build mutual-match neighbor graph (same as _extract_loop)
        neighbors = {}
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
                if nbrs:
                    neighbors[(r, c)] = nbrs

        if not neighbors:
            return None

        # Find the largest connected component via DFS
        visited, best = set(), []
        for seed in neighbors:
            if seed in visited:
                continue
            comp, stack = [], [seed]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                comp.append(node)
                for nb in neighbors.get(node, []):
                    if nb not in visited:
                        stack.append(nb)
            if len(comp) > len(best):
                best = comp

        if len(best) < 2:
            return None

        # Traverse in order: start from an endpoint (degree 1) if one exists
        best_set = set(best)
        endpoints = [c for c in best if len([nb for nb in neighbors.get(c, []) if nb in best_set]) == 1]
        start = endpoints[0] if endpoints else best[0]

        path, prev, cur = [start], None, start
        while True:
            nxt = next((nb for nb in neighbors.get(cur, []) if nb != prev and nb in best_set), None)
            if nxt is None or nxt == path[0]:
                break
            path.append(nxt)
            prev, cur = cur, nxt

        if len(path) < 2:
            return None

        # Convert to pixel edge-midpoints
        n = len(path)
        points = []
        for i, (r, c) in enumerate(path):
            nr, nc = path[(i + 1) % n]
            px = (c + nc + 1) * self._width  / (2 * GRID_W)
            py = (r + nr + 1) * self._height / (2 * GRID_H)
            points.append((px, py))
        return points

    # ── Content extraction bridge ─────────────────────────────────────
    # Overriding this makes all parent methods (simulate, render, etc.)
    # work transparently with tile-grid content dicts.

    def _genome_to_tiles(self, content):
        """Decode a blacklist genome dict to (types, rotations)."""
        return self._decode_genome(
            np.asarray(content["bl_rows"],  dtype=int),
            np.asarray(content["bl_cols"],  dtype=int),
            np.asarray(content["bl_tiles"], dtype=int),
            wfc_seed=content.get("wfc_seed", [0]),
        )

    def _extract_content(self, content):
        if isinstance(content, dict) and "bl_rows" in content:
            types, rotations = self._genome_to_tiles(content)
            pts = self._grid_to_track_points(types, rotations) \
               or self._best_effort_path(types, rotations)
            return np.array(pts) if pts is not None else np.array(self._default_track_points)
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

    def quality(self, info):
        return super().quality(info)

    # ── Tile rendering ────────────────────────────────────────────────

    def render(self, content=None, **kwargs):
        if isinstance(content, dict) and "bl_rows" in content:
            self._tile_render_content = content
        else:
            self._tile_render_content = None
        return super().render(content, **kwargs)

    def _render_track_bg(self, img_w, img_h, left_edge_f, right_edge_f, scaled_curve,
                         grass_color, edge_color, road_color, centerline_color):
        content = getattr(self, '_tile_render_content', None)
        if content is None:
            return super()._render_track_bg(img_w, img_h, left_edge_f, right_edge_f,
                                             scaled_curve, grass_color, edge_color,
                                             road_color, centerline_color)
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
