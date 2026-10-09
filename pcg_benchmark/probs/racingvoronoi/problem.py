from __future__ import annotations

from pcg_benchmark.probs.racing.problem import RacingProblem
from pcg_benchmark.probs.racingvoronoi.utils import (
    build_power_cell_graph,
    fillet_corners,
    find_boundary_cycle,
    generate_sites,
)
from pcg_benchmark.spaces import ArraySpace, FloatSpace, DictionarySpace
import numpy as np


class RacingVoronoiProblem(RacingProblem):
    """Cell selection on an evolvable power diagram.

    The sites are fixed (voronoi_seed).  Each genome gives every cell a
    selection score and a weight: the weights set the cells' shapes through
    a power diagram (Aurenhammer 1987), the scores grow a connected cluster
    of num_selected_cells cells through it, and the track is the cluster's
    rim with its corners rounded.  With every weight at 0 the diagram is the
    Voronoi diagram, which is the representation racingvoronoiold keeps.

    Why weights: a uniform scatter gives similar cells, so every straight
    comes out at the edge scale; weights let the search grow, shrink or
    remove cells, which gives the rim long edges (ceiling: _WEIGHT_MAX_M2).
    Why not another seed or fewer cells: that moves the edge scale for every
    genome (at 26 cells only 11 are selectable, so nearly all draw one loop).
    Why not the seed as a gene: one mutation would redraw every cell.  Why
    not site positions as genes: on the same 40 genomes they meet the rules
    on 10, the weights on 31, and one weight moves only its cell's edges.
    """

    _render_desc = 'Rendering voronoi frames'

    # Decoded points already trace a valid loop in order; the base class's
    # 2-opt untangle must not reorder them (it would break the loop).
    _untangle_control_points = False

    # Side of the square the selectable sites are drawn from, as a fraction of
    # the build box side (user decision).
    _SITE_BOX_FRAC = 0.9

    def __init__(self, **kwargs):
        kwargs.setdefault('num_points', 15)
        # The cell counts trade lap length against how many distinct tracks
        # can be expressed; what binds is ELIGIBLE cells (polygon inside the
        # build box).  40 random genomes, 10 selected, all weights 0, 16 m road:
        #
        #   cells  eligible  distinct  median lap (p10-p90)  tightest corner p10 / median  turns
        #      50        27     40/40  3589 m (2527-4344)    8.0 / 10.6 m                  16
        #      70        43     40/40  3271 m (2401-3987)    6.0 /  9.2 m                  17
        #      90        59     40/40  2994 m (2239-3684)    5.2 /  8.6 m                  20
        #
        # 50 cells.  More give more reach but smaller cells: more corners, a
        # shorter lap, and tighter corners that fold the road.  (Sites over
        # the whole map with the 1091 m box left 15 eligible.)
        num_cells = int(kwargs.pop('num_cells', 50))
        num_selected = int(kwargs.pop('num_selected_cells', 10))
        voronoi_seed = int(kwargs.pop('voronoi_seed', 0))
        # Lloyd relaxation (moving each site to its cell's centre) evens the
        # cells; 0, off, is the raw scatter "a Voronoi grid" means.  Over 8
        # genomes at 0 / 1 / 2 / 4 iterations it changes evenness (spacing
        # spread 0.57 / 0.36 / 0.32 / 0.18) and lap length (3280 / 3512 / 3722
        # / 3840 m), not distinctness (8 of 8), finishing or off-road.  Off:
        # the shorter lap is a length-rule matter, not a reason for a step the
        # representation should not have.
        lloyd_iterations = int(kwargs.pop('lloyd_iterations', 0))
        # Site placement, a user decision: selectable sites uniform over a
        # centred square of _SITE_BOX_FRAC of the build box (1170 m); guard
        # points uniform over a 3500 m square minus a 2500 m one, so edge
        # cells close instead of running to infinity (never selectable).
        num_guard_points = int(kwargs.pop('num_guard_points', 50))
        guard_outer_m = float(kwargs.pop('guard_outer_m', 3500.0))
        guard_inner_m = float(kwargs.pop('guard_inner_m', 2500.0))
        weight_max_m2 = float(kwargs.pop('weight_max_m2', self._WEIGHT_MAX_M2))

        super().__init__(**kwargs)

        self._num_cells = num_cells
        self._num_selected_cells = num_selected
        self._voronoi_seed = voronoi_seed
        self._lloyd_iterations = lloyd_iterations
        self._num_guard_points = num_guard_points
        self._guard_outer_m = guard_outer_m
        self._guard_inner_m = guard_inner_m
        self._weight_max_m2 = weight_max_m2

        # Two genes per cell: a selection score (_decode_selected_cells) and a
        # power weight (_build_cell_graph; an edge moves at most
        # _WEIGHT_MAX_M2 / (2 d) for sites d apart).
        self._content_space = DictionarySpace({
            "cell_scores": ArraySpace((self._num_cells,), FloatSpace(0.0, 1.0)),
            "cell_weights": ArraySpace((self._num_cells,), FloatSpace(0.0, 1.0)),
        })
        # The control space is inherited: the same targets for every representation.

        self._cell_sites = None
        self._boundary_cells = None
        self._ineligible_cells = None
        self._voronoi_vertices = None
        self._voronoi_all_edges = None
        self._voronoi_all_pairs = None
        self._cell_adjacency = None

    # ------------------------------------------------------------------
    # Fixed Voronoi grid
    # ------------------------------------------------------------------

    # Weight ceiling (m^2).  An edge moves by (w_i - w_j) / (2 d) for sites d
    # apart (median nearest 87 m), so a ceiling well above d^2 lets a heavy
    # cell remove its neighbours.  Rule (user decision, 2026-10-08): the
    # setting under which the most random genomes meet every regulation after
    # both gates.  40 genomes per seed, geometry only, 12 m road, rules met on
    # seed 21 / 22 (calibration/voronoi_rule.py):
    #
    #   sites  w_max     2250   10000   20000   40000   60000  120000
    #    50             21/23   24/25   26/26   31/30   24/25   14/19
    #    40             13/28   24/28   24/33   31/30   25/26   19/13
    #    30             23/31   24/25   18/23   12/14   10/8     4/9
    #
    # 40000 with 50 sites: 61 of 80, against 44 at 2250, the largest ceiling
    # at which no cell vanishes; beyond it heavy cells swallow the cluster
    # and laps fall under 3.5 km.  No setting passes typicality: the rim keeps
    # 16-19 FIA corners in 3.5-4 km.
    _WEIGHT_MAX_M2 = 40000.0
    # Diagrams kept by weight vector (each genome decodes at least twice).
    _GRAPH_CACHE_MAX = 512

    def _sites(self):
        """The fixed sites, selectable then guards, drawn once."""
        if getattr(self, "_all_sites", None) is None:
            w, h = float(self._width), float(self._height)
            cx, cy = 0.5 * w, 0.5 * h
            ho, hi = 0.5 * self._guard_outer_m, 0.5 * self._guard_inner_m
            bx0, by0, bx1, by1 = self._build_box()
            margin = 0.5 * (1.0 - self._SITE_BOX_FRAC) * (bx1 - bx0)
            self._all_sites = generate_sites(
                self._num_cells, (bx0 + margin, by0 + margin, bx1 - margin, by1 - margin),
                self._voronoi_seed, lloyd_iterations=self._lloyd_iterations,
                num_guard_points=self._num_guard_points,
                guard_outer=(cx - ho, cy - ho, cx + ho, cy + ho),
                guard_inner=(cx - hi, cy - hi, cx + hi, cy + hi))
            self._graph_cache = {}
        return self._all_sites

    def _cell_weights(self, content):
        """Power weight of every site (m^2): weight genes times _weight_max_m2,
        0 for guards; all zeros (plain Voronoi) for content without them."""
        sites = self._sites()
        w = np.zeros(len(sites))
        genes = content.get("cell_weights") if isinstance(content, dict) else None
        if genes is not None:
            genes = np.clip(np.asarray(genes, dtype=float).ravel(), 0.0, 1.0)
            if genes.size != self._num_cells:
                raise ValueError(f"cell_weights must have length {self._num_cells}, got {genes.size}")
            w[:self._num_cells] = genes * self._weight_max_m2
        return w

    def _build_cell_graph(self, weights=None):
        """Build (or fetch) the power diagram for these site weights and make
        it the current one.  None means all zeros."""
        sites = self._sites()
        if weights is None:
            weights = np.zeros(len(sites))
        key = np.asarray(weights, dtype=float).tobytes()
        entry = self._graph_cache.get(key)
        if entry is None:
            graph = build_power_cell_graph(sites, self._num_cells, weights,
                                           clip_box=(0.0, 0.0, float(self._width), float(self._height)))
            # Selectable iff its polygon lies inside the build box (the track
            # runs on its vertices): not empty, not clipped, no vertex outside.
            ineligible = set(graph['boundary_cells']) | set(graph['empty_cells'])
            bx0, by0, bx1, by1 = self._build_box()
            verts = graph['voronoi_vertices']
            vertex_ok = ((verts[:, 0] >= bx0) & (verts[:, 0] <= bx1) &
                         (verts[:, 1] >= by0) & (verts[:, 1] <= by1))
            # Guards (indices >= num_cells) are never selectable.
            for (va, vb), (p1, p2) in zip(graph['all_edges_full'], graph['all_edge_pairs_full']):
                if not (vertex_ok[int(va)] and vertex_ok[int(vb)]):
                    ineligible.update(int(p) for p in (p1, p2) if int(p) < self._num_cells)
            entry = (graph, ineligible)
            if len(self._graph_cache) >= self._GRAPH_CACHE_MAX:
                self._graph_cache.pop(next(iter(self._graph_cache)))
            self._graph_cache[key] = entry
        graph, ineligible = entry
        self._cell_sites = graph['cell_sites']
        self._voronoi_vertices = graph['voronoi_vertices']
        self._voronoi_all_edges = graph['all_edges_full']
        self._voronoi_all_pairs = graph['all_edge_pairs_full']
        self._cell_adjacency = graph['cell_neighbours']
        self._boundary_cells = graph['boundary_cells']
        self._ineligible_cells = ineligible

    # ------------------------------------------------------------------
    # Cell selection
    # ------------------------------------------------------------------

    @staticmethod
    def _highest_scoring_cell(cells, scores):
        """The highest-scoring cell, ties to the lowest index (a pure function
        of the genome, not of the order of `cells`)."""
        return min(cells, key=lambda i: (-float(scores[i]), i), default=None)

    def _decode_selected_cells(self, cell_scores: np.ndarray) -> list[int]:
        """Selected cells of the current diagram (_build_cell_graph must
        have been called for this genome's weights)."""
        scores = np.asarray(cell_scores, dtype=float).ravel()
        if scores.size != self._num_cells:
            raise ValueError(f"cell_scores must have length {self._num_cells}, got {scores.size}")

        ineligible = self._ineligible_cells
        selectable = [i for i in range(self._num_cells) if i not in ineligible]

        # Grow a cluster from the best eligible cell, adding the best neighbour.
        k = min(int(self._num_selected_cells), len(selectable))
        start = self._highest_scoring_cell(selectable, scores)
        if start is None:
            # No eligible cell (only if box, count or spread change; at the
            # defaults a median 21 of 50 are eligible): let _extract_content
            # fall back.
            return []
        selected: set[int] = {start}
        frontier: set[int] = {n for n in self._cell_adjacency[start] if n not in ineligible}

        while len(selected) < k:
            candidates = [c for c in frontier if c not in selected]
            if not candidates:
                # Walled in: fill with the best leftovers (the boundary-cycle
                # step then rejects the disconnected cluster).
                leftovers = sorted((i for i in selectable if i not in selected),
                                   key=lambda i: (-float(scores[i]), i))
                selected.update(leftovers[: k - len(selected)])
                break
            nxt = self._highest_scoring_cell(candidates, scores)
            selected.add(nxt)
            frontier.update(n for n in self._cell_adjacency[nxt] if n not in ineligible)

        return sorted(selected)

    # ------------------------------------------------------------------
    # Content extraction
    # ------------------------------------------------------------------

    def _extract_content(self, content):
        if content is None or (isinstance(content, dict) and 'track_points' in content):
            return super()._extract_content(content)

        if isinstance(content, dict) and 'cell_scores' in content:
            self._build_cell_graph(self._cell_weights(content))
            selected = self._decode_selected_cells(content['cell_scores'])

            # A boundary edge has exactly one of its two cells inside the
            # cluster: inside on one side, outside on the other.
            pairs = self._voronoi_all_pairs
            edges = self._voronoi_all_edges
            if len(pairs) > 0:
                sel = np.asarray(selected, dtype=int)
                first_cell_in = np.isin(pairs[:, 0], sel)
                second_cell_in = np.isin(pairs[:, 1], sel)
                mask = first_cell_in != second_cell_in
                boundary_edges = edges[mask]
            else:
                boundary_edges = np.zeros((0, 2), dtype=int)

            cycle = find_boundary_cycle(boundary_edges, self._voronoi_vertices)
            if cycle is None:
                return super()._extract_content(None)

            # The cycle is the track: the rim of the selected cluster, taken
            # as-is.  Short edges and tight vertex clusters are left alone,
            # because the corner arcs in _make_curve adapt their radius to the
            # adjacent edge lengths and absorb them.
            pts = self._voronoi_vertices[np.array(cycle, dtype=int)]
            if len(pts) < 3:
                return super()._extract_content(None)
            # One order for a given rim: anticlockwise, from its lowest
            # vertex (smallest x, then y).  find_boundary_cycle starts its walk
            # at an arbitrary vertex, and _drop_tight_corners and the
            # resampling both depend on where the rim starts: on 40 random
            # genomes the unordered rim, reversed and rotated, repairs to a
            # different track on 4.
            pts = np.asarray(pts, dtype=float)
            x, y = pts[:, 0], pts[:, 1]
            if np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) < 0:
                pts = pts[::-1]
            first = min(range(len(pts)), key=lambda i: (pts[i, 0], pts[i, 1]))
            return np.roll(pts, -first, axis=0)

        return super()._extract_content(content)

    # ------------------------------------------------------------------
    # Simulation interface
    # ------------------------------------------------------------------

    # Corner rounding: every polygon corner becomes a circular arc tangent to
    # both edges, mirroring how the tile representation drives quarter arcs.
    # The cap only binds on shallow corners, since the radius is already
    # limited to half the shorter adjacent edge.  At 110 m it leaves the arcs
    # as round as the cell geometry allows, which is what keeps the car on the
    # road: capping at 40 forces tighter arcs than the corner needs, raising
    # the off-road fraction from 0.112 to 0.166.
    _FILLET_MAX_RADIUS = 110.0

    def _drop_tight_corners(self, poly):
        """Remove rim vertices while the filleted rim fails the shared
        self-overlap test.

        A cell rim can turn back on itself at a vertex, and the fillet radius
        there is half the shorter adjacent edge over tan(turn / 2), which goes
        to zero as the turn approaches 180 degrees.  The inner road edge then
        folds over itself and the shared gate scores the track 0.  So while
        the curve fails that test, the rim vertex nearest the failure
        (_overlap_spot) is removed, merging its two edges into one.  This is
        _drop_overlap_points, the repair racing-v0 applies to its control
        points, on this decoder's geometry.  The test counts a fillet tighter
        than half the track width as a fold (utils.fold_count): on the sample
        below the failing rims' smallest local radius is 2.6-5.7 m, every
        passing rim's 6.3 m or more.

        40 random genomes (seed 7, 12 m road): 15 fail before the repair, 0
        after; only those 15 change, losing a median of 2 vertices (at most
        5) and up to 418 m of lap; the closest approach of two non-adjacent
        parts of a repaired road is 15.6 m against the 12 m road, so no fold
        is traded for a collision.  The curve costs 8.8 ms with the repair,
        0.9 ms without (each pass runs the area check).

        No 2-opt afterwards (the rim is in loop order and must follow the
        selected cells).  Below six vertices a rim keeps its fold and scores 0."""
        return self._repair_overlaps(poly, self._fillet_curve)

    def _fillet_curve(self, poly):
        """Round each corner of a rim polygon with a tangent arc and re-space
        to the curve step.  The start line is placed afterwards, by
        _place_start, as for every representation."""
        step = self._curve_step()
        rounded = np.asarray(fillet_corners(poly, max_radius=self._FILLET_MAX_RADIUS,
                                            sample_step=step), dtype=float)
        return self._resample_uniform(rounded, step=step)

    def _build_curve(self, track_points):
        """Voronoi tracks are polygons: round each corner with a tangent arc,
        then densify the remaining straights (a spline would bow the straights
        and round every corner twice)."""
        return self._fillet_curve(self._drop_tight_corners(track_points))

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def info(self, content, trajectory=None, use_cache=True):
        """The base info, with a second cache keyed by the genome (scores and
        weights): the base cache is keyed by the decoded track points, so
        without it every call would decode the genome again first."""
        scores_key = None
        if use_cache and trajectory is None and isinstance(content, dict) and 'cell_scores' in content:
            scores_key = (tuple(np.asarray(content['cell_scores'], dtype=float).ravel().tolist()),
                          tuple(self._cell_weights(content).tolist()),
                          self._driver)
            cached = self._info_cache.get(scores_key)
            if cached is not None:
                return cached

        result = super().info(content, trajectory=trajectory, use_cache=use_cache)
        if use_cache and scores_key is not None:
            self._info_cache[scores_key] = result
        return result

    # ------------------------------------------------------------------
    # Quality
    # ------------------------------------------------------------------

    # No quality parameter is overridden; the self-overlap gate runs on the
    # dense curve as everywhere.  Why not exempt Voronoi from the area check:
    # its fillets fold the inner edge (15 of 40 unrepaired rims, seed 7);
    # _drop_tight_corners removes the fold, so the gate is met, not waived.
    # Why not check the sparse polygon: on the same genomes neither finds a
    # crossing, so it would only be a second definition.
    # The straights come from the diagram's edge scale (median longest 210-260
    # m at _WEIGHT_MAX_M2), a property of the representation, scored as any.

    # Controlability and diversity are inherited (a Jaccard measure on cell
    # sets would be genotype-based and meaningful only here).

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _draw_bg_overlay(self, bg_draw, scale):
        """Draw the diagram of the genome last decoded (all edges, including
        the out-of-bounds ones) in gray between the road surface and the
        centerline, so the cell structure is visible behind the track.  With
        nothing decoded yet it draws the unweighted diagram."""
        if self._voronoi_vertices is None:
            self._build_cell_graph()
        verts = self._voronoi_vertices * scale
        for u, v in self._voronoi_all_edges:
            pu, pv = verts[int(u)], verts[int(v)]
            bg_draw.line(
                [(int(round(pu[0])), int(round(pu[1]))), (int(round(pv[0])), int(round(pv[1])))],
                fill=(70, 70, 70), width=1,
            )
