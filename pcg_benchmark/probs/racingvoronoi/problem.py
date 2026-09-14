from __future__ import annotations

from pcg_benchmark.probs.racing.problem import RacingProblem
from pcg_benchmark.probs.racingvoronoi.utils import (
    build_voronoi_cell_graph,
    fillet_corners,
    find_boundary_cycle,
)
from pcg_benchmark.spaces import ArraySpace, FloatSpace, DictionarySpace
import numpy as np


class RacingVoronoiProblem(RacingProblem):

    _render_desc = 'Rendering voronoi frames'

    # Decoded points already trace a valid loop in order; the base class's
    # 2-opt untangle must not reorder them (it would break the loop).
    _untangle_control_points = False

    def __init__(self, **kwargs):
        kwargs.setdefault('num_points', 15)
        # max_steps is NOT overridden: the shared 7000-step budget applies.
        # Now that every representation generates a lap of the same length, a
        # per-representation cap would score the budget rather than the track.
        # A 4400 m lap at the ~16 m/s the agent averages needs ~2750 steps, so
        # 7000 leaves headroom for a slow lap without truncating a good one.

        # The two counts trade lap length against how many distinct tracks the
        # representation can express, and the binding constraint is the number
        # of ELIGIBLE cells, not num_cells: a cell whose polygon leaves the
        # build box cannot be selected, and the diagram always loses its outer
        # ring to unbounded regions.
        #
        # Swept over num_cells 36/49/64 x lloyd 0/2/4 x selection 10/14/18,
        # 8 genomes each, measuring the eligible pool, the spread of
        # nearest-neighbour site spacing (std over mean), median lap length and
        # how many of the 8 genomes chose a distinct cluster:
        #
        #   cells  lloyd  k   pool  spread  medlen  distinct
        #      36      0  10    16    0.57    3278       8/8
        #      36      4  10    15    0.18    3839       8/8
        #      49      0  14    25    0.49    3941       8/8
        #      49      4  14    26    0.18    4494       8/8
        #      64      4  18    35    0.20    4653       8/8
        #
        # 49 cells with 14 selected is the setting taken.  36 cells leave only
        # 16 eligible, so most of the genome is inert, and the laps they
        # produce sit below the shared 3900-5900 m band whatever the relaxation.
        # 64 cells raise the pool further but cut cell size enough that the
        # selected rim gains corners and its tightest one tightens toward
        # undrivable.  Every setting keeps full cluster variety, so that is not
        # what decides it.
        num_cells = int(kwargs.pop('num_cells', 49))
        num_selected = int(kwargs.pop('num_selected_cells', 14))
        voronoi_seed = int(kwargs.pop('voronoi_seed', 0))
        # Lloyd relaxation: repeatedly move each site to the centre of its own
        # cell, which evens the cells out.  0 disables it and uses the raw
        # random scatter.
        #
        # 4 iterations, from the same sweep.  Raw scatter clumps: the spread of
        # nearest-neighbour spacing is 0.57 of its mean at 0 iterations, 0.32
        # at 2 and 0.18 at 4, past which it stops improving.  Rounder cells
        # also give the selected cluster a longer rim, which is what carries
        # the lap from 3941 m to 4494 m and into the shared length band.
        lloyd_iterations = int(kwargs.pop('lloyd_iterations', 4))

        super().__init__(**kwargs)

        self._num_cells = num_cells
        self._num_selected_cells = num_selected
        self._voronoi_seed = voronoi_seed
        self._lloyd_iterations = lloyd_iterations

        self._content_space = DictionarySpace({
            "cell_scores": ArraySpace((self._num_cells,), FloatSpace(0.0, 1.0)),
        })
        # Control space is the inherited one: every representation is asked for
        # the same length band, so controlability compares like with like.

        self._cell_sites = None
        self._boundary_cells = None
        self._ineligible_cells = None
        self._voronoi_vertices = None
        self._voronoi_all_edges = None
        self._voronoi_all_pairs = None
        self._cell_adjacency = None
        self._last_selected_cells = None

    # _resample_uniform is inherited from RacingProblem (shared with tile).

    # ------------------------------------------------------------------
    # Fixed Voronoi grid
    # ------------------------------------------------------------------

    def _build_cell_graph(self):
        if self._cell_sites is not None:
            return
        graph = build_voronoi_cell_graph(self._num_cells, self._build_box(),
                                         self._voronoi_seed,
                                         lloyd_iterations=self._lloyd_iterations)
        self._cell_sites             = graph['cell_sites']  # also the built flag
        self._voronoi_vertices       = graph['voronoi_vertices']
        self._voronoi_all_edges = graph['all_edges_full']
        self._voronoi_all_pairs = graph['all_edge_pairs_full']
        self._cell_adjacency         = graph['cell_neighbours']
        self._boundary_cells         = graph['boundary_cells']

        # A cell is selectable iff its polygon stays fully inside the shared
        # build box, since the track runs along the selected cells' Voronoi
        # vertices.  Two ways to fail: a semi-infinite region clipped at the
        # border, or a finite region owning a vertex outside the box
        # (near-collinear sites push Voronoi vertices arbitrarily far out).  No
        # other exclusions, so what is inside the box is selectable.
        self._ineligible_cells = set(self._boundary_cells)
        bx0, by0, bx1, by1 = self._build_box()
        verts = self._voronoi_vertices
        vertex_ok = (
            (verts[:, 0] >= bx0) & (verts[:, 0] <= bx1) &
            (verts[:, 1] >= by0) & (verts[:, 1] <= by1)
        )
        for (va, vb), (p1, p2) in zip(self._voronoi_all_edges, self._voronoi_all_pairs):
            if not (vertex_ok[int(va)] and vertex_ok[int(vb)]):
                self._ineligible_cells.add(int(p1))
                self._ineligible_cells.add(int(p2))

    # ------------------------------------------------------------------
    # Cell selection
    # ------------------------------------------------------------------

    @staticmethod
    def _highest_scoring_cell(cells, scores):
        """Return the cell with the highest score; ties go to the lowest index.

        Sorting on (-score, index) puts the best score first and the lowest
        index first within a tie, so the pick is a pure function of the genome
        and not of the order the caller happened to build `cells` in."""
        return min(cells, key=lambda i: (-float(scores[i]), i), default=None)

    def _decode_selected_cells(self, cell_scores: np.ndarray) -> list[int]:
        self._build_cell_graph()
        scores = np.asarray(cell_scores, dtype=float).ravel()
        if scores.size != self._num_cells:
            raise ValueError(f"cell_scores must have length {self._num_cells}, got {scores.size}")

        ineligible = self._ineligible_cells
        selectable = [i for i in range(self._num_cells) if i not in ineligible]

        # Grow a connected cluster: start at the highest-scoring eligible
        # cell, then repeatedly add the highest-scoring eligible neighbor.
        k = min(int(self._num_selected_cells), len(selectable))
        start = self._highest_scoring_cell(selectable, scores)
        selected: set[int] = {start}
        frontier: set[int] = {n for n in self._cell_adjacency[start] if n not in ineligible}

        while len(selected) < k:
            candidates = [c for c in frontier if c not in selected]
            if not candidates:
                # The cluster is walled in; fill the remaining slots with the
                # best leftover cells by score (connectivity can no longer
                # be satisfied, so the boundary-cycle step will reject this).
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
            self._build_cell_graph()
            selected = self._decode_selected_cells(content['cell_scores'])
            self._last_selected_cells = selected

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
            return np.asarray(pts, dtype=float)

        self._last_selected_cells = None
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

    def _make_curve(self, track_points):
        """Voronoi tracks are polygons: round each corner with a tangent arc,
        then densify the remaining straights (a spline would bow the straights
        and round every corner twice)."""
        step = self._curve_step()
        rounded = fillet_corners(track_points, max_radius=self._FILLET_MAX_RADIUS, sample_step=step)
        return self._resample_uniform(rounded, step=step)

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def info(self, content, trajectory=None, use_cache=True):
        """Same as the base info, plus:
        - an extra cache keyed by the raw cell_scores genome (decoding a
          genome to its track is itself expensive, so caching by track
          points alone would still re-decode every call), and
        - the list of selected cells, which diversity() compares."""
        scores_key = None
        if use_cache and trajectory is None and isinstance(content, dict) and 'cell_scores' in content:
            scores_key = (tuple(np.asarray(content['cell_scores'], dtype=float).ravel().tolist()),
                          self._driver)
            cached = self._info_cache.get(scores_key)
            if cached is not None:
                return cached

        result = super().info(content, trajectory=trajectory, use_cache=use_cache)
        # _extract_content (called inside super().info) records which cells
        # the genome selected; keep the value already stored on cache hits.
        if 'selected_cells' not in result:
            result['selected_cells'] = self._last_selected_cells
        if use_cache and scores_key is not None:
            self._info_cache[scores_key] = result
        return result

    # ------------------------------------------------------------------
    # Quality
    # ------------------------------------------------------------------

    # Voronoi tracks are polygons, so the self-intersection check runs on the
    # sparse polygon vertices instead of the dense curve, and the area-overlap
    # check is off (adjacent cell edges would trip it constantly).  Everything
    # else is shared with RacingProblem._quality_terms.
    _QUALITY_PARAMS = {
        **RacingProblem._QUALITY_PARAMS,
        "angles_on_curve":    False,
        "geom_area_check":    False,
        # start_straight is deliberately NOT overridden: one quality function
        # for all.  Voronoi is the tightest case, with straights pinned to the
        # diagram edge scale, but it still reaches the shared target often
        # enough to be a real discriminator rather than a lock-out.
    }

    # ------------------------------------------------------------------
    # Controlability and diversity are both inherited from RacingProblem, so
    # every representation is measured by the same function.  A Jaccard measure
    # on the selected cell sets would be genotype-based and meaningful only
    # here, which is exactly what a shared benchmark measure must not be.

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _draw_bg_overlay(self, bg_draw, scale):
        """Draw the full Voronoi diagram (all finite edges, including the
        out-of-bounds ones) in gray between the road surface and the
        centerline, so the cell structure is visible behind the track."""
        self._build_cell_graph()
        verts = self._voronoi_vertices * scale
        for u, v in self._voronoi_all_edges:
            pu, pv = verts[int(u)], verts[int(v)]
            bg_draw.line(
                [(int(round(pu[0])), int(round(pu[1]))), (int(round(pv[0])), int(round(pv[1])))],
                fill=(70, 70, 70), width=1,
            )
