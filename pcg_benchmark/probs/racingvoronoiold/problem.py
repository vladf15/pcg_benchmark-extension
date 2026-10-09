from __future__ import annotations

from pcg_benchmark.probs.racing.problem import RacingProblem
from pcg_benchmark.probs.racing.utils import (
    count_self_intersections,
    count_track_area_intersections,
)
from pcg_benchmark.probs.racingvoronoiold.utils import (
    build_voronoi_cell_graph,
    fillet_corners,
    find_boundary_cycle,
)
from pcg_benchmark.spaces import ArraySpace, FloatSpace, DictionarySpace
import numpy as np


class RacingVoronoiOldProblem(RacingProblem):
    """Cell selection on one fixed Voronoi diagram: the representation
    racingvoronoi-v0 had before its genome gained per-cell power-diagram
    weights (see racingvoronoi/problem.py).  Kept as its own problem so
    results can be compared with it; the code is otherwise that version's."""

    _render_desc = 'Rendering voronoi frames'

    # Decoded points already trace a valid loop in order; the base class's
    # 2-opt untangle must not reorder them (it would break the loop).
    _untangle_control_points = False

    # Side of the square the selectable sites are drawn from, as a fraction of
    # the build box side (user decision).
    _SITE_BOX_FRAC = 0.9

    def __init__(self, **kwargs):
        kwargs.setdefault('num_points', 15)
        # max_steps is NOT overridden: the shared 7000-step budget applies.
        # Now that every representation generates a lap of the same length, a
        # per-representation cap would score the budget rather than the track.
        # A 4400 m lap at the ~16 m/s the agent averages needs ~2750 steps, so
        # 7000 leaves headroom for a slow lap without truncating a good one.

        # The two counts trade lap length against how many distinct tracks the
        # representation can express, and the binding constraint is the number
        # of ELIGIBLE cells, not num_cells: every cell whose polygon leaves the
        # build box is excluded.  Measured over 40 random genomes, 10 cells
        # selected, no relaxation, 50 guard points, sites in 90% of the side of
        # the 1300 m build box:
        #
        #   cells  eligible  distinct  median lap (p10-p90)  tightest corner p10 / median  turns
        #      50        27     40/40  3589 m (2527-4344)    8.0 / 10.6 m                  16
        #      70        43     40/40  3271 m (2401-3987)    6.0 /  9.2 m                  17
        #      90        59     40/40  2994 m (2239-3684)    5.2 /  8.6 m                  20
        #
        # 50 cells is the setting taken.  More cells give the genome more cells
        # to reach, but cut cell size, so the rim gains corners, the lap
        # shortens and the tightest corner falls toward the 6 m at which
        # min_radius_score is 0.  With the sites over the whole 1500 m map and
        # the 1091 m build box, 50 cells left 15 eligible.
        num_cells = int(kwargs.pop('num_cells', 50))
        num_selected = int(kwargs.pop('num_selected_cells', 10))
        voronoi_seed = int(kwargs.pop('voronoi_seed', 0))
        # Lloyd relaxation: repeatedly move each site to the centre of its own
        # cell, which evens the cells out.  0 disables it and uses the raw
        # random scatter, which is what "a Voronoi grid" means with nothing
        # added on top.
        #
        # Measured over 8 genomes at 0 / 1 / 2 / 4 iterations, with the sites
        # confined to the build box and no guard points.  What it changes
        # is cell evenness (spread of nearest-neighbour spacing 0.57 / 0.36 /
        # 0.32 / 0.18) and, through that, lap length (3280 / 3512 / 3722 /
        # 3840 m), because rounder cells give the selected cluster a longer
        # rim.  What it does NOT change: every genome still picks a different
        # cluster (8 of 8 at every setting), the car still finishes with no
        # off-road time, and the quality differences are inside the noise of
        # this sample size.
        #
        # Off by default.  The cost is real and worth stating: voronoi already
        # sat below the shared 3900-5900 m length band, and this moves it about
        # 440 m further down.  That is a length-band question, not a reason to
        # keep a processing step the representation is not supposed to have.
        lloyd_iterations = int(kwargs.pop('lloyd_iterations', 0))
        # Where the sites go, a user decision.  The num_cells selectable sites
        # are scattered uniformly over a centred square with _SITE_BOX_FRAC of
        # the build box side (1170 m in the 1300 m box).  num_guard_points more
        # are scattered uniformly over a guard_outer_m square minus the
        # guard_inner_m square, both centred on the map (3500 m and 2500 m on
        # the 1500 m map).  They give every cell near the map edge a neighbour
        # beyond it, so those cells close instead of running off to infinity;
        # their own cells are never selectable.
        num_guard_points = int(kwargs.pop('num_guard_points', 50))
        guard_outer_m = float(kwargs.pop('guard_outer_m', 3500.0))
        guard_inner_m = float(kwargs.pop('guard_inner_m', 2500.0))

        super().__init__(**kwargs)

        self._num_cells = num_cells
        self._num_selected_cells = num_selected
        self._voronoi_seed = voronoi_seed
        self._lloyd_iterations = lloyd_iterations
        self._num_guard_points = num_guard_points
        self._guard_outer_m = guard_outer_m
        self._guard_inner_m = guard_inner_m

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
        w, h = float(self._width), float(self._height)
        cx, cy = 0.5 * w, 0.5 * h
        ho, hi = 0.5 * self._guard_outer_m, 0.5 * self._guard_inner_m
        bx0, by0, bx1, by1 = self._build_box()
        margin = 0.5 * (1.0 - self._SITE_BOX_FRAC) * (bx1 - bx0)
        sites = (bx0 + margin, by0 + margin, bx1 - margin, by1 - margin)
        graph = build_voronoi_cell_graph(self._num_cells, sites,
                                         self._voronoi_seed,
                                         lloyd_iterations=self._lloyd_iterations,
                                         num_guard_points=self._num_guard_points,
                                         guard_outer=(cx - ho, cy - ho, cx + ho, cy + ho),
                                         guard_inner=(cx - hi, cy - hi, cx + hi, cy + hi),
                                         clip_box=(0.0, 0.0, w, h))
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
        # Guard sites own indices num_cells and up and are never selectable, so
        # they are left out of the set: it holds selectable cell indices only,
        # and len() on it is the count of excluded cells (23 of 50 at the
        # defaults).  With the guards in, it would hold 73 entries for 50 cells.
        for (va, vb), (p1, p2) in zip(self._voronoi_all_edges, self._voronoi_all_pairs):
            if not (vertex_ok[int(va)] and vertex_ok[int(vb)]):
                self._ineligible_cells.update(
                    int(p) for p in (p1, p2) if int(p) < self._num_cells)

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
        if start is None:
            # No cell's polygon fits the build box, so there is nothing to
            # grow a cluster from.  Reachable only if the box, the site count
            # or the site spread is changed; at the defaults 27 of 50 are
            # eligible.  Returning empty lets _extract_content fall back
            # rather than raising from inside the adjacency lookup.
            return []
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
        points, run on the geometry this decoder produces.  It is driven by
        the test rather than by a radius threshold because the radius does not
        predict the fold: on the sample below the failing rims' smallest
        fillets run 4.1 to 5.7 m, but 19 of the 25 passing rims also carry a
        fillet under the 8 m half-width (down to 4.4 m), and a threshold at
        that value removed vertices from 34 of 40 rims instead of 15.

        Measured on 40 random genomes (seed 7): 15 fail the gate before the
        repair and 0 after.  Only those 15 are changed, losing a median of 1
        vertex and at most 2; the other 25 are identical to the unrepaired
        rim.  The lap shortens by up to 104 m, and the closest approach
        between two non-adjacent parts of the repaired road is 27.5 m against
        a 16 m road, so no fold is traded for a collision elsewhere.  Building
        the curve costs 11.3 ms with the repair against 1.0 ms without, since
        each pass runs the area check.

        No 2-opt pass follows the removal, unlike racing-v0: the rim is
        already in loop order, and reordering it would stop the track from
        following the cells the genome selected.  The floor of six vertices
        keeps a rim that cannot be repaired from collapsing; such a rim keeps
        its fold and scores 0, which is the honest outcome."""
        pts = np.asarray(poly, dtype=float)
        width = float(self._track_width)
        while len(pts) > 6:
            curve = self._fillet_curve(pts)
            if (count_self_intersections(curve)
                    + count_track_area_intersections(curve, track_width=width) == 0):
                break
            k = int(np.argmin(np.linalg.norm(pts - self._overlap_spot(curve), axis=1)))
            pts = np.delete(pts, k, axis=0)
        return pts

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

    # No quality parameter is overridden: one quality function for all five
    # representations.  Both halves of the self-overlap gate run on the dense
    # curve, as they do everywhere else.
    #
    # Why not exempt Voronoi from the area check: its fillets can fold the
    # inner road edge, and an exemption would score those tracks as though
    # they did not.  On 40 random genomes (seed 7), 15 unrepaired rims fold.
    # _drop_tight_corners removes the fold, so the gate is passed on the
    # geometry rather than waived.  Why not check the sparse polygon instead
    # of the curve: on the same 40 genomes neither form finds a crossing, so
    # the polygon check buys nothing and would be a second definition.
    #
    # The longest-straight band is not overridden either.  Voronoi is the
    # tightest case, with straights pinned to the diagram edge scale: on 20
    # random genomes (seed 21) the median longest straight is 215 m and none
    # reaches the 250 m full-marks figure, so the term scores them between
    # its 150 m zero and full marks rather than locking them out.

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
