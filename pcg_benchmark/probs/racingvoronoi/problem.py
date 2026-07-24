from __future__ import annotations

from pcg_benchmark.probs.racing.problem import RacingProblem
from pcg_benchmark.probs.racingvoronoi.utils import (
    build_voronoi_cell_graph,
    fillet_corners,
    find_boundary_cycle,
    merge_short_edges,
    remove_spike_vertices,
)
from pcg_benchmark.spaces import ArraySpace, FloatSpace, IntegerSpace, DictionarySpace
from pcg_benchmark.probs.utils import get_range_reward
import numpy as np


class RacingVoronoiProblem(RacingProblem):

    _render_desc = 'Rendering voronoi frames'

    # Decoded points already trace a valid loop in order; the base class's
    # 2-opt untangle must not reorder them (it would break the loop).
    _untangle_control_points = False

    def __init__(self, **kwargs):
        kwargs.setdefault('num_points', 15)
        kwargs.setdefault('max_steps', 2000)

        # Corner count is set by the number of USED cells, not by shrinking the
        # diagram (2026-07-24): 100/15 gave ~37 turns (roughly double a famous
        # European circuit's ~15) because voronoi corners are pinned to the
        # cell-edge scale.  Keeping 50 total cells (small enough that the cells
        # are not oversized, ~26 eligible -> lots of search freedom) and using
        # only 6 of them gives ~19 turns.  The trade-off is that a 6-cell loop
        # is shorter (~1600 m, length_score ~0.71) since fewer/smaller cells
        # cannot both be few AND span a long lap - voronoi stays the most
        # corner-dense representation (19 vs ~16), an inherent property of the
        # tessellation.
        num_cells = int(kwargs.pop('num_cells', 50))
        num_selected = int(kwargs.pop('num_selected_cells', 6))
        voronoi_seed = int(kwargs.pop('voronoi_seed', 0))

        super().__init__(**kwargs)

        self._num_cells = num_cells
        self._num_selected_cells = num_selected
        self._voronoi_seed = voronoi_seed

        self._content_space = DictionarySpace({
            "cell_scores": ArraySpace((self._num_cells,), FloatSpace(0.0, 1.0)),
        })
        self._control_space = DictionarySpace({
            "length":    FloatSpace(500.0, self._width * 16.0),
            "num_turns": IntegerSpace(1, num_selected * 4),
        })

        self._cell_sites = None
        self._boundary_cells = None
        self._ineligible_cells = None
        self._voronoi_vertices = None
        self._voronoi_all_edges = None
        self._voronoi_all_pairs = None
        self._cell_adjacency = None
        self._last_selected_cells = None

    # _densify_polyline is inherited from RacingProblem (shared with tile).

    # ------------------------------------------------------------------
    # Fixed Voronoi grid
    # ------------------------------------------------------------------

    def _build_cell_graph(self):
        if self._cell_sites is not None:
            return
        graph = build_voronoi_cell_graph(self._num_cells, self._width, self._height, self._voronoi_seed)
        self._cell_sites             = graph['cell_sites']
        self._voronoi_vertices       = graph['voronoi_vertices']
        self._voronoi_all_edges = graph['all_edges_full']
        self._voronoi_all_pairs = graph['all_edge_pairs_full']
        self._cell_adjacency         = graph['cell_neighbours']
        self._boundary_cells         = graph['boundary_cells']

        # Eligibility rule: a cell may be selected iff its polygon stays fully
        # inside the safe box (margin inside the map edges).  That is exactly
        # the out-of-bounds condition quality() enforces on the decoded track,
        # since the track runs along the selected cells' Voronoi vertices.
        # Two ways a cell can violate it: its region is semi-infinite (clipped
        # at the map border), or a finite region owns a vertex outside the
        # safe box (near-collinear sites push Voronoi vertices arbitrarily far
        # out).  No other exclusions (in particular no buffer ring around the
        # border cells), so eligibility is visually consistent: what is fully
        # in bounds is selectable.
        self._ineligible_cells = set(self._boundary_cells)
        margin = float(self._track_width) * 0.5 + 2.0
        verts = self._voronoi_vertices
        vertex_ok = (
            (verts[:, 0] >= margin) & (verts[:, 0] <= self._width - margin) &
            (verts[:, 1] >= margin) & (verts[:, 1] <= self._height - margin)
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
        """Return the cell with the highest score; ties go to the lowest index."""
        best = None
        for i in cells:
            if best is None:
                best = i
            elif scores[i] > scores[best]:
                best = i
            elif scores[i] == scores[best] and i < best:
                best = i
        return best

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
                leftovers = []
                for i in selectable:
                    if i not in selected:
                        leftovers.append((-float(scores[i]), i))
                leftovers.sort()
                for _score, i in leftovers[: k - len(selected)]:
                    selected.add(i)
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

            pts = self._voronoi_vertices[np.array(cycle, dtype=int)]
            pts = remove_spike_vertices(pts)
            # Collapse vertex clusters tighter than 1.5 track widths: they are
            # artifacts of the cell tessellation, not corners a track designer
            # would build.  The rounding itself happens in _make_curve.
            pts = merge_short_edges(pts, min_edge=float(self._track_width) * 1.5)
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
    _FILLET_MAX_RADIUS = 40.0

    def _make_curve(self, track_points):
        """Voronoi tracks are polygons: round each corner with a tangent arc,
        then densify the remaining straights (a spline would bow the straights
        and round every corner twice)."""
        step = float(self._track_width) * 0.35
        rounded = fillet_corners(track_points, max_radius=self._FILLET_MAX_RADIUS, sample_step=step)
        return self._densify_polyline(rounded, max_step=step)

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
            scores_key = tuple(np.asarray(content['cell_scores'], dtype=float).ravel().tolist())
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
        # start_straight is NOT overridden: the shared 50 m target (base class)
        # applies to every representation (2026-07-22 user rule).  Voronoi is the
        # tightest case (straights pinned to the diagram edge scale); measured on
        # the 750 m map, 29% of random genomes already reach 50 m and the GA can
        # push toward the ~99 m diagram-best chain, so 50 m is a real, reachable
        # discriminator (it is the one thing random voronoi content genuinely
        # fails often) without locking the representation out.
    }

    # ------------------------------------------------------------------
    # Controlability
    # ------------------------------------------------------------------
    # Diversity is inherited from RacingProblem: it compares the decoded
    # track shapes (occupancy grid + average turn), which keeps the measure
    # identical and comparable across all four representations.  A Jaccard
    # measure on the selected cell sets would be genotype-based and only
    # meaningful for this representation.

    def controlability(self, info, control):
        length_err = self._width * 0.6
        l_score = get_range_reward(
            info.get('total_length', 0.0), 0,
            control['length'] - length_err,
            control['length'] + length_err,
            self._width * 16.0,
        )
        turns_err = 2
        t_score = get_range_reward(
            info.get('num_turns', 0), 0,
            control['num_turns'] - turns_err,
            control['num_turns'] + turns_err,
            self._num_selected_cells * 4,
        )
        return (l_score + t_score) / 2.0

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
