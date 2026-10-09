from __future__ import annotations

import numpy as np
import scipy.spatial


def lloyd_relaxation(cell_sites: np.ndarray, num_iterations: int,
                     bounds: tuple[float, float, float, float] | None = None) -> np.ndarray:
    """Run Lloyd's Voronoi relaxation. Cells with infinite regions are left in place.

    `num_iterations` of 0 returns the sites untouched, which is the raw
    Voronoi diagram with no relaxation applied.

    Boundary-adjacent cells can have finite regions whose centroid lies far
    outside the map (their Voronoi vertices are unbounded in practice), which
    would drag sites — and with them the whole diagram — off the canvas.
    `bounds` (xmin, ymin, xmax, ymax) clamps every relaxed site back inside.
    """
    num_cells = len(cell_sites)
    relaxed_sites = cell_sites.copy()

    for _ in range(num_iterations):
        voronoi = scipy.spatial.Voronoi(relaxed_sites)
        new_sites = relaxed_sites.copy()
        for cell_index in range(num_cells):
            region_vertex_indices = voronoi.regions[voronoi.point_region[cell_index]]
            if -1 in region_vertex_indices or len(region_vertex_indices) < 3:
                continue
            polygon_vertices = voronoi.vertices[np.asarray(region_vertex_indices)]
            x_coords, y_coords = polygon_vertices[:, 0], polygon_vertices[:, 1]
            x_coords_next = np.roll(x_coords, -1)
            y_coords_next = np.roll(y_coords, -1)
            shoelace_cross = x_coords * y_coords_next - x_coords_next * y_coords
            signed_area = 0.5 * shoelace_cross.sum()
            if abs(signed_area) < 1e-10:
                new_sites[cell_index] = polygon_vertices.mean(axis=0)
            else:
                new_sites[cell_index] = [
                    ((x_coords + x_coords_next) * shoelace_cross).sum() / (6 * signed_area),
                    ((y_coords + y_coords_next) * shoelace_cross).sum() / (6 * signed_area),
                ]
        if bounds is not None:
            xmin, ymin, xmax, ymax = bounds
            new_sites[:, 0] = np.clip(new_sites[:, 0], xmin, xmax)
            new_sites[:, 1] = np.clip(new_sites[:, 1], ymin, ymax)
        relaxed_sites = new_sites

    return relaxed_sites


def _ray_bbox_intersect(
    start: np.ndarray, direction: np.ndarray,
    xmin: float, xmax: float, ymin: float, ymax: float,
) -> np.ndarray | None:
    """Return the first point where a ray from `start` in `direction` crosses the bbox."""
    x0, y0 = float(start[0]), float(start[1])
    dx, dy = float(direction[0]), float(direction[1])
    t_hits: list[float] = []
    if abs(dx) > 1e-12:
        for xb in (xmin, xmax):
            t = (xb - x0) / dx
            if t > 1e-9 and ymin - 1e-9 <= y0 + t * dy <= ymax + 1e-9:
                t_hits.append(t)
    if abs(dy) > 1e-12:
        for yb in (ymin, ymax):
            t = (yb - y0) / dy
            if t > 1e-9 and xmin - 1e-9 <= x0 + t * dx <= xmax + 1e-9:
                t_hits.append(t)
    if not t_hits:
        return None
    t = min(t_hits)
    return np.array([x0 + t * dx, y0 + t * dy], dtype=float)


def build_voronoi_cell_graph(num_cells: int, region, voronoi_seed: int, lloyd_iterations: int = 2,
                             num_guard_points: int = 0, guard_outer=None, guard_inner=None,
                             clip_box=None) -> dict:
    """Generate a Voronoi grid over `region`, optionally surrounded by guard sites.

    `region` is (x0, y0, x1, y1): the num_cells selectable sites are drawn
    uniformly from it.  With num_guard_points > 0, that many more sites are
    drawn uniformly from the `guard_outer` rectangle, rejecting draws inside
    the `guard_inner` rectangle (both (x0, y0, x1, y1)).  They give the cells
    near the region's edge a neighbour beyond it, so those cells close.  Guard
    sites take indices num_cells and up: they appear in all_edge_pairs_full
    but never in cell_neighbours or boundary_cells, and cell_sites holds only
    the selectable sites.  Lloyd relaxation, when used, runs on the selectable
    sites before the guards are added.

    Cells with semi-infinite edges are added to boundary_cells (ineligible for selection).
    Each semi-infinite edge is clipped to the bounding box and appended to the vertex/edge
    arrays so the full diagram renders correctly.

    Returns a dict with:
        cell_sites          (N, 2) float
        voronoi_vertices    (V, 2) float     - finite vertices + clipped ray endpoints.
        all_edges_full      (E, 2) int
        all_edge_pairs_full (E, 2) int
        cell_neighbours     list[list[int]]
        boundary_cells      set[int]
    """
    x0, y0, x1, y1 = (float(v) for v in region)
    # Where a semi-infinite ridge is cut off so it has a drawable endpoint.
    # It bounds the rendered diagram only: cells owning such a ridge are
    # boundary cells and never selectable, whatever the cut-off is.  With no
    # clip_box the cut-off is (0, 0) to the region's far corner.
    cx0, cy0, cx1, cy1 = ((float(v) for v in clip_box) if clip_box is not None
                          else (0.0, 0.0, x1, y1))

    rng = np.random.default_rng(voronoi_seed)
    cell_sites = rng.uniform(low=[x0, y0], high=[x1, y1], size=(num_cells, 2)).astype(float)

    cell_sites = lloyd_relaxation(cell_sites, lloyd_iterations, bounds=(x0, y0, x1, y1))

    # Guard sites go after the selectable ones, so those keep indices
    # 0..num_cells-1 and the genome still maps one score to each.  From here
    # on cell_sites holds both; the return value slices the selectable part.
    guards = []
    if num_guard_points > 0:
        ox0, oy0, ox1, oy1 = (float(v) for v in guard_outer)
        ix0, iy0, ix1, iy1 = (float(v) for v in guard_inner)
        while len(guards) < num_guard_points:
            gx, gy = rng.uniform(low=[ox0, oy0], high=[ox1, oy1])
            if not (ix0 <= gx <= ix1 and iy0 <= gy <= iy1):
                guards.append((float(gx), float(gy)))
    cell_sites = np.vstack([cell_sites, np.asarray(guards, dtype=float).reshape(-1, 2)])

    try:
        voronoi = scipy.spatial.Voronoi(cell_sites) 
    except scipy.spatial.QhullError:
        rng = np.random.default_rng(voronoi_seed)
        cell_sites = cell_sites + rng.normal(scale=1e-6, size=cell_sites.shape)
        voronoi = scipy.spatial.Voronoi(cell_sites)

    center   = cell_sites.mean(axis=0)
    vertices = list(np.asarray(voronoi.vertices, dtype=float))

    edges_dict:     dict[tuple[int, int], tuple[int, int]] = {}
    boundary_cells: set[int]                               = set()
    neighbour_sets: list[set[int]]                         = [set() for _ in range(num_cells)]

    for (va, vb), (p1, p2) in zip(voronoi.ridge_vertices, voronoi.ridge_points):
        p1, p2 = int(p1), int(p2)

        if va >= 0 and vb >= 0:
            edges_dict[(min(va, vb), max(va, vb))] = (p1, p2)
            if p1 < num_cells and p2 < num_cells:
                neighbour_sets[p1].add(p2)
                neighbour_sets[p2].add(p1)
        else:
            boundary_cells.update(p for p in (p1, p2) if p < num_cells)
            finite_v = va if va >= 0 else vb
            tangent  = cell_sites[p2] - cell_sites[p1]
            normal   = np.array([-tangent[1], tangent[0]], dtype=float)
            nlen     = np.linalg.norm(normal)
            if nlen < 1e-10:
                continue
            normal /= nlen
            if np.dot(normal, (cell_sites[p1] + cell_sites[p2]) * 0.5 - center) < 0:
                normal = -normal
            clipped = _ray_bbox_intersect(voronoi.vertices[finite_v], normal,
                                          cx0, cx1, cy0, cy1)
            if clipped is not None:
                new_idx = len(vertices)
                vertices.append(clipped)
                edges_dict[(finite_v, new_idx)] = (p1, p2)

    voronoi_vertices = np.array(vertices, dtype=float)
    sorted_keys      = sorted(edges_dict)
    if sorted_keys:
        all_edges_full      = np.array(sorted_keys,                          dtype=int)
        all_edge_pairs_full = np.array([edges_dict[k] for k in sorted_keys], dtype=int)
    else:
        all_edges_full      = np.zeros((0, 2), dtype=int)
        all_edge_pairs_full = np.zeros((0, 2), dtype=int)

    return {
        'cell_sites':          cell_sites[:num_cells],
        'voronoi_vertices':    voronoi_vertices,
        'all_edges_full':      all_edges_full,
        'all_edge_pairs_full': all_edge_pairs_full,
        'cell_neighbours':     [sorted(s) for s in neighbour_sets],
        'boundary_cells':      boundary_cells,
    }


def find_boundary_cycle(boundary_edges: np.ndarray, voronoi_vertices: np.ndarray) -> list[int] | None:
    """Find the longest closed loop in a set of Voronoi boundary edges."""
    edges = np.asarray(boundary_edges, dtype=int)
    if edges.ndim == 1:
        edges = edges.reshape(-1, 2)
    if len(edges) == 0:
        return None

    adjacency: dict[int, list[int]] = {}
    for vert_a, vert_b in edges:
        vert_a, vert_b = int(vert_a), int(vert_b)
        if vert_a == vert_b:
            continue
        adjacency.setdefault(vert_a, []).append(vert_b)
        adjacency.setdefault(vert_b, []).append(vert_a)

    voronoi_vertices   = np.asarray(voronoi_vertices, dtype=float)
    visited:            set[int]        = set()
    best_cycle:         list[int] | None = None
    longest_perimeter:  float            = -1.0

    for start_vertex in adjacency:
        if start_vertex in visited:
            continue

        component: set[int] = set()
        stack = [start_vertex]
        while stack:
            v = stack.pop()
            if v in component:
                continue
            component.add(v)
            for neighbour in adjacency.get(v, []):
                if neighbour not in component:
                    stack.append(neighbour)
        visited.update(component)

        # A clean cycle needs every vertex to have exactly two neighbours.
        is_clean_cycle = len(component) > 0
        for v in component:
            if len(adjacency.get(v, [])) != 2:
                is_clean_cycle = False
                break
        if not is_clean_cycle:
            continue

        # Walk the cycle: from each vertex, continue to whichever of its two
        # neighbours we did not just come from.  (next(iter(...)) just takes
        # an arbitrary element from the set; sets cannot be indexed.)
        first_vertex = next(iter(component))
        prev_vertex, current_vertex, cycle = None, first_vertex, []
        for _ in range(len(component) + 2):
            cycle.append(current_vertex)
            neighbours = adjacency[current_vertex]
            if prev_vertex is None or neighbours[0] != prev_vertex:
                next_vertex = neighbours[0]
            else:
                next_vertex = neighbours[1]
            prev_vertex, current_vertex = current_vertex, int(next_vertex)
            if current_vertex == first_vertex:
                break

        if len(cycle) < 3 or current_vertex != first_vertex:
            continue

        positions = voronoi_vertices[np.array(cycle, dtype=int)]
        perimeter = float(np.sum(np.linalg.norm(positions[1:] - positions[:-1], axis=1)))
        if perimeter > longest_perimeter:
            longest_perimeter = perimeter
            best_cycle        = cycle

    return best_cycle


def fillet_corners(points: np.ndarray, max_radius: float, sample_step: float) -> np.ndarray:
    """Replace each polygon corner with a circular arc tangent to both edges.

    The radius adapts so the arc never eats past the midpoint of either
    adjacent edge (two arcs share each edge), capped at max_radius.  Arcs are
    sampled about sample_step apart, which keeps their curvature contiguous:
    downstream corner counting sees one corner per arc no matter how large
    the radius is, unlike corner cutting whose straight chords reset it.
    """
    poly = np.asarray(points, dtype=float)
    n = len(poly)
    if n < 3:
        return poly
    out: list[np.ndarray] = []
    for i in range(n):
        prev, curr, nxt = poly[(i - 1) % n], poly[i], poly[(i + 1) % n]
        v_in, v_out = curr - prev, nxt - curr
        l_in, l_out = float(np.linalg.norm(v_in)), float(np.linalg.norm(v_out))
        if l_in < 1e-9 or l_out < 1e-9:
            out.append(curr)
            continue
        u_in, u_out = v_in / l_in, v_out / l_out
        cos_t = float(np.clip(np.dot(u_in, u_out), -1.0, 1.0))
        turn = float(np.arccos(cos_t))
        if turn < np.deg2rad(2.0):
            out.append(curr)
            continue
        half_tan = np.tan(turn / 2.0)
        tangent_allow = 0.5 * min(l_in, l_out)
        radius = min(float(max_radius), tangent_allow / max(half_tan, 1e-9))
        tangent_len = radius * half_tan
        arc_start = curr - u_in * tangent_len
        sign = 1.0 if (u_in[0] * u_out[1] - u_in[1] * u_out[0]) > 0 else -1.0
        normal_in = np.array([-u_in[1], u_in[0]]) * sign
        center = arc_start + normal_in * radius
        a0 = float(np.arctan2(arc_start[1] - center[1], arc_start[0] - center[0]))
        sweep = sign * turn
        n_samples = max(2, int(np.ceil(abs(sweep) * radius / max(float(sample_step), 1e-9))) + 1)
        for k in range(n_samples + 1):
            a = a0 + sweep * (k / n_samples)
            out.append(center + radius * np.array([np.cos(a), np.sin(a)]))
    return np.asarray(out, dtype=float)
