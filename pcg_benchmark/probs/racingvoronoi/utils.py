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
    would drag sites, and with them the whole diagram, off the canvas.
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


def generate_sites(num_cells: int, region, voronoi_seed: int, lloyd_iterations: int = 0,
                   num_guard_points: int = 0, guard_outer=None, guard_inner=None) -> np.ndarray:
    """Sites of the diagram: num_cells selectable sites drawn uniformly from
    `region` (x0, y0, x1, y1), optionally Lloyd-relaxed, followed by
    num_guard_points guard sites drawn uniformly from the `guard_outer`
    rectangle minus the `guard_inner` one.  Guards give the cells near the
    region's edge a neighbour beyond it, so those cells close; they take
    indices num_cells and up and are never selectable.  The same draws as
    racingvoronoiold's build_voronoi_cell_graph, so equal seeds give equal
    sites."""
    x0, y0, x1, y1 = (float(v) for v in region)
    rng = np.random.default_rng(voronoi_seed)
    cell_sites = rng.uniform(low=[x0, y0], high=[x1, y1], size=(num_cells, 2)).astype(float)
    cell_sites = lloyd_relaxation(cell_sites, lloyd_iterations, bounds=(x0, y0, x1, y1))
    guards = []
    if num_guard_points > 0:
        ox0, oy0, ox1, oy1 = (float(v) for v in guard_outer)
        ix0, iy0, ix1, iy1 = (float(v) for v in guard_inner)
        while len(guards) < num_guard_points:
            gx, gy = rng.uniform(low=[ox0, oy0], high=[ox1, oy1])
            if not (ix0 <= gx <= ix1 and iy0 <= gy <= iy1):
                guards.append((float(gx), float(gy)))
    return np.vstack([cell_sites, np.asarray(guards, dtype=float).reshape(-1, 2)])


def build_power_cell_graph(sites: np.ndarray, num_cells: int, weights: np.ndarray,
                           clip_box) -> dict:
    """Power diagram (Laguerre-Voronoi diagram) of weighted sites.

    The cell of site i is the set of points p where |p - s_i|^2 - w_i is
    smallest (Aurenhammer 1987).  With every weight equal it is the Voronoi
    diagram.  A larger weight moves each edge of that cell away from its site:
    the edge between i and j stays perpendicular to s_i s_j and shifts by
    (w_i - w_j) / (2 |s_i - s_j|) toward j.  A cell can also be empty, when
    its neighbours' weights exceed its own by enough.

    Built as Aurenhammer does, by lifting each site to (x, y, x^2 + y^2 - w)
    and taking the lower convex hull: each lower facet is a vertex of the
    diagram (the point with equal power to its three sites), two lower facets
    sharing an edge give the diagram edge between that edge's two sites, and
    a lower facet edge on the hull's rim gives a semi-infinite edge.  A site
    on no lower facet has an empty cell.  Why not scipy.spatial.Voronoi: it
    has no weights.  Why not a raster (label each pixel by least power): the
    track runs along the cell edges, and a raster would move them by up to a
    pixel and turn every straight edge into a staircase.

    `weights` holds one weight per site, guards included (m^2).  Semi-infinite
    edges are cut at `clip_box` (x0, y0, x1, y1) so the full diagram can be
    drawn; their cells are boundary cells and never selectable.

    Returns a dict with:
        cell_sites          (num_cells, 2) float, the selectable sites
        voronoi_vertices    (V, 2) float, diagram vertices + clipped ray ends
        all_edges_full      (E, 2) int, vertex index pairs
        all_edge_pairs_full (E, 2) int, the two sites each edge separates
        cell_neighbours     list[list[int]], selectable neighbours per cell
        boundary_cells      set[int], selectable cells with a semi-infinite edge
        empty_cells         set[int], selectable cells with no region
    """
    sites = np.asarray(sites, dtype=float)
    w = np.asarray(weights, dtype=float).ravel()
    cx0, cy0, cx1, cy1 = (float(v) for v in clip_box)
    # Centred before lifting, which the power function does not notice and
    # which keeps the lifted z coordinates small enough for Qhull.
    origin = sites.mean(axis=0)
    local = sites - origin
    lifted = np.column_stack([local, np.sum(local * local, axis=1) - w])
    try:
        hull = scipy.spatial.ConvexHull(lifted)
    except scipy.spatial.QhullError:
        rng = np.random.default_rng(0)
        hull = scipy.spatial.ConvexHull(lifted + rng.normal(scale=1e-6, size=lifted.shape))
    eq = hull.equations
    lower = eq[:, 2] < 0.0
    # The lower facet z = a x + b y + c holds the point with equal power to
    # its three sites at (a / 2, b / 2).
    facet_vertex = {}
    vertices = []
    for f in np.flatnonzero(lower):
        a, b = -eq[f, 0] / eq[f, 2], -eq[f, 1] / eq[f, 2]
        facet_vertex[int(f)] = len(vertices)
        vertices.append(origin + 0.5 * np.array([a, b]))

    edges, pairs = [], []
    boundary_cells: set[int] = set()
    neighbour_sets: list[set[int]] = [set() for _ in range(num_cells)]
    on_lower = set()
    for f in np.flatnonzero(lower):
        f = int(f)
        simplex = hull.simplices[f]
        on_lower.update(int(v) for v in simplex)
        for k in range(3):
            p1, p2 = (int(v) for v in np.delete(simplex, k))
            nb = int(hull.neighbors[f][k])
            if lower[nb]:
                if f < nb:                          # each shared edge once
                    edges.append((facet_vertex[f], facet_vertex[nb]))
                    pairs.append((p1, p2))
                    if p1 < num_cells and p2 < num_cells:
                        neighbour_sets[p1].add(p2)
                        neighbour_sets[p2].add(p1)
                continue
            # Rim of the lower hull: a semi-infinite edge, perpendicular to
            # s1 s2 and heading away from the facet's third site.
            boundary_cells.update(p for p in (p1, p2) if p < num_cells)
            tangent = sites[p2] - sites[p1]
            normal = np.array([-tangent[1], tangent[0]])
            nlen = float(np.linalg.norm(normal))
            if nlen < 1e-10:
                continue
            normal /= nlen
            if np.dot(normal, sites[int(simplex[k])] - sites[p1]) > 0:
                normal = -normal
            clipped = _ray_bbox_intersect(vertices[facet_vertex[f]], normal, cx0, cx1, cy0, cy1)
            if clipped is not None:
                edges.append((facet_vertex[f], len(vertices)))
                pairs.append((p1, p2))
                vertices.append(clipped)

    return {
        'cell_sites':          sites[:num_cells],
        'voronoi_vertices':    np.asarray(vertices, dtype=float).reshape(-1, 2),
        'all_edges_full':      np.asarray(edges, dtype=int).reshape(-1, 2),
        'all_edge_pairs_full': np.asarray(pairs, dtype=int).reshape(-1, 2),
        'cell_neighbours':     [sorted(n) for n in neighbour_sets],
        'boundary_cells':      boundary_cells,
        'empty_cells':         {i for i in range(num_cells) if i not in on_lower},
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
        # Closed perimeter: the last vertex joins the first.
        perimeter = float(np.sum(np.linalg.norm(np.roll(positions, -1, axis=0) - positions, axis=1)))
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
