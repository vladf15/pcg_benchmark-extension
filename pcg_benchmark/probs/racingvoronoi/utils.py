from __future__ import annotations

import numpy as np
import scipy.spatial


def lloyd_relaxation(cell_sites: np.ndarray, num_iterations: int,
                     bounds: tuple[float, float, float, float] | None = None) -> np.ndarray:
    """Run Lloyd's Voronoi relaxation. Cells with infinite regions are left in place.

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


def build_voronoi_cell_graph(num_cells: int, width: float, height: float, voronoi_seed: int, lloyd_iterations: int = 2) -> dict:
    """Generate a Voronoi grid without guard/boundary points.

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
    W, H = float(width), float(height)

    margin = float(min(W, H)) * 0.02
    cell_sites = np.random.default_rng(voronoi_seed).uniform(
        low=[margin, margin],
        high=[W - margin, H - margin],
        size=(num_cells, 2),
    ).astype(float)

    cell_sites = lloyd_relaxation(cell_sites, lloyd_iterations,
                                  bounds=(margin, margin, W - margin, H - margin))

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
            neighbour_sets[p1].add(p2)
            neighbour_sets[p2].add(p1)
        else:
            boundary_cells.add(p1)
            boundary_cells.add(p2)
            finite_v = va if va >= 0 else vb
            tangent  = cell_sites[p2] - cell_sites[p1]
            normal   = np.array([-tangent[1], tangent[0]], dtype=float)
            nlen     = np.linalg.norm(normal)
            if nlen < 1e-10:
                continue
            normal /= nlen
            if np.dot(normal, (cell_sites[p1] + cell_sites[p2]) * 0.5 - center) < 0:
                normal = -normal
            clipped = _ray_bbox_intersect(voronoi.vertices[finite_v], normal, 0.0, W, 0.0, H)
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
        'cell_sites':          cell_sites,
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


def smooth_corners(points: np.ndarray, max_radius: float = 20.0, passes: int = 2) -> np.ndarray:
    """Angle-adaptive corner cutting with a flat world-unit cap, repeated `passes` times.

    cut = min(angle_factor * edge_length, max_radius, 0.45 * edge_length)
    angle_factor in [0, 0.45]: 0 for straight-through, 0.45 for full hairpin.
    """
    pts = np.asarray(points, dtype=float)
    for _ in range(passes):
        n = len(pts)
        if n < 3:
            return pts

        new_pts: list[np.ndarray] = []
        for i in range(n):
            prev = pts[(i - 1) % n]
            curr = pts[i]
            nxt  = pts[(i + 1) % n]

            v_in,  v_out  = curr - prev, nxt - curr
            len_in, len_out = float(np.linalg.norm(v_in)), float(np.linalg.norm(v_out))
            if len_in < 1e-9 or len_out < 1e-9:
                new_pts.append(curr)
                continue

            cos_a        = float(np.clip(np.dot(v_in / len_in, v_out / len_out), -1.0, 1.0))
            angle_factor = 0.45 * float(np.arccos(cos_a)) / np.pi

            if angle_factor < 1e-4:
                new_pts.append(curr)
                continue

            cut_in  = min(angle_factor * len_in,  max_radius, len_in  * 0.45)
            cut_out = min(angle_factor * len_out, max_radius, len_out * 0.45)

            new_pts.append(curr - (v_in  / len_in)  * cut_in)
            new_pts.append(curr + (v_out / len_out) * cut_out)

        pts = np.array(new_pts, dtype=float)

    return pts


def remove_spike_vertices(polygon_points: np.ndarray, threshold_deg: float = 168.0) -> np.ndarray:
    """Remove vertices forming near-180° hairpin turns from a closed polygon."""
    points = list(np.asarray(polygon_points, dtype=float))
    spike_cos_threshold = np.cos(np.deg2rad(threshold_deg))

    changed = True
    while changed and len(points) >= 4:
        changed = False
        spike_indices = []
        n = len(points)
        for i in range(n):
            incoming = np.array(points[i],            dtype=float) - np.array(points[(i - 1) % n], dtype=float)
            outgoing = np.array(points[(i + 1) % n], dtype=float) - np.array(points[i],            dtype=float)
            in_len, out_len = np.linalg.norm(incoming), np.linalg.norm(outgoing)
            if in_len < 1e-9 or out_len < 1e-9:
                spike_indices.append(i)
                continue
            if float(np.dot(incoming / in_len, outgoing / out_len)) < spike_cos_threshold:
                spike_indices.append(i)
        if spike_indices:
            for idx in reversed(spike_indices):
                points.pop(idx)
            changed = True

    return np.array(points, dtype=float)
