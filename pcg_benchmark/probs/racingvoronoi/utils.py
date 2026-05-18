from __future__ import annotations

import numpy as np
import scipy.spatial


def create_voronoi_grid(seed_points, width, height, voronoi_seed):
    """Build a Voronoi diagram from seed_points and return its finite edges.

    Each Voronoi edge is the boundary between two neighbouring cells.
    Semi-infinite edges (one endpoint at infinity) are discarded.

    Returns:
        voronoi_vertices    (V, 2) float        - coordinates of every finite vertex.
        all_edges           (E, 2) int          - vertex-index pairs for every finite edge.
        all_edge_cell_pairs (E, 2) int          - the two cell indices sharing each edge.
        cell_neighbours     list[list[int]]     - for each cell, sorted neighbour cell indices.
    """
    seed_array = np.asarray(seed_points, dtype=float)
    if seed_array.ndim == 1:
        seed_array = seed_array.reshape(-1, 2)
    num_cells = len(seed_array)

    try:
        voronoi = scipy.spatial.Voronoi(seed_array)
    except scipy.spatial.QhullError:
        #Tiny random jitter to de-align collinear points
        rng = np.random.default_rng(voronoi_seed)
        jittered = seed_array + rng.normal(scale=1e-6, size=seed_array.shape)
        voronoi = scipy.spatial.Voronoi(jittered)

    voronoi_vertices = np.asarray(voronoi.vertices, dtype=float)

    #remove duplicates using dict keyed by (lower_idx, higher_idx).
    edges_dict: dict[tuple[int, int], tuple[int, int]] = {}
    for (vertex_a_idx, vertex_b_idx), (cell_left, cell_right) in zip(
        voronoi.ridge_vertices, voronoi.ridge_points
    ):
        if vertex_a_idx < 0 or vertex_b_idx < 0:
            continue
        edge_key = (min(vertex_a_idx, vertex_b_idx), max(vertex_a_idx, vertex_b_idx))
        edges_dict[edge_key] = (int(cell_left), int(cell_right))

    sorted_edge_keys = sorted(edges_dict)
    if sorted_edge_keys:
        all_edges           = np.array(sorted_edge_keys,                                dtype=int)
        all_edge_cell_pairs = np.array([edges_dict[k] for k in sorted_edge_keys],       dtype=int)
    else:
        all_edges           = np.zeros((0, 2), dtype=int)
        all_edge_cell_pairs = np.zeros((0, 2), dtype=int)

    neighbour_sets: list[set[int]] = [set() for _ in range(num_cells)]
    for cell_a, cell_b in all_edge_cell_pairs:
        neighbour_sets[cell_a].add(cell_b)
        neighbour_sets[cell_b].add(cell_a)
    cell_neighbours = [sorted(neighbours) for neighbours in neighbour_sets]

    return voronoi_vertices, all_edges, all_edge_cell_pairs, cell_neighbours


def lloyd_relaxation(cell_sites: np.ndarray, guard_points: np.ndarray, num_iterations: int) -> np.ndarray:
    """Run Lloyd's Voronoi relaxation on cell_sites, guard points are not affected."""
    num_cells = len(cell_sites)
    relaxed_sites = cell_sites.copy()

    for _ in range(num_iterations):
        voronoi = scipy.spatial.Voronoi(np.vstack([relaxed_sites, guard_points]))
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
        relaxed_sites = new_sites

    return relaxed_sites


def build_voronoi_cell_graph(num_cells: int, width: float, height: float, voronoi_seed: int, lloyd_iterations: int = 2) -> dict:
    """Generate a complete Voronoi grid for racetrack cell selection.

    Places num_cells seed points randomly inside the inner 80% of the bounding
    box, 8 guard points just outside each edge of the box.  
    Cells with any edges that go out of bounds are discarded from selection but still rendered.

    Returns a dict with:
        cell_sites          (N, 2) float     - seed positions of the N cells.
        voronoi_vertices    (V, 2) float     - all Voronoi vertex positions.
        all_edges_full      (E, 2) int       - every finite edge incl. guard-cell edges.
        all_edge_pairs_full (E, 2) int       - cell-index pairs for all_edges_full.
        cell_neighbours     list[list[int]]  - real-cell neighbour indices per cell.
        boundary_cells      set[int]         - cells excluded from selection.
    """
    W, H = float(width), float(height)

    margin = float(min(W, H)) * 0.025
    cell_sites = np.random.default_rng(voronoi_seed).uniform(
        low=[margin, margin],
        high=[W - margin, H - margin],
        size=(num_cells, 2),
    ).astype(float)

    # Guard point creation, small offset to keep most of the geometry in bounds
    guard_offset = float(min(W, H)) * 0.05
    guard_points = np.array([
        [-guard_offset,     -guard_offset    ],
        [W / 2,             -guard_offset    ],
        [W + guard_offset,  -guard_offset    ],
        [-guard_offset,      H / 2           ],
        [W + guard_offset,   H / 2           ],
        [-guard_offset,      H + guard_offset],
        [W / 2,              H + guard_offset],
        [W + guard_offset,   H + guard_offset],
    ], dtype=float)

    #optional Lloyd relaxation to avoid extremely small cells
    cell_sites = lloyd_relaxation(cell_sites, guard_points, lloyd_iterations)
    
    
    all_sites = np.vstack([cell_sites, guard_points])

    voronoi_vertices, all_edges_full, all_edge_pairs_full, cell_neighbours_all = (
        create_voronoi_grid(all_sites, W, H, voronoi_seed)
    )

    voronoi_vertices    = np.asarray(voronoi_vertices,    dtype=float)
    all_edges_full      = np.asarray(all_edges_full,      dtype=int)
    all_edge_pairs_full = np.asarray(all_edge_pairs_full, dtype=int)

    cell_neighbours = [
        [nb for nb in neighbours if nb < num_cells]
        for neighbours in cell_neighbours_all[:num_cells]
    ]

    # Mark boundary cells (with edges out of bounds) as ineligible for selection.
    boundary_cells: set[int] = set()
    for (vert_a_idx, vert_b_idx), (cell_a, cell_b) in zip(all_edges_full, all_edge_pairs_full):
        cell_a, cell_b = int(cell_a), int(cell_b)
        is_guard_edge = (cell_a < num_cells) != (cell_b < num_cells)
        if not is_guard_edge:
            continue
        real_cell = cell_a if cell_a < num_cells else cell_b
        for vert_idx in (int(vert_a_idx), int(vert_b_idx)):
            x, y = voronoi_vertices[vert_idx]
            if x < 0.0 or x > W or y < 0.0 or y > H:
                boundary_cells.add(real_cell)
                break

    return {
        'cell_sites':          cell_sites,
        'voronoi_vertices':    voronoi_vertices,
        'all_edges_full':      all_edges_full,
        'all_edge_pairs_full': all_edge_pairs_full,
        'cell_neighbours':     cell_neighbours,
        'boundary_cells':      boundary_cells,
    }


def find_boundary_cycle(boundary_edges: np.ndarray, voronoi_vertices: np.ndarray) -> list[int] | None:
    """Find the longest closed loop in a set of Voronoi boundary edges.
    This is to use the outside boundary in cases where the Voronoi cells form a loop.
    """
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

    voronoi_vertices = np.asarray(voronoi_vertices, dtype=float)
    visited:          set[int]       = set()
    best_cycle:       list[int] | None = None
    longest_perimeter: float          = -1.0

    for start_vertex in list(adjacency):
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

        if not component or any(len(adjacency.get(v, [])) != 2 for v in component):
            continue

        first_vertex = next(iter(component))
        prev_vertex, current_vertex, cycle = None, first_vertex, []
        for _ in range(len(component) + 2):
            cycle.append(current_vertex)
            neighbours   = adjacency[current_vertex]
            next_vertex  = neighbours[0] if prev_vertex is None or neighbours[0] != prev_vertex else neighbours[1]
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


def remove_spike_vertices(polygon_points: np.ndarray, threshold_deg: float = 168.0) -> np.ndarray:
    """Remove vertices that cause extreme hairpin turns in a closed polygon.

    A spike occurs when three consecutive vertices are nearly collinear, forming
    a turn angle close to 180°.  On a racetrack this looks like a dead-end
    needle that the car cannot navigate.  Vertices are removed iteratively
    until no remaining turn exceeds threshold_deg or fewer than four vertices
    remain.  The threshold cosine is negative (cos(168°) ≈ -0.978), so only
    very straight-through vertices are removed and genuine corners are kept.
    """
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
            incoming_len = np.linalg.norm(incoming)
            outgoing_len = np.linalg.norm(outgoing)
            if incoming_len < 1e-9 or outgoing_len < 1e-9:
                spike_indices.append(i)
                continue
            cos_angle = float(np.dot(incoming / incoming_len, outgoing / outgoing_len))
            if cos_angle < spike_cos_threshold:
                spike_indices.append(i)
        if spike_indices:
            for idx in reversed(sorted(set(spike_indices))):
                points.pop(idx)
            changed = True

    return np.array(points, dtype=float)
