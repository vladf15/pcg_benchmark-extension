from __future__ import annotations

from pcg_benchmark.probs.racing.engine import CarPhysicsEngine
from pcg_benchmark.probs.racing.agent import SteeringAgent
from pcg_benchmark.probs.racing.problem import (
    PX_PER_M,
    RacingProblem,
    _rotated_rect,
)
from pcg_benchmark.probs.racing.utils import count_self_intersections
from pcg_benchmark.probs.racingvoronoi.utils import (
    build_voronoi_cell_graph,
    find_boundary_cycle,
    remove_spike_vertices,
)
from pcg_benchmark.spaces import ArraySpace, FloatSpace, DictionarySpace
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class RacingVoronoiProblem(RacingProblem):

    def __init__(self, **kwargs):
        kwargs.setdefault('num_points', 15)
        kwargs.setdefault('max_steps', 2000)

        num_cells = int(kwargs.pop('num_cells', 50))
        num_selected = int(kwargs.pop('num_selected_cells', 15))

        super().__init__(**kwargs)

        self._num_cells = num_cells
        self._num_selected_cells = num_selected

        self._content_space = DictionarySpace({
            "cell_scores": ArraySpace((self._num_cells,), FloatSpace(0.0, 1.0)),
        })

        self._fixed_cell_sites = None
        self._fixed_boundary_cells = None
        self._fixed_voronoi_vertices = None
        self._fixed_voronoi_all_edges_full = None
        self._fixed_voronoi_all_pairs_full = None
        self._fixed_cell_adjacency = None
        self._last_selected_cells = None

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _densify_polyline(points: np.ndarray, *, max_step: float, closed: bool) -> np.ndarray:
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        if len(points) < 2:
            return points
        out = [points[0]]
        n = len(points)
        for i in range(n if closed else n - 1):
            a = points[i]
            b = points[(i + 1) % n]
            d = float(np.linalg.norm(b - a))
            if d <= 1e-12:
                continue
            steps = max(1, int(np.ceil(d / max(float(max_step), 1e-9))))
            for s in range(1, steps + 1):
                t = s / steps
                out.append(a * (1.0 - t) + b * t)
        out_np = np.asarray(out, dtype=float)
        if closed and len(out_np) >= 2 and not np.allclose(out_np[0], out_np[-1], atol=1e-9):
            out_np = np.vstack([out_np, out_np[0]])
        return out_np

    # ------------------------------------------------------------------
    # Fixed Voronoi grid
    # ------------------------------------------------------------------

    def _ensure_fixed_cell_graph(self):
        if self._fixed_cell_sites is not None:
            return
        graph = build_voronoi_cell_graph(self._num_cells, self._width, self._height)
        self._fixed_cell_sites             = graph['cell_sites']
        self._fixed_voronoi_vertices       = graph['voronoi_vertices']
        self._fixed_voronoi_all_edges_full = graph['all_edges_full']
        self._fixed_voronoi_all_pairs_full = graph['all_edge_pairs_full']
        self._fixed_cell_adjacency         = graph['cell_neighbours']
        self._fixed_boundary_cells         = graph['boundary_cells']

    # ------------------------------------------------------------------
    # Cell selection
    # ------------------------------------------------------------------

    def _decode_selected_cells(self, cell_scores: np.ndarray) -> list[int]:
        self._ensure_fixed_cell_graph()
        scores = np.asarray(cell_scores, dtype=float).ravel()
        if scores.size != self._num_cells:
            raise ValueError(f"cell_scores must have length {self._num_cells}, got {scores.size}")

        boundary = self._fixed_boundary_cells
        selectable = [i for i in range(self._num_cells) if i not in boundary]

        k = min(int(self._num_selected_cells), len(selectable))
        start = max(selectable, key=lambda i: (float(scores[i]), -i))
        selected: set[int] = {start}
        frontier: set[int] = {n for n in self._fixed_cell_adjacency[start] if n not in boundary}

        while len(selected) < k:
            candidates = [c for c in frontier if c not in selected]
            if not candidates:
                remaining = sorted(
                    (i for i in selectable if i not in selected),
                    key=lambda i: (-float(scores[i]), i),
                )
                selected.update(remaining[: k - len(selected)])
                break
            nxt = max(candidates, key=lambda i: (float(scores[i]), -i))
            selected.add(nxt)
            frontier.update(n for n in self._fixed_cell_adjacency[nxt] if n not in boundary)

        return sorted(selected)

    # ------------------------------------------------------------------
    # Content extraction
    # ------------------------------------------------------------------

    def _extract_content(self, content):
        if content is None or (isinstance(content, dict) and 'track_points' in content):
            return super()._extract_content(content)

        if isinstance(content, dict) and 'cell_scores' in content:
            self._ensure_fixed_cell_graph()
            selected = self._decode_selected_cells(content['cell_scores'])
            self._last_selected_cells = selected

            inside = set(selected)
            # Use full pairs (including guard-cell edges) so the boundary cycle is
            # topologically complete. Guard cells are never in `inside`, so XOR
            # correctly marks real-guard edges as boundary edges too.
            pairs = self._fixed_voronoi_all_pairs_full
            edges = self._fixed_voronoi_all_edges_full
            if len(pairs) > 0:
                mask = np.array(
                    [(int(a) in inside) ^ (int(b) in inside)
                     for a, b in pairs],
                    dtype=bool,
                )
                boundary_edges = edges[mask]
            else:
                boundary_edges = np.zeros((0, 2), dtype=int)

            cycle = find_boundary_cycle(boundary_edges, self._fixed_voronoi_vertices)
            if cycle is None:
                return super()._extract_content(None)

            pts = self._fixed_voronoi_vertices[np.array(cycle, dtype=int)]
            pts = remove_spike_vertices(pts)
            if len(pts) < 3:
                return super()._extract_content(None)
            return np.asarray(pts, dtype=float)

        self._last_selected_cells = None
        return super()._extract_content(content)

    # ------------------------------------------------------------------
    # Simulation interface
    # ------------------------------------------------------------------

    def reset(self, track_points=None):
        track_points = self._set_track_cache(track_points)
        self._curve_points = self._densify_polyline(
            track_points,
            max_step=float(self._track_width) * 0.35,
            closed=self._closed_loop,
        )

        if len(self._curve_points) >= 2:
            dx = float(self._curve_points[1][0] - self._curve_points[0][0])
            dy = float(self._curve_points[1][1] - self._curve_points[0][1])
            start_angle = float(np.arctan2(dy, dx))
        else:
            start_angle = 0.0

        if self._closed_loop and len(self._curve_points) > 0:
            self._final_target = np.asarray(self._curve_points[0], dtype=float)

        self._steps_since_reset = 0
        if not hasattr(self, '_engine') or self._engine is None:
            self._engine = CarPhysicsEngine(
                start_position=self._curve_points[0], start_angle=start_angle
            )
        else:
            self._engine.start_position = np.asarray(self._curve_points[0], dtype=float)
            self._engine.start_angle = start_angle

        if not hasattr(self, '_agent') or self._agent is None:
            self._agent = SteeringAgent(
                self._curve_points,
                track_width=self._track_width,
                enable_wander=self._enable_wander,
            )
        else:
            self._agent.curve_points = self._curve_points

        state = self._engine.reset()
        self._car_state = state
        return state

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def info(self, content, trajectory=None, use_cache=True):
        if use_cache and trajectory is None and isinstance(content, dict) and 'cell_scores' in content:
            scores_key = tuple(np.asarray(content['cell_scores'], dtype=float).ravel().tolist())
            cached = self._info_cache.get(scores_key)
            if cached is not None:
                return cached
        else:
            scores_key = None

        track_points = self._extract_content(content)
        track_points = self._normalize_track_points(track_points)

        if len(track_points) < 2:
            return {
                'num_points': self.num_points,
                'total_length': 0.0, 'avg_length': 0.0,
                'max_length': 0.0, 'min_length': 0.0,
                'avg_turn': 0.0, 'max_turn': 0.0, 'min_turn': 0.0,
                'steps': 0, 'finished': False,
                'track_points': track_points,
                'trajectory_end': track_points[0] if len(track_points) else None,
                'curve_points': track_points,
                'selected_cells': self._last_selected_cells,
            }

        cache_key = self._get_info_cache_key(track_points)
        if use_cache and trajectory is None and cache_key in self._info_cache:
            return self._info_cache[cache_key]

        diffs = track_points[1:] - track_points[:-1]
        seg_lens = np.linalg.norm(diffs, axis=1)
        turn_angles = self._compute_turn_angles(track_points)

        curve_points = self._densify_polyline(
            track_points,
            max_step=float(self._track_width) * 0.35,
            closed=self._closed_loop,
        )

        if trajectory is None:
            steps, finished, end_xy = self._get_cached_simulation_summary(
                track_points, curve_points=curve_points
            )
            trajectory_end = end_xy
        else:
            steps = len(trajectory)
            trajectory_end = trajectory[-1][:2] if trajectory else None
            finished = (
                steps < self._default_max_steps
                and self._is_finished(trajectory_end, steps_len=steps)
            )

        result = {
            'num_points': self.num_points,
            'total_length': float(np.sum(seg_lens)),
            'avg_length': float(np.mean(seg_lens)),
            'max_length': float(np.max(seg_lens)),
            'min_length': float(np.min(seg_lens)),
            'avg_turn': float(np.mean(turn_angles)) if turn_angles.size > 0 else 0.0,
            'max_turn': float(np.max(turn_angles)) if turn_angles.size > 0 else 0.0,
            'min_turn': float(np.min(turn_angles)) if turn_angles.size > 0 else 0.0,
            'steps': steps,
            'finished': finished,
            'track_points': track_points,
            'trajectory_end': trajectory_end,
            'curve_points': curve_points,
            'selected_cells': self._last_selected_cells,
        }
        if use_cache:
            self._info_cache[cache_key] = result
            if scores_key is not None:
                self._info_cache[scores_key] = result
        return result

    # ------------------------------------------------------------------
    # Quality
    # ------------------------------------------------------------------

    def _compute_cluster_shape_score(self, selected_cells: list[int]) -> float:
        """Score cluster shape: reward balanced aspect ratio and solidity."""
        self._ensure_fixed_cell_graph()
        if len(selected_cells) < 3:
            return 0.5
        sites = self._fixed_cell_sites[np.array(selected_cells, dtype=int)]
        bbox_w = float(np.max(sites[:, 0]) - np.min(sites[:, 0]))
        bbox_h = float(np.max(sites[:, 1]) - np.min(sites[:, 1]))
        longer = max(bbox_w, bbox_h, 1e-6)
        aspect_ratio = min(bbox_w, bbox_h) / longer  # 0 = line, 1 = square

        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(sites)
            # In 2D, hull.volume is the area
            solidity = float(hull.volume) / max(bbox_w * bbox_h, 1e-6)
        except Exception:
            solidity = 0.5

        # Reward aspect ~0.5 (not a line, not a square) and solidity ~0.6
        aspect_score = float(np.exp(-((aspect_ratio - 0.5) ** 2) / (2 * 0.2 ** 2)))
        solidity_score = float(np.exp(-((solidity - 0.6) ** 2) / (2 * 0.2 ** 2)))
        return (aspect_score + solidity_score) / 2.0

    def quality(self, info):
        track_points = info.get('track_points')
        if track_points is None:
            return 0.0
        points = np.asarray(track_points, dtype=float)
        if len(points) == 0:
            return 0.0

        finished = info.get('finished', False)
        curve_points = info.get('curve_points')
        if curve_points is None:
            curve_points = self._densify_polyline(
                points, max_step=float(self._track_width) * 0.35, closed=self._closed_loop
            )
        curve_points = np.asarray(curve_points, dtype=float)

        # Out-of-bounds penalty
        margin = float(self._track_width) * 0.5 + 2.0
        oob_violation = 0.0
        if len(curve_points) > 0:
            oob_violation = max(
                0.0,
                margin - float(np.min(curve_points[:, 0])),
                float(np.max(curve_points[:, 0])) - (self._width - margin),
                margin - float(np.min(curve_points[:, 1])),
                float(np.max(curve_points[:, 1])) - (self._height - margin),
            )
        oob_penalty = float(np.exp(-oob_violation / max(1e-6, float(self._track_width) * 0.5)))

        # Completion
        trajectory_end = info.get('trajectory_end')
        if trajectory_end is None and len(points) > 0:
            trajectory_end = points[-1]
        if trajectory_end is None:
            return 0.0
        final_point = curve_points[0] if (self._closed_loop and len(curve_points) > 0) else points[-1]
        dist_to_goal = float(np.linalg.norm(final_point - np.asarray(trajectory_end, dtype=float)))
        curve_len = (
            float(np.sum(np.linalg.norm(curve_points[1:] - curve_points[:-1], axis=1)))
            if len(curve_points) > 1 else 1.0
        )
        completion_pct = 1.0 if finished else (1.0 - min(dist_to_goal / (curve_len + 1e-6), 1.0))

        # Curvature — measured at actual Voronoi corners, not diluted by interpolation
        turn_angles = self._compute_turn_angles(points)
        avg_curvature = float(np.mean(turn_angles)) if turn_angles.size > 0 else 0.0
        curvature_variety = float(np.std(turn_angles)) if turn_angles.size > 0 else 0.0

        min_curvature = np.deg2rad(15)
        curvature_penalty = (avg_curvature / min_curvature) if avg_curvature < min_curvature else 1.0

        ideal_curvature = np.deg2rad(35)
        curvature_score = (
            float(np.exp(-((avg_curvature - ideal_curvature) ** 2) / (2 * (ideal_curvature / 2) ** 2)))
            if avg_curvature > 0 else 0.0
        )

        # Voronoi polygons naturally have sharper corners — threshold at 90 deg
        sharp_turns = int(np.sum(turn_angles > np.deg2rad(90)))
        sharp_turn_penalty = 1.0 - min((sharp_turns - 4) / 4.0, 1.0) if sharp_turns > 4 else 1.0
        max_turn = float(np.max(turn_angles)) if turn_angles.size > 0 else 0.0
        excess_turn = max(0.0, max_turn - np.deg2rad(150))
        turn_penalty = float(np.exp(-excess_turn / max(1e-6, np.deg2rad(15))))

        ideal_variety = np.deg2rad(15)
        variety_score = (
            float(np.exp(-((curvature_variety - ideal_variety) ** 2) / (2 * (ideal_variety / 2) ** 2)))
            if avg_curvature > 0 else 0.0
        )

        # Segment variance and total length
        if len(points) > 1:
            seg_lens = np.linalg.norm(points[1:] - points[:-1], axis=1)
            mean_len = float(np.mean(seg_lens))
            std_len = float(np.std(seg_lens))
            variance_penalty = float(np.exp(-std_len / (mean_len + 1e-6)))
            total_length = float(np.sum(seg_lens))
            min_len, max_len = 1500.0, 6500.0
            if total_length < min_len:
                length_score = total_length / min_len
            elif total_length > max_len:
                length_score = max_len / total_length
            else:
                length_score = 1.0
        else:
            variance_penalty = 1.0
            length_score = 0.0

        # For now the geometry penalties use the sparse polygon vertices, not the interpolated curve.
        center_inters = int(count_self_intersections(points, closed=self._closed_loop))
        #Self-intersection penalty is less strict here since Voronoi tracks are more prone to tight turns
        geom_penalty = float(np.exp(-1.5 * float(min(center_inters, 50))))

        segment_proximity_penalty = self._compute_segment_proximity_penalty(
            points, min_distance=self._track_width * 2.0
        )

        # Time score
        steps = info.get('steps', 0)
        min_steps = int(30 * max(len(points), 1))
        max_steps_for_scoring = int(300 * max(len(points), 1))
        if not finished:
            time_score = 0.0
        elif steps < min_steps:
            time_score = max(0.0, 1.0 - (min_steps - steps) / min_steps)
        elif steps > max_steps_for_scoring:
            time_score = max(0.0, 1.0 - (steps - max_steps_for_scoring) / max_steps_for_scoring)
        else:
            time_score = 1.0

        # Cluster shape
        selected = info.get('selected_cells')
        shape_score = self._compute_cluster_shape_score(selected) if selected else 0.5

        quality = (
            0.25 * completion_pct +
            0.15 * segment_proximity_penalty +
            0.15 * shape_score +
            0.12 * curvature_score +
            0.10 * curvature_penalty +
            0.10 * sharp_turn_penalty +
            0.08 * variance_penalty +
            0.03 * variety_score +
            0.01 * length_score +
            0.01 * time_score
        )
        return float(quality) * oob_penalty * turn_penalty * geom_penalty

    # ------------------------------------------------------------------
    # Diversity & Controlability
    # ------------------------------------------------------------------

    def diversity(self, info1, info2):
        cells1 = set(info1.get('selected_cells') or [])
        cells2 = set(info2.get('selected_cells') or [])
        if not cells1 and not cells2:
            return 0.0
        union = len(cells1 | cells2)
        intersection = len(cells1 & cells2)
        return 1.0 - intersection / max(union, 1)

    def controlability(self, info, control):
        return 1.0

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(
        self,
        content=None,
        frame_sampling=2,
        skip=None,
        *,
        fast=True,
        show_hud=True,
        progress=True,
        progress_desc=None,
        render_scale=1.0,
    ):
        if skip is None:
            skip = self._skip_render
        if skip:
            return []

        track_points = self._extract_content(content)
        trajectory = self._get_cached_trajectory(track_points)

        try:
            frame_sampling = int(frame_sampling)
        except Exception:
            frame_sampling = 5
        if frame_sampling < 1:
            frame_sampling = 1

        track_points_np = self._normalize_track_points(track_points)
        curve_np = np.asarray(
            self._densify_polyline(
                track_points_np,
                max_step=float(self._track_width) * 0.35,
                closed=self._closed_loop,
            ),
            dtype=float,
        )

        try:
            render_scale = float(render_scale)
        except Exception:
            render_scale = 1.0
        if render_scale <= 1e-9:
            render_scale = 1.0

        scale = float(PX_PER_M) * render_scale
        img_w = int(round(float(self._width) * scale))
        img_h = int(round(float(self._height) * scale))
        curve_px = curve_np * scale

        grass_color = (34, 139, 34)
        edge_color = (10, 10, 10)
        road_color = (215, 215, 215)
        centerline_color = (120, 120, 120)

        car_length = 5.0 * scale
        car_width = 2.0 * scale
        scaled_curve = [(int(round(x)), int(round(y))) for x, y in curve_px]
        half_width = float(self._track_width) * scale * 0.5

        left_edge_f: list = []
        right_edge_f: list = []
        n = len(curve_px)
        if n >= 2:
            has_dup = self._closed_loop and n >= 3 and np.allclose(curve_px[0], curve_px[-1], atol=1e-9)
            base = curve_px[:-1] if has_dup else curve_px
            m = len(base)
            for j in range(m):
                if self._closed_loop and m >= 3:
                    dir_prev = base[j] - base[(j - 1) % m]
                    dir_next = base[(j + 1) % m] - base[j]
                else:
                    dir_prev = base[1] - base[0] if j == 0 else base[j] - base[j - 1]
                    dir_next = base[j] - base[j - 1] if j == m - 1 else base[j + 1] - base[j]
                avg_dir = dir_prev + dir_next
                norm = float(np.linalg.norm(avg_dir))
                perp = np.array([-avg_dir[1], avg_dir[0]], dtype=float) / norm if norm > 0 else np.zeros(2)
                left_edge_f.append(tuple(base[j] + perp * half_width))
                right_edge_f.append(tuple(base[j] - perp * half_width))
            if has_dup and left_edge_f:
                left_edge_f.append(left_edge_f[0])
                right_edge_f.append(right_edge_f[0])
        track_background = Image.new("RGB", (img_w, img_h), grass_color)
        bg_draw = ImageDraw.Draw(track_background)

        left_edge  = [(int(round(x)), int(round(y))) for x, y in left_edge_f]
        right_edge = [(int(round(x)), int(round(y))) for x, y in right_edge_f]
        if len(left_edge) > 1:
            bg_draw.line(left_edge, fill=edge_color, width=4)
        if len(right_edge) > 1:
            bg_draw.line(right_edge, fill=edge_color, width=4)

        for j in range(len(left_edge_f) - 1):
            lj = (int(round(left_edge_f[j][0])),      int(round(left_edge_f[j][1])))
            lk = (int(round(left_edge_f[j + 1][0])),  int(round(left_edge_f[j + 1][1])))
            rj = (int(round(right_edge_f[j][0])),      int(round(right_edge_f[j][1])))
            rk = (int(round(right_edge_f[j + 1][0])), int(round(right_edge_f[j + 1][1])))
            bg_draw.polygon([lj, lk, rk, rj], fill=road_color)

        # Voronoi grid overlay — all finite edges in gray (includes out-of-bounds)
        self._ensure_fixed_cell_graph()
        vor_v = self._fixed_voronoi_vertices * scale
        for u, v in self._fixed_voronoi_all_edges_full:
            pu, pv = vor_v[int(u)], vor_v[int(v)]
            bg_draw.line(
                [(int(round(pu[0])), int(round(pu[1]))), (int(round(pv[0])), int(round(pv[1])))],
                fill=(70, 70, 70), width=1,
            )

        if len(scaled_curve) > 1:
            bg_draw.line(scaled_curve, fill=centerline_color, width=2)

        agent = None
        font = None
        dt = 0.1
        mass = 1350.0
        font_size = 24
        if show_hud:
            agent = SteeringAgent(curve_np, track_width=float(self._track_width), enable_wander=bool(self._enable_wander))
            agent.reset()
            try:
                dt = float(getattr(getattr(self, '_engine', None), 'time_step', 0.1))
            except Exception:
                dt = 0.1
            if dt <= 1e-9:
                dt = 0.1
            try:
                mass = float(getattr(getattr(self, '_engine', None), 'mass', 1350.0))
            except Exception:
                mass = 1350.0
            font_size = max(10, int(round(24.0 * render_scale)))
            for font_path in ("DejaVuSans.ttf", r"C:\Windows\Fonts\segoeui.ttf", "arial.ttf"):
                try:
                    font = ImageFont.truetype(font_path, font_size)
                    break
                except Exception:
                    pass
            if font is None:
                font = ImageFont.load_default()

        reuse_canvas = bool(fast)
        img = track_background.copy() if reuse_canvas else None
        prev_bbox = None
        dirty_pad = max(2, int(round(12.0 * render_scale)))
        frames = []

        iterator = enumerate(trajectory)
        if progress:
            try:
                from tqdm import tqdm  # type: ignore
                desc = progress_desc or 'Rendering voronoi frames'
                iterator = enumerate(tqdm(trajectory, total=len(trajectory), desc=desc, leave=False, dynamic_ncols=True))
            except Exception:
                pass

        prev_angle = None
        for i, state in iterator:
            angle = state[2] if len(state) > 2 else 0.0
            action = None
            lookahead = None
            yaw_rate = 0.0
            if show_hud and agent is not None:
                try:
                    action = agent.act(state)
                except Exception:
                    action = {'steering': 0.0, 'throttle': 0.0}
                lookahead = getattr(agent, 'last_lookahead_point', None)
                if prev_angle is not None:
                    da = (float(angle) - float(prev_angle) + np.pi) % (2.0 * np.pi) - np.pi
                    yaw_rate = float(da / dt)
                prev_angle = float(angle)

            if i % frame_sampling != 0:
                continue

            if reuse_canvas:
                if prev_bbox is not None:
                    try:
                        img.paste(track_background.crop(prev_bbox), prev_bbox)
                    except Exception:
                        pass
            else:
                img = track_background.copy()

            draw = ImageDraw.Draw(img)
            car_x = float(state[0]) * scale
            car_y = float(state[1]) * scale
            cos_a = float(np.cos(angle))
            sin_a = float(np.sin(angle))
            fwd = (cos_a, sin_a)
            rgt = (-sin_a, cos_a)

            dx = car_length / 2.0
            dy = car_width / 2.0
            wl = car_length * 0.10
            ww = car_width * 0.11
            wb = car_length * 0.34
            wt = car_width * 0.44

            wheel_pts = []
            for fs, rs in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                wx = car_x + fwd[0] * wb * fs + rgt[0] * wt * rs
                wy = car_y + fwd[1] * wb * fs + rgt[1] * wt * rs
                wheel = _rotated_rect(wx, wy, fwd, rgt, wl, ww)
                draw.polygon(wheel, fill=(25, 25, 25), outline=(0, 0, 0))
                wheel_pts.extend(wheel)

            corners = [
                (car_x + cos_a * dx - sin_a * dy, car_y + sin_a * dx + cos_a * dy),
                (car_x + cos_a * dx + sin_a * dy, car_y + sin_a * dx - cos_a * dy),
                (car_x - cos_a * dx + sin_a * dy, car_y - sin_a * dx - cos_a * dy),
                (car_x - cos_a * dx - sin_a * dy, car_y - sin_a * dx + cos_a * dy),
            ]
            draw.polygon(corners, fill=(255, 0, 0), outline=(0, 0, 0))
            front_x = car_x + cos_a * dx
            front_y = car_y + sin_a * dx
            car_line_w = max(1, int(round(3.0 * render_scale)))
            draw.line([(car_x, car_y), (front_x, front_y)], fill=(0, 0, 255), width=car_line_w)

            bbox_xs = [p[0] for p in corners] + [car_x, front_x]
            bbox_ys = [p[1] for p in corners] + [car_y, front_y]
            if wheel_pts:
                bbox_xs.extend(p[0] for p in wheel_pts)
                bbox_ys.extend(p[1] for p in wheel_pts)

            if lookahead is not None:
                try:
                    la_x = float(lookahead[0]) * scale
                    la_y = float(lookahead[1]) * scale
                    rr = max(2, int(round(6.0 * render_scale)))
                    draw.ellipse([(la_x - rr, la_y - rr), (la_x + rr, la_y + rr)],
                                 outline=(0, 255, 255), width=max(1, int(round(3.0 * render_scale))))
                    draw.line([(car_x, car_y), (la_x, la_y)], fill=(0, 200, 200),
                              width=max(1, int(round(2.0 * render_scale))))
                    bbox_xs.extend([la_x - rr, la_x + rr, car_x, la_x])
                    bbox_ys.extend([la_y - rr, la_y + rr, car_y, la_y])
                except Exception:
                    pass

            if show_hud and font is not None:
                v = float(state[3]) if len(state) > 3 else 0.0
                steer_ang = float(state[4]) * (180.0 / np.pi) if len(state) > 4 else 0.0
                steer_cmd = float(action.get('steering', 0.0)) if isinstance(action, dict) else 0.0
                throttle = float(action.get('throttle', 0.0)) if isinstance(action, dict) else 0.0
                a_lat = float(v * yaw_rate)
                lines = [
                    f"v: {v:5.1f} m/s  ({v * 3.6:5.0f} km/h)",
                    f"steer cmd: {steer_cmd:+.2f}   steer ang: {steer_ang:+5.1f} deg",
                    f"throttle: {throttle:+.2f}",
                    f"yaw rate: {yaw_rate:+6.2f} rad/s   a_lat: {a_lat:+6.2f} m/s^2",
                    f"Fy est: {mass * a_lat / 1000.0:+7.2f} kN",
                ]
                pad = max(2, int(round(6.0 * render_scale)))
                x0, y0 = pad, pad
                line_h = max(1, int(round(font_size * 1.25)))
                max_w = 0
                for txt in lines:
                    try:
                        bb = draw.textbbox((0, 0), txt, font=font)
                        w = int(bb[2] - bb[0])
                    except Exception:
                        try:
                            w, _ = draw.textsize(txt, font=font)
                            w = int(w)
                        except Exception:
                            w = int(len(txt) * font_size * 0.6)
                    if w > max_w:
                        max_w = w
                box_w = max_w + pad * 2
                box_h = pad * 2 + line_h * len(lines)
                draw.rectangle([(x0, y0), (x0 + box_w, y0 + box_h)],
                               fill=(255, 255, 255), outline=(0, 0, 0),
                               width=max(1, int(round(render_scale))))
                ty = y0 + pad
                for txt in lines:
                    draw.text((x0 + pad, ty), txt, fill=(0, 0, 0), font=font)
                    ty += line_h
                bbox_xs.extend([x0, x0 + box_w])
                bbox_ys.extend([y0, y0 + box_h])

            if reuse_canvas:
                try:
                    prev_bbox = (
                        int(max(0, min(bbox_xs) - dirty_pad)),
                        int(max(0, min(bbox_ys) - dirty_pad)),
                        int(min(img_w, max(bbox_xs) + dirty_pad)),
                        int(min(img_h, max(bbox_ys) + dirty_pad)),
                    )
                except Exception:
                    prev_bbox = None

            frames.append(img.copy() if reuse_canvas else img)

        return frames
