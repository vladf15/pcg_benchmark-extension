"""view_track.py — interactive track viewer.

Run with:  python view_track.py

Keyboard shortcuts inside the pygame window:
  Left / Right   previous / next iteration  (best chromosome auto-selected)
  Up / Down      previous / next chromosome within current iteration
  Space          restart current chromosome
  Escape         close the window

Driver: the scripted SteeringAgent by default. Set "RL model" in the
control panel to a trained policy (model_training/runs/<name>/best/*.zip)
to watch that instead; the active driver is named in the overlay. An RL
policy runs through rl_agent_adapter, which owns its own engine instance
because the policy reads an 18-float observation rather than the benchmark
engine's 5-float state.  Both drivers now use the same physics.
"""
from __future__ import annotations

import ctypes
import json
import multiprocessing
import sys
import time
from pathlib import Path
from queue import Empty

import numpy as np
import pygame
import tkinter as tk
from tkinter import filedialog, ttk


DEFAULT_RESULTS_DIR = Path(__file__).parent.parent / "benchmark_experiments-extension" / "results"

# Canvas height in pixels.  The process is DPI-aware (see run_simulation), so
# this is PHYSICAL pixels, not logical ones.  Fixed rather than derived from
# the screen so the canvas is identical on every machine and for every track.
_CANVAS_HEIGHT_PX = 1250


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def find_iteration_folders(run_folder: str | Path) -> list[Path]:
    run_path = Path(run_folder)
    return sorted(
        (p for p in run_path.iterdir() if p.is_dir() and p.name.startswith("iter_")),
        key=lambda p: int(p.name.split("_")[1]),
    )

def load_chromosomes(folder: Path) -> list[dict]:
    files = sorted(folder.glob("chromosome_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    chromosomes = []
    for filepath in files:
        with open(filepath) as f:
            chromosomes.append(json.load(f))
    return chromosomes

def sort_chromosomes(chromosomes: list[dict], sort_by: str) -> list[dict]:
    if sort_by == "Quality (best first)":
        return sorted(chromosomes, key=lambda c: -(c.get("quality") or 0.0))
    if sort_by == "Quality (worst first)":
        return sorted(chromosomes, key=lambda c:  (c.get("quality") or 0.0))
    return list(chromosomes)

def infer_problem_type(chromosomes: list[dict], results_folder: str | Path = "") -> str:
    content = chromosomes[0].get("content") or {} if chromosomes else {}
    if "cell_scores" in content:
        return "RacingVoronoi"
    if "polar_points" in content:
        return "RacingRadial"
    if "tile_prefs" in content:
        # Both tile genomes use the same key, so the env name in the results
        # path is the discriminator.  Two fallbacks: the gene range (square
        # prefs are 0-6, hex go up to 15) and the length (square 11x11 = 121
        # cells, hex 13x11 = 143).
        if "tilehex" in str(results_folder).lower():
            return "RacingTileHex"
        if int(np.max(np.asarray(content["tile_prefs"]))) > 6:
            return "RacingTileHex"
        return "RacingTile"
    return "Racing"

def numpy_content(content: dict) -> dict:
    return {k: np.asarray(v) if isinstance(v, list) else v for k, v in content.items()}

def resolve_all_iterations(config: dict) -> tuple[list[Path], str, int]:
    """Return iteration folder list and problem type without loading all chromosome data."""
    folder = Path(config["results_folder"])
    iters  = find_iteration_folders(folder)
    if not iters:
        raise FileNotFoundError(f"No iter_N folders found in {folder}")
    iter_name = config.get("iteration", "latest")
    names     = [p.name for p in iters]
    start_idx = (len(iters) - 1) if iter_name == "latest" else (
        names.index(iter_name) if iter_name in names else len(iters) - 1)
    seed_chroms = load_chromosomes(iters[start_idx])
    if not seed_chroms:
        raise FileNotFoundError(f"No chromosome files found in {iters[start_idx]}")
    return iters, infer_problem_type(seed_chroms, folder), start_idx

def build_problem(problem_type: str, num_cells: int = 36):
    if problem_type == "RacingVoronoi":
        from pcg_benchmark.probs.racingvoronoi.problem import RacingVoronoiProblem
        # num_cells comes from the saved genome's cell_scores length.  Every
        # other knob is left at the problem's own default, so the viewer
        # decodes the same track the GA scored and cannot drift out of step
        # with it the way a hardcoded copy would.
        return RacingVoronoiProblem(num_cells=num_cells)
    if problem_type == "RacingRadial":
        from pcg_benchmark.probs.racingradial.problem import RacingRadialProblem
        return RacingRadialProblem()
    if problem_type == "RacingTile":
        from pcg_benchmark.probs.racingtile.problem import RacingTileProblem
        return RacingTileProblem()
    if problem_type == "RacingTileHex":
        from pcg_benchmark.probs.racingtilehex.problem import RacingTileHexProblem
        return RacingTileHexProblem()
    from pcg_benchmark.probs.racing.problem import RacingProblem
    return RacingProblem()

def parse_speed(speed) -> float:
    """Playback speed as a REAL-TIME factor (1.0 = real time, 0.25 = quarter).

    Not a steps-per-frame count: the playback loop converts it using the
    frame rate and the physics timestep.
    """
    return float(str(speed).strip())

def _score_to_color(score: float) -> tuple[int, int, int]:
    """Map score [0, 1] from grass green to bright orange."""
    t = max(0.0, min(1.0, float(score)))
    return (int(34 + t * 221), int(139 + t * 1), int(34 - t * 34))

def _extract_cell_polygons(problem) -> tuple[dict[int, np.ndarray], set[int]]:
    """Return ({cell_index: ordered vertex positions} for every traceable cell,
    set of ineligible cell indices).

    All cells are traced, including ineligible border cells, so the score
    heatmap covers the whole diagram.  Border cells have an open edge chain
    (their region is clipped at the map edge); the chain is walked from one
    endpoint to the other and implicitly closed with a straight segment."""
    vertices   = problem._voronoi_vertices
    edges      = problem._voronoi_all_edges
    pairs      = problem._voronoi_all_pairs
    num_cells  = problem._num_cells
    ineligible = set(getattr(problem, "_ineligible_cells", None) or set())
    polygons   = {}
    for cell_idx in range(num_cells):
        mask       = (pairs[:, 0] == cell_idx) | (pairs[:, 1] == cell_idx)
        cell_edges = edges[mask]
        if len(cell_edges) < 2:
            continue
        adj = {}
        for a, b in cell_edges:
            adj.setdefault(int(a), []).append(int(b))
            adj.setdefault(int(b), []).append(int(a))
        # Closed polygons have degree 2 everywhere; clipped border cells have
        # two degree-1 endpoints. Start at an endpoint when one exists so the
        # walk covers the full chain instead of bouncing mid-chain.
        endpoints = [v for v, nbrs in adj.items() if len(nbrs) == 1]
        start = endpoints[0] if endpoints else int(cell_edges[0, 0])
        prev, curr, poly = None, start, []
        for _ in range(len(adj) + 2):
            poly.append(curr)
            nxt = next((n for n in adj[curr] if n != prev), None)
            if nxt is None or nxt == start:
                break
            prev, curr = curr, nxt
        if len(poly) >= 3:
            polygons[cell_idx] = vertices[np.array(poly)]
    return polygons, ineligible


# ---------------------------------------------------------------------------
# Pygame viewer
# ---------------------------------------------------------------------------

_CAR_LENGTH_METRES = 5.0
_CAR_WIDTH_METRES  = 2.0

def _rotated_rect_corners(cx, cy, fwd, rgt, hl, hw):
    fx, fy = fwd; rx, ry = rgt
    return [
        (int(round(cx + fx*hl + rx*hw)), int(round(cy + fy*hl + ry*hw))),
        (int(round(cx + fx*hl - rx*hw)), int(round(cy + fy*hl - ry*hw))),
        (int(round(cx - fx*hl - rx*hw)), int(round(cy - fy*hl - ry*hw))),
        (int(round(cx - fx*hl + rx*hw)), int(round(cy - fy*hl + ry*hw))),
    ]


class RaceViewer:
    """Real-time pygame window for watching a car drive around a racetrack."""

    def __init__(self, width, height, scale=None, fps=60, show_hud=True, title="Race Viewer"):
        pygame.init()
        pygame.font.init()
        self._map_w, self._map_h = float(width), float(height)
        if scale is None:
            # Fixed canvas height, so every track is drawn at the same scale
            # and tracks stay directly comparable by eye.
            scale = _CANVAS_HEIGHT_PX / self._map_h
        self._scale  = float(scale)
        self._fps    = fps
        self._show_hud      = show_hud
        self._img_w         = int(round(width  * self._scale))
        self._img_h         = int(round(height * self._scale))
        # View origin in pixels, kept at 0 so the view always covers the whole
        # map.  Every track therefore shares one canvas and one scale.
        self._ox = 0.0
        self._oy = 0.0
        self._screen        = pygame.display.set_mode((self._img_w, self._img_h))
        self._clock         = pygame.time.Clock()
        self._track_surface = None
        self._font          = None
        pygame.display.set_caption(title)
        if show_hud:
            for name in ("DejaVu Sans", "Segoe UI", "Arial", None):
                try:
                    self._font = pygame.font.SysFont(name, 18); break
                except Exception:
                    pass

    # ── View transform: world metres -> window pixels ────────────────────

    def _px(self, pts):
        """Map world points (metres) to window pixels: p * scale - origin."""
        a = np.asarray(pts, dtype=float) * self._scale
        a[..., 0] -= self._ox
        a[..., 1] -= self._oy
        return a

    def _reset_view(self):
        """Show the whole map at the fixed canvas scale.

        Deliberately NOT a per-track zoom: fitting each track to the window
        would make a compact tile track and a sprawling spline track look the
        same size, which hides exactly the difference between representations
        that the viewer is used to judge.
        """
        self._scale = float(min(self._img_w / self._map_w, self._img_h / self._map_h))
        self._ox = self._oy = 0.0

    def build_track_surface(self, curve_points, track_width,
                            voronoi_edges=None, voronoi_vertices=None,
                            cell_polygons=None, cell_scores=None,
                            ineligible_cells=None):
        """Draw the static track geometry onto a cached Surface."""
        self._reset_view()
        scale         = self._scale
        curve_px      = self._px(curve_points)
        half_width_px = float(track_width) * scale * 0.5
        surface       = pygame.Surface((self._img_w, self._img_h))
        surface.fill((34, 139, 34))
        if cell_polygons is not None and cell_scores is not None:
            ineligible = ineligible_cells or set()
            score_font = pygame.font.Font(None, max(14, int(scale * 5)))
            for cell_idx, poly_verts in cell_polygons.items():
                if cell_idx >= len(cell_scores):
                    continue
                score   = float(cell_scores[cell_idx])
                color   = _score_to_color(score)
                blocked = cell_idx in ineligible
                if blocked:
                    # Ineligible (out-of-bounds) cells: scored but never
                    # selectable, shown dimmed toward gray.
                    color = tuple(int(0.35 * c + 0.65 * g) for c, g in zip(color, (95, 95, 95)))
                poly_px = [(int(round(x)), int(round(y)))
                           for x, y in self._px(poly_verts)]
                if len(poly_px) >= 3:
                    pygame.draw.polygon(surface, color, poly_px)
                    if blocked:
                        pygame.draw.polygon(surface, (50, 50, 50), poly_px, 2)
                    ctr   = self._px(np.asarray(poly_verts, dtype=float).mean(axis=0))
                    cx    = int(round(ctr[0]))
                    cy    = int(round(ctr[1]))
                    label = score_font.render(f"{score:.2f}", True, (0, 0, 0) if not blocked else (210, 210, 210))
                    surface.blit(label, (cx - label.get_width() // 2, cy - label.get_height() // 2))
        left_edge, right_edge = self._compute_track_edges(curve_px, half_width_px)
        if len(left_edge)  > 1: pygame.draw.lines(surface, (10, 10, 10), False, left_edge,  4)
        if len(right_edge) > 1: pygame.draw.lines(surface, (10, 10, 10), False, right_edge, 4)
        for j in range(len(left_edge) - 1):
            pygame.draw.polygon(surface, (215, 215, 215),
                                [left_edge[j], left_edge[j+1], right_edge[j+1], right_edge[j]])
        if voronoi_edges is not None and voronoi_vertices is not None:
            sv = self._px(voronoi_vertices)
            for a, b in voronoi_edges:
                pygame.draw.line(surface, (70, 70, 70),
                                 (int(round(sv[int(a), 0])), int(round(sv[int(a), 1]))),
                                 (int(round(sv[int(b), 0])), int(round(sv[int(b), 1]))), 1)
        if cell_polygons is not None:
            # The edge list holds Voronoi ridges only.  A cell on the outside
            # of the diagram is closed by the map border instead of by a
            # ridge, so that part of its outline is in no edge, and stroking
            # the traced polygons is what draws it.  Same gray as the ridges,
            # so the diagram reads as one thing whether or not the score
            # heatmap is showing.
            for poly_verts in cell_polygons.values():
                poly_px = [(int(round(x)), int(round(y)))
                           for x, y in self._px(poly_verts)]
                if len(poly_px) >= 3:
                    pygame.draw.polygon(surface, (70, 70, 70), poly_px, 1)
        cl = [(int(round(x)), int(round(y))) for x, y in curve_px]
        if len(cl) > 1:
            pygame.draw.lines(surface, (120, 120, 120), False, cl, 2)
        self._track_surface = surface

    def build_tile_surface(self, types, rotations):
        """Draw the tile grid as the static track background (for RacingTile problems)."""
        import math as _m
        # This view draws the grid in cell space across the whole window, so
        # the car must be placed against the full map, not a zoomed track.
        self._reset_view()
        grid_h, grid_w = types.shape
        surface = pygame.Surface((self._img_w, self._img_h))
        surface.fill((34, 139, 34))
        cw = self._img_w / grid_w
        ch = self._img_h / grid_h
        ROAD = (210, 210, 210)
        EDGE = (25,  25,  25)
        GRASS_C = (34, 139, 34)
        GRASS, STRAIGHT = 0, 1

        for r in range(grid_h):
            for c in range(grid_w):
                x0, y0 = c * cw, r * ch
                t, rot = int(types[r, c]), int(rotations[r, c])
                pygame.draw.rect(surface, GRASS_C, (int(x0), int(y0), int(cw) + 1, int(ch) + 1))
                if t == GRASS:
                    continue
                rw, rh = cw * 0.5, ch * 0.5
                if t == STRAIGHT:
                    if rot in (0, 2):
                        ry = int(y0 + (ch - rh) / 2)
                        pygame.draw.rect(surface, ROAD, (int(x0), ry, int(cw), int(rh)))
                        pygame.draw.line(surface, EDGE, (int(x0), ry),         (int(x0+cw), ry),         2)
                        pygame.draw.line(surface, EDGE, (int(x0), ry+int(rh)), (int(x0+cw), ry+int(rh)), 2)
                    else:
                        rx = int(x0 + (cw - rw) / 2)
                        pygame.draw.rect(surface, ROAD, (rx, int(y0), int(rw), int(ch)))
                        pygame.draw.line(surface, EDGE, (rx,         int(y0)), (rx,         int(y0+ch)), 2)
                        pygame.draw.line(surface, EDGE, (rx+int(rw), int(y0)), (rx+int(rw), int(y0+ch)), 2)
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
                        pts.append((int(cx + r_out * _m.cos(a)), int(cy + r_out * _m.sin(a))))
                    for i in range(steps + 1):
                        a = _m.radians(e - (e - s) * i / steps)
                        pts.append((int(cx + r_in * _m.cos(a)), int(cy + r_in * _m.sin(a))))
                    if len(pts) >= 3:
                        pygame.draw.polygon(surface, ROAD, pts)
                    for ar in (r_in, r_out):
                        prev = None
                        for i in range(steps + 1):
                            a  = _m.radians(s + (e - s) * i / steps)
                            pt = (int(cx + ar * _m.cos(a)), int(cy + ar * _m.sin(a)))
                            if prev:
                                pygame.draw.line(surface, EDGE, prev, pt, 2)
                            prev = pt

        for i in range(grid_h + 1):
            y = int(i * ch)
            pygame.draw.line(surface, (20, 100, 20), (0, y), (self._img_w, y), 1)
        for j in range(grid_w + 1):
            x = int(j * cw)
            pygame.draw.line(surface, (20, 100, 20), (x, 0), (x, self._img_h), 1)

        self._track_surface = surface

    def build_hextile_surface(self, problem):
        """Draw the RacingTileHex structure view: the normal track ribbon with
        the hex cell grid drawn on top.

        The road comes from the ribbon, not from the per-tile arc polylines.
        Drawing it per tile produces a lumpy string of beads where the turns
        should be smooth, because each tile's arc is stroked and capped
        separately.

        Building the same filled ribbon the plain track view uses keeps the
        road identical to every other representation, and the hex outlines go
        on top, so the structure is shown ON the track rather than substituted
        for it.  Pixel coordinates are in problem space (0.._width /
        0.._height); scale maps them to the viewer surface."""
        from pcg_benchmark.probs.racingtilehex.problem import GRID_H, GRID_W

        # 1. The nice road ribbon, exactly as the plain track view draws it.
        self.build_track_surface(problem._curve_points, problem._track_width)
        surface = self._track_surface

        # 2. Hex cell outlines on top, so the grid is visible over the track.
        # Both go through self._px, so the grid stays registered with the road.
        OUTLINE = (20, 100, 20)
        for r in range(GRID_H):
            for c in range(GRID_W):
                corners = [(int(round(p[0])), int(round(p[1])))
                           for p in self._px(problem._hex_corners(r, c))]
                pygame.draw.polygon(surface, OUTLINE, corners, 1)

        self._track_surface = surface

    def draw_frame(self, state, action=None, lookahead=None, yaw_rate=0.0, mass=1350.0):
        """Blit the cached track, draw the car and HUD."""
        if self._track_surface is None:
            raise RuntimeError("Call build_track_surface() before draw_frame().")
        self._screen.blit(self._track_surface, (0, 0))
        self._draw_car(state)
        if lookahead is not None:
            self._draw_lookahead(state, lookahead)
        if self._show_hud and self._font is not None and action is not None:
            self._draw_hud(state, action, yaw_rate, mass)
        self._clock.tick(self._fps)

    @property
    def fps(self):
        """Frame rate cap, used by the playback loop to pace physics steps."""
        return self._fps

    def close(self):
        pygame.quit()

    @staticmethod
    def interpolate_state(prev_state, state, alpha):
        """Blend two physics states for drawing, `alpha` in [0, 1].

        Physics runs at 10 Hz but the display refreshes at 60, so most
        frames fall BETWEEN two simulated states. Drawing the latest state
        on every frame makes the car sit still for five frames and then
        jump, which reads as jitter even though the simulation is fine.
        Interpolating position and heading turns those five frames into
        real motion. Only x/y/angle are blended: speed and steering angle
        are HUD readouts, and showing the true last simulated value there
        is better than a smoothed one.
        """
        if prev_state is None or state is None:
            return state
        out = np.array(state, dtype=float)
        a = min(max(float(alpha), 0.0), 1.0)
        out[0] = prev_state[0] + (state[0] - prev_state[0]) * a
        out[1] = prev_state[1] + (state[1] - prev_state[1]) * a
        if len(out) > 2:
            # Shortest way round the circle, so wrapping from +pi to -pi
            # does not spin the car backwards for one frame.
            d = (float(state[2]) - float(prev_state[2]) + np.pi) % (2.0 * np.pi) - np.pi
            out[2] = float(prev_state[2]) + d * a
        return out

    @staticmethod
    def compute_yaw_rate(state, prev_angle, dt):
        current = float(state[2]) if len(state) > 2 else 0.0
        if prev_angle is None or dt <= 0.0:
            return 0.0, current
        delta = (current - prev_angle + np.pi) % (2.0 * np.pi) - np.pi
        return float(delta / dt), current

    def _draw_car(self, state):
        s  = self._scale
        cx, cy = self._px((float(state[0]), float(state[1])))
        a  = float(state[2]) if len(state) > 2 else 0.0
        ca, sa   = float(np.cos(a)), float(np.sin(a))
        fwd, rgt = (ca, sa), (-sa, ca)
        L, W     = _CAR_LENGTH_METRES * s, _CAR_WIDTH_METRES * s
        wb, wt   = L * 0.34, W * 0.44
        for sx, sy in [(1,1),(1,-1),(-1,1),(-1,-1)]:
            wc = _rotated_rect_corners(
                cx + fwd[0]*wb*sx + rgt[0]*wt*sy,
                cy + fwd[1]*wb*sx + rgt[1]*wt*sy,
                fwd, rgt, L*0.10, W*0.11,
            )
            pygame.draw.polygon(self._screen, (25, 25, 25), wc)
            pygame.draw.polygon(self._screen, (0, 0, 0),    wc, 1)
        hl, hw = L/2, W/2
        body = [
            (cx + ca*hl - sa*hw, cy + sa*hl + ca*hw),
            (cx + ca*hl + sa*hw, cy + sa*hl - ca*hw),
            (cx - ca*hl + sa*hw, cy - sa*hl - ca*hw),
            (cx - ca*hl - sa*hw, cy - sa*hl + ca*hw),
        ]
        pygame.draw.polygon(self._screen, (255, 0, 0), body)
        pygame.draw.polygon(self._screen, (0, 0, 0),   body, 1)
        pygame.draw.line(self._screen, (0, 0, 255),
                         (int(round(cx)), int(round(cy))),
                         (int(round(cx + ca*hl)), int(round(cy + sa*hl))), 3)

    def _draw_lookahead(self, state, lookahead):
        cp  = self._px((float(state[0]), float(state[1])))
        lp  = self._px((float(lookahead[0]), float(lookahead[1])))
        car = (int(round(cp[0])), int(round(cp[1])))
        la  = (int(round(lp[0])), int(round(lp[1])))
        pygame.draw.circle(self._screen, (0, 255, 255), la, 6, 3)
        pygame.draw.line(self._screen, (0, 200, 200), car, la, 2)

    def _draw_hud(self, state, action, yaw_rate, mass):
        v    = float(state[3]) if len(state) > 3 else 0.0
        sa   = float(state[4]) * (180.0/np.pi) if len(state) > 4 else 0.0
        sc   = float(action.get("steering", 0.0)) if isinstance(action, dict) else 0.0
        tc   = float(action.get("throttle", 0.0)) if isinstance(action, dict) else 0.0
        alat = float(v * yaw_rate)
        lines = [
            f"v: {v:5.1f} m/s  ({v*3.6:5.0f} km/h)",
            f"steer cmd: {sc:+.2f}   steer ang: {sa:+5.1f} deg",
            f"throttle: {tc:+.2f}",
            f"yaw rate: {yaw_rate:+6.2f} rad/s   a_lat: {alat:+6.2f} m/s^2",
            f"Fy est: {mass*alat/1000.0:+7.2f} kN",
        ]
        pad  = 6
        lh   = self._font.get_linesize()
        rend = [self._font.render(l, True, (0, 0, 0)) for l in lines]
        bw   = max(r.get_width() for r in rend) + pad*2
        bh   = lh * len(lines) + pad*2
        hud  = pygame.Surface((bw, bh))
        hud.fill((255, 255, 255))
        pygame.draw.rect(hud, (0, 0, 0), hud.get_rect(), 1)
        for i, r in enumerate(rend):
            hud.blit(r, (pad, pad + i*lh))
        self._screen.blit(hud, (6, 6))

    @staticmethod
    def _compute_track_edges(curve_px, half_width_px):
        left_edge, right_edge = [], []
        n = len(curve_px)
        if n < 2:
            return left_edge, right_edge
        has_dup = n >= 3 and np.allclose(curve_px[0], curve_px[-1], atol=1e-9)
        base    = curve_px[:-1] if has_dup else curve_px
        m       = len(base)
        for j in range(m):
            if m >= 3:
                dp = base[j] - base[(j-1) % m]
                dn = base[(j+1) % m] - base[j]
            else:
                dp = base[1] - base[0]     if j == 0     else base[j] - base[j-1]
                dn = base[j] - base[j-1]   if j == m-1   else base[j+1] - base[j]
            avg  = dp + dn
            norm = float(np.linalg.norm(avg))
            perp = np.array([-avg[1], avg[0]], dtype=float) / norm if norm > 0 else np.zeros(2)
            left_edge.append( (int(round(base[j][0] + perp[0]*half_width_px)),
                               int(round(base[j][1] + perp[1]*half_width_px))))
            right_edge.append((int(round(base[j][0] - perp[0]*half_width_px)),
                               int(round(base[j][1] - perp[1]*half_width_px))))
        if has_dup and left_edge:
            left_edge.append(left_edge[0]); right_edge.append(right_edge[0])
        return left_edge, right_edge


# ---------------------------------------------------------------------------
# Simulation subprocess
# ---------------------------------------------------------------------------

def run_simulation(config: dict, cmd_queue, status_queue) -> None:
    """Initialise pygame, load tracks, and run the main simulation loop."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try: ctypes.windll.user32.SetProcessDPIAware()
        except Exception: pass

    pygame.init(); pygame.font.init()
    sys.path.insert(0, str(Path(__file__).parent))

    from pcg_benchmark.probs.racing.agent import SteeringAgent

    try:
        iter_folders, problem_type, start_iter = resolve_all_iterations(config)
    except Exception as exc:
        print(f"[viewer] Could not load content: {exc}"); return

    sort_by     = config["sort_by"]
    total_iters = len(iter_folders)
    chrom_cache: dict[int, list[dict]] = {}

    def get_iter_chroms(idx: int) -> list[dict]:
        if idx not in chrom_cache:
            chrom_cache[idx] = sort_chromosomes(load_chromosomes(iter_folders[idx]), sort_by)
        return chrom_cache[idx]

    is_voronoi = problem_type == "RacingVoronoi"
    is_tile    = problem_type == "RacingTile"
    is_hextile = problem_type == "RacingTileHex"
    num_cells  = 36
    if is_voronoi:
        seed = get_iter_chroms(start_iter)
        if seed:
            scores = (seed[0].get("content") or {}).get("cell_scores")
            if scores is not None:
                num_cells = len(scores)

    problem = build_problem(problem_type, num_cells=num_cells)
    if is_voronoi:
        problem._build_cell_graph()
    cell_polygons, ineligible_cells = _extract_cell_polygons(problem) if is_voronoi else (None, set())
    show_structure   = config.get("show_structure", False)
    # Path to a trained SB3 model; empty means use the scripted agent.
    # Resolved against this file's directory so a relative path works
    # regardless of where the viewer was launched from.
    rl_model_path    = (config.get("rl_model", "") or "").strip()
    if rl_model_path and not Path(rl_model_path).is_absolute():
        rl_model_path = str((Path(__file__).parent / rl_model_path).resolve())
    current_content  = None
    # False when the loaded chromosome does not fit this problem, so the
    # structure overlay knows not to decode it a second time.
    content_fits     = True

    # scale=None: the viewer sizes its window to the screen and then fits the
    # view to each track as it is loaded.
    viewer  = RaceViewer(problem._width, problem._height, scale=None,
                         fps=config["fps"], show_hud=config["show_hud"],
                         title=f"Track Viewer — {problem_type}")

    speed_multiplier = parse_speed(config.get("speed", "0.5"))
    auto_cycle       = config["loop_track"]
    overlay_font     = pygame.font.SysFont("Segoe UI", 16) or pygame.font.Font(None, 16)

    iter_index = chrom_index = step = 0
    step_accumulator = yaw_rate = 0.0
    prev_angle = state = prev_state = agent = None
    last_frame_time = time.perf_counter()
    action    = {"steering": 0.0, "throttle": 0.0}
    lookahead = None
    dt        = 0.1

    def load_track():
        """Load the selected chromosome, falling back to the default track when
        the saved genome does not fit this problem.

        A results tree written before a grid size changed holds genomes of the
        wrong length (an 11x11 tile genome is 121 genes; the 13x11 hex grid
        wants 143).  Decoding one raises deep inside the representation, and
        this runs in a worker process, so an unhandled raise takes the viewer
        down with no window and only a traceback.  Report it and keep going.
        """
        nonlocal current_content, content_fits
        current_content = numpy_content(get_iter_chroms(iter_index)[chrom_index]["content"])
        content_fits = True
        try:
            track_points = problem._extract_content(current_content)
        except (ValueError, KeyError, IndexError) as exc:
            print(f"[viewer] iter {iter_index} chromosome {chrom_index} does not fit "
                  f"{problem_type}: {type(exc).__name__}: {exc}")
            print("[viewer] these results were saved for a different grid; "
                  "showing the default track instead")
            content_fits = False
            track_points = problem._extract_content(None)
        problem.reset(track_points)

    def rebuild_surface():
        if is_tile and show_structure and content_fits and current_content is not None:
            if hasattr(problem, "_decode_genome") and "tile_prefs" in current_content:
                types, rotations = problem._decode_genome(
                    np.asarray(current_content["tile_prefs"], dtype=int),
                )
                viewer.build_tile_surface(types, rotations)
            return
        if is_hextile and show_structure and content_fits and current_content is not None:
            viewer.build_hextile_surface(problem)
            return
        scores = None
        if show_structure and cell_polygons is not None:
            raw = (get_iter_chroms(iter_index)[chrom_index].get("content") or {}).get("cell_scores")
            if raw is not None:
                scores = np.asarray(raw, dtype=float)
        viewer.build_track_surface(
            problem._curve_points, problem._track_width,
            voronoi_edges=getattr(problem, "_voronoi_all_edges", None) if is_voronoi else None,
            voronoi_vertices=getattr(problem, "_voronoi_vertices", None) if is_voronoi else None,
            # Passed whether or not structure is showing: the fill below is
            # gated on cell_scores, which stays None without it, while the
            # outlines close the border cells in both modes.
            cell_polygons=cell_polygons,
            cell_scores=scores,
            ineligible_cells=ineligible_cells,
        )

    def make_agent():
        """Scripted agent by default; a trained RL policy if one is loaded.

        The RL wrapper exposes the same act()/reset() interface but drives
        its own engine instance, because it reads an 18-float observation
        that the benchmark state does not provide. Everything downstream of
        act() is unchanged.
        """
        if rl_model_path:
            try:
                from rl_agent_adapter import RLAgent
                return RLAgent(rl_model_path, problem._curve_points,
                               problem._track_width,
                               getattr(problem, "_width", None))
            except Exception as exc:
                # Never let a bad model path kill the viewer: say what went
                # wrong and fall back to the scripted agent, which the
                # overlay will then name.
                print(f"[viewer] could not load RL model {rl_model_path!r}: "
                      f"{type(exc).__name__}: {exc}")
                print("[viewer] falling back to the scripted agent")
        ag = SteeringAgent(problem._curve_points, track_width=problem._track_width,
                           engine=getattr(problem, "_engine", None))
        ag.reset()
        return ag

    def switch_to(new_iter, new_chrom=0):
        nonlocal iter_index, chrom_index, agent, state, prev_angle, step, step_accumulator
        nonlocal action, lookahead, yaw_rate, dt, prev_state
        iter_index   = new_iter % total_iters
        total_chroms = len(get_iter_chroms(iter_index))
        chrom_index  = new_chrom % total_chroms
        load_track()
        if len(problem._curve_points) >= 2:
            rebuild_surface(); agent = make_agent()
        state = problem._engine.reset()
        if hasattr(agent, "state"):          # RL agent drives its own engine
            state = agent.reset()
        # Playback pace must come from the engine actually being stepped: an
        # RL agent integrates its own engine instance, so reading the
        # benchmark engine's dt would silently mis-time playback if the two
        # ever stop sharing a control interval.
        dt    = float(getattr(getattr(agent, "_engine", None), "time_step",
                              problem._engine.time_step))
        # prev_state must reset with the episode: interpolating from the
        # previous track's final pose would fling the car across the map on
        # the first frame after a switch.
        prev_angle = None; prev_state = None; step = 0; step_accumulator = 0.0
        action = {"steering": 0.0, "throttle": 0.0}; lookahead = None; yaw_rate = 0.0
        quality = get_iter_chroms(iter_index)[chrom_index].get("quality")
        q_str   = f"  quality={quality:.3f}" if quality is not None else ""
        pygame.display.set_caption(
            f"Track Viewer — {problem_type}  iter [{iter_index+1}/{total_iters}]  [{chrom_index+1}/{total_chroms}]{q_str}")
        try:
            status_queue.put_nowait({
                "iter_index": iter_index, "total_iters": total_iters,
                "chrom_index": chrom_index, "total_chroms": total_chroms,
                "quality": quality,
            })
        except Exception:
            pass

    def draw_overlay():
        chroms   = get_iter_chroms(iter_index)
        quality  = chroms[chrom_index].get("quality")
        q_str    = f"  quality: {quality:.3f}" if quality is not None else ""
        driver   = getattr(agent, "name", None) or "scripted agent"
        text     = (f"iter [{iter_index+1}/{total_iters}]  chrom [{chrom_index+1}/{len(chroms)}]{q_str}"
                    f"  driver: {driver}"
                    f"   ◄/► iterations   ▲/▼ chromosomes   Space restart")
        surf     = overlay_font.render(text, True, (220, 220, 220))
        screen   = pygame.display.get_surface()
        screen.blit(surf, (8, screen.get_height() - surf.get_height() - 8))

    switch_to(start_iter, 0)
    if len(problem._curve_points) < 2:
        print("[viewer] Track generation failed."); viewer.close(); return

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if   event.key == pygame.K_ESCAPE: running = False
                elif event.key == pygame.K_RIGHT:  switch_to(iter_index + 1, 0)
                elif event.key == pygame.K_LEFT:   switch_to(iter_index - 1, 0)
                elif event.key == pygame.K_UP:     switch_to(iter_index, chrom_index + 1)
                elif event.key == pygame.K_DOWN:   switch_to(iter_index, chrom_index - 1)
                elif event.key == pygame.K_SPACE:  switch_to(iter_index, chrom_index)

        if not running: break

        try:
            while True:
                cmd = cmd_queue.get_nowait()
                if   cmd["cmd"] == "stop":        running = False; break
                elif cmd["cmd"] == "speed":       speed_multiplier = cmd["value"]
                elif cmd["cmd"] == "hud":         viewer._show_hud = cmd["value"]
                elif cmd["cmd"] == "next":        switch_to(iter_index + 1, 0)
                elif cmd["cmd"] == "prev":        switch_to(iter_index - 1, 0)
                elif cmd["cmd"] == "restart":     switch_to(iter_index, chrom_index)
                elif cmd["cmd"] == "show_structure":
                    show_structure = cmd["value"]
                    if len(problem._curve_points) >= 2: rebuild_surface()
        except Empty:
            pass

        if not running: break

        # speed_multiplier is a REAL-TIME factor: 1.0 plays the simulation
        # back at the speed it would happen at. Advance by MEASURED elapsed
        # time rather than assuming the frame rate held, so a heavy track
        # that drops below `fps` plays at the right speed instead of going
        # into slow motion.
        now = time.perf_counter()
        elapsed = min(now - last_frame_time, 0.25)   # cap: don't fast-forward
        last_frame_time = now                        # after a long stall
        step_accumulator += speed_multiplier * elapsed / max(dt, 1e-9)
        while step_accumulator >= 1.0:
            action    = agent.act(state)
            lookahead = getattr(agent, "last_lookahead_point", None)
            yaw_rate, prev_angle = RaceViewer.compute_yaw_rate(state, prev_angle, dt)
            # An RL agent integrates its own physics inside act(), so its
            # pose comes from the agent; the scripted agent is driven by
            # the benchmark engine as before.
            # Copy, don't alias: the engine returns one preallocated array and
            # mutates it in place, so keeping the reference would leave
            # prev_state and state pointing at the same values and the
            # interpolation below would have nothing to interpolate between.
            prev_state = np.array(state, dtype=float) if state is not None else None
            state     = (agent.state() if hasattr(agent, "state")
                         else problem._engine.step(action))
            step     += 1; step_accumulator -= 1.0

        # Physics is 10 Hz, the display is 60: draw the car partway between
        # the last two simulated states so the frames in between show motion
        # instead of the car standing still and then jumping.
        draw_state = RaceViewer.interpolate_state(prev_state, state, step_accumulator)
        viewer.draw_frame(draw_state, action=action, lookahead=lookahead, yaw_rate=yaw_rate)
        draw_overlay()
        pygame.display.flip()

        if auto_cycle and step >= problem._default_max_steps:
            switch_to(iter_index, chrom_index + 1)

    viewer.close()


# ---------------------------------------------------------------------------
# Config window (tkinter GUI)
# ---------------------------------------------------------------------------

class ConfigWindow:
    """Main application window. Call run() to enter the tkinter event loop."""

    def __init__(self, defaults: dict) -> None:
        self._defaults      = defaults
        self._process       = None
        self._cmd_queue     = None
        self._status_queue  = None
        self._root          = tk.Tk()
        self._root.title("Track Viewer")
        self._root.resizable(False, False)
        self._root.protocol("WM_DELETE_WINDOW", self._on_quit)
        self._build_ui()

    def run(self):
        self._root.mainloop()

    def _build_ui(self):
        d     = self._defaults
        outer = ttk.Frame(self._root, padding=16)
        outer.grid(sticky="nsew")
        outer.columnconfigure(1, weight=1)
        self._row = 0
        self._build_results_section(outer, d)
        self._build_viewer_section(outer, d)
        self._build_controls_section(outer)
        self._build_status_bar(outer)

    def _sep(self, parent):
        ttk.Separator(parent, orient="horizontal").grid(
            row=self._row, column=0, columnspan=3, sticky="ew", pady=(8, 4))
        self._row += 1

    def _heading(self, parent, text):
        ttk.Label(parent, text=text, font=("", 9, "bold")).grid(
            row=self._row, column=0, columnspan=3, sticky="w")
        self._row += 1

    def _field(self, parent, label, widget, extra=None):
        ttk.Label(parent, text=label).grid(row=self._row, column=0, sticky="w", padx=(0, 8))
        widget.grid(row=self._row, column=1, sticky="ew")
        if extra is not None:
            extra.grid(row=self._row, column=2, padx=(4, 0))
        self._row += 1

    def _build_results_section(self, parent, d):
        self._sep(parent); self._heading(parent, "Results")
        self._folder_var = tk.StringVar(value=d.get("results_folder", str(DEFAULT_RESULTS_DIR)))
        self._field(parent, "Run folder",
                    ttk.Entry(parent, textvariable=self._folder_var, width=32),
                    ttk.Button(parent, text="Browse…", command=self._browse_folder, width=8))
        self._iter_var   = tk.StringVar(value=d.get("iteration", "latest"))
        self._iter_combo = ttk.Combobox(parent, textvariable=self._iter_var, state="readonly", width=14)
        self._field(parent, "Iteration", self._iter_combo)
        self._sort_var = tk.StringVar(value=d.get("sort_by", "Quality (best first)"))
        self._field(parent, "Sort by",
                    ttk.Combobox(parent, textvariable=self._sort_var, state="readonly", width=22,
                                 values=["Quality (best first)", "Quality (worst first)", "Index"]))
        self._folder_status = ttk.Label(parent, text="", foreground="gray")
        self._folder_status.grid(row=self._row, column=0, columnspan=3, sticky="w")
        self._row += 1
        self._refresh_iterations()

    def _browse_folder(self):
        current = self._folder_var.get()
        initial = current if Path(current).exists() else str(DEFAULT_RESULTS_DIR)
        chosen  = filedialog.askdirectory(
            title="Select run folder (contains iter_0, iter_1, …)", initialdir=initial)
        if chosen:
            self._folder_var.set(chosen); self._refresh_iterations()

    def _refresh_iterations(self):
        folder = self._folder_var.get()
        if not folder or not Path(folder).is_dir():
            self._iter_combo["values"] = []
            self._folder_status.config(text="Folder not found", foreground="red"); return
        iters = find_iteration_folders(folder)
        if not iters:
            self._iter_combo["values"] = []
            self._folder_status.config(text="No iter_N folders found", foreground="red"); return
        names = ["latest"] + [p.name for p in iters]
        self._iter_combo["values"] = names
        if self._iter_var.get() not in names:
            self._iter_var.set("latest")
        count = len(list(iters[-1].glob("chromosome_*.json")))
        self._folder_status.config(
            text=f"{len(iters)} iterations · {count} chromosomes in latest", foreground="gray")

    def _build_viewer_section(self, parent, d):
        self._sep(parent); self._heading(parent, "Viewer")
        self._fps_var = tk.IntVar(value=d.get("fps", 60))
        self._field(parent, "FPS",
                    ttk.Spinbox(parent, from_=10, to=120, textvariable=self._fps_var, width=10))
        # Real-time factor: 1.0 = the speed the simulation would actually
        # happen at. Range reaches 0.1 for slow-motion inspection of slides
        # and corner entries, which is what watching the driver is for.
        self._speed_var   = tk.DoubleVar(value=float(d.get("speed", 1.0)))
        self._speed_label = ttk.Label(parent, text=f"{float(d.get('speed', 1.0)):.2f}×", width=6)
        speed_slider      = ttk.Scale(parent, from_=0.1, to=4.0, orient="horizontal",
                                      variable=self._speed_var,
                                      command=lambda _: self._on_speed_changed())
        self._field(parent, "Speed", speed_slider, self._speed_label)
        ttk.Label(parent, text="Speed is a real-time factor: 1.00× is real time, 0.25× is quarter speed",
                  foreground="gray", font=("", 8)).grid(row=self._row, column=0, columnspan=3, sticky="w")
        self._row += 1
        ttk.Label(parent, text="Speed, HUD and structure toggle update live without restarting",
                  foreground="gray", font=("", 8)).grid(row=self._row, column=0, columnspan=3, sticky="w")
        self._row += 1
        self._hud_var = tk.BooleanVar(value=d.get("show_hud", True))
        self._hud_var.trace_add("write", lambda *_: self._on_hud_changed())
        self._field(parent, "Show HUD", ttk.Checkbutton(parent, variable=self._hud_var))
        self._show_structure_var = tk.BooleanVar(value=d.get("show_structure", False))
        self._show_structure_var.trace_add("write", lambda *_: self._on_show_structure_changed())
        self._field(parent, "Show structure", ttk.Checkbutton(parent, variable=self._show_structure_var))
        # Driver: scripted agent, or a trained RL model (.zip from
        # model_training/runs/<name>/best/). Applied on the next start.
        self._rl_var = tk.StringVar(value=d.get("rl_model", ""))
        self._field(parent, "RL model",
                    ttk.Entry(parent, textvariable=self._rl_var, width=28),
                    ttk.Button(parent, text="Browse…", command=self._pick_rl_model))
        ttk.Label(parent, text="Leave empty for the scripted agent",
                  foreground="gray", font=("", 8)).grid(row=self._row, column=0,
                                                        columnspan=3, sticky="w")
        self._row += 1
        self._loop_var = tk.BooleanVar(value=d.get("loop_track", True))
        self._field(parent, "Auto-cycle", ttk.Checkbutton(parent, variable=self._loop_var))

    def _pick_rl_model(self):
        start = Path(__file__).parent / "model_training" / "runs"
        path = filedialog.askopenfilename(
            title="Select a trained model",
            initialdir=str(start) if start.exists() else str(Path.home()),
            filetypes=[("Trained model", "*.zip"), ("All files", "*.*")])
        if path:
            self._rl_var.set(path)

    def _build_controls_section(self, parent):
        self._sep(parent)
        nav = ttk.Frame(parent)
        nav.grid(row=self._row, column=0, columnspan=3, sticky="ew", pady=(0, 4)); self._row += 1
        self._prev_btn    = ttk.Button(nav, text="◄ Prev",    width=9, command=lambda: self._send_cmd({"cmd": "prev"}))
        self._restart_btn = ttk.Button(nav, text="↺ Restart", width=9, command=lambda: self._send_cmd({"cmd": "restart"}))
        self._next_btn    = ttk.Button(nav, text="Next ►",    width=9, command=lambda: self._send_cmd({"cmd": "next"}))
        self._prev_btn.pack(side="left", padx=(0, 2))
        self._restart_btn.pack(side="left", padx=2)
        self._next_btn.pack(side="left", padx=(2, 0))
        run = ttk.Frame(parent)
        run.grid(row=self._row, column=0, columnspan=3, sticky="e"); self._row += 1
        self._run_btn  = ttk.Button(run, text="Run ▶",  width=10, command=self._on_run)
        self._stop_btn = ttk.Button(run, text="Stop ■", width=10, command=self._on_stop)
        ttk.Button(run, text="Quit", command=self._on_quit).pack(side="left", padx=(0, 6))
        self._run_btn.pack(side="left", padx=(0, 2)); self._stop_btn.pack(side="left")
        self._set_running_state(running=False)

    def _build_status_bar(self, parent):
        self._sep(parent)
        self._status_var = tk.StringVar(value="Stopped")
        ttk.Label(parent, textvariable=self._status_var, foreground="gray").grid(
            row=self._row, column=0, columnspan=3, sticky="w")
        self._row += 1

    def _on_run(self):
        if self._process and self._process.is_alive(): return
        self._start_simulation(self._collect_config())

    def _on_stop(self):  self._stop_simulation()
    def _on_quit(self):  self._stop_simulation(); self._root.destroy()

    def _start_simulation(self, config):
        self._cmd_queue    = multiprocessing.Queue()
        self._status_queue = multiprocessing.Queue()
        self._process      = multiprocessing.Process(
            target=run_simulation, args=(config, self._cmd_queue, self._status_queue), daemon=True)
        self._process.start()
        self._set_running_state(running=True)
        self._status_var.set("Starting…")
        self._poll_status()

    def _stop_simulation(self):
        if self._process is None: return
        self._send_cmd({"cmd": "stop"})
        self._process.join(timeout=2.0)
        if self._process.is_alive(): self._process.terminate()
        self._process = None
        self._set_running_state(running=False)
        self._status_var.set("Stopped")

    def _poll_status(self):
        if self._process is None: return
        if not self._process.is_alive():
            self._process = None
            self._set_running_state(running=False)
            self._status_var.set("Stopped"); return
        latest = None
        try:
            while True: latest = self._status_queue.get_nowait()
        except Exception: pass
        if latest is not None:
            ii = latest.get("iter_index",   0) + 1
            ti = latest.get("total_iters",  1)
            ci = latest.get("chrom_index",  0) + 1
            tc = latest.get("total_chroms", 1)
            q  = latest.get("quality")
            self._status_var.set(
                f"iter [{ii}/{ti}]  chrom [{ci}/{tc}]" + (f"  quality: {q:.3f}" if q is not None else ""))
        self._root.after(150, self._poll_status)

    def _send_cmd(self, cmd):
        if self._cmd_queue is not None:
            try: self._cmd_queue.put_nowait(cmd)
            except Exception: pass

    def _on_speed_changed(self):
        v = round(float(self._speed_var.get()), 2)
        self._speed_label.config(text=f"{v:.2f}×")
        self._send_cmd({"cmd": "speed", "value": v})

    def _on_show_structure_changed(self):
        self._send_cmd({"cmd": "show_structure", "value": bool(self._show_structure_var.get())})

    def _on_hud_changed(self): self._send_cmd({"cmd": "hud", "value": bool(self._hud_var.get())})

    def _set_running_state(self, running):
        self._run_btn.configure( state="disabled" if running else "normal")
        self._stop_btn.configure(state="normal"   if running else "disabled")
        for btn in (self._prev_btn, self._restart_btn, self._next_btn):
            btn.configure(state="normal" if running else "disabled")

    def _collect_config(self):
        return {
            "results_folder": self._folder_var.get(),
            "iteration":      self._iter_var.get(),
            "sort_by":        self._sort_var.get(),
            "fps":            self._fps_var.get(),
            "speed":          round(float(self._speed_var.get()), 2),
            "show_hud":        bool(self._hud_var.get()),
            "show_structure":  bool(self._show_structure_var.get()),
            "loop_track":     bool(self._loop_var.get()),
            "rl_model":       self._rl_var.get().strip(),
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "results_folder": str(DEFAULT_RESULTS_DIR),
    "iteration":      "latest",
    "sort_by":        "Quality (best first)",
    "fps":            60,
    "speed":          1.0,          # real-time factor (1.0 = real time)
    "show_hud":        True,
    "show_structure":  False,
    "rl_model":        "",
    "loop_track":     True,
}

if __name__ == "__main__":
    multiprocessing.freeze_support()
    ConfigWindow(_DEFAULTS).run()
