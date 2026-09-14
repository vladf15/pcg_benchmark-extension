"""Bridge from generated benchmark tracks to the driving environment.

The circuits and the generated tracks share a unit system (world units are
metres in both), so a generated track needs no rescaling: take the smoothed
centerline the benchmark already computes and hand it to TrackGeometry.

    from generated_tracks import env_for_content
    env = env_for_content(problem, content)

This is the piece Phase 5 needs: it lets the driver trained on real circuits
be pointed at whatever the GA produced, scored by exactly the same code that
scores the real circuits.

Run as a script to score saved experiment output:
    python generated_tracks.py --problem racingtile-v0 --content out/best.json
"""

import argparse
import json
import os
import sys

import numpy as np

import track_loader
from racing_env import RacingEnv
from track_geometry import TrackGeometry

_BENCH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "..", ".."))
if _BENCH not in sys.path:
    sys.path.append(_BENCH)


def centerline_from_content(problem, content):
    """Smoothed centerline (world units) for one piece of generated content.

    Uses the benchmark's own `info()`, so the line scored here is exactly the
    line the benchmark's quality function was computed on. No rescaling: the
    generated tracks are already in metres.
    """
    info = problem.info(content)
    pts = np.asarray(info["curve_points"], dtype=float)
    if len(pts) < 4:
        raise ValueError(f"degenerate track: only {len(pts)} curve points")
    return pts, info


def geometry_from_content(problem, content, map_size=None):
    """TrackGeometry for generated content, on the benchmark's own map."""
    pts, info = centerline_from_content(problem, content)
    if map_size is None:
        # Generated tracks live on the benchmark's fixed square map, unlike
        # the real circuits which each carry their own bounding square.
        map_size = float(getattr(problem, "_width", 750))
    width = float(getattr(problem, "_track_width", track_loader.TRACK_WIDTH))
    return TrackGeometry(pts, width, map_size), info


def centerline_escapes_map(geometry, margin=None):
    """True if the centerline itself leaves the drivable area.

    Generated centerlines are not clamped to the map, so a sampled track can
    place points past the edge or at negative coordinates. Such a track cannot
    be lapped whatever its shape, so the driveability report must call it out
    as out of bounds rather than blame the driver for an off_map failure.
    """
    if margin is None:
        margin = track_loader.MARGIN
    pts = geometry.points
    lo, hi = margin, geometry.map_size - margin
    return bool((pts < lo).any() or (pts > hi).any())


class _PresetTrackEnv(RacingEnv):
    """RacingEnv driving one supplied geometry instead of a loaded circuit.

    Everything else (observation, reward, termination) is inherited
    unchanged, which is the point: generated tracks must be judged by
    identical rules to the real ones or the comparison means nothing.
    """

    def __init__(self, geometry, name="generated", **kwargs):
        super().__init__(track_names=[], **kwargs)
        self._geoms = {name: {(False, False): geometry,
                              (True, False): geometry.reversed(),
                              (False, True): geometry.mirrored(),
                              (True, True): geometry.mirrored().reversed()}}
        self._names = [name]


def env_for_geometry(geometry, name="generated", **kwargs):
    """Evaluation env (deterministic) for an already-built geometry."""
    kwargs.setdefault("randomize", False)
    kwargs.setdefault("seed", 0)
    return _PresetTrackEnv(geometry, name=name, **kwargs)


def env_for_content(problem, content, map_size=None, **kwargs):
    """Evaluation env for one piece of generated benchmark content."""
    geom, _ = geometry_from_content(problem, content, map_size=map_size)
    return env_for_geometry(geom, **kwargs)


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", default="racing-v0",
                        help="benchmark problem name, e.g. racingtile-v0")
    parser.add_argument("--content", required=True,
                        help="JSON file holding content (a list scores each entry)")
    parser.add_argument("--model", default=None,
                        help="trained SB3 model (default: scripted heuristic)")
    args = parser.parse_args()

    import pcg_benchmark
    from baseline import evaluate_track, make_policy

    env_bench = pcg_benchmark.make(args.problem)
    problem = env_bench._problem

    with open(args.content, "r", encoding="utf-8") as fh:
        contents = json.load(fh)
    if not isinstance(contents, list):
        contents = [contents]

    policy, policy_name = make_policy(args.model)
    print(f"problem: {args.problem}   policy: {policy_name}\n")
    header = (f"{'#':<5}{'outcome':<10}{'prog':>6}{'offroad':>9}"
              f"{'laptime':>9}{'mean v':>8}{'jerk':>7}")
    print(header)
    print("-" * len(header))

    for i, content in enumerate(contents):
        try:
            geom, _ = geometry_from_content(problem, content)
        except ValueError as exc:
            print(f"{i:<5}{'invalid':<10}  ({exc})")
            continue
        if centerline_escapes_map(geom):
            # Not a driving failure: the track is unlappable by construction.
            print(f"{i:<5}{'oob':<10}  (centerline leaves the map)")
            continue
        env = env_for_geometry(geom)
        r = evaluate_track(env, policy, "generated")
        lap = f"{r['laptime_s']}s" if r["laptime_s"] != "" else "-"
        print(f"{i:<5}{r['outcome']:<10}{r['progress_frac']:>6.2f}"
              f"{r['offroad_frac']:>9.2f}{lap:>9}{r['mean_speed']:>8.1f}"
              f"{r['steer_jerk']:>7.3f}")


if __name__ == "__main__":
    _main()
