"""Loader for the TUMFTM racetrack-database circuits.

Reads the raw CSV files (columns: x_m, y_m, w_tr_right_m, w_tr_left_m) and
converts them to the benchmark's world units.

Scale: the benchmark's simulation world is metric, 1 world unit = 1 metre
(the engine integrates position with velocity in m/s; the 750x750 map and
the 16 m track width are metres; problem.py's PX_PER_M = 5.0 is only the
world-to-pixel factor used when rendering images). So the circuits are
imported at 1:1 — every circuit at the exact same scale, with real corner
radii and lengths. Each circuit is translated so its bounding box starts at
the benchmark's edge margin; real circuits span up to ~2.2 km, so each track
carries its own `map_size` (bounding square + margins) instead of the fixed
750 m benchmark map.

The returned track dict plugs straight into the benchmark simulation:

    from track_loader import load_track
    track = load_track("Spa")
    agent = SteeringAgent(track["points"], track_width=track["track_width"])

Run as a script to print a stats table for all circuits, or with --save to
write the converted centerlines to model_training/tracks_scaled/.
"""

import argparse
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
TRACKS_DIR = os.path.normpath(os.path.join(_HERE, "..", "racetrack-database-master", "tracks"))
RACELINES_DIR = os.path.normpath(os.path.join(_HERE, "..", "racetrack-database-master", "racelines"))

# Benchmark dimensions (mirror racing/problem.py: track_width 16 m and the
# out-of-bounds margin of half a track width plus 2). One world unit is one
# metre, same as the benchmark's simulation world.
METERS_PER_UNIT = 1.0
TRACK_WIDTH = 16.0
MARGIN = TRACK_WIDTH * 0.5 + 2.0
# Distance from the world edge to the centerline bounding box. Must exceed
# MARGIN (where the out-of-world check fires) by more than the road half
# width plus the environment's off-track allowance (3 half-widths = 24 m),
# so a car at the circuit's extremes always trips off_track before off_map.
PADDING = MARGIN + 3.5 * (TRACK_WIDTH * 0.5)


def list_tracks(tracks_dir=TRACKS_DIR):
    """Sorted circuit names (CSV files without extension)."""
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(tracks_dir)
        if f.lower().endswith(".csv")
    )


def _read_csv(path):
    """Read a TUMFTM CSV (comment header starting with '#') into a float array."""
    return np.loadtxt(path, delimiter=",", comments="#", dtype=float)


def _polyline_length(points):
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def _close_loop(points, tol=1.0):
    """Append the first point if the polyline does not already end there.
    The benchmark treats reaching the end of the centerline as finishing
    the lap, so the loop must close explicitly."""
    if np.linalg.norm(points[-1] - points[0]) > tol:
        points = np.vstack([points, points[0]])
    return points


def resample(points, spacing):
    """Resample a closed polyline to (roughly) uniform arc-length spacing."""
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = arc[-1]
    n = max(int(round(total / spacing)), 8)
    targets = np.linspace(0.0, total, n + 1)
    x = np.interp(targets, arc, points[:, 0])
    y = np.interp(targets, arc, points[:, 1])
    return np.column_stack([x, y])


def load_track(name, tracks_dir=TRACKS_DIR, resample_spacing=None):
    """Load one circuit at 1:1 scale (world units = metres), translated so
    its bounding box starts at the benchmark edge margin.

    Returns a dict:
        points        Nx2 centerline in world units (= metres), closed loop
        track_width   constant benchmark width (16.0)
        map_size      side of the square world holding this track (bounding
                      box max side + a margin on each edge)
        name          circuit name
        length        centerline length (= real length in metres)
        real_width_m  mean real total road width (left + right), metres
    """
    raw = _read_csv(os.path.join(tracks_dir, name + ".csv"))
    pts = raw[:, :2] / METERS_PER_UNIT
    widths = raw[:, 2] + raw[:, 3]

    # Translate so the bounding box starts at (PADDING, PADDING).
    pts = pts - pts.min(axis=0) + PADDING
    span = pts.max(axis=0) - pts.min(axis=0)
    map_size = float(max(span[0], span[1]) + 2.0 * PADDING)

    pts = _close_loop(pts, tol=1.0)
    if resample_spacing is not None:
        pts = resample(pts, resample_spacing)

    return {
        "name": name,
        "points": pts,
        "track_width": TRACK_WIDTH,
        "map_size": map_size,
        "length": _polyline_length(pts),
        "real_width_m": float(np.mean(widths)),
    }


def load_raceline(name, tracks_dir=TRACKS_DIR, racelines_dir=RACELINES_DIR):
    """Load the TUMFTM precomputed raceline for a circuit, transformed with
    the SAME translation as its track (so both line up)."""
    track_pts = _read_csv(os.path.join(tracks_dir, name + ".csv"))[:, :2] / METERS_PER_UNIT
    offset = track_pts.min(axis=0) - PADDING
    line = _read_csv(os.path.join(racelines_dir, name + ".csv"))[:, :2] / METERS_PER_UNIT
    return line - offset


def load_all(tracks_dir=TRACKS_DIR, resample_spacing=None):
    """Load every circuit. Returns {name: track dict}."""
    return {
        name: load_track(name, tracks_dir=tracks_dir,
                         resample_spacing=resample_spacing)
        for name in list_tracks(tracks_dir)
    }


def _main():
    parser = argparse.ArgumentParser(description="Convert TUMFTM circuits to benchmark world units")
    parser.add_argument("--save", action="store_true",
                        help="write converted centerlines to model_training/tracks_scaled/")
    parser.add_argument("--spacing", type=float, default=None,
                        help="optional resample spacing in world units")
    args = parser.parse_args()

    print(f"scale: 1 world unit = {METERS_PER_UNIT:.0f} m (same as the benchmark; "
          f"PX_PER_M=5 applies only when rendering)\n")

    header = f"{'track':<16}{'length':>9}{'map_size':>10}{'points':>8}{'real width':>12}"
    print(header)
    print("-" * len(header))
    for name in list_tracks():
        t = load_track(name, resample_spacing=args.spacing)
        print(f"{name:<16}{t['length']:>8.0f}m{t['map_size']:>9.0f}m"
              f"{len(t['points']):>8}{t['real_width_m']:>11.1f}m")

    if args.save:
        out_dir = os.path.normpath(os.path.join(_HERE, "..", "tracks_scaled"))
        os.makedirs(out_dir, exist_ok=True)
        for name in list_tracks():
            t = load_track(name, resample_spacing=args.spacing)
            np.savetxt(os.path.join(out_dir, name + ".csv"), t["points"],
                       delimiter=",", header="x,y", comments="# ", fmt="%.3f")
        print(f"\nsaved converted centerlines to {out_dir}")


if __name__ == "__main__":
    _main()
