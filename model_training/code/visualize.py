"""Draw a driven lap: track outline plus the line the car actually took.

Used to see WHY a policy scores what it scores. A reward number says a lap
was bad; the picture says whether the car cut the infield, sat wide through
every sweeper, or oscillated down the straights. The same plots serve as
thesis figures.

Run:
    python visualize.py --track Spa                 # heuristic policy
    python visualize.py --model ../runs/ppo_driver/best/best_model.zip
    python visualize.py --all --out ../figures      # every circuit, to disk

Without --out the figure opens in a window.
"""

import argparse
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import track_loader
from baseline import make_policy
from racing_env import RacingEnv


def rollout(env, policy, track=None, options=None):
    """Drive one episode, recording the path. Returns (trajectory, info)."""
    opts = dict(options or {})
    if track is not None:
        opts["track"] = track
    obs, info = env.reset(options=opts)
    geom = env.geometry
    path, speeds = [], []
    done = False
    while not done:
        # Reconstruct world position from the env's own physics state, so the
        # plot shows exactly what the environment scored.
        car = env.car_state
        path.append((car.x, car.y))
        speeds.append(car.v_forward)
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        done = terminated or truncated
    car = env.car_state
    path.append((car.x, car.y))
    speeds.append(car.v_forward)
    return np.asarray(path), np.asarray(speeds), geom, info


def _track_edges(geom):
    """Left and right road edges from the centerline and half width."""
    pts = geom.points
    tan = geom.seg_tan
    normal = np.column_stack([-tan[:, 1], tan[:, 0]])   # left of travel
    left = pts + normal * geom.half_width
    right = pts - normal * geom.half_width
    return np.vstack([left, left[:1]]), np.vstack([right, right[:1]])


def plot_run(path, speeds, geom, info, ax=None, title=None):
    """Plot one recorded lap: road edges, centerline, speed-colored path."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    left, right = _track_edges(geom)
    ax.plot(left[:, 0], left[:, 1], color="0.55", lw=0.8)
    ax.plot(right[:, 0], right[:, 1], color="0.55", lw=0.8)
    ring = np.vstack([geom.points, geom.points[:1]])
    ax.plot(ring[:, 0], ring[:, 1], color="0.8", lw=0.6, ls="--")

    # Color the driven line by speed: braking points and the top-speed
    # stretches are the first thing to check when a lap looks wrong.
    seg = np.stack([path[:-1], path[1:]], axis=1)
    lc = LineCollection(seg, cmap="viridis", linewidths=1.8)
    lc.set_array(speeds[:-1])
    lc.set_clim(0.0, 85.5)     # the engine's max_speed
    ax.add_collection(lc)
    plt.colorbar(lc, ax=ax, label="speed (m/s)", shrink=0.75)

    ax.plot(*path[0], "o", color="tab:green", ms=7, label="start")
    end = ("lap" if info.get("lap_complete")
           else "off_map" if info.get("off_map")
           else "off_track" if info.get("off_track")
           else "stuck" if info.get("stuck") else "timeout")
    ax.plot(*path[-1], "x", color="tab:red", ms=8, mew=2, label=f"end ({end})")

    if title is None:
        title = (f"{info.get('track', '?')}  |  {end}  |  "
                 f"progress {info.get('progress_frac', 0):.2f}  "
                 f"offroad {info.get('offroad_frac', 0):.2f}")
        if "laptime_steps" in info:
            title += f"  |  {info['laptime_steps']} steps"
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlim(0, geom.map_size)
    ax.set_ylim(0, geom.map_size)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("metres")
    return ax


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", type=str, default="Spa")
    parser.add_argument("--all", action="store_true",
                        help="render every circuit instead of one")
    parser.add_argument("--model", type=str, default=None,
                        help="path to a trained SB3 model (default: heuristic)")
    parser.add_argument("--out", type=str, default=None,
                        help="directory to save PNGs into (default: show window)")
    args = parser.parse_args()

    if args.out:
        matplotlib.use("Agg")
        os.makedirs(args.out, exist_ok=True)

    policy, policy_name = make_policy(args.model)
    names = track_loader.list_tracks() if args.all else [args.track]
    env = RacingEnv(track_names=names, randomize=False, seed=0)

    for name in names:
        path, speeds, geom, info = rollout(env, policy, track=name)
        ax = plot_run(path, speeds, geom, info)
        ax.figure.suptitle(f"policy: {policy_name}", fontsize=9, y=0.98)
        if args.out:
            out = os.path.join(args.out, f"{name}_{policy_name}.png")
            ax.figure.savefig(out, dpi=130, bbox_inches="tight")
            plt.close(ax.figure)
            print(f"{name:<16} {info.get('progress_frac', 0):.2f} -> {out}")
        else:
            plt.show()


if __name__ == "__main__":
    main()
