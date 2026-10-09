"""Phase 2 baseline: run the scripted heuristic over every circuit and record
what it achieves. This is the bar a trained driver has to beat, and the
comparison row in the thesis.

The same script scores a trained model (--model), so the two runs are
measured by identical code on identical episodes.

Run:
    python baseline.py                          # heuristic, all circuits
    python baseline.py --model ../runs/ppo_driver/best/best_model.zip
    python baseline.py --out ../results/baseline.csv

Every episode is deterministic (fixed start at the beginning of the
centerline, forward, unmirrored), so the table is reproducible.
"""

import argparse
import csv
import math
import os

import numpy as np

import track_loader
from racing_env import RacingEnv
from sanity_check import heuristic_policy

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.normpath(os.path.join(_HERE, "..", "results", "baseline.csv"))

FIELDS = ["track", "outcome", "finished", "progress_frac", "offroad_frac",
          "laptime_steps", "laptime_s", "length_m", "mean_speed",
          "max_speed", "grip_used", "slip_p95", "steer_jerk", "reward"]

_G = 9.81


def _grip_and_slip(env):
    """Fraction of tire grip in use, and body slip angle, this step.

    Grip comes from the TIRE FORCE, not from yaw_rate * speed. The latter
    equals lateral acceleration only in steady cornering: once the car
    slides it rotates about its own axis while travelling elsewhere, so
    the yaw proxy reads spin as if it were grip and reports impossible
    values (measured up to 5.9 g on a 1.3 g tire). That artifact is what
    made the engine look broken; the tires were always inside their limit.
    """
    car = env.car_state
    engine = getattr(env._physics, "_engine", None)
    forces = getattr(engine, "debug_forces", None)
    # Against the car's own weight and tyre mu (1534 kg, 1.3).
    grip = (abs(forces[6]) / (engine.mass * _G * engine.tire_mu)
            if forces else float("nan"))
    slip = abs(math.degrees(math.atan2(car.v_lateral, max(car.v_forward, 0.5))))
    return grip, slip


def evaluate_track(env, policy, name):
    """Drive one deterministic episode and summarize it."""
    obs, info = env.reset(options={"track": name})
    geom = env.geometry
    total = 0.0
    speeds, steers, grips, slips = [], [], [], []
    done = False
    while not done:
        action = policy(obs)
        speeds.append(env.car_state.v_forward)
        steers.append(float(action[0]))
        g, s = _grip_and_slip(env)
        grips.append(g)
        slips.append(s)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        done = terminated or truncated

    outcome = ("lap" if info.get("lap_complete")
               else "off_map" if info.get("off_map")
               else "off_track" if info.get("off_track")
               else "stuck" if info.get("stuck") else "timeout")
    steps = info.get("laptime_steps")
    dt = env.physics_dt
    # Mean absolute change in steering command per step: a proxy for how
    # smooth the driving is. A policy that oscillates scores badly here even
    # when its laptime looks fine.
    jerk = float(np.mean(np.abs(np.diff(steers)))) if len(steers) > 1 else 0.0

    return {
        "track": name,
        "outcome": outcome,
        "finished": int(outcome == "lap"),
        "progress_frac": round(info.get("progress_frac", 0.0), 4),
        "offroad_frac": round(info.get("offroad_frac", 0.0), 4),
        "laptime_steps": steps if steps is not None else "",
        "laptime_s": round(steps * dt, 1) if steps is not None else "",
        "length_m": round(geom.length, 1),
        "mean_speed": round(float(np.mean(speeds)), 2),
        "max_speed": round(float(np.max(speeds)), 2),
        # 95th percentile of grip usage: how hard the policy actually
        # drives. A clean lap at 50% grip is a slow lap, not a good one.
        "grip_used": round(float(np.nanpercentile(grips, 95)), 3),
        "slip_p95": round(float(np.percentile(slips, 95)), 2),
        "steer_jerk": round(jerk, 4),
        "reward": round(total, 1),
    }


def make_policy(model_path):
    if model_path is None:
        return heuristic_policy, "heuristic"
    from stable_baselines3 import PPO
    model = PPO.load(model_path)

    def _policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action
    return _policy, os.path.splitext(os.path.basename(model_path))[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="trained SB3 model to score (default: heuristic)")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT,
                        help="CSV output path")
    parser.add_argument("--tracks", type=str, nargs="*", default=None,
                        help="subset of circuits (default: all)")
    args = parser.parse_args()

    policy, policy_name = make_policy(args.model)
    names = args.tracks if args.tracks else track_loader.list_tracks()
    env = RacingEnv(track_names=names, randomize=False, seed=0)

    header = (f"{'track':<16}{'outcome':<10}{'prog':>6}{'offroad':>9}"
              f"{'laptime':>9}{'mean v':>8}{'grip':>7}{'slip':>7}"
              f"{'jerk':>7}{'reward':>9}")
    print(f"policy: {policy_name}\n")
    print(header)
    print("-" * len(header))

    rows = []
    for name in names:
        r = evaluate_track(env, policy, name)
        rows.append(r)
        lap = f"{r['laptime_s']}s" if r["laptime_s"] != "" else "-"
        print(f"{r['track']:<16}{r['outcome']:<10}{r['progress_frac']:>6.2f}"
              f"{r['offroad_frac']:>9.2f}{lap:>9}{r['mean_speed']:>8.1f}"
              f"{r['grip_used']:>7.2f}{r['slip_p95']:>7.1f}"
              f"{r['steer_jerk']:>7.3f}{r['reward']:>9.1f}")

    finished = sum(r["finished"] for r in rows)
    print("-" * len(header))
    print(f"finished {finished}/{len(rows)}   "
          f"mean offroad {np.mean([r['offroad_frac'] for r in rows]):.3f}   "
          f"mean progress {np.mean([r['progress_frac'] for r in rows]):.3f}   "
          f"mean speed {np.mean([r['mean_speed'] for r in rows]):.1f} m/s   "
          f"mean grip used {np.mean([r['grip_used'] for r in rows]):.2f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
