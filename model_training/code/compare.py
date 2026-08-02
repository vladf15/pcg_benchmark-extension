"""Compare a trained policy against the scripted heuristic, circuit by circuit.

Both are scored by the same code (baseline.evaluate_track) on the same
deterministic episodes, so the only difference is the policy. The headline
question is laptime on circuits the agent never trained on.

    python compare.py --model ../runs/pilot/best/best_model.zip
    python compare.py --model ... --all        # all 25, not just the holdout
"""

import argparse
import os

import numpy as np

import track_loader
import train
from baseline import evaluate_track, make_policy
from racing_env import RacingEnv
from sanity_check import heuristic_policy


def score(policy, names):
    env = RacingEnv(track_names=names, randomize=False, seed=0)
    return {n: evaluate_track(env, policy, n) for n in names}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--all", action="store_true",
                   help="score all 25 circuits (default: the 5 held out)")
    args = p.parse_args()

    names = track_loader.list_tracks() if args.all else train.EVAL_TRACKS
    agent, label = make_policy(args.model)
    a = score(agent, names)
    h = score(heuristic_policy, names)

    seen = "ALL 25 circuits" if args.all else "HELD-OUT circuits (never trained on)"
    print(f"agent: {label}   vs   scripted heuristic   [{seen}]\n")
    head = (f"{'track':<14}{'heuristic':>19}{'agent':>19}{'laptime':>10}")
    print(head)
    print(f"{'':<14}{'lap      grip':>19}{'lap      grip':>19}{'delta':>10}")
    print("-" * len(head))

    deltas, agent_laps, heur_laps = [], 0, 0
    for n in names:
        ra, rh = a[n], h[n]
        agent_laps += ra["finished"]
        heur_laps += rh["finished"]
        ta = f"{ra['laptime_s']}s" if ra["laptime_s"] != "" else ra["outcome"]
        th = f"{rh['laptime_s']}s" if rh["laptime_s"] != "" else rh["outcome"]
        if ra["laptime_s"] != "" and rh["laptime_s"] != "":
            d = (float(ra["laptime_s"]) - float(rh["laptime_s"])) / float(rh["laptime_s"])
            deltas.append(d)
            ds = f"{d:+.1%}"
        else:
            ds = "-"
        print(f"{n:<14}{th:>11}{rh['grip_used']:>8.2f}"
              f"{ta:>11}{ra['grip_used']:>8.2f}{ds:>10}")

    print("-" * len(head))
    print(f"completed laps   heuristic {heur_laps}/{len(names)}   "
          f"agent {agent_laps}/{len(names)}")
    if deltas:
        m = float(np.mean(deltas))
        verdict = "FASTER" if m < 0 else "slower"
        print(f"mean laptime     agent is {abs(m):.1%} {verdict} "
              f"(negative delta = agent quicker)")
    print(f"mean grip used   heuristic "
          f"{np.mean([h[n]['grip_used'] for n in names]):.2f}   "
          f"agent {np.mean([a[n]['grip_used'] for n in names]):.2f}")
    print(f"mean offroad     heuristic "
          f"{np.mean([h[n]['offroad_frac'] for n in names]):.3f}   "
          f"agent {np.mean([a[n]['offroad_frac'] for n in names]):.3f}")
    print(f"mean steer jerk  heuristic "
          f"{np.mean([h[n]['steer_jerk'] for n in names]):.3f}   "
          f"agent {np.mean([a[n]['steer_jerk'] for n in names]):.3f}")


if __name__ == "__main__":
    main()
