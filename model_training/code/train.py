"""Train a PPO driver on the rescaled real circuits.

The car is selected with --physics: 'benchmark' (the engine the GA quality
function scores laps with) or 'legacy' (the separate simcade model).

Requires stable-baselines3 + torch (not installed yet):
    pip install stable-baselines3 torch tensorboard

Run:
    python train.py --steps 2000000 --n-envs 8

Design choices:
- 5 circuits are held out entirely for evaluation (never trained on), so the
  reported score measures generalization to unseen tracks, which is what the
  generated tracks always are.
- Training envs randomize track / start / direction / mirror every reset;
  eval envs are deterministic (fixed track, forward, no mirror).
- Checkpoints every 100k steps so a crash costs little.
"""

import argparse
import os
import sys
import time

import numpy as np

from racing_env import RacingEnv, make_physics
import track_loader

# Held-out circuits: varied and well known, used only for evaluation.
EVAL_TRACKS = ["Spa", "Silverstone", "Suzuka", "Monza", "Montreal"]

_HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.normpath(os.path.join(_HERE, "..", "runs"))


def train_track_names():
    return [n for n in track_loader.list_tracks() if n not in EVAL_TRACKS]


def make_env(track_names, randomize, seed, physics="benchmark"):
    def _thunk():
        # Monitor records episode return and length. Without it SB3 logs no
        # ep_rew_mean/ep_len_mean during training and EvalCallback reports
        # inaccurate episode rewards, so the run gives no feedback at all.
        from stable_baselines3.common.monitor import Monitor
        # One physics instance per env: each holds its own car state, so a
        # shared one would have twelve parallel envs stepping the same car.
        return Monitor(RacingEnv(track_names=track_names,
                                 randomize=randomize, seed=seed,
                                 physics=make_physics(physics)))
    return _thunk


def _progress_line_class():
    """One self-updating status line instead of PPO's per-rollout table.

    Defined lazily so this module still imports without stable-baselines3.
    """
    from stable_baselines3.common.callbacks import BaseCallback

    class ProgressLine(BaseCallback):
        def __init__(self, total):
            super().__init__()
            self.total = total
            self.t0 = time.time()

        def _on_step(self):
            return True

        def _on_rollout_end(self):
            n = self.num_timesteps
            el = time.time() - self.t0
            fps = n / max(el, 1e-9)
            eta = (self.total - n) / max(fps, 1e-9)
            buf = self.model.ep_info_buffer
            rew = np.mean([e["r"] for e in buf]) if buf else float("nan")
            ln = np.mean([e["l"] for e in buf]) if buf else float("nan")
            # \r returns to the start of the line, so each update overwrites
            # the previous one and the console keeps a single live row.
            sys.stdout.write(
                f"\r{100.0 * n / self.total:5.1f}%  {n:>9,}/{self.total:,}  "
                f"reward {rew:8.1f}  ep_len {ln:6.0f}  "
                f"{fps:5.0f} steps/s  eta {eta / 60:5.1f} min   ")
            sys.stdout.flush()

    return ProgressLine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", type=str, default="ppo_driver")
    parser.add_argument("--physics", type=str, default="benchmark",
                        choices=["benchmark", "legacy"],
                        help="car to train on; 'benchmark' is the one the GA "
                             "quality function scores laps with")
    args = parser.parse_args()

    # Imported here so the env/plan work without the RL stack installed.
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

    ProgressLine = _progress_line_class()

    os.makedirs(RUNS_DIR, exist_ok=True)
    run_dir = os.path.join(RUNS_DIR, args.name)

    train_names = train_track_names()
    train_vec = SubprocVecEnv(
        [make_env(train_names, True, args.seed + i, args.physics)
         for i in range(args.n_envs)])
    eval_vec = SubprocVecEnv(
        [make_env(EVAL_TRACKS, False, args.seed + 1000, args.physics)])

    model = PPO(
        "MlpPolicy", train_vec,
        policy_kwargs=dict(net_arch=[128, 128]),
        n_steps=2048, batch_size=4096, gae_lambda=0.95, gamma=0.995,
        learning_rate=3e-4, ent_coef=0.01, clip_range=0.2,
        tensorboard_log=os.path.join(run_dir, "tb"),
        seed=args.seed, verbose=0,
    )

    callbacks = [
        ProgressLine(args.steps),
        # n_eval_episodes = number of held-out circuits: the eval env is
        # non-randomized, so it walks EVAL_TRACKS one circuit per reset and
        # the five episodes cover the five circuits exactly once each.
        EvalCallback(eval_vec, best_model_save_path=os.path.join(run_dir, "best"),
                     log_path=os.path.join(run_dir, "eval"),
                     eval_freq=25_000, n_eval_episodes=len(EVAL_TRACKS),
                     deterministic=True),
        CheckpointCallback(save_freq=100_000, save_path=os.path.join(run_dir, "ckpt"),
                           name_prefix="ppo"),
    ]

    model.learn(total_timesteps=args.steps, callback=callbacks)
    model.save(os.path.join(run_dir, "final"))
    print()                                   # close the live status line
    print("saved", os.path.join(run_dir, "final.zip"))


if __name__ == "__main__":
    main()
