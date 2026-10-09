"""Train a PPO driver on the real circuits.

The car is the benchmark engine, the one the GA quality function scores laps
with, so a policy trained here is driving the vehicle the benchmark simulates.

Requires stable-baselines3 and torch:
    pip install stable-baselines3 torch tensorboard

Run:
    python train.py --steps 2000000 --n-envs 8

Design choices:
- 5 circuits are held out entirely for evaluation (never trained on), so the
  reported score measures generalization to unseen tracks, which is what the
  generated tracks always are.
- No generated tracks in training.  The policy is the benchmark's judge of
  all four representations; trained on their output it would drive best on
  whichever representation it saw most, and the simulation terms would
  measure that rather than the tracks.
- Training envs randomize track / start / direction / mirror every reset;
  eval envs are deterministic (fixed track, forward, no mirror).
- Checkpoints every 100k calls of the vectorised env, 100k x n_envs
  timesteps (1.2M at 12 envs), so a crash costs little.
"""

import argparse
import os
import sys
import time

import numpy as np

from racing_env import RacingEnv
import track_loader

# Held-out circuits: varied and well known, used only for evaluation.
EVAL_TRACKS = ["Spa", "Silverstone", "Suzuka", "Monza", "Montreal"]

_HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.normpath(os.path.join(_HERE, "..", "runs"))


def train_track_names():
    return [n for n in track_loader.list_tracks() if n not in EVAL_TRACKS]


def make_env(track_names, randomize, seed):
    def _thunk():
        # Monitor records episode return and length. Without it SB3 logs no
        # ep_rew_mean/ep_len_mean during training and EvalCallback reports
        # inaccurate episode rewards, so the run gives no feedback at all.
        from stable_baselines3.common.monitor import Monitor
        # One physics instance per env: each holds its own car state, so a
        # shared one would have twelve parallel envs stepping the same car.
        return Monitor(RacingEnv(track_names=track_names,
                                 randomize=randomize, seed=seed))
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


def _circuit_selection_class():
    """Checkpoint selection on the benchmark's own acceptance measure.

    EvalCallback keeps the checkpoint with the highest evaluation RETURN on
    five circuits, which is not what the benchmark needs from its driver:
    neighbouring run16 checkpoints differed by 4 to 12 circuits with off-road
    time at similar returns.  Every `every` steps this callback saves the
    policy, drives all 24 circuits with it through the benchmark problem (the
    same RacingProblem path the quality function scores laps with), and keeps
    the policy that ranks best on, in order: laps finished, fewest circuits
    with any off-road time, lowest mean off-road share, highest mean speed.
    Each line of select.csv records one evaluation."""
    from stable_baselines3.common.callbacks import BaseCallback

    class CircuitSelection(BaseCallback):
        def __init__(self, run_dir, every):
            super().__init__()
            self.dir = os.path.join(run_dir, "selected")
            self.every = int(every)
            self.next_at = self.every
            self.best = None
            os.makedirs(self.dir, exist_ok=True)

        def _on_step(self):
            if self.num_timesteps >= self.next_at:
                self.next_at += self.every
                self._evaluate()
            return True

        def _evaluate(self):
            import gc
            sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..", "..")))
            from pcg_benchmark.probs.racing.problem import RacingProblem
            cand = os.path.join(self.dir, "candidate.zip")
            self.model.save(cand)
            p = RacingProblem(width=2600.0, height=2600.0, rl_policy_path=cand)
            fin, off, speeds = 0, [], []
            for name in track_loader.list_tracks():
                if name == "IMS":
                    continue
                pts = np.asarray(track_loader.load_track(name, resample_spacing=5.0)["points"], float)
                info = p.info({"track_points": pts})
                fin += bool(info["finished"])
                off.append(float(info["offroad_frac"]))
                if info["finished"] and info.get("steps"):
                    speeds.append(info["total_length"] / (info["steps"] * p._engine.time_step))
            off = np.array(off)
            key = (fin, -int((off > 0).sum()), -float(off.mean()),
                   float(np.mean(speeds)) if speeds else 0.0)
            better = self.best is None or key > self.best
            with open(os.path.join(self.dir, "select.csv"), "a") as f:
                f.write("%d,%d,%d,%.5f,%.2f,%d\n" % (self.num_timesteps, fin, (off > 0).sum(),
                                                     off.mean(), key[3], better))
            if better:
                self.best = key
                os.replace(cand, os.path.join(self.dir, "policy.zip"))
            del p
            gc.collect()

    return CircuitSelection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", type=str, default="ppo_driver")
    # Start from a saved policy's weights instead of from scratch.  From
    # scratch the benchmark car spends its first ~2.7M steps stuck at the
    # start before it learns to drive (runs/run13/eval: return -45.1 at 101
    # steps up to 2.7M), and run13 stopped at 4.8M with its evaluation
    # return still rising, so continuing from it spends the budget on
    # driving rather than on the bootstrap.
    parser.add_argument("--init", type=str, default=None,
                        help="checkpoint .zip to start from")
    # Circuit selection interval (_circuit_selection_class): one evaluation
    # drives the 24 circuits, about 15 s, so every 600k steps costs well
    # under 1% of training time.
    parser.add_argument("--select-every", type=int, default=600_000)
    args = parser.parse_args()

    # Imported here so the env/plan work without the RL stack installed.
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

    ProgressLine = _progress_line_class()
    CircuitSelection = _circuit_selection_class()

    os.makedirs(RUNS_DIR, exist_ok=True)
    run_dir = os.path.join(RUNS_DIR, args.name)

    train_names = train_track_names()
    train_vec = SubprocVecEnv(
        [make_env(train_names, True, args.seed + i)
         for i in range(args.n_envs)])
    eval_vec = SubprocVecEnv(
        [make_env(EVAL_TRACKS, False, args.seed + 1000)])

    # PPO (Schulman et al. 2017) as implemented by stable-baselines3.
    # Provenance, per setting:
    # - n_steps 2048, learning_rate 3e-4, gae_lambda 0.95 and SB3's default
    #   10 epochs: the paper's Table 3 (MuJoCo, 1M timesteps), unchanged.
    #   clip_range 0.2: the paper's best clipping value (its Table 1, 0.82
    #   normalised score against 0.76 at 0.1 and 0.70 at 0.3).
    # - gamma 0.995 against the paper's 0.99: an effective horizon of 200
    #   steps (20 s) against 100 (10 s).  A modelling choice, so the value
    #   sees a braking zone and the corner after it; not swept here.  The
    #   lap bonus is paid when the episode ends, so it counts at full value
    #   only in the last ~200 steps before the line (0.995^200 = 0.37) and is
    #   worth ~1e-4 from the start of a 2000-step lap; over the lap, time is
    #   priced by TIME_COST per step (thesis/HISTORY.md, "Training plan",
    #   pace push).
    # - batch_size 4096 (the minibatch): 12 envs x 2048 steps = 24576 per
    #   rollout, so 6 minibatches per epoch.  A modelling choice sized to the
    #   rollout, not swept; the paper's 64 was for one env's 2048 steps.
    # - ent_coef 0.01: at 0.005 a pilot collapsed to standing still before
    #   it discovered driving (thesis/HISTORY.md, "Training plan", bootstrap).
    # - net_arch [128, 128], separate policy and value networks: twice SB3's
    #   64-unit default for a 21-float observation.  A modelling choice.
    # No observation normalisation wrapper: racing_env scales every
    # observation to about [-4, 4] by physical constants, which a running
    # mean would replace with statistics of whichever circuits were drawn.
    ppo_kwargs = dict(
        n_steps=2048, batch_size=4096, gae_lambda=0.95, gamma=0.995,
        learning_rate=3e-4, ent_coef=0.01, clip_range=0.2,
        tensorboard_log=os.path.join(run_dir, "tb"),
        seed=args.seed, verbose=0,
    )
    if args.init:
        # Same hyperparameters as a fresh run, so the only difference is
        # where the weights start.
        model = PPO.load(args.init, env=train_vec, **ppo_kwargs)
    else:
        model = PPO("MlpPolicy", train_vec,
                    policy_kwargs=dict(net_arch=[128, 128]), **ppo_kwargs)

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
        CircuitSelection(run_dir, args.select_every),
    ]

    model.learn(total_timesteps=args.steps, callback=callbacks)
    model.save(os.path.join(run_dir, "final"))
    print()                                   # close the live status line
    print("saved", os.path.join(run_dir, "final.zip"))


if __name__ == "__main__":
    main()
