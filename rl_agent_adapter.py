"""Drive a trained RL policy inside view_track, next to the scripted agent.

view_track's loop is `action = agent.act(state)` then
`state = problem._engine.step(action)`, where `state` is the benchmark
engine's 5-float [x, y, angle, speed, steering]. A trained policy cannot be
dropped in directly, because it was trained in model_training/ against:

  - an 18-float observation, not the benchmark's 5-float state, and
  - a 21-float observation built from track geometry (lateral offset,
    heading error, thirteen speed-scaled curvature bands ahead), not the
    5-float state.

`RLAgent` bridges both. It owns its own engine instance and its own
TrackGeometry for the current track, drives that engine from the policy,
and reports the resulting pose back in the benchmark's 5-float format so
the viewer can draw it unchanged. The benchmark engine is left untouched:
its step() result is simply ignored while an RL agent is selected.

Usage inside view_track (already wired):
    agent = RLAgent(model_path, problem._curve_points, problem._track_width)
    ...unchanged loop...

Standalone check:
    python rl_agent_adapter.py --model path/to/best_model.zip
"""

from __future__ import annotations

import os
import sys

import numpy as np

# model_training/code holds the env, geometry and physics the policy expects.
_MT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "model_training", "code"))
if _MT not in sys.path:
    sys.path.append(_MT)


class RLAgent:
    """Scripted-agent-compatible wrapper around a trained SB3 policy."""

    def __init__(self, model_path, curve_points, track_width, map_size=None):
        from stable_baselines3 import PPO
        from pcg_benchmark.probs.racing.engine import CarPhysicsEngine
        from track_geometry import TrackGeometry
        from racing_env import RacingEnv

        self._model = PPO.load(model_path)
        self._env_cls = RacingEnv          # source of the observation layout
        pts = np.asarray(curve_points, dtype=float)
        if map_size is None:
            map_size = float(np.max(pts) + 50.0)
        self._geom = TrackGeometry(pts, float(track_width), map_size)
        self._engine = CarPhysicsEngine(start_position=(0.0, 0.0))
        self._max_speed = self._engine.max_speed
        self._max_steer = self._engine.max_steering
        self.name = os.path.basename(str(model_path))
        self.last_lookahead_point = None       # viewer draws this if present
        self.reset()

    # ── lifecycle ─────────────────────────────────────────────────────────

    def reset(self):
        """Place the car at the start of the centerline, facing along it."""
        pos, heading = self._geom.pose_at_s(0.0)
        self._engine.reset()
        self._engine.position = np.array(pos, dtype=float)
        self._engine.angle = float(heading)
        self._engine.start_position = np.array(pos, dtype=float)
        self._engine.start_angle = float(heading)
        self._hint, self._s, _, _ = self._geom.project(pos)
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._last_horizon = 0.0
        return self.state()

    def state(self):
        """Current pose in the benchmark's [x, y, angle, speed, steering]."""
        e = self._engine
        return np.array([e.position[0], e.position[1], e.angle,
                         float(np.hypot(*e.velocity)), e.steering_angle],
                        dtype=float)

    # ── the interface view_track calls ────────────────────────────────────

    def act(self, car_state):
        """Advance the RL car one step and return the action it took.

        `car_state` is ignored: the benchmark engine's state describes a
        different car driven by a different physics model. This agent
        integrates its own engine instance instead, so the two never
        interfere. The viewer reads the resulting pose from `state()`.
        """
        obs = self._observe()
        action, _ = self._model.predict(obs, deterministic=True)
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self._engine.step({"steering": float(action[0]),
                           "throttle": float(action[1])})
        self._prev_action = action

        # Keep the arc-length position current for the next observation.
        e = self._engine
        self._hint, self._s, _, _ = self._geom.project(
            (e.position[0], e.position[1]), self._hint)
        # Show where the policy is looking: the far end of its curvature
        # bands, which now stretches and shrinks with speed.
        pt, _ = self._geom.pose_at_s(self._s + self._last_horizon)
        self.last_lookahead_point = (float(pt[0]), float(pt[1]))
        return {"steering": float(action[0]), "throttle": float(action[1])}

    # ── observation, identical to the training environment ────────────────

    def _observe(self):
        e = self._engine
        g = self._geom
        cos_a, sin_a = np.cos(e.angle), np.sin(e.angle)
        v_fwd = cos_a * e.velocity[0] + sin_a * e.velocity[1]
        v_lat = -sin_a * e.velocity[0] + cos_a * e.velocity[1]
        _, _, lateral, heading_ref = g.project((e.position[0], e.position[1]),
                                               self._hint)
        heading_err = (e.angle - heading_ref + np.pi) % (2.0 * np.pi) - np.pi

        edges = self._env_cls.lookahead_edges(v_fwd)
        self._last_horizon = float(edges[-1])

        obs = np.empty(8 + len(edges) - 1, dtype=np.float32)
        obs[0] = v_fwd / self._max_speed
        obs[1] = v_lat / 10.0
        obs[2] = e.yaw_rate / 2.0
        obs[3] = e.steering_angle / self._max_steer
        obs[4] = lateral / g.half_width
        obs[5] = heading_err / np.pi
        obs[6] = self._prev_action[0]
        obs[7] = self._prev_action[1]
        obs[8:] = g.curvature_ahead(self._s, edges) * 10.0
        return np.clip(obs, -4.0, 4.0)


def _main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    import track_loader
    t = track_loader.load_track("Monza")
    ag = RLAgent(args.model, t["points"], t["track_width"], t["map_size"])
    for i in range(400):
        ag.act(None)
    s = ag.state()
    print(f"{ag.name}: after 400 steps at ({s[0]:.0f}, {s[1]:.0f}), "
          f"speed {s[3]:.1f} m/s")


if __name__ == "__main__":
    _main()
