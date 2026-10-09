"""PPO driver for the quality function's simulation stage.

Wraps a Stable-Baselines3 policy trained by model_training/code/train.py in
the interface RacingProblem drives (act / reset / curve_points / current_idx /
_find_projection / _signed_lateral_offset), so the simulation stage can score a
track with the learned driver instead of the analytical SteeringAgent.

The observation MUST be what the policy saw in training, or the policy is fed
noise and reports a driveability that means nothing.  Two things keep it that
way:

  1. Observation and geometry are imported from the training code rather than
     reimplemented: TrackGeometry does the arc-length resampling, projection
     and curvature-ahead sampling, and RacingEnv.build_observation builds the
     vector for training and for this driver alike.  A reimplementation would
     be a second definition of the contract, free to drift from the one that
     trained.
  2. The two scaling constants that are NOT read from the track come from the
     checkpoint rather than from the car (see OBS_MAX_SPEED below).

A checkpoint trained on a different car still transfers: run10, trained on
the legacy engine, laps Monza, Silverstone and Catalunya on the benchmark
engine at off-road 0.006 / 0.000 / 0.000 once given its own scaling
constants.

Why not simpler: reading the policy's action from the car state alone, with no
centerline projection, drops obs[4] (lateral offset), obs[5] (heading error)
and obs[8:] (curvature ahead), which is 15 of the 21 inputs.  The policy is a
path follower and cannot drive without them.

Why not more complex: no frame stacking, no VecNormalize.  train.py trains on
raw observations with a plain MlpPolicy, so anything added here would be a
transform the policy never saw.
"""

from __future__ import annotations

import os
import sys

import numpy as np

# The training package is a sibling of the benchmark package.  Its modules
# import each other by bare name (import track_geometry), so its own directory
# has to be importable, not just its parent.
_TRAINING_CODE = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "model_training", "code"))


def _import_training_module(name):
    if _TRAINING_CODE not in sys.path:
        sys.path.append(_TRAINING_CODE)
    return __import__(name)


# Observation scaling the default checkpoint was trained with: the benchmark
# engine's 85.5 m/s top speed and 30 degree steering lock.  A checkpoint
# trained on the legacy engine needs 83.0 and 33 degrees instead, passed
# RacingProblem's rl_obs_max_speed / rl_obs_max_steering.  These belong to the
# checkpoint and not to the car: driving the benchmark engine with run10
# (a legacy-engine policy) normalised by 85.5 and 30 fails Silverstone at 0.05
# progress, where its own 83.0 and 33 lap it at 0.000 off-road.
OBS_MAX_SPEED = 85.5
OBS_MAX_STEERING = np.deg2rad(30.0)

# Centerline sampling interval, metres.  TrackGeometry's own default, and the
# value every run in model_training/runs was trained with, so the curvature
# observation is sampled at the spacing the policy learned to read.
_GEOM_SPACING = 3.0


def _numpy_actor(model):
    """The policy's deterministic action as a plain numpy forward pass.

    Returns None if the checkpoint is not the architecture train.py produces,
    in which case the caller keeps using SB3's predict().

    Why do this at all: for a continuous action space SB3's predict() with
    deterministic=True returns the Gaussian's mean, which is exactly
    action_net(policy_net(obs)).  Everything else predict() does is framework
    overhead.  Measured on run12_stuckfix, 3000 calls each: predict() 87.4 us,
    calling policy(obs) directly 101.2 us, the actor MLP and head alone
    18.9 us, this numpy path 6.6 us, agreeing with predict() to 4.9e-08.
    torch.set_num_threads(1) only moves predict() to 79.5 us, so the cost is
    tensor and distribution construction, not arithmetic or threading.

    Why not simpler: reusing model.policy directly still pays for tensor
    construction, which the 101.2 us above shows is the bulk of it.  Why not
    more complex: ONNX or torch.compile would add a build step and a
    dependency to replace two matmuls that already cost 6.6 us.
    """
    from torch import nn
    policy = model.policy
    # A squashed or non-flatten policy is a different function; so is any
    # observation preprocessing beyond a 1-D Box cast.
    if getattr(policy, "squash_output", False):
        return None
    if type(policy.features_extractor).__name__ != "FlattenExtractor":
        return None
    if len(getattr(model.observation_space, "shape", ())) != 1:
        return None

    acts = {nn.Tanh: np.tanh, nn.ReLU: lambda x: np.maximum(x, 0.0)}
    layers = []
    for m in list(policy.mlp_extractor.policy_net) + [policy.action_net]:
        if isinstance(m, nn.Linear):
            layers.append([m.weight.detach().numpy().astype(np.float64).copy(),
                           m.bias.detach().numpy().astype(np.float64).copy(),
                           None])
        elif type(m) in acts and layers:
            layers[-1][2] = acts[type(m)]
        else:
            return None
    if not layers:
        return None

    def forward(obs):
        h = obs
        for weight, bias, activation in layers:
            h = weight @ h + bias
            if activation is not None:
                h = activation(h)
        return h
    return forward


class RLAgent:
    """Path-following agent driven by a trained PPO policy.

    Exposes the members RacingProblem reads off its agent, so it is a drop-in
    for SteeringAgent in the simulation stage.
    """

    def __init__(self, curve_points, track_width=12.0, model=None,
                 obs_max_speed=None, obs_max_steering=None):
        """`model` is a loaded SB3 policy (see load_policy).

        obs_max_speed and obs_max_steering are the divisors for obs[0] and
        obs[3].  They belong to the CHECKPOINT, not to the car being driven:
        the policy learned one numeric scaling and reads any other as a
        different speed or steering angle.  Measured on run10 (trained on
        the legacy engine, 83.0 m/s and 33 deg) driving the benchmark engine:
        with its own constants it laps Monza, Silverstone and Catalunya
        (progress 1.00 / 1.00 / 1.00, off-road 0.006 / 0.000 / 0.000);
        normalised by the benchmark engine's 85.5 and 30 deg instead,
        Silverstone fails at 0.05 progress with 0.050 off-road.  Defaults
        match the default checkpoint.
        """
        if model is None:
            raise ValueError("RLAgent needs a loaded policy; see load_policy()")
        self._model = model
        self._actor = _numpy_actor(model)
        self.track_width = float(track_width)

        racing_env = _import_training_module("racing_env")
        self._env_cls = racing_env.RacingEnv
        self._track_geometry_cls = _import_training_module(
            "track_geometry").TrackGeometry

        self._max_speed_norm = float(
            OBS_MAX_SPEED if obs_max_speed is None else obs_max_speed)
        self._max_steering_norm = float(
            OBS_MAX_STEERING if obs_max_steering is None else obs_max_steering)

        self._geom = None
        self._seg_hint = None
        self._s = 0.0
        self._last_action = np.zeros(2, dtype=np.float32)
        self.current_idx = 0

        self.curve_points = curve_points

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    @property
    def curve_points(self):
        return self._curve_points

    @curve_points.setter
    def curve_points(self, points):
        """Rebuild the arc-length geometry the observation is read from.

        map_size is passed as 0.0: TrackGeometry stores it for the training
        env's out-of-map check and never uses it in projection, curvature or
        resampling, which is all this agent asks of it.  The benchmark does its
        own bounds check in _get_cached_simulation_summary.
        """
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        self._curve_points = pts
        # TrackGeometry needs a closed ring of at least a few points to
        # resample; below that the caller is not describing a lap.
        if len(pts) < 4:
            self._geom = None
        else:
            self._geom = self._track_geometry_cls(
                pts, self.track_width, 0.0, spacing=_GEOM_SPACING)
        self.reset()

    @property
    def progress_fraction(self):
        """Fraction of the lap reached, in [0, 1].

        Read off the arc length TrackGeometry reports, so it does not depend on
        that geometry's 3 m resampling.  SteeringAgent exposes the same
        quantity from the benchmark's own curve.
        """
        if self._geom is None or self._geom.length <= 0.0:
            return 0.0
        return float(min(max(self._s, 0.0), self._geom.length)) / float(self._geom.length)

    def reset(self):
        self._seg_hint = None
        self._s = 0.0
        self._last_action = np.zeros(2, dtype=np.float32)
        self.current_idx = 0

    # ------------------------------------------------------------------
    # Projection helpers that RacingProblem calls directly
    # ------------------------------------------------------------------

    def _find_projection(self, point, start_idx):
        """(segment index, projection parameter) for a position.

        The benchmark uses the returned index as the progress marker and feeds
        it back as the next start_idx, matching SteeringAgent's contract.
        """
        if self._geom is None:
            return 0, 0.0
        seg_idx, s, _lateral, _heading_ref = self._geom.project(
            point, start_idx if start_idx else None)
        return int(seg_idx), float(s)

    def _signed_lateral_offset(self, point, seg_idx):
        """Signed distance from the centerline; positive is left of travel."""
        if self._geom is None:
            return 0.0
        _seg, _s, lateral, _heading_ref = self._geom.project(point, seg_idx)
        return float(lateral)

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------

    def _build_observation(self, car_state, lateral, heading_ref):
        """RacingEnv.build_observation on the benchmark car's state, with the
        checkpoint's scaling constants (OBS_MAX_SPEED, OBS_MAX_STEERING)."""
        _x, _y, heading, v_forward, steering = car_state
        return self._env_cls.build_observation(
            self._geom, self._s, lateral, heading_ref, heading, v_forward,
            self._v_lateral, self._yaw_rate, steering, self._last_action,
            self._max_speed_norm, self._max_steering_norm)

    def act(self, car_state):
        """Return the engine action dict for a benchmark car state.

        `car_state` is the engine's state vector
        [x, y, heading, speed, steering, v_lateral, yaw_rate]; the engine's
        get_state appends the last two, so the observation is built without
        reaching into the engine from here.
        """
        state = np.asarray(car_state, dtype=float).ravel()
        pos = state[:2]
        self._v_lateral = float(state[5]) if state.size > 5 else 0.0
        self._yaw_rate = float(state[6]) if state.size > 6 else 0.0

        if self._geom is None:
            return {"steering": 0.0, "throttle": 0.0}

        seg_idx, s, lateral, heading_ref = self._geom.project(
            pos, self._seg_hint)
        self._seg_hint = seg_idx
        self._s = s
        self.current_idx = int(seg_idx)

        obs = self._build_observation(
            (state[0], state[1], state[2], state[3],
             state[4] if state.size > 4 else 0.0),
            lateral, heading_ref)
        if self._actor is not None:
            action = self._actor(obs)
        else:
            action, _ = self._model.predict(obs, deterministic=True)
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self._last_action = action
        return {"steering": float(action[0]), "throttle": float(action[1])}


def load_policy(path, device="cpu"):
    """Load an SB3 PPO checkpoint.

    device is cpu because the simulation stage runs one car at a time: a
    21-input MLP forward pass per step is dominated by the per-call overhead,
    which a GPU adds to rather than removes.
    """
    from stable_baselines3 import PPO
    return PPO.load(str(path), device=device)
