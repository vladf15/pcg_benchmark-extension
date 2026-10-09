"""Gymnasium environment for training a driver on the real circuits (1:1, metres).

The physics sits behind a small adapter (EngineBackedPhysics), so the engine
can be swapped without touching observation or reward.  Centreline and
constant track width are the benchmark's own.

Observation (21 floats, all roughly in [-4, 4]):
    0  forward speed          / 85.5 (engine top speed)
    1  lateral speed          / 10
    2  yaw rate               / 2 rad/s
    3  steering angle         / 30 deg
    4  lateral offset         / half track width (signed, + = left of center)
    5  heading error          / pi (car heading vs centerline tangent)
    6  previous steer action
    7  previous throttle action
    8-20  worst signed curvature in 13 bands ahead, times 10 (+ = left; a
          10 m hairpin reads 1.0, a 100 m sweeper 0.1).  Bands, not samples,
          so short corners cannot hide.  Edges are fixed TIMES ahead (0.2 to
          8.25 s) scaled by speed: at 85.5 m/s the last band is ~705 m out
          (braking from top speed takes ~330 m), at low speed the bands wrap
          the car for apex placement; frozen below LOOKAHEAD_MIN_SPEED.

Action (2 floats in [-1, 1]): steering, throttle/brake.

Reward per step:
    + 0.1  * forward progress along the centerline (metres, only while
             within the kerb limit)
    - 0.05 per step (time cost: the reason to go fast rather than far)
    - 0.05 * (change in steer action)^2  (suppresses oscillation)
    off track (past the kerb, within 2 half-widths): - 1.0 - overshoot *
           (1.5 + 4 v/vmax) per step, overshoot past the outside wheels
           (CAR_HALF_WIDTH); a hard cliff gave no gradient on entry speed
    lap complete: +20, plus up to +200 for steps to spare
    departed (beyond 2 half-widths) or off the map: -(20 + 30 v/vmax), end
    stuck (no progress for stuck_patience steps): -40, end (more than a
           crash, or standing still is the cheapest ending)

The racing RL literature's shape: dense progress plus penalties for leaving
the track (GT Sophy, Wurman et al. 2022: progress masked off course, an
off-course penalty proportional to squared speed), and DDPG's TORCS reward
(Lillicrap et al. 2016, Sec. 9.1: "the velocity of the car projected along
the track direction", episodes ended after 500 frames without progress),
which ds per step already is.  ABSENT on purpose: a flat alignment bonus
(a survival stipend that recreates slow creeping) and a centreline-distance
penalty (it makes lane keepers hug the middle).  The reward never says
WHERE to drive: progress is along the centreline whatever the line (which
blocks infield shortcuts), so a racing line must emerge.

Episodes end on lap complete, on failure (departed, off the map, or stuck),
or when the per-track step budget runs out (truncated).
"""

import os
import sys
from collections import namedtuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import track_loader
from track_geometry import TrackGeometry

# Make the editable pcg_benchmark import work even outside the venv.
# model_training lives inside pcg_benchmark-extension, so the benchmark root
# is two levels up from code/.
_BENCH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "..", ".."))
if _BENCH not in sys.path:
    sys.path.append(_BENCH)


CarState = namedtuple(
    "CarState", ["x", "y", "heading", "v_forward", "v_lateral", "yaw_rate", "steering"])


class EngineBackedPhysics:
    """Adapter around the benchmark's CarPhysicsEngine; a replacement needs
    `dt`, `reset(position, heading) -> CarState` and `step(steer, throttle)
    -> CarState` (inputs in [-1, 1])."""

    def __init__(self):
        from pcg_benchmark.probs.racing.engine import CarPhysicsEngine
        self._engine = CarPhysicsEngine(start_position=(0.0, 0.0))
        self.dt = float(self._engine.time_step)
        self.max_speed = float(self._engine.max_speed)
        self.max_steering = float(self._engine.max_steering)

    def _state(self):
        e = self._engine
        cos_a, sin_a = np.cos(e.angle), np.sin(e.angle)
        v_fwd = cos_a * e.velocity[0] + sin_a * e.velocity[1]
        v_lat = -sin_a * e.velocity[0] + cos_a * e.velocity[1]
        return CarState(float(e.position[0]), float(e.position[1]),
                        float(e.angle), float(v_fwd), float(v_lat),
                        float(e.yaw_rate), float(e.steering_angle))

    def reset(self, position, heading, speed=0.0):
        self._engine.start_position = np.asarray(position, dtype=float)
        self._engine.start_angle = float(heading)
        self._engine.reset()
        if speed:
            # Rolling start: world-frame velocity alone is a valid state (the
            # gearbox picks its gear on the first step).
            self._engine.velocity = speed * np.array(
                [np.cos(heading), np.sin(heading)], dtype=float)
        return self._state()

    def step(self, steer, throttle):
        self._engine.step({"steering": float(steer), "throttle": float(throttle)})
        return self._state()


class RacingEnv(gym.Env):
    """Drive one lap on a real circuit."""

    metadata = {"render_modes": []}

    # Curvature band edges in SECONDS ahead, scaled by speed (lookahead_edges):
    # fixed metres serve both ends badly (on the legacy engine at 83 m/s a
    # 325 m horizon left 0.4 s beyond the braking distance; at 8 m/s it
    # described track 40 s away).  Near edges ~1.2x apart for apexes, far
    # ones stretched for braking points.  A MODELLING CHOICE, not swept.
    LOOKAHEAD_TIMES = (0.0, 0.20, 0.40, 0.65, 0.95, 1.30, 1.70, 2.20,
                       2.80, 3.55, 4.45, 5.50, 6.75, 8.25)
    # Below it the scale freezes: 0.2 s x 15 m/s = 3 m, one geometry sample.
    LOOKAHEAD_MIN_SPEED = 15.0
    _LOOKAHEAD_T = np.array(LOOKAHEAD_TIMES)

    @classmethod
    def lookahead_edges(cls, v_forward):
        """Band edges in metres for the given forward speed (m/s)."""
        return max(float(v_forward), cls.LOOKAHEAD_MIN_SPEED) * cls._LOOKAHEAD_T

    # Reward constants.  PROGRESS_SCALE pays per METRE, so a lap pays the same
    # however fast: on Spa driving 2.3x faster raised the return only from
    # 608 to 620, and the policy crept.  TIME_COST makes it a lap-time
    # objective: the only way to keep more progress reward is fewer steps.
    PROGRESS_SCALE = 0.1
    TIME_COST = 0.05          # per step; ~0.5 per second of lap time.
                              # At 0.03 run5 drove clean but at 16-17 m/s
                              # mean; 0.05 prices time higher to push lap
                              # time down
    STEER_RATE_SCALE = 0.05   # per (steer action change)^2: cheap for small
                              # corrections, expensive for full-lock sawing;
                              # a modelling choice, not swept
    LAP_BONUS = 20.0          # a modelling choice, not swept
    LAP_TIME_BONUS = 200.0    # ~a third of a lap's reward (30 was ~5%)
    # Off-track penalties scale with how far off AND how fast (as GT Sophy's
    # scale with squared speed): a hard cliff (-10 and done) gave no gradient,
    # and run4 plateaued 2.5M steps carrying 21-32 m/s into corners
    # survivable at 10-17.  Per step: overshoot * (OFFROAD_SCALE +
    # OFFROAD_SPEED v/vmax); on termination FAIL_PENALTY + FAIL_SPEED_PENALTY
    # v/vmax.
    OFFROAD_SCALE = 1.5       # set with TIME_COST (1.0 with 0.03) so the
                              # extra pace pressure cannot be paid for by
                              # running wide more often
    OFFROAD_SPEED = 4.0       # a modelling choice, not swept
    # Flat cost per step past the edge: above the most one step can earn
    # (0.1 x 85.5 m/s x 0.1 s = 0.855), so no step off pays for itself.
    # Without it (run14) the circuits went 33.4 m/s against run13's 20.4 but
    # off the road on 22 against 11 (share 0.0128 against 0.0048).  Not a
    # termination: that is run4's cliff again.
    OFFROAD_STEP = 1.0
    # Both modelling choices, not swept; STUCK_PENALTY below is sized
    # against FAIL_PENALTY.
    FAIL_PENALTY = 20.0       # also the off-map penalty (at v ~ 0)
    FAIL_SPEED_PENALTY = 30.0
    # Stopping must cost more than driving.  With it equal to FAIL_PENALTY
    # the returns were flat (standing -25.1, driving at 10 m/s -21.1,
    # crashing at 30 m/s -25.0) and run11 sat at -25.1 for all 5M steps.  40
    # restores run10's slope (-45.1 standing against -21.1 driving, a 24-point
    # gap against its 23), sized to that measurement; far above the worst
    # crash would teach driving off deliberately.  Validated (run12_stuckfix,
    # 5M steps): return -45.1 -> 482.6, 4 of 5 held-out circuits lapped
    # against 0; the escape comes at ~3.7M steps, so shorter runs look failed.
    STUCK_PENALTY = 40.0
    # Progress stops at 1.0 half-widths, the benchmark's off-road line, so
    # the policy trains on what it is scored by; cutting the infield earns
    # nothing.  Why not 1.1: a slightly wide line was net positive (+0.25
    # progress against 0.13 penalty per step), and run13 spent up to 3.8% of
    # a lap past the edge.
    KERB_LIMIT = 1.0
    # Edge terms act at the outside wheels, half the 992's published 1852 mm
    # width inside the edge, so the centre stays 0.93 m inside.  At the centre
    # (run15) the policy ran apexes on the edge, off the road on 9 of 24.
    CAR_HALF_WIDTH = 0.926
    # Termination at two half-widths from the centreline: a modelling choice.
    OFF_TRACK_LIMIT = 2.0

    def __init__(self, track_names=None, physics=None, randomize=True,
                 stuck_patience=100, seed=None):
        """
        track_names: circuits to use (default: all in the database).
        physics:     object with the EngineBackedPhysics interface.
        randomize:   pick random track / start / direction / mirror each
                     reset. Turn off for evaluation: resets then start at
                     the beginning of the centerline, forward, unmirrored,
                     and walk through `track_names` one circuit per reset.
                     reset(options=...) overrides any of those.
        """
        super().__init__()
        self._physics = physics if physics is not None else EngineBackedPhysics()
        self._randomize = randomize
        self._stuck_patience = int(stuck_patience)
        self._rng = np.random.default_rng(seed)

        names = track_names if track_names is not None else track_loader.list_tracks()
        self._geoms = {}
        for name in names:
            t = track_loader.load_track(name)
            # 1:1 scale (world units = metres); each circuit carries its own
            # bounding square, since real circuits span 813-2171 m and the
            # benchmark map is one fixed square for every track.
            base = TrackGeometry(t["points"], t["track_width"], t["map_size"])
            # Precompute the four augmentation variants once.
            self._geoms[name] = {
                (False, False): base,
                (True, False): base.reversed(),
                (False, True): base.mirrored(),
                (True, True): base.mirrored().reversed(),
            }
        self._names = list(self._geoms.keys())

        self._margin = track_loader.MARGIN  # benchmark edge margin (8 m: half the width + 2)

        n_obs = 8 + len(self.LOOKAHEAD_TIMES) - 1
        self.observation_space = spaces.Box(low=-4.0, high=4.0,
                                            shape=(n_obs,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(2,), dtype=np.float32)

        self._geom = None
        self._track_name = None
        self._car = None
        # Non-randomised envs walk the track list, one circuit per reset, so an
        # evaluation covers the whole held-out set.
        self._cycle_idx = 0

    # ── read-only accessors (for visualization and analysis) ──────────────

    @property
    def geometry(self):
        """Geometry of the circuit currently loaded (None before reset)."""
        return self._geom

    @property
    def car_state(self):
        """Current CarState, so a caller can record the driven path."""
        return self._car

    @property
    def physics_dt(self):
        """Simulation timestep, for converting step counts to seconds."""
        return self._physics.dt

    # ── episode setup ─────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        options = options or {}

        name = options.get("track")
        if name is None:
            if self._randomize:
                name = self._names[self._rng.integers(len(self._names))]
            else:
                # Round-robin over the circuits.
                name = self._names[self._cycle_idx % len(self._names)]
                self._cycle_idx += 1
        if self._randomize:
            reverse = bool(options.get("reverse", self._rng.integers(2)))
            mirror = bool(options.get("mirror", self._rng.integers(2)))
            start_frac = float(options.get("start_frac", self._rng.random()))
            # Rolling start: from rest, ~100 near-random steps cover under a
            # metre and PPO collapses to "never move" (a pilot sat at -13.2
            # for 1.5M steps).  Low speeds stay in the range so launching is
            # still learned; evaluation keeps standing starts.
            start_speed = float(options.get("start_speed",
                                            self._rng.uniform(0.0, 30.0)))
        else:
            reverse = bool(options.get("reverse", False))
            mirror = bool(options.get("mirror", False))
            start_frac = float(options.get("start_frac", 0.0))
            start_speed = float(options.get("start_speed", 0.0))

        self._track_name = name
        self._geom = self._geoms[name][(reverse, mirror)]
        self._map_size = self._geom.map_size
        # Step budget for a slow lap (8 m/s), capped at 10000 (Spa needs ~8750).
        self.max_episode_steps = int(np.clip(
            self._geom.length / (self._physics.dt * 8.0), 1000, 10000))

        pos, heading = self._geom.pose_at_s(start_frac * self._geom.length)
        car = self._car = self._physics.reset(pos, heading, start_speed)

        self._seg_hint, s0, lateral0, heading_ref0 = self._geom.project((car.x, car.y))
        self._s = s0
        self._progress = 0.0
        self._steps = 0
        self._offroad_steps = 0
        self._last_action = np.zeros(2, dtype=np.float32)
        self._progress_marker = (0, 0.0)  # (step, progress) for stuck detection

        return (self._observe(car, lateral0, heading_ref0),
                {"track": name, "reverse": reverse,
                 "mirror": mirror, "start_frac": start_frac})

    # ── stepping ──────────────────────────────────────────────────────────

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        car = self._car = self._physics.step(action[0], action[1])
        self._steps += 1

        self._seg_hint, s, lateral, heading_ref = self._geom.project(
            (car.x, car.y), self._seg_hint)
        ds = self._geom.delta_s(s, self._s)
        self._s = s
        self._progress += ds

        half_width = self._geom.half_width
        offset_hw = abs(lateral) / half_width          # offset in half-widths
        # Past the wheel line (see CAR_HALF_WIDTH).  offset_hw, the car's
        # centre, still decides the off-track termination.
        wheel_line = half_width - self.CAR_HALF_WIDTH
        overshoot = max(0.0, abs(lateral) - wheel_line)
        if offset_hw > 1.0:
            self._offroad_steps += 1        # the benchmark's off-road: centre past the edge

        # Progress pays only up to the kerb: arc swept while off the track
        # is arc the car cut, not drove.
        progress_reward = self.PROGRESS_SCALE * ds if offset_hw <= self.KERB_LIMIT else 0.0
        steer_delta = float(action[0]) - float(self._last_action[0])
        speed_frac = max(car.v_forward, 0.0) / self._physics.max_speed
        reward = (progress_reward
                  - self.TIME_COST
                  - self.STEER_RATE_SCALE * steer_delta * steer_delta)
        if overshoot > 0.0:
            # Graded off-track band: the further off and the faster, the
            # worse. This is the gradient that teaches corner-entry speed.
            reward -= self.OFFROAD_STEP + (overshoot / half_width) * (
                self.OFFROAD_SCALE + self.OFFROAD_SPEED * speed_frac)
        self._last_action = action

        terminated = False
        truncated = False
        info = {}

        # Failures first, exclusive with a finished lap (otherwise one step
        # could award the lap bonus and the failure penalty at once).
        m = self._margin
        if not (m <= car.x <= self._map_size - m and m <= car.y <= self._map_size - m):
            # Failure: left the map entirely.
            terminated = True
            reward -= self.FAIL_PENALTY + self.FAIL_SPEED_PENALTY * speed_frac
            info["off_map"] = True
        elif offset_hw > self.OFF_TRACK_LIMIT:
            # Failure: departed the track, costing more the faster (~2.5x).
            terminated = True
            reward -= self.FAIL_PENALTY + self.FAIL_SPEED_PENALTY * speed_frac
            info["off_track"] = True
        elif self._progress >= self._geom.length:
            # Lap complete: accumulated forward progress covers the whole loop.
            terminated = True
            reward += self.LAP_BONUS + self.LAP_TIME_BONUS * (
                1.0 - self._steps / self.max_episode_steps)
            info["lap_complete"] = True
            info["laptime_steps"] = self._steps

        # Failure: stuck (less than 1 map unit of progress since the marker).
        mark_step, mark_progress = self._progress_marker
        if self._progress > mark_progress + 1.0:
            self._progress_marker = (self._steps, self._progress)
        elif not terminated and self._steps - mark_step > self._stuck_patience:
            terminated = True
            reward -= self.STUCK_PENALTY
            info["stuck"] = True

        if not terminated and self._steps >= self.max_episode_steps:
            truncated = True

        if terminated or truncated:
            info["track"] = self._track_name
            info["progress_frac"] = self._progress / self._geom.length
            info["offroad_frac"] = self._offroad_steps / max(self._steps, 1)

        obs = self._observe(car, lateral, heading_ref)
        return obs, float(reward), terminated, truncated, info

    # ── observation ───────────────────────────────────────────────────────

    def _observe(self, car, lateral, heading_ref):
        """Build the observation. The caller passes the projection results it
        already computed; projecting is the most expensive thing done per
        step, so it is done once and shared."""
        return self.build_observation(
            self._geom, self._s, lateral, heading_ref, car.heading, car.v_forward,
            car.v_lateral, car.yaw_rate, car.steering, self._last_action,
            self._physics.max_speed, self._physics.max_steering)

    @classmethod
    def build_observation(cls, geom, s, lateral, heading_ref, heading, v_forward,
                          v_lateral, yaw_rate, steering, last_action,
                          max_speed, max_steering):
        """The policy's observation, from the car's state and its projection
        onto the centreline (arc position s, signed lateral offset, heading
        of the centreline there).  The benchmark's driver
        (pcg_benchmark/probs/racing/rl_agent.py) calls this too, so the
        vector the policy is scored with is the one it was trained on: the
        input layer is positional, and swapping two entries gives a driver
        that still runs and still returns actions, just bad ones."""
        heading_err = (heading - heading_ref + np.pi) % (2.0 * np.pi) - np.pi
        curv = geom.curvature_ahead(s, cls.lookahead_edges(v_forward))

        obs = np.empty(8 + len(cls.LOOKAHEAD_TIMES) - 1, dtype=np.float32)
        obs[0] = v_forward / max_speed
        obs[1] = v_lateral / 10.0
        obs[2] = yaw_rate / 2.0
        obs[3] = steering / max_steering
        obs[4] = lateral / geom.half_width
        obs[5] = heading_err / np.pi
        obs[6] = last_action[0]
        obs[7] = last_action[1]
        obs[8:] = curv * 10.0
        return np.clip(obs, -4.0, 4.0)
