"""Gymnasium environment for training a driver on the rescaled real circuits.

The physics is deliberately hidden behind a small adapter (`EngineBackedPhysics`)
so the engine can be swapped (for example for a TORCS-style model) without
touching the environment, observation, or reward. Everything else is fixed:
centerline + constant track width, the same dimensions the benchmark uses.

Observation (21 floats, all roughly in [-4, 4]):
    0  forward speed          / 83 (engine top speed)
    1  lateral speed          / 10
    2  yaw rate               / 2 rad/s
    3  steering angle         / 30 deg
    4  lateral offset         / half track width (signed, + = left of center)
    5  heading error          / pi (car heading vs centerline tangent)
    6  previous steer action
    7  previous throttle action
    8-20  worst signed curvature in 13 SPEED-SCALED bands ahead, times 10
          (positive = left turn; 10 x curvature = 10/radius in metres, so a
          10 m hairpin reads 1.0 and a 100 m sweeper 0.1). Bands, not point
          samples, so short corners cannot hide between samples. Band edges
          are fixed TIMES ahead (0.2 s to 8.25 s), converted to metres with
          the current speed, so the horizon breathes with pace like a real
          driver's: at 83 m/s the last band reaches ~685 m (braking from
          top speed takes ~293 m, so the braking point is visible with
          seconds to spare), while at low speed the same bands wrap tightly
          around the car for apex placement instead of describing track
          half a minute away. Below LOOKAHEAD_MIN_SPEED the scaling is
          frozen so the car is never blind at a standstill.

Action (2 floats in [-1, 1]): steering, throttle/brake.

Reward per step:
    + 0.1  * forward progress along the centerline (metres, only while
             within the kerb limit)
    - 0.05 per step (time cost: the reason to go fast rather than far)
    - 0.05 * (change in steer action)^2  (suppresses oscillation)
    off track (past the kerb, within 2 half-widths):
           - overshoot * (1.5 + 4 * v/vmax) per step: the further off and
             the faster, the worse. A hard cliff at the edge gave no
             gradient about corner-entry speed and produced a constant-
             pace policy that never braked for tight corners.
    lap complete:  +20, plus up to +200 for finishing with steps to spare
    departed the track (beyond 2 half-widths), off the map, or stuck:
           -(20 + 30 * v/vmax) and the episode ends: failing fast is
           much worse than failing slow, which is the point.

This is the shape used by the racing RL literature: dense progress along
the track plus penalties for leaving it (GT Sophy, Wurman et al. 2022) and
the classic TORCS formulation v*cos(heading_error) (Lau 2016), which is
what ds per step already equals. Two terms that appear in lane-keeping
work are deliberately ABSENT here:

  - no flat alignment bonus (e.g. +c * cos(heading_error)): a constant
    per-step bonus is a survival stipend that pays the agent for merely
    existing on track, which re-creates the slow-creeping optimum. The
    literature's version is speed-multiplied, and ds already is that.
  - no centerline-distance penalty (e.g. -c * (lateral/half_width)^2):
    that term is what makes lane keepers hug the centerline. A racing
    agent must be free to use the full width; staying on track is handled
    by termination, not by attraction to the middle.

Note the reward never says WHERE on the road to drive. Progress is
measured along the centerline whatever line the car takes (which is what
blocks infield shortcuts), so a racing line has to emerge because it is
the quickest way round, not because it was rewarded directly.

Episodes end on: lap complete (terminated); failure (terminated) when the
car departs the track (more than OFF_TRACK_LIMIT half-widths from the
centerline), leaves the map, or makes no progress for `stuck_patience`
steps; or the per-track step budget running out (truncated).
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
    """Adapter around the benchmark's CarPhysicsEngine.

    A replacement physics model (e.g. TORCS-style) only has to provide the
    same three members: `dt`, `reset(position, heading) -> CarState`, and
    `step(steer, throttle) -> CarState`, with steer/throttle in [-1, 1].
    """

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
            # Rolling start: both engines store velocity in the world frame,
            # and the auto gearbox picks the right gear from wheel speed on
            # the first step, so setting velocity alone is a valid state.
            self._engine.velocity = speed * np.array(
                [np.cos(heading), np.sin(heading)], dtype=float)
        return self._state()

    def step(self, steer, throttle):
        self._engine.step({"steering": float(steer), "throttle": float(throttle)})
        return self._state()


class V2Physics(EngineBackedPhysics):
    """Simcade physics v2 (engine_v2.py): substepped, load transfer, real
    grip limits, human-limited inputs. Default for the driving model; the
    benchmark's own engine stays frozen and available as EngineBackedPhysics.
    """

    def __init__(self):
        from engine_v2 import CarPhysicsEngineV2
        self._engine = CarPhysicsEngineV2(start_position=(0.0, 0.0))
        self.dt = float(self._engine.time_step)
        self.max_speed = float(self._engine.max_speed)
        self.max_steering = float(self._engine.max_steering)


def make_physics(name="v2"):
    """Physics by name: 'v2' (simcade, default) or 'v1' (benchmark engine)."""
    if name == "v2":
        return V2Physics()
    if name == "v1":
        return EngineBackedPhysics()
    raise ValueError(f"unknown physics '{name}' (use 'v1' or 'v2')")


class RacingEnv(gym.Env):
    """Drive one lap on a (real, rescaled) circuit."""

    metadata = {"render_modes": []}

    # Band edges for the curvature observation, in SECONDS ahead: slot i
    # reports the worst curvature between edge i and i+1, so a short corner
    # can never hide between two sample points. Edges are converted to
    # metres with the current speed (see lookahead_edges), because fixed
    # distances serve both ends badly: at 83 m/s a 325 m horizon left 0.4 s
    # of look beyond the braking distance (blind at speed), while at 8 m/s
    # the same horizon described track 40 s away (noise when slow). Near
    # edges grow ~1.2x apart for apex resolution, far ones stretch for
    # braking-point planning. 14 edges = 13 slots.
    LOOKAHEAD_TIMES = (0.0, 0.20, 0.40, 0.65, 0.95, 1.30, 1.70, 2.20,
                       2.80, 3.55, 4.45, 5.50, 6.75, 8.25)
    # Below this speed the band scale freezes (0.2 s * 15 m/s = 3 m, the
    # centerline sampling interval, so the nearest band never collapses
    # below one geometry sample and the car is never blind at a standstill).
    LOOKAHEAD_MIN_SPEED = 15.0
    _LOOKAHEAD_T = np.array(LOOKAHEAD_TIMES)

    @classmethod
    def lookahead_edges(cls, v_forward):
        """Band edges in metres for the given forward speed (m/s)."""
        return max(float(v_forward), cls.LOOKAHEAD_MIN_SPEED) * cls._LOOKAHEAD_T

    # Reward constants (see module docstring).
    #
    # PROGRESS_SCALE is per METRE of centerline arc covered, so a lap pays
    # the same total however fast it is driven. That, plus a lap bonus worth
    # only ~5% of a lap's reward, made speed almost worthless: measured on
    # Spa, driving 2.3x faster raised the episode return from 608 to 620,
    # about 2%. The agent correctly concluded that creeping safely beats
    # driving quickly, which is exactly the "sticks to the centerline and
    # wobbles slowly" behaviour that resulted.
    #
    # TIME_COST fixes it: every step costs, so the ONLY way to keep more of
    # the progress reward is to cover the lap in fewer steps. This makes the
    # reward a lap-time objective rather than a distance objective.
    PROGRESS_SCALE = 0.1
    TIME_COST = 0.05          # per step; ~0.5 per second of lap time
                              # (raised from 0.03 after run5: clean but
                              # conservative at 16-17 m/s mean, so price
                              # time higher to push lap time down)
    STEER_RATE_SCALE = 0.05   # per (steer action change)^2: cheap for small
                              # corrections, expensive for full-lock sawing
    LAP_BONUS = 20.0
    LAP_TIME_BONUS = 200.0    # was 30: now worth ~a third of a lap's reward
    # Off-track penalties. A hard cliff at the edge (on-track = no signal,
    # one step over = flat -10 and done) gives the policy NO gradient about
    # how badly it missed: run4 plateaued for 2.5M steps at a constant-pace
    # policy that carried 21-32 m/s into corners survivable at 10-17,
    # because nothing in the reward distinguished "clipped the kerb" from
    # "flew off at speed". So the penalty scales with BOTH how far off the
    # car is and how fast it is going (GT Sophy scales its off-course and
    # wall penalties with kinetic energy for the same reason): per step in
    # the off-track band, overshoot * (OFFROAD_SCALE + OFFROAD_SPEED * v/vmax),
    # and on termination FAIL_PENALTY + FAIL_SPEED_PENALTY * v/vmax.
    OFFROAD_SCALE = 1.5       # raised from 1.0 alongside TIME_COST so the
                              # extra pace pressure cannot be paid for by
                              # running wide more often
    OFFROAD_SPEED = 4.0
    FAIL_PENALTY = 20.0        # also the stuck/off-map penalty (at v ~ 0)
    FAIL_SPEED_PENALTY = 30.0
    # The edge is 1.0 half-widths (point car); KERB_LIMIT adds a tolerance
    # inside which progress still counts, so clipping an apex kerb is not
    # treated as leaving the track. Beyond OFF_TRACK_LIMIT the car has
    # genuinely departed: terminate. Progress reward stops at the kerb, so
    # cutting across the infield earns nothing (blocks the shortcut exploit).
    KERB_LIMIT = 1.1
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
        self._physics = physics if physics is not None else V2Physics()
        self._randomize = randomize
        self._stuck_patience = int(stuck_patience)
        self._rng = np.random.default_rng(seed)

        names = track_names if track_names is not None else track_loader.list_tracks()
        self._geoms = {}
        for name in names:
            t = track_loader.load_track(name)
            # 1:1 scale (world units = metres); each circuit carries its own
            # bounding square since real circuits exceed the 750 m benchmark map.
            base = TrackGeometry(t["points"], t["track_width"], t["map_size"])
            # Precompute the four augmentation variants once.
            self._geoms[name] = {
                (False, False): base,
                (True, False): base.reversed(),
                (False, True): base.mirrored(),
                (True, True): base.mirrored().reversed(),
            }
        self._names = list(self._geoms.keys())

        self._margin = track_loader.MARGIN  # benchmark edge margin (10 m)

        n_obs = 8 + len(self.LOOKAHEAD_TIMES) - 1
        self.observation_space = spaces.Box(low=-4.0, high=4.0,
                                            shape=(n_obs,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(2,), dtype=np.float32)

        self._geom = None
        self._track_name = None
        self._car = None
        # Non-randomized envs walk the track list in order, one circuit per
        # reset. Without this every evaluation episode would replay the first
        # circuit, so a held-out set of five would only ever test one of them.
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
                # Deterministic round-robin: consecutive resets visit every
                # circuit in turn, so an eval callback that just calls
                # reset() repeatedly covers the whole held-out set.
                name = self._names[self._cycle_idx % len(self._names)]
                self._cycle_idx += 1
        if self._randomize:
            reverse = bool(options.get("reverse", self._rng.integers(2)))
            mirror = bool(options.get("mirror", self._rng.integers(2)))
            start_frac = float(options.get("start_frac", self._rng.random()))
            # Rolling start. From a standstill, ~100 steps of near-random
            # throttle cover under a metre, so every early episode ends in
            # the same stuck-failure and PPO collapses to "never move" (the
            # reshaped pilot sat at reward -13.2 / 101 steps for all 1.5M
            # steps). Spawning already moving means progress reward and its
            # gradient are felt from step one; low speeds stay in the range
            # so launching from rest is still learned. Eval keeps standing
            # starts.
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
        # Step budget: enough for a slow-but-moving lap (8 m/s average),
        # bounded so no episode drags on forever. Real circuits are 2.3-7 km,
        # so the cap sits at 10000 steps (Spa at 8 m/s needs ~8750).
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
        overshoot = max(0.0, abs(lateral) - half_width)
        if overshoot > 0.0:
            self._offroad_steps += 1

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
            reward -= (overshoot / half_width) * (
                self.OFFROAD_SCALE + self.OFFROAD_SPEED * speed_frac)
        self._last_action = action

        terminated = False
        truncated = False
        info = {}

        # Failures are checked first and are exclusive with a finished lap:
        # a car that is off the map or off the track has not completed a
        # valid lap, whatever its progress counter says. Checking the lap
        # first would let a single step award the lap bonus and the failure
        # penalty at once, and report both flags in info.
        m = self._margin
        if not (m <= car.x <= self._map_size - m and m <= car.y <= self._map_size - m):
            # Failure: left the map entirely.
            terminated = True
            reward -= self.FAIL_PENALTY + self.FAIL_SPEED_PENALTY * speed_frac
            info["off_map"] = True
        elif offset_hw > self.OFF_TRACK_LIMIT:
            # Failure: genuinely departed the track. Penalty scales with the
            # speed it happened at: flying off costs ~2.5x dribbling off.
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
            reward -= self.FAIL_PENALTY
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
        g = self._geom
        heading_err = (car.heading - heading_ref + np.pi) % (2.0 * np.pi) - np.pi
        curv = g.curvature_ahead(self._s, self.lookahead_edges(car.v_forward))

        obs = np.empty(8 + len(self.LOOKAHEAD_TIMES) - 1, dtype=np.float32)
        obs[0] = car.v_forward / self._physics.max_speed
        obs[1] = car.v_lateral / 10.0
        obs[2] = car.yaw_rate / 2.0
        obs[3] = car.steering / self._physics.max_steering
        obs[4] = lateral / g.half_width
        obs[5] = heading_err / np.pi
        obs[6] = self._last_action[0]
        obs[7] = self._last_action[1]
        obs[8:] = curv * 10.0
        return np.clip(obs, -4.0, 4.0)
