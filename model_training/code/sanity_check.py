"""Sanity checks for RacingEnv, no RL libraries needed.

1. Gymnasium's built-in env checker (spaces, reset/step contract).
2. Random-action episodes: should fail quickly (stuck or off map) but not crash.
3. A tiny hand-written policy (steer at heading error and lateral offset,
   slow down for upcoming curvature): should make real forward progress and
   ideally complete slow laps. If this policy makes no progress, the
   observation or reward wiring is broken and training would be pointless.

Run:  python sanity_check.py [--episodes N]
"""

import argparse

import numpy as np
from gymnasium.utils.env_checker import check_env

from racing_env import EngineBackedPhysics, RacingEnv
from track_loader import TRACK_WIDTH


# The car the observation describes: obs[0] is forward speed over the
# engine's max_speed and a steering command of 1 is its max_steering.
_PHYSICS = EngineBackedPhysics()
_TOP_SPEED = _PHYSICS.max_speed                    # 85.5 m/s
_STEER_LOCK = _PHYSICS.max_steering                # 30 degrees
_WHEELBASE = float(_PHYSICS._engine.length)        # 2.45 m
# Usable fractions of what the car can do (tyre mu 1.3; physics_tests
# measure 1.11 g sustained cornering and 1.20 g braking). A
# competent-but-not-heroic driver leaves some margin; these are the two
# knobs that set how hard the scripted baseline pushes.  0.65 is a MODELLING
# CHOICE, not swept.
_GRIP_FRAC = 0.65
_MU_G = _GRIP_FRAC * 1.3 * 9.81      # usable cornering accel (m/s^2)
# Assumed braking decel must be close to the real value. Setting it far
# BELOW the truth does not make the car brake earlier and safer, it makes
# the braking stutter: the car over-brakes, the speed target is met, the
# proportional throttle releases, the corner is still there, and it brakes
# again. Traces showed -1.00, -0.61, -0.27, -0.10, -1.00 in nine steps.
_BRAKE = _GRIP_FRAC * 1.2 * 9.81


def heuristic_policy(obs):
    """Drive on the observation alone, like the trained policy would.

    Not tuned for fast laps: it exists only to prove the environment is
    drivable (a competent controller can make real progress and complete
    laps). A trained policy should comfortably beat it.
    """
    heading_err = obs[5] * np.pi
    lateral = obs[4]                       # in half-widths, + = left of center
    curv_now = obs[8] / 10.0               # signed 1/radius (1/m) at the car
    speed = obs[0] * _TOP_SPEED            # m/s
    v_lateral = obs[1] * 10.0              # m/s, + = sliding left

    # Feedback works in CURVATURE space, not wheel-angle space: a wheel
    # angle demands lateral acceleration proportional to v^2, so gains
    # tuned at 15 m/s saturate the tires at 40. Ask for an acceleration,
    # convert to curvature, then to a wheel angle. Lateral error is the P
    # term, v * heading_err its derivative along the path.
    lateral_m = lateral * 0.5 * TRACK_WIDTH    # lateral is in half-widths
    a_des = -0.5 * lateral_m - 1.4 * speed * heading_err
    # The correction budget scales with heading error: a small cap trims
    # the line nicely but is far too weak to catch a slide (at 1.3 rad of
    # heading error the car was commanding 5% of lock while sliding off).
    urgency = min(abs(heading_err) / 0.35, 1.0)
    a_cap = 4.0 + urgency * (_MU_G - 4.0)
    kappa_cmd = curv_now + np.clip(a_des, -a_cap, a_cap) / max(speed, 5.0) ** 2

    # Wheel angle: kinematic (the car's wheelbase) + small understeer allowance,
    # then countersteer into any slide. Countersteer is what a path-only
    # controller lacks: a slide is not a path error yet, so without this
    # the car leaves the track sideways with the wheel nearly straight.
    body_slip = np.arctan2(v_lateral, max(speed, 3.0))
    wheel_angle = (_WHEELBASE + 0.002 * speed ** 2) * kappa_cmd + 0.75 * body_slip
    steer = wheel_angle / _STEER_LOCK

    # Entry-speed rule per lookahead sample: in a corner of radius r the car
    # can do v_corner = sqrt(mu*g*r); a corner d metres ahead allows a higher
    # speed NOW because there is d metres of braking first:
    # v_now = sqrt(v_corner^2 + 2*a_brake*d). The binding sample sets the
    # target, so the car runs fast on straights and brakes early enough.
    curv_ahead = np.abs(obs[8:]) / 10.0
    v_corner = np.sqrt(_MU_G / np.maximum(curv_ahead, 1e-4))
    # Bands are speed-scaled: recover their metre positions from the same
    # helper the env uses. The worst corner in a band could sit right at
    # its start, so braking distance assumes the NEAR edge of each band.
    dist_m = RacingEnv.lookahead_edges(speed)[:-1]
    v_allowed = np.sqrt(v_corner ** 2 + 2.0 * _BRAKE * dist_m)
    target = min(float(np.min(v_allowed)), 0.9 * _TOP_SPEED)

    # Commit to the braking zone. A plain proportional response releases
    # the brake as soon as speed nears the target, so the car re-enters
    # the zone and brakes again; traces showed the pedal cycling
    # -1.00, -0.61, -0.27, -0.10, -1.00 within a single corner entry.
    # Braking hard while over the target and only then blending out is
    # both smoother and what a real driver does.
    # The blend-out band is wide (3 m/s) rather than tight: with a narrow
    # band the car crosses back and forth over the target and the pedal
    # still flickers between full brake and coast.
    over = speed - target
    if over > 3.0:
        throttle = -1.0
    else:
        throttle = np.clip(0.25 * (target - speed), -1.0, 1.0)

    # Off the road: never accelerate away from it (the lookahead may read
    # "clear ahead" while the car slides wide), but never brake to a
    # standstill either, which trips the stuck detector.
    if abs(lateral) > 1.0:
        throttle = min(throttle, -0.4) if speed > 10.0 else min(throttle, 0.2)

    # Braking and steering share one grip pool, so demanding both at the
    # limit spins the car. Steering gets priority; braking uses what is
    # left, with a floor of 0.35 because refusing to brake at all in a
    # corner loses more than the spin risk it avoids (trail-braking).
    steer = float(np.clip(steer, -1.0, 1.0))
    spare = max(1.0 - (abs(kappa_cmd) * speed ** 2 / _MU_G) ** 2, 0.0) ** 0.5
    if throttle < 0.0:
        throttle = max(throttle, -max(spare, 0.35))

    return np.array([steer, throttle], dtype=np.float32)


def run_episodes(env, policy, n, label, options=None):
    print(f"\n{label}:")
    for _ in range(n):
        obs, info = env.reset(options=options)
        total, done = 0.0, False
        while not done:
            action = policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            done = terminated or truncated
        end = ("lap" if info.get("lap_complete")
               else "off_map" if info.get("off_map")
               else "off_track" if info.get("off_track")
               else "stuck" if info.get("stuck") else "timeout")
        lap = f" laptime={info['laptime_steps']}" if "laptime_steps" in info else ""
        print(f"  {info['track']:<14} {end:<8} progress={info['progress_frac']:.2f} "
              f"offroad={info['offroad_frac']:.2f} reward={total:8.1f}{lap}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    print("1) gymnasium check_env ...")
    check_env(RacingEnv(track_names=["Spielberg"], seed=0), skip_render_check=True)
    print("   passed")

    env = RacingEnv(seed=1)
    rng = np.random.default_rng(2)
    run_episodes(env, lambda obs: rng.uniform(-1, 1, 2).astype(np.float32),
                 args.episodes, "2) random policy (expected: quick failures)")
    run_episodes(env, heuristic_policy, args.episodes,
                 "3) heuristic policy (expected: real progress, maybe slow laps)")

    env_fixed = RacingEnv(track_names=["Spielberg", "Spa"], randomize=False, seed=3)
    run_episodes(env_fixed, heuristic_policy, 1,
                 "4) deterministic eval reset (Spielberg, forward, no mirror)",
                 options={"track": "Spielberg"})


if __name__ == "__main__":
    main()
