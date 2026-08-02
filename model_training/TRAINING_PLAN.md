# Plan: training a driving model on the real circuits

Goal: train a driver on the 25 TUMFTM circuits (loaded at the benchmark's own
scale by `code/track_loader.py`), then use it as an independent judge of how
driveable the generated tracks are. The driveability score is used post hoc on
finished experiment outputs, never inside the GA fitness.

## Physics v2 (2026-07-27)

`code/engine_v2.py` replaces the benchmark engine for the driving model
(the benchmark's own engine is untouched and still reachable via
`make_physics("v1")`). Simcade feel: substepped integration, real load
transfer, genuine grip limits with power-on oversteer, human-limited inputs
(33 deg lock at 70 deg/s, 1.0 s lock-to-lock), and none of v1's hidden
driver aids. Acceptance suite `code/physics_tests.py` passes 7/7. Full
detail, including the two bugs this uncovered, is in PHYSICS_PLAN.md.

Two bugs found while validating, both of which had been distorting results:

- The integrator double-counted the Coriolis terms (adding `yaw * v` to the
  accelerations while also re-decomposing world-frame velocity with the
  updated heading each step). The car could pull only 0.63 g with 1.3 g of
  tire grip. **The benchmark engine has the same defect**, so its car has
  been cornering at about half its intended grip; that is a decision point
  for Rafa, not something to fix silently.
- `curvature_ahead` compared a band's upper offset against the absolute
  lower index, so every band scanned out to roughly twice the car's arc
  position. Phantom corners pinned the braking target everywhere, which is
  the real reason the old baseline crawled at 8.9 m/s.

Baseline after both fixes: **25/25 circuits lapped, mean offroad 0.000,
mean speed 26.7 m/s** (was 16/25, 0.076, 8.9 m/s). Cost: env.step 57 to
83 us, so 2M training steps is 2.8 minutes of environment time.

The scripted heuristic needed a substantial rebuild, and each change is a
skill the RL policy will have to learn: feedback in curvature space rather
than wheel-angle space, a correction budget that scales with grip, sharing
the grip pool between braking and steering, and above all COUNTERSTEER.
That last one took the test suite from 3/6 to 6/6: all remaining failures
were the car sliding off sideways (lateral velocity 15-25 m/s) while
commanding almost no steering, because a path-following controller cannot
see a slide that is not yet a path error. Under v1 a controller that
ignored the lateral-velocity and yaw-rate channels could lap 16 circuits;
under v2 it laps none.

## Reward reshaping and rolling starts (2026-07-28)

Problem reported: trained models stick religiously to the centerline and
wobble around it at low speed. Measured cause: progress reward paid per
metre regardless of speed, so on Spa driving 2.3x faster changed episode
return by about 2%. Creeping safely was almost as good as racing, and much
less risky, so creeping is what was learned.

Reward changes in `racing_env.py` (the racing line itself is deliberately
NOT rewarded; it has to emerge because it is the fastest way round):

- `TIME_COST = 0.03` subtracted every step. The only way to keep more of a
  lap's progress reward is now to finish it in fewer steps: the objective
  becomes lap time, not distance. It also makes standing still strictly
  negative.
- `LAP_TIME_BONUS` 30 -> 200: finishing fast is worth ~a third of a lap's
  reward instead of 5%.
- `STEER_RATE_SCALE` 0.02 -> 0.05: full-lock sawing all lap now costs ~500,
  not ~200, which is what the wobble was.

Incentive profile after the change (Spa): creep (900 s) 329, old-agent pace
505, heuristic pace 672, fast 701. Before, all of these landed within a few
percent of each other.

First pilot with the new reward (`runs/reshaped`, 1.5M steps) exposed a
bootstrap failure: eval sat at reward -13.2 / episode length exactly 101
from the first eval to the last. That is the stuck-detector kill (-10 fail
plus 101 time costs): from a standstill, ~100 steps of near-random throttle
cover under a metre, so every early episode ends identically and PPO
collapses to "never move" before discovering driving. Two fixes:

- Rolling starts: randomized (training) resets now spawn the car already
  moving at uniform(0, 30) m/s along the track (`start_speed` reset option;
  both engines store world-frame velocity and the auto gearbox recovers the
  right gear on the first step). Progress reward and its gradient are felt
  from step one. Low speeds stay in the range so launches are still
  learned; eval keeps standing starts.
- `ent_coef` 0.005 -> 0.01 in train.py so exploration survives longer.

Second pilot: `runs/reshaped2`, 1.5M steps, 12 envs.

Follow-up (same day), after checking the reward against the racing RL
literature (GT Sophy: dense course progress + off-course penalties; classic
TORCS DDPG: v*cos(heading) - centerline terms):

- Leaving the track now terminates IMMEDIATELY: `OFF_TRACK_LIMIT` 3.0 ->
  1.1 half-widths (1.0 = edge, 0.1 kerb allowance), -10 as before.
- The overshoot penalty (`OFFROAD_SCALE`) was removed as dead code: with
  immediate termination there is no lingering off-road state to penalise.
- Steer-rate penalty is now squared (0.05 * delta^2): near-free for small
  corrections, expensive for full-lock sawing.
- Two lane-keeping-style terms were considered and deliberately left out:
  a flat cos(heading_error) alignment bonus (a per-step survival stipend
  that re-creates the creeping optimum; the literature's version is
  speed-multiplied, which ds per step already is) and a quadratic
  centerline-distance penalty (that term is exactly what makes agents hug
  the centerline; racing needs the full track width, and staying on track
  is handled by termination). Rationale in the racing_env.py docstring.

Consequence to expect: the scripted heuristic occasionally clips beyond
the edge (offroad_frac ~0.003), so under strict termination it may now
fail circuits it used to lap. That is the environment getting stricter,
not the driver getting worse; judge the heuristic's numbers accordingly.

## Pace push (2026-07-28, after run5)

run5 (graded penalties + speed-scaled bands) drives cleanly on both the
held-out circuits (5/5 laps, offroad <= 0.007) and the user's generated
tracks, but conservatively: 16-17 m/s mean. User wants a slightly harsher
off-track penalty and stronger encouragement for fast lap times. Two
constants changed, nothing else:

- TIME_COST 0.03 -> 0.05. This is the only lever that actually prices lap
  time: LAP_TIME_BONUS is paid ~3000 steps away and gamma=0.995 discounts
  it to ~1e-4, so the per-step cost is what the agent feels. Break-even
  speed (where progress stops out-earning the clock) rises from 3 to
  5 m/s; every second of lap time now costs 0.5 reward instead of 0.3.
- OFFROAD_SCALE 1.0 -> 1.5. Raised together with TIME_COST so the extra
  pace pressure cannot be paid for by running wide: the off-track band
  must stay more expensive relative to the clock than before, not less.

Observation unchanged (21 floats). Value scales shifted, so train from
scratch or fine-tune from run5; fine-tuning with lower ent_coef (~0.003)
is the cheaper first try.

## Graded off-track penalties (2026-07-28, after run4)

run4 (5M steps, speed-scaled bands, cliff termination at 1.1 hw) learned a
constant-pace policy: ~25 m/s everywhere, 0/5 held-out laps, dying at
tight corners carrying 21-32 m/s where 10-17 was survivable, eval reward
plateaued at ~40-47 for the last 2.5M steps. Diagnosis: a hard cliff at
the track edge gives NO gradient about corner-entry speed. On-track = no
signal, one step over = flat -10; "clipped the kerb" and "flew off at
40 m/s" cost the same. The model that drove well (reshaped2) was trained
when the graded overshoot penalty and lenient 3.0 hw limit still existed.

Fix (user-requested: harsher, scaled with how far off the car goes; also
scaled with speed, as GT Sophy scales off-course penalties with kinetic
energy):

- On track (<= 1.0 hw, +0.1 kerb allowance): progress pays as before.
- Off-track band (kerb to 2.0 hw): no progress credit, per-step penalty
  overshoot * (1.0 + 4.0 * v/vmax). At 30 m/s and half a half-width off
  that is -1.22/step vs the +0.30 the same step would have paid on track.
- Beyond 2.0 hw: terminate, penalty 20 + 30 * v/vmax (also for off-map;
  stuck keeps the base 20). Flying off at 30 m/s costs -30.8, dribbling
  off at 5 costs -21.8.

Observation unchanged (21 floats), so run4 checkpoints still load, but
value scales changed: retrain from scratch. If the next run still fails
to brake for hairpins, the next single-variable suspect is the
speed-scaled bands themselves (braking makes an approaching corner recede
in time-band space, which may blur the brake-now signal).

## Speed-scaled lookahead bands (2026-07-28)

The fixed 325 m band edges served both ends badly, measured in time:
at 83 m/s braking takes 293 of the 325 visible metres, leaving 0.4 s of
free look (the car was blind at speed, committed to everything it could
see), while at 8 m/s seven of ten bands described track more than 4 s
away (noise the network must learn to ignore exactly where hairpins
live). The near end was also too coarse for apexes: at 70 m/s the first
band spanned one control step.

Band edges are now fixed TIMES ahead, converted to metres with current
speed: `LOOKAHEAD_TIMES` = 0.2 s to 8.25 s over 13 bands (was 10),
`lookahead_edges(v)` = `max(v, 15) * times`. The 15 m/s floor keeps the
nearest band at the 3 m centerline sampling interval and the car sighted
at a standstill. Result: horizon is a constant 8.2 s at any pace, 685 m
at top speed (braking point visible with ~4.7 s to spare), 124 m when
slow, with 3 m near-band resolution for apex placement.

Observation grows 18 -> 21 floats. OLD MODELS ARE INCOMPATIBLE (shape
mismatch fails loudly on load; view_track falls back to the scripted
agent with a warning). The heuristic and rl_agent_adapter now derive
band distances from the same `lookahead_edges` helper, so all three
consumers stay in lockstep. Sanity suite passes; heuristic laptimes
unchanged (Norisring 785 vs 784 steps).

## Fixes and additions (2026-07-27)

Two bugs in the training harness, both of which would have silently spoiled
the first run:

1. The eval callback only ever tested one circuit. A non-randomized env
   always reset to the first track in its list, and `EvalCallback` passes no
   reset options, so all five "held-out" episodes ran on the same circuit and
   best-model selection was based on it alone. Non-randomized envs now walk
   their track list in order, one circuit per reset, so the five eval
   episodes cover the five held-out circuits exactly once each.
2. No `Monitor` wrapper. It was imported but never applied, so
   stable-baselines3 would have logged no `ep_rew_mean` or `ep_len_mean` and
   `EvalCallback` would have reported inaccurate episode rewards. Every env
   is now wrapped.

Smaller environment fixes:

- A lap completing on the same step as a failure awarded the lap bonus and
  the failure penalty together and set both info flags. Failures are now
  checked first and are exclusive with a finished lap.
- The car was projected onto the centerline twice per step (once in `step`,
  once in `_observe`). Projection is the most expensive per-step operation;
  it is now done once and shared.
- Added read-only `geometry`, `car_state`, and `physics_dt` accessors so
  analysis code can record a driven path without touching internals.

New files:

- `visualize.py`: draws a driven lap (road edges, centerline, path colored by
  speed, start/end markers, outcome in the title). This is how to see WHY a
  policy scores what it scores, and it produces thesis figures directly.
- `render_run.py`: plays a lap back as an animated GIF, so the driving
  itself can be watched rather than just its trace. Follow-camera by
  default (`--whole-track` fits the whole circuit), HUD showing speed,
  throttle/brake, steering, elapsed time, and body slip angle whenever the
  car is sliding more than 3 degrees. `--model` renders a trained policy.
- `baseline.py`: Phase 2. Runs a policy over every circuit deterministically
  and writes `results/baseline.csv` (outcome, progress, offroad fraction,
  laptime, mean and max speed, steering jerk, reward). The same script scores
  a trained model with `--model`, so both are measured by identical code.
- `generated_tracks.py`: the bridge to generated tracks, which Phase 5 needs.
  Real circuits and generated tracks share a unit system, so a generated
  track needs no rescaling: its centerline comes from the benchmark's own
  `info()["curve_points"]` and goes straight into `TrackGeometry`. The env
  subclass inherits observation, reward, and termination unchanged, which is
  the point, since generated tracks must be judged by identical rules.

## What the v1 baseline revealed (historical)

Superseded by the v2 baseline above, but kept because the diagnostic method
is reusable and one finding still applies.

The v1 heuristic finished 16 of 25 circuits at mean offroad 0.076 and a mean
speed of only 8.9 m/s. Most of that slowness turned out to be the
`curvature_ahead` band bug (see the v2 section), not conservatism.

Two distinct failure modes hid behind similar offroad numbers, and the
`steer_jerk` column separated them: IMS (offroad 0.47, jerk 0.034 against
0.004-0.008 elsewhere) was the controller weaving across a nearly straight
road, while Sochi and Shanghai (offroad ~0.49, jerk 0.005) were not weaving
at all but sitting on the inside edge through 97-99% of corner time, which
is apex-cutting. Using jerk plus the sign of lateral offset against
curvature to tell oscillation from a steady offset is worth reusing.

Still true under v2: do NOT add a yaw-rate damping term to a steering
controller here. A gain sweep of `-kd * (yaw_rate - speed * curvature)` over
kd = 0.15 to 0.8 made every circuit monotonically worse and RAISED jerk,
because the feedback arrives a full timestep late at dt = 0.1 s and pumps
the oscillation instead of damping it. Countersteer on body slip angle (see
the v2 section) is the correct way to use those observation channels.

## Generated centerlines can leave the map

Sampled `racing-v0` tracks routinely put their centerline outside the 750 m
map (observed x up to 768, y down to -16). Such a track cannot be lapped
whatever its shape, so `generated_tracks.centerline_escapes_map` reports it
as out of bounds instead of letting the driver take an `off_map` failure for
it. Blaming the driver there would make the driveability score depend on a
property the track shape does not explain.

First bridge test, three random samples per representation: `racing-v0` all
out of bounds; `racingradial-v0` stuck or off-track at 0.13-0.38 progress;
`racingtile-v0` clean laps (offroad 0.00-0.03); `racingtilehex-v0` 0.93-0.97
progress, clean, timing out on the same conservative-speed budget;
`racingvoronoi-v0` mostly clean laps. That is the discrimination Phase 4
asks for, already visible with the scripted policy.

## Current state (2026-07-25)

- Scale settled: the benchmark's simulation world is metric, 1 world unit =
  1 metre (the engine integrates position with velocity in m/s; PX_PER_M = 5
  in problem.py is only the world-to-pixel factor for rendering). Circuits are
  therefore imported at 1:1, every one at the exact same scale, with real
  corner radii and lengths (2.3 to 7.0 km). Real circuits span up to ~2.2 km,
  so each track carries its own bounding-square `map_size` instead of the
  fixed 750 m benchmark map; the Gym env uses that per-track size.
- Sanity suite passes at 1:1: random policy fails fast (0-3% progress,
  negative reward); the scripted heuristic laps Catalunya, Austin, Budapest,
  Oschersleben cleanly (offroad 0.00-0.03, reward +380 to +550) and reaches
  96-97% on MexicoCity/Spielberg before the conservative-speed step budget
  runs out. That separation confirms observation, reward, and termination are
  wired correctly and the env is trainable.

## Watching the driver

Two views of the same rollout, both working with the scripted heuristic now
and with a trained model later (`--model <path>`):

    python visualize.py --track Spa            # static, path coloured by speed
    python visualize.py --all --out ../figures

    python render_run.py --track Spa           # animated GIF, camera follows
    python render_run.py --track Monza --whole-track --speedup 4
    python render_run.py --all --out ../renders

Use the static plot to judge the line over a whole lap, the GIF to see what
the car is doing moment to moment (braking points, slides, throttle on
exit). `--speedup N` draws every Nth step and keeps the GIF playing in real
time.

Both sets are checked in as of 2026-07-27 and show physics v2: `figures/`
has a PNG per circuit, `renders/all_tracks/` a GIF per circuit (31 MB
total, all 25 completed laps at offroad 0.00). The IMS pair is the clearest
before/after in the project: under v1 the car sawtoothed across the full
track width (offroad 0.47, 2972 steps), under v2 it holds a clean line on
the banking at 40-50 m/s (offroad 0.00, 1003 steps).

## Phase 0: prerequisites (before any training)

1. Settle the physics question with Rafa (see thesis/physics-model-decision.md
   and PHYSICS_PLAN.md, the staged TORCS-like upgrade plan with measured
   costs and the three decision points).
   The driver must be trained on the physics that will be used for evaluation.
   Training first and swapping physics later invalidates the model.
2. Freeze the agent-facing interface: centerline polyline + constant width 16.

## Phase 1: Gymnasium environment  [BUILT 2026-07-25]

Done and sanity-checked. Files in `code/`:

- `track_loader.py`: TUMFTM CSVs to world units at 1:1 (see scale note
  above), per-track `map_size`, raceline loader, stats table / --save CLI.
- `track_geometry.py`: precomputes centerline projection (warm-start, with a
  full-scan fallback so a stale hint never fabricates a lateral offset),
  arc-length, signed curvature, and banded curvature-ahead (each observation
  slot reports the worst curvature in its distance band, so short corners
  cannot hide between sample points). Provides reversed / mirrored variants
  for augmentation.
- `racing_env.py`: `RacingEnv` (Gymnasium). Physics hidden behind
  `EngineBackedPhysics`, so swapping to a TORCS-style engine touches nothing
  else. 18-float observation (curvature bands out to 325 m, because braking
  from 83 m/s takes ~270 m), 2-float action, reward and termination as
  documented in the file. Off-track (>3 half-widths) ends the episode and
  blocks the infield-shortcut exploit (progress reward is not credited past
  that limit).
- `train.py`: PPO harness (needs sb3+torch, not yet installed). 20 train / 5
  eval circuit split, held-out eval = Spa, Silverstone, Suzuka, Monza,
  Montreal.
- `sanity_check.py`: gymnasium check_env + random vs scripted-heuristic
  policies (results in the current-state section above). The heuristic uses
  the engine's speed-blended steering lock and an understeer feed-forward;
  both were needed to drive clean at real scale and are worth remembering
  when judging what the RL policy has to learn.

Original design notes for reference:

- Observation (normalized, car frame):
  speed, lateral velocity, yaw rate,
  signed lateral offset divided by half width,
  heading error against the track tangent,
  10 curvature samples at fixed arc distances ahead (for example 5, 10, ... 50
  map units), which is the "what is coming" signal.
- Action: continuous, steer in [-1, 1] and throttle/brake in [-1, 1].
- Reward per step: progress along the centerline (delta arc distance), minus a
  penalty proportional to how far beyond half width the car is, minus a small
  steering-rate penalty. Lap completion gives a bonus scaled by the remaining
  step budget so faster laps score more.
- Termination: map out of bounds, no forward progress for ~100 steps, or lap
  complete.
- Reset: pick a random circuit, random start point on the centerline, random
  direction, optional mirror. That turns 25 circuits into ~100 layouts with
  unlimited start variation, which is the main defense against overfitting.

## Phase 2: baseline before training  [DONE 2026-07-27]

`python baseline.py` writes `results/baseline.csv` for all 25 circuits. With
physics v2: **25/25 finished, mean offroad 0.000, mean speed 26.7 m/s**,
laptimes 87 s (Norisring) to 243 s (Spa). Rerun with `--model <path>` to
score a trained driver through identical code.

Every row also reports `grip_used` and `slip_p95`. After the 2026-07-28
driving audit (PHYSICS_PLAN.md) the scripted driver actually uses the car:
**87% of tire grip**, 30.9 m/s average, and **2.0x real F1 lap records**,
which is about right for a GT3-class car (F1 has ~5 g of grip against this
car's 1.3 g). It brakes hard into corners instead of coasting down to
them, and still laps 25/25 at 0.003 mean offroad.

That makes it a demanding baseline. The trained driver has to beat a
controller already near the limit, so the comparison is laptime and
smoothness, not completion. The one obvious weakness left to exploit is
steering smoothness: the scripted policy twitches on straights (9-19% of
steps reverse the steering sign, `steer_jerk` 0.003-0.027).

## Phase 3: training (1 day setup, then iteration)

- stable-baselines3 PPO, MlpPolicy with two hidden layers of 128, 4 to 8
  parallel envs. Budget 2 to 5 million steps.
- Hold out 5 circuits entirely (suggest Spa, Silverstone, Suzuka, Monza,
  Montreal): the eval callback runs only on these. Generalization to unseen
  tracks is the whole point, because generated tracks are always unseen.
- Local compute: the env is tiny (a polyline and 6 floats of car state), so RAM
  is not the bottleneck this time; a few million PPO steps on CPU is hours to
  overnight. Run headless, checkpoint every 100k steps so a crash loses little.
  Fallback if the PC struggles: Colab or DelftBlue.
- Expect 2 to 4 reward-shaping iterations. The classic failure modes to watch:
  the car learns to creep (too much offroad penalty), or learns to cut corners
  through the grass (too little), or oscillates (no steering-rate penalty).

## Phase 4: validate the instrument (1-2 days)

Before using the driver to judge anything:

1. It must finish the 5 held-out real circuits cleanly (offroad well under the
   scripted agent's 0.33-0.40). Real circuits are driveable by construction, so
   failing here means the driver is broken, not the tracks.
2. Run it on a batch of random and evolved tracks from all five
   representations. It should separate them: clean laps on good evolved tracks,
   struggles on random garbage. If it fails everything unfamiliar, it overfitted
   and needs more augmentation.
3. Evaluate with the deterministic policy (deterministic=True) and fixed seeds,
   so the same track always gets the same score. Save the exact model file used
   for any thesis number.

## Phase 5: driveability score (1 day)

`driveability(track) -> dict` over: finished, laptime ratio against the scripted
agent on the same track, offroad fraction, steering jerk, mean speed. Report the
components separately in the thesis rather than collapsing them into one number;
a single scalar hides which property failed. A post-hoc script sweeps the saved
GA experiment outputs and produces one table per representation.

The plumbing is in place: `generated_tracks.py` turns benchmark content into
the same environment the circuits use (`env_for_content(problem, content)`),
`baseline.evaluate_track` computes the metric row, and
`centerline_escapes_map` separates unlappable tracks from driving failures.
What remains for Phase 5 is the sweep over saved experiment output and the
per-representation table.

## Deliberate scope choices

- The driver is shared by all five representations, like the car model, so it
  cannot bias one representation over another.
- It stays out of the GA fitness: fitness stays the shared quality function;
  the learned driver is an independent judge applied afterwards.
- Circuits are imported at 1:1 (world units are metres, exactly like the
  benchmark), so corner radii, lengths, and speeds are real. The only
  difference from the benchmark setup is the world size: real circuits get a
  per-track bounding square instead of the fixed 750 m map, which only the
  Gym env uses, so nothing in the benchmark changes.

## Dependencies to install (in the venv)

    pip install gymnasium stable-baselines3 torch tensorboard
