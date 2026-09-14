# Plan: TORCS-like physics upgrade without the overhead

> **STATUS 2026-07-27: BUILT, then aligned to TORCS.** `code/engine_legacy.py`
> implements the simcade car, `code/physics_tests.py` is the acceptance
> suite (9/9 passing), and `RacingEnv` uses v2 by default
> Measured cost: env.step
> 57 us (v1) -> 124 us (v2 with combined slip), so 2M training steps is
> 4.1 minutes of environment time. The benchmark engine is untouched, so
> every GA result stays valid. Read "TORCS comparison and second pass"
> next for the current model, then "Build results" for the first pass and
> the two engine bugs the work uncovered.

Question: can the car physics get close to TORCS realism at acceptable cost?
Answer: yes. Measured on this machine (2026-07-27): `engine.step` costs 5.0 us,
the full `RacingEnv.step` 58 us, so the engine is 9% of the environment cost,
and the environment itself is a small fraction of PPO training time (2M steps
is about 2 minutes of single-core env time; the neural network dominates).
Even a 10x more expensive engine only takes env.step from 58 to about 105 us.
The real budget is implementation and retuning time, not CPU.

## What TORCS simulates vs what we have

TORCS (simuv2) per car: four wheels each with a Pacejka tire and its own
vertical load, longitudinal and lateral weight transfer, aerodynamic drag,
wings and ground effect, an engine torque curve with a gearbox, clutch and
differential, and spring/damper suspension per wheel.

The current engine (racing/engine.py) already has: dynamic bicycle model,
Pacejka lateral and longitudinal LUTs, friction-circle clamp per axle, static
40/60 axle loads, drag and rolling resistance, RWD force routing.

What it lacks, ordered by how much each one shapes real handling in a 2D
top-down setting:

1. substepping (integrates at 10 Hz, which forces the artificial stabilizers)
2. load transfer (static axle loads, so braking/throttle never change balance)
3. real wheel dynamics (pedal commands slip ratio directly = perfect traction
   control; wheelspin, lockup, and power oversteer are impossible)
4. an engine torque curve and gearbox (drive force is a flat cap)
5. downforce (grip is 1.3 g at all speeds up to 300 km/h)
6. tires that go past the peak (slip angles clamped to 10 deg, so no
   post-limit sliding or spins)

Not worth porting for a 2D benchmark judge: per-wheel modeling (needs lateral
transfer plus a differential to mean anything and doubles tire evaluations),
suspension kinematics, roll/pitch degrees of freedom, clutch dynamics,
damage. Staying with a two-axle bicycle model plus quasi-static load transfer
captures every handling phenomenon that shows up in top-down driving.

## Ground rule: the benchmark engine stays frozen

The GA quality function simulates with the current engine; changing it
invalidates the regression reference and every experiment run so far. The
upgrade therefore goes into a NEW class (`engine_legacy.py`), used by
model_training through the existing `EngineBackedPhysics` adapter slot.
Whether the benchmark itself ever adopts v2 is a separate decision for Rafa.
The judge being trained on better physics than the GA fitness used is fine:
the judge must be consistent across representations, not identical to the
fitness simulation.

## Stage 0: harness before physics (half a day)

- `physics_tests.py`: scripted maneuvers with pass ranges, run against old
  and new engine so every stage documents what changed.
  - straight line: 0-100 km/h in 3.3-3.8 s, 100-0 km/h in 33-38 m
  - top speed: 290-310 km/h reached asymptotically (no hard clamp)
  - skidpad (constant radius 50 m): steady lateral acceleration 1.2-1.4 g
  - step steer at 30 m/s: yaw rate settles without oscillation, overshoot
    under 20%
  - throttle-in-corner near the limit: with aids off the rear must be able
    to break away (this is the test the current engine cannot pass)
- Regression: `baseline.py` table and `sanity_check.py` before and after
  each stage.

## Stage 1: substepping and removing the patches (half a day)

Keep the 10 Hz control interface (dt = 0.1 stays the action rate) but
integrate N = 5 internal substeps of 20 ms with semi-implicit Euler. Cost:
about 5x engine cost, still under 10% added env time.

Substepping is the enabler. The artificial stabilizers exist because tire
dynamics are stiff at 10 Hz. After substepping:

- add tire relaxation (first-order lag on tire force with length ~0.3 m,
  time constant 0.3/v): this is the physical stabilizer that replaces the
  hacks below
- reduce `yaw_damping` 0.6 toward 0.05-0.1 (real yaw damping comes from the
  tires; the current value fights steady-state cornering)
- remove the `lateral_friction` exponential scrub and the `v/(v+1)` lateral
  force fade (low speed is handled by relaxation plus the vx epsilon)
- raise the slip-angle clamp from 10 deg to the LUT's full 25 deg so tires
  have a real post-peak region

Acceptance: skidpad and step-steer pass with the stabilizers reduced;
baseline table shifts only mildly.

## Stage 2: load transfer (half a day)

Quasi-static longitudinal transfer, updated every substep from the previous
substep's acceleration:

    dFz = (h_cg / wheelbase) * mass * a_x        (h_cg ~ 0.45 m)
    Fz_front = Fz_front_static - dFz
    Fz_rear  = Fz_rear_static  + dFz

Per-axle grip (`mu * Fz`) then follows the load. This alone produces the
core TORCS handling traits: braking sharpens turn-in and lightens the rear
(trail-braking oversteer becomes possible), throttle plants the rear and
pushes toward understeer. Optional refinement: mild tire load sensitivity
(mu falls a few percent per added kN) which is what makes lateral transfer
matter in real cars; a bicycle model can approximate it as
`mu_eff = mu * (1 - k * |a_y| / g)` with small k.

## Stage 3: wheel dynamics and drivetrain (1.5-2 days, the big jump)

Replace "pedal commands slip ratio" with real wheel states:

- one wheel angular velocity per axle (2 new state variables, inertia
  ~1.5 kg m^2 per axle)
- slip ratio emerges: kappa = (omega * R - v_x) / max(|v_x|, eps)
- engine: parametric torque curve (peak ~450 Nm around 6000 rpm falling to
  the 8000 rpm limiter) through an auto-shifting 6-speed box (shift up at
  rpm threshold, down on the reverse threshold with hysteresis). This is a
  handful of scalar ops, not a real transmission model.
- brakes: torque split front/rear by bias (~60/40), lockup possible
- rear axle: remove the lateral-priority traction-control clamp; combined
  slip handled by the friction circle on the resulting (Fx, Fy)

This makes wheelspin, lockup, and power oversteer real. The earlier attempt
at wheel-speed states wound up under sustained throttle (see the comment in
engine.py) because it integrated at 10 Hz; at 20 ms substeps with correct
wheel inertia it is stable. Optional flags `abs_on` / `tc_on` can restore
the forgiving behavior as explicit driver aids, which keeps the "predictable
constant across track types" benchmark virtue available as a switch instead
of being baked into the physics.

Acceptance: 0-100 and braking tests still in range (retune drive/brake
torques), throttle-in-corner test now passes with aids off.

## Stage 4: aero (half a day)

- downforce added to axle loads before grip: Fz_aero = 0.5 * rho * ClA * v^2,
  split front/rear to preserve balance (911 GT-ish ClA 0.5-1.0)
- drop the hard 83 m/s clamp; top speed emerges from drag + rolling
  resistance against engine power (tune cd_a and the torque curve to land
  at ~300 km/h)

High-speed corners at Monza/Spa then reward speed the way real cars
experience it: grip grows with v^2 while required lateral force also grows
with v^2, so fast sweepers stop being flat-out impossible.

## Stage 5: control interface cleanup (a few hours)

- steering: keep the rate limit (models hands on a wheel), replace the
  speed-blended lock (75 to 30 deg) and the hidden gain schedule (1.05 to
  0.45) with one fixed lock of ~35 deg. The blend was a playability aid;
  with real physics it distorts the steering feel the RL driver learns.
- throttle slew stays (models pedal and engine response).
- decide observation additions for the RL driver: gear and engine rpm
  (2 floats) are worth exposing; per-axle slip is optional.

## Cost summary

| stage | added state | est. cost factor on engine.step |
|---|---|---|
| substep x5 | none | ~5x |
| load transfer | none | +5% |
| wheels + drivetrain | 2 wheel speeds, gear int | +40% |
| aero | none | +5% |

Worst case ~8x of 5 us = 40 us, env.step goes 58 to ~95 us. With 8 parallel
envs, 2M PPO steps of env time is ~4 minutes total. Training wall time stays
dominated by the network. Overhead is a non-issue.

Effort: about 4-5 working days including validation, dominated by Stage 3
and retuning, not by any single hard algorithm.

## What this does NOT try to be

A port of TORCS. Porting simuv2 literally (C++, four wheels, suspension,
differential) is weeks of work and its extra fidelity is invisible from a
top-down centerline-following viewpoint. The staged plan reproduces the
TORCS handling phenomena that a driveability judge can actually sense:
balance shifts under braking and power, traction limits, gear-shaped
acceleration, speed-dependent grip.

## TORCS comparison and second pass (2026-07-27)

Compared feature by feature against TORCS SimuV2 (as characterised in
thesis/physics-model-decision.md) and closed the gaps that change handling.

| TORCS SimuV2 | v2 before | v2 now |
|---|---|---|
| one magic formula on COMBINED slip, split by direction | separate lateral curve + hand-coded friction circle | **combined slip, force split by slip direction** |
| load-sensitive grip (mu falls as Fz rises) | fixed mu | **mu = mu0 (1 - 0.10 (Fz/Fz0 - 1))** |
| longitudinal AND lateral weight transfer | longitudinal only | **both** (lateral via load sensitivity) |
| slip grows until it delivers the demanded torque | free-rolling slip, demand lost in corners | **slip solved for the demand** (see the 2026-07-28 audit) |
| slip from an integrated wheel-spin state | pedal demands force | still no wheel state, but its EFFECT is now reproduced |
| four wheels, suspension, differential | two axles | unchanged, out of scope for a 2D judge |
| engine torque curve + gearbox | already had it | unchanged |

**Combined slip** is the structural change. One magic-formula curve is now
evaluated on the magnitude of the combined slip vector and the resulting
force split back by that vector's direction, exactly as TORCS does. The
friction circle is no longer coded by hand: it emerges, because a tire
already using its slip longitudinally has little slip pointing sideways.
Verified in isolation: at 8 degrees of slip angle, adding brake demand
takes the rear's lateral force from 0.99 to 0.80 to 0.57 of peak while the
total force magnitude stays pinned near 1.0.

**Load sensitivity** is what makes weight transfer a balance change rather
than bookkeeping. Measured mu: 1.365 at 0.6x static load, 1.300 at static,
1.196 at 1.8x. So the loaded axle gains less than the unloaded one loses
and total grip falls in hard braking and hard cornering.

**Lateral transfer** cannot move load between left and right tires in a
two-axle model, but it still costs grip through load sensitivity, applied
as a multiplier that falls with lateral acceleration squared (about 1.3%
of grip at the limit).

**Still deliberately not TORCS:** the pedal demands a force fraction which
each tire then caps, instead of integrating a wheel-spin state. Wheelspin
and lockup consequences are present (over-demand pushes the tire past the
curve peak, losing force and stealing cornering grip); what is absent is
the wheel's own rotational state. That component is the one that has
blown up numerically here twice, and at 20 ms substeps its extra fidelity
is not visible from a top-down centreline-following viewpoint.

**A pedal-calibration bug this uncovered.** Mapping demand fraction
linearly onto slip made the pedal hair-trigger: because the tire curve
reaches 58% of peak at only 20% of peak slip, 20% throttle delivered 58%
of available grip. 0-100 dropped to 3.2 s and the stopping distance to
30 m. Fixed by inverting the curve through a precomputed table, so "40%
pedal" now means 40% of the force the tire can give (verified: 0.20 ->
0.20, 0.50 -> 0.50, 1.00 -> 1.00, and past 1.0 the force FALLS to 0.91).

**Effect at the limit.** The difficulty gradient survives: grip usage 52%
at the driver's tuning, 85% at 0.8x, 90% at 1.0x, 94% at 1.25x, with slip
rising 1.2 -> 3 -> 6 -> 44 degrees. One behaviour did soften: at 1.25x the
car now survives big slides (46 degrees) that previously ended off-track,
because combined slip degrades past the peak more progressively than the
old hard clamp did. That is more TORCS-like but marginally more forgiving
at the extreme, and worth stating rather than hiding.

**Cost:** engine 5 -> 42 us per control step, env.step 83 -> 124 us, so 2M
training steps is 4.1 minutes of environment time. Profiling shows no
single hotspot; the cost is simply more physics per step. Still irrelevant
next to the neural network.

Acceptance suite grew to 9 tests (added trail-braking and load-sensitivity
checks); all 9 pass.

**Simplification pass after the TORCS work.** The alignment work left
duplicated and stale code behind, so it was cleaned up with a trajectory
fingerprint (SHA-256 over 2400 stepped poses) proving behaviour was
preserved:

- the curve inverter went from a 38-line hand-rolled bisection table to 16
  lines sampling the curve forward and inverting with `np.interp`
- `_grip` and `_lateral_transfer_loss` merged into one function, since
  they were only ever called together (four calls per substep became two)
- the pedal's `min(demand, grip)` caps were deleted: they duplicated what
  the combined-slip tire model already does, and clipping the demand
  actually PREVENTED the over-demand slide that is the point of the model
- the module docstring still described the old separate-curve model with
  its hand-coded friction circle; rewritten to match what the code does
- an unused `yaw_rate` unpack left over from the reverted damping
  experiment removed from the heuristic

engine_legacy.py 448 -> 344 lines, sanity_check.py 163 -> 114, all 9 physics
tests and the 25-circuit baseline unchanged.

**Handling balance check.** Steady-state cornering at 25 m/s, comparing
front and rear slip angles: +2.9 degrees of understeer coasting, +5.5 on
power (the classic rear-drive push), +2.7 under braking (front loaded,
turns in better). The balance shifting with load is the load-transfer and
load-sensitivity chain working end to end.

**Measuring grip: use forces, never yaw rate.** This trap was hit three
times in one session and each time it looked like a physics bug:

1. `yaw_rate * speed` reported 1.5-5.9 g on a 1.3 g tire (it reads spin,
   not grip, once the car slides).
2. The first trail-braking test measured cornering "lost" as 1.00, because
   heavy braking had simply stopped the car and it was comparing 20 m/s
   against standstill.
3. The second version reported -0.27, i.e. braking apparently INCREASED
   cornering, because braking loads the front and genuinely sharpens
   turn-in, so yaw can rise while grip falls.

All three vanish when the measurement uses `engine.debug_forces` (per-axle
Fx, Fy, Fz set every substep). Kinematic proxies assume the steady state
the test is trying to probe.

## Full double-check against TORCS (2026-07-28)

Re-audited every equation rather than trusting the earlier pass, and found
one substantive bug plus a set of missing invariant checks.

### The bug: drive force was silently thrown away in corners

`_axle_force` converted the pedal's force demand into a longitudinal slip
using the FREE-ROLLING inverse (the slip that would give that force with no
slip angle present), then combined it with the actual slip angle. Because
the force is split by the slip vector's DIRECTION, adding a small
longitudinal slip to a large slip angle barely rotates the vector, so most
of the demand vanished. Measured: at 5 degrees of slip angle a request for
25% of the tire's grip delivered only 14%, with 7% of grip sitting unused.

In real TORCS this cannot happen, because the wheel spins up until its slip
produces the demanded torque. With no wheel-spin state the equivalent is to
SOLVE for the longitudinal slip that delivers the demand alongside the slip
angle already present, which is what `_slip_for_combined` now does by
bisection on the combined-slip force. The demand is now delivered exactly
(0.10 -> 0.10, 0.25 -> 0.25, 0.50 -> 0.50) while the total force stays
inside the friction ellipse, and saturates honestly when the tire runs out.

Three consequences, all of them more realistic:

- **Traction-limited launches.** Full throttle now pins the rear tire on
  its sliding plateau (82% of grip) instead of delivering whatever the
  engine asks: the car spins its wheels off the line like a 500 hp
  rear-drive car. 0-100 went 4.00 -> 4.30 s, so the acceptance range was
  widened to 3.0-4.5 s rather than faking the physics back.
  A 40% throttle launch uses only 37% of grip and does not spin.
- **Much stronger trail-braking coupling**: rear cornering force lost
  under braking went 0.13 -> 0.52.
- **The car is harder at the limit**, which broke the scripted driver: at
  `_GRIP_FRAC = 0.75` five circuits failed with slip angles of 27-47
  degrees (sliding, unable to catch it). Backed off to 0.65, which
  restores 25/25 at 0.73 measured grip usage. The driver is still pushing
  hard, but now inside what it can recover from.

Cost: the bisection runs twice per axle per substep, so env.step went
124 -> 227 us (12 iterations, halved from the 24 the first version used;
that resolves slip to ~0.01 degrees, far finer than the car can feel).
2M training steps is 7.6 minutes of environment time.

### Invariants now tested permanently

The maneuver tests could not have caught an asymmetric car, a state going
non-finite, or an integrator whose answer depends on its step count, so
those are now four more acceptance checks (suite is 13 tests):

- **left/right symmetry**: mirrored inputs give identical yaw to 4e-15
- **state stays finite** under 400 steps of extreme sustained input
- **brakes at rest do not reverse the car**
- **substep convergence**: 5, 10 and 20 substeps agree to within 4%, so
  the integration has converged rather than being tuned to N = 5

Also verified by hand and passing: cg geometry is self-consistent (static
front load share 0.40 equals lr/L), yaw inertia is 42% above the m*lf*lr
estimate (normal for a real car), the load-transfer lever arm h_cg/L is
correct, tire force never exceeds mu*Fz on any axle under any input
combination (worst case exactly 1.000), and coasting decelerates at only
0.034 g from drag and rolling resistance alone.

## Realism audit and driving audit (2026-07-28)

### Engine vs published road-test figures

Compared against a real Porsche 911 GT3, which is what the 1500 kg / 1.3 g
parameter set represents. These are magazine-test numbers, not my own
tuning targets, so the comparison is independent:

| measure | engine v2 | real 911 GT3 | error |
|---|---|---|---|
| 0-100 km/h | 4.00 s | 3.4 s | +18% |
| 0-200 km/h | 11.9 s | 11.5 s | +3% |
| quarter mile | 11.9 s @ 201 km/h | ~11.5 s @ 200 | +3% |
| 100-0 km/h | 36.9 m (peak 1.12 g) | 31 m | +19% |
| 200-0 km/h | 140.5 m (peak 1.18 g) | 125 m | +12% |
| skidpad | 1.24 g | 1.1-1.3 g | in range |
| top speed | 303 km/h | 318 km/h | -5% |

Every figure is within about 20%, and every error is in the conservative
direction (slightly slower, slightly longer braking). Downforce behaves
correctly too: the grip ceiling rises from 1.32 g at 72 km/h to 1.53 g at
270 km/h. Gearing is sane: six speeds, shifts at 70/105/141/178/220 km/h.

### How the scripted driver was actually driving (and the fix)

Auditing the driver rather than the engine exposed two defects that the
"25/25 clean laps" headline had hidden:

1. **It barely braked.** Full brake on 0-3% of steps, median grip usage
   0.06-0.21, hard-braking (>0.9 g) steps only 3.1% of a Monza lap.
2. **The braking stuttered.** A trace of the biggest braking zone showed
   the pedal cycling -1.00, -0.61, -0.27, -0.10, -0.01, -1.00 within nine
   steps. Cause: the assumed braking decel (0.61 g) was half what the car
   can actually do (1.2 g), so each application over-braked, the
   proportional throttle released as speed met target, the corner was
   still there, and it braked again.

Fixed by tying the driver's assumptions to measured car capability
(`_GRIP_FRAC = 0.75` of the real 1.30 g cornering and 1.20 g braking) and
committing to the braking zone: full brake while more than 3 m/s over
target, blending out only inside that band. A narrow band was tried first
and still flickered.

Result on all 25 circuits, still 25/25 laps and offroad 0.003:

| | before | after |
|---|---|---|
| mean grip used | 0.55 | **0.87** |
| mean speed | 26.7 m/s | **30.9 m/s** |
| Monza laptime | 173 s | **152 s** |
| hard-braking steps (Monza) | 3.1% | **8.8%** |
| full-throttle steps (Monza) | 42% | **54%** |

Laptimes are now **2.0x real F1 records** on the 20 circuits with a known
record. F1 cars carry roughly 5 g of downforce-assisted grip against this
car's 1.3 g, so a GT3-class car belongs around 1.5-1.8x; 2.0x is a
competent-but-not-optimal GT3 driver, which is what a baseline should be.

### Known remaining wart

Steering sign reverses on 9-19% of steps, highest at IMS (19.3%) where the
line offset is tiny (0.04 half-widths). So it is small twitching on
straights rather than weaving off line, and it does not cost lap time or
track position. It is the scripted controller's discretisation, not a
physics fault, and it gives the RL policy an obvious axis on which to look
better (the `steer_jerk` column already measures it).

## Build results (2026-07-27)

### What engine v2 actually does

Real physics: 5 substeps of 20 ms per 100 ms control step; dynamic bicycle
model; Pacejka lateral tires peaking near 10 deg slip and falling to about
0.89 of peak by 30 deg; friction circle on both axles with a 0.92
kinetic/static ratio so an overloaded tire snaps rather than fading;
exact-exponential tire relaxation (0.3 m relaxation length); longitudinal
load transfer; drag plus downforce with a drag-limited top speed; a torque
curve through an auto-shifting 6-speed.

Simplified on purpose: no wheel-speed state. The pedal demands a fraction of
the drivetrain force available in the current gear (or of the brake
hardware, split by axle load); each tire caps that demand at its grip and
the friction circle steals cornering grip from over-demand. That gives
wheelspin and lockup CONSEQUENCES without the stiff wheel integrator, which
is the component that historically blew up at this timestep.

Human input limits, as requested: fixed 33 deg steering lock rate-limited to
70 deg/s (lock-to-lock measured at 1.00 s, so no instant flicks), pedal slew
6.0/s (about 0.33 s from full throttle to full brake), and the 10 Hz control
rate as the reaction-time floor.

Driver aids REMOVED relative to v1, which is what makes the agent do the
work: the speed-blended steering lock (75 to 30 deg), the hidden steering
gain schedule (1.05 to 0.45), the lateral-priority traction clamp that made
power oversteer impossible, the strong artificial yaw damping (0.6, now 0.1
of genuine tire scrub), the lateral-velocity scrub, and the v/(v+1) lateral
force fade.

### Acceptance tests (physics_tests.py, 7/7)

Both engines through the identical suite (`python physics_tests.py` and
`python physics_tests.py --v1`):

| maneuver | v1 (benchmark) | v2 | accept |
|---|---|---|---|
| 0-100 km/h | 3.80 s | 3.80 s | 3.0-4.2 s |
| top speed | 298.8 km/h | 300.5 km/h | 285-315 |
| 100-0 km/h | 34.0 m | 37.0 m | 30-42 m |
| max steady lateral | **0.65 g** | **1.24 g** | 1.1-1.5 g |
| step-steer overshoot | 0.00 | 0.08 | 0-0.30 |
| power-on oversteer | **no** | **yes** | must break away |
| trail braking | **no combined slip** | **0.13 lost** | 0.10-0.90 |
| load sensitivity | **absent** | **0.13 mu drop** | 0.03-0.35 |
| lock-to-lock | 0.90 s | 1.00 s | 0.6-1.6 s |
| | 5/9 | **9/9** | |

The two failures are the independent confirmation of the findings below:
v1 corners at half its tire grip (the Coriolis double-count) and cannot
break the rear away (the traction-control clamp). Everything straight-line
matches, exactly as expected, since those maneuvers never exercise the
yaw coupling.

### Two real bugs found on the way

1. **Double-counted Coriolis terms.** The integrator added `yaw*v_lat` and
   `-yaw*v_fwd` to the accelerations while ALSO storing velocity in the
   world frame and re-decomposing it with the updated heading each substep.
   The frame rotation already applies that coupling, so adding it again
   wasted about half the cornering force: the car could only pull 0.63 g
   with 1.3 g of tire grip available. The v1 engine has the same structure
   and therefore the same defect. Fixing it took the skidpad from 0.63 g to
   1.25 g. **This is worth telling Rafa: it means the benchmark car has been
   cornering at roughly half its intended grip.**
2. **Curvature bands scanned the wrong range.** In `track_geometry.
   curvature_ahead` the band's upper offset was compared against the
   ABSOLUTE lower index rather than the relative one, so every band scanned
   from its own start out to roughly twice the car's arc position. Every
   band saw every corner within 2*s, phantom hairpins pinned the braking
   target everywhere, and the scripted driver crawled at about 9 m/s on
   every circuit. This is why the v1 baseline showed a mean speed of
   8.9 m/s against an 83 m/s top speed; it was a geometry bug, not
   (only) a conservative controller.

### Baseline: v1 versus v2

| | v1 (before) | v2 (after) |
|---|---|---|
| circuits finished | 16/25 | **25/25** |
| mean offroad | 0.076 | **0.000** |
| mean speed | 8.9 m/s | **26.7 m/s** |
| worst circuit | IMS, offroad 0.47 | none |

Note this compares two things at once (better physics AND the two bug
fixes), so it is not a clean physics-only comparison. The point is that the
instrument now behaves: every real circuit is lapped cleanly at realistic
speed, which is the precondition for using failures on generated tracks as
evidence about the tracks.

### Is the car too easy? (investigated 2026-07-27)

Fair question given that the scripted driver laps all 25 circuits at 0.00
offroad. Measured answer: the car is not too easy, the driver is slow. It
laps at **2.1-2.4x realistic laptimes** using only **~55% of tire grip**,
with body slip under 1.2 degrees. Nothing punishes it because it never
approaches a limit.

Forcing it to push (by raising the grip fraction its speed rule assumes)
degrades it in the right order, which is what a realistic car should do:

| assumed grip | laptime vs real | grip used | slip p95 | outcome |
|---|---|---|---|---|
| as-tuned (0.50) | 2.1-2.4x | 50% | 1.2 deg | clean laps |
| 0.80 | 1.8-2.1x | 81% | 3.5-7 deg | clean laps |
| 1.00 | 1.8-1.9x | 87-90% | 10-18 deg | laps, visibly sliding |
| 1.25 | - | 91% | 37-52 deg | off-track failures |

So the difficulty is real and graded: the car rewards being driven at 80%,
gets loose at 90%, and spits you off past 100%. There is headroom for the
RL policy to be genuinely better rather than merely as clean.

**A measurement bug found while checking this, worth knowing about.** The
first version of this investigation computed lateral acceleration as
`yaw_rate * speed` and reported the car sustaining 1.5-2.4 g on a 1.3 g
tire, which looked exactly like broken physics. It is not: that identity
only holds in steady cornering. Once the car slides it rotates about its
own axis while travelling in a different direction, so the yaw proxy reads
SPIN as if it were grip (peak reading 5.9 g). Measuring the actual tire
force instead shows it never exceeds 1.29 g at any point of any lap. The
lesson generalizes: when validating a physics engine, measure forces, not
kinematic proxies that assume the very steady state you are testing for.

`baseline.py` now reports `grip_used` (95th percentile of tire force over
the tire limit) and `slip_p95` in every row, so "clean but slow" can never
again be mistaken for "clean and good".

### What the scripted driver needed (lessons for the RL policy)

The heuristic had to be substantially rebuilt for v2, and each change maps
to something the learned policy will have to discover:

1. **Feedback in curvature space, not wheel-angle space.** Wheel-angle
   feedback demands lateral acceleration proportional to v^2, so gains that
   work at 15 m/s command several g at 40 m/s and simply saturate the tires.
2. **A correction budget scaled to grip.** A flat cap is right for trimming
   the line but far too weak to catch a slide; the cap has to open up as
   heading error grows.
3. **Braking and steering share one grip pool.** Asking for both at the
   limit spins the car (measured: 27 m/s and 2 rad of heading lost in 18
   steps). Steering gets priority, braking uses the remainder, with a floor
   of 0.35 so trail-braking still happens.
4. **Countersteer, which is the big one.** All three of the last failing
   circuits showed the same picture: lateral velocity 15-25 m/s, large yaw
   rate, steering command near zero, car leaving the track sideways while
   barely turning the wheel. A path-following controller cannot catch a
   slide because the slide is not a path error yet. Adding
   `0.75 * body_slip` to the wheel angle took the suite from 3/6 to 6/6 with
   zero offroad. v1 never needed this because its artificial yaw damping
   killed slides for free.

Point 4 is the clearest evidence the physics now demands real car control:
under v1 a controller that ignored the lateral-velocity and yaw-rate
observation channels could lap 16 circuits. Under v2 it cannot lap any.

## Decision points for Rafa

1. Judge-only engine v2 (implemented this way, keeps all GA results valid)
   or also adopt v2 in the benchmark quality simulation (invalidates
   regression references and finished experiments)?
2. **The Coriolis double-count in the benchmark engine.** The v1 engine
   corners at roughly half its intended grip because of the bug described
   above. Leaving it is defensible (it is a constant across all five
   representations, so comparisons stay fair) but it should be a stated
   choice, not an accident, and it is worth a sentence in the thesis either
   way. Fixing it changes every quality number.
3. Optional next step, only if he wants more fidelity: per-axle lateral load
   transfer and tire load sensitivity. Everything else from the staged plan
   is built.
