"""Porsche 2020 performance figures against driveline efficiency, drag area
and launch rpm, on the published gears (8 overall ratios) and 1534 kg.
Reports 0-60 mph, 0-100 km/h, quarter mile and top speed.
    python memcap.py 1500 porsche_fit.py"""
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
REPO = _os.path.normpath(_os.path.join(HERE, '..', '..', '..'))
import math, sys
sys.path.insert(0, REPO)
from pcg_benchmark.probs.racing.engine import CarPhysicsEngine
REAL = (15.25, 9.89, 6.71, 4.87, 3.68, 2.93, 2.37, 1.90)
def run(eff, cd_a, launch=3000.0, up=7200.0, mass=1534.0):
    e = CarPhysicsEngine(start_position=(0.0, 0.0))
    e.mass = mass; e.inertia_z = mass * e.lf * e.lr
    e.normal_load = mass * e.g
    e.load_front_static = e.normal_load * e.lr / e.length
    e.load_rear_static = e.normal_load * e.lf / e.length
    e.gear_ratios = REAL; e.cd_a = cd_a; e.launch_rpm = launch; e.shift_up_rpm = up
    e.driveline_eff = eff
    t = x = 0.0; t60 = t100 = tq = None; pv = 0.0
    while t < 300.0:
        e.step({"steering": 0.0, "throttle": 1.0}); t += e.time_step
        v = math.hypot(*e.velocity); dx = 0.5 * (v + pv) * e.time_step; x += dx
        lerp = lambda a, b, y: t - e.time_step * (b - y) / max(b - a, 1e-9)
        if t60 is None and v >= 26.82: t60 = lerp(pv, v, 26.82)
        if t100 is None and v >= 27.78: t100 = lerp(pv, v, 27.78)
        if tq is None and x >= 402.3: tq = lerp(x - dx, x, 402.3)
        if t > 50 and v - pv < 0.00002: break
        pv = v
    return t60, t100, tq, v * 3.6, e.gear + 1
for args in sys.argv[1:]:
    eff, cd, la, up = map(float, args.split(","))
    print("eff %.2f cdA %.3f launch %4.0f up %4.0f: 0-60 %.2f  0-100 %.2f  1/4 %.2f  top %.1f km/h (gear %d)"
          % ((eff, cd, la, up) + run(eff, cd, la, up)))
