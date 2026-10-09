"""physics_tests' power-on oversteer case at several cg heights: per case the
settled body slip, the largest slip under full throttle and the threshold."""
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
REPO = _os.path.normpath(_os.path.join(HERE, '..', '..', '..'))
import sys
sys.path.insert(0, _os.path.join(REPO, "model_training", "code"))
import physics_tests as T
for h in (0.47, 0.48, 0.49, 0.50, 0.52):
    row = []
    for speed, steer in ((14.0, 0.25), (14.0, 0.32), (14.0, 0.40), (20.0, 0.12), (20.0, 0.16), (20.0, 0.20)):
        eng = T._make(); eng.h_cg = h
        if not T._accelerate_to(eng, speed):
            row.append("noacc"); continue
        for _ in range(40):
            thr = min(max(0.3 * (speed - T._speed(eng)), -1.0), 1.0)
            eng.step({"steering": steer, "throttle": thr})
        base = abs(T._beta(eng))
        if base > 25.0:
            row.append("slid"); continue
        mx = base
        for _ in range(40):
            eng.step({"steering": steer, "throttle": 1.0}); mx = max(mx, abs(T._beta(eng)))
        row.append("%.1f->%.1f/%.1f" % (base, mx, max(2 * base, base + 6)))
    print("h_cg %.2f  " % h + "  ".join(row))
