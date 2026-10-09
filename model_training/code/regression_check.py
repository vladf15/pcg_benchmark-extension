"""Regression reference for the racing problems.

Scores a fixed sample and compares it with the stored reference, so a code
change that moves any quality, controllability or diversity score is seen.

    python memcap.py 2500 regression_check.py record   (write regression_reference.json)
    python memcap.py 2500 regression_check.py          (compare with it)

Sample, per problem (racing, racingtile, racingtilehex, racingtilediag, racingtilehexdiag,
racingvoronoi, racingradial): 6 random genomes (content and control space
seeded with 21) through the benchmark's own env.evaluate, with the default
driver; and 20 more through the geometry terms alone (_quality_terms without
a lap).  Plus the 24 reference circuits' quality with the default driver.
Scores match when they agree to 1e-9.  Timing is reported, not compared.
"""
import gc
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
os.environ["PCG_BENCHMARK_WORKERS"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)

import pcg_benchmark  # noqa: E402
import track_loader  # noqa: E402
from pcg_benchmark.probs.racing.problem import RacingProblem  # noqa: E402

REFERENCE = os.path.join(HERE, "regression_reference.json")
PROBLEMS = ("racing-v0", "racingtile-v0", "racingtilehex-v0", "racingtilediag-v0", "racingtilehexdiag-v0",
            "racingvoronoi-v0", "racingradial-v0")


def scores():
    out = {}
    for name in PROBLEMS:
        env = pcg_benchmark.make(name)
        env.seed(21)
        p = env._problem
        content = [env.content_space.sample() for _ in range(26)]
        control = [env.control_space.sample() for _ in range(6)]
        t0 = time.perf_counter()
        _, _, _, det, _ = env.evaluate(content[:6], control)
        dt = (time.perf_counter() - t0) / 6
        geo = []
        for c in content[6:]:
            raw = p._normalize_track_points(p._extract_content(c))
            t = p._quality_terms({"track_points": raw, "curve_points": p._make_curve(raw)})
            geo.append({k: float(v) for k, v in t.items() if np.isscalar(v) and np.isfinite(v)})
        out[name] = {"quality": [float(x) for x in det["quality"]],
                     "controlability": [float(x) for x in det["controlability"]],
                     "diversity": [float(x) for x in det["diversity"]],
                     "geometry": geo, "s_per_eval": dt}
        print("%-20s quality %s  %.2f s/eval" % (name, np.round(det["quality"], 4), dt), flush=True)
        del env, p
        gc.collect()
    p = RacingProblem(width=2600.0, height=2600.0)
    circ = {}
    for n in track_loader.list_tracks():
        if n == "IMS":
            continue
        pts = np.asarray(track_loader.load_track(n, resample_spacing=5.0)["points"], dtype=float)
        circ[n] = float(p.quality(p.info({"track_points": pts})))
    out["circuits"] = circ
    print("circuits feasible %d of %d" % (sum(q >= 1.0 for q in circ.values()), len(circ)))
    return out


def compare(now, ref):
    diffs = []
    for name in PROBLEMS:
        for k in ("quality", "controlability", "diversity"):
            for i, (a, b) in enumerate(zip(now[name][k], ref[name][k])):
                if abs(a - b) > 1e-9:
                    diffs.append("%s %s[%d]: %.6f -> %.6f" % (name, k, i, b, a))
        for i, (a, b) in enumerate(zip(now[name]["geometry"], ref[name]["geometry"])):
            for k in b:
                if k in a and abs(a[k] - b[k]) > 1e-9:
                    diffs.append("%s geometry[%d] %s: %.6f -> %.6f" % (name, i, k, b[k], a[k]))
        print("%-20s s/eval reference %.2f now %.2f" % (name, ref[name]["s_per_eval"], now[name]["s_per_eval"]))
    for n, b in ref["circuits"].items():
        if abs(now["circuits"].get(n, -1.0) - b) > 1e-9:
            diffs.append("circuit %s: %.6f -> %.6f" % (n, b, now["circuits"].get(n, -1.0)))
    return diffs


def main():
    now = scores()
    if len(sys.argv) > 1 and sys.argv[1] == "record":
        with open(REFERENCE, "w") as f:
            json.dump(now, f, indent=1)
        print("recorded", REFERENCE)
        return
    with open(REFERENCE) as f:
        ref = json.load(f)
    diffs = compare(now, ref)
    print("\n".join(diffs[:50]))
    print("MATCH" if not diffs else "%d DIFFERENCES" % len(diffs))
    sys.exit(1 if diffs else 0)


if __name__ == "__main__":
    main()
