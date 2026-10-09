"""The reference circuits driven by TORCS's berniw robot (racing/torcs.py).

Sets what the driven-lap stage asks of a TORCS lap (problem.py,
_TORCS_LAP): every circuit of the typicality reference set should pass it,
as every circuit passes the benchmark's own off-road limit.  Prints, per
circuit: laps finished, best lap, penalty time (corner cutting), damage, and
the mean speed in TORCS and with the benchmark's default driver.

    python ../memcap.py 3000 torcs_circuits.py
"""
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "..")))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))

import track_loader  # noqa: E402
from pcg_benchmark.probs.racing import torcs  # noqa: E402
from pcg_benchmark.probs.racing.problem import RacingProblem  # noqa: E402


def main():
    p = RacingProblem(width=2600.0, height=2600.0)
    rows = []
    for name in track_loader.list_tracks():
        if name == "IMS":
            continue
        pts = np.asarray(track_loader.load_track(name, resample_spacing=5.0)["points"], dtype=float)
        track = p._normalize_track_points(pts)
        curve = p._make_curve(track)
        t = p._quality_terms({"track_points": track, "curve_points": curve, "finished": True})
        ref = t["oob_score"] >= 0.999 and t["self_overlaps"] == 0 and t["length_score"] >= 0.999
        info = p.info({"track_points": track})
        r = torcs.run(curve, width=p._track_width)
        L = r["length_m"]
        rows.append((name, ref, r))
        print("%-14s %-9s laps %d  best %6.1f s  penalty %5.1f s  damage %5.0f  speed torcs %4.1f / "
              "benchmark %4.1f m/s  off-road %.4f  %.2f s wall" % (
                  name, "reference" if ref else "left out", r["laps"], r["best_lap"], r["penalty"], r["damage"],
                  L / r["best_lap"] if r["finished"] else float("nan"),
                  info["total_length"] / (info["steps"] * 0.1) if info["finished"] else float("nan"),
                  info["offroad_frac"], r["wall_s"]))
    ref = [r for _, ok, r in rows if ok]
    print("\nreference circuits: %d; finished %d; largest penalty %.1f s; largest damage %.0f"
          % (len(ref), sum(r["finished"] for r in ref), max(r["penalty"] for r in ref),
             max(r["damage"] for r in ref)))


if __name__ == "__main__":
    main()
