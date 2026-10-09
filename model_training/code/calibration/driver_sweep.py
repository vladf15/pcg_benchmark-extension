"""SteeringAgent sweep: one line per variant over the 24 circuits and N random
genomes per representation (seed 21).

    python driver_sweep.py "label|attr=value,attr=value" ...

Reported per variant: laps finished, tracks with any off-road step, mean
off-road fraction, the car's largest distance from the centreline (median and
max over tracks; the road edge is 8 m), mean lap speed on the circuits and on
the generated tracks, and seconds per evaluation.
"""
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
REPO = _os.path.normpath(_os.path.join(HERE, '..', '..', '..'))
import gc, os, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
os.environ["PCG_BENCHMARK_WORKERS"] = "1"
ROOT = REPO
sys.path.insert(0, ROOT); sys.path.insert(0, _os.path.join(ROOT, "model_training", "code"))
import track_loader
from pcg_benchmark.probs.racing import agent as agent_mod

REPS = (("racing", "RacingProblem"), ("racingtile", "RacingTileProblem"),
        ("racingtilehex", "RacingTileHexProblem"), ("racingvoronoi", "RacingVoronoiProblem"))
BASE_INIT = agent_mod.SteeringAgent.__init__
DERIVED = ("ay_max", "ax_max_brake_tires", "ax_max_accel_tires", "ax_max_drive", "ax_max_brake")


def configure(kv):
    def init(self, curve_points, track_width=12.0, max_speed=None, engine=None):
        BASE_INIT(self, [], track_width, max_speed, engine)
        for k, v in kv.items():
            if k == "grip_utilisation":
                f = float(v) / self.grip_utilisation
                for a in DERIVED:
                    setattr(self, a, getattr(self, a) * f)
            if k == "steer_deadzone_deg":
                self.steer_deadzone_rad = np.deg2rad(float(v))
                continue
            setattr(self, k, float(v))
        self.curve_points = curve_points
    agent_mod.SteeringAgent.__init__ = init


def drive(p, content):
    t0 = time.perf_counter()
    info = p.info(content, use_cache=False)
    dt = time.perf_counter() - t0
    tr = p.evaluate(content)
    ag = p._agent
    idx, worst = 0, 0.0
    for s in tr:
        j, _ = ag._find_projection(s[:2], idx); idx = j
        worst = max(worst, abs(ag._signed_lateral_offset(s[:2], j)))
    v = np.array([s[3] for s in tr])
    return bool(info["finished"]), float(info["offroad_frac"]), worst, float(v.mean()), dt


def main():
    n_gen = int(os.environ.get("N_GEN", "20"))
    from pcg_benchmark.probs.racing.problem import RacingProblem
    for arg in sys.argv[1:]:
        label, _, over = arg.partition("|")
        configure(dict(x.split("=") for x in over.split(",") if x))
        p = RacingProblem(width=2600.0, height=2600.0, driver=os.environ.get("DRIVER", "steering"))
        circ = [drive(p, {"track_points": np.asarray(track_loader.load_track(n, resample_spacing=5.0)["points"], float)})
                for n in track_loader.list_tracks() if n != "IMS"]
        del p; gc.collect()
        gen = []
        for mod, cls in REPS:
            C = getattr(__import__("pcg_benchmark.probs.%s.problem" % mod, fromlist=[cls]), cls)
            q = C(driver=os.environ.get("DRIVER", "steering")); sp = q._content_space; sp.seed(int(os.environ.get("SEED", "21")))
            gen += [drive(q, sp.sample()) for _ in range(n_gen)]
            del q; gc.collect()
        allr = circ + gen
        worst = np.array([r[2] for r in allr])
        print("%-34s laps %3d/%3d  tracks off-road %2d  off-road %.4f  |offset| p50 %4.1f p90 %4.1f max %5.1f m  speed circuits %4.1f generated %4.1f  %.2f s/eval"
              % (label, sum(r[0] for r in allr), len(allr), sum(r[1] > 0 for r in allr), np.mean([r[1] for r in allr]),
                 np.median(worst), np.percentile(worst, 90), worst.max(), np.mean([r[3] for r in circ]),
                 np.mean([r[3] for r in gen]), np.mean([r[4] for r in allr])))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
