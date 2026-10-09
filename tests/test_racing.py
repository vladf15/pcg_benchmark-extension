"""Checks behind the claims the thesis makes about the racing problems.

    python model_training/code/memcap.py 3000 tests/test_racing.py
    (or: python -m unittest discover tests; set RACING_SLOW=0 to skip the
    physics and regression harnesses, which take several minutes)

What is checked, and which claim it backs:
  determinism     every decode and score is a pure function of the genome
                  (fresh instances agree), so a run is reproducible from its
                  seed
  gates           a road that leaves the map or overlaps itself scores 0
  rules           the FIA rule terms are at full marks exactly inside the
                  published limits
  calibration     the typicality constants in racing/problem.py are the ones
                  calibrate_typicality.py fits on the reference circuits now,
                  so a change to the pipeline cannot leave them stale
  circuits        which reference circuits reach quality 1.0 (geometry
                  stages; the driven lap is assumed)
  torcs           (only with TORCS installed at TORCS_DIR) the TORCS export
                  closes on itself, and the TORCS driver races a reference
                  circuit cleanly, as the clean-race rule requires of all 21
  harnesses       physics_tests.py passes and regression_check.py matches
"""
import os
import subprocess
import sys
import unittest
import warnings

import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("PCG_BENCHMARK_WORKERS", "1")
REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CODE = os.path.join(REPO, "model_training", "code")
sys.path.insert(0, REPO)
sys.path.insert(0, CODE)

import track_loader  # noqa: E402
from pcg_benchmark.probs.racing import torcs  # noqa: E402
from pcg_benchmark.probs.racing.problem import RacingProblem  # noqa: E402

PROBLEMS = (
    ("pcg_benchmark.probs.racing.problem", "RacingProblem"),
    ("pcg_benchmark.probs.racingtile.problem", "RacingTileProblem"),
    ("pcg_benchmark.probs.racingtilehex.problem", "RacingTileHexProblem"),
    ("pcg_benchmark.probs.racingtilediag.problem", "RacingTileDiagProblem"),
    ("pcg_benchmark.probs.racingtilehexdiag.problem", "RacingTileHexDiagProblem"),
    ("pcg_benchmark.probs.racingvoronoi.problem", "RacingVoronoiProblem"),
)
SLOW = os.environ.get("RACING_SLOW", "1") != "0"


def _problem(mod, cls, **kw):
    return getattr(__import__(mod, fromlist=[cls]), cls)(**kw)


def _geometry_info(p, pts):
    curve = np.asarray(p._make_curve(p._normalize_track_points(np.asarray(pts, float))))
    return {"track_points": pts, "curve_points": curve, "finished": True, "offroad_frac": 0.0}


def _rounded_rectangle(w, h, r=60.0, cx=1300.0, cy=1300.0, step=5.0):
    """A closed rounded rectangle, w x h metres, corner radius r."""
    pts = []
    for (x0, y0, a0) in ((cx + w / 2 - r, cy - h / 2 + r, -90), (cx + w / 2 - r, cy + h / 2 - r, 0),
                         (cx - w / 2 + r, cy + h / 2 - r, 90), (cx - w / 2 + r, cy - h / 2 + r, 180)):
        for a in np.deg2rad(np.arange(a0, a0 + 90, 5)):
            pts.append((x0 + r * np.cos(a), y0 + r * np.sin(a)))
    return np.array(pts)


class Determinism(unittest.TestCase):
    def test_fresh_instances_agree(self):
        for mod, cls in PROBLEMS:
            with self.subTest(problem=cls):
                a, b = _problem(mod, cls), _problem(mod, cls)
                a._content_space.seed(7)
                b._content_space.seed(7)
                for _ in range(3):
                    ca, cb = a._content_space.sample(), b._content_space.sample()
                    pa, pb = a._extract_content(ca), b._extract_content(cb)
                    np.testing.assert_array_equal(np.asarray(pa), np.asarray(pb))
                    self.assertEqual(a.quality(_geometry_info(a, pa)), b.quality(_geometry_info(b, pb)))


class Gates(unittest.TestCase):
    def setUp(self):
        self.p = RacingProblem(width=2600.0, height=2600.0)

    def test_self_overlap_scores_zero(self):
        t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
        eight = np.c_[1300 + 900 * np.sin(t), 1300 + 450 * np.sin(2 * t)]   # crosses itself
        info = {"track_points": eight, "curve_points": np.vstack([eight, eight[:1]]),
                "finished": True, "offroad_frac": 0.0}
        self.assertGreater(self.p._quality_terms(info)["self_overlaps"], 0)
        self.assertEqual(self.p.quality(info), 0.0)

    def test_off_map_scores_zero(self):
        rect = _rounded_rectangle(1500, 800) + np.array([1000.0, 0.0])      # past the right edge
        self.assertEqual(self.p.quality(_geometry_info(self.p, rect)), 0.0)


class Rules(unittest.TestCase):
    def setUp(self):
        self.p = RacingProblem(width=2600.0, height=2600.0)

    def test_inside_limits(self):
        t = self.p._quality_terms(_geometry_info(self.p, _rounded_rectangle(1800, 900)))
        self.assertGreaterEqual(t["length_m"], 3500.0)
        for k in self.p._RULE_TERMS:
            self.assertGreaterEqual(t[k], 0.999, k)

    def test_straight_over_two_km(self):
        t = self.p._quality_terms(_geometry_info(self.p, _rounded_rectangle(2400, 300)))
        self.assertGreater(t["longest_straight_m"], 2000.0)
        self.assertLess(t["straight_max_score"], 0.999)

    def test_short_lap(self):
        t = self.p._quality_terms(_geometry_info(self.p, _rounded_rectangle(900, 500)))
        self.assertLess(t["length_m"], 3500.0)
        self.assertLess(t["length_score"], 0.999)


class Calibration(unittest.TestCase):
    def test_typicality_constants_current(self):
        import calibrate_typicality
        fitted, names, loo = calibrate_typicality.calibrate()
        stored = RacingProblem._TYPICALITY
        for key in ("mean", "sd", "inv_cov"):
            np.testing.assert_allclose(np.asarray(stored[key]), np.asarray(fitted[key]), rtol=1e-4, atol=1e-6,
                                       err_msg=key)
        self.assertAlmostEqual(stored["threshold"], fitted["threshold"], places=2)
        self.assertEqual(len(names), 22)


class Circuits(unittest.TestCase):
    # Reference circuits that do not reach quality 1.0 on geometry, and why:
    # Suzuka crosses itself (its bridge), Norisring is 2.3 km.
    EXPECTED_FAIL = {"Norisring", "Suzuka"}

    def test_reference_circuits(self):
        p = RacingProblem(width=2600.0, height=2600.0)
        fail = set()
        for name in track_loader.list_tracks():
            if name == "IMS":
                continue
            pts = np.asarray(track_loader.load_track(name, resample_spacing=5.0)["points"], float)
            if p.quality(_geometry_info(p, pts)) < 0.999:
                fail.add(name)
        self.assertEqual(fail, self.EXPECTED_FAIL)


@unittest.skipUnless(torcs.available(), "no TORCS install at TORCS_DIR")
class Torcs(unittest.TestCase):
    def test_export_closes(self):
        p = RacingProblem(width=2600.0, height=2600.0)
        curve = p._make_curve(p._normalize_track_points(_rounded_rectangle(1800, 900)))
        c = np.asarray(curve)[:-1]
        segs = torcs.segments(c)
        pts, h = torcs.rebuild(segs, 0.5 * (c[0] + c[1]), np.arctan2(*(c[1] - c[0])[::-1]))
        self.assertLess(np.linalg.norm(pts[-1] - 0.5 * (c[0] + c[1])), 0.05)
        self.assertAlmostEqual(abs(h - np.arctan2(*(c[1] - c[0])[::-1])) / (2 * np.pi), 1.0, places=6)

    def test_reference_circuit_races_clean(self):
        # the clean-race rule of _torcs_summary holds on a reference circuit
        p = RacingProblem(width=2600.0, height=2600.0, driver="torcs")
        pts = np.asarray(track_loader.load_track("Spielberg", resample_spacing=5.0)["points"], float)
        info = p.info({"track_points": pts})
        self.assertTrue(info["finished"])
        self.assertEqual(info["offroad_frac"], 0.0)


@unittest.skipUnless(SLOW, "RACING_SLOW=0")
class Harnesses(unittest.TestCase):
    def run_script(self, script):
        return subprocess.run([sys.executable, os.path.join(CODE, "memcap.py"), "3000", script],
                              cwd=CODE, capture_output=True, text=True, stdin=subprocess.DEVNULL)

    def test_physics(self):
        r = self.run_script("physics_tests.py")
        self.assertEqual(r.returncode, 0, r.stdout[-2000:])
        self.assertIn("12/12 passed", r.stdout)

    def test_regression(self):
        r = self.run_script("regression_check.py")
        self.assertIn("MATCH", r.stdout, r.stdout[-2000:])


if __name__ == "__main__":
    unittest.main(verbosity=2)
