"""Calibrate the typicality stage of the racing quality function.

Prints the _TYPICALITY dict for pcg_benchmark/probs/racing/problem.py and the
leave-one-out distance of every reference circuit.

Reference set: the TUMFTM circuits (IMS left out) that pass the validity
gates and the length rule, scored through the problem's own
_quality_terms on a 2600 m map so Spa fits.  Features: _TYPICALITY_FEATURES.
Each feature is standardised by the reference mean and standard deviation;
the distance is sqrt(z' (C + ridge I)^-1 z) with C the covariance of the
standardised features.  The threshold is the largest leave-one-out distance,
so every reference circuit scores full marks when held out.

    python memcap.py 1500 calibrate_typicality.py
"""
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)

import track_loader  # noqa: E402
from pcg_benchmark.probs.racing.problem import RacingProblem  # noqa: E402

# Ridge on the standardised covariance: with 21 circuits and 5 features the
# plain covariance is invertible but noisy.  0.1 is a modelling choice, the
# smaller of the two values (0.1, 0.5) tried on an earlier feature set, where
# both passed the same random tracks.
RIDGE = 0.1


def calibrate(verbose=False):
    """Fit the typicality constants on the reference circuits.  Returns
    (dict in _TYPICALITY's layout, circuit names, leave-one-out distances)."""
    p = RacingProblem(width=2600.0, height=2600.0)
    keys = p._TYPICALITY_FEATURES
    names, rows = [], []
    for name in track_loader.list_tracks():
        if name == "IMS":
            continue
        pts = np.asarray(track_loader.load_track(name, resample_spacing=5.0)["points"], dtype=float)
        track = p._normalize_track_points(pts)
        curve = p._make_curve(track)
        t = p._quality_terms({"track_points": track, "curve_points": curve, "finished": True})
        valid = t["oob_score"] >= 0.999 and t["self_overlaps"] == 0 and t["length_score"] >= 0.999
        if verbose:
            print("%-14s %s  %s" % (name, "reference" if valid else "left out ",
                                    "  ".join("%s %.3f" % (k, t[k]) for k in keys)))
        if valid:
            names.append(name)
            rows.append([float(t[k]) for k in keys])
    X = np.array(rows)

    def fit(A):
        mu, sd = A.mean(0), A.std(0)
        Z = (A - mu) / sd
        return mu, sd, np.linalg.inv(np.cov(Z.T) + RIDGE * np.eye(A.shape[1]))

    def dist(x, f):
        mu, sd, ci = f
        z = (x - mu) / sd
        return float(np.sqrt(z @ ci @ z))

    loo = np.array([dist(X[i], fit(np.delete(X, i, 0))) for i in range(len(X))])
    mu, sd, ci = fit(X)
    return {"mean": mu, "sd": sd, "inv_cov": ci, "threshold": float(loo.max())}, names, loo


def main():
    T, names, loo = calibrate(verbose=True)
    print("\n%d reference circuits; leave-one-out distances, largest first:" % len(names))
    for i in np.argsort(loo)[::-1]:
        print("  %-14s %.2f" % (names[i], loo[i]))
    fmt = lambda v: "(" + ", ".join("%.6g" % x for x in v) + ")"
    print("\n    _TYPICALITY = {")
    print('        "mean": %s,' % fmt(T["mean"]))
    print('        "sd": %s,' % fmt(T["sd"]))
    print('        "inv_cov": (%s),' % ",\n                    ".join(fmt(r) for r in T["inv_cov"]))
    print('        "threshold": %.4g,' % T["threshold"])
    print("    }")


if __name__ == "__main__":
    main()
