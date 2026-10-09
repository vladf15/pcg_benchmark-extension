"""Corners one tile builds between two 300 m straights, on the FIA line, with
the pipeline's uniform arc-length resampling (open path)."""
import numpy as np


def resample_open(pts, step):
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.r_[0.0, np.cumsum(seg)]
    t = np.linspace(0.0, arc[-1], max(3, int(round(arc[-1] / step))) + 1)
    return np.column_stack([np.interp(t, arc, pts[:, 0]), np.interp(t, arc, pts[:, 1])])


def tile_corners(p, pts, detail=False):
    pts = np.asarray(pts, float)
    u0 = pts[1] - pts[0]; u0 /= np.linalg.norm(u0); u1 = pts[-1] - pts[-2]; u1 /= np.linalg.norm(u1)
    path = np.vstack([pts[0] - 300 * u0, pts, pts[-1] + 300 * u1])
    c = resample_open(path, p._curve_step())
    ang, w = p._curve_turn_profile(c)
    t, a = p._find_corners(ang, w, p._QUALITY_PARAMS["fia_corner_curv_deg_per_m"])
    out = [(round(float(np.rad2deg(x)), 1), round(float(y / x), 1)) for x, y in zip(t, a)]
    if detail:
        return out, float(np.rad2deg(np.sum(ang)))
    return out
