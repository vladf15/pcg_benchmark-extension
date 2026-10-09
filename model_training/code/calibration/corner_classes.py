"""Corner classes of a curve, as corner_census.py counts them."""
import numpy as np
CLASSES = ("hairpin", "slow", "medium", "fast", "sweeper")
CENSUS = np.array([0.13, 0.29, 0.24, 0.26, 0.08])   # 24 reference circuits, 360 corners; fast excludes hairpins

def corners(p, curve):
    P = p._QUALITY_PARAMS
    ang, w = p._curve_turn_profile(curve)
    turns, arcs = p._find_corners(ang, w, P["fia_corner_curv_deg_per_m"])
    return np.rad2deg(np.array(turns)), np.array(arcs) / np.maximum(np.array(turns), 1e-9)

def shares(turn, rad):
    if len(turn) == 0:
        return np.zeros(5)
    h = turn >= 150
    m = [h, (rad < 44) & ~h, (rad >= 44) & (rad < 75) & ~h, (rad >= 75) & (rad < 150) & ~h, (rad >= 150) & ~h]
    return np.array([x.mean() for x in m])
