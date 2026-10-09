"""Racetrack generation with Genetic-WFC (Bailly and Levieux 2023, IEEE ToG
15(1):36-45).

The pipeline is the paper's: a chromosome of one boost zone per grid cell
(Sec. III-E(a)) drives a Simple Tiled WFC (Sec. III-B), the resulting level is
evaluated, and the placed modules are written back into the chromosome
(Sec. III-E(b), Alg. 1 l.20).  Selection, crossover, mutation and elitist
reinsertion come from the benchmark's shared GA, which follows Alg. 1.

DEVIATIONS FROM THE SOURCE, and why:

1.  Adjacency is derived, not learned from sample grids (Sec. III-B): a
    tile's compatibility is its connectivity (faces open or closed, and
    neighbours must agree), so _wfc_compat computes it exactly; there is no
    corpus of authored tile racetracks, and learning could only approximate it.
2.  Module frequencies are built in, not counted from samples (there are
    none): curve tiles come in the proportions of the 24 circuits' corner
    classes and share one weight; grass and the straight have swept weights
    of their own (table above _WFC_WEIGHTS), grass outweighing every road
    tile so a circuit is drawn through the grid instead of filling it.
3.  Cell choice is the fewest remaining possibilities, the classic WFC
    heuristic, not the paper's Eq. 1, which as printed sums to zero for any
    distribution (its intended form is ambiguous); the two agree when the
    modules are equiprobable.
4.  No greyblock second pass (Sec. III-D): the tiles ARE the road geometry.
5.  Grass is both air and border module (Sec. III-C): it already has the
    closed socket both need, so separate modules would cost WFC time for no
    expressiveness.  The outer ring is forced to grass (the paper's border
    trick), which also keeps the track inside the oob margin.
6.  Only the largest closed loop is kept (_keep_largest_component).  The
    paper sets the fitness of a map without exactly one spawn to -inf
    (Sec. V); a racetrack needs one circuit, WFC makes several loops, and -inf
    would leave the GA no gradient (spawns are easy to satisfy, loops not).
7.  Re-encoding is kept for the other half of the paper's reason (Sec.
    III-E(b)): weights here are static, so the decode does not drift, but only
    14% of road requests survive propagation on random genomes, so without it
    crossover would recombine unexpressed wishes.
8.  An all-grass genome is seeded with one straight at the centre: unlike a
    level, an all-grass grid is no track at all.
9.  11x11 rather than 15x15, sized for turn count: a median of 17.5 turns on
    random content against the ~15 of real circuits.

IMPOSED BY THE HARNESS:

10. Crossover is uniform, not the paper's one-point half-grid split (Sec.
    III-E(c)): the benchmark's generator calls contentSwap(a, b, 0.5), which
    scatters genes, so a child cannot inherit a whole corner sequence.  The
    shared generator is not changed; this is recorded as a departure.
"""
from __future__ import annotations

import numpy as np
from pcg_benchmark.probs.racing.problem import RacingProblem
from pcg_benchmark.spaces import ArraySpace, FloatSpace, IntegerSpace, DictionarySpace
from PIL import Image, ImageDraw


# 11x11: median 17.5 turns on random content (7-25 over 12, straight and
# quarter-arc tiles only) against ~15 on real circuits.  A cell is 136 m.
GRID_H = 11
GRID_W = 11

GRASS       = 0
STRAIGHT    = 1
CORNER      = 2
CHICANE     = 3
SHARP       = 4
HAIRPIN     = 5
KINK        = 6
KINK_MIRROR = 7
S_CHICANE   = 8
TIGHTEN     = 9
TIGHTEN_MIRROR = 10
DOUBLE_APEX = 11
ESS         = 12
SWEEP       = 13

N, E, S, W = 0, 1, 2, 3

_DIR_DELTA = {N: (-1, 0), E: (0, 1), S: (1, 0), W: (0, -1)}
_OPPOSITE  = {N: S, S: N, E: W, W: E}

# Tiles joining opposite faces, and adjacent faces.  WFC sees only faces, so a
# family behaves like the plain straight or quarter arc; members differ in the
# road drawn inside the cell.
_STRAIGHT_FAMILY = (STRAIGHT, CHICANE, S_CHICANE, ESS, SWEEP)
_CORNER_FAMILY   = (SHARP, TIGHTEN, TIGHTEN_MIRROR, CORNER, DOUBLE_APEX, KINK, KINK_MIRROR, HAIRPIN)
# The mirrored kinds are drawn as their base kind reflected (_tile_points).
_MIRROR_OF = {KINK_MIRROR: KINK, TIGHTEN_MIRROR: TIGHTEN}

# (tile_type, rotation) -> frozenset of open edge directions
_OPEN_EDGES = {
    **{(GRASS, r): frozenset() for r in range(4)},
    **{(t, r): frozenset({E, W}) if r % 2 == 0 else frozenset({N, S})
       for t in _STRAIGHT_FAMILY for r in range(4)},
    **{(t, r): frozenset({(N + r) % 4, (E + r) % 4})
       for t in _CORNER_FAMILY for r in range(4)},
}

# Road inside each tile, a turtle path in cell apothems (68.2 m square, 68.4 m
# hex), y north: straights from face (-1, 0) to (1, 0); corners turn `turn`
# degrees left from face (0, 1) to (sin turn, -cos turn).  Segments are
# ("L", length) or ("A", radius, signed degrees, left positive).  Every path
# starts and ends square to a face midpoint, so neighbours join without a
# kink.  racingtilehex builds its tiles from the same functions.  The shapes
# are modelling choices placed against the shared quality thresholds, not
# tuned on generated tracks.  Metres for the square cell:
#
#   CORNER   one arc tangent to both faces.  Hex: radius 1/tan(turn/2), the
#            arc starting on the faces (39.5 and 118.5 m).  Square: 0.9 (61 m)
#            with 0.1 (7 m) straights, a user decision a little sharper than
#            the 68 m face-to-face arc; still medium (44-75 m).
#   SHARP    the same turn, tighter, with longer straights (hex: tangent
#            0.6 of 1/tan(turn/2); square: the shape gene, set so the walker
#            reads it slow, under 44 m, see _SHAPE_RANGES).
#   TIGHTEN  45 degrees at 2 r then 45 at r, straights solved to end on the
#            exit face: tightening toward the exit (TIGHTEN_MIRROR opens).
#   DOUBLE_APEX  45 degrees at 0.3 (20 m), a straight (the gene, 14-27 m), 45
#            at 0.3 again; with transitions the walker counts one 88-90
#            degree corner.
#   CHICANE  left 45, right 90, left 45 at 0.4 (27 m): a 16 m sideways step
#            and back.  Between straights the walker counts three corners at
#            45 degrees, one at the gene's 30 and 37.5 (the transitions smooth
#            the outer swings under corner_min_turn_deg).
#   S_CHICANE  left 70, right 140, a 2 r tan 35 (24 m) crossing straight,
#            left 140, right 70, all at 0.25 (17 m): 22 m out each side and
#            back (a user request for a sharper S).  The straight returns the
#            road to the centre line; 70 keeps the 140s under the 150 degree
#            hairpin test.
#   HAIRPIN  (90 - turn/2) away at 0.45 (31 m), 180 back at 0.3 (20 m) behind
#            the cell centre, then the first turn again: passes the hairpin
#            test and sits at the circuits' 19.2 m median tightest corner.
#            Legs 0.6 (41 m) apart leave 25 m of grass; their length sets
#            how far the bulb reaches (_SQUARE_HAIRPIN_LEG, hex _HAIRPIN_LEG).
#   KINK     the turn plus a counter-swing c at a strong radius, then c back
#            at a second, straights solved to the exit face (driven the other
#            way: slight one way, strong the other).  Hex: c = 20 at
#            _KINK_STRONG and 0.6.  Square: c = the gene, 20-45, at 0.3 and
#            0.4 (20, 27 m), the largest radii where 45 still fits; the walker
#            counts one 105-135 degree corner, and from c = 30 a 23-44 degree
#            counter-swing.  KINK_MIRROR puts the swing on the other leg.
#   ESS, SWEEP  on a straight tile: swing s, 2 s back, s again, on radius
#            1 / (2 sin s) arcs filling the tile, a 7-13 m sideways step.  The
#            walker counts the middle arc: 30-39 degrees at 103-134 m for ESS
#            (s 16-21, fast), 23-26 at 153-172 m for SWEEP (s 12.5-14,
#            sweeper).  With no end straight the transitions vanish, so
#            curvature steps to under 0.6 deg/m at the face.  Why not one bend:
#            a straight tile must leave on its entry heading.
def _chicane_path(swing_deg, radius):
    """Chicane segments across a straight tile: left swing, right 2 swing,
    left swing, all at `radius` (apothems), with equal straights at both ends
    so the path spans the 2 apothems between the faces (the three arcs
    advance 4 r sin(swing))."""
    end = (2.0 - 4.0 * radius * np.sin(np.deg2rad(swing_deg))) / 2.0
    if end < 0.0:
        raise ValueError(f"chicane {swing_deg} deg at radius {radius} does not fit the tile")
    return [("L", end), ("A", radius, swing_deg), ("A", radius, -2.0 * swing_deg),
            ("A", radius, swing_deg), ("L", end)]


def _s_chicane_path(swing_deg, radius):
    """S chicane segments across a straight tile: left swing, right 2 swing,
    a straight across the centre line, left 2 swing, right swing, all at
    `radius` (apothems), with equal straights at both ends so the path spans
    the 2 apothems between the faces.  Each arc of radius r from heading a to
    heading b advances r |sin b - sin a| along the axis, so the four arcs
    advance 6 r sin(swing) and the crossing straight, at heading -swing, its
    length times cos(swing); the crossing is 2 r tan(swing / 2), which brings
    the road back to the centre line."""
    a = np.deg2rad(swing_deg)
    cross = 2.0 * radius * np.tan(0.5 * a)
    end = (2.0 - 6.0 * radius * np.sin(a) - cross * np.cos(a)) / 2.0
    if end < 0.0:
        raise ValueError(f"S chicane {swing_deg} deg at radius {radius} does not fit the tile")
    return [("L", end), ("A", radius, swing_deg), ("A", radius, -2.0 * swing_deg),
            ("L", cross), ("A", radius, 2.0 * swing_deg), ("A", radius, -swing_deg),
            ("L", end)]

_STRAIGHT_PATHS = {
    STRAIGHT:  [("L", 2.0)],
    CHICANE:   _chicane_path(45.0, 0.4),
    S_CHICANE: _s_chicane_path(70.0, 0.25),
}
# Puts the centre of the 180 degree arc 0.5 (34 m) from the cell centre along
# the diagonal; the road edge stays 5.7 m inside the cell.
_SQUARE_HAIRPIN_LEG = np.sqrt(2.0) * 0.5 + 0.3 - 0.45 * np.tan(np.deg2rad(22.5))
# Square CORNER radius, see the table above.
_SQUARE_CORNER_RADIUS = 0.9

# Shape gene (square tiles): a second gene in [0, 1] per cell sets one shape
# parameter of its tile, linearly over the range below (faces, and so WFC,
# unaffected).  Without it, on 80 random genomes, 252 hairpins took 19
# distinct shapes against 43 of 47 on the circuits.  Each range keeps the
# tile in one census class (_TYPE_WEIGHTS), measured alone between two
# straights as (turn, mean radius) at the range ends:
#   SHARP        tangent length 0.25-0.40 apothems (17-27 m): 88-90 degrees
#                at 32-42 m, slow.
#   TIGHTEN      inner radius 0.20-0.25 (14-17 m, outer twice that): 90
#                degrees at 38-42 m, slow.
#   CORNER       tangent length 0.5-0.9 (34-61 m): 87-89 degrees at 49-67 m,
#                medium.
#   DOUBLE_APEX  straight between the apexes 0.2-0.4 (14-27 m): 88-90 degrees
#                at 48-58 m, medium.
#   KINK         counter-swing 20-45 degrees: 110-134 degrees at 26-34 m,
#                slow.
#   HAIRPIN      leg length 0.5-1.0 of _SQUARE_HAIRPIN_LEG: how far the bulb
#                reaches toward the cell corner; 179-183 degrees at 29-33 m.
#   CHICANE      swing 30-45 degrees at radius 0.4 (27 m).
#   S_CHICANE    swing 35-70 degrees at radius 0.25 (17 m).
#   ESS          swing 16-21 degrees: 30-39 degrees at 103-134 m, fast.
#   SWEEP        swing 12.5-14 degrees: 23-26 degrees at 153-172 m, sweeper.
# Quantised to 1/_SHAPE_LEVELS so the shape cache stays bounded; content
# without the gene decodes at 0.5.  calibration/proto_sq.py.
_SHAPE_LEVELS = 20
_SHAPE_RANGES = {
    SHARP: (0.25, 0.40), TIGHTEN: (0.20, 0.25), TIGHTEN_MIRROR: (0.20, 0.25),
    CORNER: (0.5, 0.9), DOUBLE_APEX: (0.2, 0.4), HAIRPIN: (0.5, 1.0),
    KINK: (20.0, 45.0), KINK_MIRROR: (20.0, 45.0),
    CHICANE: (30.0, 45.0), S_CHICANE: (35.0, 70.0), ESS: (16.0, 21.0), SWEEP: (12.5, 14.0),
}
# Square KINK: strong and second radius (apothems), see the shape table.
_SQUARE_KINK_RADII = (0.3, 0.4)
# Clothoid transition length (_walk_smooth), in metres, before the cap at
# twice a tile's shorter end straight.  Swept on 40 random genomes (seed 21)
# against the time the car spends at its grip limit in the
# quasi-steady-state lap (the share of lap time where the cornering limit
# sets the speed), median over the tracks:
#
#   transition               0 m      10       20       30       40
#   at the limit             0.090    0.085    0.078    0.076    0.070    circuits 0.052
#   distinct hairpin shapes  4 / 89   4 / 89   6 / 90   9 / 93   9 / 96
#   class L1 from the census 0.96     0.68     0.64     0.55     0.47
#
# The pure arcs of a tile hold the car at the limit for the whole arc, where a
# real corner reaches it at the apex, and at 0 m the slow class takes 0.77 of
# the corners, since without transitions the walker measures each tile's arc
# at its own radius.  40 m comes closest to the circuits;
# longer transitions are cut to the end straights of most tiles anyway (6.8 m
# either side of the widest corner allows 13.6 m).  Script:
# model_training/code/calibration/tile_sweep.py.
_TRANSITION_M = 40.0


def _exit_port(turn):
    """Exit face midpoint of a corner-family path turning `turn` degrees."""
    t = np.deg2rad(turn)
    return np.array([np.sin(t), -np.cos(t)])


def _walk_path(segments, start, heading_deg, n_arc):
    """Points along a turtle path, starting with `start`.  n_arc(radius, turn
    in radians) gives the samples per arc; a straight adds only its end."""
    pos = np.asarray(start, dtype=float)
    heading = np.deg2rad(heading_deg)
    pts = [pos]
    for seg in segments:
        if seg[0] == "L":
            if seg[1] > 1e-9:
                pos = pos + seg[1] * np.array([np.cos(heading), np.sin(heading)])
                pts.append(pos)
            continue
        radius, turn = seg[1], np.deg2rad(seg[2])
        centre = pos + np.sign(turn) * radius * np.array([-np.sin(heading), np.cos(heading)])
        a0 = np.arctan2(pos[1] - centre[1], pos[0] - centre[0])
        n = n_arc(radius, turn)
        pts.extend(centre + radius * np.array([np.cos(a0 + turn * s / n), np.sin(a0 + turn * s / n)])
                   for s in range(1, n + 1))
        pos = pts[-1]
        heading += turn
    return np.array(pts)


def _walk_smooth(segments, start, heading_deg, transition, step, end):
    """A turtle path with clothoid transitions: the curvature profile is
    averaged over `transition` (apothems), so it ramps linearly into and out
    of every arc (a clothoid) with the total turn kept; cut to twice the
    shorter end straight so the path still meets its faces straight.  The
    small miss this causes at `end` (0.3 m for a 90 degree corner at 20 m) is
    added back over the curved part with a smoothstep weight.  Points `step`
    apart, ending exactly on `end`."""
    lengths, turns = [], []
    for seg in segments:
        if seg[0] == "L":
            if seg[1] > 1e-12:
                lengths.append(float(seg[1])); turns.append(0.0)
        else:
            lengths.append(float(seg[1]) * abs(np.deg2rad(seg[2])))
            turns.append(np.deg2rad(seg[2]))
    edges = np.r_[0.0, np.cumsum(lengths)]
    total = edges[-1]
    ds = step / 8.0
    n = max(int(np.ceil(total / ds)), 2)
    ds = total / n
    # Exact turn per fine sample: differences of the cumulative turn, which
    # is piecewise linear in arc length.
    cum = np.interp(np.arange(n + 1) * ds, edges, np.r_[0.0, np.cumsum(turns)])
    k = np.diff(cum) / ds
    first = lengths[0] if turns[0] == 0.0 else 0.0
    last = lengths[-1] if turns[-1] == 0.0 else 0.0
    width = int(round(min(transition, 2.0 * first, 2.0 * last) / ds)) // 2 * 2 + 1
    if width >= 3:
        k = np.convolve(k, np.ones(width) / width, mode="same")
    heading = np.deg2rad(heading_deg) + np.r_[0.0, np.cumsum(k * ds)]
    mid = 0.5 * (heading[:-1] + heading[1:])
    xy = np.asarray(start, dtype=float) + np.vstack(
        [[0.0, 0.0], np.cumsum(np.column_stack([np.cos(mid), np.sin(mid)]) * ds, axis=0)])
    err = np.asarray(end, dtype=float) - xy[-1]
    curved = np.nonzero(np.abs(k) > 1e-12)[0]
    if len(curved):
        a, b = curved[0], curved[-1] + 1
        u = np.clip((np.arange(n + 1) - a) / max(b - a, 1), 0.0, 1.0)
    else:
        u = np.linspace(0.0, 1.0, n + 1)
    xy = xy + np.outer(u * u * (3.0 - 2.0 * u), err)
    keep = np.r_[np.arange(0, n, max(int(round(step / ds)), 1)), n]
    return xy[keep]


def _solve_straights(segs, start, heading_deg, end):
    """The two straight lengths (a, b) that put the end of the turtle path
    segs(a, b), walked from `start` at `heading_deg`, on `end`: the end point
    is affine in them, so three walks give the linear system.  Shared by
    every tile whose straights are solved (racingtilediag, racingtilehexdiag)."""
    def walk_end(a, b):
        return _walk_path(segs(a, b), start, heading_deg, lambda radius, t: 1)[-1]
    e0 = walk_end(0.0, 0.0)
    return np.linalg.solve(np.column_stack([walk_end(1.0, 0.0) - e0, walk_end(0.0, 1.0) - e0]),
                           np.asarray(end) - e0)


def _corner_path(kind, turn, hairpin_leg, kink_strong, corner_radius, sharp_radius=0.6,
                 kink_counter=20.0, kink_second=0.6, tighten_radius=0.25, double_gap=0.3):
    """Turtle path of a corner-family tile, as described in the table above.
    corner_radius is CORNER's tangent length in apothems: 1.0 starts the arc
    on the faces, less adds a straight either side.  sharp_radius is SHARP's
    tangent length the same way.  kink_counter (degrees) and kink_second
    (apothems) are KINK's counter-swing and the radius it is taken back at;
    tighten_radius is TIGHTEN's inner radius and double_gap DOUBLE_APEX's
    straight between the apexes."""
    half = np.deg2rad(turn / 2.0)
    if kind == CORNER:
        return [("L", 1.0 - corner_radius), ("A", corner_radius / np.tan(half), turn),
                ("L", 1.0 - corner_radius)]
    if kind == SHARP:
        return [("L", 1.0 - sharp_radius), ("A", sharp_radius / np.tan(half), turn),
                ("L", 1.0 - sharp_radius)]
    if kind == HAIRPIN:
        swing = 90.0 - turn / 2.0
        big, small = 0.45, 0.3
        # Entry straight that puts both legs `small` from the bisector.
        entry = 1.0 - (small + big * (1.0 - np.cos(np.deg2rad(swing)))) / np.sin(np.deg2rad(swing))
        return [("L", entry), ("A", big, -swing), ("L", hairpin_leg), ("A", small, 180.0),
                ("L", hairpin_leg), ("A", big, -swing), ("L", entry)]
    if kind == TIGHTEN:
        def segs(a, b):
            return [("L", a), ("A", 2.0 * tighten_radius, turn / 2.0),
                    ("A", tighten_radius, turn / 2.0), ("L", b)]
        return segs(*_solve_straights(segs, (0.0, 1.0), -90.0, _exit_port(turn)))
    if kind == DOUBLE_APEX:
        # Symmetric about the bisector, so the path ends on the exit face
        # when its middle does: the entry straight that puts the midpoint of
        # the gap on the bisector, found from two walks (it is affine in it).
        def half_way(a):
            q = _walk_path([("L", a), ("A", 0.3, turn / 2.0), ("L", double_gap / 2.0)],
                           (0.0, 1.0), -90.0, lambda radius, t: 1)[-1]
            return float(np.dot(q, _exit_port(turn) - np.array([0.0, 1.0])))
        f0, f1 = half_way(0.0), half_way(1.0)
        a = (0.5 * float(np.dot(_exit_port(turn) + np.array([0.0, 1.0]),
                                _exit_port(turn) - np.array([0.0, 1.0]))) - f0) / (f1 - f0)
        return [("L", a), ("A", 0.3, turn / 2.0), ("L", double_gap),
                ("A", 0.3, turn / 2.0), ("L", a)]

    def kink(lead, mid):
        return [("L", lead), ("A", kink_strong, turn + kink_counter), ("L", mid),
                ("A", kink_second, -kink_counter), ("L", 0.1)]

    return kink(*_solve_straights(kink, (0.0, 1.0), -90.0, _exit_port(turn)))


def _tile_points(kind, turn, n_arc, mirror=False, hairpin_leg=0.0, kink_strong=0.4,
                 corner_radius=1.0, s_chicane=None, sharp_radius=0.6, chicane=None,
                 transition=0.0, step=0.05, kink_counter=20.0, kink_second=0.6,
                 tighten_radius=0.25, double_gap=0.3, ess=None):
    """Centre line of one tile in the canonical frame above, from the entry
    face midpoint to the exit face midpoint.  `mirror` reflects a
    straight-family path across its axis, and a corner-family path across the
    bisector of its two faces, walked backwards so it still starts on the
    entry face.  `s_chicane` and `chicane`, (swing degrees, radius) pairs,
    replace the default chicane shapes, and `ess` (swing degrees) gives ESS
    and SWEEP their swing.  `transition` > 0 (apothems) adds
    clothoid transitions (_walk_smooth), with points `step` apart; at 0 the
    arcs join their straights directly, sampled by n_arc.  racingtilehex
    passes only the arguments it needs, so its tiles keep the defaults."""
    if kind in _STRAIGHT_FAMILY:
        if kind in (ESS, SWEEP):
            path = _chicane_path(ess, 0.5 / np.sin(np.deg2rad(ess)))
        elif kind == S_CHICANE and s_chicane is not None:
            path = _s_chicane_path(*s_chicane)
        elif kind == CHICANE and chicane is not None:
            path = _chicane_path(*chicane)
        else:
            path = _STRAIGHT_PATHS[kind]
        segs = [(seg[0], seg[1], -seg[2]) if mirror and seg[0] == "A" else seg
                for seg in path]
        if transition > 0.0 and kind != STRAIGHT:
            return _walk_smooth(segs, (-1.0, 0.0), 0.0, transition, step, (1.0, 0.0))
        pts = _walk_path(segs, (-1.0, 0.0), 0.0, n_arc)
        pts[-1] = (1.0, 0.0)
        return pts
    exit_port = _exit_port(turn)
    segs = _corner_path(kind, turn, hairpin_leg, kink_strong, corner_radius, sharp_radius,
                        kink_counter, kink_second, tighten_radius, double_gap)
    if transition > 0.0:
        pts = _walk_smooth(segs, (0.0, 1.0), -90.0, transition, step, exit_port)
    else:
        pts = _walk_path(segs, (0.0, 1.0), -90.0, n_arc)
    # Every path is built to end on the exit face midpoint; pin it there so
    # rounding cannot open a gap at the join with the next tile.
    pts[-1] = exit_port
    if mirror:
        b = np.array([0.0, 1.0]) + exit_port
        b = b / np.linalg.norm(b)
        pts = (pts @ (2.0 * np.outer(b, b) - np.eye(2)).T)[::-1].copy()
    return pts


# Every tile with a curve in it, in the order they sit in _WFC_TILES.
_CURVE_KINDS = _CORNER_FAMILY + (CHICANE, S_CHICANE, ESS, SWEEP)

# WFC weight per tile type, every rotation alike.  All curve tiles share one
# weight, so the vocabulary, not a weight per tile, sets the mix (census
# table above _WFC_WEIGHTS); grass and the straight have their own (sweep
# below it).
_CURVE_WEIGHT = 0.125
_TYPE_WEIGHTS = {GRASS: 3.0, STRAIGHT: 32.0, **{k: _CURVE_WEIGHT for k in _CURVE_KINDS}}

# frozenset of two open directions -> (tile_type, rotation)
_EDGES_TO_TILE = {
    frozenset({E, W}): (STRAIGHT, 0),
    frozenset({N, S}): (STRAIGHT, 1),
    frozenset({N, E}): (CORNER, 0),
    frozenset({E, S}): (CORNER, 1),
    frozenset({S, W}): (CORNER, 2),
    frozenset({W, N}): (CORNER, 3),
}


_N_WFC_TILES    = 3 + 4 * len(_CURVE_KINDS)    # len(RacingTileProblem._WFC_TILES), 51


class _TileGenomeSpace(DictionarySpace):
    """Genome = one boost zone per cell (Genetic-WFC, Bailly and Levieux 2023,
    Sec. III-E): each gene is the tile boosted when WFC collapses that cell
    (0 grass, else a road tile).  It steers without overriding: a tile
    propagation eliminated has probability zero, and boosting zero leaves
    zero, so no adjacency violation can arise.  One gene per cell, so
    contentSwap makes local changes.  On squares tile_shape holds one float
    per cell for its tile's shape (_SHAPE_RANGES); WFC never reads it and it
    is not re-encoded.  _reencode (Sec. III-E(b)) rewrites the genome to the
    built layout after info().  Decoding is pure (every individual gets a
    full WFC pass from the same seed)."""

    def __init__(self, problem_ref, n_tiles, shape_genes):
        spaces = {
            # IntegerSpace max is exclusive: tile indices 0..n_tiles-1.
            "tile_prefs": ArraySpace((GRID_H * GRID_W,), IntegerSpace(0, n_tiles)),
        }
        if shape_genes:
            # One shape gene per cell (see _SHAPE_RANGES).
            spaces["tile_shape"] = ArraySpace((GRID_H * GRID_W,), FloatSpace(0.0, 1.0))
        super().__init__(spaces)
        self._prob = problem_ref

    def seed(self, seed):
        """Seed the genome draw itself: GenericSpace.seed reaches only the
        nested spaces, which sample() (init_content) never reads, so without
        this two identical runs draw different populations."""
        super().seed(seed)
        self._random = np.random.default_rng(seed)

    def sample(self):
        return self._prob.init_content(self._random)


# ── Lattice-independent graph walks ──────────────────────────────────
# A neighbour graph maps each road cell (r, c) to the cells it is joined to
# through an open face (or port) both of them use; every tile problem builds
# one with _build_neighbor_graph and walks it with these.

def _connected_components(neighbors):
    """Group the cells of a neighbour graph into connected components."""
    visited, components = set(), []
    for seed in neighbors:
        if seed in visited:
            continue
        comp, stack = [], [seed]
        while stack:
            cell = stack.pop()
            if cell in visited:
                continue
            visited.add(cell)
            comp.append(cell)
            for nb in neighbors.get(cell, []):
                if nb not in visited:
                    stack.append(nb)
        components.append(comp)
    return components


def _largest_loop(neighbors):
    """The cells of the largest component that is a closed loop (every cell
    of degree exactly 2), or None when there is no loop.  Mutual connectivity
    rather than grid adjacency, so two loops that merely touch in the grid
    are separate candidates."""
    loops = [comp for comp in _connected_components(neighbors)
             if all(len(neighbors.get(cell, [])) == 2 for cell in comp)]
    return set(max(loops, key=len)) if loops else None


def _walk_loop(neighbors):
    """The closed loop through the degree-2 cells, in order, or None if
    there is none of at least 4 cells."""
    cycle_cells = {cell for cell, nbrs in neighbors.items() if len(nbrs) == 2}
    if not cycle_cells:
        return None
    # From each cell continue to the neighbour not just come from.
    start = next(iter(cycle_cells))
    loop, prev, cur = [start], None, start
    while True:
        nxt = None
        for nb in neighbors.get(cur, []):
            if nb != prev and nb in cycle_cells:
                nxt = nb
                break
        if nxt is None or nxt == start:
            break
        loop.append(nxt)
        prev, cur = cur, nxt
    if len(loop) < 4 or start not in neighbors.get(loop[-1], []):
        return None
    return loop


def _walk_chain(neighbors):
    """The largest connected chain of road cells, loop or not, in order from
    an endpoint (a cell of degree 1) if it has one; None under 2 cells."""
    components = _connected_components(neighbors)
    best = max(components, key=len) if components else []
    if len(best) < 2:
        return None
    best_set = set(best)
    start = best[0]
    for cell in best:
        in_chain = [nb for nb in neighbors.get(cell, []) if nb in best_set]
        if len(in_chain) == 1:
            start = cell
            break
    path, prev, cur = [start], None, start
    while True:
        nxt = None
        for nb in neighbors.get(cur, []):
            if nb != prev and nb in best_set:
                nxt = nb
                break
        if nxt is None or nxt == path[0]:
            break
        path.append(nxt)
        prev, cur = cur, nxt
    return path if len(path) >= 2 else None


def _rectangle_loop(rng):
    """The cells of a random rectangle of interior cells, clockwise from its
    top-left corner: the last-resort layout when every WFC attempt fails."""
    min_dim = 3
    r0 = int(rng.integers(1, GRID_H - min_dim - 1))
    c0 = int(rng.integers(1, GRID_W - min_dim - 1))
    r1 = int(rng.integers(r0 + min_dim, min(r0 + min_dim + 5, GRID_H - 1) + 1))
    c1 = int(rng.integers(c0 + min_dim, min(c0 + min_dim + 5, GRID_W - 1) + 1))
    loop = []
    for c in range(c0, c1):
        loop.append((r0, c))
    for r in range(r0, r1):
        loop.append((r, c1))
    for c in range(c1, c0, -1):
        loop.append((r1, c))
    for r in range(r1, r0, -1):
        loop.append((r, c0))
    return loop


class _GeneticWFCProblem(RacingProblem):
    """The Genetic-WFC pipeline the tile problems share (Bailly and Levieux
    2023): decode a boost-zone genome with WFC, keep the largest closed loop,
    trace its road, and write the placed tiles back into the genome.  The
    subclass supplies the lattice: its tiles and weights, neighbours,
    propagation, road geometry and drawing.

    Decoded tiles are what _decode_genome returns: a tuple of grids on
    squares (types, rotations) and one grid of tile ids on hexes.  Methods
    that take them, such as _extract_loop, take them as positional
    arguments (_tile_args).  The subclass hooks:
      _n_tiles()              size of the vocabulary;
      _tile_weights()         WFC weight of every tile id;
      _ids_to_tiles(ids)      a grid of tile ids to decoded tiles;
      _tile_ids(tiles)        decoded tiles to a grid of tile ids;
      _fallback_tiles(rng)    the last-resort loop;
      _content_track_points(content, tiles), _tile_background(...)."""

    # Decoded points already trace a valid loop in order; the base class's
    # 2-opt untangle must not reorder them (it would break the loop).
    _untangle_control_points = False
    _SHAPE_GENES = False   # whether the genome carries tile_shape
    _SEED_TILE = 1         # tile forced at the centre of an all-grass genome

    def __init__(self, **kwargs):
        kwargs.setdefault('num_points', 15)
        super().__init__(**kwargs)     # the shared 7000-step budget, as for every representation
        self._content_space = _TileGenomeSpace(self, self._n_tiles(), self._SHAPE_GENES)
        # genome bytes -> decoded tiles (keys in insertion order for eviction),
        # and -> raw WFC output before loop pruning (for _reencode).  The bound
        # is well above one GA population.
        self._decode_cache: dict = {}
        self._decode_cache_keys: list = []
        self._raw_wfc: dict = {}
        self._DECODE_CACHE_MAX = int(kwargs.get("decode_cache_max", 512))
        self._tile_polylines: dict = {}   # tile key -> (entry face, road points), _tile_polyline

    @staticmethod
    def _tile_args(tiles):
        """Decoded tiles as positional arguments: (types, rotations) on
        squares, (tile_ids,) on hexes."""
        return tiles if isinstance(tiles, tuple) else (tiles,)

    # ── WFC runner (Merrell's Model Synthesis) ────────────────────────

    def _run_wfc(self, wave, rng, compat, boosts=None):
        """Merrell's model-synthesis loop: observe the uncollapsed cell with
        the fewest possibilities (ties random), collapse it to its requested
        tile (`boosts`, the genome's (GRID_H, GRID_W) boost zones) while that
        is still legal, else a draw weighted by _tile_weights, then AC-3.
        Returns the decoded tiles, or None on contradiction."""
        weights = self._tile_weights()
        while True:
            min_e, candidates = float('inf'), []
            for r in range(GRID_H):
                for c in range(GRID_W):
                    n = len(wave[r][c])
                    if n > 1:
                        if n < min_e:
                            min_e, candidates = n, [(r, c)]
                        elif n == min_e:
                            candidates.append((r, c))
            if not candidates:
                break
            r, c = candidates[int(rng.integers(len(candidates)))]
            possible = list(wave[r][c])
            if boosts is not None and int(boosts[r, c]) in wave[r][c]:
                # The request (a ruled-out tile is absent from the wave, and
                # then the weights decide: nothing to roll back).
                chosen = int(boosts[r, c])
            else:
                w = weights[possible].astype(float)
                w = w / w.sum()  # normalize to probabilities
                chosen = possible[int(rng.choice(len(possible), p=w))]
            wave[r][c] = {chosen}
            if not self._wfc_propagate(wave, [(r, c)], compat):
                return None
        ids = np.zeros((GRID_H, GRID_W), dtype=int)
        for r in range(GRID_H):
            for c in range(GRID_W):
                ids[r, c] = next(iter(wave[r][c])) if wave[r][c] else GRASS
        return self._ids_to_tiles(ids)

    @staticmethod
    def _copy_wave(wave):
        """Deep-copy a wave: a new grid where every possibility set is a copy,
        so changes to the copy never leak back into the original."""
        return [[set(cell) for cell in row] for row in wave]

    def _decode_cache_store(self, prefs_arr, tiles, raw):
        """Cache a decoded genome (decoding is pure, so a hit equals a
        recompute), with `raw`, the WFC output before loop pruning, evicted
        together with it (_reencode needs both).  Oldest out first."""
        key = prefs_arr.tobytes()
        if key not in self._decode_cache:
            self._decode_cache_keys.append(prefs_arr.copy())
            while len(self._decode_cache_keys) > self._DECODE_CACHE_MAX:
                oldest = self._decode_cache_keys.pop(0)
                self._decode_cache.pop(oldest.tobytes(), None)
                self._raw_wfc.pop(oldest.tobytes(), None)
        self._decode_cache[key] = tiles
        self._raw_wfc[key] = np.asarray(raw).copy()

    def _decode_genome(self, tile_prefs):
        """Decode a boost-zone genome (flat, GRID_H*GRID_W tile ids) with one
        full WFC pass: Bailly and Levieux's (2023) Alg. 1 l. 18, `l <-
        generate(c)`.  Deterministic, as in the paper ("the same random
        generator seed every time"); _fallback_tiles if every attempt
        contradicts."""
        tile_prefs = np.asarray(tile_prefs, dtype=int)
        if tile_prefs.size != GRID_H * GRID_W:
            raise ValueError(
                "tile_prefs must have %d entries for this %dx%d grid, got %d"
                % (GRID_H * GRID_W, GRID_H, GRID_W, tile_prefs.size))
        key = tile_prefs.tobytes()
        if key in self._decode_cache:
            return self._decode_cache[key]

        compat   = self._wfc_compat()
        prefs_2d = tile_prefs.reshape(GRID_H, GRID_W)

        # Base wave with the grass border forced (level structure, not a
        # request); the genome enters only as boost zones in _run_wfc.
        n_tiles = self._n_tiles()
        base_wave = [[set(range(n_tiles)) for _ in range(GRID_W)]
                     for _ in range(GRID_H)]
        border = []
        for r in range(GRID_H):
            for c in range(GRID_W):
                if r == 0 or r == GRID_H - 1 or c == 0 or c == GRID_W - 1:
                    base_wave[r][c] = {GRASS}
                    border.append((r, c))
        self._wfc_propagate(base_wave, border, compat)

        # An all-grass genome still needs a road seed: a straight at the centre.
        if not np.any(prefs_2d[1:GRID_H - 1, 1:GRID_W - 1]):
            sr, sc = GRID_H // 2, GRID_W // 2
            if self._SEED_TILE in base_wave[sr][sc]:
                base_wave[sr][sc] = {self._SEED_TILE}
                self._wfc_propagate(base_wave, [(sr, sc)], compat)

        # A fixed seed sequence, NOT genome-derived: parent and child get the
        # same dice, so the genome is their only difference and the layout is
        # heritable; a genome seed would let one gene redraw the whole track.
        for attempt in range(100):
            rng  = np.random.default_rng(attempt * 1_000_003 + 7)
            wave = self._copy_wave(base_wave)
            result = self._run_wfc(wave, rng, compat, boosts=prefs_2d)
            if result is None:
                continue
            tiles = self._keep_largest_component(*self._tile_args(result))
            if self._extract_loop(*self._tile_args(tiles)) is not None:
                # With the RAW output: _reencode records what WFC placed.
                self._decode_cache_store(tile_prefs, tiles, raw=self._tile_ids(result))
                return tiles

        # Total failure: a deterministic last-resort loop.
        rng   = np.random.default_rng(int(np.sum(tile_prefs)) % (2**31))
        tiles = self._fallback_tiles(rng)
        self._decode_cache_store(tile_prefs, tiles, raw=self._tile_ids(tiles))
        return tiles

    def init_content(self, rng=None):
        """A random genome, one boost per cell ("We use one boost zone per grid
        cell", Sec. III-E): sparse boosts are decided by neighbours before
        their gene is read (at ~15% of cells only 14% of requests reach the
        layout on squares, 8% on hexes).  Genes are UNIFORM over the
        vocabulary (Sec. III-E(c), random chromosomes), not grass-heavy: a 15%
        road rate gives 1066 m mean tracks against 3031 m (squares; hexes
        1401 against 2706), and contentSwap draws from here, so most mutations
        would be grass onto grass.  Sparse loops come from the grass weight in
        the WFC collapse instead.  Shape genes uniform in [0, 1]."""
        if rng is None:
            rng = np.random.default_rng()
        elif isinstance(rng, int):
            rng = np.random.default_rng(rng)
        n = GRID_H * GRID_W
        content = {"tile_prefs": rng.integers(0, self._n_tiles(), size=n).astype(int)}
        if self._SHAPE_GENES:
            content["tile_shape"] = rng.random(n)
        return content

    # ── Decoding: tile grid -> track waypoints ────────────────────────

    def _extract_loop(self, *tiles):
        """The closed loop of mutually-connected road tiles, as an ordered
        list of (r, c), or None if no valid cycle exists."""
        return _walk_loop(self._build_neighbor_graph(*tiles))

    def _best_effort_path(self, *tiles):
        """For rendering: the largest connected chain of road tiles even if
        it is not a closed loop, as edge-midpoint waypoints, or None."""
        path = _walk_chain(self._build_neighbor_graph(*tiles))
        return None if path is None else self._edge_midpoints(path)

    # ── Content extraction bridge ─────────────────────────────────────
    # Overriding this makes all parent methods (simulate, render, etc.)
    # work transparently with tile-grid content dicts.

    def _genome_to_tiles(self, content):
        """Decode a tile-preference genome dict to its tiles."""
        return self._decode_genome(np.asarray(content["tile_prefs"], dtype=int))

    def _extract_content(self, content):
        if isinstance(content, dict) and "tile_prefs" in content:
            tiles = self._genome_to_tiles(content)
            pts = self._content_track_points(content, tiles)
            if pts is None:
                pts = self._best_effort_path(*self._tile_args(tiles))
            if pts is None:
                pts = self._default_track_points
            return np.array(pts)
        return super()._extract_content(content)

    def _build_curve(self, track_points):
        """Tile waypoints already trace the exact road (straights + sampled
        arcs), so the curve is only re-spaced to the shared arc-length step;
        a spline through them would re-introduce wobble."""
        return self._resample_uniform(track_points, step=self._curve_step())

    def _reencode(self, content, tiles):
        """Write the tiles WFC placed back into the genome (Alg. 1 l. 20, "as
        if it was the chromosome's choice in the first place"): only 14% of
        road requests survive on squares (8% on hexes), so otherwise crossover
        mixes wishes, not layouts.  In place: generators/search.py passes the
        chromosome's own content to evaluate(), so this updates the
        individual the GA breeds from (generate, evaluate, re-encode).  The
        RAW output, not the pruned loop ("the asset that has been placed"):
        writing the pruned loop deletes off-loop road every generation and
        ratchets the population to tiny loops (29 road genes to 4 in one
        round)."""
        prefs = content.get("tile_prefs", None)
        if prefs is None:
            return
        raw = self._raw_wfc.get(np.asarray(prefs, dtype=int).tobytes(), None)
        flat = np.asarray(raw if raw is not None else self._tile_ids(tiles), dtype=int).reshape(-1)
        prefs_arr = np.asarray(prefs, dtype=int)
        if flat.shape != prefs_arr.shape:
            return
        if isinstance(prefs, np.ndarray) and prefs.shape == flat.shape:
            prefs[:] = flat          # keep the GA's own array object
        else:
            content["tile_prefs"] = flat

    def info(self, content, trajectory=None, use_cache=True):
        """Decode the genome, re-encode it, score it.  Content that is not a
        tile genome goes to the base problem (the same test as
        _extract_content)."""
        if not (isinstance(content, dict) and "tile_prefs" in content):
            return super().info(content, trajectory=trajectory, use_cache=use_cache)
        # The genome before _reencode, for view_track's display only.
        requested = np.asarray(content["tile_prefs"], dtype=int).tolist()
        tiles = self._genome_to_tiles(content)
        self._reencode(content, tiles)
        track_pts = self._content_track_points(content, tiles)

        if track_pts is None or len(track_pts) < 3:
            return {
                'requested_tile_prefs': requested,
                'num_points': 0, 'total_length': 0.0,
                'steps': 0, 'finished': False,
                'track_points': np.zeros((0, 2)), 'trajectory_end': None,
                'curve_points': np.zeros((0, 2)),
            }

        # A copy: two genomes decoding to one track would share the cached dict.
        result = dict(super().info(
            {"track_points": np.array(track_pts)},
            trajectory=trajectory,
            use_cache=use_cache,
        ))
        result["requested_tile_prefs"] = requested
        return result

    # ── Rendering ─────────────────────────────────────────────────────

    def render(self, content=None, **kwargs):
        if isinstance(content, dict) and "tile_prefs" in content:
            self._tile_render_content = content
        else:
            self._tile_render_content = None
        return super().render(content, **kwargs)

    def _render_track_bg(self, img_w, img_h, left_edge_f, right_edge_f, scaled_curve,
                         grass_color, edge_color, road_color, centerline_color, scale=1.0):
        content = getattr(self, '_tile_render_content', None)
        if content is None:
            return super()._render_track_bg(img_w, img_h, left_edge_f, right_edge_f,
                                             scaled_curve, grass_color, edge_color,
                                             road_color, centerline_color, scale=scale)
        return self._tile_background(img_w, img_h, content, self._genome_to_tiles(content))


class RacingTileProblem(_GeneticWFCProblem):
    """Genetic-WFC on the square lattice; see the module docstring."""

    _SHAPE_GENES = True
    _SEED_TILE = 1          # (STRAIGHT, 0) in _WFC_TILES

    # No quality parameter is overridden (one function for all).  Each
    # cell's road stays disjoint: on 80 random genomes (seeds 7 and 21) no
    # crossing, no fold (the tightest tile arc, TIGHTEN at 14 m, is over twice
    # the 6 m fold radius), and parts of the
    # centreline more than _overlap_spot's separation apart come no closer
    # than 35.2 m.  The area check (6.6 ms) rejects nothing here and runs
    # because the gate is shared.  Lap length is tuned in the generator
    # (_WFC_WEIGHTS), not by moving the rule.

    # ── WFC tile index constants (used internally) ────────────────────
    # Indices into _WFC_TILES: 0=grass, 1=straight_H, 2=straight_V, then
    #   rotations 0-3 of each kind in _CURVE_KINDS: 3-6 sharp, 7-10 tighten,
    #   11-14 tighten mirror, 15-18 corner NE/ES/SW/WN, 19-22 double apex,
    #   23-26 kink, 27-30 kink mirror, 31-34 hairpin, 35-38 chicane, 39-42
    #   S chicane (rotations 0-1, then its mirror image at rotations 2-3),
    #   43-46 ess, 47-50 sweep.
    _WFC_TILES = [
        (GRASS, 0),
        (STRAIGHT, 0), (STRAIGHT, 1),
        *[(t, r) for t in _CURVE_KINDS for r in range(4)],
    ]
    # Vocabulary.  The census of the 24 circuits' 360 corners (FIA line) by
    # swept angle and mean radius, against the share of this vocabulary's
    # curve tiles (all rotations, over the shape gene; each alone between
    # straights) whose largest corner falls in each cell:
    #
    #   angle       slow          medium        fast          sweeper      (tiles | circuits)
    #   20-52       0.00 | 0.01   0.07 | 0.04   0.13 | 0.12   0.08 | 0.07
    #   52-75       0.02 | 0.06   0    | 0.08   0    | 0.05   0    | 0.01
    #   75-105      0.25 | 0.12   0.18 | 0.07   0    | 0.06   0    | 0
    #   105-127     0.12 | 0.04   0    | 0.03   0    | 0.02   0    | 0
    #   127-150     0.07 | 0.05   0    | 0.03   0    | 0.01   0    | 0
    #   150 and up  0.08 | 0.05   0    | 0.05   0    | 0.03   0    | 0
    #
    # The empty cells (0.38 of the circuits' corners) do not fit a square
    # cell: right angles wider than the 68 m half cell, and 52-75 degree turns
    # the four faces do not offer.  Over the reachable cells the vocabulary
    # is 0.35 from the census in L1 (the plain arc, SHARP, HAIRPIN, kinks and
    # chicanes alone: 0.80, reaching cells with 0.50 of the corners).
    # calibration/vocab_share.py.
    #
    # Weights, swept on 40 random genomes per seed: L1 of the built corner
    # class shares from the circuits' (0.13, 0.29, 0.24, 0.26, 0.08); tracks
    # inside the circuits' spread on length, turns, longest straight and
    # curvature entropy ("all four"); median turns; and tracks passing the
    # gates, rules and typicality on geometry ("pass", tile_weights.py):
    #
    #   grass straight curve   class L1 (s21, s22, s23)   all four   turns      pass
    #     3      4     0.125   0.41  0.41  0.37           0  0  2    37 36 40   14 13 11
    #     3     16     0.125   0.47  0.46  0.46           2  0  0    30 36 37   12 15 14
    #     3     32     0.125   0.46  0.48  0.43           6  2  1    32 37 34   14 12 16   (in use)
    #     8      8     0.125   0.37  0.44  0.35           2  0  0    32 36 27   15  8 12
    #
    # (seed 21 only, grass 3 / straight 4 at curve 0.0625, 0.25, 0.5: L1
    # 0.37, 0.35, 0.39, turns 36, 39, 58.)  Rule: the most tracks that pass
    # (42 of 120 in use, against 35-41; the same at 12 m), also the most with
    # all four.  The straight weight moves the mix since ESS, SWEEP and the
    # chicanes compete with the straight for its cells.  No setting reaches
    # the circuits' median of 15 turns: slow tiles need straights either
    # side, so their corners stay apart.  The plain-arc vocabulary gives 28
    # turns and L1 0.72 (seed 21).  calibration/mix_sweep.py, tile_weights.py.
    _WFC_WEIGHTS = np.array([_TYPE_WEIGHTS[t] for t, _ in _WFC_TILES])
    _WFC_COMPAT  = None  # populated once on first use; never changes

    # Bailly and Levieux boost a request by "a fixed and very high boosting
    # factor"; here a legal request wins outright (the factor's limit), so a
    # buildable genome decodes to exactly its layout.  At a factor of 1000 a
    # road request loses ~4% of the time, and one lost cell breaks its loop.

    @classmethod
    def _wfc_compat(cls):
        if cls._WFC_COMPAT is not None:
            return cls._WFC_COMPAT
        compat = {}
        for d in (N, E, S, W):
            opp = _OPPOSITE[d]
            compat[d] = []
            for t1, r1 in cls._WFC_TILES:
                # B may sit in direction d of A when the touching faces agree.
                a_open = d in _OPEN_EDGES[(t1, r1)]
                allowed = set()
                for j, (t2, r2) in enumerate(cls._WFC_TILES):
                    if (opp in _OPEN_EDGES[(t2, r2)]) == a_open:
                        allowed.add(j)
                compat[d].append(frozenset(allowed))
        cls._WFC_COMPAT = compat
        return compat

    def _wfc_propagate(self, wave, stack, compat):
        """AC-3 constraint propagation. Returns False on contradiction."""
        while stack:
            r, c = stack.pop()
            cur = wave[r][c]
            for d, (dr, dc) in _DIR_DELTA.items():
                nr, nc = r + dr, c + dc
                if not (0 <= nr < GRID_H and 0 <= nc < GRID_W):
                    continue
                allowed = set()
                for ti in cur:
                    allowed |= compat[d][ti]
                new = wave[nr][nc] & allowed
                if not new:
                    return False
                if new != wave[nr][nc]:
                    wave[nr][nc] = new
                    stack.append((nr, nc))
        return True

    # ── Single-loop enforcement ────────────────────────────────────────

    def _build_neighbor_graph(self, types, rotations):
        """Each road cell (r, c) to the neighbours it is mutually open to
        (grid adjacency is not enough)."""
        neighbors: dict = {}
        for r in range(GRID_H):
            for c in range(GRID_W):
                edges = _OPEN_EDGES[(int(types[r, c]), int(rotations[r, c]))]
                if not edges:
                    continue
                nbrs = []
                for d in edges:
                    dr, dc = _DIR_DELTA[d]
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < GRID_H and 0 <= nc < GRID_W:
                        if _OPPOSITE[d] in _OPEN_EDGES[(int(types[nr, nc]), int(rotations[nr, nc]))]:
                            nbrs.append((nr, nc))
                neighbors[(r, c)] = nbrs
        return neighbors

    def _edge_midpoints(self, cells):
        """Waypoints of an ordered cell list: per cell, the midpoint of the
        face shared with the next cell."""
        n = len(cells)
        points = []
        for i, (r, c) in enumerate(cells):
            nr, nc = cells[(i + 1) % n]
            px = (c + nc + 1) * self._width  / (2 * GRID_W)
            py = (r + nr + 1) * self._height / (2 * GRID_H)
            points.append((px, py))
        return points

    def _keep_largest_component(self, types, rotations):
        """Keep only road tiles forming the largest single closed loop
        (_largest_loop); without one the tiles are returned unchanged and
        _extract_loop rejects them."""
        largest = _largest_loop(self._build_neighbor_graph(types, rotations))
        if largest is None:
            return types, rotations
        types, rotations = types.copy(), rotations.copy()
        for r in range(GRID_H):
            for c in range(GRID_W):
                if types[r, c] != GRASS and (r, c) not in largest:
                    types[r, c], rotations[r, c] = GRASS, 0
        return types, rotations

    # ── Hooks of _GeneticWFCProblem ───────────────────────────────────

    def _n_tiles(self):
        return len(self._WFC_TILES)

    def _tile_weights(self):
        return self._WFC_WEIGHTS

    def _ids_to_tiles(self, ids):
        """A grid of tile ids to (types, rotations)."""
        types     = np.zeros((GRID_H, GRID_W), dtype=int)
        rotations = np.zeros((GRID_H, GRID_W), dtype=int)
        for r in range(GRID_H):
            for c in range(GRID_W):
                types[r, c], rotations[r, c] = self._WFC_TILES[ids[r, c]]
        return types, rotations

    def _tile_ids(self, tiles):
        return self._types_to_wave(*tiles)

    def _fallback_tiles(self, rng):
        fb = self._rectangular_fallback(rng)
        return np.asarray(fb["types"], dtype=int), np.asarray(fb["rotations"], dtype=int)

    def _content_track_points(self, content, tiles):
        return self._grid_to_track_points(*tiles, self._content_shapes(content))

    def _tile_background(self, img_w, img_h, content, tiles):
        return self._make_tile_bg(img_w, img_h, *tiles, self._content_shapes(content))

    # ── Wave ↔ tile-array conversion helpers ─────────────────────────

    _WFC_TILE_IDX: dict | None = None  # (type, rotation) → WFC tile index

    @classmethod
    def _tile_to_idx(cls):
        if cls._WFC_TILE_IDX is None:
            cls._WFC_TILE_IDX = {(t, r): i for i, (t, r) in enumerate(cls._WFC_TILES)}
        return cls._WFC_TILE_IDX

    def _types_to_wave(self, types, rotations):
        """Convert (types, rotations) to a (GRID_H, GRID_W) tile-index array."""
        idx = self._tile_to_idx()
        return np.array(
            [[idx.get((int(types[r, c]), int(rotations[r, c])), 0)
              for c in range(GRID_W)]
             for r in range(GRID_H)],
            dtype=int,
        )

    def _rectangular_fallback(self, rng):
        """A rectangle of straights and quarter arcs (_rectangle_loop)."""
        loop = _rectangle_loop(rng)
        types     = np.zeros((GRID_H, GRID_W), dtype=int)
        rotations = np.zeros((GRID_H, GRID_W), dtype=int)
        n = len(loop)
        for i, (r, c) in enumerate(loop):
            pr, pc = loop[(i - 1) % n]
            nr, nc = loop[(i + 1) % n]
            from_dir = self._direction_toward(r, c, pr, pc)
            to_dir   = self._direction_toward(r, c, nr, nc)
            t, rot = _EDGES_TO_TILE.get(frozenset({from_dir, to_dir}), (GRASS, 0))
            types[r, c], rotations[r, c] = t, rot
        return {"types": types, "rotations": rotations}

    @staticmethod
    def _direction_toward(r, c, target_r, target_c):
        """Return the compass direction that steps from (r, c) to the target cell."""
        for d, (dr, dc) in _DIR_DELTA.items():
            if r + dr == target_r and c + dc == target_c:
                return d
        return None

    # ── Decoding: tile grid -> track waypoints ────────────────────────

    def _tile_polyline(self, tile_type, rotation, shape=0.5):
        """(entry face, centre line relative to the cell centre in metres),
        from the entry face midpoint to the other open face's.  The canonical
        _tile_points path (half-cell units, y north) is turned `rotation`
        quarter turns clockwise and flipped to the map's y-down frame.
        `shape` is the cell's shape gene; every tile but the straight gets
        _TRANSITION_M clothoid transitions."""
        q = round(float(shape) * _SHAPE_LEVELS) / _SHAPE_LEVELS
        key = (int(tile_type), int(rotation), q)
        if key in self._tile_polylines:
            return self._tile_polylines[key]
        hx = 0.5 * self._width / GRID_W
        hy = 0.5 * self._height / GRID_H
        kind = key[0]
        corner_family = kind in _CORNER_FAMILY
        lo, hi = _SHAPE_RANGES.get(kind, (0.0, 0.0))
        value = lo + q * (hi - lo)
        # Half a turn maps an S chicane onto itself, so rotations 2-3 carry its
        # mirror (CHICANE, ESS, SWEEP mirror under half a turn already).
        mirror = kind in _MIRROR_OF or (kind == S_CHICANE and key[1] >= 2)
        base = _MIRROR_OF.get(kind, kind)
        pts = _tile_points(base,
                           90.0 if corner_family else 0.0,
                           lambda radius, turn: self._arc_samples(radius * hx * abs(turn)),
                           mirror=mirror,
                           hairpin_leg=_SQUARE_HAIRPIN_LEG * (value if kind == HAIRPIN else 1.0),
                           kink_strong=_SQUARE_KINK_RADII[0],
                           kink_counter=value if base == KINK else 20.0,
                           kink_second=_SQUARE_KINK_RADII[1],
                           corner_radius=value if kind == CORNER else _SQUARE_CORNER_RADIUS,
                           sharp_radius=value if kind == SHARP else 0.6,
                           tighten_radius=value if base == TIGHTEN else 0.25,
                           double_gap=value if kind == DOUBLE_APEX else 0.3,
                           chicane=(value, 0.4) if kind == CHICANE else None,
                           s_chicane=(value, 0.25) if kind == S_CHICANE else None,
                           ess=value if kind in (ESS, SWEEP) else None,
                           transition=_TRANSITION_M / hx,
                           step=0.5 * self._curve_step() / hx)
        for _ in range(key[1]):
            pts = np.column_stack([pts[:, 1], -pts[:, 0]])
        pts = np.column_stack([pts[:, 0] * hx, -pts[:, 1] * hy])
        entry = ((N if corner_family else W) + key[1]) % 4
        self._tile_polylines[key] = (entry, pts)
        return self._tile_polylines[key]

    def _loop_to_track_points(self, loop, types, rotations, shapes=None):
        """Waypoints along the tiles' road: each tile's centre line
        (_tile_polyline), reversed when entered by its other face, without
        its first point.  The polyline IS the drawn road (a spline through
        face midpoints bowed the straights and wobbled at corners)."""
        n = len(loop)
        points = []
        for i, (r, c) in enumerate(loop):
            pr, pc = loop[(i - 1) % n]
            entry, local = self._tile_polyline(
                types[r, c], rotations[r, c], 0.5 if shapes is None else shapes[r, c])
            if self._direction_toward(r, c, pr, pc) != entry:
                local = local[::-1]
            cx = (c + 0.5) * self._width / GRID_W
            cy = (r + 0.5) * self._height / GRID_H
            points.extend((cx + x, cy + y) for x, y in local[1:])
        return points

    def _grid_to_track_points(self, types, rotations, shapes=None):
        """Convert tile grid to ordered pixel waypoints along the tile road."""
        loop = self._extract_loop(types, rotations)
        if loop is None:
            return None
        return self._loop_to_track_points(loop, types, rotations, shapes)

    @staticmethod
    def _content_shapes(content):
        """The shape genes as a (GRID_H, GRID_W) array, or None for content
        saved without them (decoded at 0.5, see _SHAPE_RANGES)."""
        shapes = content.get("tile_shape") if isinstance(content, dict) else None
        if shapes is None:
            return None
        shapes = np.clip(np.asarray(shapes, dtype=float).reshape(-1), 0.0, 1.0)
        if shapes.size != GRID_H * GRID_W:
            raise ValueError("tile_shape must have %d entries, got %d" % (GRID_H * GRID_W, shapes.size))
        return shapes.reshape(GRID_H, GRID_W)

    # ── Tile rendering ────────────────────────────────────────────────

    def _make_tile_bg(self, img_w, img_h, types, rotations, shapes=None):
        bg   = Image.new("RGB", (img_w, img_h), (34, 139, 34))
        draw = ImageDraw.Draw(bg)
        cw   = img_w / GRID_W
        ch   = img_h / GRID_H
        for r in range(GRID_H):
            for c in range(GRID_W):
                self._draw_tile_pil(draw, c * cw, r * ch, cw, ch,
                                    int(types[r, c]), int(rotations[r, c]),
                                    0.5 if shapes is None else shapes[r, c])
        for i in range(GRID_H + 1):
            y = int(i * ch)
            draw.line([0, y, img_w, y], fill=(20, 100, 20), width=1)
        for j in range(GRID_W + 1):
            x = int(j * cw)
            draw.line([x, 0, x, img_h], fill=(20, 100, 20), width=1)
        return bg

    def _draw_tile_pil(self, draw, x0, y0, cw, ch, t, rot, shape=0.5):
        """Draw one tile's road at the track width along its centre line, so
        the picture is the road the car drives: a dark line two pixels wider
        per side under the grey road."""
        ROAD = (210, 210, 210)
        EDGE = (25,  25,  25)
        GRASS_C = (34, 139, 34)
        draw.rectangle([x0, y0, x0 + cw - 1, y0 + ch - 1], fill=GRASS_C)
        if t == GRASS:
            return
        scale = cw / (self._width / GRID_W)
        _, local = self._tile_polyline(t, rot, shape)
        pts = [(x0 + 0.5 * cw + x * scale, y0 + 0.5 * ch + y * scale) for x, y in local]
        road_px = max(1, int(round(self._track_width * scale)))
        draw.line(pts, fill=EDGE, width=road_px + 4, joint="curve")
        draw.line(pts, fill=ROAD, width=road_px, joint="curve")
