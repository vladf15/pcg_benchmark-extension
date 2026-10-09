"""TORCS as an optional driver: export a centreline as a TORCS track, race it
with one of TORCS's shipped robots, and read back what TORCS records.

Used by RacingProblem when driver="torcs" (or PCG_RACING_DRIVER=torcs), and
by benchmark_experiments-extension/tools/torcs_check.py.  Nothing here runs unless
one of those asks for it; the default driver is the benchmark's own.

TORCS (Wymann et al., The Open Racing Car Simulator; references.bib
wymann2000torcs), version 1.3.10.  TORCS loads tracks only from its own
tracks/ folder, so it is installed where this module can write: TORCS_DIR,
by default the folder torcs next to the two repos.

One race, run():
  1. the centreline becomes TORCS segments (segments(), below);
  2. a track file goes to TORCS_DIR/tracks/road/<name>/<name>.xml and a race
     file to TORCS_DIR/config/raceman/<name>.xml;
  3. `wtorcs.exe -r <race file>` races it with no graphics, as fast as the
     machine allows (0.27-0.45 s for two laps of a 3.5-6.6 km track);
  4. the results file TORCS writes under %LOCALAPPDATA%/torcs/results/<name>/
     is read, and all three files are deleted.
<name> holds the process id and a counter, so races in parallel processes
never share a file; 8 races at once give the same results as one at a time.

Geometry.  The benchmark's centreline is closed, at a 5 m arc-length step
(problem._resample_uniform), in metres with y up (the driver's left normal
is the tangent turned anticlockwise, agent._seg_norm).  TORCS lists a track
as straights and constant-radius arcs in driving order (track manual,
torcs.sourceforge.net/api/track_manual.html).  Each centreline vertex
becomes one arc from the midpoint of the edge before it to the midpoint of
the edge after it, turning by the polygon's angle at that vertex.  On equal
edges that arc is tangent to both at its ends, so the TORCS centreline runs
through every edge midpoint with the edge's heading; the headings close
exactly (a simple closed polygon turns by 2 pi).  Measured on one final
track per representation (torcs_check.py selftest): closure within 4 cm,
largest distance from the centreline 10 cm (Voronoi; 0.2-1.3 cm for the
rest), 243-458 segments.  Runs of vertices turning less than STRAIGHT_RAD
merge into one straight, their turn carried into the next arc so the
headings still close.  Why not one straight per edge: TORCS segments must
join with matching headings, so a polygon of straights is not a valid
track.  Why not a fitted spline or clothoid: TORCS has none; its varying-
radius turn ("end radius") is the next rung up and would cut the segment
count, not the error.  The benchmark's renders draw y down, so a TORCS
track is the mirror image of its PNG.
"""
import itertools
import math
import os
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET

import numpy as np

# A vertex turning less than this counts as straight.  At the 5 m step a
# turn of 1e-4 rad is a radius of 50 km; dropping it moves the next 5 m by
# 0.25 mm, and the turn is not lost (it is carried into the next arc).
STRAIGHT_RAD = 1e-4

# Road width when the caller gives none; RacingProblem passes its own
# _track_width, so a TORCS track is as wide as the benchmark's road.
WIDTH_M = 12.0
# Beyond each road edge: 10 m of grass, then a wall; a MODELLING CHOICE.
# The benchmark has no run-off (a car past the edge counts as off-road and
# drives on), so some run-off is needed for a TORCS lap to mean the same
# thing; 10 m of the shipped "grass" grip (friction 0.4) lets a robot that
# runs wide recover, as on g-track-1 (6 m sides, then a wall).  Why not a
# wall at the road edge: every excursion would be a crash, stricter than
# the benchmark's own measure.
SIDE_M = 10.0
# The sides keep this width everywhere.  Inside a corner tighter than the
# run-off is deep (16.5 m to the wall) the side folds back on itself, and
# where two stretches of the lap run closer than two run-offs one's grass
# and wall overlap the other's road.  Both show in TORCS's window; in the
# race they did not cost a lap: Shanghai (tightest radius 6.5 m) races
# clean, as do the final spline and Voronoi tracks of seed 1, the Voronoi
# one with two stretches 17.8 m apart centre to centre.  Narrowing the inside
# side to keep the wall off the road was tried and put Shanghai's inner
# wall at the road edge (berniw hit it, damage 538), so it was dropped.
BARRIER_M = 0.5
# Ground outside the walls in the 3D model only (cars cannot reach it): the
# shipped "grass3" texture, so it reads apart from the run-off.  Without it
# trackgen textures the ground with a file TORCS does not ship (grass.rgb).
TERRAIN = "grass3"

# The race: one robot, LAPS laps from a standing start, in race mode, the
# mode whose results carry the penalty time (raceresults.cpp,
# ReStoreRaceResults: laps, time, penalty time, best lap time, top speed,
# damage).  Corner cutting is penalised, so leaving the road on the inside
# of a corner shows up as penalty time.  Tyre temperature and wear off
# ("tire factor" 0, TORCS 1.3.8 changelog), since the benchmark's car has
# neither.  Damage and fuel at the shipped practice.xml values (1.0).
LAPS = 2
# berniw (Bernhard Wymann, shipped with TORCS), its driver 3: car1-trb1, a
# TORCS Racing Board GT car (drivers/berniw/berniw.xml).  bt 3 drives the
# same car with a different robot.
ROBOT, ROBOT_IDX = "berniw", 3
# Wall-clock limit of one race.  A two-lap race takes under 0.5 s, so a race
# still running after this has a robot that cannot get round.
TIMEOUT_S = 60

TORCS_DIR = os.environ.get("TORCS_DIR", os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "torcs")))
# TORCS's settings folder: results, and a copy of every race file it runs.
LOCAL_DIR = os.path.join(os.environ.get("LOCALAPPDATA", ""), "torcs")
RESULTS_DIR = os.path.join(LOCAL_DIR, "results")
_COUNTER = itertools.count()


def available():
    return os.path.isfile(os.path.join(TORCS_DIR, "wtorcs.exe"))


def segments(curve):
    """TORCS segments [(kind, length m, arc rad)] for a closed centreline
    (first point repeated at the end or not), kind 'str', 'lft' or 'rgt',
    starting at the midpoint of the first edge."""
    c = np.asarray(curve, dtype=float).reshape(-1, 2)
    if len(c) > 1 and np.allclose(c[0], c[-1], atol=1e-9):
        c = c[:-1]
    edge = np.roll(c, -1, axis=0) - c
    half = 0.5 * np.linalg.norm(edge, axis=1)
    head = np.arctan2(edge[:, 1], edge[:, 0])
    turn = (np.roll(head, -1) - head + math.pi) % (2.0 * math.pi) - math.pi   # at vertex i + 1
    out, run, carry = [], 0.0, 0.0
    for i in range(len(c)):
        a = turn[i] + carry
        d = half[i] + half[(i + 1) % len(c)]          # midpoint to midpoint, along the edges
        if abs(a) < STRAIGHT_RAD:
            run += d
            carry = a
            continue
        if run:
            out.append(("str", run, 0.0))
            run = 0.0
        carry = 0.0
        # the arc tangent to both half-edges at their far ends: chord |m1 m2| = d cos(a/2) on
        # equal halves, so radius = chord / (2 sin(|a|/2)) and length = radius |a|
        chord = d * math.cos(0.5 * a)
        radius = chord / (2.0 * math.sin(0.5 * abs(a)))
        out.append(("lft" if a > 0 else "rgt", radius * abs(a), abs(a)))
    if run:
        out.append(("str", run, 0.0))
    return out


def rebuild(segs, start, heading):
    """Points at every segment end, integrating the segments from `start`
    at `heading` the way TORCS lays them out."""
    p, h, pts = np.asarray(start, dtype=float), float(heading), []
    for kind, length, arc in segs:
        if kind == "str":
            p = p + length * np.array([math.cos(h), math.sin(h)])
        else:
            s = 1.0 if kind == "lft" else -1.0
            r = length / arc
            mid = h + s * 0.5 * arc
            p = p + 2.0 * r * math.sin(0.5 * arc) * np.array([math.cos(mid), math.sin(mid)])
            h += s * arc
        pts.append(p)
    return np.array(pts), h


def track_xml(name, segs, width=WIDTH_M):
    """A TORCS track file, laid out as the shipped tracks (tracks/road/
    g-track-1/g-track-1.xml): Header, Graphic, Main Track with its sides and
    barriers, then the segments in driving order, then the Objects and
    Surfaces sections that pull in TORCS's shipped object and surface lists.
    Without the Surfaces section TORCS knows none of the names used here and
    gives road, grass and wall one default surface, in grip and in the 3D
    model alike.  Flat and unbanked, as the benchmark's tracks are 2D; no pits."""
    rows = []
    for i, (kind, length, arc) in enumerate(segs):
        rows.append('      <section name="s%04d">' % i)
        rows.append('        <attstr name="type" val="%s"/>' % kind)
        if kind == "str":
            rows.append('        <attnum name="lg" unit="m" val="%.4f"/>' % length)
        else:
            rows.append('        <attnum name="radius" unit="m" val="%.4f"/>' % (length / arc))
            rows.append('        <attnum name="arc" unit="deg" val="%.6f"/>' % math.degrees(arc))
        rows.append('      </section>')
    return _TRACK % {"name": name, "width": width, "side": SIDE_M, "terrain": TERRAIN,
                     "barrier": BARRIER_M, "segments": "\n".join(rows)}


def run(curve, robot=ROBOT, idx=ROBOT_IDX, laps=LAPS, keep=False, name=None, width=WIDTH_M):
    """Race one closed centreline in TORCS.  Returns a dict: laps completed,
    finished (all `laps`), time (s, all laps), best_lap (s), penalty (s),
    damage, top_speed (m/s), segments, length_m, wall_s, log, track.
    keep=True leaves the track in TORCS (tracks/road/<track>), under `name`
    when given; build_3d() then makes it viewable in TORCS's own window.
    The race and results files are always deleted."""
    if not available():
        raise FileNotFoundError("no wtorcs.exe in %s; set TORCS_DIR to a writable copy of the TORCS install"
                                % TORCS_DIR)
    name = name or "pcg-%d-%d" % (os.getpid(), next(_COUNTER))
    segs = segments(curve)
    folder = os.path.join(TORCS_DIR, "tracks", "road", name)
    race = os.path.join(TORCS_DIR, "config", "raceman", name + ".xml")
    out_dir = os.path.join(RESULTS_DIR, name)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, name + ".xml"), "w") as f:
        f.write(track_xml(name, segs, width))
    with open(race, "w") as f:
        f.write(_RACE % {"race": name, "track": name, "laps": laps, "robot": robot, "idx": idx})
    t0 = time.time()
    try:
        proc = subprocess.run([os.path.join(TORCS_DIR, "wtorcs.exe"), "-r", race], cwd=TORCS_DIR,
                              stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=TIMEOUT_S)
        log = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired as e:     # subprocess.run has killed wtorcs.exe, which starts no children
        log = "timeout after %d s\n%s" % (TIMEOUT_S, e.stdout or "")
    wall = time.time() - t0
    res = {"laps": 0, "time": float("nan"), "best_lap": float("nan"), "penalty": 0.0, "damage": 0.0,
           "top_speed": float("nan")}
    files = sorted(os.listdir(out_dir)) if os.path.isdir(out_dir) else []
    if files:
        root = ET.parse(os.path.join(out_dir, files[-1])).getroot()
        for sec in root.iter("section"):
            if sec.get("name") == "Rank":
                first = sec.find("section")
                if first is not None:
                    v = {a.get("name"): float(a.get("val")) for a in first.findall("attnum")}
                    res = {"laps": int(v.get("laps", 0)), "time": v.get("time", float("nan")),
                           "best_lap": v.get("best lap time") or float("nan"),   # 0 when a wall touch voided it
                           "penalty": v.get("penaltytime", 0.0), "damage": v.get("dammages", 0.0),
                           "top_speed": v.get("top speed", float("nan"))}
                break
    shutil.rmtree(out_dir, ignore_errors=True)
    # TORCS copies the race file into LOCAL_DIR; either copy left behind shows in its race menus
    for f in (race, os.path.join(LOCAL_DIR, "config", "raceman", name + ".xml")):
        if os.path.exists(f):
            os.remove(f)
    if not keep:
        shutil.rmtree(folder, ignore_errors=True)
    res.update({"finished": res["laps"] >= laps, "segments": len(segs),
                "length_m": float(sum(s[1] for s in segs)), "wall_s": wall, "log": log, "track": name})
    return res


def build_3d(name):
    """Build the 3D model of a kept track (tracks/road/<name>/<name>.ac) with
    TORCS's trackgen, so it can be opened in TORCS's window.  Races do not
    need it: the physics reads the track from the XML, and results-only mode
    draws nothing.  Returns the model's path, or None when trackgen failed."""
    subprocess.run([os.path.join(TORCS_DIR, "trackgen.exe"), "-c", "road", "-n", name, "-a"], cwd=TORCS_DIR,
                   stdin=subprocess.DEVNULL, capture_output=True, timeout=600)
    path = os.path.join(TORCS_DIR, "tracks", "road", name, name + ".ac")
    return path if os.path.isfile(path) else None


def watch(curve, name, width=WIDTH_M, robot=ROBOT, idx=ROBOT_IDX):
    """Open TORCS's window on a centreline: write it as track `name`, build
    its 3D model, set TORCS's Practice race to that track with `robot` and
    tyre wear off, and start wtorcs.exe without waiting for it.  TORCS
    starts no race with graphics from the command line, so the window opens
    at the main menu: Race, Practice, New Race.  Returns the track folder."""
    if not available():
        raise FileNotFoundError("no wtorcs.exe in %s" % TORCS_DIR)
    folder = os.path.join(TORCS_DIR, "tracks", "road", name)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, name + ".xml"), "w") as f:
        f.write(track_xml(name, segments(curve), width))
    if build_3d(name) is None:
        raise RuntimeError("trackgen could not build the 3D model of " + name)
    # TORCS reads the Practice settings from LOCAL_DIR once it has run, from its install before that
    for d in (LOCAL_DIR, TORCS_DIR):
        raceman = os.path.join(d, "config", "raceman")
        if os.path.isdir(raceman):
            with open(os.path.join(raceman, "practice.xml"), "w") as f:
                f.write(_PRACTICE % {"track": name, "robot": robot, "idx": idx})
    subprocess.Popen([os.path.join(TORCS_DIR, "wtorcs.exe")], cwd=TORCS_DIR, stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return folder


_TRACK = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE params SYSTEM "../../../src/libs/tgf/params.dtd" [
<!ENTITY default-surfaces SYSTEM "../../../data/tracks/surfaces.xml">
<!ENTITY default-objects SYSTEM "../../../data/tracks/objects.xml">
]>
<params name="%(name)s" type="param" mode="mw">
  <section name="Header">
    <attstr name="name" val="%(name)s"/>
    <attstr name="category" val="road"/>
    <attnum name="version" val="4"/>
    <attstr name="author" val="pcg benchmark export"/>
    <attstr name="description" val="generated track"/>
  </section>
  <section name="Graphic">
    <attstr name="3d description" val="%(name)s.ac"/>
    <section name="Terrain Generation">
      <attstr name="surface" val="%(terrain)s"/>
    </section>
  </section>
  <section name="Main Track">
    <attnum name="width" unit="m" val="%(width).1f"/>
    <attstr name="surface" val="asphalt2"/>
    <attnum name="profil steps length" unit="m" val="5"/>
    <section name="Left Side">
      <attnum name="width" unit="m" val="%(side).1f"/>
      <attstr name="surface" val="grass"/>
    </section>
    <section name="Right Side">
      <attnum name="width" unit="m" val="%(side).1f"/>
      <attstr name="surface" val="grass"/>
    </section>
    <section name="Left Barrier">
      <attstr name="style" val="wall"/>
      <attnum name="height" unit="m" val="1.0"/>
      <attnum name="width" unit="m" val="%(barrier).1f"/>
      <attstr name="surface" val="wall"/>
    </section>
    <section name="Right Barrier">
      <attstr name="style" val="wall"/>
      <attnum name="height" unit="m" val="1.0"/>
      <attnum name="width" unit="m" val="%(barrier).1f"/>
      <attstr name="surface" val="wall"/>
    </section>
    <section name="Track Segments">
%(segments)s
    </section>
  </section>
  <section name="Objects">
    &default-objects;
  </section>
  <section name="Surfaces">
    &default-surfaces;
  </section>
</params>
"""


_RACE = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE params SYSTEM "params.dtd">
<params name="%(race)s" type="param" mode="mw">
  <section name="Header">
    <attstr name="name" val="Race"/>
    <attnum name="priority" val="100"/>
  </section>
  <section name="Tracks">
    <attnum name="maximum number" val="1"/>
    <section name="1">
      <attstr name="name" val="%(track)s"/>
      <attstr name="category" val="road"/>
    </section>
  </section>
  <section name="Races">
    <section name="1">
      <attstr name="name" val="Race"/>
    </section>
  </section>
  <section name="Race">
    <attnum name="laps" val="%(laps)d"/>
    <attstr name="type" val="race"/>
    <attstr name="starting order" val="drivers list"/>
    <attstr name="restart" val="no"/>
    <attstr name="display mode" val="results only"/>
    <attstr name="display results" val="yes"/>
    <attstr name="corner cutting time penalty" val="yes"/>
    <attnum name="fuel consumption factor" val="1.0"/>
    <attnum name="damage factor" val="1.0"/>
    <attnum name="tire factor" val="0"/>
    <section name="Starting Grid">
      <attnum name="rows" val="1"/>
      <attnum name="distance to start" val="100"/>
      <attnum name="distance between columns" val="20"/>
      <attnum name="offset within a column" val="10"/>
      <attnum name="initial speed" unit="km/h" val="0"/>
      <attnum name="initial height" unit="m" val="0.2"/>
    </section>
  </section>
  <section name="Drivers">
    <attnum name="maximum number" val="1"/>
    <section name="1">
      <attnum name="idx" val="%(idx)d"/>
      <attstr name="module" val="%(robot)s"/>
    </section>
  </section>
</params>
"""

# The shipped practice.xml with the track, the driver (also the one the
# camera follows) and tyre wear set; its Configuration section keeps TORCS's
# Configure Race menu working.
_PRACTICE = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE params SYSTEM "../libs/tgf/params.dtd">
<params name="practice" type="param" mode="mw">
  <section name="Header">
    <attstr name="name" val="Practice"/>
    <attstr name="description" val="Practice"/>
    <attnum name="priority" val="100"/>
    <attstr name="menu image" val="data/img/splash-practice.png"/>
    <attstr name="run image" val="data/img/splash-run-practice.png"/>
  </section>
  <section name="Tracks">
    <attnum name="maximum number" val="1"/>
    <section name="1">
      <attstr name="name" val="%(track)s"/>
      <attstr name="category" val="road"/>
    </section>
  </section>
  <section name="Races">
    <section name="1">
      <attstr name="name" val="Practice"/>
    </section>
  </section>
  <section name="Practice">
    <attnum name="laps" val="20"/>
    <attstr name="type" val="practice"/>
    <attstr name="starting order" val="drivers list"/>
    <attstr name="restart" val="yes"/>
    <attstr name="display mode" val="normal"/>
    <attstr name="display results" val="yes"/>
    <attstr name="invalidate best lap on wall touch" val="yes"/>
    <attstr name="invalidate best lap on corner cutting" val="yes"/>
    <attstr name="corner cutting time penalty" val="yes"/>
    <attnum name="fuel consumption factor" min="0.0" max="5.0" val="1.0"/>
    <attnum name="damage factor" min="0.0" max="5.0" val="1.0"/>
    <attnum name="tire factor" min="0" max="5" val="0"/>
    <section name="Starting Grid">
      <attnum name="rows" val="1"/>
      <attnum name="distance to start" val="100"/>
      <attnum name="distance between columns" val="20"/>
      <attnum name="offset within a column" val="10"/>
      <attnum name="initial speed" unit="km/h" val="0"/>
      <attnum name="initial height" unit="m" val="0.2"/>
    </section>
  </section>
  <section name="Drivers">
    <attnum name="maximum number" val="1"/>
    <attstr name="focused module" val="%(robot)s"/>
    <attnum name="focused idx" val="%(idx)d"/>
    <section name="1">
      <attnum name="idx" val="%(idx)d"/>
      <attstr name="module" val="%(robot)s"/>
    </section>
  </section>
  <section name="Configuration">
    <section name="1">
      <attstr name="type" val="track select"/>
    </section>
    <section name="2">
      <attstr name="type" val="drivers select"/>
    </section>
    <section name="3">
      <attstr name="type" val="race config"/>
      <attstr name="race" val="Practice"/>
      <section name="Options">
        <section name="1">
          <attstr name="type" val="race length"/>
        </section>
        <section name="2">
          <attstr name="type" val="display mode"/>
        </section>
      </section>
    </section>
  </section>
</params>
"""
