"""Render a driven lap on a TUMFTM circuit as an animated GIF.

Companion to visualize.py: that draws the whole lap as one static picture,
this plays it back so the driving itself can be watched (where it brakes,
whether it slides, how it picks up the throttle on exit).

    python render_run.py --track Spa
    python render_run.py --track Monza --model ../runs/ppo_driver/best/best_model.zip
    python render_run.py --all --out ../renders

The camera follows the car by default; --whole-track holds the full circuit
in frame instead. Real circuits are 2-7 km long, so a full lap at 10 Hz is
1500-2500 frames; --speedup keeps the file small by drawing every Nth step
(the gif still plays in real time).

Two GIF details, same as the benchmark's own gif writer:
- every frame is mapped to the FIRST frame's palette, otherwise the colors
  shimmer as each frame picks its own
- Pillow merges identical consecutive frames, which breaks the frame rate,
  so one corner pixel alternates between two identical palette slots to
  keep every frame distinct
"""

import argparse
import math
import os

import numpy as np
from PIL import Image, ImageDraw

import track_loader
from baseline import make_policy
from racing_env import RacingEnv

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.normpath(os.path.join(_HERE, "..", "renders"))

# Colors (RGB).
C_BG = (30, 32, 36)
C_GRASS = (44, 74, 48)
C_ROAD = (72, 74, 80)
C_EDGE = (235, 235, 235)
C_CENTER = (120, 120, 128)
C_CAR = (228, 64, 52)
C_TRAIL = (250, 196, 60)
C_TEXT = (240, 240, 240)

# Drawn to scale: the 992 is 4519 x 1852 mm (Porsche 2020 technical data).
CAR_LEN = 4.5
CAR_WID = 1.9


def _edges(geom):
    """Left and right road edges from the centerline."""
    pts = geom.points
    tan = geom.seg_tan
    nrm = np.column_stack([-tan[:, 1], tan[:, 0]])
    return pts + nrm * geom.half_width, pts - nrm * geom.half_width


class _Camera:
    """Maps world metres to pixels for one frame."""

    def __init__(self, size_px, view_m, center_xy):
        self.size = size_px
        self.scale = size_px / view_m
        self.cx, self.cy = center_xy

    def __call__(self, p):
        x = (p[0] - self.cx) * self.scale + self.size * 0.5
        # Screen y grows downward; world y grows up.
        y = self.size * 0.5 - (p[1] - self.cy) * self.scale
        return (x, y)

    def many(self, pts):
        out = np.empty((len(pts), 2))
        out[:, 0] = (pts[:, 0] - self.cx) * self.scale + self.size * 0.5
        out[:, 1] = self.size * 0.5 - (pts[:, 1] - self.cy) * self.scale
        return [tuple(q) for q in out]


def _draw_frame(geom, left, right, state, trail, size_px, view_m, follow,
                label=""):
    """One frame: road, driven trail so far, car, HUD."""
    x, y, heading, speed, throttle, steer, step, slip = state
    center = (x, y) if follow else (geom.map_size * 0.5, geom.map_size * 0.5)
    cam = _Camera(size_px, view_m, center)

    img = Image.new("RGB", (size_px, size_px), C_GRASS if follow else C_BG)
    d = ImageDraw.Draw(img)

    # Road surface as a thick line along the centerline, then the edges.
    road_px = max(int(geom.track_width * cam.scale), 1)
    ring = np.vstack([geom.points, geom.points[:1]])
    d.line(cam.many(ring), fill=C_ROAD, width=road_px, joint="curve")
    d.line(cam.many(np.vstack([left, left[:1]])), fill=C_EDGE, width=2)
    d.line(cam.many(np.vstack([right, right[:1]])), fill=C_EDGE, width=2)
    if road_px > 6:
        d.line(cam.many(ring), fill=C_CENTER, width=1)

    if len(trail) > 1:
        d.line(cam.many(np.asarray(trail)), fill=C_TRAIL,
               width=max(int(0.9 * cam.scale), 2))

    # Car body as a rotated rectangle, with a nose marker so its heading is
    # readable even when a slide has it pointing away from its direction of
    # travel (which is exactly the moment worth watching).
    c, s = math.cos(heading), math.sin(heading)

    def body(px, py):
        return cam((x + px * c - py * s, y + px * s + py * c))

    hl, hw = CAR_LEN * 0.5, CAR_WID * 0.5
    d.polygon([body(hl, hw), body(hl, -hw), body(-hl, -hw), body(-hl, hw)],
              fill=C_CAR, outline=C_EDGE)
    d.line([body(0.0, 0.0), body(hl * 1.9, 0.0)], fill=C_EDGE, width=2)

    # HUD: speed, pedal, steering, elapsed time, and body slip angle (the
    # number that says whether the car is sliding, which is the thing the
    # physics made possible and the agent has to manage).
    bar = 90
    d.text((8, 6), f"{speed * 3.6:5.0f} km/h", fill=C_TEXT)
    d.text((8, 20), f"t {step * 0.1:6.1f} s", fill=C_TEXT)
    if label:
        d.text((size_px - 8 - 6 * len(label), 6), label, fill=C_TEXT)
    if abs(slip) > 3.0:
        d.text((8, 62), f"slip {slip:+5.1f} deg", fill=(255, 170, 60))
    mid = 8 + bar * 0.5

    def centered_bar(top, value, color):
        """Bar filling from the centre; Pillow needs x0 <= x1, so the ends
        are sorted rather than assuming the value is positive."""
        end = mid + bar * 0.5 * float(np.clip(value, -1.0, 1.0))
        d.rectangle([8, top, 8 + bar, top + 8], outline=C_TEXT)
        d.rectangle([min(mid, end), top + 1, max(mid, end), top + 7], fill=color)

    # Throttle/brake: green for power, red for brake.
    centered_bar(36, throttle, (90, 210, 90) if throttle >= 0 else (220, 80, 60))
    centered_bar(48, steer, (90, 160, 230))
    return img


def _slip_deg(car):
    """Body slip angle: the angle between where the car points and where it
    is actually going. Large values mean it is sliding."""
    return math.degrees(math.atan2(car.v_lateral, max(car.v_forward, 0.5)))


def rollout_frames(env, policy, track, size_px, view_m, follow, every):
    """Drive one episode, yielding a frame every `every` steps."""
    obs, info = env.reset(options={"track": track})
    geom = env.geometry
    left, right = _edges(geom)
    trail, i = [], 0
    done = False
    last = info
    while not done:
        car = env.car_state
        trail.append((car.x, car.y))
        action = policy(obs)
        if i % every == 0:
            yield _draw_frame(geom, left, right,
                              (car.x, car.y, car.heading, car.v_forward,
                               float(action[1]), float(action[0]), i,
                               _slip_deg(car)),
                              trail, size_px, view_m, follow, track), None
        obs, reward, terminated, truncated, last = env.step(action)
        done = terminated or truncated
        i += 1
    car = env.car_state
    trail.append((car.x, car.y))
    # Hold the final frame briefly so the outcome is readable.
    final = _draw_frame(geom, left, right,
                        (car.x, car.y, car.heading, car.v_forward, 0.0, 0.0, i,
                         _slip_deg(car)),
                        trail, size_px, view_m, follow, track)
    for _ in range(10):
        yield final, last


def save_gif(path, frames, duration_ms, palette_colors=128):
    """Stream frames into a GIF (see the module docstring for the two tricks)."""
    it = iter(frames)
    try:
        first, _ = next(it)
    except StopIteration:
        raise RuntimeError("render produced no frames")

    # Build the palette from a frame that already contains every colour.
    # The first frame has no trail drawn yet, so an adaptive palette taken
    # from it has no slot for the trail colour and quantization snaps the
    # trail to the nearest neighbour (it came out red, the car's colour).
    # A swatch strip of the known palette guarantees each colour a slot.
    swatch = first.copy()
    sd = ImageDraw.Draw(swatch)
    for i, col in enumerate((C_BG, C_GRASS, C_ROAD, C_EDGE, C_CENTER,
                             C_CAR, C_TRAIL, C_TEXT, (90, 210, 90),
                             (220, 80, 60), (90, 160, 230), (255, 170, 60))):
        sd.rectangle([i * 4, 0, i * 4 + 3, 3], fill=col)
    pal_src = swatch.convert("P", palette=Image.Palette.ADAPTIVE,
                             colors=palette_colors, dither=Image.Dither.NONE)
    first = first.quantize(palette=pal_src, dither=Image.Dither.NONE)
    palette = list(first.getpalette() or [])
    palette += [0] * (768 - len(palette))
    corner = palette[first.getpixel((0, 0)) * 3:][:3]
    palette[254 * 3: 254 * 3 + 3] = corner
    palette[255 * 3: 255 * 3 + 3] = corner
    first.putpalette(palette)
    first.putpixel((0, 0), 254)

    info = {}

    def rest():
        flip = True
        for img, last in it:
            if last:
                info.update(last)
            q = img.quantize(palette=first, dither=Image.Dither.NONE)
            q.putpixel((0, 0), 255 if flip else 254)
            flip = not flip
            yield q

    first.save(path, save_all=True, append_images=rest(), loop=0,
               duration=duration_ms, optimize=False, disposal=1)
    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--track", default="Spa")
    p.add_argument("--all", action="store_true", help="render every circuit")
    p.add_argument("--model", default=None,
                   help="trained SB3 model (default: scripted heuristic)")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--size", type=int, default=520, help="frame size in pixels")
    p.add_argument("--view", type=float, default=140.0,
                   help="metres across the frame when following the car "
                        "(smaller = closer; at 140 m the car is ~17 px)")
    p.add_argument("--whole-track", action="store_true",
                   help="fit the whole circuit instead of following the car")
    p.add_argument("--speedup", type=int, default=2,
                   help="draw every Nth step (gif still plays in real time)")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    policy, policy_name = make_policy(args.model)
    names = track_loader.list_tracks() if args.all else [args.track]
    env = RacingEnv(track_names=names, randomize=False, seed=0)

    for name in names:
        view = env._geoms[name][(False, False)].map_size if args.whole_track \
            else args.view
        path = os.path.join(args.out, f"{name}_{policy_name}.gif")
        frames = rollout_frames(env, policy, name, args.size, view,
                                not args.whole_track, max(args.speedup, 1))
        info = save_gif(path, frames,
                        duration_ms=int(round(100 * max(args.speedup, 1))))
        end = ("lap" if info.get("lap_complete")
               else "off_map" if info.get("off_map")
               else "off_track" if info.get("off_track")
               else "stuck" if info.get("stuck") else "timeout")
        size_mb = os.path.getsize(path) / 1e6
        print(f"{name:<16}{end:<10}progress={info.get('progress_frac', 0):.2f} "
              f"offroad={info.get('offroad_frac', 0):.2f}  {size_mb:.1f} MB  -> {path}")


if __name__ == "__main__":
    main()
