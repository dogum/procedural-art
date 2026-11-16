"""Pomegranates, from the scene: the still life that procedural-realism path-traces
(scenes/pomegranate.json: two pomegranates, apricots, a copper ball, a glass of juice and a
crystal ball on walnut, window light from the left) set up on the stage from the same file and
painted. The photograph and the painting share one description of the world; the painting gets
its drawing, cast shadows and the reflection in the copper from the geometry, and its colour
from a painter's light and shadow families. The eye should land on the big pomegranate.

    python pomegranates_from_the_scene.py --out pom.png [--height 1600] [--film pom.mp4] [--study s.png] [--tape pom.tape]
                                          [--scene path/to/scene.json]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'scripts'))
from atelier import Painting, rgb, lerp, smoothstep, blur, F32  # noqa: E402
from stage import load_pathtracer_scene, paint, masses  # noqa: E402
from PIL import Image  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--out', default='pomegranates.png')
ap.add_argument('--height', type=int, default=1000)
ap.add_argument('--film', default=None)
ap.add_argument('--study', default=None)
ap.add_argument('--tape', default=None)
ap.add_argument('--seed', type=int, default=6)
ap.add_argument('--scene', default=None, help='a procedural-realism path-tracer scene (default: its pomegranate.json)')
a = ap.parse_args()

scene_file = a.scene
if scene_file is None:
    for d in (os.path.join(HERE, '..', '..', 'procedural-realism', 'scenes'), os.environ.get('PT_SCENES', ''), HERE):
        if d and os.path.exists(os.path.join(d, 'pomegranate.json')):
            scene_file = os.path.join(d, 'pomegranate.json')
            break
    if scene_file is None:
        raise SystemExit("pomegranate.json not found: pass --scene FILE (it ships with the procedural-realism skill)")

ASPECT = 2.0
H = a.height
W = int(H * ASPECT)
p = Painting(W, H, seed=a.seed, ground='#5a3d2a', film=bool(a.film), tape=bool(a.tape))
x, y = p.x, p.y

sc = load_pathtracer_scene(scene_file, aspect=ASPECT, shadow=0.34)
sc.bounce = np.array([0.75, 0.42, 0.26], F32)                # light coming back up off the walnut
sc.things[1].color = rgb('#6f6450')                          # the wall: a painter's dark, warm ground
for th in sc.things:
    if th.name.startswith('juice'):
        th.color, th.gloss = rgb('#5e0f1e'), 0.5             # the juice: deeper and cooler than the fruit, so the glass reads
B = sc.render(p)

# the painter's adjustments: the wall falls away into shadow on the right, the table glows where the light lands
study = B.study.copy()
wall = B.masks.get('wall', np.zeros_like(x))
study = lerp(study, study * rgb('#5a5648') * 1.25, wall * smoothstep(0.3, 1.8, x))
study = lerp(study, np.clip(study * 1.18 * rgb('#ffe2b8'), 0, 1), B.masks['table'] * smoothstep(0.4, 0.9, B.light) * 0.6)
study = np.clip(study, 0, 1).astype(F32)
if a.study:
    Image.fromarray((study * 255).astype(np.uint8)).save(a.study)

plan = masses(p, B, focus='hero_pomegranate')
for m in plan:
    if m['name'] == 'wall':
        m.update(sizes=(0.08, 0.04, 0.018), pickup=0.45, soft=0.22, jitter=0.04, hue=0.03, tol=0.045, flow=0.5 + 0.3 * p.noise(0.6, 2, seed=3))
    if m['name'] == 'table':
        m.update(sizes=(0.06, 0.026, 0.011, 0.005), length=8, tol=0.04, ctol=0.1, jitter=0.05)
    if m['name'].startswith(('hero', 'second', 'apricot', 'copper', 'crystal')):
        m.update(pickup=0.45, soft=0.14)                     # round things: let the turn of the form melt
paint(p, B, focus='hero_pomegranate', study=study, plan=plan)

p.save(a.out)
if a.film:
    p.film(a.film)
if a.tape:
    from tape import save_tape
    save_tape(p, a.tape, title='Pomegranates, from the scene')
