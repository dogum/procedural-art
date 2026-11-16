"""The white cup: a woman at a table by a window, half length, a white cup lifted in one hand, the
other hand resting on the table. A test of the stage's figure up close: the head turned three
quarters with the planes of brow, nose and jaw in side light, a hand that holds something, and
sleeves that fold. Everything is set up in 3D with stage.py and painted from its buffers; the
painter then restates the few accents a stage cannot rank (the eyes, the mouth, the cup's rim).

    python the_white_cup.py --out cup.png [--height 1600] [--film cup.mp4] [--study study.png] [--tape cup.tape]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
from atelier import Painting, rgb, lerp, smoothstep, blur, F32  # noqa: E402
from stage import Scene, Camera, box, capsule, cylinder, lathe, torus, plane, ellipsoid, figure, paint, masses, mottle  # noqa: E402
from PIL import Image  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--out', default='the_white_cup.png')
ap.add_argument('--height', type=int, default=1000)
ap.add_argument('--film', default=None)
ap.add_argument('--study', default=None)
ap.add_argument('--tape', default=None)
ap.add_argument('--seed', type=int, default=7)
a = ap.parse_args()

H = a.height
W = int(H * 0.82)
p = Painting(W, H, seed=a.seed, ground='#6b5a4a', film=bool(a.film), tape=bool(a.tape))

# ---- the sitter (metres; the camera looks into +z; the window is off to the left) ----------------
SEAT = 0.46
POSE = dict(sit=True, lean=7, hip_l=84, hip_r=84, knee_l=86, knee_r=86,
            shoulder_l=16, elbow_l=73, abduct_l=13, palm_l=78, grip_l=14,          # a forearm lying on the table
            shoulder_r=34, elbow_r=112, abduct_r=10, grip_r=66, wrist_r=8,          # the cup lifted
            head_yaw=-22, head_pitch=13, head_tilt=-3)


def sitter(scene):
    return figure(scene, 'sitter', at=(0.0, 0, 0.55), facing=198, height=1.66, pose=POSE, seat=SEAT, skirt=True,
                  top='#b9793a', bottom='#2b2f3a', skin='#d6a585', hair='#2e2019', hair_style='bob', importance=1.6)


cam = Camera((0.42, 1.19, -1.02), (0.0, 1.02, 0.42), fov=40)
j = sitter(Scene(cam))                                        # a first pass, only to learn where the hands come
TOP = float(j['wrist_l'][1]) - 0.030                          # the table top is where the resting hand finds it

sc = Scene(cam, key=(-0.74, 0.42, -0.40), key_color=(1.0, 0.95, 0.84), sky=(0.52, 0.60, 0.74), bounce=(0.66, 0.50, 0.38),
           shadow=0.34, soft=14.0, backdrop=(0.2, 0.22, 0.22),
           fills=[((0.9, 1.3, -0.6), (0.70, 0.66, 0.60), 0.16, 1.6)])            # the room's own light on the shadow side
sc.add('wall', box((0, 1.5, 1.75), (4, 2.2, 0.1)), '#51605c', grain=(0, 1, 0), pattern=mottle(0.7, 0.06))
sc.add('floor', plane((0, 0, 0), (0, 1, 0)), '#5a4630')
sc.add('picture frame', box((0.78, 1.52, 1.64), (0.30, 0.24, 0.02)), '#3a2a1c')
sc.add('picture', box((0.78, 1.52, 1.62), (0.25, 0.19, 0.012)), '#9a8a6a', pattern=mottle(0.2, 0.25))
sc.add('chair', [capsule((0.06 + sx_ * 0.17, 0.5, 0.80 - sx_ * 0.05), (0.06 + sx_ * 0.17, 1.10, 0.84 - sx_ * 0.05), 0.016) for sx_ in (-1, 1)] +
       [box((0.06, 1.06, 0.835), (0.17, 0.035, 0.012), rot=(0, 18, 0)), box((0.06, 0.90, 0.825), (0.17, 0.025, 0.012), rot=(0, 18, 0))],
       '#3a2a1e', grain=(0, 1, 0))
sc.add('table', box((0.0, TOP - 0.02, 0.02), (0.95, 0.02, 0.36), rot=(0, 18, 0)), '#7a5234', grain=(1, 0, -0.3),
       pattern=mottle(0.5, 0.07))
j = sitter(sc)
# the cup in the lifted hand, a saucer and a spoon left on the table, two pears
g = j['grip_r']
cup_prof = [(0.0, 0.020), (0.008, 0.027), (0.035, 0.036), (0.066, 0.040), (0.072, 0.041)]
cup_at = g + j['palm_r'] * 0.030 - np.array([0, 0.034, 0], F32)
sc.add('cup', [lathe(cup_at, cup_prof), torus(cup_at + j['thumb_r'] * 0.004 - j['palm_r'] * 0.046 + (0, 0.040, 0), 0.017, 0.0045,
                                              rot=np.stack([np.cross(j['palm_r'], (0, 1, 0)), j['palm_r'], (0, 1, 0)], 1))],
       '#efe9dc', gloss=0.7, importance=2.2)
sc.add('coffee', cylinder(cup_at + (0, 0.0695, 0), 0.004, 0.0345), '#3a2416', importance=1.5)
sx, sz = -0.02, -0.14
sc.add('saucer', lathe((sx, TOP, sz), [(0.0, 0.040), (0.006, 0.060), (0.012, 0.074)]), '#e9e2d3', gloss=0.6, importance=1.6)
sc.add('pears', [ellipsoid((0.33, TOP + 0.046, -0.04), (0.040, 0.046, 0.040)), ellipsoid((0.335, TOP + 0.092, -0.035), (0.022, 0.034, 0.022)),
                 ellipsoid((0.43, TOP + 0.040, 0.05), (0.044, 0.040, 0.056), rot=(0, 30, 12)),
                 ellipsoid((0.475, TOP + 0.046, 0.075), (0.024, 0.022, 0.034), rot=(0, 30, 12))], '#a8a04a', k=0.02, importance=1.4)

B = sc.render(p, res=720)
study = B.study.copy()
# the painter's adjustments: the wall falls away from the light, warmth in the lit flesh, a breath of
# red in the cheek and the ear
wall = B.masks.get('wall', np.zeros((p.H, p.W), F32))
study = lerp(study, study * rgb('#8f9a94'), 0.55 * wall * smoothstep(0.25, 0.95, p.x / (W / H)))
flesh = np.maximum(B.masks.get('sitter.skin', 0), np.maximum(B.masks.get('sitter.skin.hand_l', 0), B.masks.get('sitter.skin.hand_r', 0)))
study = lerp(study, np.clip(study * rgb('#ffd8c4') * 1.10, 0, 1), 0.5 * flesh * smoothstep(0.3, 0.8, B.light))
hx, hy = B.project(j['head'] + j['head_dir'] * 0.05 - np.array([0, 0.035, 0], F32))
cheek = np.exp(-(((p.x - hx) / 0.035) ** 2 + ((p.y - hy) / 0.03) ** 2)) * B.masks.get('sitter.skin', 0)
study = lerp(study, study * rgb('#ffb6a0'), 0.35 * cheek)
study = np.clip(study, 0, 1).astype(F32)
if a.study:
    Image.fromarray((study * 255).astype(np.uint8)).save(a.study)

# ---- painting ----------------------------------------------------------------------------------
plan = masses(p, B, focus='sitter')
for m in plan:
    if m['name'] in ('wall', 'floor'):
        m.update(sizes=(0.08, 0.04, 0.018), length=6, pickup=0.35, soft=0.25, jitter=0.03, hue=0.02, tol=0.035,
                 clip=0.0, overshoot=0.3)                                                               # a quiet ground
    if m['name'] == 'table':
        m.update(sizes=(0.05, 0.022, 0.010), length=8, tol=0.04)
paint(p, B, focus='sitter', study=study, plan=plan)

p.save(a.out)
if a.film:
    p.film(a.film)
if a.tape:
    from tape import save_tape
    save_tape(p, a.tape, title='The white cup')
