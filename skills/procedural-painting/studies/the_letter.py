"""The letter: a woman in a dark dress sits by a tall window reading, the sun raking across the
floorboards, a table with a blue jug against the far wall. Set up as a 3D scene with stage.py
(room, window, chair, table, a posed figure) so the perspective, the patch of light and every
cast shadow come from the geometry; then painted from its buffers. The eye should land on the
letter, the lightest thing in the room, held against the darkest.

    python the_letter.py --out letter.png [--height 1600] [--film letter.mp4] [--study study.png] [--tape letter.tape]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
from atelier import Painting, rgb, lerp, smoothstep, blur, F32  # noqa: E402
from stage import Scene, Camera, box, capsule, cylinder, lathe, torus, plane, ellipsoid, figure, paint, masses, planks, mottle  # noqa: E402
from PIL import Image  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--out', default='the_letter.png')
ap.add_argument('--height', type=int, default=1000)
ap.add_argument('--film', default=None)
ap.add_argument('--study', default=None)
ap.add_argument('--tape', default=None)
ap.add_argument('--seed', type=int, default=4)
a = ap.parse_args()

H = a.height
W = int(H * 1.12)
p = Painting(W, H, seed=a.seed, ground='#7b6a58', film=bool(a.film), tape=bool(a.tape))
x, y = p.x, p.y

# ---- the room (metres; y up, the camera looks into +z, the window wall is on the left) -----------
WX = -2.3                                                    # inner face of the window wall
WZ = 2.25                                                    # the window's place along the wall
sc = Scene(Camera((1.55, 1.36, -3.3), (-0.50, 0.86, 1.6), fov=40),
           key=(-0.66, 0.50, 0.34), key_color=(1.0, 0.93, 0.76), sky=(0.60, 0.66, 0.78), bounce=(0.70, 0.52, 0.36),
           shadow=0.40, soft=45.0, backdrop=(0.86, 0.90, 0.92),
           fills=[((WX + 0.5, 1.7, WZ), (0.80, 0.86, 0.95), 0.50, 1.6),       # the window's cool glow on what is near it
                  ((-0.9, 0.05, 1.3), (1.0, 0.78, 0.5), 0.40, 1.4)])            # the sunlit floor throwing warm light up
sc.add('floor', plane((0, 0, 0), (0, 1, 0)), '#8a6238', grain=(1, 0, 0.12), pattern=planks((1, 0, 0.12), 0.13, 0.10))
sc.add('far wall', box((2.4, 1.7, 3.5), (4.8, 1.9, 0.2)), '#a5a79b', grain=(0, 1, 0), pattern=mottle(0.9, 0.04))
win = box((WX - 0.14, 1.52, WZ), (0.4, 1.18, 0.72))
sc.add('window wall', box((WX - 0.14, 1.6, 1.0), (0.14, 1.9, 5.0)), '#979a8e', cut=win, grain=(0, 1, 0), pattern=mottle(0.9, 0.04))
sc.add('outside', plane((WX - 0.31, 0, 0), (1, 0, 0)), '#e9ece4', kind='emit', shadow=False)
sc.add('ceiling', box((2.0, 3.45, 1.0), (4.6, 0.1, 5.0)), '#b8b8ae')
sc.add('window frame', [box((WX - 0.10, 1.52, WZ), (0.025, 1.18, 0.022)), box((WX - 0.10, 1.95, WZ), (0.025, 0.02, 0.72)),
                        box((WX - 0.10, 1.12, WZ), (0.025, 0.02, 0.72)),
                        box((WX - 0.02, 0.32, WZ), (0.10, 0.025, 0.78))], '#d9d6c8', importance=1.3)
sc.add('skirting', box((0, 0.07, 3.28), (6, 0.07, 0.02)), '#c9c6b8')
# a picture on the far wall, a table under it with a jug and a bowl
sc.add('picture frame', box((0.55, 1.78, 3.27), (0.36, 0.28, 0.02)), '#3a2c20')
sc.add('picture', box((0.55, 1.78, 3.25), (0.31, 0.23, 0.012)), '#7d8a86', pattern=mottle(0.25, 0.22))
TX, TZ = 0.6, 2.85
sc.add('table', [box((TX, 0.74, TZ), (0.55, 0.02, 0.33))] +
       [capsule((TX + sx * 0.5, 0.0, TZ + sz * 0.28), (TX + sx * 0.5, 0.73, TZ + sz * 0.28), 0.022, 0.028) for sx in (-1, 1) for sz in (-1, 1)],
       '#4a3020', grain=(1, 0, 0))
jug_prof = [(0.0, 0.055), (0.03, 0.075), (0.10, 0.095), (0.17, 0.07), (0.23, 0.042), (0.29, 0.05), (0.31, 0.056)]
sc.add('jug', [lathe((TX - 0.18, 0.76, TZ - 0.02), jug_prof), torus((TX - 0.085, 0.94, TZ - 0.02), 0.055, 0.011, rot=(90, 0, 0))],
       'cobalt blue', gloss=0.9, importance=1.5)
sc.add('bowl', lathe((TX + 0.24, 0.76, TZ - 0.06), [(0.0, 0.04), (0.02, 0.075), (0.06, 0.11), (0.075, 0.115)]), '#e4dccb', gloss=0.4)
sc.add('lemons', [ellipsoid((TX + 0.21, 0.845, TZ - 0.07), (0.045, 0.034, 0.034)), ellipsoid((TX + 0.28, 0.84, TZ - 0.03), (0.036, 0.032, 0.042))],
       'cadmium yellow', importance=1.4)
# the chair and the reader, facing the window
FX, FZ, SEAT = -0.62, 1.55, 0.45
sc.add('chair', [box((FX - 0.16, SEAT - 0.02, FZ), (0.23, 0.02, 0.22)), box((FX + 0.085, 0.80, FZ), (0.018, 0.36, 0.21), rot=(0, 0, -6))] +
       [capsule((FX - 0.16 + sx * 0.2, 0, FZ + sz * 0.19), (FX - 0.16 + sx * 0.2, SEAT - 0.03, FZ + sz * 0.19), 0.018) for sx in (-1, 1) for sz in (-1, 1)],
       '#3b2a1e', grain=(0, 1, 0))
j = figure(sc, 'reader', at=(FX, 0, FZ), facing=-90, height=1.66, pose='read', seat=SEAT, skirt=True,
           top='#d9d3c5', bottom='#23252f', skin='#d2a184', hair='#33241c', shoes='#17130f', hair_style='bun',
           head_pitch=34, lean=10)
mid = 0.5 * (j['hand_l'] + j['hand_r'])
sc.add('reader.letter', box(mid + (-0.03, 0.03, 0.0), (0.012, 0.105, 0.075), rot=(0, 0, 34)), '#f3ecd9', importance=2.0)

B = sc.render(p)
study = B.study.copy()
# the painter's adjustments to what the stage gives: a little air in the far corner, warmth where the sun lands
lit_floor = B.masks['floor'] * smoothstep(0.35, 0.8, B.light)
study = lerp(study, np.clip(study * rgb('#ffd9a0') * 1.18, 0, 1), 0.55 * lit_floor)
study = lerp(study, rgb('#aab3b2'), 0.10 * smoothstep(3.0, 6.5, B.depth))
study = np.clip(study, 0, 1).astype(F32)
if a.study:
    Image.fromarray((study * 255).astype(np.uint8)).save(a.study)

# ---- painting ----------------------------------------------------------------------------------
plan = masses(p, B, focus='reader')
for m in plan:
    if m['name'] in ('far wall', 'window wall', 'ceiling'):
        m.update(sizes=(0.07, 0.034, 0.016, 0.008), length=7, pickup=0.3, soft=0.22, jitter=0.015, hue=0.012, tol=0.03,
                 clip=0.0, overshoot=0.3, marble=0.15, dry=0.0, load=1.2)                                    # quiet walls
    if m['name'] == 'floor':
        m.update(sizes=(0.05, 0.022, 0.009, 0.0045), length=8, tol=0.045, ctol=0.1)                    # boards, and the edge of the light
paint(p, B, focus='reader', study=study, plan=plan)
# the things a stage cannot know: the brightest notes, laid on last in thick paint
p.dry(0.6)
edge = B.masks['floor'] * smoothstep(0.55, 0.95, B.light)
p.strokes(edge, study, flow=B.flow, size=0.007, length=7, density=0.5, pickup=0.1, load=1.2, dry=0.3, thickness=2.0,
          ctol=0.1, name='sun on the boards')
p.strokes(B.masks.get('outside', np.zeros_like(x)), '#f3f1e4', flow=np.pi / 2, size=0.012, length=5, pickup=0.1, load=1.2,
          thickness=1.5, jitter=0.02, clip=1.0, overshoot=0.0, name='daylight in the window')

p.save(a.out)
if a.film:
    p.film(a.film)
if a.tape:
    from tape import save_tape
    save_tape(p, a.tape, title='The letter')
