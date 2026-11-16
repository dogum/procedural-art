"""Estuary, a clearing sky: a loose watercolour.

The sky is painted into wet paper and left to do what it does; the far hills are one graded wash
on dry paper; the headland is charged with darker and warmer paint while wet; the water is a pale
wash with dry-brush sparkle; the foreground is dry brush over a varied wash.

    python estuary_in_watercolour.py --out estuary.png [--height 1600] [--film estuary.mp4]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
from aquarelle import *  # noqa: E402,F401,F403

ap = argparse.ArgumentParser()
ap.add_argument('--out', default='estuary.png')
ap.add_argument('--height', type=int, default=1600)
ap.add_argument('--film', default=None)
ap.add_argument('--seed', type=int, default=7)
a = ap.parse_args()

H = a.height
W = int(round(H * 4 / 3))
p = Paper(W, H, seed=a.seed, film=bool(a.film))
x, y = p.x, p.y
rng = np.random.default_rng(a.seed)
AS = W / H
HORIZON = 0.56

# ---- the drawing: a few masks -----------------------------------------------------------
n1, n2 = p.noise(0.3, 3), p.noise(0.08, 3)
sky = fill(y - HORIZON, 0.002)
# clouds: big soft heaps drifting up to the right, left as bare paper
heap = n1 + 0.9 * np.sin(2.4 * x - 3.0 * y + 0.6) + 0.5 * np.sin(5.2 * x + 4.0 * y) - 5.5 * (y - 0.24) ** 2 / 0.25
cloud = smoothstep(0.15, 1.0, heap + 0.25 * n2) * smoothstep(HORIZON - 0.03, HORIZON - 0.2, y)
ridge_far = HORIZON - 0.07 - 0.03 * np.sin(2.3 * x + 0.4) - 0.015 * np.sin(6.1 * x) \
    - 0.005 * np.interp(x[0], np.linspace(0, AS, 50), rng.standard_normal(50))[None, :]
hills = fill(ridge_far - y, 0.0012) * sky
ridge_near = HORIZON - 0.010 - 0.07 * smoothstep(AS * 0.42, AS * 0.95, x) * (1 + 0.2 * np.sin(9 * x)) \
    - 0.011 * np.clip(np.interp(x[0], np.linspace(0, AS, 150), rng.standard_normal(150))[None, :], -1, 1.3) * smoothstep(AS * 0.45, AS * 0.6, x)
head = fill(ridge_near - y, 0.001) * fill(y - HORIZON - 0.003, 0.0012) * smoothstep(AS * 0.40, AS * 0.47, x)
shore = 0.83 + 0.035 * np.sin(2.2 * x + 1.0) - 0.06 * smoothstep(AS * 0.55, AS, x) + 0.008 * n2
water = fill(HORIZON - y, 0.001) * fill(y - shore, 0.002)
bank = fill(shore - y, 0.002)

# ---- sky: wet the paper, float the blue round the clouds, grey their undersides ------------------
p.wet(sky, amount=0.9)
p.wash(sky * (1 - cloud), mixture({'cobalt blue': 3, 'cerulean': 2}), strength=p.grade(90, 0.30, 0.05, sky), wetness=0.8, name='sky blue')
p.wash(sky * smoothstep(0.34, HORIZON, y) * (1 - 0.5 * cloud), 'naples yellow', strength=0.16, wetness=0.5, name='warm low sky')
under = cloud * smoothstep(0.25, 0.75, cloud) * smoothstep(-0.03, 0.06, y - 0.27 - 0.07 * n1)
p.wash(under, mixture({'ultramarine': 2, 'burnt sienna': 1.5, 'alizarin crimson': 0.1}), strength=0.08, wetness=0.6, name='cloud shadows')
p.dry()

# ---- far hills: one graded wash on dry paper ---------------------------------------------------
p.wash(hills, mixture({'ultramarine': 3, 'burnt sienna': 1, 'cobalt violet': 0.5}), strength=p.grade(90, 0.26, 0.07, hills), wetness=0.65, name='far hills')
p.dry()

# ---- the water: broad dry-brush pulls that leave the paper sparkling, a pale wash through them ------
p.wash(water * smoothstep(HORIZON + 0.03, HORIZON + 0.09, y), mixture({'cerulean': 2, 'naples yellow': 1}), strength=0.035, wetness=0.5, name='water, first tint')
p.dry()
pulls = []
for i in range(60):
    yy = HORIZON + 0.012 + (0.30 * rng.random())
    cx, ln = rng.uniform(-0.1, AS + 0.1), rng.uniform(0.25, 0.7)
    pulls.append([(cx - ln, yy + rng.normal(0, 0.003)), (cx + ln, yy + rng.normal(0, 0.003))])
p.stroke(pulls, mixture({'cerulean': 2, 'cobalt blue': 1.5, 'burnt sienna': 0.25}), size=rng.uniform(0.02, 0.05, len(pulls)), strength=0.13 * water * (1 + 0.25 * n1), dry=0.5, taper=0.2,
         name='water, dragged')
calm = water * smoothstep(HORIZON + 0.10, HORIZON + 0.26, y)
p.wash(calm, mixture({'cerulean': 2, 'cobalt blue': 1.4}), strength=p.grade(90, 0.02, 0.17, calm), wetness=0.8, name='near water')
refl = water * smoothstep(AS * 0.44, AS * 0.54, x) * smoothstep(HORIZON + 0.11, HORIZON, y)
p.wet(refl, amount=0.6)
p.wash(refl, mixture({'sap green': 2, 'ultramarine': 1.5, 'burnt umber': 1}), strength=0.26, wetness=0.6, name='reflection')
p.dry()

# ---- the headland and its trees: hard top edge, wet inside ----------------------------------------
dark_green = mixture({'sap green': 3, 'ultramarine': 2, 'burnt sienna': 1})
p.wash(head, dark_green, strength=0.42 + 0.12 * n2, wetness=0.8, name='headland')
p.drop([(AS * 0.64, HORIZON - 0.03), (AS * 0.80, HORIZON - 0.05), (AS * 0.93, HORIZON - 0.03)], mixture({'prussian blue': 1, 'burnt umber': 1.2}),
       size=[0.028, 0.036, 0.03], strength=0.6, water=0.8, within=head, name='tree darks')
p.drop([(AS * 0.72, HORIZON - 0.028), (AS * 0.87, HORIZON - 0.06)], 'raw sienna', size=0.02, strength=0.55, water=1.1, within=head, name='warm in the trees')
p.dry()
ripples = []
for i in range(14):
    yy = HORIZON + 0.02 + 0.24 * rng.random() ** 0.7
    cx, ln = rng.uniform(0, AS), rng.uniform(0.03, 0.14) * (0.5 + 3.0 * (yy - HORIZON))
    if not (0.70 < yy < 0.84 and 0.2 < cx < 0.66):                     # keep the posts clear
        ripples.append([(cx - ln, yy), (cx + ln, yy + rng.normal(0, 0.0015))])
p.stroke(ripples, mixture({'cobalt blue': 2, 'paynes grey': 1}), size=rng.uniform(0.002, 0.0045, len(ripples)), strength=0.22, dry=0.6, taper=0.7,
         name='ripples')

# ---- foreground bank: a varied wash, charged while wet, then a little dry brush -----------------------
p.wash(bank, mixture({'raw sienna': 3, 'yellow ochre': 1}), strength=0.30 * (1 + 0.4 * n1), wetness=0.9, name='bank')
p.wash(bank * smoothstep(-0.1, 1.0, n2 + n1), mixture({'sap green': 2, 'yellow ochre': 1}), strength=0.34, wetness=0.7, name='grass in the bank')
low = bank * smoothstep(0.90, 1.0, y + 0.03 * n1) * smoothstep(-0.6, 0.8, n1)
p.wash(low, mixture({'burnt umber': 2, 'ultramarine': 1}), strength=0.30, wetness=0.7, name='mud darks')
p.settle(16)
p.drop([(0.40, 0.955), (0.92, 0.95)], None, size=[0.035, 0.03], water=1.2, within=bank, name='backruns')
p.dry()
blades = []
for i in range(170):
    bx = rng.uniform(0, AS) if i % 3 else rng.choice([0.12, 0.2, 0.95, 1.2]) + rng.normal(0, 0.04)
    by = min(float(np.interp(bx, x[0], shore[0])) + rng.uniform(-0.008, 0.11), 0.995)
    hgt = rng.uniform(0.025, 0.08)
    lean = rng.normal(0.3, 0.3)
    blades.append([(bx, by), (bx + 0.3 * lean * hgt, by - 0.6 * hgt), (bx + lean * hgt, by - hgt)])
p.stroke(blades[:110], mixture({'sap green': 2, 'burnt umber': 2, 'ultramarine': 1}), size=rng.uniform(0.002, 0.0045, 110), strength=0.7, dry=0.3,
         taper=0.95, name='grasses')
p.stroke(blades[110:], mixture({'raw sienna': 2, 'burnt sienna': 1}), size=rng.uniform(0.002, 0.004, len(blades) - 110), strength=0.6, dry=0.3,
         taper=0.95, name='dry stalks')

# ---- three posts in the shallows, and gulls --------------------------------------------------------
posts = [(0.36, 0.805, 0.085), (0.44, 0.79, 0.07), (0.51, 0.778, 0.055)]
p.stroke([[(px_, py_ - h_), (px_ + 0.004, py_)] for px_, py_, h_ in posts], mixture({'burnt umber': 2, 'ultramarine': 1.5}),
         size=[0.010, 0.008, 0.007], strength=0.8, taper=0.0, name='posts')
p.stroke([[(px_ + 0.004, py_ + 0.004), (px_ + 0.012, py_ + 0.55 * h_)] for px_, py_, h_ in posts], mixture({'burnt umber': 1, 'ultramarine': 2}),
         size=[0.009, 0.007, 0.006], strength=0.3, dry=0.35, taper=0.6, name='post reflections')
gulls = [[(gx - 0.014, gy - 0.004), (gx - 0.005, gy - 0.009), (gx, gy), (gx + 0.006, gy - 0.008), (gx + 0.016, gy - 0.003)]
         for gx, gy in [(0.42, 0.40), (0.485, 0.425), (0.37, 0.45)]]
p.stroke(gulls, 'paynes grey', size=0.0026, strength=0.7, taper=0.5, name='gulls')
p.dry()

p.save(a.out)
if a.film:
    p.film(a.film)
