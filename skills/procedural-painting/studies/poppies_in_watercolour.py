"""Poppies: a loose watercolour.

Each flower is one wet shape on dry paper, charged with darker red at the base and touched with
clean water so it backruns; when dry, a second transparent layer draws the overlapping petals.
The background is floated in round the flowers on wet paper.

    python poppies_in_watercolour.py --out poppies.png [--height 1600] [--film poppies.mp4]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
from aquarelle import *  # noqa: E402,F401,F403

ap = argparse.ArgumentParser()
ap.add_argument('--out', default='poppies.png')
ap.add_argument('--height', type=int, default=1600)
ap.add_argument('--film', default=None)
ap.add_argument('--seed', type=int, default=11)
a = ap.parse_args()

H = a.height
W = int(round(H * 3 / 4))
p = Paper(W, H, seed=a.seed, film=bool(a.film))
x, y = p.x, p.y
rng = np.random.default_rng(a.seed)
n1, n2 = p.noise(0.22, 3), p.noise(0.05, 3)

# ---- the drawing ---------------------------------------------------------------------------
# flowers: centre, size, how far the cup is tipped (0 = seen from the side, 1 = open to us)
FLOWERS = [(0.27, 0.27, 0.125, 0.75), (0.53, 0.42, 0.105, 0.35), (0.20, 0.56, 0.085, 0.2)]
BUDS = [(0.60, 0.20, 0.022), (0.43, 0.14, 0.018)]
BASE = (0.40, 1.02)


def petals(cx, cy, r, tip, k):
    """The petal shapes of one poppy: a list of masks, back to front."""
    out = []
    n = 5
    for i in range(n):
        ang = -np.pi / 2 + (i - (n - 1) / 2) * (1.15 + 0.25 * tip) + 0.25 * np.sin(k + i)
        px_, py_ = cx + 0.52 * r * np.cos(ang), cy + (0.30 + 0.25 * tip) * r * np.sin(ang) - 0.12 * r
        rx, ry = r * (0.62 + 0.1 * np.sin(2.1 * i + k)), r * (0.50 + 0.22 * tip)
        d = sd_ellipse(x, y, px_, py_, rx, ry, ang + np.pi / 2) + 0.10 * r * n2 + 0.05 * r * n1
        out.append(fill(d, 0.0012))
    return out


flower_petals = [petals(cx, cy, r, tip, 1.7 * i) for i, (cx, cy, r, tip) in enumerate(FLOWERS)]
flower = [np.clip(sum(ps), 0, 1) for ps in flower_petals]
all_flowers = np.clip(sum(flower), 0, 1)


def stem(x0, y0, x1, y1, bend):
    mx, my = 0.5 * (x0 + x1) + bend, 0.5 * (y0 + y1)
    return [tuple(q) for q in curve([(x0, y0), (mx, my), (x1, y1)], 14)]


stems = [stem(cx + 0.01, cy + (0.62 + 0.3 * tip) * r, BASE[0] + 0.03 * (i - 1), BASE[1], 0.07 * (1 - i)) for i, (cx, cy, r, tip) in enumerate(FLOWERS)]
stems += [stem(bx, by + r, BASE[0] + 0.02, BASE[1], 0.10 - 0.16 * j) for j, (bx, by, r) in enumerate(BUDS)]

# ---- the background, floated in round the flowers on wet paper -------------------------------------
halo = blur(all_flowers, 0.012 * H)
bg = np.clip(1 - 1.6 * halo, 0, 1)
p.wet(bg, amount=0.85)
p.wash(bg * smoothstep(-0.4, 0.9, n1 + 1.6 * (y - 0.45)), mixture({'sap green': 2, 'yellow ochre': 1, 'cerulean': 0.6}), strength=0.22, wetness=0.8,
       name='green ground')
p.wash(bg * smoothstep(0.2, 1.2, -n1 + 1.2 * (0.5 - y)), mixture({'cerulean': 2, 'cobalt violet': 0.5}), strength=0.12, wetness=0.8, name='cool air')
p.wash(bg * smoothstep(0.5, 1.3, n1 + 0.6 * n2 + 0.8 * (x - 0.3)), 'naples yellow', strength=0.25, wetness=0.7, name='warm light')
p.drop([(0.62, 0.72), (0.12, 0.83), (0.66, 0.93)], mixture({'sap green': 2, 'ultramarine': 1}), size=[0.07, 0.06, 0.08], strength=0.34, water=0.9,
       within=bg, name='deep greens')
p.settle(14)
p.drop([(0.66, 0.60), (0.10, 0.70)], None, size=[0.04, 0.035], water=1.2, within=bg, name='backruns')
p.dry()

# ---- stems and leaves, a loaded brush on dry paper ----------------------------------------------------
green = mixture({'sap green': 3, 'ultramarine': 1, 'burnt sienna': 0.6})
p.stroke(stems, green, size=[0.009, 0.008, 0.008, 0.006, 0.006], strength=0.6, taper=0.1, wetness=0.25, name='stems')
leaves = []
for (lx, ly, dx, dy) in [(0.39, 0.80, -0.16, -0.07), (0.41, 0.86, 0.17, -0.10), (0.40, 0.92, -0.20, -0.02), (0.42, 0.74, 0.12, -0.12)]:
    leaves.append([(lx, ly), (lx + 0.5 * dx, ly + 0.8 * dy - 0.01), (lx + dx, ly + dy)])
p.stroke(leaves, green, size=[0.034, 0.03, 0.036, 0.026], strength=0.5, taper=0.95, wetness=0.7, name='leaves')
p.drop([(0.32, 0.775), (0.49, 0.80), (0.31, 0.905)], mixture({'prussian blue': 1, 'sap green': 1}), size=0.014, strength=0.7, water=0.8, name='leaf darks')
p.dry()

# ---- the flowers: one wet shape each, charged and left to run ---------------------------------------------
red = mixture({'cadmium red': 3, 'vermilion': 2})
for i, (cx, cy, r, tip) in enumerate(FLOWERS):
    m = flower[i]
    p.wash(m, red, strength=0.40 * (1 + 0.35 * n1 + 0.2 * n2), wetness=0.9, name=f'poppy {i + 1}')
    p.wash(m * smoothstep(0.75 * r, 0.1 * r, np.hypot(x - cx, y - cy - 0.15 * r)), 'alizarin crimson', strength=0.55, wetness=0.6, name='  deeper at the base')
    p.drop([(cx - 0.45 * r, cy - 0.45 * r)], 'cadmium orange', size=0.3 * r, strength=0.5, water=1.0, within=m, name='  orange where the light comes through')
    p.settle(12)
    p.drop([(cx + 0.5 * r, cy - 0.35 * r)], None, size=0.22 * r, water=1.2, within=m, name='  a backrun')
p.dry()
# the petals in front: a second, transparent layer with hard edges
for i, (cx, cy, r, tip) in enumerate(FLOWERS):
    front = np.clip(flower_petals[i][1] + flower_petals[i][3], 0, 1) * flower[i]
    p.wash(front, mixture({'alizarin crimson': 2, 'cadmium red': 1}), strength=0.30 * smoothstep(-0.9 * r, 0.4 * r, y - cy), wetness=0.6,
           name='petal overlaps')
p.dry()
# the dark hearts and stamens
for i, (cx, cy, r, tip) in enumerate(FLOWERS):
    hx, hy = cx + 0.02 * r, cy + 0.12 * r - 0.1 * r * tip
    heart = fill(sd_ellipse(x, y, hx, hy, 0.20 * r, 0.20 * r * (0.35 + 0.65 * tip)) + 0.03 * r * n2, 0.001)
    p.wash(heart, mixture({'dioxazine violet': 1, 'ivory black': 1}), strength=0.75, wetness=0.5, name='heart')
    spokes = [[(hx, hy), (hx + 0.34 * r * np.cos(t), hy + 0.34 * r * (0.35 + 0.65 * tip) * np.sin(t))] for t in rng.uniform(0, 2 * np.pi, 16)]
    p.stroke(spokes, mixture({'dioxazine violet': 1, 'ivory black': 1}), size=0.0022, strength=0.7, dry=0.3, taper=0.8, name='stamens')
# buds
for bx, by, r in BUDS:
    bud = fill(sd_ellipse(x, y, bx, by, 0.75 * r, 1.25 * r, 0.3), 0.001)
    p.wash(bud, green, strength=0.5, wetness=0.6, name='bud')
    p.drop([(bx + 0.2 * r, by - 0.5 * r)], red, size=0.5 * r, strength=0.6, water=0.9, within=bud, name='  red showing')
p.dry()

# a few dry-brush grasses and spatters to finish
grass = []
for i in range(24):
    gx = rng.uniform(0.02, 0.73)
    gy = rng.uniform(0.80, 1.0)
    h_ = rng.uniform(0.08, 0.26)
    l_ = rng.normal(0.0, 0.3)
    grass.append([(gx, gy), (gx + 0.4 * l_ * h_, gy - 0.6 * h_), (gx + l_ * h_ * 1.4, gy - h_)])
p.stroke(grass, mixture({'sap green': 2, 'burnt umber': 1}), size=rng.uniform(0.002, 0.005, len(grass)), strength=0.5, dry=0.45, taper=0.95, name='grasses')
spat = np.stack([rng.uniform(0.05, 0.7, 26), rng.uniform(0.05, 0.75, 26)], -1)
p.drop(spat, red, size=rng.uniform(0.0015, 0.005, len(spat)), strength=0.45, water=0.4, ragged=0.2, name='spatter')
p.dry()

p.save(a.out)
if a.film:
    p.film(a.film)
