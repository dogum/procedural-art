"""Rain at blue hour: a wet street between two rows of houses, lamps coming on, two people
under a red umbrella. Painted from imagination; nothing is traced or sampled from an image.

    python rain_at_blue_hour.py --out rain.png [--height 1600] [--film rain.mp4] [--study study.png]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
from atelier import (Painting, rgb, lerp, mix, smoothstep, fill, blur, sd_box, sd_circle, sd_ellipse,  # noqa: E402
                     sd_segment, sd_poly, along, F32)
from PIL import Image  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--out', default='rain_at_blue_hour.png')
ap.add_argument('--height', type=int, default=1000)
ap.add_argument('--film', default=None)
ap.add_argument('--study', default=None, help='also save the flat colour study the painter works from')
ap.add_argument('--seed', type=int, default=11)
a = ap.parse_args()

H = a.height
W = int(H * 1.5)
p = Painting(W, H, seed=a.seed, ground='#6b4a3a', film=bool(a.film))
x, y = p.x, p.y
rng = np.random.default_rng(a.seed)
HOR, VX = 0.50, 0.60                        # horizon and vanishing point


def screen(base, light, colour):
    """Add light the way a lamp does: toward the colour, never past white."""
    return 1 - (1 - base) * (1 - np.clip(light, 0, 1)[..., None] * rgb(colour))


# ---- the colour study ------------------------------------------------------------------
# sky: ultramarine overhead, a band of violet, the last peach light sitting on the roofs
t = smoothstep(0.0, HOR, y)
sky = lerp(rgb('#1b2a5e'), rgb('#3c3f86'), smoothstep(0.0, 0.55, t))
sky = lerp(sky, rgb('#8a5f8f'), smoothstep(0.45, 0.85, t))
sky = lerp(sky, rgb('#f0a868'), smoothstep(0.72, 1.0, t) * np.exp(-((x - VX) / 0.55) ** 2))
cloud = smoothstep(0.1, 0.9, p.noise(0.35, 4, seed=1) * 0.6 + p.noise(0.08, 2, seed=2) * 0.2 + (0.55 - t) * 0.8)
sky = lerp(sky, rgb('#2a2f5c'), 0.55 * cloud)

# houses: two walls running to the vanishing point, roofline broken into separate fronts
def roofline(x0, y0, steps, seed):
    """Height of the roof at each x for a wall whose eave runs from (x0,y0) to the vanishing point."""
    r = np.random.default_rng(seed)
    tt = np.clip((x - x0) / (VX - x0), 0, 1)                     # 0 near, 1 at the vanishing point
    base = y0 + (HOR - y0) * tt
    edges = np.sort(r.uniform(0, 1, steps)) ** 0.75
    k = np.searchsorted(edges, tt)
    bump = r.uniform(-0.055, 0.04, steps + 1)[k] * (1 - tt)
    chim = (np.abs(((tt * 23 + r.uniform()) % 1) - 0.5) < 0.06) * r.uniform(0, 1, steps + 1)[k] > 0.55
    return base + bump - 0.022 * chim * (1 - tt), tt, k, base + bump

roofL, tL, kL, eaveL = roofline(0.0, 0.06, 7, 3)
roofR, tR, kR, eaveR = roofline(1.5, 0.15, 6, 5)
ground_y = HOR + (0.585 - HOR) * 0 + 0.0 * x
curbL = HOR + (0.74 - HOR) * (1 - tL)                              # foot of the left wall
curbR = HOR + (0.70 - HOR) * (1 - tR)
wallL = (x < VX - 0.012) & (y > roofL) & (y < curbL)
wallR = (x > VX + 0.03) & (y > roofR) & (y < curbR)
tonesL = np.random.default_rng(7).uniform(0, 1, 9)[kL]
tonesR = np.random.default_rng(8).uniform(0, 1, 9)[kR]
facL = lerp(rgb('#232a52'), rgb('#3a3560'), tonesL)
facR = lerp(rgb('#1c2248'), rgb('#34305a'), tonesR)
# atmospheric perspective: far fronts lose contrast and go toward the sky's violet
facL = lerp(facL, rgb('#6f5a86'), 0.75 * smoothstep(0.3, 1.0, tL))
facR = lerp(facR, rgb('#6f5a86'), 0.75 * smoothstep(0.3, 1.0, tR))


def windows(tt, roof, curb, seed, lit=0.42):
    """Rows of windows in perspective: returns (frame mask, lit mask, warmth 0..1)."""
    r = np.random.default_rng(seed)
    hgt = np.maximum(curb - roof, 1e-3)
    v = (y - roof) / hgt                                           # 0 at the eave, 1 at the pavement
    cols = -np.log(np.maximum(1 - tt * 0.985, 1e-3)) * 9.0          # even spacing in depth
    rows = v * 4.6 - 0.35
    ci, ri = np.floor(cols), np.floor(rows)
    fc, fr = cols - ci, rows - ri
    inside = (np.abs(fc - 0.5) < 0.2) & (np.abs(fr - 0.45) < 0.3) & (ri >= 0) & (ri < 3)
    hsh = np.sin(ci * 127.1 + ri * 311.7 + seed * 17.0) * 43758.5453
    hsh = hsh - np.floor(hsh)
    on = inside & (hsh < lit)
    return inside, on, (hsh / lit) % 1.0

study = sky.copy()
study = np.where(wallL[..., None], facL, study)
study = np.where(wallR[..., None], facR, study)
glow = np.zeros_like(x)                                             # warm light in the air
for wall, tt, roof, curb, sd in ((wallL, tL, eaveL, curbL, 21), (wallR, tR, eaveR, curbR, 22)):
    ins, on, warm = windows(tt, roof, curb, sd)
    dark = wall & ins & ~on
    study = np.where(dark[..., None], study * 0.72 + 0.28 * rgb('#3d4f86'), study)   # glass holding the sky
    wcol = lerp(rgb('#f08a2c'), rgb('#ffd98a'), warm)
    lit = wall & on
    study = np.where(lit[..., None], lerp(wcol, study, 0.55 * smoothstep(0.55, 1.0, tt)), study)
    glow += blur(lit.astype(F32), 0.012 * H) * 0.9

# shop fronts: a band of orange light along the foot of each wall
shopL = wallL & (y > curbL - 0.36 * (curbL - roofL)) & (tL < 0.93)
shopR = wallR & (y > curbR - 0.30 * (curbR - roofR)) & (tR < 0.9)
bays = lambda tt, n: (np.abs(((-np.log(np.maximum(1 - tt * 0.985, 1e-3)) * n) % 1) - 0.5) < 0.36)
shop_lit_L = shopL & bays(tL, 4.0) & (np.sin(np.floor(-np.log(np.maximum(1 - tL * 0.985, 1e-3)) * 4.0) * 2.4) > -0.55)
shop_lit_R = shopR & bays(tR, 3.2) & (np.sin(np.floor(-np.log(np.maximum(1 - tR * 0.985, 1e-3)) * 3.2) * 1.9 + 1) > -0.2)
for m_, tt, c0, c1 in ((shop_lit_L, tL, '#ffc163', '#f2741e'), (shop_lit_R, tR, '#ffd28a', '#ef8a2e')):
    v = p.noise(0.05, 2, seed=31)
    curb, roof = (curbL, roofL) if m_ is shop_lit_L else (curbR, roofR)
    up = np.clip((curb - y) / (0.3 * (curb - roof)), 0, 1)             # 0 at the pavement, 1 at the awning
    shop = lerp(rgb(c1), rgb(c0), smoothstep(-0.8, 0.8, v + 1.2 * up - 0.5))
    shop = lerp(rgb('#7a3a22'), shop, smoothstep(0.0, 0.22, up))        # stall riser under the glass
    cc = -np.log(np.maximum(1 - tt * 0.985, 1e-3)) * (4.0 if m_ is shop_lit_L else 3.2)
    mull = (np.abs(((cc * 3) % 1) - 0.5) > 0.44) | ((np.sin(cc * 37.0) > 0.86) & (up < 0.7) & (up > 0.1))
    shop = np.where(mull[..., None], shop * 0.38, shop)                 # glazing bars, things in the window
    study = np.where(m_[..., None], lerp(shop, study, 0.6 * smoothstep(0.6, 1.0, tt)), study)
    glow += blur(m_.astype(F32), 0.03 * H) * 1.1
# awnings: dark stripes above the lit bays
aw = (shopL & (y < curbL - 0.30 * (curbL - roofL))) | (shopR & (y < curbR - 0.25 * (curbR - roofR)))
study = np.where(aw[..., None], rgb('#3a1f33'), study)

# street lamps down the left pavement and three on the right
lamps = []
for tt_ in (0.08, 0.40, 0.62, 0.76, 0.86, 0.92):
    lamps.append((0.33 + (VX - 0.33) * tt_, 0.175 + (HOR - 0.175) * tt_, HOR + (0.80 - HOR) * (1 - tt_), 1 - tt_))
for tt_ in (0.05, 0.68, 0.86):
    lamps.append((1.16 + (VX - 1.16) * tt_, 0.20 + (HOR - 0.20) * tt_, HOR + (0.76 - HOR) * (1 - tt_), 1 - tt_))
post = np.zeros_like(x, bool)
core = np.zeros_like(x)
halo = np.zeros_like(x)
for lx, ly, lb, s in lamps:
    post |= sd_segment(x, y, lx, ly, lx, lb, 0.0035 * s + 0.001) < 0
    post |= sd_box(x, y, lx, ly + 0.012 * s, 0.014 * s, 0.006 * s) < 0
    d = np.hypot(x - lx, (y - ly) * 1.05)
    core = np.maximum(core, smoothstep(0.022 * s, 0.008 * s, d))
    halo += 0.85 * np.exp(-d / (0.045 * s + 0.004)) + 0.35 * np.exp(-d / (0.16 * s + 0.01))

# the street: a mirror of everything above it, dragged down and broken by the rain
above = study.copy()
above = screen(above, 0.55 * glow + 0.8 * halo, '#ffb75e')
above = np.where(post[..., None] & (core[..., None] < 0.5), rgb('#15152c'), above)
above = lerp(above, rgb('#fff3cf'), core)
street_m = (y > HOR) & ~wallL & ~wallR
yy = np.clip(((2 * HOR - y) * H).astype(int), 0, H - 1)            # mirror row
xi = np.broadcast_to(np.arange(W), (H, W))
ripple = (p.noise(0.012, 2, seed=41) * 0.006 * H * smoothstep(HOR, 1.0, y) * 2.2).astype(int)
refl = above[yy, np.clip(xi + ripple, 0, W - 1)]
import scipy.ndimage as ndi  # noqa: E402
refl = ndi.uniform_filter1d(refl, max(3, int(0.09 * H)), axis=0, mode='nearest')    # long vertical drags
depth = smoothstep(HOR, 1.0, y)
wet = lerp(rgb('#262b55'), rgb('#141a3a'), depth)                   # the asphalt itself
road = lerp(wet, refl, 0.80 - 0.18 * depth)
# puddle sheen: bands where the mirror is cleaner, bands where the rough road shows
bands = smoothstep(-0.2, 0.9, p.noise(0.25, 3, seed=42) + 0.6 * np.sin(y * 70 + 4 * p.noise(0.3, 2, seed=43)))
road = lerp(road * 0.62, road * 1.12 + 0.02, bands)
# cool sky light down the middle of the street
skyrefl = np.exp(-((x - VX - 0.05 * (y - HOR)) / (0.10 + 0.9 * (y - HOR))) ** 2) * (1 - 0.5 * depth)
road = lerp(road, rgb('#8f79a8'), 0.28 * skyrefl * bands)
# lamp and shop reflections: long warm streaks straight down from each light
streak = np.zeros_like(x)
for lx, ly, lb, s in lamps:
    wdt = 0.012 * s + 0.004 + 0.02 * (y - lb) * s
    fall = np.exp(-np.maximum(y - lb, 0) / (0.55 * s + 0.05)) * (y > lb - 0.004)
    streak += np.exp(-((x - lx) / wdt) ** 2) * fall * (0.55 + 0.6 * s)
brk = smoothstep(-0.9, 0.5, p.noise(0.006, 1, seed=44) * 0.9 + np.sin(y * 260 + 9 * p.noise(0.2, 2, seed=45)))
road = screen(road, streak * (0.35 + 0.65 * brk), '#ffbe6a')
road = lerp(road, rgb('#fff0c4'), np.clip(streak - 0.75, 0, 1) * brk * 0.8)
kerb = street_m & ((np.abs(y - curbL) < 0.004 * (1 - tL) + 0.001) & (x < VX) | (np.abs(y - curbR) < 0.004 * (1 - tR) + 0.001) & (x > VX))
road = np.where(kerb[..., None], road * 0.5, road)
study = np.where(street_m[..., None], road, above)

# two people under a red umbrella, walking away, a third far off
def figure(cx, foot, h, lean=0.0, hem=0.30, flare=0.115):
    """A standing figure seen from behind: long coat to the hem, two legs, head."""
    sh, head, hm = foot - 0.82 * h, foot - 0.91 * h, foot - hem * h
    d = sd_poly(x, y, [(cx - 0.095 * h + lean * h, sh), (cx + 0.095 * h + lean * h, sh),
                       (cx + flare * h, hm), (cx - flare * h, hm)]) - 0.012 * h
    d = np.minimum(d, sd_segment(x, y, cx - 0.05 * h, hm, cx - 0.055 * h, foot, 0.036 * h, 0.026 * h))
    d = np.minimum(d, sd_segment(x, y, cx + 0.05 * h, hm, cx + 0.07 * h, foot - 0.012 * h, 0.036 * h, 0.026 * h))
    d = np.minimum(d, sd_circle(x, y, cx + lean * h, head, 0.055 * h))
    return d

FX, FOOT, FH = 0.93, 0.845, 0.30
dA = figure(FX - 0.05, FOOT, FH, 0.015)
dB = figure(FX + 0.058, FOOT + 0.004, FH * 0.92, -0.015, hem=0.2, flare=0.14)
people = np.minimum(dA, dB)
ucx, ucy = FX + 0.005, FOOT - 0.93 * FH
dome = np.maximum(sd_ellipse(x, y, ucx, ucy, 0.135, 0.075), (y - ucy) - 0.012)   # upper half of an ellipse
scallop = 0.008 * np.abs(np.sin((x - ucx) * 60))
dome = np.maximum(dome, (y - ucy) - 0.004 - scallop)
far_fig = figure(0.70, 0.60, 0.085)
ribs = np.abs(np.sin(np.arctan2(y - (ucy + 0.02), x - ucx) * 5.0))
umb = lerp(rgb('#8f1420'), rgb('#e23a2a'), smoothstep(0.1, -0.06, (y - ucy) + 0.4 * (x - ucx) + 0.05))
umb = lerp(umb, rgb('#ff8a4a'), 0.5 * smoothstep(0.03, -0.05, sd_ellipse(x, y, ucx - 0.05, ucy - 0.045, 0.05, 0.02)))
umb = umb * (0.86 + 0.14 * ribs[..., None])
coat = lerp(rgb('#0e1024'), rgb('#3b2140'), smoothstep(-0.012, 0.012, x - FX - 0.004))    # his navy, her plum
pm0 = fill(people, 0.003)
rim = pm0 * (1 - np.roll(pm0, int(0.005 * H), axis=1)) * (y < FOOT - 0.3 * FH)
coat = lerp(coat, rgb('#d98a4c'), 0.5 * rim)                         # lamp light catching a shoulder
# their reflections
fy = np.clip(((2 * FOOT - y) * H).astype(int), 0, H - 1)
pm, um = fill(people, 0.003), fill(dome, 0.003)
rp = ndi.uniform_filter1d(pm[fy, np.clip(xi + ripple * 2, 0, W - 1)], max(3, int(0.03 * H)), axis=0) * (y > FOOT)
ru = ndi.uniform_filter1d(um[fy, np.clip(xi + ripple * 2, 0, W - 1)], max(3, int(0.05 * H)), axis=0) * (y > FOOT)
study = lerp(study, rgb('#0e1024'), 0.8 * rp * (0.5 + 0.5 * brk))
study = lerp(study, rgb('#b3202a'), 0.75 * ru * (0.45 + 0.55 * brk))
study = lerp(study, coat, pm)
study = lerp(study, umb, um)
study = lerp(study, rgb('#1a1a30'), fill(sd_segment(x, y, ucx, ucy, ucx + 0.004, FOOT - 0.55 * FH, 0.002), 0.002))
study = lerp(study, rgb('#1c1c36'), fill(far_fig, 0.002))
study = np.clip(study, 0, 1).astype(F32)

if a.study:
    Image.fromarray((study * 255).astype(np.uint8)).save(a.study)

# ---- painting it ------------------------------------------------------------------------
one = np.ones_like(x)
sky_m = (y < np.where(x < VX, roofL, roofR) + 0.01).astype(F32) * (y < HOR + 0.01)
wall_m = (wallL | wallR).astype(F32)
st_m = street_m.astype(F32)
vert = np.pi / 2

# 1. lay-in: the whole canvas, thin and fast, big brush
p.strokes(one, blur(study, 0.02 * H), flow=0.2 * p.noise(0.6, 2), size=0.085, length=4, load=0.8, opacity=0.9,
          pickup=0.0, dry=0.35, thickness=0.35, jitter=0.07, name='lay-in')
# 2. sky: long sweeps that fall toward the gap between the houses
sky_flow = 0.25 * (x - VX) + 0.35 * p.noise(0.5, 2, seed=51)
p.mass(sky_m, study, flow=sky_flow, sizes=(0.055, 0.028), length=7, pickup=0.55, jitter=0.05, hue=0.04,
       overshoot=0.6, tol=0.05, name='sky')
# 3. houses: upright strokes, each front its own tone
p.mass(wall_m, study, flow=vert + 0.12 * p.noise(0.3, 2, seed=52), sizes=(0.04, 0.02, 0.009), length=4.5,
       pickup=0.4, jitter=0.06, hue=0.05, ctol=0.16, tol=0.07, overshoot=0.15, name='houses')
# 4. the street: reflections are dragged straight down, wet into wet...
p.mass(st_m, study, flow=vert + 0.05 * p.noise(0.4, 2, seed=53), sizes=(0.05, 0.024), length=8, pickup=0.7,
       jitter=0.06, hue=0.05, ctol=0.2, tol=0.06, overshoot=0.1, soft=0.2, name='street')
# ...then cut across with short flat strokes: the surface of the water
p.strokes(st_m * (0.35 + 0.65 * brk), study, flow=0.02 * p.noise(0.3, 2, seed=54), size=0.012, length=7,
          density=0.5, pickup=0.5, load=0.8, dry=0.3, ctol=0.1, thickness=1.2, name='ripples')
p.dry(0.6)
# 5. the lights: thick, clean paint laid on last and left alone
light_m = np.clip(core + (np.clip(streak - 0.55, 0, 1) * brk * st_m) + 0.9 * (shop_lit_L | shop_lit_R), 0, 1)
p.strokes(light_m, study, flow=vert, size=0.012, length=2.5, pickup=0.15, load=1.3, thickness=2.6, jitter=0.04,
          hue=0.03, ctol=0.2, overshoot=0.0, name='lights')
win_m = np.zeros_like(x)
for wall, tt, roof, curb, sd in ((wallL, tL, eaveL, curbL, 21), (wallR, tR, eaveR, curbR, 22)):
    win_m = np.maximum(win_m, (wall & windows(tt, roof, curb, sd)[1]).astype(F32))
p.strokes(win_m, study, flow=vert, size=0.008, length=2.0, pickup=0.1, load=1.3, thickness=2.2, density=1.6,
          ctol=0.25, overshoot=0.0, clip=0.5, name='windows')
p.strokes(np.clip(halo * 0.9, 0, 1) * (1 - core), study, flow=along(halo, 2.0), size=0.016, length=4,
          density=0.8, opacity=0.8, pickup=0.45, dry=0.2, soft=0.25, ctol=0.1, name='halos')
# 6. the figures: dark first, then the umbrella in two reds, then the lamp-lit edges
fig_m = np.maximum(pm, fill(far_fig, 0.002))
p.mass(fig_m, study, flow=vert, sizes=(0.016, 0.007), length=3.5, pickup=0.25, clip=0.8, overshoot=0.0,
       ctol=0.3, tol=0.05, jitter=0.04, name='figures')
p.mass(um, study, flow=along(sd_ellipse(x, y, ucx, ucy + 0.06, 0.135, 0.13), 1.5), sizes=(0.016, 0.007),
       length=4, pickup=0.2, clip=0.85, overshoot=0.0, ctol=0.3, tol=0.045, thickness=1.6, name='umbrella')
p.strokes(np.clip(rp + ru, 0, 1) * st_m, study, flow=vert, size=0.012, length=6, pickup=0.6, density=0.8,
          ctol=0.25, refine=0.06, soft=0.2, name='figure reflections')
# 7. last touches everywhere, small brush, only where the picture still asks for it
p.strokes(one, study, flow=vert * (wall_m + st_m > 0) + sky_flow * sky_m, size=0.006, length=3.5, pickup=0.3,
          refine=0.085, ctol=0.2, thickness=1.3, name='touches')
# 8. drawn marks: lamp posts in one pull each, the umbrella's shaft, then the lamps themselves
p.marks([[(lx, ly + 0.01 * s), (lx, lb)] for lx, ly, lb, s in lamps], '#14142a',
        size=[0.0075 * s + 0.002 for *_, s in lamps], pickup=0.3, taper=0.15, name='lamp posts')
p.marks([[(lx - 0.016 * s, ly + 0.014 * s), (lx + 0.016 * s, ly + 0.014 * s)] for lx, ly, lb, s in lamps],
        '#14142a', size=[0.008 * s + 0.002 for *_, s in lamps], taper=0.0, name='lamp brackets')
p.marks([[(ucx, ucy + 0.01), (ucx + 0.004, FOOT - 0.55 * FH)]], '#17172c', size=0.004, name='umbrella shaft')
p.marks([[(lx - 0.004 * s, ly - 0.004 * s), (lx + 0.007 * s, ly + 0.003 * s)] for lx, ly, lb, s in lamps],
        '#fff4cf', size=[0.036 * s + 0.005 for *_, s in lamps], pickup=0.0, load=1.5, thickness=3.5, taper=0.1,
        jitter=0.015, name='lamps')

p.save(a.out)
if a.film:
    p.film(a.film)
