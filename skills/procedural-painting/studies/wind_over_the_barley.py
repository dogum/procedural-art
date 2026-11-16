"""Wind over the barley: a wide field of ripe barley leaning in the wind under towering summer
cumulus, poplars and a red-roofed farm on the horizon, a rutted track coming in from the corner,
poppies in the near field. Painted from imagination; nothing is traced or sampled from an image.

    python wind_over_the_barley.py --out barley.png [--height 1600] [--film barley.mp4] [--study study.png] [--tape out.tape]
"""
import argparse
import os
import sys

import numpy as np
import scipy.ndimage as ndi

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
from atelier import (Painting, rgb, lerp, smoothstep, fill, blur, sd_box, sd_ellipse, sd_poly,  # noqa: E402
                     along, F32)
from PIL import Image  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--out', default='wind_over_the_barley.png')
ap.add_argument('--height', type=int, default=1000)
ap.add_argument('--film', default=None)
ap.add_argument('--tape', default=None, help='also save the stroke tape (for the browser player)')
ap.add_argument('--study', default=None, help='also save the flat colour study the painter works from')
ap.add_argument('--seed', type=int, default=5)
a = ap.parse_args()

H = a.height
W = int(H * 1.5)
p = Painting(W, H, seed=a.seed, ground='#b98a5a', film=bool(a.film), tape=bool(a.tape))
x, y = p.x, p.y
rng = np.random.default_rng(a.seed)
HOR = 0.66                                   # low horizon: two thirds of the picture is sky
SUN = np.array([-0.58, -0.62, 0.53], F32)    # light from the upper left, a little toward us
SUN /= np.linalg.norm(SUN)


def ramp(v, stops):
    """Colour ramp: stops = [(value, colour), ...] ascending."""
    out = np.broadcast_to(rgb(stops[0][1]), v.shape + (3,)).astype(F32)
    for (v0, _), (v1, c1) in zip(stops[:-1], stops[1:]):
        out = lerp(out, rgb(c1), smoothstep(v0, v1, v))
    return out


# ---- the colour study ------------------------------------------------------------------
# sky: cobalt overhead, cerulean lower, a warm milky band on the horizon
t = np.clip(y / HOR, 0, 1)
sky = ramp(t, [(0.0, '#2c5aa6'), (0.45, '#4f86c2'), (0.78, '#9cc3dc'), (1.0, '#e3dcc6')])
sky = lerp(sky, rgb('#3a6fb6'), 0.25 * smoothstep(-0.5, 1.0, p.noise(0.5, 3, seed=1)) * (1 - t))

# clouds: heaps of spheres with flat bases, lit from the upper left
wx = x + 0.018 * p.noise(0.09, 3, seed=2) + 0.006 * p.noise(0.025, 2, seed=3)
wy = y + 0.014 * p.noise(0.09, 3, seed=4) + 0.006 * p.noise(0.025, 2, seed=5)


def cumulus(puffs, base, far=0.0, lumps=True):
    """Returns (mask, colour) for one cloud; far 0..1 pushes it back into the haze."""
    hgt = np.zeros_like(x)
    sd = np.full_like(x, 1e9)
    rr = np.random.default_rng(int(base * 977) + len(puffs))
    extra = []
    for _ in range(int(1.6 * len(puffs)) if lumps else 0):          # cauliflower lumps round the top and sides
        cx, cy, r = puffs[rr.integers(len(puffs))]
        ang = rr.uniform(-np.pi * 1.05, 0.05 * np.pi)
        extra.append((cx + 0.95 * r * np.cos(ang), cy + 0.95 * r * np.sin(ang), r * rr.uniform(0.25, 0.45)))
    for cx, cy, r in list(puffs) + extra:
        d2 = ((wx - cx) ** 2 + (wy - cy) ** 2) / (r * r)
        hgt = np.maximum(hgt, r * np.sqrt(np.clip(1 - d2, 0, 1)))
        sd = np.minimum(sd, np.sqrt(d2) * r - r)
    base_w = base + 0.008 * p.noise(0.12, 2, seed=int(base * 1000))
    sd = np.maximum(sd, wy - base_w)
    hs = blur(hgt, 0.006 * H)
    gy, gx = np.gradient(hs)
    gx, gy = gx * H * 1.6, gy * H * 1.6
    nz = 1 / np.sqrt(1 + gx * gx + gy * gy)
    lam = (-gx * SUN[0] - gy * SUN[1] + SUN[2]) * nz
    hc = blur(hgt, 0.035 * H)                                          # the heap as a whole also turns from the sun
    cy_, cx_ = np.gradient(hc)
    cx_, cy_ = cx_ * H * 2.2, cy_ * H * 2.2
    lam = 0.6 * lam + 0.4 * (-cx_ * SUN[0] - cy_ * SUN[1] + SUN[2]) / np.sqrt(1 + cx_ * cx_ + cy_ * cy_)
    under = smoothstep(base - 0.13, base + 0.005, y)                 # the shaded underside
    v = lam - 0.7 * under + 0.13 * p.noise(0.06, 3, seed=7)
    col = ramp(v, [(-0.3, '#566088'), (0.1, '#7c84a8'), (0.30, '#a9a6bc'), (0.44, '#d6cbc8'), (0.58, '#f6e7c8'), (0.9, '#fffaea')])
    col = lerp(col, rgb('#a9907f'), 0.22 * smoothstep(base - 0.05, base, y))      # warm bounce off the field
    col = lerp(col, rgb('#cfd6dc'), far * 0.75)
    soft = 0.003 + 0.022 * smoothstep(0.45, 0.0, v) + 0.012 * far      # hard where lit, lost in shadow
    mask = np.clip((soft - sd) / (2 * soft), 0, 1)
    mask = mask * mask * (3 - 2 * mask)
    return mask.astype(F32), col, hc, v


hero = [(0.93, 0.40, 0.115), (1.07, 0.36, 0.135), (1.21, 0.40, 0.105), (1.02, 0.25, 0.115), (1.13, 0.20, 0.10),
        (0.99, 0.135, 0.075), (1.075, 0.095, 0.062), (0.83, 0.435, 0.07), (1.32, 0.44, 0.065), (1.17, 0.30, 0.09),
        (0.90, 0.30, 0.07)]
left = [(0.20, 0.50, 0.07), (0.31, 0.46, 0.085), (0.42, 0.49, 0.07), (0.29, 0.385, 0.065), (0.37, 0.36, 0.05),
        (0.10, 0.515, 0.05), (0.50, 0.51, 0.045)]
top = [(0.06, 0.07, 0.10), (0.20, 0.04, 0.085), (-0.04, 0.10, 0.08), (0.31, 0.075, 0.05), (0.13, 0.12, 0.06)]
rightc = [(1.44, 0.14, 0.07), (1.52, 0.11, 0.07), (1.38, 0.17, 0.04)]
clouds = [cumulus(top, 0.165, 0.0), cumulus(rightc, 0.20, 0.1), cumulus(left, 0.545, 0.35),
          cumulus(hero, 0.475, 0.0)]
# a far rank of small flat clouds low on the horizon
far_puffs = []
r2 = np.random.default_rng(9)
for cx in np.arange(0.02, 1.5, 0.085):
    cx += r2.uniform(-0.02, 0.02)
    yy = 0.615 - r2.uniform(0, 0.03)
    for k in range(3):
        far_puffs.append((cx + (k - 1) * 0.022, yy - r2.uniform(0.0, 0.012), r2.uniform(0.013, 0.024)))
m_far, c_far, h_far, v_far = cumulus(far_puffs, 0.625, 0.8, lumps=False)
gap = smoothstep(-0.3, 0.4, p.noise(0.25, 2, seed=11))               # break the rank up
m_far = m_far * gap

study = sky.copy()
study = lerp(study, c_far, m_far * 0.9)
cloud_m = m_far * 0.9
cloud_h = h_far.copy()
cloud_v = v_far.copy()
for m_, c_, h_, v_ in clouds:
    study = lerp(study, c_, m_)
    cloud_h = np.where(m_ > 0.5, h_, cloud_h)
    cloud_v = np.where(m_ > 0.5, v_, cloud_v)
    cloud_m = np.maximum(cloud_m, m_)

# the land: ground-plane coordinates so that everything shrinks toward the horizon
dy = np.maximum(y - HOR, 0)
z = 1.0 / (dy + 0.022)                       # depth (large = far)
gu, gv = (x - 0.75) * z, z


def ground_noise(scale_u, scale_v, seed, octaves=3, shear=0.0):
    """Noise laid on the ground plane: scale_* are features per unit of depth, across and into the picture."""
    tex = p.noise(0.08, octaves, seed=seed)
    row = gv * scale_v * H * 0.02
    col = gu * scale_u * H * 0.02 + shear * row + 0.5 * W
    return ndi.map_coordinates(tex, [row % (2 * H), col % (2 * W)], order=1, mode='mirror').astype(F32)


near = smoothstep(0.0, 0.34, dy)                                     # 0 at the horizon, 1 at our feet
waves = ground_noise(1.3, 2.4, 21, shear=0.6)                                  # wind running across the crop
fine = ground_noise(7.0, 6.0, 22, 2)
shadow = smoothstep(0.25, 0.6, ground_noise(0.55, 0.8, 23, 2, shear=0.3) * 0.55
                    + 0.85 * np.exp(-((dy - 0.075 - 0.02 * np.sin(x * 2.3 + 1)) / 0.04) ** 2)
                    - 1.2 * smoothstep(0.15, 0.24, dy))              # cloud shadow across the middle distance
barley = ramp(waves + 0.35 * fine, [(-1.2, '#b9812c'), (-0.2, '#d6a53e'), (0.6, '#e3b94c'), (1.5, '#f0d37c')])
barley = lerp(barley, rgb('#d9cfa0'), 0.70 * (1 - smoothstep(0.0, 0.10, dy)))     # haze takes the far field
barley = lerp(barley, lerp(barley * 0.62, rgb('#8c7f86'), 0.42), 0.85 * shadow)       # cool violet shadow
# the nearest stalks: darker gaps between heads, warm ochres
gaps = smoothstep(0.3, 1.2, ground_noise(16.0, 9.0, 24, 2)) * smoothstep(0.12, 0.30, dy)
barley = lerp(barley, rgb('#8a5a22'), 0.45 * gaps)

# green verge and a strip of rough grass in the lower right
verge_n = ground_noise(4.0, 2.0, 25)
grass = ramp(verge_n, [(-1, '#4f6a2c'), (0.3, '#7d9440'), (1.3, '#b3b85a')])

# the track: two ruts and a grassy crown, in from the lower left toward the farm
s = np.clip(dy / (1 - HOR), 0, 1)
xc = 0.93 - 0.80 * s ** 1.5 + 0.03 * np.sin(s * 5.0)
hw = 0.007 + 0.20 * s ** 1.2
tu = (x - xc) / hw                                                   # -1..1 across the track
track_m = smoothstep(1.08, 0.92, np.abs(tu)) * (y > HOR + 0.004)
ruts = smoothstep(0.30, 0.12, np.abs(np.abs(tu) - 0.55))
earth = ramp(ground_noise(5.0, 3.0, 26), [(-1, '#a17a58'), (0.2, '#cfa883'), (1.2, '#e8cfae')])
earth = lerp(earth, rgb('#6f4d3a'), 0.7 * ruts * (0.6 + 0.4 * smoothstep(-0.5, 0.5, fine)))
earth = lerp(earth, grass * 0.95, smoothstep(0.30, 0.10, np.abs(tu)) * (0.55 + 0.45 * smoothstep(-0.3, 0.6, verge_n)))
earth = lerp(earth, lerp(earth * 0.62, rgb('#857c8c'), 0.45), 0.85 * shadow)
edge_grass = smoothstep(1.55, 1.0, np.abs(tu)) * smoothstep(0.03, 0.2, dy) * smoothstep(-0.6, 0.5, verge_n)

land = barley
land = lerp(land, lerp(grass, grass * rgb('#8d86a0'), 0.5 * shadow), 0.8 * edge_grass)
land = lerp(land, earth, track_m)

# poppies: a drift of red through the near field on the right, thinning with distance
pop_band = np.exp(-((dy - 0.19 - 0.05 * np.sin(x * 4)) / 0.065) ** 2) * smoothstep(0.5, 0.8, x) \
    * smoothstep(-0.6, 0.5, ground_noise(1.5, 1.2, 27))
pop_band = (pop_band * (1 - track_m) * (1 - edge_grass)).astype(F32)
land = lerp(land, rgb('#c9552e'), 0.22 * pop_band)                    # the study only blushes; dabs come last

# the horizon: blue hills, a wood, a row of poplars, the farm
hills = HOR - 0.022 - 0.012 * np.sin(x * 3.1 + 0.6) - 0.008 * np.sin(x * 7.3) - 0.004 * p.noise(0.1, 2, seed=31)
hill_m = (y > hills) & (y <= HOR + 0.002)
far_land = lerp(rgb('#8fa7b8'), rgb('#a9b59a'), smoothstep(hills, HOR, y)[..., None] if False else smoothstep(0, 1, (y - hills) / 0.03))
wood = HOR - 0.012 - 0.010 * smoothstep(-0.5, 0.8, p.noise(0.03, 3, seed=32)) * smoothstep(0.02, 0.2, x) \
    * (1 - smoothstep(0.50, 0.62, x)) - 0.008 * smoothstep(1.2, 1.35, x) * smoothstep(-0.5, 0.8, p.noise(0.03, 3, seed=33))
wood_m = (y > wood) & (y <= HOR + 0.004) & ((x < 0.62) | (x > 1.2))

study = np.where((y > HOR)[..., None], land, study)
study = np.where(hill_m[..., None], far_land, study)
study = np.where(wood_m[..., None], lerp(rgb('#4a6a5c'), rgb('#6a8470'), smoothstep(-1, 1, p.noise(0.02, 2, seed=34))), study)

# poplars: tall dark flames, the sunny side a warmer green
POPLARS = [(0.655, 0.105), (0.690, 0.118), (0.722, 0.098), (0.752, 0.112), (0.785, 0.092), (0.842, 0.072)]
pop_sd = np.full_like(x, 1e9)
for px_, ph in POPLARS:
    wob = 0.0025 * p.noise(0.012, 2, seed=int(px_ * 1000))
    pop_sd = np.minimum(pop_sd, sd_ellipse(x + wob, y, px_, HOR - ph * 0.5 + 0.004, 0.013 + ph * 0.035, ph * 0.5))
pop_m = fill(pop_sd, 0.002)
pop_lit = smoothstep(0.004, -0.008, pop_sd + 0.9 * (np.sin((x - 0.64) * 190.0) * 0.004 + 0.004))
gy_, gx_ = np.gradient(blur(pop_m, 0.004 * H))
side = smoothstep(-0.2, 0.6, gx_ * H * 0.02)                         # 1 on the left (sun) side
pop_col = lerp(rgb('#22382e'), rgb('#56743a'), side)
pop_col = lerp(pop_col, rgb('#7f9a8c'), 0.18)                        # a little air in front of them
study = lerp(study, pop_col, pop_m)

# the farm: a long house and a barn, lit gable ends, red roofs
FX, FY = 1.035, HOR + 0.004
walls = np.minimum(sd_box(x, y, FX, FY - 0.013, 0.036, 0.013), sd_box(x, y, FX + 0.075, FY - 0.011, 0.024, 0.011))
roof = np.minimum(sd_poly(x, y, [(FX - 0.042, FY - 0.025), (FX + 0.042, FY - 0.025), (FX + 0.032, FY - 0.046),
                                 (FX - 0.030, FY - 0.046)]),
                  sd_poly(x, y, [(FX + 0.047, FY - 0.021), (FX + 0.103, FY - 0.021), (FX + 0.094, FY - 0.038),
                                 (FX + 0.056, FY - 0.038)]))
wall_col = lerp(rgb('#f4e6c8'), rgb('#a79aa6'), smoothstep(FX - 0.02, FX - 0.005, x) * (x < FX + 0.045)
                + smoothstep(FX + 0.06, FX + 0.07, x))
roof_col = lerp(rgb('#d2603c'), rgb('#9c3f30'), smoothstep(FY - 0.046, FY - 0.022, y))
oak = np.minimum(sd_ellipse(x + 0.004 * p.noise(0.015, 2, seed=36), y, FX - 0.075, FY - 0.03, 0.036, 0.03),
                 sd_ellipse(x, y, FX + 0.135, FY - 0.022, 0.03, 0.022))
oak_col = lerp(rgb('#2f4a30'), rgb('#6e8a3c'), smoothstep(0.01, -0.02, (y - (FY - 0.035)) + 0.6 * (x - (FX - 0.085))))
oak_col = np.where((x > FX + 0.1)[..., None],
                   lerp(rgb('#2f4a30'), rgb('#6e8a3c'), smoothstep(0.01, -0.02, (y - (FY - 0.027)) + 0.6 * (x - (FX + 0.125)))),
                   oak_col)
study = lerp(study, oak_col, fill(oak, 0.003))
study = lerp(study, wall_col, fill(walls, 0.0015))
study = lerp(study, roof_col, fill(roof, 0.0015))
farm_m = np.maximum.reduce([fill(walls, 0.002), fill(roof, 0.002), fill(oak, 0.003)])
study = np.clip(study, 0, 1).astype(F32)

if a.study:
    Image.fromarray((study * 255).astype(np.uint8)).save(a.study)

# ---- painting it ------------------------------------------------------------------------
one = np.ones_like(x)
sky_m = (y < HOR + 0.005).astype(F32)
land_m = (y > HOR - 0.002).astype(F32)
far_m = land_m * (dy < 0.045)
mid_m = land_m * smoothstep(0.03, 0.05, dy) * (1 - smoothstep(0.13, 0.17, dy))
near_m = land_m * smoothstep(0.13, 0.17, dy)
lean = -np.pi / 2 + 0.55 + 0.35 * waves * near + 0.15 * p.noise(0.2, 2, seed=51)     # stalks pushed over to the right
trk_flow = np.arctan2(np.ones_like(x), np.gradient(xc, axis=0) * H)                # along the track

# 1. lay-in: the big shapes, thin, scrubbed, the warm ground left showing through
p.strokes(one, blur(study, 0.018 * H), flow=0.25 * p.noise(0.6, 2, seed=52), size=0.08, length=4, load=0.8,
          opacity=0.9, pickup=0.0, dry=0.4, thickness=0.3, jitter=0.06, name='lay-in')
# 2. sky: long arm-length sweeps, lifting to the right the way the clouds are moving
sky_flow = -0.42 * (1 - t) ** 1.5 + 0.28 * (1 - t) * p.noise(0.5, 2, seed=53)
p.mass(sky_m, lerp(study, sky, cloud_m),  flow=sky_flow, sizes=(0.055, 0.026), length=7, pickup=0.5,
       jitter=0.04, hue=0.035, overshoot=0.5, tol=0.045, soft=0.15, name='sky')
# 3. clouds: shadows first, worked soft into the wet sky; the brush rolls around each heap
big_m = np.maximum.reduce([m_ for m_, *_ in clouds])
cl_flow = np.where(big_m > 0.05, along(blur(cloud_h * big_m, 0.02 * H)) + 0.25 * p.noise(0.15, 2, seed=54), 0.04)
p.mass(cloud_m, study, flow=cl_flow, sizes=(0.045, 0.022, 0.010), length=5, pickup=0.55, jitter=0.035, hue=0.03,
       ctol=0.11, tol=0.045, overshoot=0.25, soft=0.2, name='clouds')
#    ...then the lights, loaded and left alone
lights = cloud_m * smoothstep(0.50, 0.68, cloud_v)
p.strokes(lights, study, flow=cl_flow, size=0.022, length=3.5, pickup=0.2, load=1.35, thickness=2.6, jitter=0.025,
          hue=0.02, ctol=0.12, overshoot=0.1, density=0.9, name='cloud lights')
p.strokes(cloud_m * smoothstep(0.66, 0.85, cloud_v), study, flow=cl_flow + 0.3, size=0.011, length=2.5, pickup=0.1,
          load=1.5, thickness=3.4, jitter=0.02, ctol=0.12, overshoot=0.0, density=0.8, name='cloud crests')
# 4. the far country: small flat touches, soft, cool
hz_m = np.clip(hill_m + wood_m + far_m, 0, 1).astype(F32)
p.mass(hz_m, study, flow=0.02, sizes=(0.012, 0.006), length=5, pickup=0.5, jitter=0.03, ctol=0.12, tol=0.04,
       overshoot=0.1, soft=0.15, thickness=0.7, name='far country')
# 5. the field: flat strokes far off, leaning ones nearer, growing as they come toward us
p.mass(mid_m, study, flow=0.06 + 0.25 * lean * 0 + 0.1 * waves, sizes=(0.022, 0.010), length=5, pickup=0.45,
       jitter=0.05, hue=0.04, ctol=0.14, tol=0.05, name='middle field')
p.mass(near_m * (1 - track_m), study, flow=lean, sizes=(0.034, 0.016), length=4, pickup=0.4, jitter=0.07,
       hue=0.05, ctol=0.16, tol=0.06, thickness=1.2, name='near field')
p.mass(track_m, study, flow=trk_flow, sizes=(0.028, 0.012, 0.006), length=6, pickup=0.45, jitter=0.05,
       ctol=0.14, tol=0.05, overshoot=0.1, name='track')
p.strokes(pop_band * (dy < 0.16) * (y > HOR + 0.03), lerp(study, rgb('#c8402a'), 0.45), flow=0.03, size=0.006,
          length=5, density=0.35,
          pickup=0.4, ctol=0.3, name='far poppies')
#    ears and stalks: flicks of light and dark, bigger and fewer toward the bottom edge
for lo, hi, sz, ln in ((0.05, 0.13, 0.0045, 3.0), (0.11, 0.22, 0.007, 4.5), (0.19, 0.36, 0.010, 6.0)):
    band = land_m * smoothstep(lo, lo + 0.02, dy) * (1 - smoothstep(hi - 0.03, hi, dy)) * (1 - track_m)
    p.strokes(band, study, flow=lean, size=sz, length=ln, density=0.75, pickup=0.3, load=1.1, jitter=0.11,
              hue=0.06, ctol=0.2, thickness=1.5, taper=0.7, wobble=0.35, bidir=False, name='barley flicks')
p.dry(0.5)
# 6. poplars and the farm: small, exact, darkest darks and the one hard-edged white wall
p.mass(pop_m, study, flow=np.pi / 2, sizes=(0.010, 0.005), length=4, pickup=0.25, clip=0.7, overshoot=0.0,
       ctol=0.3, tol=0.04, jitter=0.04, name='poplars')
p.mass(farm_m, study, flow=0.0, sizes=(0.008, 0.004), length=3, pickup=0.15, clip=0.85, overshoot=0.0, ctol=0.2,
       tol=0.035, jitter=0.03, thickness=1.3, name='farm')
# 7. last touches: only where the picture still asks, then a few long stalks across the bottom
p.strokes(one, study, flow=np.where(y > HOR, lean, np.where(cloud_m > 0.3, cl_flow, sky_flow)), size=0.0055, length=3.5, pickup=0.3, refine=0.085,
          ctol=0.2, thickness=1.2, name='touches')
# 8. poppies: single dabs of clean red, thick, smaller and closer together with distance
cand = 9000
cx_, cy_ = rng.uniform(0.45, 1.5, cand), rng.uniform(HOR + 0.06, 1.0, cand)
pb = pop_band[np.clip((cy_ * H).astype(int), 0, H - 1), np.clip((cx_ * H).astype(int), 0, W - 1)]
clump = smoothstep(-0.7, 0.5, ground_noise(9.0, 7.0, 29, 2))            # they grow in patches
cl = clump[np.clip((cy_ * H).astype(int), 0, H - 1), np.clip((cx_ * H).astype(int), 0, W - 1)]
keep = rng.random(cand) < pb * cl * 0.62 / (0.25 + 2.5 * (cy_ - HOR))
px_, py_ = cx_[keep], cy_[keep]
szs = (0.003 + 0.034 * (py_ - HOR)) * rng.uniform(0.55, 1.45, keep.sum())
order = np.argsort(py_)
reds = ('#e23a22', '#b81d1a', '#f2603a', '#d92c20', '#9c1a1c')
p.marks([[(px_[i] - 0.4 * szs[i], py_[i]), (px_[i] + 0.5 * szs[i], py_[i] - 0.25 * szs[i])] for i in order],
        [rgb(reds[rng.integers(5)]) for i in order], size=[float(szs[i]) for i in order], pickup=0.05, load=1.4,
        thickness=3.0, taper=0.1, jitter=0.05, name='poppies')
stalks, cols = [], []
for _ in range(90):
    sx = rng.uniform(0.0, 1.5)
    sy = rng.uniform(0.93, 1.03)
    if abs((sx - 0.13) / 0.2) < 1.1:          # keep the mouth of the track open
        continue
    ln = rng.uniform(0.07, 0.15)
    ang = -np.pi / 2 + 0.45 + rng.normal(0, 0.18)
    bend = rng.uniform(0.01, 0.035)
    stalks.append([(sx, sy), (sx + 0.5 * ln * np.cos(ang), sy + 0.5 * ln * np.sin(ang)),
                   (sx + ln * np.cos(ang) + bend, sy + ln * np.sin(ang) + 0.3 * bend)])
    cols.append(('#f3dc94', '#e2b955', '#9a6a26', '#f8ecbe', '#7a5a2a')[rng.integers(5)])
p.marks(stalks, [rgb(c) for c in cols], size=0.0038, pickup=0.2, load=1.2, thickness=1.8, taper=0.6, jitter=0.06, name='stalks')

p.save(a.out)
if a.film:
    p.film(a.film)
if a.tape:
    from tape import save_tape
    print(f"wrote {a.tape}  {save_tape(p, a.tape, title='Wind over the barley') / 1e6:.2f} MB")
