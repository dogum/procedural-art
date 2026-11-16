"""Lemons and a blue jug: a cobalt-glazed jug, three lemons and a cut half on rumpled linen over
a dark table, window light from the upper left. Painted from imagination; nothing is traced or
sampled from an image.

    python lemons_and_blue_jug.py --out lemons.png [--height 1600] [--film lemons.mp4] [--study study.png] [--tape out.tape]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
from atelier import (Painting, rgb, lerp, mix, smoothstep, fill, blur, sd_circle, sd_ellipse,  # noqa: E402
                     sd_segment, sd_poly, sd_curve, smin, curve, along, around, F32)
from PIL import Image  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--out', default='lemons_and_blue_jug.png')
ap.add_argument('--height', type=int, default=1000)
ap.add_argument('--film', default=None)
ap.add_argument('--tape', default=None, help='also save the stroke tape (for the browser player)')
ap.add_argument('--study', default=None, help='also save the flat colour study the painter works from')
ap.add_argument('--seed', type=int, default=5)
a = ap.parse_args()

H = a.height
W = int(H * 1.25)
p = Painting(W, H, seed=a.seed, ground='#5a3b26', film=bool(a.film), tape=bool(a.tape))
x, y = p.x, p.y
LIGHT = np.array([-0.70, -0.56, 0.44], F32)          # from the upper left, a little in front
LIGHT /= np.linalg.norm(LIGHT)
TABLE = 0.565                                         # back edge of the table


def form(n, light, half, core, refl, bounce=(0.25, 0.9, 0.3), refl_amt=0.6, turn=0.0):
    """Light and shadow families of a round form from its normal field n = (nx, ny, nz)."""
    lam = n[0] * LIGHT[0] + n[1] * LIGHT[1] + n[2] * LIGHT[2] - turn      # turn > 0 brings the shadow round
    c = lerp(rgb(core), rgb(half), smoothstep(-0.02, 0.30, lam))
    c = lerp(c, rgb(light), smoothstep(0.25, 0.80, lam))
    b = np.asarray(bounce, F32)
    b /= np.linalg.norm(b)
    up = np.clip(n[0] * b[0] + n[1] * b[1] + n[2] * b[2], 0, 1)        # light coming back off the table
    c = lerp(c, rgb(refl), refl_amt * smoothstep(0.1, 0.9, up) * smoothstep(0.02, -0.35, lam))
    return c, lam


def ellipsoid(cx, cy, rx, ry, ang):
    """Normal field of an ellipsoid seen face on (valid inside its outline)."""
    c, s = np.cos(ang), np.sin(ang)
    u, v = ((x - cx) * c + (y - cy) * s) / rx, (-(x - cx) * s + (y - cy) * c) / ry
    k = np.minimum(np.hypot(u, v), 0.999)
    sc = k / np.maximum(np.hypot(u, v), 1e-6)
    u, v = u * sc, v * sc
    nz = np.sqrt(1 - u * u - v * v)
    return (u * c - v * s, u * s + v * c, nz)


def spec(n, power, tilt=(0.0, 0.0)):
    hx, hy, hz = LIGHT[0] + tilt[0], LIGHT[1] + tilt[1], LIGHT[2] + 1.0
    hn = np.sqrt(hx * hx + hy * hy + hz * hz)
    return np.clip((n[0] * hx + n[1] * hy + n[2] * hz) / hn, 0, 1) ** power


def cast(ax, ay, bx, by, r0, r1, squash=0.36, soft0=0.006, soft1=0.05):
    """A cast shadow lying on the table: a tapered capsule squashed into the table plane, crisp at
    the object and soft at its far end. Returns 0..1."""
    yy = ay + (y - ay) / squash
    byy = ay + (by - ay) / squash
    d = sd_segment(x, yy, ax, ay, bx, byy, r0, r1)
    far = np.clip(((x - ax) * (bx - ax) + (yy - ay) * (byy - ay)) / ((bx - ax) ** 2 + (byy - ay) ** 2), 0, 1)
    soft = soft0 + (soft1 - soft0) * far
    return smoothstep(soft, -soft, d).astype(F32)


# ---- the colour study ------------------------------------------------------------------
# background: dark and cool behind the lit side of things, a warmer, lighter air behind their shadows
bg = lerp(rgb('#14130f'), rgb('#4a3722'), smoothstep(0.15, 1.15, x + 0.25 * (y - 0.3)))
bg = lerp(bg, rgb('#17251f'), 0.55 * smoothstep(-0.6, 0.9, p.noise(0.5, 3, seed=1)) * smoothstep(0.9, 0.2, x))
bg = lerp(bg, rgb('#6a4f2e'), 0.35 * np.exp(-(((x - 0.86) / 0.3) ** 2 + ((y - 0.36) / 0.22) ** 2)))
bg = bg * (0.9 + 0.1 * p.noise(0.12, 3, seed=2)[..., None])

# table: dark wood, a dull shine along its back edge
wood = lerp(rgb('#2a1a10'), rgb('#4a2c18'), smoothstep(-1, 1, p.noise(0.3, 3, seed=3) + 4 * np.sin(y * 90 + 3 * p.noise(0.4, 2, seed=4)) * 0.2))
wood = lerp(wood, rgb('#7a4a28'), 0.6 * np.exp(-((y - TABLE - 0.012) / 0.012) ** 2))
study = np.where((y > TABLE)[..., None], wood, bg)

# linen: a pale cloth thrown over the table, its far edge wandering, folds running off to the right
wob = 0.012 * p.noise(0.12, 2, seed=5)
cloth_sd = sd_poly(x + wob, y + wob, [(-0.2, 0.615), (0.30, 0.600), (0.78, 0.612), (1.00, 0.640), (1.085, 0.72),
                                      (1.04, 0.83), (1.10, 0.93), (1.17, 1.2), (-0.2, 1.2)])
cloth_m = fill(cloth_sd, 0.003)
fold_long = (0.55 * np.sin(13 * (x * 0.55 + y * 0.83) + 2.2 * p.noise(0.35, 2, seed=6))
             + 0.45 * np.sin(21 * (x * 0.9 - y * 0.42) + 2.5 * p.noise(0.3, 2, seed=7)) * smoothstep(0.55, 1.05, x))
fold = fold_long + 0.4 * p.noise(0.16, 3, seed=8) + 0.9 * smoothstep(0.05, -0.03, cloth_sd + 0.035)
fold = blur(fold, 0.004 * H)
gy_, gx_ = np.gradient(fold)
slope = -(gx_ * LIGHT[0] + gy_ * LIGHT[1]) * H * 0.05
pool = np.exp(-(((x - 0.42) / 0.75) ** 2 + ((y - 0.72) / 0.42) ** 2))       # where the window light lands
clam = np.clip(0.60 + slope, 0, 1) * (0.42 + 0.58 * pool)
cloth = lerp(rgb('#5d5d74'), rgb('#b3a894'), smoothstep(0.10, 0.44, clam))
cloth = lerp(cloth, rgb('#efe4c9'), smoothstep(0.36, 0.66, clam))
cloth = lerp(cloth, rgb('#fbf5e3'), 0.7 * smoothstep(0.66, 0.9, clam))
cloth_thick = smoothstep(0.0, -0.02, cloth_sd)                              # the hem turns under: a darker lip
cloth = lerp(cloth * 0.72, cloth, cloth_thick)

# where things stand
JX, JTOP, JBOT = 0.405, 0.215, 0.735
LEMONS = [  # cx, cy, rx, ry, angle
    (0.735, 0.715, 0.098, 0.072, -0.18),
    (0.905, 0.672, 0.085, 0.064, 0.35),
    (0.170, 0.742, 0.088, 0.066, 0.25),
]
CUT = (0.585, 0.842, 0.082, 0.070, -0.25)

# cast shadows on the cloth and the wood (light comes from the left, so they run right and a little back)
sh = cast(JX + 0.02, JBOT - 0.008, JX + 0.62, JBOT - 0.060, 0.10, 0.075, soft1=0.06)
for cx, cy, rx, ry, ang in LEMONS:
    sh = np.maximum(sh, cast(cx + 0.01, cy + ry * 0.62, cx + rx * 2.3, cy + ry * 0.45, rx * 0.86, rx * 0.45, soft1=0.03))
cxc, cyc, rxc, ryc, angc = CUT
sh = np.maximum(sh, cast(cxc + 0.01, cyc + ryc * 0.55, cxc + rxc * 2.0, cyc + ryc * 0.5, rxc * 0.8, rxc * 0.4, soft1=0.025))
sh_col = lerp(rgb('#38405e'), rgb('#59618a'), smoothstep(0.2, 1.0, pool))   # the shadow is where the blue lives
ground = lerp(study, cloth, cloth_m)
shadowed = lerp(ground * 0.42, sh_col * (0.8 + 0.2 * clam[..., None]), cloth_m)
# warm light bouncing off each lemon into the root of its own shadow
bounce_y = np.zeros_like(x)
for cx, cy, rx, ry, ang in LEMONS + [CUT]:
    bounce_y = np.maximum(bounce_y, np.exp(-(((x - cx - rx * 0.9) / (rx * 0.7)) ** 2 + ((y - cy - ry * 0.75) / (ry * 0.45)) ** 2)))
shadowed = lerp(shadowed, mix(shadowed, rgb('#d9a520'), 0.3), 0.8 * bounce_y * cloth_m)
study = lerp(ground, shadowed, 0.93 * sh)

# ---- the jug: a surface of revolution -------------------------------------------------
prof = curve([(0.00, 0.070), (0.035, 0.062), (0.13, 0.050), (0.24, 0.058), (0.40, 0.108), (0.58, 0.140),
              (0.76, 0.128), (0.92, 0.094), (1.0, 0.082)], 12)
tj = (y - JTOP) / (JBOT - JTOP)
rj = np.interp(tj, prof[:, 0], prof[:, 1]).astype(F32)
drj = np.gradient(rj, axis=0) * H
uj = (x - JX) / rj
foot = JBOT + 0.020 * np.sqrt(np.clip(1 - uj * uj, 0, 1))                    # the foot is an ellipse seen from above
jug_sd = np.maximum(np.abs(x - JX) - rj, np.maximum(JTOP - 0.004 - y, y - foot))
jug_m = fill(jug_sd, 0.0025)
ujc = np.clip(uj, -0.999, 0.999)
inv = 1 / np.sqrt(1 + drj * drj)
nj = (ujc * inv, -drj * inv, np.sqrt(1 - ujc * ujc) * inv)
jug, lam_j = form(nj, '#5a8cec', '#1f47a6', '#0a1440', '#22397a', bounce=(0.75, 0.6, 0.2), refl_amt=0.45, turn=0.16)
jug = lerp(jug, rgb('#0a1238'), 0.82 * smoothstep(0.05, 0.62, uj) * (1 - 0.45 * smoothstep(0.84, 0.97, uj)))
# glaze: the window as a hard highlight on the shoulder, a softer one below, a thin light down the far edge
s1 = spec(nj, 120)
s2 = spec(nj, 26)
jug = lerp(jug, rgb('#a9c2f5'), 0.6 * smoothstep(0.25, 0.95, s2))
win = smoothstep(0.50, 0.66, s1)
bar = (np.abs(x - (JX - 0.062)) < 0.0035) | (np.abs(y - (JTOP + 0.218)) < 0.003)   # the window's glazing bars
jug = lerp(jug, rgb('#fbfaf2'), win * (1 - 0.75 * bar))
# yellow from the nearest lemon comes back green-gold on the blue, low on the shadow side
lb = np.exp(-(((x - JX - 0.095) / 0.05) ** 2 + ((y - 0.655) / 0.075) ** 2)) * smoothstep(0.1, -0.3, lam_j)
jug = lerp(jug, mix(jug, rgb('#f0cc20'), 0.62), 0.95 * lb)
# the cloth lights the underside of the belly
jug = lerp(jug, rgb('#5a6aa0'), 0.5 * smoothstep(0.80, 1.0, tj) * smoothstep(-0.2, 0.6, uj) * smoothstep(0.2, -0.2, lam_j))
jug = lerp(jug, rgb('#8aa4e6'), 0.6 * smoothstep(0.90, 0.985, -uj) * smoothstep(0.3, 0.55, tj) * smoothstep(0.95, 0.7, tj))
# mouth: the rim catches light, the inside is dark with the far wall lit
mouth_sd = sd_ellipse(x, y, JX, JTOP + 0.012, 0.066, 0.017)
mouth = fill(mouth_sd, 0.002)
inside = lerp(rgb('#070b22'), rgb('#2c4c9c'), smoothstep(-0.2, 0.9, (x - JX) / 0.066) * smoothstep(JTOP + 0.022, JTOP + 0.004, y))
rim = smoothstep(0.006, 0.0, np.abs(mouth_sd)) * (y < JTOP + 0.05)
# spout and handle
spout_sd = sd_poly(x, y, [(JX - 0.056, JTOP - 0.002), (JX - 0.090, JTOP - 0.006), (JX - 0.056, JTOP + 0.040)])
spout_m = fill(spout_sd - 0.004, 0.0025) * (1 - jug_m)
hpts = [(JX + 0.050, JTOP + 0.055), (JX + 0.135, JTOP + 0.030), (JX + 0.205, JTOP + 0.105), (JX + 0.185, JTOP + 0.235),
        (JX + 0.118, JTOP + 0.300)]
handle_sd = sd_curve(x, y, hpts, 0.017, 0.014)
handle_m = fill(handle_sd, 0.0025) * (1 - jug_m)
gyh, gxh = np.gradient(blur(handle_sd, 0.002 * H))
gn = np.hypot(gxh, gyh) + 1e-6
hk = np.clip(-handle_sd / 0.017, 0, 1)
nh = (gxh / gn * (1 - hk), gyh / gn * (1 - hk), np.sqrt(np.clip(1 - (1 - hk) ** 2, 0, 1)))
handle, _ = form(nh, '#4d7fe0', '#2450ad', '#0c1a4c', '#263f80', refl_amt=0.4)
handle = lerp(handle, rgb('#f4f6fb'), smoothstep(0.75, 0.95, spec(nh, 40)))
spout = lerp(rgb('#3f6fd0'), rgb('#18307a'), smoothstep(-0.02, 0.05, y - JTOP - 0.01 + 0.5 * (x - JX + 0.08)))
study = lerp(study, handle, handle_m)
study = lerp(study, spout, spout_m)
study = lerp(study, jug, jug_m)
study = lerp(study, inside, mouth)
study = lerp(study, rgb('#b9cdf7'), 0.9 * rim * smoothstep(0.5, -0.6, (x - JX) / 0.066 + 0.4))
study = lerp(study, rgb('#1a2d6e'), 0.8 * rim * smoothstep(0.0, 0.8, (x - JX) / 0.066))

# ---- lemons ---------------------------------------------------------------------------
lemon_m = np.zeros_like(x)
lemon_lam = np.zeros_like(x)
lemon_masks = []
for i, (cx, cy, rx, ry, ang) in enumerate(LEMONS):
    c, s = np.cos(ang), np.sin(ang)
    d = sd_ellipse(x, y, cx, cy, rx, ry, ang)
    d = smin(d, sd_circle(x, y, cx + c * rx * 1.04, cy + s * rx * 1.04, ry * 0.24), 0.035)       # the nipple
    d = smin(d, sd_circle(x, y, cx - c * rx * 0.97, cy - s * rx * 0.97, ry * 0.16), 0.025)      # the stem end
    m = fill(d, 0.0025)
    n = ellipsoid(cx, cy, rx * 1.08, ry * 1.04, ang)
    dimple = 0.035 * p.noise(0.012, 2, seed=20 + i)                                               # peel
    n = (n[0] + dimple, n[1] + dimple, n[2])
    col, lam = form(n, '#fbe24a', '#e3a91a', '#6e4a0e', '#b88422', bounce=(0.2, 0.95, 0.2), refl_amt=0.6, turn=0.14)
    col = lerp(col, rgb('#a9a23a'), 0.35 * smoothstep(-0.1, -0.5, lam) * smoothstep(0.3, 1.0, n[2]))  # green in the turn
    col = lerp(col, rgb('#fffbe0'), 0.9 * smoothstep(0.90, 0.99, spec(n, 40)))
    if i == 0:                                              # the big lemon sits in the edge of the jug's shadow
        inj = cast(JX + 0.02, JBOT - 0.008, JX + 0.62, JBOT - 0.060, 0.10, 0.075, soft1=0.06)
        col = lerp(col, col * rgb('#8a7a5a') * 1.05, 0.55 * inj * smoothstep(cy - ry * 0.2, cy + ry, y))
    if i == 0:                                              # blue of the jug in its near side
        col = lerp(col, mix(col, rgb('#2a55b5'), 0.35), 0.45 * smoothstep(cx - rx * 0.4, cx - rx, x) * smoothstep(0.1, -0.3, lam))
    study = lerp(study, col, m)
    lemon_m = np.maximum(lemon_m, m)
    lemon_lam = np.where(m > 0.5, lam, lemon_lam)
    lu, lv = ((x - cx) * c + (y - cy) * s) / rx, (-(x - cx) * s + (y - cy) * c) / ry
    lflow = ang + np.arctan(-lv * lu / (1 - np.clip(lu, -0.95, 0.95) ** 2 + 0.25) * ry / rx)   # tip to tip, bowed
    lemon_masks.append((m, lflow))

# the cut half: the face tipped toward us and up to the light
c, s = np.cos(angc), np.sin(angc)
back_sd = sd_ellipse(x, y, cxc + 0.022, cyc + 0.016, rxc * 0.98, ryc * 0.98, angc)
back_m = fill(back_sd, 0.0025)
nb = ellipsoid(cxc + 0.022, cyc + 0.016, rxc, ryc, angc)
backc, _ = form(nb, '#e9c02a', '#c9981a', '#7c5f12', '#b98f2a', refl_amt=0.7)
face_sd = sd_ellipse(x, y, cxc, cyc, rxc * 0.93, ryc * 0.93, angc)
face_m = fill(face_sd, 0.002)
fu, fv = ((x - cxc) * c + (y - cyc) * s) / (rxc * 0.93), (-(x - cxc) * s + (y - cyc) * c) / (ryc * 0.93)
fr, fa = np.hypot(fu, fv), np.arctan2(fv, fu)
seg = np.abs(np.sin(fa * 4.5 + 0.4))                                    # nine segments
pulp = lerp(rgb('#e9bd28'), rgb('#f6dd5e'), smoothstep(-0.8, 0.9, p.noise(0.02, 2, seed=30) + 0.9 * np.sin(fr * 9)))
pulp = lerp(pulp, rgb('#d9a81c'), 0.5 * smoothstep(0.3, 0.9, fr) * smoothstep(0.2, 0.9, fu * 0.6 + fv * 0.8 + 0.3))
face = lerp(rgb('#fbf3c8'), pulp, smoothstep(0.10, 0.34, seg) * smoothstep(0.80, 0.72, fr) * smoothstep(0.07, 0.16, fr))
face = lerp(face, rgb('#f3d24a'), smoothstep(0.90, 0.99, fr))            # zest at the very edge
face = lerp(face, rgb('#fffdf0'), 0.8 * smoothstep(-0.4, 0.9, p.noise(0.006, 1, seed=31)) *
            np.exp(-(((fu + 0.35) / 0.25) ** 2 + ((fv + 0.3) / 0.22) ** 2)))   # wet sparkle toward the light
study = lerp(study, backc, back_m)
study = lerp(study, face, face_m)
cut_m = np.maximum(back_m, face_m)

# contact: the dark line where each thing meets the table
occ = np.zeros_like(x)
for cx, cy, rx, ry, ang in LEMONS:
    occ = np.maximum(occ, fill(sd_ellipse(x, y, cx + rx * 0.12, cy + ry * 0.93, rx * 0.62, ry * 0.11), 0.006))
occ = np.maximum(occ, fill(sd_ellipse(x, y, cxc + 0.03, cyc + ryc * 0.98, rxc * 0.6, ryc * 0.1), 0.006))
occ = np.maximum(occ, fill(sd_ellipse(x, y, JX + 0.02, JBOT + 0.021, 0.082, 0.010), 0.006))
objs = np.clip(jug_m + lemon_m + cut_m + handle_m + spout_m, 0, 1)
study = lerp(study, rgb('#1d1820'), 0.85 * occ * (1 - objs))

# lost edges: where a shadow side meets a dark of the same value, let the edge go
edge_band = smoothstep(0.022, 0.0, np.abs(jug_sd)) * smoothstep(0.35, 0.9, uj) * (y < TABLE + 0.02) * (y > JTOP + 0.06)
edge_band = np.maximum(edge_band, 0.8 * smoothstep(0.02, 0.0, np.abs(handle_sd)) * smoothstep(JX + 0.10, JX + 0.20, x))
study = lerp(study, blur(study, 0.012 * H), np.clip(edge_band, 0, 1))
study = np.clip(study, 0, 1).astype(F32)

if a.study:
    Image.fromarray((study * 255).astype(np.uint8)).save(a.study)

# ---- painting it ------------------------------------------------------------------------
one = np.ones_like(x)
bg_m = (y < TABLE + 0.01).astype(F32) * (1 - objs)
table_m = (y > TABLE - 0.005).astype(F32) * (1 - cloth_m)
jug_all = np.clip(jug_m + handle_m + spout_m, 0, 1)

# 1. lay-in: thin, scrubbed, the big light and dark shapes only
p.strokes(one, blur(study, 0.025 * H), flow=0.6 + 0.5 * p.noise(0.6, 2, seed=50), size=0.09, length=3.5, load=0.8,
          opacity=0.9, pickup=0.0, dry=0.35, thickness=0.3, jitter=0.06, name='lay-in')
# 2. background: broad, loose, left thin; warmer strokes broken into the cool
p.mass(bg_m, study, flow=0.95 + 0.35 * p.noise(0.7, 2, seed=51), sizes=(0.07, 0.04), length=3.5, pickup=0.6,
       jitter=0.045, hue=0.05, marble=0.45, wobble=0.08, overshoot=0.5, tol=0.035, thickness=0.55, dry=0.15,
       soft=0.2, name='background')
# 3. table and cloth: the wood along its grain, the linen along its folds
p.mass(table_m, study, flow=0.03 * p.noise(0.3, 2, seed=52), sizes=(0.035, 0.016), length=7, pickup=0.5, jitter=0.07,
       hue=0.04, tol=0.06, overshoot=0.2, name='table')
cloth_flow = along(fold_long + 0.08 * p.noise(0.3, 2, seed=55), 0.035 * H)
p.mass(cloth_m * (1 - objs), study, flow=cloth_flow, sizes=(0.045, 0.022, 0.010), length=5, pickup=0.5, jitter=0.03,
       hue=0.025, marble=0.5, wobble=0.1, ctol=0.10, tol=0.04, overshoot=0.1, clip=0.6, soft=0.16, thickness=1.1, name='linen')
p.dry(0.5)
# 4. the jug: strokes wrap the belly, following the turn of the light
jug_flow = along(blur(np.clip(uj, -1.2, 1.2), 0.008 * H)) + 0.25 * p.noise(0.2, 2, seed=56)
p.mass(jug_all, study, flow=np.where(jug_m > 0.5, jug_flow, along(handle_sd, 0.004 * H)), sizes=(0.034, 0.016, 0.007),
       length=5, pickup=0.45, jitter=0.05, hue=0.03, marble=0.6, ctol=0.16, tol=0.045, clip=1.0, overshoot=0.0,
       soft=0.14, thickness=1.2, name='jug')
# 5. lemons: round the form, thicker paint in the lights
for m, lflow in lemon_masks:
    p.mass(m, study, flow=lflow, sizes=(0.026, 0.012, 0.006), length=3, pickup=0.4,
           jitter=0.04, hue=0.025, ctol=0.14, tol=0.04, clip=0.9, overshoot=0.0, soft=0.14, thickness=1.5,
           name='lemon')
p.mass(cut_m, study, flow=np.where(face_m > 0.5, np.arctan2(y - cyc, x - cxc), around(x, y, cxc, cyc)),
       sizes=(0.016, 0.007, 0.0035), length=3, pickup=0.3, jitter=0.03, hue=0.02, ctol=0.16, tol=0.035, clip=0.9,
       overshoot=0.0, thickness=1.5, name='cut lemon')
# cutting in: the ground is painted back up to the objects, which is what makes their outlines
near = smoothstep(0.0, 0.5, blur(objs, 0.012 * H)) * (1 - objs)
p.strokes(near, study, flow=along(blur(objs, 0.01 * H)), size=0.010, length=4, pickup=0.35, clip=1.0, overshoot=0.0,
          density=1.5, ctol=0.12, jitter=0.03, soft=0.12, name='cutting in')
# 6. come back over the whole picture with a small brush where it still asks
p.strokes(one, study, flow=np.where(jug_m > 0.5, jug_flow, cloth_flow), size=0.0055, length=3.5, pickup=0.3,
          refine=0.07, ctol=0.18, thickness=1.2, name='touches')
p.dry(0.7)
# 7. accents and highlights: thick, clean, put down once
p.strokes(np.clip(occ * 1.5, 0, 1) * (1 - objs), study, flow=0.0, size=0.007, length=5, pickup=0.3, ctol=0.3,
          name='contact shadows')
hl = np.clip(win * jug_m * (1 - 0.9 * bar), 0, 1)
p.strokes(hl, '#fdfcf4', flow=jug_flow, size=0.011, length=2.2, pickup=0.05, load=1.5, thickness=3.2, jitter=0.015,
          overshoot=0.0, clip=1.0, density=1.4, name='window on the glaze')
marks_hl, sizes_hl = [], []
for cx, cy, rx, ry, ang in LEMONS:
    marks_hl.append(curve([(cx - rx * 0.50, cy - ry * 0.44), (cx - rx * 0.36, cy - ry * 0.56), (cx - rx * 0.16, cy - ry * 0.58)]))
    sizes_hl.append(ry * 0.15)
p.marks(marks_hl, '#fff7c4', size=sizes_hl, pickup=0.4, load=1.4, thickness=2.8, taper=0.5, soft=0.2,
        name='lemon lights')
p.marks([curve([(JX - 0.056, JTOP + 0.008), (JX - 0.02, JTOP + 0.001), (JX + 0.02, JTOP + 0.001)])], '#9db4ec',
        size=0.0045, pickup=0.45, thickness=1.6, taper=0.6, soft=0.2, name='rim light')
p.marks([[(cxc - rxc * 0.45, cyc - ryc * 0.42), (cxc - rxc * 0.25, cyc - ryc * 0.5)],
         [(cxc - rxc * 0.1, cyc - ryc * 0.2), (cxc - rxc * 0.0, cyc - ryc * 0.24)]], '#fffef2', size=0.006,
        pickup=0.0, load=1.4, thickness=3.0, name='wet pulp')

spokes = [[(cxc + 0.10 * rxc * np.cos(t), cyc + 0.10 * ryc * np.sin(t)),
           (cxc + 0.93 * 0.74 * (rxc * np.cos(t) * c - ryc * np.sin(t) * s),
            cyc + 0.93 * 0.74 * (rxc * np.cos(t) * s + ryc * np.sin(t) * c))]
          for t in (np.arange(9) + 0.5) * 2 * np.pi / 9 + (-0.4) / 4.5]
p.marks(spokes, '#fbf3cc', size=0.0032, pickup=0.25, load=1.0, thickness=1.2, taper=0.5, name='segments')
ring = [(cxc + 0.86 * (rxc * np.cos(t) * c - ryc * np.sin(t) * s), cyc + 0.86 * (rxc * np.cos(t) * s + ryc * np.sin(t) * c))
        for t in np.linspace(2.2, 2.2 + 2 * np.pi, 48)]
p.marks([ring], '#fbf4d2', size=0.0045, pickup=0.2, load=1.3, thickness=1.6, taper=0.1, name='pith')

p.save(a.out)
if a.film:
    p.film(a.film)
if a.tape:
    from tape import save_tape
    print(f"wrote {a.tape}  {save_tape(p, a.tape, title='Lemons and a blue jug') / 1e6:.2f} MB")
