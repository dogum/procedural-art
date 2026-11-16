"""harbour: painted in sittings with sitting.py."""
import argparse
import os
import sys

import numpy as np

# the engine: beside the skill (this file in studies/), $ATELIER, or where it was when bundled
for _d in (os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'),
           os.environ.get('ATELIER', '')):
    if _d and os.path.exists(os.path.join(_d, 'atelier.py')):
        sys.path.insert(0, _d)
        break
from atelier import *  # noqa: E402,F401,F403

ap = argparse.ArgumentParser()
ap.add_argument('--out', default='painting.png')
ap.add_argument('--height', type=int, default=800)
ap.add_argument('--film', default=None)
ap.add_argument('--study', default=None)
ap.add_argument('--tape', default=None)
ap.add_argument('--palette', default='ultramarine,burnt sienna,yellow ochre,cadmium orange,alizarin crimson,titanium white', help='tube colours, comma-separated, or full')
ap.add_argument('--knife', action='store_true', help='lay every pass with a palette knife')
ap.add_argument('--seed', type=int, default=5)
a = ap.parse_args()

H = a.height
W = int(round(H * 1200 / 800))
PALETTE = None if a.palette == 'full' else [t.strip() for t in a.palette.split(',')]
p = Painting(W, H, seed=a.seed, ground='#8a6f5a', palette=PALETTE, film=bool(a.film),
             tape=bool(a.tape), knife=a.knife or False)
x, y = p.x, p.y

# ---- study ------------------------------------------------------------
"""Harbour at first light. Contre-jour: the sun is just up behind a veil of haze, low and left of
centre, so the sky and the water glow and everything standing in front of them (the quay, three
moored boats, the headland) is a cool dark shape with a warm edge. Six tubes: ultramarine, burnt
sienna, yellow ochre, cadmium orange, alizarin crimson, white. The eye should land on the big
boat's cabin, where the darkest dark meets the path of light on the water."""
import scipy.ndimage as ndi

HOR = 0.47
SUN = (0.52, 0.415)
xi = np.broadcast_to(np.arange(W), (H, W))

# ---- sky: pale gold at the horizon, rose above it, grey-violet overhead; flat bars of cloud lit from below
t = smoothstep(0.0, HOR, y)
sky = lerp(rgb('#6f7fa6'), rgb('#a99bb5'), smoothstep(0.0, 0.45, t))
sky = lerp(sky, rgb('#e9b9a0'), smoothstep(0.35, 0.78, t))
sky = lerp(sky, rgb('#f6dca0'), smoothstep(0.70, 1.0, t))
dsun = np.hypot((x - SUN[0]) * 0.8, y - SUN[1])
sky = lerp(sky, rgb('#fff0c0'), 0.85 * np.exp(-dsun / 0.11))
bars = smoothstep(0.35, 0.75, p.noise(0.5, 3, seed=1) * 0.5 + 0.9 * np.sin(y * 34 + 2.2 * p.noise(0.9, 2, seed=2) + 0.6 * x)
                  * smoothstep(0.42, 0.10, y) + 0.1) * smoothstep(0.02, 0.10, HOR - y)
under = smoothstep(-0.006, 0.012, ndi.sobel(bars, axis=0) * -1)            # the lit underside of each bar
cloud = lerp(rgb('#7d6f8f'), rgb('#f2a67c'), under * smoothstep(0.0, 0.5, t))
sky = lerp(sky, cloud, 0.8 * bars)
sun_disc = fill(sd_circle(x, y, SUN[0], SUN[1], 0.016), 0.004)
sky = lerp(sky, rgb('#fffbe6'), sun_disc)

# ---- far shore: a headland with a town and a lighthouse, two tones of haze
ridge1 = HOR - 0.028 - 0.05 * smoothstep(0.75, 1.35, x) - 0.012 * p.noise(0.25, 3, seed=3) * smoothstep(0.7, 1.0, x)
ridge1 = np.where(x < 0.72, HOR + 0.01, ridge1)
ridge2 = HOR - 0.012 - 0.022 * smoothstep(0.0, 0.5, 0.45 - x) - 0.006 * p.noise(0.2, 3, seed=4)
far_m = ((y > ridge1) & (y < HOR)).astype(F32)
far2_m = ((y > ridge2) & (y < HOR) & (x < 0.5)).astype(F32)
far_c = lerp(rgb('#8f8aa6'), rgb('#6b6f93'), smoothstep(0.8, 1.5, x))
study = lerp(sky, rgb('#b3a3b3'), far2_m)
study = lerp(study, far_c, far_m)
LH = (1.19, float(HOR - 0.085))                                             # lighthouse
lh = sd_segment(x, y, LH[0], LH[1] - 0.05, LH[0], HOR - 0.07, 0.006, 0.009)
study = lerp(study, rgb('#d8c4c0'), fill(lh, 0.002))
study = lerp(study, rgb('#fff1b8'), fill(sd_circle(x, y, LH[0], LH[1] - 0.052, 0.006), 0.002))
hs = np.floor(x * 95 + 3 * np.sin(x * 7))
town = (np.sin(hs * 5.1) > -0.3) & (y > HOR - 0.010 - 0.012 * (0.5 + 0.5 * np.sin(hs * 12.9))) & (y < HOR) & (x > 0.9) & (x < 1.36)
study = np.where(town[..., None], lerp(rgb('#62648a'), rgb('#cfb0a4'), (np.sin(hs * 7.3) > 0.3) * 0.55), study)

# ---- the water: the sky upside down, a path of light under the sun, darker and bluer toward us
above = study.copy()
depth = smoothstep(HOR, 1.0, y)
yy = np.clip(((2 * HOR - y) * 0.92 + 0.08 * HOR) * H, 0, H - 1).astype(int)
rip = (p.noise(0.02, 2, seed=5) * 0.012 * H * (0.3 + depth * 2.5)).astype(int)
water = above[yy, np.clip(xi + rip, 0, W - 1)]
water = ndi.uniform_filter1d(water, max(3, int(0.03 * H)), axis=0, mode='nearest')
water = lerp(water, rgb('#4a5f8a'), 0.15 + 0.5 * depth ** 1.3)
swell = p.ground_noise(HOR, 0.5, 9.0, seed=6, octaves=2)
swell2 = p.ground_noise(HOR, 1.6, 30.0, seed=7, octaves=2)
water = water * (1 + 0.10 * swell[..., None] * depth[..., None] + 0.05 * swell2[..., None])
pathw = 0.022 + 0.22 * (y - HOR)
glit = np.exp(-((x - SUN[0] - 0.03 * (y - HOR)) / pathw) ** 2) * (y > HOR)
spark = smoothstep(0.1, 0.9, swell2 + 0.6 * swell) * (0.35 + 0.65 * smoothstep(0.0, 0.12, y - HOR))
water = lerp(water, rgb('#ffe9ae'), np.clip(glit * (0.55 + 0.95 * spark), 0, 1) * (1 - 0.3 * depth))
study = np.where((y > HOR)[..., None], water, study)

# ---- the quay: a dark stone arm from the left, a beacon on its end
qy0, qy1 = 0.565, 0.60
quay = sd_poly(x, y, [(-0.1, qy0 - 0.03), (0.33, qy0 - 0.012), (0.345, qy0), (0.33, qy1), (-0.1, qy1 + 0.04)])
quay_m = fill(quay, 0.003)
qcol = lerp(rgb('#3a3550'), rgb('#57465a'), smoothstep(-1, 1, p.noise(0.05, 3, seed=8)))
qcol = lerp(qcol, rgb('#e2a070'), 0.8 * smoothstep(0.008, 0.0, np.abs(quay + 0.004)) * (y < qy0))    # light along the top
beacon = np.minimum(sd_segment(x, y, 0.315, qy0 - 0.012, 0.315, qy0 - 0.075, 0.006, 0.004), sd_box(x, y, 0.315, qy0 - 0.082, 0.009, 0.008))
bollards = np.minimum.reduce([sd_box(x, y, bx_, qy0 - 0.03 + 0.045 * (bx_ / 0.33) * 0.4, 0.006, 0.011) for bx_ in (0.05, 0.13, 0.22)])


def boat(bx, wy, L, h, hull, band, cabin, tilt=0.0, flip=1):
    """A moored fishing boat: returns (mask, colour, rigging lines, top of the mast)."""
    X = lambda u: bx + flip * u * L
    Y = lambda v: wy - v * h + tilt * 0
    hull_sd = sd_poly(x, y, [(X(-0.50), Y(0.75)), (X(-0.10), Y(0.62)), (X(0.36), Y(0.80)), (X(0.56), Y(1.32)),
                             (X(0.42), Y(0.0)), (X(-0.44), Y(0.0))]) - 0.004 * L
    cab_sd = sd_box(x, y, X(-0.16), Y(1.22), 0.13 * L, 0.52 * h, r=0.01 * L)
    roof_sd = sd_box(x, y, X(-0.15), Y(1.80), 0.16 * L, 0.07 * h)
    m_h, m_c, m_r = fill(hull_sd, 0.0025), fill(cab_sd, 0.0025), fill(roof_sd, 0.002)
    vh = np.clip((wy - y) / h, 0, 1.4)
    col = lerp(rgb(hull) * 0.72, rgb(hull), smoothstep(0.0, 0.9, vh))
    col = np.where(((vh > 0.52) & (vh < 0.68))[..., None], rgb(band), col)               # the painted band
    col = lerp(col, rgb('#f0b078'), 0.75 * smoothstep(0.012 * L, 0.0, np.abs(hull_sd + 0.004 * L)) * (vh > 0.6))  # sun on the gunwale
    ccol = lerp(rgb(cabin), rgb(cabin) * 0.6, smoothstep(-0.02 * L, 0.10 * L, x - X(-0.2)))
    ccol = lerp(ccol, rgb('#f3bd8a'), 0.85 * smoothstep(0.014 * L, 0.0, np.abs(cab_sd + 0.004 * L)) * (x < X(-0.16) - 0.0 * L) * (cab_sd < 0))   # the sunward edge
    win = (np.abs(((x - X(-0.27)) / (0.075 * L)) % 1 - 0.5) < 0.3) & (np.abs(y - Y(1.36)) < 0.13 * h) & (cab_sd < -0.012 * L)
    ccol = np.where(win[..., None], rgb('#2a2c48'), ccol)
    col = lerp(col, ccol, m_c)
    col = lerp(col, rgb('#2d2a40'), m_r)
    mask = np.clip(m_h + m_c + m_r, 0, 1)
    mast_top = (X(0.10), Y(4.6))
    lines = [[(X(0.10), Y(0.9)), mast_top],                                  # mast
             [(X(0.10), Y(2.5)), (X(-0.42), Y(2.05))],                       # boom
             [mast_top, (X(0.54), Y(1.3))], [mast_top, (X(-0.48), Y(0.8))],  # stays
             [(X(-0.02), Y(1.9)), (X(-0.02), Y(3.1))]]                       # aerial
    return mask, col, lines, mast_top


BOATS = [boat(0.93, 0.80, 0.50, 0.105, '#22344f', '#a2382c', '#8f8794'),              # the big one, near
         boat(0.50, 0.655, 0.26, 0.050, '#a79fa6', '#3f5f7e', '#6a5348', flip=-1),    # white hull, mid
         boat(1.30, 0.60, 0.15, 0.028, '#4d4a66', '#8a4a40', '#857c8a')]              # far
WL = [0.80, 0.655, 0.60]
boats_m = np.zeros_like(x)
refl_m = np.zeros_like(x)
for (bm, bc, bl, mt), wy in zip(BOATS[::-1], WL[::-1]):
    fy = np.clip(((2 * wy - y) * H).astype(int), 0, H - 1)
    rr = (p.noise(0.012, 2, seed=11) * 0.010 * H * (0.5 + 3 * np.clip(y - wy, 0, 1))).astype(int)
    rm = ndi.uniform_filter1d(bm[fy, np.clip(xi + rr, 0, W - 1)], max(3, int(0.02 * H)), axis=0) * (y > wy)
    rc = bc[fy, np.clip(xi + rr, 0, W - 1)]
    brk = smoothstep(-0.7, 0.4, swell2 + 0.5 * swell)
    rm = rm * (0.55 + 0.45 * brk) * np.exp(-np.maximum(y - wy, 0) / 0.22)
    study = lerp(study, rc * 0.62 + 0.10 * rgb('#3a4a78'), 0.85 * rm)
    refl_m = np.maximum(refl_m, rm)
    study = lerp(study, bc, bm)
    boats_m = np.maximum(boats_m, bm)
# quay last of the middle ground (it stands in front of the white boat's stern? no: behind it), with its reflection
qr = ndi.uniform_filter1d(quay_m[np.clip(((2 * qy1 - y) * H).astype(int), 0, H - 1), np.clip(xi + rip, 0, W - 1)], max(3, int(0.03 * H)), axis=0) * (y > qy1)
study = lerp(study, rgb('#3d4468'), 0.6 * qr * (1 - boats_m))
study = lerp(study, qcol, quay_m * (1 - BOATS[1][0]))
study = lerp(study, rgb('#2c2a44'), fill(np.minimum(beacon, bollards), 0.002) * (1 - BOATS[1][0]))
study = lerp(study, rgb('#ff8a4a'), fill(sd_circle(x, y, 0.315, qy0 - 0.094, 0.006), 0.002))
study = np.clip(study, 0, 1).astype(F32)

# masks and directions the sittings share
sky_m = (y < HOR).astype(F32) * (1 - far_m) * (1 - far2_m)
water_m = (y >= HOR).astype(F32)
land_m = np.clip(far_m + far2_m, 0, 1)
objs_m = np.clip(boats_m + quay_m, 0, 1)
gu, gv, near = ground_plane(x, y, HOR, cx=SUN[0])
sky_flow = 0.05 * np.sin(x * 2.0 + 0.5) + 0.10 * p.noise(0.6, 2, seed=20)

if a.study:
    from PIL import Image
    Image.fromarray((np.clip(study, 0, 1) * 255).astype(np.uint8)).save(a.study)

# ---- 10_lay_in ------------------------------------------------------------
# Thin and fast over the whole canvas with the biggest brush: the value pattern, nothing else.
p.strokes(np.ones_like(x), blur(study, 0.025 * H), flow=0.15 * p.noise(0.6, 2, seed=30), size=0.09, length=4,
          load=0.8, opacity=0.9, pickup=0.0, dry=0.35, thickness=0.3, jitter=0.06, name='lay-in')

# ---- 20_sky ------------------------------------------------------------
# The sky: long level sweeps. (First try fanned toward the sun and went muddy with too much
# pickup; now the strokes lie level like the cloud bars and the bars get their own smaller pass.)
p.mass(sky_m, study, flow=sky_flow, sizes=(0.05, 0.022, 0.01), length=10, pickup=0.4, soft=0.2, jitter=0.03,
       hue=0.025, overshoot=0.5, tol=0.04, ctol=0.07, name='sky')
p.strokes(sky_m * under * bars, study, flow=sky_flow, size=0.008, length=9, pickup=0.25, load=1.15, thickness=1.5,
          refine=0.035, ctol=0.06, name='cloud undersides')
p.mass(land_m, study, flow=0.0, sizes=(0.02, 0.008), length=6, pickup=0.5, soft=0.2, overshoot=0.2, tol=0.04,
       jitter=0.025, name='far shore')

# ---- 30_water ------------------------------------------------------------
# The water: the sky's colours dragged straight down while everything is wet, then the surface
# laid across them in level strokes that get shorter and finer toward the horizon.
p.mass(water_m, study, flow=np.pi / 2, sizes=(0.045, 0.02), length=7, pickup=0.65, soft=0.22, jitter=0.04, hue=0.03,
       ctol=0.16, tol=0.05, overshoot=0.0, name='water, dragged')
p.strokes(water_m, study, flow=0.0, size=0.004 + 0.022 * near, length=9, density=0.75, pickup=0.45, load=0.9, dry=0.2,
          ctol=0.07, jitter=0.04, hue=0.03, thickness=1.1, name='water, surface')
p.strokes(water_m, study, flow=0.0, size=0.003 + 0.008 * near, length=8, pickup=0.3, refine=0.05, ctol=0.06,
          jitter=0.03, name='water, drawing')

# ---- 40_boats ------------------------------------------------------------
# The dark shapes that stand in front of the light: quay, then boats far to near. Small brushes,
# cut at the outline, thin paint (darks stay thin). Then the water is cut back in around them.
p.dry(0.5)
p.mass(quay_m, study, flow=0.04, sizes=(0.018, 0.007), length=6, pickup=0.25, clip=0.9, overshoot=0.0, ctol=0.2,
       tol=0.045, jitter=0.05, hue=0.04, thickness=0.7, name='quay')
cab_flow = np.where(y < 0.74, np.pi / 2, 0.03)                       # cabins upright, hulls along the sheer
p.mass(boats_m, study, flow=cab_flow, sizes=(0.014, 0.006, 0.003), length=5, pickup=0.2, clip=0.9, overshoot=0.0,
       ctol=0.16, tol=0.04, jitter=0.035, hue=0.03, thickness=0.8, name='boats')
p.cut_in(objs_m, study, width=0.012, size=0.006)
p.strokes(refl_m * water_m * (1 - objs_m), study, flow=np.pi / 2, size=0.008, length=6, pickup=0.55, soft=0.2,
          refine=0.045, ctol=0.14, name='reflections')
p.strokes(refl_m * water_m * (1 - objs_m), study, flow=0.0, size=0.004, length=9, density=0.5, pickup=0.4, dry=0.25,
          ctol=0.06, name='reflections, cut across')

# ---- 50_light ------------------------------------------------------------
# Last: the light itself, in thick clean paint, and the drawn lines. The glitter goes on with a
# knife (flat flakes that skip), the sun and lamps as single dabs, masts and stays in one pull each.
p.dry(0.7)
rs = np.random.default_rng(9)
gl = np.clip(glit * (0.45 + spark) * 1.25 - 0.3, 0, 1) * water_m * (1 - objs_m) * (1 - 0.85 * refl_m)
p.strokes(gl, '#ffeeb8', flow=0.0, size=0.0025 + 0.011 * near, length=6, density=0.7, load=1.0, dry=0.4,
          pickup=0.1, thickness=2.4, jitter=0.04, hue=0.03, size_var=0.45, overshoot=0.0, name='glitter')
p.strokes(np.clip(gl * 1.5 - 0.7, 0, 1), '#fffbe6', flow=0.0, size=0.002 + 0.006 * near, length=5, density=0.5,
          load=1.3, dry=0.3, pickup=0.0, thickness=3.0, overshoot=0.0, name='glitter, whitest')
p.dabs([SUN], "#fffbe8", size=0.046, thickness=3.2, soft=0.2)
p.strokes(np.clip(np.exp(-dsun / 0.05) - 0.2, 0, 1) * sky_m, '#fff2c4', flow=0.0, size=0.012, length=5, density=0.5, opacity=0.6,
          pickup=0.5, soft=0.25, dry=0.2, name='veil round the sun')
for (bm, bc, bl, mt), sz in zip(BOATS, (0.0055, 0.0032, 0.002)):
    p.marks(bl[:2], '#2a2740', size=[sz, sz * 0.8], pickup=0.2, taper=0.2, name='mast and boom')
    p.marks(bl[2:], '#4a4460', size=sz * 0.4, pickup=0.3, taper=0.1, load=0.9, name='stays')
    p.marks([[(mt[0] - 0.002, mt[1] + 0.02), (mt[0] - 0.002, mt[1] + 0.10 * sz / 0.0055)]], '#f0b27c', size=sz * 0.45,
            pickup=0.1, name='sun on the mast')
# mast reflections: broken verticals
p.marks([[(0.93 + 0.05, 0.86 + i * 0.03), (0.93 + 0.05 + 0.004 * np.sin(i * 2.1), 0.878 + i * 0.03)] for i in range(4)],
        '#2f3052', size=0.005, pickup=0.5, soft=0.2, name='mast reflection')
p.dabs([(0.315, qy0 - 0.094)], 'cadmium orange', size=0.012, thickness=2.6)
p.dabs([(LH[0], LH[1] - 0.052)], '#fff3c4', size=0.008, thickness=2.4)
p.marks([[(LH[0], LH[1] - 0.045), (LH[0], HOR - 0.068)]], '#e3d2cc', size=0.006, pickup=0.3, name='lighthouse')
# gulls: two-stroke ticks, dark against the bright sky
gulls = [(0.70, 0.30, 0.016), (0.78, 0.26, 0.012), (0.64, 0.335, 0.009), (1.02, 0.22, 0.010)]
p.marks([c for gx_, gy_, s_ in gulls for c in (curve([(gx_ - s_, gy_ - 0.2 * s_), (gx_ - 0.5 * s_, gy_ - 0.5 * s_), (gx_, gy_)], 6),
                                               curve([(gx_, gy_), (gx_ + 0.5 * s_, gy_ - 0.55 * s_), (gx_ + s_, gy_ - 0.15 * s_)], 6))],
        '#3a3552', size=0.0022, pickup=0.1, taper=0.5, name='gulls')

p.save(a.out)
if a.film:
    p.film(a.film)
if a.tape:
    import tape
    tape.save_tape(p, a.tape)
