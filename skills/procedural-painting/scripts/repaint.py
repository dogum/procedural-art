"""Paint an existing picture (a terrain render, a path-traced still life, a photo) on the atelier
canvas: the image supplies the colours, the engine supplies pigment mixing, bristles and relief.

    python repaint.py --in render.png --out painted.png [--height 1600] [--looseness 1.0]
                      [--ground "burnt sienna"] [--film process.mp4] [--seed 1]
                      [--geom WORK/geom_HASH | --aov BANDS/aov] [--push 1.0]

With --aov (the folder pbr_terrain.py --aov writes inside its bands folder) the painter knows
the ground under every pixel of a terrain render: which way is downhill, how steep it is, where
the snow, the water and the cloud are. Slopes are painted with strokes that run down the fall
lines, plains with level strokes, snow goes on late in thick clean paint, water is dragged
straight down and cut across, and the skyline is cut in against the sky.

With --geom (the folder pbr_terrain.py --geom-cache leaves in its --work folder) the painter
knows the depth of every pixel of a terrain render: the sky is swept in first as its own mass,
far ground is kept simple and soft, near ground gets broken colour in flat strokes, and the
mountain is drawn over the sky with a found edge. --push is the painter's eye: more chroma,
warmer lights, cooler shadows (0 keeps the picture's own colour).

Brush direction follows the picture's edges where it has any, and sweeps in long gentle arcs
where it has none (skies, still water). For a picture composed from nothing, write a study
script instead (see ../studies and ../SKILL.md): there the brush direction is chosen per mass.
"""
import argparse
import glob
import os

import numpy as np

from atelier import Painting, along, blur, image_flow, load_image, lerp, smoothstep, F32, _resize


def load_aov(folder, W, H):
    """The painter buffers of a pbr_terrain.py --aov render, stitched and resized to W x H."""
    files = sorted(glob.glob(os.path.join(folder, '[0-9]*.npz')))
    if not files:
        raise SystemExit(f"no painter buffers in {folder} (render with pbr_terrain.py --aov)")
    rows = [(int(os.path.basename(f)[:5]), np.load(f)) for f in files]
    w0 = rows[0][1]['hit'].shape[1]
    h0 = max(r + d['hit'].shape[0] for r, d in rows)
    if sum(d['hit'].shape[0] for _, d in rows) != h0:
        raise SystemExit(f"{folder} has missing rows: finish the render first")
    if abs(w0 / h0 - W / H) > 0.02:
        raise SystemExit("the painter buffers and the image have different shapes")
    A = {}
    for key in ('hit', 'fall', 'slope', 'elev', 'snow', 'water', 'cloud', 'near', 'depth'):
        full = np.zeros((h0, w0), F32)
        for r, d in rows:
            v = d[key].astype(F32)
            full[r:r + v.shape[0]] = np.where(np.isfinite(v), v, 0)
        A[key] = full
    hit = A['hit'] > 0.5
    lt = np.where(hit, np.log(np.maximum(A['depth'], 1e-3)), np.nan)
    lo, hi = np.nanpercentile(lt, [2, 98])
    A['near'] = np.clip(A['near'], 0, 1)
    A['easel'] = 1 - np.clip((np.nan_to_num(lt, nan=hi) - lo) / (hi - lo + 1e-6), 0, 1)       # 1 = at the painter's feet
    land = (hit & (A['water'] < 0.5)).astype(F32)
    c, s_ = np.cos(A['fall']) * land, np.sin(A['fall']) * land
    flip = np.where(s_ < 0, -1, 1)                                     # downhill is down the picture or level
    A['fx'], A['fy'] = c * flip, s_ * flip
    out = {k: _resize(v, W, H) for k, v in A.items() if k not in ('fall', 'depth')}
    return out


def paint_terrain(p, src, A, k, arcs, L, film_note=None):
    """Sky, far ranges, slopes, plain, water, snow and skyline, each handled as its own mass."""
    H, W = p.H, p.W
    hit = smoothstep(0.3, 0.7, A['hit'])
    water = smoothstep(0.3, 0.7, A['water']) * hit
    sky = 1 - hit
    land = hit * (1 - water)
    cloud = smoothstep(0.15, 0.5, A['cloud']) * sky
    snow = smoothstep(0.35, 0.65, A['snow']) * land
    easel = A['easel']
    # fall lines: smoothed over a brush width so strokes read as planes, not as noise
    s1 = 0.006 * H
    w = blur(land, s1) + 1e-4
    fx, fy = blur(A['fx'], s1) / w, blur(A['fy'], s1) / w
    steep = smoothstep(0.10, 0.38, blur(A['slope'] * land, s1) / w)
    coh = np.clip(np.hypot(fx, fy), 0, 1)                             # agreement of directions under the brush
    sw = steep * smoothstep(0.25, 0.7, coh)
    fall = np.arctan2(fy, fx + 1e-6)
    lean = 0.10 * p.noise(0.25, 2)
    landflow = np.arctan2(sw * np.sin(fall) + (1 - sw) * np.sin(lean), sw * np.cos(fall) + (1 - sw) * np.cos(lean))
    vert = np.pi / 2

    # sky: one swept mass; the cloud is modelled inside it afterwards
    p.mass(sky, src, flow=arcs, sizes=(0.06 * k, 0.03 * k), length=8, pickup=0.6, jitter=0.035, hue=0.03,
           overshoot=0.8, soft=0.2, tol=0.05 * k, name='sky')
    if cloud.max() > 0.2:
        cflow = image_flow(blur(src, 0.004 * H), 0.012 * H)
        p.mass(cloud, src, flow=cflow, sizes=(0.022 * k, 0.010 * k, 0.005 * k), length=6, pickup=0.55, soft=0.25,
               jitter=0.03, overshoot=0.5, tol=0.04 * k, ctol=0.1, name='cloud')
    # far ranges: few values, soft, no texture
    far = land * smoothstep(0.42, 0.22, easel) * (1 - sw)
    p.mass(far, blur(src, 0.003 * H), flow=landflow, sizes=(0.028 * k, 0.012 * k), length=7, pickup=0.5, soft=0.2,
           jitter=0.03, hue=0.025, overshoot=0.0, tol=0.05 * k, ctol=0.1, name='far ranges')
    # slopes: strokes run down the fall lines, one plane of the mountain after another
    slopes = land * np.maximum(sw, smoothstep(0.22, 0.42, easel) * 0) * (1 - snow * 0.6)
    p.mass(slopes, src, flow=landflow, sizes=(0.034 * k, 0.016 * k, 0.008 * k), length=7, pickup=0.45, jitter=0.05,
           hue=0.04, marble=0.7, wobble=0.12, overshoot=0.0, tol=0.05 * k, ctol=0.11, name='slopes')
    # the plain and the ground at the painter's feet: level strokes, bigger and more broken toward the easel
    flats = land * (1 - sw) * smoothstep(0.22, 0.42, easel)
    nearw = smoothstep(0.45, 0.85, easel)
    p.mass(flats * (1 - nearw), src, flow=landflow, sizes=(0.03 * k, 0.014 * k, 0.007 * k), length=7, pickup=0.45,
           jitter=0.05, hue=0.045, tol=0.055 * k, ctol=0.12, name='plain')
    if (flats * nearw).max() > 0.2:
        p.mass(flats * nearw, src, flow=landflow * 0.5, sizes=(0.05 * k, 0.024 * k, 0.012 * k), length=5, pickup=0.4,
               jitter=0.08, hue=0.07, marble=0.8, tol=0.07 * k, ctol=0.16, name='near ground')
        p.strokes(flats * nearw, src, flow=landflow * 0.5, size=0.012 * k, length=7, density=0.3, pickup=0.35,
                  dry=0.35, load=0.8, jitter=0.07, hue=0.06, thickness=1.4, name='broken colour')
    # water: the reflection is dragged straight down, wet into wet, then the surface is cut across it
    if water.max() > 0.2:
        p.mass(water, src, flow=vert + 0.04 * p.noise(0.4, 2), sizes=(0.045 * k, 0.022 * k), length=8, pickup=0.7,
               soft=0.22, jitter=0.035, hue=0.03, overshoot=0.05, clip=0.6, tol=0.06 * k, ctol=0.2, name='water')
        p.strokes(water, src, flow=vert, size=0.010 * k, length=6, pickup=0.6, soft=0.2, refine=0.075 * k, ctol=0.14,
                  clip=0.6, jitter=0.03, name='reflections')
        p.strokes(water, src, flow=0.02 * p.noise(0.3, 2), size=0.007 * k, length=12, density=0.12, pickup=0.45,
                  load=0.8, dry=0.3, jitter=0.04, ctol=0.09, clip=0.8, thickness=1.1, name='ripples')
    # trees and houses near the camera: upright dabs
    if A['near'].max() > 0.3:
        nm = smoothstep(0.3, 0.7, A['near'])
        p.mass(nm, src, flow=vert + 0.25 * p.noise(0.05, 2), sizes=(0.012 * k, 0.005 * k), length=2.5, pickup=0.3,
               jitter=0.07, hue=0.06, overshoot=0.1, tol=0.05 * k, ctol=0.14, thickness=1.3, name='trees')
    # drawing: small strokes where the form still asks for them, still running with the ground
    p.strokes(land, src, flow=landflow, size=0.0035 * k, length=5, pickup=0.3, refine=0.062 * k, ctol=0.08,
              jitter=0.02, name='drawing')
    p.dry(0.7)
    # snow: late, thick and clean, pulled down the fall line; its blue shadows come from the picture
    if snow.max() > 0.2:
        sflow = np.where(coh > 0.2, fall, vert)
        p.mass(snow, src, flow=sflow, sizes=(0.014 * k, 0.006 * k), length=5, pickup=0.12, load=1.3, jitter=0.02,
               hue=0.015, marble=0.3, overshoot=0.05, clip=0.5, tol=0.035 * k, ctol=0.09, thickness=2.3, name='snow')
    # the skyline is a found edge: cut the sky back in against the land
    hard = (A['hit'] < 0.02).astype(F32)                                # certainly sky
    wsky = blur(hard, 0.006 * H) + 1e-4
    skycol = blur(src * hard[..., None], 0.006 * H) / wsky[..., None]   # the sky's own colour right up to the ridge
    skycol = np.where(hard[..., None] > 0, src, skycol)
    spill = smoothstep(0.0, 0.06, blur(hit, 0.025 * H)) * (A['hit'] < 0.5) * (1 - cloud)
    p.strokes(spill, skycol, flow=arcs, size=0.014 * k, length=4, pickup=0.0, clip=1.0, overshoot=0.0, refine=0.04,
              jitter=0.015, hue=0.01, ctol=0.2, name='sky back over the land')
    edge = smoothstep(0.01, 0.2, blur(hit, 0.008 * H)) * (A['hit'] < 0.5) * (1 - cloud)
    p.strokes(edge, skycol, flow=along(blur(hit, 0.012 * H)), size=0.011 * k, length=5, density=1.5, pickup=0.0,
              clip=1.0, overshoot=0.0, jitter=0.015, hue=0.01, ctol=0.2, soft=0.04, name='skyline')
    return land


def painters_eye(src, push=1.0):
    """More chroma, warmer lights, cooler shadows: how a painter would key the picture."""
    if push <= 0:
        return src
    L = src @ np.array([0.299, 0.587, 0.114], F32)
    g = L[..., None]
    src = np.clip(g + (src - g) * (1 + 0.3 * push), 0, 1)                         # chroma
    warm = smoothstep(0.45, 0.85, L)[..., None] * push
    cool = smoothstep(0.5, 0.15, L)[..., None] * push
    src = src * (1 + warm * np.array([0.06, 0.02, -0.07], F32)) + cool * np.array([-0.025, -0.005, 0.05], F32)
    src = np.clip(0.5 + (src - 0.5) * (1 + 0.1 * push), 0, 1).astype(F32)

    return src


def paint_picture(p, src, k=1.0, aov=None, geom=None, push=0.0):
    """Paint the image `src` (H x W x 3) on the canvas p. aov / geom: folders of terrain buffers."""
    H, W = p.H, p.W
    # direction: along edges where the picture has structure, long arcs where it is empty
    L = src @ np.array([0.299, 0.587, 0.114], F32)
    gy, gx = np.gradient(blur(L, 0.004 * H))
    energy = blur(np.hypot(gx, gy), 0.02 * H)
    has = smoothstep(0.5, 1.6, energy / (np.percentile(energy, 80) + 1e-6))
    arcs = 0.12 * np.sin(p.x * 2.1 + 1.3) + 0.3 * p.noise(0.6, 2)
    ef = image_flow(src, 0.012 * H)
    d = np.where(np.cos(ef) < 0, ef + np.pi, ef)                      # keep directions in one half-plane
    flow = np.arctan2(has * np.sin(d) + (1 - has) * np.sin(arcs), has * np.cos(d) + (1 - has) * np.cos(arcs))

    src = painters_eye(src, push)

    one = np.ones((H, W), F32)
    p.strokes(one, blur(src, 0.02 * H), flow=flow, size=0.08 * k, length=4, load=0.8, opacity=0.9, pickup=0.0,
              dry=0.35, thickness=0.35, jitter=0.06, name='lay-in')
    if aov:
        land = paint_terrain(p, src, load_aov(aov, W, H), k, arcs, L)
    elif geom:
        files = sorted(glob.glob(os.path.join(geom, '*.npz')))
        if not files:
            raise SystemExit(f"no geometry bands in {geom}")
        t = np.concatenate([np.load(f)['t'] for f in files])
        hit = np.concatenate([np.load(f)['hit'] for f in files])
        h0 = int(os.path.basename(files[-1])[:-4].split('_')[1])
        t, hit = t.reshape(h0, -1).astype(F32), hit.reshape(h0, -1)
        if abs(t.shape[1] / h0 - W / H) > 0.02:
            raise SystemExit("the geometry cache and the image have different shapes")
        sky = 1 - smoothstep(0.3, 0.7, _resize(hit.astype(F32), W, H))
        lt = np.log(np.where(hit, np.maximum(t, 1e-3), np.nan))
        lo, hi_ = np.nanpercentile(lt, [2, 98])
        near = 1 - np.clip((_resize(np.nan_to_num(lt, nan=hi_), W, H) - lo) / (hi_ - lo + 1e-6), 0, 1)   # 1 = at the easel
        land = 1 - sky
        flat = np.arctan2((1 - has) * 0.0 + has * np.sin(d), (1 - has) + has * np.cos(d))   # level strokes on plains
        p.mass(sky, src, flow=arcs, sizes=(0.06 * k, 0.03 * k, 0.012 * k), length=8, pickup=0.6, jitter=0.035,
               hue=0.03, overshoot=0.8, soft=0.2, tol=0.05 * k, name='sky')
        far = land * smoothstep(0.55, 0.3, near)
        p.mass(far, src, flow=flat, sizes=(0.03 * k, 0.014 * k, 0.007 * k), length=6, pickup=0.45, jitter=0.04,
               hue=0.035, clip=0.6, overshoot=0.1, tol=0.05 * k, ctol=0.1, name='far ground')
        nearm = land * smoothstep(0.3, 0.55, near)
        p.mass(nearm, src, flow=flat * 0.5, sizes=(0.05 * k, 0.024 * k, 0.012 * k), length=5, pickup=0.4,
               jitter=0.09, hue=0.07, marble=0.8, tol=0.07 * k, ctol=0.16, name='near ground')
        p.strokes(nearm, src, flow=flat * 0.5, size=0.012 * k, length=7, density=0.3, pickup=0.35, dry=0.35,
                  load=0.8, jitter=0.07, hue=0.06, thickness=1.4, name='broken colour')
        p.strokes(far, src, flow=flat, size=0.0035 * k, length=5, pickup=0.3, refine=0.05 * k, ctol=0.08, clip=0.8,
                  jitter=0.02, name='drawing')
    else:
        p.mass(one, src, flow=flow, sizes=(0.045 * k, 0.022 * k, 0.011 * k), length=6, pickup=0.5, jitter=0.05,
               hue=0.035, tol=0.06 * k, ctol=0.12, name='masses')
        p.strokes(one, src, flow=flow, size=0.0055 * k, length=4, pickup=0.35, refine=0.055 * k, ctol=0.1,
                  jitter=0.03, name='drawing')
        p.strokes(has, src, flow=flow, size=0.003 * k, length=4, pickup=0.25, refine=0.05 * k, ctol=0.08,
                  jitter=0.02, thickness=0.8, name='small drawing')
    p.dry(0.6)
    # lights last: thick, clean, only where the picture is brighter than its surroundings
    lum = blur(L, 0.002 * H)
    hi = smoothstep(0.06, 0.16, lum - blur(lum, 0.03 * H)) * smoothstep(0.55, 0.8, lum)
    if geom or aov:
        hi = hi * smoothstep(0.5, 0.9, blur(land, 0.004 * H))                 # not along the skyline
    p.strokes(hi, src, flow=flow, size=0.007 * k, length=2.5, pickup=0.1, load=1.3, thickness=2.4, jitter=0.02,
              ctol=0.12, overshoot=0.0, fade=True, name='lights')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='src', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--height', type=int, default=None, help='output height in pixels (default: the input height)')
    ap.add_argument('--looseness', type=float, default=1.0, help='above 1: bigger brushes, fewer small strokes')
    ap.add_argument('--ground', default='#7a5a40', help='colour of the toned canvas (hex or pigment name)')
    ap.add_argument('--film', default=None)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--geom', default=None, help='geometry cache folder of a pbr_terrain.py --geom-cache render')
    ap.add_argument('--aov', default=None, help='painter buffers folder of a pbr_terrain.py --aov render (<bands>/aov)')
    ap.add_argument('--push', type=float, default=None, help="painter's-eye colour push (default 1 with --geom or --aov, else 0)")
    a = ap.parse_args()

    from PIL import Image
    im = Image.open(a.src)
    H = a.height or im.height
    W = int(round(H * im.width / im.height))
    src = load_image(a.src, W, H)
    p = Painting(W, H, seed=a.seed, ground=a.ground, film=bool(a.film))
    push = a.push if a.push is not None else (1.0 if (a.geom or a.aov) else 0.0)
    paint_picture(p, src, a.looseness, a.aov, a.geom, push)
    p.save(a.out)
    if a.film:
        p.film(a.film)


if __name__ == '__main__':
    main()
