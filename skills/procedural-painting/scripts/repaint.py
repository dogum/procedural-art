"""Paint an existing picture (a terrain render, a path-traced still life, a photo) on the atelier
canvas: the image supplies the colours, the engine supplies pigment mixing, bristles and relief.

    python repaint.py --in render.png --out painted.png [--height 1600] [--looseness 1.0]
                      [--ground "burnt sienna"] [--film process.mp4] [--seed 1]
                      [--geom WORK/geom_HASH] [--push 1.0]

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

from atelier import Painting, blur, image_flow, load_image, lerp, smoothstep, F32, _resize


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
    ap.add_argument('--push', type=float, default=None, help="painter's-eye colour push (default 1 with --geom, else 0)")
    a = ap.parse_args()

    from PIL import Image
    im = Image.open(a.src)
    H = a.height or im.height
    W = int(round(H * im.width / im.height))
    src = load_image(a.src, W, H)
    p = Painting(W, H, seed=a.seed, ground=a.ground, film=bool(a.film))
    k = a.looseness

    # direction: along edges where the picture has structure, long arcs where it is empty
    L = src @ np.array([0.299, 0.587, 0.114], F32)
    gy, gx = np.gradient(blur(L, 0.004 * H))
    energy = blur(np.hypot(gx, gy), 0.02 * H)
    has = smoothstep(0.5, 1.6, energy / (np.percentile(energy, 80) + 1e-6))
    arcs = 0.12 * np.sin(p.x * 2.1 + 1.3) + 0.3 * p.noise(0.6, 2)
    ef = image_flow(src, 0.012 * H)
    d = np.where(np.cos(ef) < 0, ef + np.pi, ef)                      # keep directions in one half-plane
    flow = np.arctan2(has * np.sin(d) + (1 - has) * np.sin(arcs), has * np.cos(d) + (1 - has) * np.cos(arcs))

    push = a.push if a.push is not None else (1.0 if a.geom else 0.0)
    if push > 0:
        g = L[..., None]
        src = np.clip(g + (src - g) * (1 + 0.3 * push), 0, 1)                         # chroma
        warm = smoothstep(0.45, 0.85, L)[..., None] * push
        cool = smoothstep(0.5, 0.15, L)[..., None] * push
        src = src * (1 + warm * np.array([0.06, 0.02, -0.07], F32)) + cool * np.array([-0.025, -0.005, 0.05], F32)
        src = np.clip(0.5 + (src - 0.5) * (1 + 0.1 * push), 0, 1).astype(F32)

    one = np.ones((H, W), F32)
    p.strokes(one, blur(src, 0.02 * H), flow=flow, size=0.08 * k, length=4, load=0.8, opacity=0.9, pickup=0.0,
              dry=0.35, thickness=0.35, jitter=0.06, name='lay-in')
    if a.geom:
        files = sorted(glob.glob(os.path.join(a.geom, '*.npz')))
        if not files:
            raise SystemExit(f"no geometry bands in {a.geom}")
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
    if a.geom:
        hi = hi * smoothstep(0.5, 0.9, blur(land, 0.004 * H))                 # not along the skyline
    p.strokes(hi, src, flow=flow, size=0.007 * k, length=2.5, pickup=0.1, load=1.3, thickness=2.4, jitter=0.02,
              ctol=0.12, overshoot=0.0, name='lights')
    p.save(a.out)
    if a.film:
        p.film(a.film)


if __name__ == '__main__':
    main()
