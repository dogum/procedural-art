"""A watercolour of an existing picture: a few transparent washes, light to dark, on white paper.

    python aquarelle_repaint.py --in photo.png --out watercolour.png [--height 1200] [--seed 1]
                                [--aov BANDS/aov] [--planes 4] [--key 1.0] [--border 0.035] [--accents 1.0]
                                [--film process.mp4]

This is not repaint.py with thinner paint. A watercolourist works the other way round:

  1. The lights are reserved. Snow, the tops of clouds and sparkle on water are paper left bare;
     nothing is ever painted over them.
  2. The sky goes on wet paper, in one go, so it has no edges.
  3. The land is a handful of flat washes, each over everything from one skyline down. A wash
     keeps the hard edge of its ridge at the top and is covered by the next one below, so every
     nearer plane is the sum of the washes behind it: distance is the number of layers.
     With --aov (the painter buffers of a pbr_terrain.py --aov render) the planes are the real
     ranges, found from depth, and the far ones granulate. Without it the planes are steps of
     tone found in the picture.
  4. A last, smaller wash puts the darkest notes in, and a dry brush drags a few accents down
     the slopes (down the real fall lines with --aov).

Every wash is given exactly the pigment that is still missing at that point (the absorption the
picture has there, less what the washes before it left), so the layers add up to the picture's
colour without a single opaque touch.
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aquarelle import Paper, _Mix, _ks                                   # noqa: E402
from atelier import F32, NB, blur, image_flow, load_image, smoothstep, _resize   # noqa: E402

LUM = np.array([0.299, 0.587, 0.114], F32)


def keyed(src, key=1.0):
    """The picture as a watercolourist would key it: lighter, a little cleaner in colour."""
    if key <= 0:
        return src
    L = (src @ LUM)[..., None]
    warm = smoothstep(0.45, 0.85, L) * key                                   # warmer lights, cooler shadows
    cool = smoothstep(0.5, 0.15, L) * key
    src = np.clip(src * (1 + warm * np.array([0.05, 0.015, -0.06], F32)) + cool * np.array([-0.02, 0.0, 0.04], F32), 0, 1)
    c = np.clip(L + (src - L) * (1 + 0.22 * key), 0, 1)
    return np.clip(c ** (1 - 0.14 * key), 0, 1).astype(F32)


def masked_blur(f, m, r):
    """Blur `f` (h x w x c) inside the mask only: colours do not leak across its edge."""
    den = np.maximum(blur(m, r), 1e-4)
    return np.stack([blur(f[..., i] * m, r) / den for i in range(f.shape[-1])], -1)


def valleys(v, n, sep=0.07):
    """Up to n thresholds for the values v (0..1), placed where few pixels are: between ranges."""
    h, edges = np.histogram(v, bins=96, range=(0, 1))
    h = ndi.gaussian_filter1d(h.astype(F32), 2.0)
    c = np.cumsum(h) / max(h.sum(), 1)
    cand = [i for i in range(3, 93) if h[i] <= h[i - 1] and h[i] <= h[i + 1] and 0.06 < c[i] < 0.94]
    cand.sort(key=lambda i: h[i])
    pick = []
    for i in cand:
        t = edges[i] + 0.5 / 96
        if all(abs(t - q) > sep for q in pick):
            pick.append(t)
        if len(pick) == n:
            break
    for q in np.quantile(v, np.linspace(0, 1, n + 2)[1:-1]):                # not enough gaps: fill in by area
        if len(pick) < n and all(abs(q - t) > sep for t in pick):
            pick.append(float(q))
    return sorted(pick)


def find_sky(src):
    """Without buffers: the smooth region that hangs from the top edge of the picture."""
    H = src.shape[0]
    L = blur(src @ LUM, 0.004 * H)
    gy, gx = np.gradient(L)
    busy = blur(np.hypot(gx, gy), 0.006 * H)
    calm = busy < max(1.6 * np.percentile(busy, 40), 1e-4)
    lab, _ = ndi.label(calm)
    top = np.unique(lab[0][lab[0] > 0])
    if len(top) == 0:
        return np.zeros(L.shape, F32)
    sky = np.isin(lab, top)
    sky = ndi.binary_fill_holes(ndi.binary_closing(sky, iterations=max(1, H // 300)))
    # up to the skyline, which is never calm: take in what lies next to the sky and is still its colour
    near = masked_blur(src, sky.astype(F32), 0.02 * H)
    same = np.abs(src - near).max(-1) < 0.07
    sky = sky | (ndi.binary_dilation(sky, iterations=max(2, int(0.03 * H))) & same)
    sky = ndi.binary_dilation(ndi.binary_opening(sky, iterations=1), iterations=max(1, int(0.002 * H)))
    if sky.mean() > 0.85 or sky.mean() < 0.03:                              # everything, or nothing worth the name
        return np.zeros(L.shape, F32)
    return sky.astype(F32)


def paint_watercolour(p, src, aov=None, planes=4, key=1.0, border=0.035, accents=1.0):
    """Lay the picture `src` (H x W x 3, 0..1) on the Paper p. aov: folder of painter buffers."""
    H, W, hs, ws = p.H, p.W, p.hs, p.ws
    src = keyed(src, key)
    small = lambda f, how=Image.BOX: _resize(np.asarray(f, F32), ws, hs, how)         # noqa: E731
    img_s = np.clip(np.stack([small(src[..., i]) for i in range(3)], -1), 0, 1)
    L = src @ LUM

    # ---- what is where
    A = None
    if aov:
        from repaint import load_aov
        A = load_aov(aov, W, H)
        if float((A['hit'] * (1 - np.clip(A['water'], 0, 1)) > 0.5).mean()) < 0.002:
            print("  the painter buffers show no land in this view: painting from the picture alone", flush=True)
            A = None
    if A is not None:
        hit = smoothstep(0.3, 0.7, A['hit'])
        sky = 1 - hit
        water = smoothstep(0.3, 0.7, A['water']) * hit
        land = hit * (1 - water)
        depth = 1 - A['easel']                                               # 0 at the painter's feet, 1 far
        th = valleys(depth[land > 0.5], planes - 1)
        level = np.zeros((H, W), np.int32)                                   # 0 = the farthest range
        for t in th:
            level += (depth < t)
        level = ndi.median_filter(level, size=max(3, int(0.006 * H) | 1))
        gran = np.linspace(0.8, 0.3, len(th) + 1)
    else:
        sky = find_sky(src)
        water = np.zeros((H, W), F32)
        land = 1 - sky
        dark = 1 - blur(L, 0.006 * H)
        v = dark[land > 0.5]
        th = [float(q) for q in np.quantile(v, np.linspace(0, 1, planes + 1)[1:-1])] if v.size else []
        level = np.zeros((H, W), np.int32)                                   # 0 = the lightest tone
        for t in th:
            level += (dark > t)
        level = ndi.median_filter(level, size=max(3, int(0.008 * H) | 1))
        gran = np.full(len(th) + 1, 0.45)
    n_lev = len(th) + 1

    # ---- the lights: paper
    spark = smoothstep(0.03, 0.09, L - blur(L, 0.02 * H)) * smoothstep(0.62, 0.8, L)
    lights = np.maximum(smoothstep(0.84, 0.93, blur(L, 0.0015 * H)), spark * (water if A is not None else land))
    if A is not None:
        lights = np.maximum(lights, smoothstep(0.4, 0.7, A['snow']) * smoothstep(0.62, 0.8, L) * land)
        lights = np.maximum(lights * (1 - sky), smoothstep(0.25, 0.6, A['cloud']) * sky * smoothstep(0.7, 0.85, L))
    keep = 1 - np.clip(lights, 0, 1)                                         # 1 where paint may go
    # a sketch stops short of the edge of the sheet: a loose margin of bare paper
    if border > 0:
        xx, yy = p.x, p.y
        asp = W / H
        inset = np.minimum(np.minimum(xx, asp - xx), np.minimum(yy, 1 - yy)) - border * (1 + 0.3 * p.noise(0.3, 2))
        sheet = (inset > 0).astype(F32)
    else:
        sheet = np.ones((H, W), F32)

    # ---- what the paper has to end up absorbing, on the water grid
    T = np.clip(_ks(img_s) - p._ksp, 0, None) * (0.85 if key > 0 else 1.0)
    lev_s = small(level.astype(F32), Image.NEAREST).round().astype(np.int32)
    sky_s, land_s, water_s = small(sky), small(land), small(water)
    Ts = np.zeros_like(T)
    r = 0.014 * hs
    for i in range(n_lev):                                                   # each plane simplified on its own
        m = ((lev_s == i) & (sky_s < 0.5)).astype(F32)
        if m.sum() > 0:
            Ts += masked_blur(T, m, r) * m[..., None]
    if sky_s.max() > 0:
        m = (sky_s >= 0.5).astype(F32)
        Ts += masked_blur(T, m, 0.03 * hs) * m[..., None]
    D = Ts.mean(-1)
    put = np.zeros_like(T)                                                   # what the washes so far have left

    def wash(mask, target, g, name, **kw):
        nonlocal put
        delta = np.clip(target - put, 0, None)
        if float((delta.mean(-1) * small(mask)).max()) < 1e-4:
            return
        p.wash(mask, _Mix(delta, g, 0.3), strength=1.0, name=name, **kw)
        put = np.maximum(put, target * (small(mask) > 0.5)[..., None])

    # ---- 1. the sky, wet in wet
    if sky.max() > 0:
        p.wet(sky * sheet, amount=0.9, feather=0.004)
        wash(sky * sheet * (0.15 + 0.85 * keep), Ts * 0.68, 0.55, 'sky', wetness=0.9, vary=0.3, settle=10, hand=0.8)
        p.dry()

    # ---- 2. the land: one wash from each skyline down
    ground = np.clip(land + water, 0, 1)
    for i in range(n_lev):
        m = (level >= i).astype(F32) * (ground > 0.5) * (keep > 0.5) * sheet
        if m.sum() < 16:
            continue
        here = lev_s >= i
        Li = float(np.median(D[(lev_s == i) & (land_s + water_s > 0.5)])) if ((lev_s == i) & (land_s + water_s > 0.5)).any() else 0.0
        if A is not None and i < n_lev - 1:
            cap = np.minimum(1.0, Li / np.maximum(D, 1e-5))                  # nearer planes get this range's tone for now
            target = Ts * np.where(lev_s == i, 1.0, cap)[..., None] * here[..., None]
        else:
            target = Ts * here[..., None] if A is not None else Ts * np.minimum(1.0, (Li if i < n_lev - 1 else 1e9) / np.maximum(D, 1e-5))[..., None] * here[..., None]
            if A is None:
                target = np.where((lev_s == i)[..., None], Ts * here[..., None], target)
        wash(m, target, float(gran[i]), f'plane {i + 1}/{n_lev}', wetness=0.7, vary=0.22, settle=5,
             hand=0.9 if i < n_lev - 1 else 0.6)
        p.dry()

    # ---- 3. the darks: a smaller wash where the picture is darker than its plane
    for name, rad, over, g in (('shadow shapes', 0.007, 1.10, 0.5), ('darks', 0.003, 1.22, 0.3)):
        Tf = np.stack([blur(T[..., b], rad * hs) for b in range(NB)], -1)
        base = np.maximum(put.mean(-1), 0.02)
        more = (Tf.mean(-1) - over * base) / base
        dm = _resize(smoothstep(-0.05, 0.05, more), W, H, Image.BICUBIC) * ground * keep * sheet
        dm = (blur(dm, 0.0025 * H) > 0.5).astype(F32)
        if dm.mean() > 0.002:
            wash(dm, np.where((small(dm) > 0.5)[..., None], np.maximum(Tf, put), put), g, name, wetness=0.5, vary=0.15, settle=3, hand=0.4)
            p.dry()

    # ---- 4. dry brush: a few accents down the slopes, and across the water
    rng = np.random.default_rng([p.seed, 5])
    Ld = blur(L, 0.004 * H)
    if A is not None:
        fx, fy = blur(A['fx'], 0.01 * H), blur(A['fy'], 0.01 * H)
        want = land * keep * sheet * smoothstep(0.25, 0.6, A['slope'] / (np.percentile(A['slope'][land > 0.5], 90) + 1e-6)) * (0.3 + 0.7 * A['easel'])
    else:
        ang = image_flow(src, 0.012 * H)
        fx, fy = np.cos(ang), np.sin(ang)
        gy_, gx_ = np.gradient(Ld)
        tex = blur(np.hypot(gx_, gy_), 0.01 * H)
        want = land * keep * sheet * smoothstep(0.5, 1.5, tex / (np.percentile(tex, 75) + 1e-6)) * smoothstep(0.75, 0.4, Ld)
    want = want * (blur(sheet, 0.04 * H) > 0.97)                             # accents stay inside the sketch
    n_acc = int(30 * accents * W / H)
    pr = want.ravel().astype(np.float64)
    if pr.sum() > 0:
        idx = rng.choice(pr.size, size=min(n_acc, int((pr > 0.05).sum())), replace=False, p=pr / pr.sum())
        lines, cols, sizes = [], [], []
        for j in idx:
            y0, x0 = divmod(int(j), W)
            px_, py_ = float(x0), float(y0)
            pts = [(px_ / H, py_ / H)]
            ln = rng.uniform(0.03, 0.09) * H
            for _ in range(6):
                ix, iy = int(np.clip(px_, 0, W - 1)), int(np.clip(py_, 0, H - 1))
                d = np.array([fx[iy, ix], fy[iy, ix]])
                d = d / (np.hypot(*d) + 1e-6)
                px_, py_ = px_ + d[0] * ln / 6, py_ + d[1] * ln / 6
                pts.append((px_ / H, py_ / H))
            lines.append(pts)
            cols.append(np.clip(src[y0, x0] * 0.7, 0, 1))
            sizes.append(rng.uniform(0.006, 0.014))
        # grouped by tone, so each group is one loading of the brush
        order = np.argsort([c @ LUM for c in cols])[::-1]
        for part in np.array_split(order, 4):
            if len(part):
                c = np.mean([cols[k] for k in part], 0)
                p.stroke([lines[k] for k in part], c, size=[sizes[k] for k in part], strength=0.3, dry=0.7, name='dry brush')
    if water.max() > 0:
        ys, xs = np.where((water > 0.5) & (keep > 0.5) & (sheet > 0.5))
        if len(ys):
            sel = rng.choice(len(ys), size=min(60, len(ys)), replace=False)
            lines = [[(xs[k] / H - l_, ys[k] / H), (xs[k] / H + l_, ys[k] / H)] for k, l_ in zip(sel, rng.uniform(0.02, 0.09, len(sel)))]
            c = np.clip(np.mean(src[water > 0.5], 0) * 0.6, 0, 1)
            p.stroke(lines, c, size=0.005, strength=0.5, dry=0.7, name='water')
    p.dry()
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--height', type=int, default=None)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--aov', default=None, help='painter buffers of the same view (pbr_terrain.py --aov)')
    ap.add_argument('--planes', type=int, default=4, help='washes the land is built from (3 to 6)')
    ap.add_argument('--key', type=float, default=1.0, help='how much lighter and cleaner than the picture (0 = as it is)')
    ap.add_argument('--grain', type=float, default=1.0, help='tooth of the paper (0 smooth, 1 cold-press, 1.6 rough)')
    ap.add_argument('--border', type=float, default=0.035, help='margin of bare paper round the sketch, in picture heights (0 = paint to the edge)')
    ap.add_argument('--accents', type=float, default=1.0, help='how many dry-brush accents (0 = none)')
    ap.add_argument('--film', default=None)
    a = ap.parse_args()
    im = Image.open(a.inp)
    H = a.height or im.height
    W = int(round(H * im.width / im.height))
    src = load_image(a.inp, W, H)
    p = Paper(W, H, seed=a.seed, grain=a.grain, tilt=0.05, film=bool(a.film))
    paint_watercolour(p, src, a.aov, int(np.clip(a.planes, 2, 6)), a.key, a.border, a.accents)
    p.save(a.out)
    if a.film:
        p.film(a.film)


if __name__ == '__main__':
    main()
