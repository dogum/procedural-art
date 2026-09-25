#!/usr/bin/env python3
"""Stroke-based painterly renderer: turns any image into a painting, procedurally.

    python paint.py --in photo.png --out painting.png --style oil --size 3000x1000 --seed 1

Styles: oil, impasto, gouache, watercolor, ink.  numpy + scipy + pillow only.
"""
import argparse
import os
import sys
import time

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

sys.dont_write_bytecode = True                   # the installed skill folder may be read-only
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paint_strokes import (F32, fblur, luminance, orientation_field, seed_grid, trace,  # noqa: E402
                           smooth_subdivide, Strokes, rasterize, decode, bilerp,
                           bristle_tile, smoothstep, tile_sample)
import paint_media as pm  # noqa: E402
import paint_wash as pw  # noqa: E402

T0 = time.time()


def log(*a):
    print('[%6.1fs]' % (time.time() - T0), *a, flush=True)


# ----------------------------------------------------------------------------
# presets for stroke styles.  Radii are in "units": 1 unit = 1 px at 3000x1000.
# ----------------------------------------------------------------------------
PRESETS = {
    'oil': dict(
        kind='strokes', radii=[30, 16, 8, 4, 2.2], blur=[0.9, 0.7, 0.5, 0.5, 0.4],
        grid=[1.5, 1.2, 1.0, 1.0, 1.0], T=[0, 0.07, 0.065, 0.06, 0.06],
        maxlen=[6, 8, 10, 10, 8], step=1.0, fc=[0.35, 0.5, 0.7, 0.75, 0.75],
        cthresh=[0.14, 0.12, 0.11, 0.1, 0.1],
        tensor=2.5, fallback=0.0, fallback_w=0.3, field_jitter=0.10, jitter_scale=180,
        wjit=0.25, angle_jit=0.15, len_jit=0.6, taper='oil', rag=0.04, cap='flat', capk=0.9,
        jit=dict(lum=0.025, hue=0.03, sat=0.05), wet=0.25, lift_end=0.8, end_blob=0.0, prof_pow=1.5, edge_blend=0.6,
        bristle_amp=0.05, detail_mix=0.12, dry=0.35,
        fine=dict(levels=[(2.2, 0.035), (1.1, 0.035)], grid=1.3, maxlen=4, cthresh=0.08,
                  hl_lo=0.07, hl_hi=0.18, hl_close=1.5, hl_radius=1.0),
        ground=(0.55, 0.36, 0.22), ground_mix=0.12, ground_dark=0.97,
        impasto=1.0, relief=0.45, ridge=0.0, groove=0.5, spec=0.08, shin=30, canvas=0.12,
        grade=dict(sat=1.05, contrast=1.05, gamma=1.0)),
    'impasto': dict(
        kind='strokes', radii=[20, 11, 6, 3.4], blur=[0.8, 0.6, 0.5, 0.5],
        grid=[1.2, 1.1, 1.0, 1.0], T=[0, 0.06, 0.06, 0.06],
        maxlen=[6, 6, 6, 5], step=1.0, fc=[0.6, 0.65, 0.7, 0.75],
        cthresh=[0.16, 0.14, 0.12, 0.12],
        tensor=2.0, fallback=0.0, fallback_w=0.25, field_jitter=0.28, jitter_scale=150,
        wjit=0.2, taper='oil', rag=0.05, cap='flat', capk=0.8,
        jit=dict(lum=0.06, hue=0.09, sat=0.15), wet=0.1,
        bristle_amp=0.14, detail_mix=0.05, dry=0.25,
        fine=dict(levels=[(1.7, 0.05)], grid=1.4, maxlen=4, cthresh=0.1,
                  hl_lo=0.07, hl_hi=0.18, hl_close=1.5, hl_radius=1.4),
        ground=(0.55, 0.36, 0.22), ground_mix=0.3, ground_dark=0.75,
        impasto=2.2, relief=0.55, ridge=0.12, spec=0.06, shin=25, canvas=0.08,
        grade=dict(sat=1.3, contrast=1.08, gamma=1.0)),
    'gouache': dict(
        kind='strokes', radii=[26, 13, 6.5, 3.2], blur=[1.0, 0.8, 0.6, 0.5],
        grid=[1.0, 1.0, 1.0, 1.0], T=[0, 0.07, 0.07, 0.07],
        maxlen=[8, 10, 10, 8], step=1.0, fc=[0.4, 0.55, 0.7, 0.75],
        cthresh=[0.16, 0.14, 0.12, 0.12],
        tensor=2.5, fallback=0.0, fallback_w=0.3, field_jitter=0.08, jitter_scale=180,
        wjit=0.12, taper='flat', rag=0.03, cap='flat', capk=0.5,
        jit=dict(lum=0.015, hue=0.015, sat=0.03), wet=0.0, palette=18, snap=0.8,
        bristle_amp=0.035, detail_mix=0.0, dry=0.0,
        fine=dict(levels=[(3.0, 0.045), (1.5, 0.045)], grid=1.4, maxlen=4, cthresh=0.1,
                  hl_lo=0.07, hl_hi=0.18, hl_close=1.5, hl_radius=1.3, dry=0.0),
        ground=(0.95, 0.93, 0.88), ground_mix=0.2, ground_dark=1.0,
        impasto=0.5, relief=0.25, ridge=0.35, spec=0.0, shin=10, canvas=0.0, paper=0.25,
        grade=dict(sat=1.1, contrast=1.02, gamma=0.95, lift=0.03)),
    'ink': dict(
        kind='ink', paper=(0.95, 0.925, 0.87), ink=(1.0, 1.04, 1.12),
        lift=0.3, gamma=1.2, dmax=1.45, local=3.0, stretch=0.3, empty=0.12, chroma_light=0.35, chroma_light_dark=0.45,
        taus=[0.06, 0.25, 0.5, 0.8], sigmas=[10, 6, 4, 3],
        soft=0.03, edge_px=1.0, rim=2.5, edge=1.0, wobble=5.0, flow=0.15, mottle=0.04,
        bleed=0.35, paper_relief=0.10, deckle=0.0,
        strokes=dict(dry=0.9, layers=[
            # (kind, radius, threshold, stop factor, half length, darkness[, spacing])
            ('texture', 3.0, 0.08, 0.5, 6, 0.9, 2.6),
            ('contour', 5.0, 0.08, 0.35, 12, 1.6, 4.0),
            ('contour', 2.5, 0.07, 0.4, 12, 1.4, 3.5),
            ('contour', 1.4, 0.06, 0.5, 8, 1.2, 3.0),
        ]),
        mode='auto',
        wash_lines=[(2.0, 1.1, 0.045, 0.1, 80, 16, 12, 0.7)],
        lines=dict(paper_pct=(40, 97), local=0.5, tone_gamma=1.0, dmax=1.0, taus=[0.25, 0.65, 0.9], sigmas=[8, 5, 3], flow=0.25,
                   dry=0.8, bleed=0.3, max_pressure=1.5, hl_lo=0.07, hl_hi=0.18, hl_close=2.0,
                   layers=[
                       # (edge sigma, brush radius, low, high threshold, min chain length,
                       #  seed spacing, half length in steps, darkness)
                       (4.0, 2.4, 0.045, 0.14, 60, 28, 14, 2.2),
                       (2.0, 1.3, 0.07, 0.16, 45, 14, 10, 1.8),
                   ])),
    'watercolor': dict(
        kind='watercolor', paper=(0.975, 0.965, 0.935), density=0.85, dmax=1.3, key_target=0.4, key_min=0.7, sat=1.08, gran=0.12,
        paper_relief=0.12, deckle=0.022, deckle_rough=0.5,
        taus=[0.05, 0.22, 0.45, 0.75, 1.15, 1.7],
        sigmas=[16, 11, 7, 5, 3.5, 2.5],
        soft=0.005, edge_px=0.7, rim=2.0, edge=1.3, wobble=4.0, flow=0.2, mottle=0.03, hue_var=0.05,
        detail_sigma=1.2, detail_thr=0.045, detail_frac=0.5,
        marks=dict(cthresh=0.15, dry=0.8, rim=0.5, base=0.12, layers=[
            # (radius, residual threshold, half length, fraction of missing pigment)
            (7, 0.03, 5, 0.7),
            (3.5, 0.035, 5, 0.8),
            (1.8, 0.04, 4, 0.8),
            (1.0, 0.045, 3, 0.8),
        ]),
        reserve=0.85, hl_lo=0.07, hl_hi=0.18, hl_close=1.5, lift=0.7, lift_thr=0.06),
}


# ----------------------------------------------------------------------------
# stroke painter (Hertzmann-style coarse-to-fine)
# ----------------------------------------------------------------------------
def paint_strokes(src, P, rng, unit, detail=1.0, sscale=1.0, palette=None):
    H, W = src.shape[:2]
    # toned ground: a thin warm underpainting of the big shapes
    under = fblur(src, 40 * unit) * P.get('ground_dark', 0.8)
    canvas = (under * (1 - P['ground_mix']) + np.array(P['ground'], F32) * P['ground_mix']).astype(F32)
    plain = canvas.copy()          # flat stroke colours only: drives the error
    Hmap = np.zeros((H, W), F32)
    btile = bristle_tile(rng)
    nL = len(P['radii'])
    fine_ref = fblur(src, 0.8 * unit)
    nL = min(nL, int(os.environ.get('PAINT_LAYERS', nL)))
    for li in range(nL):
        R = max(P['radii'][li] * unit * sscale, 0.9)
        ref = fblur(src, P['blur'][li] * R)
        err = fblur(np.linalg.norm(plain - ref, axis=2), 0.5 * R)
        seeds = seed_grid(err, P['grid'][li] * R, P['T'][li] / detail, rng, full=(li == 0))
        if len(seeds) == 0:
            continue
        lp = dict(fc=P['fc'][li] if np.ndim(P['fc']) else P['fc'], cthresh=P['cthresh'][li],
                  maxlen=P['maxlen'][li], tensor=P['tensor'])
        n, cov = stroke_layer(canvas, plain, Hmap, seeds, ref, R, P, lp, rng, unit, fine_ref, btile, li, palette)
        log('layer %d  R=%.1f  strokes=%d  coverage=%.0f%%' % (li, R, n, 100 * cov))
    if P.get('fine') and nL == len(P['radii']):
        detail_pass(src, canvas, plain, Hmap, P, rng, unit, detail, sscale, btile, palette)
    return canvas, Hmap


def stroke_layer(canvas, plain, Hmap, seeds, ref, R, P, lp, rng, unit, fine_ref, btile, li, palette=None,
                 col_override=None):
    """Trace, rasterise and shade one layer of strokes seeded at `seeds`."""
    H, W = Hmap.shape
    wx, wy = orientation_field(luminance(ref), max(0.5 * R, lp.get('grad_min', 1.0)), lp['tensor'] * R,
                               P['fallback'], P['fallback_w'], lp.get('field_jitter', P['field_jitter']), rng,
                               unit * P.get('jitter_scale', 40) / 40.0)
    col = bilerp(ref, seeds[:, 0], seeds[:, 1]) if col_override is None else col_override
    n_half = max(1, lp['maxlen'] // 2)
    M = len(seeds)
    rot = rng.standard_normal(M).astype(F32) * P.get('angle_jit', 0.0) if P.get('angle_jit') else None
    maxk = None
    if P.get('len_jit'):
        maxk = np.ceil(n_half * (1 - P['len_jit'] * rng.random(M))).astype(np.int32)
    Pts, lo, hi = trace(seeds, col, ref, wx, wy, lp.get('step', P['step']) * R, n_half, lp['fc'],
                        lp['cthresh'], rng=rng, rot=rot, maxk=maxk)
    # stroke colour: blend of seed colour and mean along the path
    K = Pts.shape[1]
    kk = np.arange(K)
    vm = (kk[None] >= lo[:, None]) & (kk[None] <= hi[:, None])
    pc = bilerp(ref, Pts[..., 0].ravel(), Pts[..., 1].ravel()).reshape(M, K, 3)
    pmean = (pc * vm[..., None]).sum(1) / vm.sum(1)[:, None]
    ms = lp.get('seed_mix', 0.5)
    col = ms * col + (1 - ms) * pmean
    if palette is not None:
        sn = P.get('snap', 1.0) * lp.get('snap_mul', 1.0)
        col = col * (1 - sn) + pm.nearest_palette(col, palette) * sn
    col0 = col
    col = pm.jitter_colours(col, rng, **{k: v * lp.get('jit_mul', 1.0) for k, v in P['jit'].items()})
    if R >= 5:
        P2, lo2, hi2 = smooth_subdivide(Pts, lo, hi)
    else:
        P2, lo2, hi2 = Pts, lo, hi
    wj = lp.get('wjit', P['wjit'])
    width = R * np.clip(1 + wj * rng.standard_normal(M), 0.6, 1.5).astype(F32)
    if lp.get('width_mul') is not None:
        width = width * lp['width_mul']
    key = rng.permutation(M).astype(np.int64)
    S = Strokes(P2, lo2, hi2, width, key, taper=lp.get('taper', P['taper']), rag=lp.get('rag', P['rag']),
                cap=lp.get('cap', P.get('cap', 'flat')), capk=lp.get('capk', P.get('capk', 0.45)))
    PP = dict(P, **lp.get('shade', {}))
    hsum = np.zeros(H * W, F32) if (PP['impasto'] > 0 or PP.get('edge_blend', 0) > 0) else None
    csum = wsum = None
    if PP.get('edge_blend', 0) > 0:
        S.col = col
        csum = np.zeros((H * W, 3), np.float64)
        wsum = np.zeros(H * W, np.float64)
    zbuf, Ls = rasterize(S, H, W, hsum=hsum, csum=csum, wsum=wsum)
    if hsum is not None:
        hsum = ndi.gaussian_filter(hsum.reshape(H, W), 0.9 * max(unit, 0.6))
    avg = None
    if csum is not None:
        avg = (csum / np.maximum(wsum, 1e-9)[:, None]).astype(F32)
        avg_w = np.clip(wsum / 0.5, 0, 1).astype(F32)       # trust of the average
    inv = np.argsort(key)
    has, sid, un, vn = decode(zbuf, inv)
    shade_layer(canvas, Hmap, has, sid, un, vn, col, width, Ls, fine_ref, btile, PP, li, R, rng, unit, hsum,
                (avg, avg_w) if avg is not None else None)
    plain.reshape(-1, 3)[has] = col0[sid]
    return M, has.mean()


def detail_pass(src, canvas, plain, Hmap, P, rng, unit, detail, sscale, btile, palette=None):
    """Last, small brushes: edges and thin structures the big layers missed,
    then the highlights, placed as crisp dabs the way a painter adds them last.

    Seeds are where the painting so far differs from a lightly blurred source
    AND the source has an edge (or a small bright spot), so flat passages keep
    their brushwork.  Colours come from the source; no pixels are copied.
    """
    F = P['fine']
    ref = fblur(src, F.get('ref_blur', 0.7) * unit)
    # coherence: lines and contours have one direction; render noise and
    # speckle do not, and are left to the bigger strokes
    Jxx = Jyy = Jxy = 0
    for c in range(3):
        gy, gx = np.gradient(fblur(ref[..., c], 0.6 * unit))
        Jxx, Jyy, Jxy = Jxx + gx * gx, Jyy + gy * gy, Jxy + gx * gy
    s2 = 2.0 * unit
    a, b, tr = fblur(Jxx - Jyy, s2), fblur(2 * Jxy, s2), fblur(Jxx + Jyy, s2)
    coh = smoothstep(0.25, 0.6, np.sqrt(a * a + b * b) / (tr + 1e-6))
    del Jxx, Jyy, Jxy, a, b, tr
    for k, (r0, T) in enumerate(F['levels']):
        R = max(r0 * unit * sscale, 0.8)
        # missing structure at this brush's scale: compare band-passed
        # painting and source, so a slightly different flat colour (brush
        # jitter) does not count, but a lost line, rim or small shape does
        bs = 3.0 * R
        cb = fblur(canvas, 0.5 * R)                  # ignore the bristle texture
        bref = ref - fblur(ref, bs)
        err = fblur(np.linalg.norm((cb - fblur(cb, bs)) - bref, axis=2), 0.5 * R)
        struct = smoothstep(F.get('struct_lo', 0.02), F.get('struct_hi', 0.06),
                            fblur(np.linalg.norm(bref, axis=2), 0.5 * R))
        score = err * coh * struct
        seeds = seed_grid(score, F['grid'] * R, T / detail, rng, jitter=0.4)
        if len(seeds) == 0:
            continue
        lref = fblur(src, 0.5 * R) if R > 1.5 * unit else ref
        lp = dict(fc=0.8, cthresh=F['cthresh'], maxlen=F['maxlen'], tensor=1.5, grad_min=0.7 * unit,
                  field_jitter=0.0, seed_mix=0.7, jit_mul=0.5, wjit=0.15, snap_mul=0.5, rag=0.02,
                  shade=dict(wet=0.0, detail_mix=0.0, edge_blend=0.0, dry=F.get('dry', 0.15),
                             bristle_amp=P['bristle_amp'] * 0.6, impasto=P['impasto'] * F.get('impasto', 0.6)))
        n, cov = stroke_layer(canvas, plain, Hmap, seeds, lref, R, P, lp, rng, unit, lref, btile, 90 + k, palette)
        log('detail %d R=%.1f  strokes=%d  coverage=%.0f%%' % (k, R, n, 100 * cov))
    # highlights: spots and rims clearly brighter than their surroundings,
    # a closed hard-edged shape (a speckled render highlight becomes one
    # shape).  Dab size grows toward the inside of large highlights.
    Rh = max(F['hl_radius'] * unit * sscale, 0.8)
    hm = pw.highlight_mask(src, unit, F)
    if hm.max() < 0.5:
        return
    inside = hm > 0.5
    dist = ndi.distance_transform_edt(inside).astype(F32)
    seeds = seed_grid(hm, 1.1 * Rh, 0.5, rng, jitter=0.6)
    wm = np.clip(0.7 * bilerp(dist, seeds[:, 0], seeds[:, 1]) / Rh, 1.0, F.get('hl_grow', 4.0))
    keep = rng.random(len(seeds)) < 1.0 / wm ** 2      # fewer, bigger dabs inside
    seeds, wm = seeds[keep], wm[keep].astype(F32)
    if len(seeds):
        sm = fblur(src, F.get('hl_blur', 1.5) * unit)
        Ls = luminance(sm)
        hi_ref = ndi.maximum_filter(Ls, size=max(3, int(round(2 * unit)) | 1))
        pick = sm * (hi_ref / np.maximum(Ls, 1e-3))[..., None]
        col = np.clip(bilerp(pick, seeds[:, 0], seeds[:, 1]) * F.get('hl_boost', 1.03), 0, 1)
        lp = dict(fc=0.6, cthresh=0.25, maxlen=2, tensor=1.0, grad_min=0.7 * unit,
                  field_jitter=0.0, seed_mix=1.0, jit_mul=0.3, wjit=0.2, snap_mul=0.0, rag=0.05,
                  taper='dab', cap='round', width_mul=wm,
                  shade=dict(wet=0.0, detail_mix=0.0, edge_blend=0.0, dry=0.0,
                             bristle_amp=P['bristle_amp'] * 0.5, impasto=P['impasto'] * F.get('hl_impasto', 1.0)))
        n, cov = stroke_layer(canvas, plain, Hmap, seeds, sm, Rh, P, lp, rng, unit, sm, btile, 91, None,
                              col_override=col)
        log('highlights R=%.1f  dabs=%d' % (Rh, n))


def shade_layer(canvas, Hmap, has, sid, un, vn, col, width, Ls, fine_ref, btile, P, li, R, rng, unit, hsum=None,
                avg=None):
    H, W = Hmap.shape
    cf = canvas.reshape(-1, 3)
    hf = Hmap.reshape(-1)
    idx = np.nonzero(has)[0]
    M = len(col)
    w = width[sid]
    ul = un * Ls[sid]
    # bristle streaks: tile sampled along (u) and across (v) the stroke
    offu = rng.integers(0, 256, M)
    offv = rng.integers(0, 256, M)
    bsp = 0.9 + 0.10 * R                     # bristle spacing in px
    b = tile_sample(btile, ul / 2.0 + offu[sid], (vn + 1) * w / bsp + offv[sid])
    c = col[sid]
    if avg is not None:
        # wet-into-wet: near its edges a stroke merges into the paint around it
        av, aw = avg
        e = 1 - P['edge_blend'] * smoothstep(0.3, 1.0, np.abs(vn)) * aw[idx]
        c = av[idx] * (1 - e[:, None]) + c * e[:, None]
    c = c * (1 + P['bristle_amp'] * b[:, None])
    if li > 0 and P.get('wet', 0) > 0:
        # wet-in-wet: the stroke picks up some of the paint already there
        c = c * (1 - P['wet']) + cf[idx] * P['wet']
    if P['detail_mix'] > 0:
        c = c * (1 - P['detail_mix']) + fine_ref.reshape(-1, 3)[idx] * P['detail_mix']
    # alpha: antialiased edge x dry-brush break-up towards the stroke end
    a = np.ones_like(un)
    load = 1 - P['dry'] * smoothstep(0.55, 1.0, un)
    if P['dry'] > 0:
        a *= smoothstep(-0.2, 0.2, b + 2.8 - 5.5 * (1 - load))
    cf[idx] = cf[idx] * (1 - a[:, None]) + np.clip(c, 0, 1) * a[:, None]
    if P['impasto'] > 0:
        # paint accumulates: the body is the additive sum of all strokes of the
        # layer (continuous, order-independent); the top stroke adds bristle
        # grooves and an optional ridge along its edges
        thick = P['impasto'] * unit * (0.4 + 0.6 * np.sqrt(R / unit / 8))
        body = np.minimum(hsum, P.get('body_cap', 2.5))
        cov = np.minimum(hsum, 1.0)
        Hmap *= (1 - P.get('cover', 0.45) * cov)
        Hmap += thick * body
        rd = P.get('ridge', 0.0)
        v2 = np.clip(1 - vn * vn, 0, 1)
        top = P.get('groove', 0.25) * b * v2 * load
        if rd > 0:
            top = top + rd * np.exp(-((np.abs(vn) - 0.8) / 0.12) ** 2)
        if P.get('end_blob', 0) > 0:
            top = top + P['end_blob'] * smoothstep(0.85, 1.0, un) * v2
        hf[idx] += thick * top * a


def finish_strokes(canvas, Hmap, P, unit, rng):
    H, W = Hmap.shape
    out = canvas
    if P.get('canvas', 0) > 0:
        weave = pm.canvas_weave(H, W, unit, rng)
        thin = np.exp(-np.maximum(Hmap, 0) / (0.8 * unit))
        Hmap = Hmap + weave * unit * P['canvas'] * (0.35 + 0.65 * thin)
    if P.get('paper', 0) > 0:
        Hmap = Hmap + pm.fine_grain(H, W, unit, rng) * unit * P['paper']
    diff, sp = pm.light_height(Hmap, P['relief'], spec=P['spec'], shin=P['shin'], blur=0.5 * unit)
    diff = np.clip(diff, 0.55, 1.35)
    out = out * (1 + 0.55 * (diff[..., None] - 1))
    if sp is not None:
        out = out + sp[..., None] * np.array([1.0, 0.97, 0.9], F32)
    g = P.get('grade')
    if g:
        out = pm.saturate(np.clip(out, 0, 1), g['sat'])
        out = pm.curve(out, g['contrast'], g.get('lift', 0.0), g['gamma'])
    return np.clip(out, 0, 1)


# ----------------------------------------------------------------------------
def load_input(path, size):
    im = Image.open(path).convert('RGB')
    if size:
        W, H = size
        iw, ih = im.size
        s = max(W / iw, H / ih)
        nw, nh = int(round(iw * s)), int(round(ih * s))
        im = im.resize((nw, nh), Image.LANCZOS)
        l, t = (nw - W) // 2, (nh - H) // 2
        im = im.crop((l, t, l + W, t + H))
    return np.asarray(im, F32) / 255.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--in', dest='inp', required=True, help='input image (png/jpg)')
    ap.add_argument('--out', required=True, help='output image (.png or .jpg)')
    ap.add_argument('--style', default='oil', choices=sorted(PRESETS), help='painting medium')
    ap.add_argument('--size', default=None, help='WxH output size (default: input size); input is cover-cropped')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--detail', type=float, default=1.0, help='>1 more small strokes / sharper, <1 looser')
    ap.add_argument('--strokes-scale', type=float, default=1.0, help='multiply all brush sizes')
    ap.add_argument('--relief', type=float, default=None, help='override impasto relief strength (0 = flat)')
    ap.add_argument('--ink-mode', default=None, choices=['auto', 'lines', 'wash'],
                    help='ink only: lines = contour strokes + sparse washes (objects), '
                         'wash = sumi-e landscape; auto (default) picks wash when the top is a light sky')
    ap.add_argument('--jpg', action='store_true', help='also write a .jpg next to a .png output')
    a = ap.parse_args()
    size = tuple(int(v) for v in a.size.lower().split('x')) if a.size else None
    src = load_input(a.inp, size)
    H, W = src.shape[:2]
    unit = np.sqrt(W * H) / 1732.0
    rng = np.random.default_rng(a.seed)
    P = dict(PRESETS[a.style])
    if a.relief is not None:
        P['relief'] = a.relief
    if P['kind'] == 'ink':
        P['mode'] = a.ink_mode or P.get('mode', 'auto')
        if P['mode'] == 'auto':
            P['mode'] = pw.ink_mode_auto(src)
    log('%s%s  %dx%d  unit=%.2f' % (a.style, ' (%s)' % P['mode'] if P['kind'] == 'ink' else '', W, H, unit))
    if P['kind'] in ('ink', 'watercolor') and a.strokes_scale != 1.0:
        # brush sizes of the wash media: glaze blur scales and mark radii
        s = a.strokes_scale
        P['sigmas'] = [v * s for v in P['sigmas']]
        for k, ri in (('marks', 0), ('strokes', 1), ('lines', 1)):
            if k in P:
                P[k] = dict(P[k])
                P[k]['layers'] = [tuple(v * s if i == ri else v for i, v in enumerate(l))
                                  for l in P[k]['layers']]
        if P.get('wash_lines'):
            P['wash_lines'] = [tuple(v * s if i == 1 else v for i, v in enumerate(l)) for l in P['wash_lines']]
    if P['kind'] == 'strokes':
        pal = None
        if P.get('palette'):
            pal = pm.kmeans_palette(fblur(src, 3 * unit), P['palette'], rng)
        canvas, Hmap = paint_strokes(src, P, rng, unit, a.detail, a.strokes_scale, pal)
        out = finish_strokes(canvas, Hmap, P, unit, rng)
    elif P['kind'] == 'ink':
        out = pw.ink_wash(src, rng, unit, a.detail, P)
    elif P['kind'] == 'watercolor':
        out = pw.watercolor(src, rng, unit, a.detail, P)
    else:
        raise SystemExit('unknown kind')
    img = Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    img.save(a.out, quality=92)
    if a.jpg and a.out.lower().endswith('.png'):
        img.save(a.out[:-4] + '.jpg', quality=90)
    try:
        import resource
        mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        log('wrote %s  (peak memory %.0f MB)' % (a.out, mb))
    except ImportError:
        log('wrote', a.out)


if __name__ == '__main__':
    main()
