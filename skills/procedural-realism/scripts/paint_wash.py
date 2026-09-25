"""Transparent-media renderers for paint.py: watercolour and ink wash.

Both work in optical density (Beer-Lambert: colour = paper * exp(-density)),
so overlapping glazes darken the way real transparent pigment does.
"""
import numpy as np
from scipy import ndimage as ndi

from paint_strokes import (F32, fblur, smooth_noise, luminance, smoothstep, bilerp,
                           orientation_field, seed_grid, trace, smooth_subdivide,
                           Strokes, rasterize, decode, bristle_tile)
import paint_media as pm


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def deckle_mask(H, W, unit, rng, margin=0.035, rough=1.0):
    """Paint stops short of the paper edge along an irregular line."""
    y, x = np.mgrid[0:H, 0:W].astype(F32)
    m = margin * np.sqrt(W * H)
    d = np.minimum(np.minimum(x, W - 1 - x), np.minimum(y, H - 1 - y))
    n = (0.6 * smooth_noise(H, W, 120 * unit, rng) + 0.4 * smooth_noise(H, W, 25 * unit, rng))
    edge = m * (1 + 0.55 * rough * n)
    return smoothstep(-1.5 * unit, 1.5 * unit, d - edge).astype(F32)


def glaze_stack(D, levels, rng, unit, paper, gran, lum_fn):
    """Layer transparent washes, light to dark.

    D      target density (H,W,C) (C = 3 for colour, 1 for ink)
    levels list of dicts: sigma, thr, frac, flat, soft, rim, edge, wobble, flow
    """
    H, W, C = D.shape
    cur = np.zeros_like(D)
    for lev in levels:
        s = lev['sigma'] * unit
        tgt = fblur(D, s)
        res = tgt - cur
        r = lum_fn(res)
        # wobbly, organic edges: perturb the field that is thresholded
        wob = lev.get('wobble', 0.0)
        if wob > 0:
            r = r + wob * (0.65 * smooth_noise(H, W, 30 * unit, rng) + 0.35 * smooth_noise(H, W, 7 * unit, rng))
        hard = (r > lev['thr']).astype(F32)
        mask = fblur(hard, lev['soft'] * unit) if lev['soft'] > 0 else hard
        amt = np.clip(res, 0, None) * lev['frac']
        if lev.get('flat', 0) > 0:
            sr = 3 * s + 4 * unit
            reg = fblur(amt * mask[..., None], sr) / (fblur(mask, sr)[..., None] + 1e-3)
            amt = amt * (1 - lev['flat']) + reg * lev['flat']
        # wet edge: pigment is carried to the rim of the wash as it dries
        rim = np.clip(mask - fblur(mask, lev['rim'] * unit), 0, None) * 2.0
        m = mask * (1 + lev['edge'] * rim)
        flow = 1 + lev.get('flow', 0.15) * smooth_noise(H, W, 70 * unit, rng, octaves=3)
        floc = 1 + 0.08 * smooth_noise(H, W, 3 * unit, rng)
        cur = cur + amt * (m * flow * floc * gran)[..., None]
    return cur


def warp(img, amp, unit, rng):
    """Displace an image by a smooth random field (amp in px)."""
    H, W = img.shape
    y, x = np.mgrid[0:H, 0:W].astype(F32)
    dx = amp * (0.75 * smooth_noise(H, W, 45 * unit, rng) + 0.25 * smooth_noise(H, W, 9 * unit, rng))
    dy = amp * (0.75 * smooth_noise(H, W, 45 * unit, rng) + 0.25 * smooth_noise(H, W, 9 * unit, rng))
    return ndi.map_coordinates(img, [y + dy, x + dx], order=1, mode='nearest').astype(F32)


# ----------------------------------------------------------------------------
def tonal_glazes(D, taus, sigmas, soft, rng, unit, gran, rim_px=3.0, edge=1.0, wobble=0.02,
                 flow=0.15, hue_var=0.0, lum_fn=None, edge_px=0.8, mottle=0.1):
    """Nested washes: glaze k covers where the (blurred) density exceeds taus[k].

    Each glaze adds (taus[k+1]-taus[k]) of density with the local hue.  The
    sigmoid is in density units, so a glaze edge is crisp across a real image
    edge and soft (wet-in-wet) across a smooth gradient.
    """
    H, W, C = D.shape
    lum_fn = lum_fn or (lambda x: x.mean(-1))
    cur = np.zeros_like(D)
    taus = list(taus)
    for k, tau in enumerate(taus):
        dt = (taus[k + 1] - tau) if k + 1 < len(taus) else (tau - taus[k - 1])
        s = sigmas[k] * unit
        Db = fblur(D, s)
        d = lum_fn(Db)
        chroma = Db / (d[..., None] + 1e-3)
        chroma = np.clip(chroma, 0, 4)
        if hue_var > 0 and C == 3:
            chroma = chroma * (1 + hue_var * rng.standard_normal(3).astype(F32))
        if wobble > 0:
            d = warp(d, wobble * unit, unit, rng)
        # (d - tau)/|grad d| is ~ the distance to the wash edge in px: crisp
        # edges of fixed width where the image has an edge, soft (wet-in-wet)
        # where it only has a gentle gradient.
        gy, gx = np.gradient(d)
        gm = np.hypot(gx, gy)
        x = (d - tau) / (gm * edge_px * unit + soft)
        mask = _sigmoid(x)
        rim = np.clip(mask - fblur(mask, rim_px * unit), 0, None) * 2.0
        # flow blotches belong to the big wet washes; later glazes are smoother
        fl = 1 + flow * (1 - 0.7 * k / max(len(taus) - 1, 1)) * smooth_noise(H, W, 110 * unit, rng, octaves=2)
        floc = (1 + 0.07 * smooth_noise(H, W, 2.5 * unit, rng)
                + mottle * smooth_noise(H, W, 14 * unit, rng, octaves=2))
        cur = cur + (dt * mask * (1 + edge * rim) * fl * floc * gran)[..., None] * chroma
    return cur


def wash_strokes(D, cur, rng, unit, detail, paper_h, M_):
    """Transparent brush marks where the washes still miss detail.

    Strokes follow the image structure, carry the missing pigment colour,
    darken at their rims and break up on the paper tooth (dry brush).
    """
    H, W, C = D.shape
    add = np.zeros_like(D)
    btile = bristle_tile(rng, along=12.0, across=0.7)
    for R0, thr, nh, amt in M_['layers']:
        R = max(R0 * unit, 0.8)
        Df = fblur(D, 0.6 * R)
        # darker-than-surroundings detail, plus a share of the local colour
        res = np.clip(Df - fblur(D, 4 * R), 0, None) + M_['base'] * Df
        r = np.clip(Df - fblur(D, 4 * R), 0, None).mean(-1)
        seeds = seed_grid(r, 1.6 * R, thr / detail, rng, jitter=0.8)
        if len(seeds) == 0:
            continue
        L = -Df.mean(-1)
        wx, wy = orientation_field(L, max(0.5 * R, 1.0), 2.5 * R, 0.0, 0.3)
        col = bilerp(res, seeds[:, 0], seeds[:, 1])
        Ms = len(seeds)
        Pts, lo, hi = trace(seeds, col, res, wx, wy, 1.0 * R, nh, 0.7, M_['cthresh'], rng=rng)
        P2, lo2, hi2 = smooth_subdivide(Pts, lo, hi) if R >= 4 else (Pts, lo, hi)
        width = R * np.clip(1 + 0.2 * rng.standard_normal(Ms), 0.6, 1.5).astype(F32)
        key = rng.permutation(Ms).astype(np.int64)
        St = Strokes(P2, lo2, hi2, width, key, taper='oil', rag=0.08, cap='round')
        zbuf, Ls = rasterize(St, H, W)
        has, sid, un, vn = decode(zbuf, np.argsort(key))
        ul = un * Ls[sid]
        offu = rng.integers(0, 256, Ms)
        offv = rng.integers(0, 256, Ms)
        tu = ((ul / 2.0).astype(np.int32) + offu[sid]) & 255
        tv = (((vn + 1) * width[sid] / (1.0 + 0.1 * R)).astype(np.int32) + offv[sid]) & 255
        b = btile[tu, tv]
        ph = paper_h.reshape(-1)[has]
        # dry brush: toward the end of the stroke only the paper peaks take paint
        dry = M_['dry'] * smoothstep(0.4, 1.0, un)
        a = smoothstep(-0.3, 0.3, 0.6 * b + ph + 2.2 - 5.0 * dry)
        rim = 1 + M_['rim'] * smoothstep(0.6, 0.95, np.abs(vn))
        dens = col[sid] * (amt * a * rim)[:, None]
        af = add.reshape(-1, C)
        af[has] = af[has] + dens
    return add


def watercolor(src, rng, unit, detail, P):
    H, W = src.shape[:2]
    paper_h = pm.cold_press_paper(H, W, unit, rng)
    pc = np.array(P['paper'], F32)
    src = pm.saturate(src, P.get('sat', 1.0))
    # high-key: watercolour is a light medium, so lift dark images toward a
    # mean luminance of ~0.4 (bright images are left alone)
    mL = float(luminance(src).mean())
    key = np.clip(np.log(P.get('key_target', 0.4)) / np.log(max(mL, 1e-3)), P.get('key_min', 0.7), 1.0)
    src = np.clip(src, 0, 1) ** key
    C = np.clip(src / pc, 0.01, 1.0)
    D = -np.log(C) * P['density']
    dmax = P.get('dmax', 0)
    if dmax > 0:                       # high-key: soft-clip the darkest passages
        D = dmax * (1 - np.exp(-D / dmax))
    gran = np.exp(-P['gran'] * paper_h)
    gran = (gran / gran.mean()).astype(F32)
    taus = np.array(P['taus'], F32) / detail ** 0.3
    cur = tonal_glazes(D, taus, P['sigmas'], P['soft'], rng, unit, 1.0, P['rim'], P['edge'],
                       P['wobble'], P['flow'], P.get('hue_var', 0.0), None,
                       P.get('edge_px', 0.8), P.get('mottle', 0.1))
    # final accents: small dark marks where detail is still missing
    Df = fblur(D, P['detail_sigma'] * unit)
    res = (Df - cur).mean(-1)
    acc = _sigmoid((res - P['detail_thr'] / detail) / 0.02)
    acc = fblur(acc, 0.5 * unit)
    cur = cur + np.clip(Df - cur, 0, None) * (acc * P['detail_frac'] * gran)[..., None]
    if P.get('marks'):
        cur = cur + wash_strokes(D, cur, rng, unit, detail, paper_h, P['marks'])
    # granulation: pigment settles into the paper's valleys; strongest in
    # mid-density washes, weaker in heavy darks (which would turn gritty)
    cur = cur * (1 + (gran[..., None] - 1) / (1 + 1.5 * cur))
    if P.get('deckle', 0) > 0:
        dm = deckle_mask(H, W, unit, rng, P['deckle'], P.get('deckle_rough', 0.5))
        rim = np.clip(dm - fblur(dm, 3 * unit), 0, None) * 2
        cur = cur * (dm * (1 + 0.8 * rim))[..., None]
    out = pc * np.exp(-cur)
    diff, _ = pm.light_height(paper_h * unit * 0.5, 1.0, blur=0.0)
    out = out * (1 + P['paper_relief'] * (diff[..., None] - 1))
    return np.clip(out, 0, 1)


# ----------------------------------------------------------------------------
def ink_strokes(L, dark_map, rng, unit, detail, S, dens_scale=1.0):
    """Calligraphic contour strokes along image edges.  Returns density map."""
    H, W = L.shape
    out = np.zeros((H, W), F32)
    btile = bristle_tile(rng, along=14.0, across=0.6)
    for lay in S['layers']:
        kind, R0, thr, stopf, nh, dmul = lay[:6]
        spacing = lay[6] if len(lay) > 6 else 2.2
        R = max(R0 * unit, 0.8)
        Lb = fblur(L, max(0.6 * R, 0.8 * unit))
        if kind == 'contour':
            gy, gx = np.gradient(Lb)
            E = np.hypot(gx, gy) * R            # contrast across one brush width
            cth = 10.0
        else:                                   # texture strokes on dark passages
            E = dark_map
            cth = 0.25
        seeds = seed_grid(E, spacing * R, thr / detail, rng, jitter=0.9 if kind != 'contour' else 0.3)
        if kind != 'contour' and len(seeds):
            # sparser in lighter passages
            keep = rng.random(len(seeds)) < np.clip(bilerp(E, seeds[:, 0], seeds[:, 1]) * 1.3, 0, 1)
            seeds = seeds[keep]
        if len(seeds) == 0:
            continue
        wx, wy = orientation_field(Lb, max(0.5 * R, 1.0), 1.5 * R if kind == 'contour' else 3 * R,
                                   0.0, 0.02 if kind == 'contour' else 0.2)
        col = bilerp(Lb, seeds[:, 0], seeds[:, 1])
        M = len(seeds)
        Pts, lo, hi = trace(seeds, col, Lb, wx, wy, 1.0 * R, nh, 0.8, cth,
                            stopmap=E, stopthr=stopf * thr / detail, rng=rng)
        P2, lo2, hi2 = smooth_subdivide(Pts, lo, hi) if R >= 3 else (Pts, lo, hi)
        width = R * np.clip(1 + 0.25 * rng.standard_normal(M), 0.5, 1.6).astype(F32)
        key = rng.permutation(M).astype(np.int64)
        St = Strokes(P2, lo2, hi2, width, key, taper='ink', rag=0.10, cap='round')
        zbuf, Ls = rasterize(St, H, W)
        has, sid, un, vn = decode(zbuf, np.argsort(key))
        # stroke darkness follows the edge contrast at the seed
        es = bilerp(E, seeds[:, 0], seeds[:, 1])
        if kind == 'contour':
            sd = dmul * np.clip(0.5 + 0.4 * es / (3 * thr), 0.5, 1.0) * (0.8 + 0.3 * rng.random(M))
        else:
            sd = dmul * np.clip(es, 0, 1) * (0.7 + 0.6 * rng.random(M))
        ul = un * Ls[sid]
        offu = rng.integers(0, 256, M)
        offv = rng.integers(0, 256, M)
        tu = ((ul / 2.0).astype(np.int32) + offu[sid]) & 255
        tv = (((vn + 1) * width[sid] / (0.8 + 0.12 * R)).astype(np.int32) + offv[sid]) & 255
        b = btile[tu, tv]
        dryness = S['dry'] * smoothstep(0.35, 1.0, un) + 0.15 * np.abs(vn) ** 3
        a = smoothstep(-0.25, 0.25, b + 2.6 - 5.0 * dryness)
        prof = 0.8 + 0.2 * np.sqrt(np.clip(1 - vn * vn, 0, 1))
        d = sd[sid] * prof * a * (1 + 0.12 * b)
        o = out.reshape(-1)
        o[has] = o[has] + d * dens_scale
    return out


def saliency(src, unit, scales):
    """Multi-scale centre-surround colour contrast, 0..1: where the subject is."""
    H, W = src.shape[:2]
    # rough opponent space: lightness + two chroma axes
    Lc = luminance(src)
    rg = src[..., 0] - src[..., 1]
    by = 0.5 * (src[..., 0] + src[..., 1]) - src[..., 2]
    X = np.stack([Lc, 0.7 * rg, 0.5 * by], -1).astype(F32)
    sal = np.zeros((H, W), F32)
    for s in scales:
        c = fblur(X, s * unit / 5)
        sr = fblur(X, s * unit)
        sal += np.linalg.norm(c - sr, axis=2)
    sal = fblur(sal, 8 * unit)
    sal = smoothstep(0.15, 0.85, sal / (np.percentile(sal, 98) + 1e-9))
    return sal.astype(F32)


def ink_wash(src, rng, unit, detail, P):
    H, W = src.shape[:2]
    paper_h, fib = pm.rice_paper(H, W, unit, rng)
    pc = np.array(P['paper'], F32)
    L = luminance(src)
    if P.get('decolor', 0) > 0:
        # contrast-preserving grey: blend luminance with the first principal
        # component of the colours (keeps red-vs-grey separations visible)
        X = src.reshape(-1, 3)[::7]
        Xm = X.mean(0)
        w, V = np.linalg.eigh(np.cov((X - Xm).T))
        pcv = V[:, -1] * np.sign(V[:, -1] @ np.array([0.299, 0.587, 0.114]))
        proj = (src - Xm) @ pcv.astype(F32)
        proj = (proj - proj.min()) / (proj.max() - proj.min() + 1e-6)
        L = (1 - P['decolor']) * L + P['decolor'] * (proj * (L.max() - L.min()) + L.min())
    if P.get('chroma_light', 0) > 0:
        # ink painters read saturated colour as light: lift chromatic areas.
        # In dark images colour contrast is what separates the subject, so
        # lift more there.
        cl = P['chroma_light'] + P.get('chroma_light_dark', 0) * (1 - smoothstep(0.25, 0.5, float(L.mean())))
        L = L + cl * (src.max(-1) - src.min(-1))
    lo, hi = np.percentile(L[::3, ::3], [0.5, 99.8])
    t = np.clip((L - lo) / (hi - lo + 1e-6), 0, 1) * P['stretch'] + L * (1 - P['stretch'])
    dark = np.clip((1 - t - P['lift']) / (1 - P['lift']), 0, 1) ** P['gamma']
    # local contrast: darker than the surroundings -> more ink (shadows, gullies)
    loc = np.clip(fblur(t, 25 * unit) - t, 0, None) * P['local']
    dark = np.clip(dark + loc, 0, 1.2)
    # notan: ink goes where there is form; flat passages (sky, plain walls)
    # stay close to bare paper
    if P['empty'] < 1:
        # light, featureless passages (skies) become bare paper; dark masses
        # (a dark wall) keep their ink
        form = saliency(src, unit, P.get('form_scales', (30, 90, 250)))
        keep = np.maximum(form, smoothstep(0.35, 0.7, fblur(dark, 20 * unit)))
        dark = dark * (P['empty'] + (1 - P['empty']) * keep)
    D = (dark * P['dmax'])[..., None]
    taus = np.array(P['taus'], F32) / detail ** 0.3
    cur = tonal_glazes(D, taus, P['sigmas'], P['soft'], rng, unit, 1.0, P['rim'], P['edge'],
                       P['wobble'], P['flow'], 0.0, None, P.get('edge_px', 1.0), P.get('mottle', 0.08))[..., 0]
    # ink bleeds a little into the fibres
    cur = cur * (1 - P['bleed']) + fblur(cur, 2.5 * unit) * P['bleed']
    # texture strokes go where it is dark AND there is structure (not flat sky)
    gy, gx = np.gradient(fblur(t, 1.5 * unit))
    struct = fblur(np.hypot(gx, gy), 12 * unit) * unit
    struct = smoothstep(0.2, 1.0, struct / (np.percentile(struct, 90) + 1e-6))
    tex_map = fblur(dark, 2 * unit) * struct
    strokes = ink_strokes(L, tex_map, rng, unit, detail, P['strokes'])
    strokes = strokes + P['bleed'] * 0.5 * fblur(strokes, 1.5 * unit) * (0.5 + fib)
    dens = cur + strokes
    if P.get('deckle', 0) > 0:
        dens = dens * deckle_mask(H, W, unit, rng, P['deckle'], 0.4)
    ink = np.array(P['ink'], F32)
    out = pc * np.exp(-dens[..., None] * ink)
    # fibres: slightly lighter strands on the paper, slightly darker where inked
    out = out * (1 + 0.035 * fib[..., None] * (1 - 2 * np.clip(dens, 0, 1)[..., None]))
    diff, _ = pm.light_height(paper_h * unit * 0.35, 1.0, blur=0.0)
    out = out * (1 + P['paper_relief'] * (diff[..., None] - 1))
    return np.clip(out, 0, 1)
