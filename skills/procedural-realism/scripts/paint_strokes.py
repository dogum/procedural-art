"""Stroke engine for paint.py: orientation fields, vectorised stroke tracing and
z-buffered stroke rasterisation.  numpy + scipy + pillow only.

Coordinates are pixel centres: x in [0, W), y in [0, H).
"""
import numpy as np
from scipy import ndimage as ndi
from PIL import Image

F32 = np.float32
LUMA = np.array([0.299, 0.587, 0.114], F32)


# ----------------------------------------------------------------------------
# small image helpers
# ----------------------------------------------------------------------------
def luminance(img):
    return (img @ LUMA).astype(F32)


def resize_f(ch, W, H, resample=Image.BILINEAR):
    """Resize a single float32 channel."""
    return np.asarray(Image.fromarray(np.ascontiguousarray(ch, F32), 'F').resize((W, H), resample), F32)


def _blur2(ch, s):
    if s < 0.3:
        return ch
    if s <= 6:
        return ndi.gaussian_filter(ch, s, mode='nearest')
    # large sigma: box-downsample, blur small, upsample
    k = int(s / 3)
    H, W = ch.shape
    Hs, Ws = max(2, H // k), max(2, W // k)
    small = resize_f(ch, Ws, Hs, Image.BOX)
    sx = W / Ws
    small = ndi.gaussian_filter(small, np.sqrt(max(s * s - (sx * sx) / 4.0, 1.0)) / sx, mode='nearest')
    return resize_f(small, W, H, Image.BICUBIC)


def fblur(img, s):
    """Fast approximate gaussian blur (2D or HxWxC)."""
    img = img.astype(F32, copy=False)
    if img.ndim == 3:
        return np.stack([_blur2(img[..., c], s) for c in range(img.shape[2])], -1)
    return _blur2(img, s)


def smooth_noise(H, W, scale, rng, octaves=1, gain=0.5):
    """Zero-mean, ~unit-std band-limited noise with feature size ~scale px."""
    out = np.zeros((H, W), F32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        sc = max(scale / (2 ** o), 0.7)
        if sc < 3:
            n = ndi.gaussian_filter(rng.standard_normal((H, W)).astype(F32), sc * 0.5, mode='wrap')
        else:
            h, w = int(np.ceil(H / sc)) + 3, int(np.ceil(W / sc)) + 3
            g = ndi.gaussian_filter(rng.standard_normal((h, w)).astype(F32), 0.8, mode='wrap')
            n = resize_f(g, int(w * sc), int(h * sc), Image.BICUBIC)[:H, :W]
        n = (n - n.mean()) / (n.std() + 1e-8)
        out += amp * n
        tot += amp * amp
        amp *= gain
    return out / np.sqrt(tot)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def bilerp(img, x, y):
    """Bilinear sample img (H,W) or (H,W,C) at float coords (x, y)."""
    H, W = img.shape[:2]
    x = np.clip(x, 0, W - 1.001)
    y = np.clip(y, 0, H - 1.001)
    x0 = x.astype(np.int32)
    y0 = y.astype(np.int32)
    fx = (x - x0).astype(F32)
    fy = (y - y0).astype(F32)
    flat = img.reshape(H * W, -1)
    i = y0 * W + x0
    if img.ndim == 3:
        fx = fx[:, None]
        fy = fy[:, None]
    a = flat[i]
    b = flat[i + 1]
    c = flat[i + W]
    d = flat[i + W + 1]
    if img.ndim == 2:
        a, b, c, d = a[:, 0], b[:, 0], c[:, 0], d[:, 0]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


# ----------------------------------------------------------------------------
# orientation field
# ----------------------------------------------------------------------------
def orientation_field(L, s_grad, s_tensor, fallback_angle=0.0, fallback_w=0.08,
                      jitter=0.0, rng=None, unit=1.0):
    """Smoothed structure tensor in doubled-angle form.

    Returns (wx, wy): sampling them and taking 0.5*atan2(wy, wx) gives the
    dominant *gradient* angle; strokes run perpendicular to it (along contours).
    fallback_angle is the stroke (tangent) angle used where the image is flat.
    """
    Ls = fblur(L, s_grad)
    gy, gx = np.gradient(Ls)
    Jxx = fblur(gx * gx, s_tensor)
    Jyy = fblur(gy * gy, s_tensor)
    Jxy = fblur(gx * gy, s_tensor)
    wx = Jxx - Jyy
    wy = 2 * Jxy
    m = np.hypot(wx, wy)
    ref = np.percentile(m[::5, ::5], 85) + 1e-12
    ga = 2 * (fallback_angle + np.pi / 2)
    wx = wx + fallback_w * ref * np.cos(ga)
    wy = wy + fallback_w * ref * np.sin(ga)
    if jitter > 0 and rng is not None:
        H, W = L.shape
        a = 2 * jitter * smooth_noise(H, W, 40 * unit, rng, octaves=2)
        c, s = np.cos(a), np.sin(a)
        wx, wy = wx * c - wy * s, wx * s + wy * c
    return wx.astype(F32), wy.astype(F32)


def tangent_at(wx, wy, x, y):
    a = bilerp(wx, x, y)
    b = bilerp(wy, x, y)
    g = 0.5 * np.arctan2(b, a)
    return np.stack([-np.sin(g), np.cos(g)], 1).astype(F32)


# ----------------------------------------------------------------------------
# seeding
# ----------------------------------------------------------------------------
def seed_grid(err, spacing, T, rng, full=False, jitter=0.5):
    """Hertzmann seeding: one seed per grid cell whose mean error exceeds T,
    placed at the cell's max-error pixel (or randomly if full)."""
    H, W = err.shape
    g = max(1, int(round(spacing)))
    gh, gw = int(np.ceil(H / g)), int(np.ceil(W / g))
    pad = np.zeros((gh * g, gw * g), F32)
    pad[:H, :W] = err
    blocks = pad.reshape(gh, g, gw, g).transpose(0, 2, 1, 3).reshape(gh, gw, g * g)
    cy, cx = np.mgrid[0:gh, 0:gw]
    if full:
        sel = np.ones((gh, gw), bool)
        oy = rng.random((gh, gw)) * g
        ox = rng.random((gh, gw)) * g
    else:
        mean = blocks.mean(-1)
        sel = mean > T
        am = blocks.argmax(-1)
        oy = (am // g) + (rng.random((gh, gw)) - 0.5) * g * jitter
        ox = (am % g) + (rng.random((gh, gw)) - 0.5) * g * jitter
    x = (cx * g + ox)[sel]
    y = (cy * g + oy)[sel]
    x = np.clip(x, 0, W - 1)
    y = np.clip(y, 0, H - 1)
    return np.stack([x, y], 1).astype(F32)


# ----------------------------------------------------------------------------
# tracing
# ----------------------------------------------------------------------------
def trace(seeds, col, ref, wx, wy, step, n_half, fc=0.7, cthresh=0.1,
          stopmap=None, stopthr=None, min_fwd=1, rng=None, turn_noise=0.0, rot=None, maxk=None):
    """Trace curved strokes in lockstep, both directions from each seed.

    A stroke stops when the reference colour along it differs from the stroke
    colour by more than cthresh (per stroke array or scalar), or when
    stopmap < stopthr.  Returns (P, lo, hi): points (M, 2*n_half+1, 2) with the
    valid run P[i, lo[i]:hi[i]+1].
    """
    M = len(seeds)
    H, W = ref.shape[:2]
    K = 2 * n_half + 1
    P = np.empty((M, K, 2), F32)
    P[:, n_half] = seeds
    if rot is not None:
        rc, rs = np.cos(rot).astype(F32), np.sin(rot).astype(F32)

    def tan_(x, y):
        t = tangent_at(wx, wy, x, y)
        if rot is not None:
            t = np.stack([t[:, 0] * rc - t[:, 1] * rs, t[:, 0] * rs + t[:, 1] * rc], 1)
        return t
    d0 = tan_(seeds[:, 0], seeds[:, 1])
    if rng is not None:
        flip = rng.random(M) < 0.5
        d0[flip] *= -1
    counts = []
    cth = np.broadcast_to(np.asarray(cthresh, F32), (M,))
    for sgn in (1, -1):
        p = seeds.copy()
        d = d0 * sgn
        act = np.ones(M, bool)
        cnt = np.zeros(M, np.int32)
        for k in range(n_half):
            t = tan_(p[:, 0], p[:, 1])
            if turn_noise > 0 and rng is not None:
                a = rng.standard_normal(M).astype(F32) * turn_noise
                c, s = np.cos(a), np.sin(a)
                t = np.stack([t[:, 0] * c - t[:, 1] * s, t[:, 0] * s + t[:, 1] * c], 1)
            neg = (t * d).sum(1) < 0
            t[neg] *= -1
            nd = fc * t + (1 - fc) * d
            nd /= np.linalg.norm(nd, axis=1, keepdims=True) + 1e-8
            q = p + step[:, None] * nd if np.ndim(step) else p + step * nd
            inb = (q[:, 0] > -2) & (q[:, 0] < W + 1) & (q[:, 1] > -2) & (q[:, 1] < H + 1)
            c = bilerp(ref, q[:, 0], q[:, 1])
            diff = np.linalg.norm(c - col, axis=1) if ref.ndim == 3 else np.abs(c - col)
            ok = diff < cth
            if stopmap is not None:
                ok &= bilerp(stopmap, q[:, 0], q[:, 1]) >= stopthr
            if sgn == 1 and k < min_fwd:
                ok[:] = True
            if maxk is not None:
                ok &= k < maxk
            ok &= act & inb
            p = np.where(ok[:, None], q, p)
            d = np.where(ok[:, None], nd, d)
            act = ok
            cnt += ok
            if sgn == 1:
                P[:, n_half + 1 + k] = p
            else:
                P[:, n_half - 1 - k] = p
            if not act.any():
                # fill the rest with the final positions
                for kk in range(k + 1, n_half):
                    if sgn == 1:
                        P[:, n_half + 1 + kk] = p
                    else:
                        P[:, n_half - 1 - kk] = p
                break
        counts.append(cnt)
    lo = n_half - counts[1]
    hi = n_half + counts[0]
    return P, lo, hi


def smooth_subdivide(P, lo, hi, passes=1):
    """[1,2,1] smoothing along the stroke + Catmull-Rom midpoint subdivision."""
    Q = P.copy()
    for _ in range(passes):
        Qp = np.concatenate([Q[:, :1], Q, Q[:, -1:]], 1)
        Q = 0.25 * Qp[:, :-2] + 0.5 * Qp[:, 1:-1] + 0.25 * Qp[:, 2:]
        # keep true endpoints
        idx = np.arange(len(P))
        Q[idx, lo] = P[idx, lo]
        Q[idx, hi] = P[idx, hi]
    M, K, _ = Q.shape
    Qp = np.concatenate([Q[:, :1], Q, Q[:, -1:]], 1)
    mid = (-Qp[:, :-3] + 9 * Qp[:, 1:-2] + 9 * Qp[:, 2:-1] - Qp[:, 3:]) / 16.0
    out = np.empty((M, 2 * K - 1, 2), F32)
    out[:, 0::2] = Q
    out[:, 1::2] = mid
    return out, 2 * lo, 2 * hi


# ----------------------------------------------------------------------------
# rasterisation
# ----------------------------------------------------------------------------
class Strokes:
    """Per-stroke attributes for one layer."""

    def __init__(self, P, lo, hi, width, key, taper='oil', rag=0.12, seedval=None, cap='round', capk=0.4,
                 pressure=None):
        self.cap, self.capk = cap, capk
        self.pressure = pressure        # optional (M, K) width factor per path point (brush pressure)
        self.P, self.lo, self.hi = P, lo, hi
        self.width = width.astype(F32)
        self.key = key.astype(np.int64)
        self.taper = taper
        self.rag = rag
        self.M = len(P)
        self.seedval = seedval


def taper_profile(kind, un):
    if kind == 'oil':
        # press at start, lift at the end
        return (0.82 + 0.18 * smoothstep(0.0, 0.18, un)) * (1 - 0.35 * smoothstep(0.7, 1.0, un))
    if kind == 'dab':
        return 1 - 0.25 * smoothstep(0.6, 1.0, un)
    if kind == 'flat':
        return np.ones_like(un)
    if kind == 'ink':
        # thin entry, swell, long thin tail
        return 0.25 + 0.75 * smoothstep(0.0, 0.25, un) * (1 - 0.8 * smoothstep(0.45, 1.0, un))
    return np.ones_like(un)


_RAG = None


def _rag_table():
    global _RAG
    if _RAG is None:
        r = np.random.default_rng(12345).standard_normal(4096).astype(F32)
        r = ndi.gaussian_filter1d(r, 1.5, mode='wrap')
        _RAG = np.clip(r / r.std(), -2, 2)
    return _RAG


def rasterize(S, H, W, max_cand=3_000_000, sp=0.7, hsum=None, csum=None, wsum=None):
    """Z-buffer rasterisation of all strokes of a layer.

    Each segment is sampled on a small grid in its own (t, v) frame (spacing
    0.7 px, so every covered pixel is hit), samples are rounded to pixels and
    the exact (t, v) is recomputed there.  Returns a flat int64 buffer: -1
    where empty, else key<<24 | uq<<12 | vq, where uq = position along the
    stroke (0..4095 of its length) and vq = across position (0..4095 = -1..1),
    plus the stroke lengths L.
    """
    P, lo, hi = S.P, S.lo, S.hi
    M, K, _ = P.shape
    j = np.arange(K - 1)
    valid = (j[None] >= lo[:, None]) & (j[None] < hi[:, None])
    D = P[:, 1:] - P[:, :-1]
    sl = np.where(valid, np.hypot(D[..., 0], D[..., 1]), 0).astype(F32)
    cum = np.cumsum(sl, 1)
    u0 = cum - sl
    L = np.maximum(cum[:, -1], 1e-3).astype(F32)
    si, sj = np.nonzero(valid)
    ax, ay = P[si, sj, 0], P[si, sj, 1]
    bx, by = P[si, sj + 1, 0], P[si, sj + 1, 1]
    slen = sl[si, sj]
    ok = slen > 1e-6
    tx = np.where(ok, (bx - ax) / np.maximum(slen, 1e-6), 1.0).astype(F32)
    ty = np.where(ok, (by - ay) / np.maximum(slen, 1e-6), 0.0).astype(F32)
    # turn angle with the previous segment -> join extension
    first = sj == lo[si]
    last = sj == hi[si] - 1
    ptx = np.where(first, tx, np.roll(tx, 1))
    pty = np.where(first, ty, np.roll(ty, 1))
    cross = np.abs(ptx * ty - pty * tx)
    dotp = ptx * tx + pty * ty
    sin_turn = np.where(dotp < 0, 1.0, cross).astype(F32)
    u0s = u0[si, sj]
    rag = _rag_table()
    wmax = S.width if S.pressure is None else S.width * S.pressure.max(1)
    wb = wmax[si] * (1 + 2.0 * S.rag) + 0.6               # width bound per segment
    ea = np.where(first, wb, wb * sin_turn + 0.7).astype(F32)
    eb = np.where(last, wb, 0.5).astype(F32)
    tlen = slen + ea + eb
    zbuf = np.full(H * W, -1, np.int64)
    N = len(si)
    if N == 0:
        return zbuf, L
    nt = int(np.ceil(tlen.max() / sp)) + 1
    nv = int(np.ceil(2 * wb.max() / sp)) + 1
    ti = (np.arange(nt) * sp).astype(F32)
    vj = ((np.arange(nv) - (nv - 1) / 2) * sp).astype(F32)
    chunk = max(1, max_cand // (nt * nv))
    width, key = S.width, S.key
    seedoff = (np.arange(M) * 977) % 4096
    for c0 in range(0, N, chunk):
        c = slice(c0, c0 + chunk)
        tt = ti[None, :, None] - ea[c, None, None]
        pre = (tt <= slen[c, None, None] + eb[c, None, None]) & (np.abs(vj)[None, None, :] <= wb[c, None, None])
        seg, it, iv = np.nonzero(pre)
        seg = seg + c0
        t_ = ti[it] - ea[seg]
        v_ = vj[iv]
        px = np.rint(ax[seg] + t_ * tx[seg] - v_ * ty[seg]).astype(np.int32)
        py = np.rint(ay[seg] + t_ * ty[seg] + v_ * tx[seg]).astype(np.int32)
        inb = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        seg, px, py = seg[inb], px[inb], py[inb]
        s = si[seg]
        dx = px.astype(F32) - ax[seg]
        dy = py.astype(F32) - ay[seg]
        txs, tys = tx[seg], ty[seg]
        t = dx * txs + dy * tys
        v = dy * txs - dx * tys
        ln = slen[seg]
        u = u0s[seg] + np.clip(t, 0, ln)
        un = u / L[s]
        w = width[s] * taper_profile(S.taper, un)
        if S.pressure is not None:
            j0 = sj[seg]
            f = np.clip(t / np.maximum(ln, 1e-6), 0, 1)
            w = w * (S.pressure[s, j0] * (1 - f) + S.pressure[s, j0 + 1] * f)
        if S.rag > 0:
            ridx = ((u / np.maximum(width[s] * 0.2, 1.0)).astype(np.int32)
                    + seedoff[s] + (v > 0) * 1531) & 4095
            w = w * (1 + S.rag * rag[ridx])
        w2 = w * w
        fs, ls = first[seg], last[seg]
        body = (t >= np.where(fs, 0, -ea[seg])) & (t < np.where(ls, ln, ln + eb[seg]))
        d2a = dx * dx + dy * dy
        ex, ey = dx - ln * txs, dy - ln * tys
        d2b = ex * ex + ey * ey
        capa = fs & (t < 0)
        capb = ls & (t >= ln)
        if S.cap == 'round':
            inside = (body & (v * v <= w2)) | (capa & (d2a <= w2)) | (capb & (d2b <= w2))
            r = np.where(body, np.abs(v), np.sqrt(np.where(capa, d2a, d2b)))
        else:
            # flat brush: rounded-rectangle ends (superellipse, exponent 4)
            te = np.where(capa, -t, t - ln) / np.maximum(S.capk * w, 1e-3)
            q = te * te
            q = q * q
            vv = v * v / np.maximum(w2, 1e-6)
            inside = (body & (vv <= 1)) | ((capa | capb) & (q + vv * vv <= 1))
            r = np.abs(v)
        if not inside.any():
            continue
        vn = np.sign(v + 1e-6) * np.minimum(r / np.maximum(w, 1e-3), 1.0)
        if hsum is not None:
            # additive paint body: every stroke contributes, order-independent.
            # Only body/cap samples (not join extensions) to avoid double counts.
            hm = inside & ((t >= 0) & (t < ln) | capa | capb)
            # flat-topped body falling off over the outer 40% of the width
            hw = (smoothstep(0.0, 0.4, 1 - np.abs(vn[hm])) * smoothstep(0.0, 0.15, un[hm])
                  * (1 - 0.8 * smoothstep(0.75, 1.0, un[hm])))
            pix_h = (py[hm] * W + px[hm]).astype(np.intp)
            hsum += np.bincount(pix_h, weights=hw * (sp * sp), minlength=H * W).astype(F32)
            if csum is not None:
                ch = S.col[s[hm]]
                for c_ in range(3):
                    csum[:, c_] += np.bincount(pix_h, weights=hw * ch[:, c_], minlength=H * W)
                wsum += np.bincount(pix_h, weights=hw, minlength=H * W)
        vq = np.clip(((vn[inside] + 1) * 2047.5).astype(np.int64), 0, 4095)
        uq = np.clip((un[inside] * 4095).astype(np.int64), 0, 4095)
        pack = (key[s[inside]] << 24) | (uq << 12) | vq
        np.maximum.at(zbuf, (py[inside] * W + px[inside]).astype(np.intp), pack)
    return zbuf, L


def decode(zbuf, inv_key):
    has = zbuf >= 0
    z = zbuf[has]
    sid = inv_key[z >> 24]
    un = ((z >> 12) & 4095).astype(F32) / 4095
    vn = (z & 4095).astype(F32) / 4095 * 2 - 1
    return has, sid, un, vn


# ----------------------------------------------------------------------------
# textures used by the stroke shaders
# ----------------------------------------------------------------------------
def bristle_tile(rng, n=256, along=10.0, across=0.7):
    """Tileable noise, smooth along axis 0 (stroke direction), streaky across."""
    t = rng.standard_normal((n, n)).astype(F32)
    t = ndi.gaussian_filter(t, (along, across), mode='wrap')
    t = (t - t.mean()) / (t.std() + 1e-8)
    return t


def tile_sample(tile, u, v):
    """Bilinear, wrapping lookup in a square tile at float coords (u rows, v cols)."""
    n = tile.shape[0]
    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    fu = (u - u0).astype(F32)
    fv = (v - v0).astype(F32)
    u0 &= n - 1
    v0 &= n - 1
    u1 = (u0 + 1) & (n - 1)
    v1 = (v0 + 1) & (n - 1)
    return ((tile[u0, v0] * (1 - fv) + tile[u0, v1] * fv) * (1 - fu)
            + (tile[u1, v0] * (1 - fv) + tile[u1, v1] * fv) * fu)
