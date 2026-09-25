"""Surfaces (canvas, papers), colour helpers and lighting for paint.py."""
import numpy as np
from scipy import ndimage as ndi

from paint_strokes import F32, smooth_noise, luminance


# ----------------------------------------------------------------------------
# colour helpers
# ----------------------------------------------------------------------------
def saturate(img, amount, pivot_lum=None):
    L = luminance(img)[..., None]
    return np.clip(L + (img - L) * amount, 0, 1)


def curve(img, contrast=1.0, lift=0.0, gamma=1.0):
    x = np.clip(img, 0, 1) ** gamma
    x = 0.5 + (x - 0.5) * contrast
    return np.clip(lift + x * (1 - lift), 0, 1)


def hue_rotate(cols, ang):
    """Rotate RGB colours (N,3) about the grey axis by ang (N,) radians."""
    k = np.array([1, 1, 1], F32) / np.sqrt(3)
    c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
    kxc = np.cross(k[None], cols)
    kdc = (cols @ k)[:, None] * k[None]
    return cols * c + kxc * s + kdc * (1 - c)


def jitter_colours(cols, rng, lum=0.04, hue=0.05, sat=0.08):
    M = len(cols)
    out = hue_rotate(cols, rng.standard_normal(M).astype(F32) * hue)
    Lc = (out @ np.array([0.299, 0.587, 0.114], F32))[:, None]
    out = Lc + (out - Lc) * (1 + sat * rng.standard_normal((M, 1)).astype(F32))
    out = out * (1 + lum * rng.standard_normal((M, 1)).astype(F32))
    return np.clip(out, 0, 1).astype(F32)


def kmeans_palette(img, k, rng, iters=12, sample=60000):
    X = img.reshape(-1, 3)
    X = X[rng.choice(len(X), min(sample, len(X)), replace=False)]
    # k-means++ init
    C = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        d = np.min(((X[:, None] - np.array(C)[None]) ** 2).sum(-1), 1)
        C.append(X[rng.choice(len(X), p=d / d.sum())])
    C = np.array(C, F32)
    for _ in range(iters):
        lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), 1)
        for i in range(k):
            m = lab == i
            if m.any():
                C[i] = X[m].mean(0)
    return C


def nearest_palette(cols, C):
    return C[np.argmin(((cols[:, None] - C[None]) ** 2).sum(-1), 1)]


# ----------------------------------------------------------------------------
# surfaces (height fields in roughly [-1, 1])
# ----------------------------------------------------------------------------
def canvas_weave(H, W, unit, rng, period=5.0):
    p = period * unit
    y, x = np.mgrid[0:H, 0:W].astype(F32)
    xj = x + 0.35 * p * smooth_noise(H, W, 25 * unit, rng)
    yj = y + 0.35 * p * smooth_noise(H, W, 25 * unit, rng)
    fx, fy = xj / p, yj / p
    ix, iy = np.floor(fx), np.floor(fy)
    rx, ry = fx - ix, fy - iy
    over = ((ix + iy) % 2) == 0
    warp = np.sin(np.pi * rx) ** 0.8 * (0.55 + 0.45 * np.sin(np.pi * ry))
    weft = np.sin(np.pi * ry) ** 0.8 * (0.55 + 0.45 * np.sin(np.pi * rx))
    h = np.where(over, warp, weft)
    # slubs: thickness variation along threads
    h = h * (1 + 0.25 * smooth_noise(H, W, 12 * unit, rng))
    h += 0.12 * smooth_noise(H, W, 0.8 * unit, rng)
    return ((h - h.mean()) / (h.std() + 1e-6)).astype(F32)


def cold_press_paper(H, W, unit, rng):
    h = (0.45 * smooth_noise(H, W, 1.6 * unit, rng) +
         1.0 * smooth_noise(H, W, 4.0 * unit, rng) +
         0.55 * smooth_noise(H, W, 10 * unit, rng))
    # rounded bumps: soft-clip the peaks
    h = np.tanh(0.9 * h)
    return ((h - h.mean()) / (h.std() + 1e-6)).astype(F32)


def rice_paper(H, W, unit, rng, n_fibres=None):
    """Fine grain + long thin fibres, as in xuan paper."""
    h = 0.5 * smooth_noise(H, W, 1.2 * unit, rng) + 0.5 * smooth_noise(H, W, 6 * unit, rng)
    fib = np.zeros((H, W), F32)
    if n_fibres is None:
        n_fibres = int(H * W / (unit * unit) / 900)
    # each fibre is a random-walk polyline, splatted
    L = 24
    x = rng.random(n_fibres) * W
    y = rng.random(n_fibres) * H
    a = rng.random(n_fibres) * np.pi * 2
    for k in range(L):
        a += rng.standard_normal(n_fibres) * 0.25
        x += np.cos(a) * 2.2 * unit
        y += np.sin(a) * 2.2 * unit
        xi = np.clip(x.astype(int), 0, W - 1)
        yi = np.clip(y.astype(int), 0, H - 1)
        np.add.at(fib, (yi, xi), 1.0)
    fib = ndi.gaussian_filter(fib, 0.6 * unit)
    fib = fib / (fib.max() + 1e-6)
    return ((h - h.mean()) / (h.std() + 1e-6)).astype(F32), np.clip(fib * 4, 0, 1).astype(F32)


def fine_grain(H, W, unit, rng):
    h = 0.7 * smooth_noise(H, W, 1.0 * unit, rng) + 0.5 * smooth_noise(H, W, 3.0 * unit, rng)
    return ((h - h.mean()) / (h.std() + 1e-6)).astype(F32)


# ----------------------------------------------------------------------------
# lighting of a height map
# ----------------------------------------------------------------------------
def light_height(Hmap, strength, light=(-0.55, -0.7, 0.9), spec=0.0, shin=40.0, blur=0.6):
    """Returns (diffuse_factor, specular) for a relief height map (pixels)."""
    h = ndi.gaussian_filter(Hmap, blur) if blur > 0 else Hmap
    gy, gx = np.gradient(h)
    nx, ny, nz = -gx * strength, -gy * strength, np.ones_like(h)
    inv = 1.0 / np.sqrt(nx * nx + ny * ny + 1)
    nx, ny, nz = nx * inv, ny * inv, nz * inv
    l = np.array(light, F32)
    l /= np.linalg.norm(l)
    ndl = nx * l[0] + ny * l[1] + nz * l[2]
    diff = ndl / l[2]
    sp = None
    if spec > 0:
        hv = l + np.array([0, 0, 1], F32)
        hv /= np.linalg.norm(hv)
        ndh = np.clip(nx * hv[0] + ny * hv[1] + nz * hv[2], 0, 1)
        sp = spec * ndh ** shin
    return diff.astype(F32), sp
