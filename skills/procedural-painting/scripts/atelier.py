"""atelier: a small oil-paint simulator for pictures composed in code.

A painting is a short Python script. It describes masses (a mask, the colour the mass should
be, the direction the brush moves over it) and the order they go down in; this module lays
them as brush strokes on a canvas, back to front, big brushes first.

What the paint does:
  * Colour mixes like pigment, not like light. Every colour is a 12-band reflectance spectrum
    and mixing happens on Kubelka-Munk absorption/scattering, so blue dragged into wet yellow
    goes green and a red worked into green goes brown, the way they do on a palette.
  * Strokes are bristle ribbons. Each bristle carries its own amount of paint, runs dry toward
    the end of the stroke, skips over the valleys of the canvas weave when it is dry, and picks
    up the wet paint it crosses (wet-into-wet), which drags streaks of the colour underneath.
  * Paint has thickness. Stroke bodies, edge ridges and bristle grooves build a relief that is
    lit by a raking light with a gloss highlight at the end.

numpy, scipy and pillow only (ffmpeg for the process film).  See ../SKILL.md.
"""
import glob
import json
import os
import subprocess

import numpy as np
from scipy import ndimage as ndi
from PIL import Image

F32 = np.float32
TAU = 2 * np.pi

# ----------------------------------------------------------------------------------------
# colour: sRGB <-> 12-band reflectance <-> Kubelka-Munk
# ----------------------------------------------------------------------------------------
# Smooth reflectance curves for the three sRGB primaries, 405..695 nm in 12 bands. They sum to
# one in every band (white is a flat spectrum) and _SPD2RGB maps them back to the identity, so
# an unmixed colour survives the round trip exactly. Solved once with scripts-side SLSQP from
# the CIE 1931 observer under D65; kept as a table so results never depend on a solver.
_BASIS = np.array([
    [0.66480, 0.04914, 0.28606], [0.00000, 0.06231, 0.93769], [0.00000, 0.00000, 1.00000],
    [0.00000, 0.24246, 0.75754], [0.00000, 0.94994, 0.05006], [0.00000, 0.99791, 0.00209],
    [0.00000, 1.00000, 0.00000], [0.44173, 0.48132, 0.07695], [1.00000, 0.00000, 0.00000],
    [1.00000, 0.00000, 0.00000], [1.00000, 0.00000, 0.00000], [1.00000, 0.00000, 0.00000]], F32)
_SPD2RGB = np.array([
    [0.007770, 0.048250, 0.009250, -0.101660, -0.205220, -0.157430, 0.152580, 0.469430, 0.489990, 0.230670,
     0.051220, 0.005150],
    [-0.005760, -0.046640, -0.031610, 0.079620, 0.253700, 0.385080, 0.295600, 0.109210, -0.015230, -0.020980,
     -0.003400, 0.000420],
    [0.039770, 0.356880, 0.491830, 0.201430, 0.019300, -0.036250, -0.038860, -0.023070, -0.008130, -0.002120,
     -0.000630, -0.000160]], F32)
_FIX = np.linalg.inv(_SPD2RGB @ _BASIS).astype(F32)      # removes the rounding of the tables above
NB = _BASIS.shape[0]
_RMIN = 0.02     # the darkest paint still reflects 2% (this sets how strongly darks tint lights)


def srgb_to_lin(c):
    c = np.asarray(c, F32)
    return np.where(c <= 0.04045, c / 12.92, ((np.maximum(c, 0) + 0.055) / 1.055) ** 2.4).astype(F32)


def lin_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055).astype(F32)


def rgb_to_q(rgb):
    """sRGB (...,3) in 0..1 -> paint state (...,NB). The state is sqrt(K/S): averaging two states
    is a pigment mix (the square root tames how fast a dark paint overpowers a light one)."""
    R = _RMIN + (1 - _RMIN) * np.clip(srgb_to_lin(rgb) @ _BASIS.T, 0, 1)
    return ((1 - R) / np.sqrt(2 * R)).astype(F32)


def q_to_lin(q):
    ks = q * q
    R = 1 + ks - np.sqrt(ks * ks + 2 * ks)
    return (((R - _RMIN) / (1 - _RMIN)) @ (_FIX @ _SPD2RGB).T).astype(F32)


def q_to_rgb(q):
    return lin_to_srgb(q_to_lin(q))


def _tcol(a, b, t):
    """A blend weight that broadcasts against colours: fields get a trailing axis."""
    t = np.asarray(t, F32)
    colour = a.shape[-1:] == (3,) or b.shape[-1:] == (3,)
    has_axis = (t.ndim >= 2 and t.shape[-1] in (1, 3)) or (t.shape == (3,) and max(a.ndim, b.ndim) == 1)
    if colour and t.ndim and not has_axis:
        t = t[..., None]
    return t


def mix(a, b, t):
    """Pigment mix of two sRGB colours (arrays broadcast): t=0 gives a, t=1 gives b."""
    a, b = np.asarray(a, F32), np.asarray(b, F32)
    t = _tcol(a, b, t)
    return q_to_rgb(rgb_to_q(a) * (1 - t) + rgb_to_q(b) * t)


def lerp(a, b, t):
    """Plain (light) blend of colours or fields; t broadcasts over a trailing colour axis."""
    a, b = np.asarray(a, F32), np.asarray(b, F32)
    t = _tcol(a, b, t)
    return a * (1 - t) + b * t


def rgb(h):
    """'#rrggbb' or an (r,g,b) triple in 0..1 -> float32 array."""
    if isinstance(h, str):
        h = PIGMENTS.get(h, h).lstrip('#')
        return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)], F32)
    return np.asarray(h, F32)


# Tube colours (masstone, approximate). Use them by name anywhere a colour is expected.
PIGMENTS = {
    'titanium white': '#f7f5ee', 'lead white': '#f1ecdd', 'naples yellow': '#f0d58c',
    'cadmium lemon': '#f5df2e', 'cadmium yellow': '#f6c21c', 'indian yellow': '#e8a317',
    'yellow ochre': '#c8963c', 'raw sienna': '#b0742e', 'cadmium orange': '#f0781e',
    'cadmium red': '#d8301f', 'vermilion': '#e0452a', 'alizarin crimson': '#8c1c2c',
    'quinacridone rose': '#c2265f', 'burnt sienna': '#8a4524', 'venetian red': '#a5432c',
    'burnt umber': '#4a2d1c', 'raw umber': '#5e4a30', 'ultramarine': '#1f2f93',
    'cobalt blue': '#2550a8', 'cerulean': '#3d8cc4', 'prussian blue': '#11284a',
    'phthalo blue': '#0f2f7a', 'cobalt turquoise': '#1f9c98', 'viridian': '#1c6a56',
    'sap green': '#4f6b25', 'chromium green': '#3f7438', 'terre verte': '#6c7a52',
    'dioxazine violet': '#3c2060', 'cobalt violet': '#8a4f9a', 'ivory black': '#1c1b1a',
    'paynes grey': '#2b3440',
}


def palette(recipe):
    """Mix tube colours by parts: palette({'ultramarine': 2, 'burnt sienna': 1, 'titanium white': 3})."""
    tot = float(sum(recipe.values()))
    q = sum(rgb_to_q(rgb(k)) * (v / tot) for k, v in recipe.items())
    return q_to_rgb(q)


# ----------------------------------------------------------------------------------------
# fields: noise, shapes, flow
# ----------------------------------------------------------------------------------------
def smoothstep(a, b, x):
    t = np.clip((np.asarray(x, F32) - a) / (b - a), 0, 1)
    return (t * t * (3 - 2 * t)).astype(F32)


def _resize(ch, W, H, resample=Image.BICUBIC):
    return np.asarray(Image.fromarray(np.ascontiguousarray(ch, F32), 'F').resize((W, H), resample), F32)


def blur(a, s):
    """Gaussian blur of a field or an image by s pixels (large s is done at reduced size)."""
    a = np.asarray(a, F32)
    if a.ndim == 3:
        return np.stack([blur(a[..., c], s) for c in range(a.shape[2])], -1)
    if s < 0.3:
        return a
    if s <= 5:
        return ndi.gaussian_filter(a, s, mode='nearest')
    k = int(s / 2.5)
    H, W = a.shape
    Hs, Ws = max(2, H // k), max(2, W // k)
    small = ndi.gaussian_filter(_resize(a, Ws, Hs, Image.BOX), s * Ws / W, mode='nearest')
    return _resize(small, W, H)


def value_noise(H, W, scale_px, rng, octaves=1, gain=0.5):
    """Zero-mean, unit-std smooth noise with features about scale_px across."""
    out = np.zeros((H, W), F32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        sc = max(scale_px / 2 ** o, 0.7)
        if sc < 3:
            n = ndi.gaussian_filter(rng.standard_normal((H, W)).astype(F32), sc * 0.5, mode='wrap')
        else:
            h, w = int(np.ceil(H / sc)) + 3, int(np.ceil(W / sc)) + 3
            g = ndi.gaussian_filter(rng.standard_normal((h, w)).astype(F32), 0.8, mode='wrap')
            n = _resize(g, max(int(w * sc), W), max(int(h * sc), H))[:H, :W]
        out += amp * (n - n.mean()) / (n.std() + 1e-8)
        tot += amp * amp
        amp *= gain
    return out / np.sqrt(tot)


def sd_circle(x, y, cx, cy, r):
    return np.hypot(x - cx, y - cy) - r


def sd_ellipse(x, y, cx, cy, rx, ry, angle=0.0):
    """Approximate signed distance to an ellipse (exact on the axes, good near the outline)."""
    c, s = np.cos(angle), np.sin(angle)
    u, v = (x - cx) * c + (y - cy) * s, -(x - cx) * s + (y - cy) * c
    k = np.hypot(u / rx, v / ry)
    g = np.hypot(u / rx ** 2, v / ry ** 2) + 1e-9
    return (k * (k - 1) / g).astype(F32)


def sd_box(x, y, cx, cy, hx, hy, angle=0.0, r=0.0):
    c, s = np.cos(angle), np.sin(angle)
    u, v = np.abs((x - cx) * c + (y - cy) * s) - hx + r, np.abs(-(x - cx) * s + (y - cy) * c) - hy + r
    return (np.hypot(np.maximum(u, 0), np.maximum(v, 0)) + np.minimum(np.maximum(u, v), 0) - r).astype(F32)


def sd_segment(x, y, ax, ay, bx, by, r0=0.0, r1=None):
    """Distance to a line segment with radius r0 at a and r1 at b (a tapered capsule)."""
    r1 = r0 if r1 is None else r1
    px, py, dx, dy = x - ax, y - ay, bx - ax, by - ay
    t = np.clip((px * dx + py * dy) / (dx * dx + dy * dy + 1e-12), 0, 1)
    return (np.hypot(px - t * dx, py - t * dy) - (r0 + (r1 - r0) * t)).astype(F32)


def sd_poly(x, y, pts):
    """Signed distance to a closed polygon [(x,y), ...]."""
    pts = np.asarray(pts, F32)
    d = np.full(np.shape(x), 1e9, F32)
    inside = np.zeros(np.shape(x), bool)
    n = len(pts)
    for i in range(n):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % n]
        d = np.minimum(d, sd_segment(x, y, ax, ay, bx, by))
        cond = (ay > y) != (by > y)
        xi = ax + (y - ay) * (bx - ax) / (by - ay + 1e-12)
        inside ^= cond & (x < xi)
    return np.where(inside, -d, d).astype(F32)


def sd_curve(x, y, pts, r0, r1=None, n=24):
    """Distance to a smooth curve through the points (Catmull-Rom), radius r0 -> r1 along it."""
    pts = np.asarray(pts, F32)
    r1 = r0 if r1 is None else r1
    P = np.concatenate([pts[:1], pts, pts[-1:]])
    out = np.full(np.shape(x), 1e9, F32)
    segs = []
    for i in range(len(pts) - 1):
        p0, p1, p2, p3 = P[i], P[i + 1], P[i + 2], P[i + 3]
        for t in np.linspace(0, 1, n, endpoint=False):
            segs.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                               + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    segs.append(pts[-1])
    m = len(segs) - 1
    for i in range(m):
        a, b = segs[i], segs[i + 1]
        out = np.minimum(out, sd_segment(x, y, a[0], a[1], b[0], b[1],
                                         r0 + (r1 - r0) * i / m, r0 + (r1 - r0) * (i + 1) / m))
    return out


def curve(pts, n=16):
    """Points along a smooth curve through pts (Catmull-Rom), for Painting.marks()."""
    pts = np.asarray(pts, F32)
    P = np.concatenate([pts[:1], pts, pts[-1:]])
    out = []
    for i in range(len(pts) - 1):
        p0, p1, p2, p3 = P[i], P[i + 1], P[i + 2], P[i + 3]
        for t in np.linspace(0, 1, n, endpoint=False):
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(pts[-1])
    return np.asarray(out, F32)


def smin(a, b, k):
    """Smooth union of two distance fields (k = blend radius)."""
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return (b + (a - b) * h - k * h * (1 - h)).astype(F32)


def fill(sd, soft=0.004):
    """Distance field -> mask: 1 inside, 0 outside, soft is the edge width."""
    return smoothstep(soft, -soft, sd)


def _with_fallback(ang, gx, gy, fallback):
    """Where a field is flat its gradient direction is noise: lean on `fallback` there."""
    if fallback is None:
        return ang.astype(F32)
    g = np.hypot(gx, gy)
    conf = smoothstep(0.03, 0.15, g / (np.percentile(g, 97) + 1e-12))
    fb = np.asarray(fallback, F32)
    a2 = np.where(np.cos(ang - fb) < 0, ang + np.pi, ang)              # same half-plane as the fallback
    return np.arctan2(conf * np.sin(a2) + (1 - conf) * np.sin(fb),
                      conf * np.cos(a2) + (1 - conf) * np.cos(fb)).astype(F32)


def along(field, smooth_px=0.0, fallback=None):
    """Brush direction that follows the contour lines of a field (across its gradient). Where the
    field is flat the direction is undefined; give `fallback` (an angle or a field) to use there."""
    f = blur(field, smooth_px) if smooth_px else np.asarray(field, F32)
    gy, gx = np.gradient(f)
    return _with_fallback(np.arctan2(gx, -gy), gx, gy, fallback)


def across(field, smooth_px=0.0, fallback=None):
    """Brush direction that runs down the gradient of a field (fall lines)."""
    f = blur(field, smooth_px) if smooth_px else np.asarray(field, F32)
    gy, gx = np.gradient(f)
    return _with_fallback(np.arctan2(gy, gx), gx, gy, fallback)


def around(x, y, cx, cy):
    """Brush direction circling a point."""
    return (np.arctan2(y - cy, x - cx) + np.pi / 2).astype(F32)


def image_flow(img, sigma_px=6.0):
    """Brush direction along the edges of an image (smoothed structure tensor)."""
    L = (np.asarray(img, F32)[..., :3] @ np.array([0.299, 0.587, 0.114], F32))
    L = blur(L, max(sigma_px * 0.3, 0.8))
    gy, gx = np.gradient(L)
    jxx, jyy, jxy = blur(gx * gx, sigma_px), blur(gy * gy, sigma_px), blur(gx * gy, sigma_px)
    return (0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2).astype(F32)


# ----------------------------------------------------------------------------------------
# forms: normals, light families, cast shadows, the ground plane
# ----------------------------------------------------------------------------------------
def light_dir(v):
    """Unit light direction from (x, y, z): x right, y down, z toward the viewer."""
    v = np.asarray(v, F32)
    return v / np.linalg.norm(v)


def ellipsoid_normal(x, y, cx, cy, rx, ry=None, angle=0.0):
    """Normal field (nx, ny, nz) of a ball or an ellipsoid seen face on. Valid inside its outline
    (sd_ellipse with the same arguments gives the outline)."""
    ry = rx if ry is None else ry
    c, s = np.cos(angle), np.sin(angle)
    u, v = ((x - cx) * c + (y - cy) * s) / rx, (-(x - cx) * s + (y - cy) * c) / ry
    k = np.hypot(u, v)
    sc = np.minimum(k, 0.999) / np.maximum(k, 1e-6)
    u, v = u * sc, v * sc
    nz = np.sqrt(np.maximum(1 - u * u - v * v, 0))
    return (u * c - v * s).astype(F32), (u * s + v * c).astype(F32), nz.astype(F32)


def lathe(x, y, cx, profile):
    """A turned form (jug, bottle, vase, bowl) standing upright on the axis x = cx.
    profile: [(y, radius), ...] from top to bottom. Returns (sd, normal, slope): the outline as a
    distance field, the normal field (nx, ny, nz), and dr/dy for flow_lathe()."""
    prof = np.asarray(profile, F32)
    ys = np.linspace(prof[0, 0], prof[-1, 0], 256)
    rs = ndi.gaussian_filter1d(np.interp(ys, prof[:, 0], prof[:, 1]), 128.0 / len(prof), mode='nearest')
    r = np.interp(y, ys, rs).astype(F32)
    dr = np.interp(y, ys, np.gradient(rs, ys)).astype(F32)
    u = np.clip((x - cx) / np.maximum(r, 1e-5), -0.999, 0.999)
    nz = np.sqrt(1 - u * u)
    n = np.stack([u, -dr * np.ones_like(u), nz])
    n /= np.linalg.norm(n, axis=0, keepdims=True)
    sd = np.maximum(np.abs(x - cx) - r, np.maximum(prof[0, 0] - y, y - prof[-1, 0]))
    return sd.astype(F32), (n[0].astype(F32), n[1].astype(F32), n[2].astype(F32)), dr


def shade(n, light, lit, half, core, refl=None, refl_from=(0.25, 0.9, 0.3), refl_amt=0.6, turn=0.0):
    """The light families of a form: `lit` on the side facing the light, `half` in the halftone,
    `core` in the shadow, and `refl` (light bounced back from the table or ground, arriving from
    `refl_from`) inside the shadow. turn > 0 brings the shadow further round the form.
    Returns (colour H x W x 3, lambert): lambert > 0 is the light side."""
    L = light_dir(light)
    lam = n[0] * L[0] + n[1] * L[1] + n[2] * L[2] - turn
    c = lerp(rgb(core), rgb(half), smoothstep(-0.02, 0.30, lam))
    c = lerp(c, rgb(lit), smoothstep(0.25, 0.80, lam))
    if refl is not None:
        b = light_dir(refl_from)
        up = np.clip(n[0] * b[0] + n[1] * b[1] + n[2] * b[2], 0, 1)
        c = lerp(c, rgb(refl), refl_amt * smoothstep(0.1, 0.9, up) * smoothstep(0.02, -0.35, lam))
    return c.astype(F32), lam.astype(F32)


def specular(n, light, power=40.0):
    """Highlight strength 0..1 of a glossy form (Blinn), for the white note laid on last."""
    L = light_dir(light)
    h = light_dir((L[0], L[1], L[2] + 1.0))
    return (np.clip(n[0] * h[0] + n[1] * h[1] + n[2] * h[2], 0, 1) ** power).astype(F32)


def cast_shadow(x, y, ax, ay, bx, by, r0, r1=None, squash=0.36, soft0=0.006, soft1=0.05):
    """A cast shadow lying on a table or the ground, 0..1: a tapered shape from the foot of the
    object (ax, ay) to where the shadow ends (bx, by), squashed into the ground plane, crisp at
    the object and soft at its far end."""
    r1 = r0 if r1 is None else r1
    yy, byy = ay + (y - ay) / squash, ay + (by - ay) / squash
    d = sd_segment(x, yy, ax, ay, bx, byy, r0, r1)
    far = np.clip(((x - ax) * (bx - ax) + (yy - ay) * (byy - ay)) / ((bx - ax) ** 2 + (byy - ay) ** 2 + 1e-12), 0, 1)
    soft = soft0 + (soft1 - soft0) * far
    return smoothstep(soft, -soft, d)


def flow_ellipse(x, y, cx, cy, rx, ry, angle=0.0, bow=1.0):
    """Brush direction for an egg, a lemon, a head: tip to tip along the long axis, bowing out
    with the form (no whorls, no target rings)."""
    c, s = np.cos(angle), np.sin(angle)
    u, v = ((x - cx) * c + (y - cy) * s) / rx, (-(x - cx) * s + (y - cy) * c) / ry
    dv = -2 * u * v / np.maximum(1 - np.clip(u, -0.95, 0.95) ** 2, 0.1) * bow
    return (angle + np.arctan2(dv * ry, rx)).astype(F32)


def flow_lathe(x, y, cx, profile, bow=0.8):
    """Brush direction down the profile of a turned form (see lathe())."""
    sd, n, dr = lathe(x, y, cx, profile)
    return np.arctan2(1.0, dr * n[0] * bow).astype(F32)


def ground_plane(x, y, horizon, cx=None, eye=0.022):
    """Coordinates on a level ground seen in perspective, for everything that must shrink toward
    the horizon. Returns (gu, gv, near): across, into the picture (large = far), and 0 at the
    horizon to 1 at the bottom edge."""
    cx = float(np.mean(x)) if cx is None else cx
    dy = np.maximum(y - horizon, 0)
    z = 1.0 / (dy + eye)
    return ((x - cx) * z).astype(F32), z.astype(F32), smoothstep(0.0, float(np.max(y)) - horizon, dy)


# ----------------------------------------------------------------------------------------
# a limited palette: every colour on the brush is a mix of a few tubes
# ----------------------------------------------------------------------------------------
def rgb_to_oklab(c):
    l = srgb_to_lin(c) @ np.array([[0.4122214708, 0.2119034982, 0.0883024619],
                                   [0.5363325363, 0.6806995451, 0.2817188376],
                                   [0.0514459929, 0.1073969566, 0.6299787005]], F32)
    return (np.cbrt(l) @ np.array([[0.2104542553, 1.9779984951, 0.0259040371],
                                   [0.7936177850, -2.4285922050, 0.7827717662],
                                   [-0.0040720468, 0.4505937099, -0.8086757660]], F32)).astype(F32)


class Gamut:
    """Everything a set of tube colours can mix: a table of mixes searched by perceptual distance."""

    def __init__(self, tubes):
        from scipy.spatial import cKDTree
        self.tubes = list(tubes)
        n = len(self.tubes)
        from itertools import combinations, product
        rows = [np.eye(n, dtype=F32)]
        for k, parts in ((2, (1, 2, 4, 8, 16, 32, 64, 128)), (3, (1, 4, 16, 64))):   # a painter mixes two or three tubes
            for idx in combinations(range(n), min(k, n)):
                for pr in product(parts, repeat=len(idx)):
                    if min(pr) != 1:
                        continue                          # the same recipe scaled up
                    w = np.zeros(n, F32)
                    w[list(idx)] = pr
                    rows.append((w / w.sum())[None])
        wts = np.concatenate(rows)
        self.q = (wts @ np.stack([rgb_to_q(rgb(t)) for t in self.tubes])).astype(F32)
        self.rgb = q_to_rgb(self.q)
        self.tree = cKDTree(rgb_to_oklab(self.rgb))

    def nearest(self, cols):
        return self.tree.query(rgb_to_oklab(np.asarray(cols, F32).reshape(-1, 3)))[1]

    def project(self, img):
        """The nearest mixable colour for every pixel or colour of img (sRGB)."""
        img = np.asarray(img, F32)
        return self.rgb[self.nearest(img)].reshape(img.shape)


def canvas_weave(H, W, period, rng):
    """Plain-weave canvas height in 0..1 (thread crowns are 1)."""
    y, x = np.mgrid[0:H, 0:W].astype(F32)
    x = x + 0.35 * period * value_noise(H, W, 25 * period / 5, rng)
    y = y + 0.35 * period * value_noise(H, W, 25 * period / 5, rng)
    fx, fy = x / period, y / period
    rx, ry = fx - np.floor(fx), fy - np.floor(fy)
    over = ((np.floor(fx) + np.floor(fy)) % 2) == 0
    warp = np.sin(np.pi * rx) ** 0.8 * (0.55 + 0.45 * np.sin(np.pi * ry))
    weft = np.sin(np.pi * ry) ** 0.8 * (0.55 + 0.45 * np.sin(np.pi * rx))
    h = np.where(over, warp, weft) * (1 + 0.2 * value_noise(H, W, 12 * period / 5, rng))
    h += 0.08 * value_noise(H, W, 0.8, rng)
    lo, hi = np.percentile(h, [1, 99])
    return np.clip((h - lo) / (hi - lo), 0, 1).astype(F32)


# ----------------------------------------------------------------------------------------
# the painting
# ----------------------------------------------------------------------------------------
class Painting:
    """A canvas. Lengths are fractions of the picture height; x runs 0..aspect, y runs 0 (top)..1.

    p = Painting(1500, 1000, seed=3, ground='burnt sienna')
    p.x, p.y          coordinate fields for building masks and colours
    p.noise(...)      smooth noise field
    p.strokes(...)    one pass of brush strokes
    p.mass(...)       block-in plus refining passes for one mass
    p.glaze(...)      a transparent film of colour
    p.dry(...)        let the surface set between stages
    p.save(path)      light the relief and write the picture
    p.film(path)      write the process film (pass film=True to the constructor)
    p.dabs / p.marks / p.cut_in / p.look / p.checkpoint: see references/atelier.md
    """

    def __init__(self, width, height, seed=1, ground=(0.62, 0.50, 0.36), weave=1.0, film=False, film_px=900,
                 log=True, palette=None, tape=False, knife=False):
        self.W, self.H = int(width), int(height)
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.px = self.H / 1000.0                       # pixels per "1000-px-high picture" pixel
        yy, xx = np.mgrid[0:self.H, 0:self.W].astype(F32)
        self.x, self.y = (xx + 0.5) / self.H, (yy + 0.5) / self.H
        self.aspect = self.W / self.H
        self.tooth = canvas_weave(self.H, self.W, 3.6 * self.px, self.rng)
        self.weave = weave
        self.q = np.empty((self.H, self.W, NB), F32)
        self.q[:] = rgb_to_q(rgb(ground))
        # a toned ground is scrubbed on, never flat
        self.q *= (1 + 0.05 * value_noise(self.H, self.W, 60 * self.px, self.rng, 3))[..., None]
        self.h = np.zeros((self.H, self.W), F32)          # paint relief, in pixels
        self.wet = np.zeros((self.H, self.W), F32)
        self._btile = self._bristle_tile(self.rng)
        self._film = None
        self._film_dir = None
        if isinstance(film, str):                         # frames on disk: survives checkpoint/resume
            os.makedirs(film, exist_ok=True)
            self._film_dir = film
        elif film:
            self._film = []
        self._film_px = film_px
        self._log = log
        self.n_strokes = 0
        self.gamut = Gamut(palette) if palette else None
        self.knife = bool(knife)                          # stroke passes use a palette knife unless told otherwise (marks and dabs keep their own tool)
        self.tape = [] if tape else None                  # raw stroke records, for replay elsewhere
        self._tape_full = tape == 'full'                  # keep full-size clip masks (for Painting.replay)
        self._ground = rgb(ground)

    # ---- helpers for the script -------------------------------------------------------
    def noise(self, scale=0.2, octaves=4, seed=None, gain=0.5):
        """Smooth noise in about -1..1 (unit std) with features `scale` of the height across."""
        rng = np.random.default_rng(self.seed * 7919 + seed) if seed is not None else self.rng
        return value_noise(self.H, self.W, scale * self.H, rng, octaves, gain)

    def blur(self, a, s):
        return blur(a, s * self.H)

    def view(self):
        """The flat colour on the canvas right now, sRGB (H,W,3), without relief lighting."""
        return q_to_rgb(self.q)

    def _field(self, v, channels=None):
        if channels:
            if isinstance(v, str) or np.ndim(v) == 1:
                return np.broadcast_to(rgb(v), (self.H, self.W, 3))
            return np.asarray(v, F32)
        if np.ndim(v) == 0:
            return np.full((self.H, self.W), v, F32)
        return np.asarray(v, F32)

    @staticmethod
    def _bristle_tile(rng, n=256):
        t = rng.standard_normal((n, n)).astype(F32)
        t = ndi.gaussian_filter(t, (9.0, 0.6), mode='wrap')        # long along the stroke, fine across
        t += 0.5 * ndi.gaussian_filter(rng.standard_normal((n, n)).astype(F32), (25.0, 2.5), mode='wrap') * 6
        return ((t - t.mean()) / t.std()).astype(F32)

    def _tile(self, a, b):
        return self._btile[(a * 256).astype(np.int64) & 255, (b * 256).astype(np.int64) & 255]

    # ---- one pass of strokes ----------------------------------------------------------
    def strokes(self, mask, color, flow=0.0, size=0.03, length=5.0, density=1.0, load=1.0, opacity=1.0,
                pickup=0.35, dry=0.1, jitter=0.05, hue=0.03, marble=0.5, wobble=0.2, overshoot=0.3,
                clip=0.0, ctol=0.14, refine=None, size_var=0.25, thickness=1.0, taper=0.45, bidir=True,
                soft=0.08, batches=3, knife=None, fade=False, name=None):
        """Lay one pass of strokes of one brush size.

        mask       where the mass is, H x W in 0..1
        color      what colour it should be: H x W x 3 sRGB, a triple, '#hex' or a pigment name
        flow       brush direction in radians (0 = along +x, pi/2 = down), a number or H x W
        size       brush width as a fraction of the picture height; an H x W field gives every
                   stroke the size under its start (strokes that shrink into the distance)
        length     stroke length in brush widths
        density    1 covers the mass about twice over; lower leaves the layer below showing
        load       how full the brush is; low values run dry sooner
        opacity    0..1, how much a full-bodied stroke covers
        pickup     0..1 wet-into-wet: how much wet paint under the stroke is dragged into it
        dry        0..1 dry-brush: paint catches only on the crowns of the canvas weave
        jitter     per-stroke value variation; hue: per-stroke hue variation (radians)
        marble     0..1 a second, imperfectly mixed colour on the brush
        wobble     how much strokes wander off the flow direction
        overshoot  how far strokes may run past the edge of the mask, in brush widths (loose edges)
        clip       0..1 cut the strokes at the mask outline (1 = crisp silhouette)
        ctol       a stroke stops when the target colour under it drifts this far from its own
        refine     tolerance: place strokes only where the canvas is further than this from `color`
        thickness  impasto relief of this pass
        soft       how far the sides of a stroke melt into what is under them, in brush widths
        batches    the pass is laid in this many successive sittings (each sees the ones before)
        knife      lay the pass with a palette knife: flat slabs, hard edges, ridges, skipped patches
        fade       for moving paintings (Painting.replay with another study): lay these strokes only
                   as strongly as their spot is lit in that study (lamps, stars, highlights)
        """
        H, W, rng = self.H, self.W, self.rng
        knife = self.knife if knife is None else knife
        mask = np.clip(self._field(mask), 0, 1)
        col = self._field(color, 3)
        size_f = None
        if np.ndim(size) == 2:
            size_f = np.maximum(np.asarray(size, F32) * H, 2.0)
            inside = size_f[mask > 0.5]
            w_px = float(np.percentile(inside, 5)) if inside.size else float(size_f.min())
        else:
            w_px = max(size * H, 2.0)
        fl = self._field(flow)
        fcos, fsin = np.cos(fl), np.sin(fl)

        # --- where: a jittered grid thinned by the mask (and by the error, when refining)
        wgt = mask
        if refine is not None:
            err = np.sqrt(((self.view() - col) ** 2).mean(-1))
            err = blur(err, max(0.35 * w_px, 0.8))
            wgt = mask * smoothstep(refine, 2.2 * refine, err)
        sp = max(w_px * 0.58 / np.sqrt(density), 1.0)
        ny, nx = int(np.ceil(H / sp)), int(np.ceil(W / sp))
        gy, gx = np.mgrid[0:ny, 0:nx]
        sx = ((gx + rng.random((ny, nx))) * sp).ravel()
        sy = ((gy + rng.random((ny, nx))) * sp).ravel()
        ok = (sx < W) & (sy < H)
        sx, sy = sx[ok], sy[ok]
        prob = wgt[sy.astype(int), sx.astype(int)] ** 0.8
        if size_f is not None:                            # bigger brushes need fewer strokes
            prob = prob * np.minimum(w_px / size_f[sy.astype(int), sx.astype(int)], 1.0) ** 2
        keep = rng.random(len(sx)) < prob
        sx, sy = sx[keep].astype(F32), sy[keep].astype(F32)
        N = len(sx)
        if N == 0:
            return 0
        w_at = size_f[sy.astype(int), sx.astype(int)] if size_f is not None else w_px
        wi = (w_at * np.exp(np.clip(rng.standard_normal(N), -2, 2) * size_var)).astype(F32)
        wi = np.maximum(wi, 1.6)
        Li = (length * wi * rng.uniform(0.55, 1.25, N)).astype(F32)

        # --- path: follow the flow from the seed
        step = max(1.0, 0.4 * w_px)
        K = int(np.ceil(Li.max() / step)) + 2
        pos = np.zeros((N, K, 2), F32)
        pos[:, 0, 0], pos[:, 0, 1] = sx, sy
        msoft = blur(mask, overshoot * float(np.median(wi))) if overshoot > 0 else mask
        ix, iy = sx.astype(int), sy.astype(int)
        dx, dy = fcos[iy, ix].copy(), fsin[iy, ix].copy()
        if bidir:
            sgn = np.where(rng.random(N) < 0.5, -1, 1).astype(F32)
            dx, dy = dx * sgn, dy * sgn
        curl = (rng.standard_normal(N) * wobble / np.maximum(Li, 1)).astype(F32) * 2.0
        scol = col[iy, ix].astype(F32)
        alive = np.ones(N, bool)
        stop = np.full(N, np.inf, F32)
        m0 = np.maximum(msoft[iy, ix], 0.05)              # a soft mask thins the strokes, it does not stop them
        for k in range(1, K):
            px, py = pos[:, k - 1, 0], pos[:, k - 1, 1]
            jx, jy = np.clip(px, 0, W - 1).astype(int), np.clip(py, 0, H - 1).astype(int)
            fx, fy = fcos[jy, jx], fsin[jy, jx]
            flip = np.where(fx * dx + fy * dy < 0, -1, 1)
            ndx, ndy = 0.55 * dx + 0.45 * fx * flip, 0.55 * dy + 0.45 * fy * flip
            ca, sa = np.cos(curl * step), np.sin(curl * step)
            ndx, ndy = ndx * ca - ndy * sa, ndx * sa + ndy * ca
            nrm = np.hypot(ndx, ndy) + 1e-9
            dx, dy = ndx / nrm, ndy / nrm
            pos[:, k, 0], pos[:, k, 1] = px + dx * step, py + dy * step
            jx = np.clip(pos[:, k, 0], 0, W - 1).astype(int)
            jy = np.clip(pos[:, k, 1], 0, H - 1).astype(int)
            out = (msoft[jy, jx] < 0.5 * np.minimum(m0, 1.0)) | (np.abs(col[jy, jx] - scol).max(-1) > ctol)
            out |= (pos[:, k, 0] < -wi) | (pos[:, k, 0] > W + wi) | (pos[:, k, 1] < -wi) | (pos[:, k, 1] > H + wi)
            newly = alive & out
            stop[newly] = (k - 1) * step                  # end on the last point that was still inside
            alive &= ~out
        early = stop < 1.1 * wi                           # ran into the edge at once: pull the other way
        k1 = np.minimum(np.ceil(1.1 * wi / step).astype(int), K - 1)
        back = 2 * pos[:, 0] - pos[np.arange(N), k1]      # where the stroke would end if pulled the other way
        bx_ = np.clip(back[:, 0], 0, W - 1).astype(int)
        by_ = np.clip(back[:, 1], 0, H - 1).astype(int)
        early &= msoft[by_, bx_] >= 0.5 * np.minimum(m0, 1.0)
        pos[early] = 2 * pos[early, 0:1] - pos[early]
        Li = np.maximum(np.minimum(Li, stop), np.minimum(wi * 1.1, Li)).astype(F32)

        k = dict(load=load, opacity=opacity, pickup=pickup, dry=dry, marble=marble, clip=clip, thickness=thickness,
                 taper=taper, soft=soft, knife=bool(knife), shape='knife' if knife else 'brush',
                 field=not isinstance(color, str) and np.ndim(color) == 3, jitter=jitter, hue=hue, fade=bool(fade))
        if size_f is None:
            self._deposit(pos, step, wi, Li, scol, mask, k, jitter, hue, batches)
        else:                                             # big brushes first, then each smaller octave
            octave = np.floor(np.log(wi / wi.min()) / np.log(1.7)).astype(int)
            for o in range(octave.max(), -1, -1):
                sel = np.flatnonzero(octave == o)
                if len(sel):
                    kk = int(np.ceil(Li[sel].max() / step)) + 2
                    self._deposit(pos[sel, :kk], step, wi[sel], Li[sel], scol[sel], mask, k, jitter, hue, batches)
        self.n_strokes += N
        if self._log:
            sz = f"{wi.min():.0f}-{wi.max():.0f}px" if size_f is not None else f"{w_px:6.1f}px"
            print(f"  {name or 'pass':<18s} brush {sz}  {N:6d} strokes", flush=True)
        return N

    def marks(self, lines, color, size=0.006, mask=None, load=1.2, opacity=1.0, pickup=0.15, dry=0.0, jitter=0.03,
              hue=0.02, marble=0.3, thickness=1.5, taper=0.3, soft=0.05, wobble=0.0, knife=False, shape=None,
              name=None, _at=None):
        """Deliberate single strokes: each line is a list of (x, y) points in picture units and is
        drawn once, in order (a mast, a branch, a signature). `color` is one colour for all, a list
        with one colour per line, or an H x W x 3 field sampled at each line's start. `size` may
        also be a list. Curved lines: pass more points (see curve())."""
        H, W, rng = self.H, self.W, self.rng
        N = len(lines)
        if N == 0:
            return 0
        sizes = np.broadcast_to(np.asarray(size, F32), (N,))
        wi = np.maximum(sizes * H, 1.6).astype(F32)
        step = max(1.0, 0.35 * float(wi.min()))
        res, Ls = [], []
        for ln in lines:
            P = np.asarray(ln, F32) * H
            if wobble:
                P = P + rng.standard_normal(P.shape).astype(F32) * wobble * H
            seg = np.hypot(*np.diff(P, axis=0).T)
            cum = np.concatenate([[0], np.cumsum(seg)])
            L = max(float(cum[-1]), 1.0)
            n = int(np.ceil(L / step)) + 2
            t = np.arange(n) * step
            te = np.minimum(t, L)
            pts = np.stack([np.interp(te, cum, P[:, 0]), np.interp(te, cum, P[:, 1])], -1)
            if len(P) > 1:                                    # run on straight past the end (padding)
                d = (P[-1] - P[-2]) / (np.hypot(*(P[-1] - P[-2])) + 1e-9)
                pts += (t - te)[:, None] * d[None]
            res.append(pts)
            Ls.append(L)
        K = max(len(r) for r in res)
        pos = np.zeros((N, K, 2), F32)
        for i, r in enumerate(res):
            pos[i, :len(r)] = r
            if len(r) < K:
                d = r[-1] - r[-2]
                pos[i, len(r):] = r[-1] + d[None] * np.arange(1, K - len(r) + 1)[:, None]
        Li = np.asarray(Ls, F32)
        if isinstance(color, (list, tuple)) and len(color) == N and (isinstance(color[0], str) or np.ndim(color[0]) == 1):
            scol, at = np.stack([rgb(c) for c in color]).astype(F32), None
        elif not isinstance(color, str) and np.ndim(color) == 3:
            at = pos[:, 0] if _at is None else np.asarray(_at, F32)       # where each mark reads its colour
            jx = np.clip(at[:, 0], 0, W - 1).astype(int)
            jy = np.clip(at[:, 1], 0, H - 1).astype(int)
            scol = np.asarray(color, F32)[jy, jx]
        else:
            at = None
            scol = np.broadcast_to(rgb(color), (N, 3)).astype(F32)
        m = np.ones((H, W), F32) if mask is None else np.clip(self._field(mask), 0, 1)
        self._deposit(pos, step, wi, Li, scol, m, dict(load=load, opacity=opacity, pickup=pickup, dry=dry,
                      marble=marble, clip=0.0 if mask is None else 1.0, thickness=thickness, taper=taper,
                      soft=soft, knife=bool(knife), shape=shape or ('knife' if knife else 'brush'),
                      field=at is not None), jitter, hue, batches=None, at=at)
        self.n_strokes += N
        if self._log:
            print(f"  {name or 'marks':<18s} brush {float(wi.mean()):6.1f}px  {N:6d} marks", flush=True)
        return N

    def dabs(self, points, color, size=0.01, stretch=1.0, angle=None, load=1.3, pickup=0.1, thickness=2.0,
             soft=0.12, jitter=0.04, hue=0.03, name=None, **kw):
        """Single touches of a loaded brush: a round dab at each (x, y) point. stretch > 1 makes
        petals and leaves, pointing along `angle` (radians; one value, one per point, or None for
        any direction). `color` and `size` as in marks(): one, one per point, or a study field."""
        pts = np.asarray(points, F32).reshape(-1, 2)
        n = len(pts)
        if n == 0:
            return 0
        sz = np.broadcast_to(np.asarray(size, F32), (n,))
        ang = self.rng.uniform(0, TAU, n) if angle is None else np.broadcast_to(np.asarray(angle, F32), (n,))
        half = 0.5 * sz * np.broadcast_to(np.asarray(stretch, F32), (n,))
        d = np.stack([np.cos(ang), np.sin(ang)], -1) * half[:, None]
        if not isinstance(color, str) and np.ndim(color) == 3:          # the study is read at each dab's centre
            kw['_at'] = pts * self.H
        return self.marks([[p_ - d_, p_ + d_] for p_, d_ in zip(pts, d)], color, size=list(sz), load=load,
                          pickup=pickup, thickness=thickness, soft=soft, jitter=jitter, hue=hue, taper=0.0,
                          shape='round', name=name or 'dabs', **kw)

    def cut_in(self, objects, color, width=0.02, size=0.012, **kw):
        """Paint the background back up to the objects: a band of strokes around `objects` (their
        mask), cut exactly at the outline. This is what gives a painted object a clean silhouette;
        do it after the objects are painted. `color` is the study."""
        obj = np.clip(self._field(objects), 0, 1)
        near = blur(obj, max(width * self.H, 1.0))
        band = smoothstep(0.02, 0.3, near) * (1 - obj)
        k = dict(length=4.0, clip=1.0, overshoot=0.0, pickup=0.3, soft=0.12, ctol=0.12, jitter=0.03, density=1.5,
                 name='cutting in')
        k.update(kw)
        return self.strokes(band, color, flow=along(near, fallback=0.0), size=size, **k)

    def ground_noise(self, horizon, scale_u=1.0, scale_v=1.0, seed=0, octaves=3, shear=0.0, cx=None):
        """Smooth noise lying on the ground plane, so its features shrink toward the horizon:
        furrows, cloud shadows, drifts of flowers, swell on water. scale_* are features per unit
        of depth, across and into the picture; shear slants them."""
        gu, gv, _ = ground_plane(self.x, self.y, horizon, cx)
        tex = self.noise(0.08, octaves, seed=seed)
        row = gv * scale_v * self.H * 0.02
        col = gu * scale_u * self.H * 0.02 + shear * row + 0.5 * self.W
        return ndi.map_coordinates(tex, [row % (2 * self.H), col % (2 * self.W)], order=1, mode='mirror').astype(F32)

    def _deposit(self, pos, step, wi, Li, scol, mask, k, jitter, hue, batches, at=None):
        """Lay traced strokes: batches=None keeps the given order (one sitting per stroke budget)."""
        rng = self.rng
        N = len(wi)
        # --- colour on the brush: the target at the seed, never quite the same twice
        c1 = _jitter(scol, rng, jitter, hue)
        c2 = _jitter(scol, rng, jitter * 2.2 + 0.02, hue * 2.0)
        if self.gamut is not None:                        # only what the tubes on the palette can mix
            i1, i2 = self.gamut.nearest(c1), self.gamut.nearest(c2)
            q1, q2 = self.gamut.q[i1], self.gamut.q[i2]
            c1, c2 = self.gamut.rgb[i1], self.gamut.rgb[i2]
        else:
            q1, q2 = rgb_to_q(c1), rgb_to_q(c2)
        runout = rng.uniform(0.35, 0.8, N).astype(F32)
        offs = rng.random((N, 4)).astype(F32)

        # --- lay the pass in sittings of at most a few million ribbon samples
        du, dvn = 0.75, 0.6
        nu_all = np.ceil(Li / du).astype(int) + 1
        nv = int(np.ceil(wi.max() / dvn)) + 1
        order = rng.permutation(N) if batches else np.arange(N)
        budget = 3_500_000
        groups, cur, cnt = [], [], 0
        # drawn marks are laid one at a time when there are few, so a later line's dry gaps show the
        # earlier line under it, not the canvas from before the pass
        per_sitting = int(np.ceil(N / max(batches, 1))) if batches else (1 if N <= 64 else int(np.ceil(N / 64)))
        for i in order:
            c = nu_all[i] * nv
            if cur and (cnt + c > budget or len(cur) >= per_sitting):
                groups.append(np.array(cur))
                cur, cnt = [], 0
            cur.append(i)
            cnt += c
        groups.append(np.array(cur))
        vn = np.linspace(-0.5, 0.5, nv).astype(F32)
        if self.tape is not None:
            lo = None
            if k['clip'] > 0:                             # the outline strokes are cut at, kept small
                st = max(1, self.H // 256)
                lo = (np.clip(mask[::st, ::st], 0, 1) * 255).astype(np.uint8)
            self.tape.append(dict(op='strokes', H=self.H, W=self.W, pos=pos.copy(), step=float(step), width=wi.copy(),
                                  length=Li.copy(), c1=c1.copy(), c2=c2.copy(), runout=runout, offs=offs,
                                  seed=np.asarray(scol, F32).copy(), at=None if at is None else np.asarray(at, F32).copy(),
                                  clip_full=mask if (self._tape_full and k['clip'] > 0) else None,
                                  sittings=[g.copy() for g in groups], params=dict(k), clip_mask=lo))
        for g in groups:
            self._lay(g, pos, step, wi, Li, vn, du, q1, q2, runout, offs, mask, k)
            self._frame()

    def _profile(self, u, L, w, taper, shape='brush'):
        """Half-width of the stroke footprint (in widths) at distance u from its start."""
        if shape == 'round':                                              # a dab: an ellipse
            return 0.5 * np.sqrt(np.clip(1 - (2 * u / L - 1) ** 2, 0, 1))
        if shape == 'knife':                                              # a slab: square ends
            return 0.5 * np.clip(u / (0.05 * w), 0, 1) ** 0.5 * np.clip((L - u) / (0.05 * w), 0, 1) ** 0.5
        cap = np.clip(u / (0.16 * w), 0, 1) ** 0.35                       # rounded heel
        tail = 1 - taper * smoothstep(0.55, 1.0, u / L) ** 1.5           # lifts off toward the end
        endcap = np.sqrt(np.clip((L - u) / (0.22 * w), 0, 1))
        return 0.5 * cap * tail * endcap

    def _path(self, sid, u, pos, step):
        """Position and unit normal of stroke sid at arc length u."""
        K = pos.shape[1]
        f = np.clip(u / step, 0, K - 1.001)
        i0 = f.astype(np.int64)
        t = (f - i0).astype(F32)[..., None]
        a, b = pos[sid, i0], pos[sid, i0 + 1]
        p = a + (b - a) * t
        d = b - a
        d /= (np.hypot(d[..., 0], d[..., 1])[..., None] + 1e-9)
        return p, np.stack([-d[..., 1], d[..., 0]], -1)

    def _lay(self, g, pos, step, wi, Li, vn, du, q1, q2, runout, offs, mask, k):
        load, opacity, pickup, dry, marble = k['load'], k['opacity'], k['pickup'], k['dry'], k['marble']
        clip, thickness, taper, soft = k['clip'], k['thickness'], k['taper'], k['soft']
        knife, shape = k.get('knife', False), k.get('shape', 'brush')
        H, W = self.H, self.W
        n, nv = len(g), len(vn)
        nu = int(np.ceil(Li[g].max() / du)) + 1
        u = (np.arange(nu, dtype=F32) * du)[None, :]                       # (1,nu)
        sid = g[:, None]
        w, L = wi[g][:, None], Li[g][:, None]
        p, nrm = self._path(np.broadcast_to(sid, (n, nu)), np.broadcast_to(u, (n, nu)), pos, step)
        prof = self._profile(u, L, w, taper, shape)                        # (n,nu)
        rag = 1 - 0.10 * np.abs(self._tile(u / (w * 1.2) + offs[g, 0:1], offs[g, 1:2] + 0 * u))
        half = (prof * rag)[..., None]
        valid = (u <= L)[..., None] & (np.abs(vn)[None, None, :] <= half)
        X = p[..., 0:1] + vn[None, None, :] * w[..., None] * nrm[..., 0:1]
        Y = p[..., 1:2] + vn[None, None, :] * w[..., None] * nrm[..., 1:2]
        ix, iy = np.rint(X).astype(np.int32), np.rint(Y).astype(np.int32)
        valid &= (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        flat = np.flatnonzero(valid)
        if len(flat) == 0:
            return
        zb = np.full(H * W, -1, np.int64)
        zb[(iy.ravel()[flat].astype(np.int64) * W + ix.ravel()[flat])] = flat   # later strokes win
        pp = np.flatnonzero(zb >= 0)
        idx = zb[pp]
        del zb, X, Y, ix, iy, valid, flat
        r, rem = np.divmod(idx, nu * nv)
        iu, iv = np.divmod(rem, nv)
        s = g[r]
        uu, vv = (iu * du).astype(F32), vn[iv]
        ws, Ls = wi[s], Li[s]
        un = uu / Ls
        bpx = 1.5 * max(self.px, 0.6)                                      # one bristle track
        ac = (vv + 0.5) * ws / (bpx * 256)
        al = uu / (bpx * 256 * 3.0)
        b1 = self._tile(al + offs[s, 0], ac + offs[s, 1])
        b2 = self._tile(al * 0.6 + offs[s, 2], ac + offs[s, 3])
        b3 = self._tile(al * 0.25 + offs[s, 1], ac * 0.45 + offs[s, 2])
        edge = self._profile(uu, Ls, ws, taper, shape) - np.abs(vv)        # distance inside the footprint
        py_, px_ = np.divmod(pp, W)
        tooth = self.tooth[py_, px_]
        if knife:
            # a blade lays an even slab and skips whole patches where it rides over the tooth
            amt = load * (1 - 0.45 * runout[s] * un * un) * (1 + 0.12 * b3)
            skip = smoothstep(0.35, 1.1, b2 * 0.8 + (1 - tooth) - 0.35) * (0.25 + 2.2 * dry + 0.6 * un)
            cov = smoothstep(-0.06, 0.06, amt - 0.3 - skip)
            a = _per(opacity, s) * cov * smoothstep(0.0, np.maximum(0.25 * soft, 1.0 / ws), edge)
            b1 = 0.3 * b3
        elif shape == 'round':
            amt = load * (1 + 0.25 * b1)
            cov = smoothstep(-0.18, 0.18, amt - 0.3 - (1 - tooth) * dry * 1.5)
            a = _per(opacity, s) * cov * smoothstep(0.0, np.maximum(soft, 1.2 / ws), edge)
        else:
            amt = load * (1 - runout[s] * un ** 1.5) * (1 + 0.42 * b1)
            need = (1 - tooth) * (dry + 0.35 * un * un) * 1.5
            cov = smoothstep(need - 0.18, need + 0.18, amt - 0.3)
            a = _per(opacity, s) * cov * smoothstep(0.0, np.maximum(soft, 1.2 / ws), edge)
        if clip > 0:
            a = a * (1 - clip + clip * mask[py_, px_])

        # wet-into-wet: the bristle has been dragging through the paint behind it
        streak = smoothstep(0.0, 0.25, b3) if knife else smoothstep(-0.3, 1.3, b3)   # a knife leaves hard bands
        qs = q1[s] + (q2[s] - q1[s]) * (marble * streak)[:, None]
        if pickup > 0:
            qc = np.zeros_like(qs)
            wc = np.zeros(len(pp), F32)
            for back, wt in ((0.10, 0.45), (0.28, 0.35), (0.6, 0.20)):
                pb, nb = self._path(s, np.maximum(uu - back * Ls - 1.0, 0), pos, step)
                bx = np.clip(np.rint(pb[:, 0] + vv * ws * nb[:, 0]), 0, W - 1).astype(np.int64)
                by = np.clip(np.rint(pb[:, 1] + vv * ws * nb[:, 1]), 0, H - 1).astype(np.int64)
                qc += wt * self.q[by, bx]
                wc += wt * self.wet[by, bx]
            m = pickup * wc * (1 - np.exp(-uu / (0.7 * ws))) * np.clip(0.75 + 0.6 * b2, 0.1, 1.6)
            m = np.clip(m, 0, 0.92)[:, None]
            qs = qs * (1 - m) + qc * m
        a3 = a[:, None]
        self.q[py_, px_] = self.q[py_, px_] * (1 - a3) + qs * a3
        body = _per(thickness, s) * self.px * (0.55 + 0.45 * np.sqrt(ws / (30 * self.px)))
        if knife:                                                         # a flat plateau, a lip where the blade lifts
            ridge = 1.3 * np.exp(-((edge - 0.02) / 0.03) ** 2) + 0.9 * smoothstep(0.86, 0.98, un)
            t = body * a * (1.0 * np.clip(amt, 0, 2) + 0.1 * b3 + ridge)
            flatten = 0.92
        elif shape == 'round':                                            # a blob, highest off-centre
            t = body * a * (0.4 + 1.1 * np.clip(edge / 0.5, 0, 1) ** 0.6 + 0.2 * b1 + 0.5 * smoothstep(0.5, 0.9, un))
            flatten = 0.8
        else:
            ridge = 0.45 * np.exp(-((edge - 0.05) / 0.06) ** 2) * (un > 0.03)        # paint squeezed to the sides
            t = body * a * (0.5 * np.clip(amt, 0, 2) + 0.22 * b1 + ridge + 0.35 * un * (un < 0.97))
            flatten = 0.65
        self.h[py_, px_] = self.h[py_, px_] * (1 - flatten * a) + np.maximum(t, 0)
        self.wet[py_, px_] = self.wet[py_, px_] * (1 - a) + a

    # ---- bigger gestures --------------------------------------------------------------
    def mass(self, mask, color, flow=0.0, sizes=(0.06, 0.03, 0.014), tol=0.06, name=None, **kw):
        """Paint one mass the way a painter does: cover it with the big brush, then come back with
        smaller ones only where it is still wrong. `sizes` are brush widths, largest first; `tol`
        is how close is close enough (smaller = tighter, more small strokes)."""
        total = 0
        for i, s in enumerate(sizes):
            k = dict(kw)
            if i > 0:
                k['refine'] = tol * (1.0 if i == 1 else 0.8)
                k.setdefault('length', 4.0)
                k['thickness'] = k.get('thickness', 1.0) * 0.85
            total += self.strokes(mask, color, flow, size=s, name=f"{name or 'mass'} {i + 1}/{len(sizes)}", **k)
        return total

    def glaze(self, mask, color, strength=0.3):
        """A transparent film: adds the glaze colour's absorption on top of what is there."""
        m = (np.clip(self._field(mask), 0, 1) * strength)[..., None]
        gq = rgb_to_q(self._field(color, 3))
        self.q = np.sqrt(self.q * self.q + m * gq * gq).astype(F32)
        if self.tape is not None:
            st = max(1, self.H // 256)
            self.tape.append(dict(op='glaze', mask=(m[::st, ::st, 0] * 255).astype(np.uint8),
                                  color=(self._field(color, 3)[::st, ::st] * 255).astype(np.uint8)))
        self._frame()

    def dry(self, amount=1.0):
        """Let the surface set: later strokes drag less of it (1 = touch dry)."""
        self.wet *= (1 - amount)
        if self.tape is not None:
            self.tape.append(dict(op='dry', amount=float(amount)))

    # ---- stopping and coming back ----------------------------------------------------
    def checkpoint(self, path):
        """Save the wet canvas so a later run can carry on from here (Painting.resume)."""
        meta = dict(W=self.W, H=self.H, seed=self.seed, weave=self.weave, n_strokes=self.n_strokes, knife=self.knife,
                    ground=[float(v) for v in self._ground], palette=self.gamut.tubes if self.gamut else None,
                    rng=self.rng.bit_generator.state)
        np.savez(path, q=self.q.astype(np.float16), h=self.h, wet=(self.wet * 255).astype(np.uint8),
                 meta=json.dumps(meta))
        return path

    @classmethod
    def resume(cls, path, film=False, log=True, tape=False):
        """A Painting exactly as checkpoint() left it."""
        d = np.load(path, allow_pickle=False)
        m = json.loads(str(d['meta']))
        p = cls(m['W'], m['H'], seed=m['seed'], ground=m['ground'], weave=m['weave'], film=film, log=log,
                palette=m['palette'], tape=tape, knife=m.get('knife', False))
        p.q, p.h, p.wet = d['q'].astype(F32), d['h'].astype(F32), d['wet'].astype(F32) / 255
        p.n_strokes = m['n_strokes']
        p.rng.bit_generator.state = m['rng']
        return p

    def replay(self, tape, study=None):
        """Lay a recorded tape again on this canvas (same size as when it was recorded). With
        `study`, every stroke that took its colour from a study takes it from this one instead,
        at the same spot and with the same personal variation: the same brushwork under another
        light. Strokes recorded with fade=True (lights, stars, lamps) are only laid as strongly as
        their spot is lit now. Replaying records nothing on this canvas's own tape."""
        lum = np.array([0.299, 0.587, 0.114], F32)
        for r in tape:
            if r['op'] == 'dry':
                self.wet *= (1 - r['amount'])
                continue
            if r['op'] == 'glaze':
                m = _resize(r['mask'].astype(F32) / 255, self.W, self.H)[..., None]
                c = np.stack([_resize(r['color'][..., i].astype(F32) / 255, self.W, self.H) for i in range(3)], -1)
                gq = rgb_to_q(c)
                self.q = np.sqrt(self.q * self.q + m * gq * gq).astype(F32)
                self._frame()
                continue
            if r['H'] != self.H or r['W'] != self.W:
                raise ValueError("a tape replays on a canvas of the size it was recorded on")
            k = dict(r['params'])
            c1, c2, pos = r['c1'], r['c2'], r['pos']
            if study is not None and k.get('field'):
                at = r['at'] if r.get('at') is not None else pos[:, 0]
                ix = np.clip(at[:, 0], 0, self.W - 1).astype(int)
                iy = np.clip(at[:, 1], 0, self.H - 1).astype(int)
                now = np.asarray(study, F32)[iy, ix]
                was = r['seed']
                g = np.clip((now @ lum + 0.08) / (was @ lum + 0.08), 0.3, 1.6)[:, None]
                c1 = np.clip(now + (c1 - was) * g, 0, 1).astype(F32)
                c2 = np.clip(now + (c2 - was) * g, 0, 1).astype(F32)
                if k.get('fade'):                         # a light is painted only while it is lit
                    here = smoothstep(0.35, 0.8, (now @ lum + 0.03) / (was @ lum + 0.03))
                    k['opacity'] = (k['opacity'] * here).astype(F32)
                    k['thickness'] = (k['thickness'] * here).astype(F32)
            if self.gamut is not None:
                q1, q2 = self.gamut.q[self.gamut.nearest(c1)], self.gamut.q[self.gamut.nearest(c2)]
            else:
                q1, q2 = rgb_to_q(c1), rgb_to_q(c2)
            mask = r.get('clip_full')
            if mask is None:
                mask = (_resize(r['clip_mask'].astype(F32) / 255, self.W, self.H, Image.BILINEAR)
                        if r.get('clip_mask') is not None else np.ones((self.H, self.W), F32))
            vn = np.linspace(-0.5, 0.5, int(np.ceil(r['width'].max() / 0.6)) + 1).astype(F32)
            for g_ in r['sittings']:
                self._lay(g_, pos, r['step'], r['width'], r['length'], vn, 0.75, q1, q2, r['runout'], r['offs'], mask, k)
                self._frame()
            self.n_strokes += len(r['width'])

    def look(self, path, study=None, width=1600):
        """Write a sheet for judging the canvas: the painting, and with `study` the study beside
        it, both again small and in grey (the squint test), and a map of where they still differ."""
        img = self.render()
        if study is None:
            sheet = img
        else:
            st = np.clip(np.asarray(study, F32), 0, 1)
            g = np.array([0.299, 0.587, 0.114], F32)
            err = blur(np.sqrt(((self.view() - st) ** 2).mean(-1)), 0.006 * self.H)
            e = np.clip(err / 0.25, 0, 1)[..., None]
            heat = (1 - e) * 0.12 + e * np.array([1.0, 0.25, 0.1], F32) + 0.25 * (img @ g)[..., None] * (1 - e)
            k = 3
            small = lambda a_: np.asarray(Image.fromarray((np.clip(a_, 0, 1) * 255).astype(np.uint8)).resize(
                (self.W // k, self.H // k), Image.LANCZOS), F32) / 255
            grey = lambda a_: np.repeat((a_ @ g)[..., None], 3, -1)
            top = np.concatenate([img, st], 1)
            row = np.concatenate([small(grey(img)), small(grey(st)), small(heat)], 1)
            row = np.asarray(Image.fromarray((row * 255).astype(np.uint8)).resize(
                (top.shape[1], round(row.shape[0] * top.shape[1] / row.shape[1])), Image.LANCZOS), F32) / 255
            sheet = np.concatenate([top, row], 0)
        im = Image.fromarray((np.clip(sheet, 0, 1) * 255 + 0.5).astype(np.uint8))
        if im.width > width:
            im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
        im.save(path, quality=90) if str(path).lower().endswith(('.jpg', '.jpeg')) else im.save(path)
        return path

    # ---- output -----------------------------------------------------------------------
    def render(self, relief=1.0, gloss=0.35, light=(-0.55, -0.62), scale=1.0):
        """Light the paint relief and return sRGB (H,W,3)."""
        lin = q_to_lin(self.q)
        hgt = blur(self.h, 0.7) + self.weave * 0.55 * self.px * self.tooth * np.exp(-self.h / (1.2 * self.px))
        gy, gx = np.gradient(hgt)
        k = 1.15 * relief
        nz = 1.0 / np.sqrt(1 + (k * gx) ** 2 + (k * gy) ** 2)
        nx, ny = -k * gx * nz, -k * gy * nz
        l = np.array([light[0], light[1], 0.62], F32)
        l /= np.linalg.norm(l)
        diff = (nx * l[0] + ny * l[1] + nz * l[2]) / l[2]
        occ = np.clip((hgt - blur(hgt, 3.0 * self.px)) / (1.5 * self.px), -1, 1)
        shade = np.clip(1 + 0.42 * (diff - 1) + 0.06 * occ * relief, 0.35, 1.9)
        hv = l + np.array([0, 0, 1], F32)
        hv /= np.linalg.norm(hv)
        nh = np.clip(nx * hv[0] + ny * hv[1] + nz * hv[2], 0, 1)
        spec = gloss * (nh ** 60 - hv[2] ** 60) * np.clip(self.h / (0.6 * self.px), 0, 1)
        out = lin * shade[..., None] + 0.5 * np.maximum(spec, -0.02)[..., None]
        return lin_to_srgb(out)

    def save(self, path, **kw):
        img = Image.fromarray((self.render(**kw) * 255 + 0.5).astype(np.uint8))
        if str(path).lower().endswith(('.jpg', '.jpeg')):
            img.save(path, quality=93, subsampling=0)
        else:
            img.save(path)
        if self._log:
            print(f"wrote {path}  {self.W}x{self.H}  {self.n_strokes} strokes", flush=True)
        return path

    def _frame(self):
        if self._film is None and self._film_dir is None:
            return
        st = max(1, int(round(max(self.W, self.H) / self._film_px)))
        sub = Painting.__new__(Painting)
        sub.q, sub.h, sub.tooth = self.q[::st, ::st], self.h[::st, ::st] / st, self.tooth[::st, ::st]
        sub.px, sub.weave = self.px / st, self.weave
        fr = (sub.render() * 255 + 0.5).astype(np.uint8)
        if self._film_dir:
            n = len(glob.glob(os.path.join(self._film_dir, '*.jpg')))
            Image.fromarray(fr).save(os.path.join(self._film_dir, f'{n:06d}.jpg'), quality=92)
        else:
            self._film.append(fr)

    def film(self, path, seconds=8.0, hold=2.0, fps=30):
        """Write the painting's progress, one frame per sitting, as an mp4 (or .gif)."""
        if self._film is None and self._film_dir is None:
            raise RuntimeError("construct the Painting with film=True (or film='folder') to record frames")
        self._frame()
        fr = self._film
        if self._film_dir:
            fr = [np.asarray(Image.open(f).convert('RGB')) for f in sorted(glob.glob(os.path.join(self._film_dir, '*.jpg')))]
        n = max(int(seconds * fps), 1)
        pick = [fr[min(int((i / n) ** 1.0 * len(fr)), len(fr) - 1)] for i in range(n)] + [fr[-1]] * int(hold * fps)
        h, w = fr[0].shape[:2]
        h2, w2 = h - h % 2, w - w % 2
        if str(path).lower().endswith('.gif'):
            ims = [Image.fromarray(f) for f in pick[::2]]
            ims[0].save(path, save_all=True, append_images=ims[1:], duration=int(2000 / fps), loop=0)
            return path
        cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w2}x{h2}',
               '-r', str(fps), '-i', '-', '-pix_fmt', 'yuv420p', '-crf', '18', '-movflags', '+faststart', str(path)]
        pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for f in pick:
            pr.stdin.write(np.ascontiguousarray(f[:h2, :w2]).tobytes())
        pr.stdin.close()
        pr.wait()
        if self._log:
            print(f"wrote {path}  {len(pick)} frames from {len(fr)} sittings", flush=True)
        return path


def _per(v, s):
    """A pass parameter that may be one number or one value per stroke."""
    return v[s] if isinstance(v, np.ndarray) else v


def _jitter(cols, rng, lum, hue):
    """Per-stroke variation: rotate the hue about the grey axis and scale the value."""
    M = len(cols)
    k = np.array([1, 1, 1], F32) / np.sqrt(3)
    ang = rng.standard_normal(M).astype(F32) * hue
    c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
    out = cols * c + np.cross(k[None], cols) * s + (cols @ k)[:, None] * k[None] * (1 - c)
    out = out * (1 + lum * rng.standard_normal((M, 1)).astype(F32))
    return np.clip(out, 0, 1).astype(F32)


def load_image(path, width=None, height=None):
    """Read an image as float sRGB, optionally resized and centre-cropped to width x height."""
    im = Image.open(path).convert('RGB')
    if width and height:
        k = max(width / im.width, height / im.height)
        im = im.resize((max(width, round(im.width * k)), max(height, round(im.height * k))), Image.LANCZOS)
        l, t = (im.width - width) // 2, (im.height - height) // 2
        im = im.crop((l, t, l + width, t + height))
    return np.asarray(im, F32) / 255.0
