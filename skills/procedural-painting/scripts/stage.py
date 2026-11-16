"""stage: set the subject up in 3D, then paint it.

A flat study has to be drawn by hand: perspective, the turn of a form into shadow, where a cast
shadow falls. Here the subject is a small scene of solid shapes (and posable figures), a camera
and a light. A sphere tracer turns it into what a painter needs: the colour study in light and
shadow families, a mask and a depth for every object, and a brush direction that follows each
form. paint() then lays it on an atelier canvas back to front.

    from atelier import *
    from stage import *
    sc = Scene(camera=Camera((0, 1.5, -4), (0, 1.0, 2), fov=38), key=(-0.6, 0.6, -0.3))
    sc.add('floor', plane((0, 0, 0), (0, 1, 0)), color='raw sienna', grain=(0, 0, 1))
    sc.add('jug', lathe((0.4, 0.8, 2), [(0, 0.06), (0.10, 0.10), (0.24, 0.05), (0.30, 0.06)]), color='cobalt blue', gloss=0.6)
    figure(sc, 'reader', at=(0, 0, 2), facing=90, pose='sit', top='#3a3f5a', bottom='#2a2a33')
    p = Painting(1200, 800)
    B = sc.render(p)             # B.study, B.masks['jug'], B.flow, B.depth, B.spec, B.group('reader'), B.project(point)
    paint(p, B, focus='reader')

Units are whatever you like (metres for rooms, centimetres for a table top): y is up, and with
the camera looking along +z, +x is to the right. numpy, scipy and pillow only.
"""
import json

import numpy as np
from scipy import ndimage as ndi
from PIL import Image

from atelier import F32, rgb, lerp, smoothstep, blur, along

__all__ = ['Scene', 'Camera', 'Buffers', 'sphere', 'ellipsoid', 'box', 'capsule', 'cylinder', 'cone', 'torus',
           'plane', 'lathe', 'figure', 'POSES', 'paint', 'masses', 'load_pathtracer_scene', 'planks', 'mottle']


# ----------------------------------------------------------------------------------------
# primitives
# ----------------------------------------------------------------------------------------
def _rot(r):
    """(rx, ry, rz) degrees, applied x then y then z; a list of ['axis', degrees] steps; or a 3x3."""
    if r is None:
        return np.eye(3, dtype=F32)
    r_ = np.asarray(r, dtype=object) if isinstance(r, (list, tuple)) and len(r) and isinstance(r[0], (list, tuple)) \
        and isinstance(r[0][0], str) else None
    if r_ is None:
        a = np.asarray(r, F32)
        if a.shape == (3, 3):
            return a
        steps = [('x', a[0]), ('y', a[1]), ('z', a[2])]
    else:
        steps = [(s[0], float(s[1])) for s in r]
    M = np.eye(3)
    for ax, deg in steps:
        c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        R = {'x': [[1, 0, 0], [0, c, -s], [0, s, c]], 'y': [[c, 0, s], [0, 1, 0], [-s, 0, c]],
             'z': [[c, -s, 0], [s, c, 0], [0, 0, 1]]}[ax]
        M = np.asarray(R) @ M
    return M.astype(F32)


class Prim:
    """One solid shape: a signed distance in its own frame, a bounding sphere, and a long axis."""

    def __init__(self, fn, center, R, radius, axis=None, lip=1.0):
        self.fn, self.c, self.R = fn, np.asarray(center, F32), np.asarray(R, F32)
        self.radius = radius                               # None = unbounded (a plane)
        self.axis = None if axis is None else (self.R @ np.asarray(axis, F32))
        self.lip = lip
        self.plain = bool(np.allclose(self.R, np.eye(3)))

    def sd(self, P):
        q = P - self.c
        return self.fn(q if self.plain else q @ self.R) * self.lip


def sphere(center, r):
    return Prim(lambda q: np.sqrt((q * q).sum(-1)) - r, center, np.eye(3), r, (0, 1, 0))


def ellipsoid(center, radii, rot=None):
    rr = np.asarray(radii, F32)

    def fn(q):
        k0 = np.sqrt(((q / rr) ** 2).sum(-1))
        k1 = np.sqrt(((q / (rr * rr)) ** 2).sum(-1)) + 1e-9
        return np.where(k0 < 1e-4, -rr.min(), k0 * (k0 - 1) / k1)
    ax = np.eye(3)[int(np.argmax(rr))] if rr.max() > 1.15 * np.sort(rr)[1] else (0, 1, 0)
    return Prim(fn, center, _rot(rot), float(rr.max()), ax, lip=0.85)


def box(center, half, rot=None, r=0.0):
    hh = np.asarray(half, F32)

    def fn(q):
        d = np.abs(q) - hh + r
        return np.sqrt((np.maximum(d, 0) ** 2).sum(-1)) + np.minimum(d.max(-1), 0) - r
    return Prim(fn, center, _rot(rot), float(np.linalg.norm(hh)), np.eye(3)[int(np.argmax(hh))])


def capsule(a, b, r0, r1=None):
    """A limb: from a to b, radius r0 at a and r1 at b."""
    a, b = np.asarray(a, F32), np.asarray(b, F32)
    r1 = r0 if r1 is None else r1
    ba = b - a
    L2 = float(ba @ ba) + 1e-12

    def fn(q):
        h = np.clip((q @ ba) / L2, 0, 1)
        return np.sqrt(((q - ba * h[:, None]) ** 2).sum(-1)) - (r0 + (r1 - r0) * h)
    P = Prim(fn, a, np.eye(3), float(np.sqrt(L2)) + max(r0, r1), ba / np.sqrt(L2),
             lip=1 / np.sqrt(1 + (r1 - r0) ** 2 / L2))
    return P


def _capped(dr, dy):
    return np.minimum(np.maximum(dr, dy), 0) + np.hypot(np.maximum(dr, 0), np.maximum(dy, 0))


def cone(base, height, r0, r1=0.0, rot=None):
    """Upright (local +y) from the centre of its base: radius r0 at the base, r1 at the top."""
    f = 1 / np.sqrt(1 + ((r1 - r0) / height) ** 2)

    def fn(q):
        rr = r0 + (r1 - r0) * np.clip(q[:, 1] / height, 0, 1)
        return _capped((np.hypot(q[:, 0], q[:, 2]) - rr) * f, np.abs(q[:, 1] - height / 2) - height / 2)
    return Prim(fn, base, _rot(rot), float(np.hypot(height, max(r0, r1))), (0, 1, 0))


def cylinder(base, height, r, rot=None):
    return cone(base, height, r, r, rot)


def torus(center, R, r, rot=None):
    """A ring lying flat (its axis is local +y)."""
    return Prim(lambda q: np.hypot(np.hypot(q[:, 0], q[:, 2]) - R, q[:, 1]) - r, center, _rot(rot), R + r, (1, 0, 0))


def plane(point, normal):
    n = np.asarray(normal, F32) / np.linalg.norm(normal)
    P = Prim(lambda q: q @ n, point, np.eye(3), None, None)
    P.normal = n
    return P


def lathe(base, profile, rot=None):
    """A turned form standing on `base`: profile = [(height, radius), ...] from the foot up."""
    pr = np.asarray(profile, F32)
    hs = np.linspace(pr[0, 0], pr[-1, 0], 160)
    rs = ndi.gaussian_filter1d(np.interp(hs, pr[:, 0], pr[:, 1]), 80.0 / len(pr) / 2.5, mode='nearest')
    slope = np.abs(np.gradient(rs, hs)).max()
    f = 1 / np.sqrt(1 + slope ** 2)
    h0, h1 = float(hs[0]), float(hs[-1])

    def fn(q):
        rr = np.interp(q[:, 1], hs, rs)
        return _capped((np.hypot(q[:, 0], q[:, 2]) - rr) * f, np.abs(q[:, 1] - (h0 + h1) / 2) - (h1 - h0) / 2)
    return Prim(fn, base, _rot(rot), float(np.hypot(h1, rs.max())), (0, 1, 0))


# ----------------------------------------------------------------------------------------
# objects, camera, scene
# ----------------------------------------------------------------------------------------
class Thing:
    """A named object: one or more shapes melted together, with one paint colour."""

    def __init__(self, name, parts, color, cut=None, k=0.0, kind='matte', gloss=0.0, grain=None, flow='along',
                 pattern=None, importance=1.0, shadow=True, f0=None, ior=1.45):
        self.name = name
        self.parts = list(parts) if isinstance(parts, (list, tuple)) else [parts]
        self.cut = [] if cut is None else (list(cut) if isinstance(cut, (list, tuple)) else [cut])
        self.color = rgb(color)
        self.k, self.kind, self.gloss, self.flow = k, kind, gloss, flow
        self.grain = None if grain is None else np.asarray(grain, F32) / np.linalg.norm(grain)
        self.pattern, self.importance, self.shadow = pattern, importance, shadow
        self.f0 = None if f0 is None else np.asarray(f0, F32)
        self.ior = ior                                     # glass only: 1 = thin-walled, shows what is behind unbent
        self.is_plane = any(p.radius is None for p in self.parts)
        if not self.is_plane:
            cs = np.stack([p.c for p in self.parts])
            c = cs.mean(0)
            self.bound = (c, float(max(np.linalg.norm(p.c - c) + p.radius for p in self.parts)))
        else:
            self.bound = None

    def sd(self, P, which=False):
        ds = [p.sd(P) for p in self.parts]
        d = ds[0]
        for e in ds[1:]:
            if self.k > 0:
                h = np.clip(0.5 + 0.5 * (e - d) / self.k, 0, 1)
                d = e + (d - e) * h - self.k * h * (1 - h)
            else:
                d = np.minimum(d, e)
        for c in self.cut:
            d = np.maximum(d, -c.sd(P))
        if which:
            return d, np.argmin(np.stack(ds), 0)
        return d


class Camera:
    def __init__(self, position, look_at, fov=40.0, up=(0, 1, 0)):
        """fov is the vertical field of view in degrees."""
        self.pos = np.asarray(position, F32)
        f = np.asarray(look_at, F32) - self.pos
        self.dist = float(np.linalg.norm(f))
        self.f = f / self.dist
        r = np.cross(np.asarray(up, F32), self.f)            # +x is to the right when looking along +z
        self.r = r / np.linalg.norm(r)
        self.u = np.cross(self.f, self.r)
        self.tan = float(np.tan(np.radians(fov) / 2))

    def rays(self, W, H):
        ys, xs = np.mgrid[0:H, 0:W].astype(F32)
        sx = ((xs + 0.5) / H - 0.5 * W / H) * 2 * self.tan
        sy = (0.5 - (ys + 0.5) / H) * 2 * self.tan
        D = self.f[None, None] + sx[..., None] * self.r + sy[..., None] * self.u
        D /= np.linalg.norm(D, axis=-1, keepdims=True)
        return D.reshape(-1, 3).astype(F32)

    def project(self, P, aspect):
        """World point(s) -> picture coordinates (x in 0..aspect, y in 0..1 from the top)."""
        v = np.asarray(P, F32) - self.pos
        z = np.maximum(v @ self.f, 1e-6)
        x = (v @ self.r) / z / (2 * self.tan) + 0.5 * aspect
        y = 0.5 - (v @ self.u) / z / (2 * self.tan)
        return np.stack([x, y], -1)


def planks(axis=(0, 0, 1), width=0.12, contrast=0.12, seed=1):
    """Floorboards running along `axis`: a pattern for Scene.add(..., pattern=planks())."""
    a = np.asarray(axis, F32) / np.linalg.norm(axis)
    side = np.cross(a, (0, 1, 0))
    rs = np.random.default_rng(seed).uniform(-1, 1, 512)

    def fn(P):
        u = P @ side / width
        i = np.floor(u).astype(int)
        joint = np.abs(u - i - 0.5) > 0.47
        tone = 1 + contrast * rs[i % 512] + 0.03 * np.sin((P @ a) * 9 / width * 0.1 + i * 2.1)
        return (tone * np.where(joint, 0.84, 1.0))[:, None]
    return fn


def mottle(scale=0.6, contrast=0.05):
    """Slow uneven tone, for plaster walls and cloths."""
    def fn(P):
        u = P / scale
        v = np.sin(u[:, 0] * 1.3 + 1.3 + 1.7 * np.sin(u[:, 1] * 0.9)) * np.sin(u[:, 1] * 1.1 + 0.4 + 1.3 * np.sin(u[:, 0] * 0.7 + u[:, 2])) + \
            0.6 * np.sin(u[:, 2] * 1.6 + u[:, 1] * 0.7 + 2.0 * np.sin(u[:, 0] * 0.5))
        return (1 + contrast * v)[:, None]
    return fn


class Scene:
    """Objects, a camera and a light.

    key        direction TO the light (x right, y up, z into the picture) for a sun or a window
    key_color  colour of that light; sky and bounce colour the shadows (from above and from below)
    shadow     how dark the shadow family is (0.2 dramatic, 0.5 airy)
    soft       softness of cast shadows (small = soft, 30 = crisp sun)
    backdrop   colour seen where a ray leaves the scene (through a window, past the table edge)
    fills      [(position, colour, strength, radius), ...]: unshadowed local lights (a lamp, a
               window's glow on the wall beside it)
    air        (colour, density per unit distance): haze
    """

    def __init__(self, camera, key=(-0.5, 0.7, -0.4), key_color=(1.0, 0.94, 0.80), sky=(0.55, 0.62, 0.78),
                 bounce=(0.62, 0.48, 0.36), shadow=0.36, soft=10.0, backdrop=(0.80, 0.86, 0.92), fills=(),
                 air=None, scale=None):
        self.cam = camera
        self.key = np.asarray(key, F32) / np.linalg.norm(key)
        self.key_color, self.sky, self.bounce = (np.asarray(v, F32) for v in (key_color, sky, bounce))
        self.shadow, self.soft = shadow, soft
        self.backdrop = backdrop
        self.fills = list(fills)
        self.air = air
        self.scale = scale or camera.dist * 0.25
        self.things = []

    def add(self, name, parts, color='#808080', **kw):
        t = Thing(name, parts, color, **kw)
        self.things.append(t)
        return t

    # ---- tracing ---------------------------------------------------------------------
    def _cull(self, things, O, D):
        """For every thing: which rays pass through its bounding sphere."""
        ms = []
        for t in things:
            if t.bound is None:
                ms.append(None)
                continue
            oc = t.bound[0][None] - O
            tca = (oc * D).sum(-1)
            ms.append((((oc * oc).sum(-1) - tca * tca) < (t.bound[1] * 1.05) ** 2) & (tca > -t.bound[1]))
        return ms

    def _sd(self, things, P, masks, idx):
        d = np.full(len(P), 1e9, F32)
        for t, m in zip(things, masks):
            if m is None:
                d = np.minimum(d, t.sd(P))
            else:
                sub = m[idx]
                if sub.any():
                    d[sub] = np.minimum(d[sub], t.sd(P[sub]))
        return d

    def _march(self, things, O, D, tmax, t0=0.0, steps=120):
        N = len(O)
        masks = self._cull(things, O, D)
        t = np.full(N, t0, F32)
        hit = np.zeros(N, bool)
        idx = np.arange(N)
        eps = 0.0012 * self.scale
        for _ in range(steps):
            if len(idx) == 0:
                break
            P = O[idx] + t[idx, None] * D[idx]
            d = self._sd(things, P, masks, idx)
            h = d < eps * (1 + t[idx] / self.scale * 0.5)
            hit[idx[h]] = True
            t[idx] += np.maximum(d, eps * 0.5) * 0.9
            idx = idx[~h & (t[idx] < tmax)]
        return t, hit, masks

    def _shadow(self, things, P, n, L, tmax):
        things = [t for t in things if t.shadow and t.kind not in ('glass', 'emit')]
        N = len(P)
        O = P + n * (0.006 * self.scale)
        D = np.broadcast_to(L, (N, 3)).astype(F32)
        masks = self._cull(things, O, D)
        t = np.full(N, 0.01 * self.scale, F32)
        res = np.ones(N, F32)
        idx = np.arange(N)
        for _ in range(40):
            if len(idx) == 0:
                break
            d = self._sd(things, O[idx] + t[idx, None] * D[idx], masks, idx)
            res[idx] = np.minimum(res[idx], self.soft * d / t[idx])
            t[idx] += np.maximum(d, 0.004 * self.scale)
            idx = idx[(res[idx] > 0.01) & (t[idx] < tmax)]
        return np.clip(res, 0, 1)

    def _shade(self, things, O, D, bounces=1):
        """Trace rays and return the painter's colour and what was hit."""
        N = len(O)
        tmax = 60 * self.scale
        t, hit, masks = self._march(things, O, D, tmax)
        col = np.broadcast_to(np.asarray(self.backdrop, F32), (N, 3)).copy()
        out = dict(t=np.where(hit, t, np.inf).astype(F32), id=np.full(N, -1, np.int32), n=np.zeros((N, 3), F32),
                   spec=np.zeros(N, F32), light=np.zeros(N, F32), tan=np.zeros((N, 3), F32))
        hi = np.flatnonzero(hit)
        if len(hi) == 0:
            out['col'] = col
            return out
        P = O[hi] + t[hi, None] * D[hi]
        # which thing
        best = np.full(len(hi), 1e9, F32)
        tid = np.zeros(len(hi), np.int32)
        for j, (th, m) in enumerate(zip(things, masks)):
            if m is None:
                d = th.sd(P)
            else:
                d = np.full(len(hi), 1e9, F32)
                sub = m[hi]
                if sub.any():
                    d[sub] = th.sd(P[sub])
            cl = d < best
            best[cl], tid[cl] = d[cl], j
        n = np.zeros((len(hi), 3), F32)
        base = np.zeros((len(hi), 3), F32)
        axis = np.zeros((len(hi), 3), F32)
        e = 0.002 * self.scale
        tet = np.array([[1, -1, -1], [-1, -1, 1], [-1, 1, -1], [1, 1, 1]], F32)
        for j, th in enumerate(things):
            s = np.flatnonzero(tid == j)
            if len(s) == 0:
                continue
            g = sum(tet[k][None] * th.sd(P[s] + tet[k] * e)[:, None] for k in range(4))
            n[s] = g / (np.linalg.norm(g, axis=-1, keepdims=True) + 1e-12)
            c = np.broadcast_to(th.color, (len(s), 3))
            base[s] = c * th.pattern(P[s]) if th.pattern else c
            if th.grain is not None:
                axis[s] = th.grain
            elif th.is_plane:
                nn = th.parts[0].normal
                axis[s] = (0, 0, 1) if abs(nn[1]) > 0.7 else (0, 1, 0)
            else:
                _, w = th.sd(P[s], which=True)
                ax = np.stack([p.axis if p.axis is not None else np.array([0, 1, 0], F32) for p in th.parts])
                axis[s] = ax[w]
        n = np.where(((n * D[hi]).sum(-1) > 0)[:, None], -n, n)                   # face the viewer
        L = self.key
        lam = n @ L
        sh = np.ones(len(hi), F32)
        lit_side = lam > 0
        if lit_side.any():
            sh[lit_side] = self._shadow(things, P[lit_side], n[lit_side], L, tmax)
        # contact occlusion
        ao = np.ones(len(hi), F32)
        for hh, w in ((0.03, 0.5), (0.09, 0.35), (0.22, 0.25)):
            h = hh * self.scale
            Q = P + n * h
            d = np.full(len(hi), 1e9, F32)
            for th in things:
                if th.kind == 'emit':
                    continue
                if th.bound is None:
                    d = np.minimum(d, th.sd(Q))
                else:
                    near = ((Q - th.bound[0]) ** 2).sum(-1) < (th.bound[1] + h) ** 2
                    if near.any():
                        d[near] = np.minimum(d[near], th.sd(Q[near]))
            ao -= w * np.clip((h - d) / h, 0, 1)
        ao = np.clip(ao, 0, 1)
        # a surface the light reaches is in the light family even at a raking angle: what a painter sees
        lamn = np.clip(np.abs(lam) / 0.72, 0, 1) ** 0.85 * np.sign(lam)
        l = np.where(lit_side, lamn * sh - (1 - sh) * 0.08, lamn)
        core = base * self.shadow
        c = lerp(core, base * 0.74, smoothstep(-0.05, 0.28, l))
        c = lerp(c, np.clip(base * self.key_color * 1.22 + 0.02, 0, 1), smoothstep(0.2, 0.8, l))
        dark = 1 - smoothstep(0.0, 0.35, l)
        up = n[:, 1] * 0.5 + 0.5
        c = c + (dark * up * 0.30)[:, None] * base * self.sky + (dark * (1 - up) * 0.42)[:, None] * base * self.bounce
        for fp, fc, fs, fr in self.fills:
            v = np.asarray(fp, F32)[None] - P
            dist = np.linalg.norm(v, axis=-1)
            c = c + (fs * np.clip((n * v).sum(-1) / (dist + 1e-9), 0, 1) / (1 + (dist / fr) ** 2))[:, None] * base * rgb(fc)
        c = c * (1 - (0.55 * (1 - ao) * (0.45 + 0.55 * dark)))[:, None]
        hv = L - D[hi]
        hv /= np.linalg.norm(hv, axis=-1, keepdims=True)
        gl = np.array([th.gloss for th in things], F32)[tid]
        sp = np.clip((n * hv).sum(-1), 0, 1) ** 48 * sh * gl * lit_side
        kinds = np.array([th.kind for th in things])[tid]
        em = kinds == 'emit'
        c[em] = base[em]
        me = np.flatnonzero(kinds == 'metal')
        if len(me) and bounces > 0:                                               # a mirror with a colour
            Dr = D[hi][me] - 2 * (D[hi][me] * n[me]).sum(-1, keepdims=True) * n[me]
            r = self._shade(things, P[me] + n[me] * 0.01 * self.scale, Dr.astype(F32), bounces - 1)
            f0 = np.stack([th.f0 if th.f0 is not None else th.color for th in things])[tid[me]]
            fres = (1 - np.clip(-(D[hi][me] * n[me]).sum(-1), 0, 1)) ** 5
            c[me] = r['col'] * (f0 + (1 - f0) * fres[:, None]) * 0.92 + 0.04 * base[me]
        if self.air is not None:
            a = 1 - np.exp(-t[hi] * self.air[1])
            c = lerp(c, rgb(self.air[0]), a)
        g = axis - (axis * n).sum(-1, keepdims=True) * n                           # the grain laid on the surface
        gn = np.linalg.norm(g, axis=-1, keepdims=True)
        alt = np.cross(n, np.array([1, 0, 0], F32))
        g = np.where(gn > 0.15, g / (gn + 1e-9), alt / (np.linalg.norm(alt, axis=-1, keepdims=True) + 1e-9))
        ar = np.array([th.flow == 'around' for th in things])[tid]
        g = np.where(ar[:, None], np.cross(n, g), g)
        col[hi] = np.clip(c, 0, 1)
        out['id'][hi], out['n'][hi], out['spec'][hi], out['light'][hi], out['tan'][hi] = tid, n, sp, np.clip(l, 0, 1), g
        out['col'] = col
        return out

    def render(self, size, res=560):
        """Trace the scene for a canvas. `size` is a Painting or (W, H); the tracing is done at
        `res` pixels high at most and the buffers are enlarged to the canvas."""
        W, H = (size.W, size.H) if hasattr(size, 'W') else size
        h = min(H, res)
        w = int(round(W * h / H))
        D = self.cam.rays(w, h)
        O = np.broadcast_to(self.cam.pos, D.shape).astype(F32)
        solid = [t for t in self.things if t.kind != 'glass']
        glass = [t for t in self.things if t.kind == 'glass']
        r = self._shade(solid, O, D)
        col, depth, ids, nrm, spec, light, tan = r['col'], r['t'], r['id'], r['n'], r['spec'], r['light'], r['tan']
        order = solid + glass
        if glass:
            tg, hg, mg = self._march(glass, O, D, 60 * self.scale)
            front = np.flatnonzero(hg & (tg < depth))
            if len(front):
                P = O[front] + tg[front, None] * D[front]
                best = np.full(len(front), 1e9, F32)
                gid = np.zeros(len(front), np.int32)
                for j, th in enumerate(glass):
                    d = th.sd(P)
                    cl = d < best
                    best[cl], gid[cl] = d[cl], j
                n = np.zeros((len(front), 3), F32)
                e = 0.002 * self.scale
                for k in range(3):
                    dv = np.zeros(3, F32)
                    dv[k] = e
                    for j, th in enumerate(glass):
                        s = gid == j
                        if s.any():
                            n[s, k] = th.sd(P[s] + dv) - th.sd(P[s] - dv)
                n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
                tint = np.stack([th.color for th in glass])[gid]
                facing = np.clip(-(n * D[front]).sum(-1), 0, 1)
                rim = (1 - facing) ** 2.2
                # what shows through: the scene behind, seen along the ray as the glass bends it
                cosi = np.clip(-(n * D[front]).sum(-1), 0, 1)
                eta = (1 / np.array([th.ior for th in glass], F32)[gid])[:, None]
                ci = cosi[:, None]
                kk = 1 - eta * eta * (1 - ci * ci)
                Dr = eta * D[front] + (eta * ci - np.sqrt(np.maximum(kk, 0))) * n
                Dr /= np.linalg.norm(Dr, axis=-1, keepdims=True)
                thru = self._shade(solid, (P + Dr * 0.002 * self.scale).astype(F32), Dr.astype(F32), bounces=0)['col']
                body = thru * (0.5 + 0.5 * tint) * (0.95 - 0.3 * rim[:, None]) + 0.06 * rim[:, None] * self.sky
                hv = self.key - D[front]
                hv /= np.linalg.norm(hv, axis=-1, keepdims=True)
                sp = np.clip((n * hv).sum(-1), 0, 1) ** 90
                col[front] = np.clip(body, 0, 1)
                spec[front] = np.maximum(spec[front], sp)
                ids[front] = len(solid) + gid
                depth[front] = tg[front]
                nrm[front] = n
                up = np.array([0, 1, 0], F32)
                g = up - (n @ up)[:, None] * n
                tan[front] = g / (np.linalg.norm(g, axis=-1, keepdims=True) + 1e-9)
        # brush direction: the surface grain seen on the picture plane
        aspect = W / H
        P = O + np.where(np.isfinite(depth), depth, 1.0)[:, None] * D
        a = self.cam.project(P, aspect)
        b = self.cam.project(P + tan * (0.02 * self.scale), aspect)
        ang = np.arctan2(b[:, 1] - a[:, 1], b[:, 0] - a[:, 0])
        ang = np.where(ids >= 0, ang, 0.0)

        def up_(a_, order_=Image.BICUBIC):
            a_ = np.asarray(a_, F32).reshape(h, w)
            if (h, w) == (H, W):
                return a_
            return np.asarray(Image.fromarray(a_, 'F').resize((W, H), order_), F32)
        B = Buffers()
        B.W, B.H, B.cam, B.scene = W, H, self.cam, self
        B.study = np.clip(np.stack([up_(col[:, k]) for k in range(3)], -1), 0, 1)
        dfar = np.where(np.isfinite(depth), depth, np.nan)
        far = np.nanmax(dfar) * 1.5 if np.isfinite(dfar).any() else 1.0
        B.depth = up_(np.where(np.isfinite(depth), depth, far), Image.BILINEAR)
        B.spec = np.clip(up_(spec, Image.BILINEAR), 0, 1)
        B.light = np.clip(up_(light, Image.BILINEAR), 0, 1)
        B.normal = np.stack([up_(nrm[:, k], Image.BILINEAR) for k in range(3)])
        B.flow = (0.5 * np.arctan2(up_(np.sin(2 * ang), Image.BILINEAR), up_(np.cos(2 * ang), Image.BILINEAR))).astype(F32)
        B.names = [t.name for t in order]
        B.things = {t.name: t for t in order}
        B.masks = {}
        idim = ids.reshape(h, w)
        for j, t in enumerate(order):
            m = (idim == j)
            if m.any():
                m = np.clip(up_(m.astype(F32), Image.BILINEAR), 0, 1)
                B.masks[t.name] = np.maximum(B.masks[t.name], m) if t.name in B.masks else m     # two objects, one name: one mask
        B.masks['backdrop'] = np.clip(up_((idim < 0).astype(F32), Image.BILINEAR), 0, 1)
        if B.masks['backdrop'].max() < 0.5:
            del B.masks['backdrop']
        return B


class Buffers:
    """What Scene.render() hands the painter.

    study   H x W x 3 colour study          masks   {object name: H x W mask}, plus 'backdrop'
    depth   distance from the camera        flow    brush direction following each form
    spec    highlight strength 0..1         light   how lit each pixel is, 0 (shadow) .. 1
    normal  (3, H, W)                       things  {name: Thing}
    """

    def group(self, prefix):
        """One mask for every object whose name is `prefix` or starts with `prefix.`"""
        m = np.zeros((self.H, self.W), F32)
        for k, v in self.masks.items():
            if k == prefix or k.startswith(prefix + '.'):
                m = np.maximum(m, v)
        return m

    def project(self, point):
        """A point in the scene -> (x, y) on the picture, for marks and dabs."""
        return tuple(float(v) for v in self.cam.project(point, self.W / self.H))

    def far_to_near(self):
        ds = [(float(np.median(self.depth[m > 0.5])) if (m > 0.5).any() else 0.0, k) for k, m in self.masks.items()]
        return [k for _, k in sorted(ds, reverse=True)]


# ----------------------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------------------
POSES = {
    # degrees. hip/shoulder: + swings the limb forward; knee/elbow: + bends it; abduct: + away from the body
    'stand': dict(),
    'contrapposto': dict(lean=2, hip_l=6, knee_l=10, shoulder_l=6, shoulder_r=-4, elbow_l=12, elbow_r=8, head_yaw=-18),
    'walk': dict(lean=4, hip_l=26, knee_l=8, hip_r=-20, knee_r=22, shoulder_l=-24, shoulder_r=26, elbow_l=22,
                 elbow_r=28, head_pitch=3),
    'walk2': dict(lean=4, hip_r=26, knee_r=8, hip_l=-20, knee_l=22, shoulder_r=-24, shoulder_l=26, elbow_r=22,
                  elbow_l=28, head_pitch=3),
    'sit': dict(sit=True, lean=4, hip_l=84, hip_r=84, knee_l=86, knee_r=86, shoulder_l=14, shoulder_r=14, elbow_l=50,
                elbow_r=50),
    'read': dict(sit=True, lean=12, hip_l=82, hip_r=86, knee_l=80, knee_r=90, shoulder_l=30, shoulder_r=34, elbow_l=98,
                 elbow_r=96, abduct_l=6, abduct_r=6, head_pitch=30),
    'reach': dict(lean=6, shoulder_r=128, elbow_r=10, abduct_r=10, shoulder_l=-8, elbow_l=14, hip_l=8, hip_r=-8,
                  knee_l=4, head_pitch=-14),
    'carry': dict(lean=-3, shoulder_l=24, shoulder_r=24, elbow_l=88, elbow_r=88, hip_l=14, knee_l=10, hip_r=-10,
                  knee_r=12),
}


def figure(scene, name, at=(0, 0, 0), facing=0.0, height=1.7, pose='stand', skin='#c99a7a', hair='#3a2a22',
           top='#4a5878', bottom='#2e2c34', shoes='#1e1a18', skirt=False, coat=False, hat=None, build=1.0,
           seat=None, importance=1.4, **joints):
    """A posable figure standing on `at` (the point between the feet; for a sitting pose, the point
    on the floor under the pelvis). facing: degrees about the vertical, 0 = looking along +z (away
    from a camera that looks along +z), 180 = toward it, 90 = toward +x. pose: a name from POSES or
    a dict of joint angles; keyword joints override it (hip_l, knee_r, shoulder_l, elbow_r,
    abduct_l, lean, head_yaw, head_pitch, ...). skirt, coat: clothing volumes; hat: a colour.
    seat: height of the seat for sitting poses (default 0.26 of the height).
    Adds the objects name.skin, name.hair, name.top, name.bottom, name.shoes (and name.hat) and
    returns the joint positions, e.g. j['hand_l'], j['head'], j['lap'] to put things in hands."""
    J = dict(lean=0, twist=0, head_yaw=0, head_pitch=0, sit=False)
    for s in 'lr':
        J.update({f'hip_{s}': 0, f'knee_{s}': 0, f'shoulder_{s}': 0, f'elbow_{s}': 6, f'abduct_{s}': 4, f'leg_out_{s}': 0})
    J.update(POSES[pose] if isinstance(pose, str) else pose)
    J.update(joints)
    Hh, b = height, build
    yaw = np.radians(facing)
    F = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]], F32)   # local -> world
    at = np.asarray(at, F32)

    def W_(v):
        return at + F @ np.asarray(v, F32)

    def rx(deg):                                             # + swings a hanging limb forward (+z)
        a = np.radians(deg)
        return np.array([[1, 0, 0], [0, np.cos(a), np.sin(a)], [0, -np.sin(a), np.cos(a)]], F32)

    def rz(deg):
        a = np.radians(deg)
        return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]], F32)

    def ry(deg):
        a = np.radians(deg)
        return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]], F32)
    down = np.array([0, -1, 0], F32)
    thigh, shin, hipw = 0.245 * Hh, 0.245 * Hh, 0.052 * Hh * b
    if J['sit']:
        pel_y = (seat if seat is not None else 0.26 * Hh) + 0.055 * Hh
    else:
        # stand on the lower foot
        reach = max(np.cos(np.radians(J[f'hip_{s}'])) * thigh + np.cos(np.radians(J[f'hip_{s}'] - J[f'knee_{s}'])) * shin
                    for s in 'lr')
        pel_y = reach + 0.045 * Hh
    pel = np.array([0, pel_y, 0], F32)
    T = rx(-J['lean'])                                       # the trunk's frame: + lean tips it forward
    Hm = T @ ry(J['head_yaw']) @ rx(-J['head_pitch'])        # the head's frame: + pitch bows it
    spine = T @ np.array([0, 1, 0], F32)
    chest = pel + spine * 0.30 * Hh
    neck = chest + spine * 0.055 * Hh
    hd = Hm @ np.array([0, 1, 0], F32)
    hfw = Hm @ np.array([0, 0, 1], F32)
    head = neck + spine * 0.035 * Hh + hd * 0.062 * Hh
    j = dict(pelvis=W_(pel), chest=W_(chest), neck=W_(neck), head=W_(head))
    skin_p, top_p, bot_p, shoe_p, hair_p = [], [], [], [], []
    # trunk
    top_p += [ellipsoid(W_(chest - spine * 0.035 * Hh), (0.122 * Hh * b, 0.105 * Hh, 0.066 * Hh * b), F @ T),
              capsule(W_(pel + spine * 0.04 * Hh), W_(chest - spine * 0.02 * Hh), 0.074 * Hh * b, 0.082 * Hh * b)]
    bot_p += [ellipsoid(W_(pel), (0.098 * Hh * b, 0.075 * Hh, 0.072 * Hh * b), F @ rx(-J['lean'] * 0.5))]
    skin_p += [capsule(W_(neck - spine * 0.02 * Hh), W_(neck + spine * 0.04 * Hh), 0.028 * Hh, 0.026 * Hh),
               ellipsoid(W_(head), (0.045 * Hh, 0.064 * Hh, 0.054 * Hh), F @ Hm)]
    skin_p += [ellipsoid(W_(head + hfw * 0.05 * Hh - hd * 0.012 * Hh), (0.011 * Hh, 0.017 * Hh, 0.014 * Hh), F @ Hm)]   # nose
    hair_p += [ellipsoid(W_(head + hd * 0.016 * Hh - hfw * 0.02 * Hh), (0.048 * Hh, 0.058 * Hh, 0.050 * Hh),
                         F @ Hm)]
    for s, sg in (('l', -1.0), ('r', 1.0)):
        sh = chest + T @ np.array([sg * 0.118 * Hh * b, 0.022 * Hh, 0], F32)
        ud = T @ rz(sg * J[f'abduct_{s}']) @ rx(J[f'shoulder_{s}']) @ down
        el = sh + ud * 0.165 * Hh
        fd = T @ rz(sg * J[f'abduct_{s}'] * 0.3) @ rx(J[f'shoulder_{s}'] + J[f'elbow_{s}']) @ down
        wr = el + fd * 0.145 * Hh
        hand = wr + fd * 0.045 * Hh
        top_p += [capsule(W_(sh), W_(el), 0.034 * Hh * b, 0.029 * Hh * b), capsule(W_(el), W_(wr), 0.028 * Hh * b, 0.021 * Hh * b)]
        skin_p += [capsule(W_(wr), W_(hand + fd * 0.012 * Hh), 0.022 * Hh, 0.019 * Hh)]
        j[f'shoulder_{s}'], j[f'elbow_{s}'], j[f'wrist_{s}'], j[f'hand_{s}'] = W_(sh), W_(el), W_(wr), W_(hand)
        hp = pel + np.array([sg * hipw, -0.02 * Hh, 0], F32)
        td = ry(sg * -J[f'leg_out_{s}']) @ rx(J[f'hip_{s}']) @ down
        kn = hp + td * thigh
        sd_ = ry(sg * -J[f'leg_out_{s}']) @ rx(J[f'hip_{s}'] - J[f'knee_{s}']) @ down
        an = kn + sd_ * shin
        an[1] = max(an[1], 0.035 * Hh)
        fwd = ry(sg * -J[f'leg_out_{s}']) @ np.array([0, 0, 1], F32)
        toe = an + fwd * 0.105 * Hh
        toe[1] = 0.022 * Hh if an[1] < 0.06 * Hh else an[1] - 0.03 * Hh
        bot_p += [capsule(W_(hp), W_(kn), 0.058 * Hh * b, 0.044 * Hh * b), capsule(W_(kn), W_(an), 0.042 * Hh * b, 0.027 * Hh * b)]
        shoe_p += [capsule(W_(an - fwd * 0.02 * Hh - np.array([0, 0.008 * Hh, 0], F32)), W_(toe), 0.03 * Hh, 0.024 * Hh)]
        j[f'hip_{s}'], j[f'knee_{s}'], j[f'ankle_{s}'], j[f'foot_{s}'] = W_(hp), W_(kn), W_(an), W_(toe)
    j['lap'] = 0.5 * (j['knee_l'] + j['knee_r']) * 0.6 + 0.4 * j['pelvis'] + np.array([0, 0.07 * Hh, 0], F32)
    if skirt:
        kn_mid = 0.5 * (j['knee_l'] + j['knee_r'])
        if J['sit']:
            hem = kn_mid.copy()
            hem[1] = at[1] + 0.12 * Hh
            fw = F @ np.array([0, 0, 1], F32)
            bot_p += [capsule(j['pelvis'], kn_mid + np.array([0, 0.008 * Hh, 0], F32), 0.098 * Hh * b, 0.088 * Hh * b),
                      capsule(kn_mid - fw * 0.01 * Hh, hem + fw * 0.03 * Hh, 0.082 * Hh * b, 0.10 * Hh * b)]
        else:
            hem = j['pelvis'].copy()
            hem[1] = at[1] + (0.10 if skirt is True else float(skirt)) * Hh
            bot_p += [capsule(j['pelvis'] + np.array([0, 0.03 * Hh, 0], F32), hem, 0.095 * Hh * b, 0.16 * Hh * b)]
    if coat:
        hem = j['pelvis'] * 0.35 + 0.65 * (0.5 * (j['knee_l'] + j['knee_r']))
        top_p += [capsule(j['chest'] - F @ spine * 0.05 * Hh, hem, 0.098 * Hh * b, 0.118 * Hh * b)]
    kw = dict(importance=importance)
    scene.add(f'{name}.bottom', bot_p, bottom, k=0.03 * Hh, **kw)
    scene.add(f'{name}.shoes', shoe_p, shoes, k=0.01 * Hh, **kw)
    scene.add(f'{name}.top', top_p, top, k=0.035 * Hh, **kw)
    scene.add(f'{name}.skin', skin_p, skin, k=0.012 * Hh, **kw)
    scene.add(f'{name}.hair', hair_p, hair, **kw)
    if hat:
        hw = F @ hd
        scene.add(f'{name}.hat', [cylinder(j['head'] + hw * 0.036 * Hh, 0.008 * Hh, 0.105 * Hh, F @ Hm),
                                  cylinder(j['head'] + hw * 0.036 * Hh, 0.06 * Hh, 0.052 * Hh, F @ Hm)],
                  hat, **kw)
    j['facing'] = F @ np.array([0, 0, 1], F32)
    j['head_dir'] = F @ hfw
    return j


# ----------------------------------------------------------------------------------------
# painting the buffers
# ----------------------------------------------------------------------------------------
def masses(p, B, focus=None, loose=1.0):
    """The plan paint() follows: one entry per object, far to near, with the brush sizes and
    tolerances it would use. Edit the list (or individual entries) and pass it back to paint()."""
    out = []
    for name in B.far_to_near():
        m = B.masks[name]
        area = float(m.mean())
        if area < 0.0004:
            continue
        th = B.things.get(name)
        is_bg = th is None or th.is_plane or area > 0.14
        in_focus = bool(focus) and (name == focus or name.startswith(focus + '.'))
        imp = (th.importance if th is not None else 1.0) * (1.5 if in_focus else 1.0)
        ext = np.sqrt(area)
        if is_bg:
            sizes = (0.06 * loose, 0.028 * loose, 0.012 * loose)
            k = dict(length=6, pickup=0.5, soft=0.16, overshoot=0.4, clip=0.0, tol=0.055, jitter=0.05, hue=0.035, ctol=0.12)
        else:
            s0 = float(np.clip(0.30 * ext * loose / imp, 0.006, 0.05))
            sizes = [s0, max(s0 * 0.45, 0.004), max(s0 * 0.2, 0.0028)]
            if in_focus:
                sizes.append(max(s0 * 0.1, 0.002))
            k = dict(length=4.5, pickup=0.3, soft=0.1, overshoot=0.0, clip=0.85, tol=0.04 if in_focus else 0.05,
                     jitter=0.035, hue=0.025, ctol=0.16)
        out.append(dict(name=name, mask=m, flow=B.flow, sizes=tuple(sizes), background=is_bg, focus=in_focus, **k))
    return out


def paint(p, B, focus=None, study=None, plan=None, loose=1.0, lay_in=True, cut_in=True, highlights=True, touches=True):
    """Paint the stage on an atelier canvas: a thin lay-in, every object far to near with its own
    brush direction and brushes sized by how big and how important it is, the background cut back
    in round the objects, then the highlights. focus: the object (or figure name) that gets the
    tightest finish. study: your own edited copy of B.study. plan: an edited masses() list."""
    st = B.study if study is None else study
    one = np.ones((p.H, p.W), F32)
    if lay_in:
        p.strokes(one, blur(st, 0.02 * p.H), flow=B.flow, size=0.085 * loose, length=4, load=0.8, opacity=0.9,
                  pickup=0.0, dry=0.35, thickness=0.3, jitter=0.06, name='lay-in')
    plan = masses(p, B, focus, loose) if plan is None else plan
    things = np.zeros((p.H, p.W), F32)
    for m in plan:
        k = {a: b for a, b in m.items() if a not in ('name', 'mask', 'flow', 'sizes', 'background', 'focus')}
        p.mass(m['mask'], st, flow=m['flow'], sizes=m['sizes'], name=m['name'][:14], thickness=0.8 if m['background'] else 1.0, **k)
        if not m['background']:
            things = np.maximum(things, m['mask'])
    if cut_in and things.max() > 0:
        p.cut_in(things, st, width=0.012, size=0.006)
    if touches:
        w = one if not focus else np.clip(0.35 + 0.65 * blur(B.group(focus), 0.03 * p.H) * 3, 0, 1)
        p.strokes(w, st, flow=B.flow, size=0.004, length=3.5, pickup=0.3, refine=0.07, ctol=0.14, thickness=1.2,
                  name='touches')
    if highlights and B.spec.max() > 0.3:
        p.dry(0.5)
        hm = smoothstep(0.35, 0.8, B.spec)
        p.strokes(hm, lerp(st, rgb('#fff6e0'), smoothstep(0.3, 0.9, B.spec) * 0.85), flow=B.flow, size=0.006, length=2.2,
                  pickup=0.05, load=1.35, thickness=2.8, jitter=0.02, overshoot=0.0, ctol=0.3, name='highlights')
    return plan


# ----------------------------------------------------------------------------------------
# scenes written for procedural-realism's path tracer
# ----------------------------------------------------------------------------------------
_TEX = {'walnut': '#5a3a22', 'oak': '#9a7448', 'marble': '#d8d4cc', 'linen': '#d9d2c2', 'plaster': '#8f8676',
        'pom': '#a3222a', 'apricot': '#e89a3c', 'crown': '#8a6a3a'}


def load_pathtracer_scene(path, fov_from='hfov', aspect=3.0, shadow=0.30):
    """A stage Scene from a procedural-realism path-tracer scene (JSON). Maps camera, the first
    light (as the key direction), planes, sphere, ellipsoid, cylinder, box, disk, tube and bowl
    (as solid glass with their fill as a coloured liquid) and scatter (as spheres). Not mapped:
    prism and gem (stood in for by a box and an ellipsoid of the same size), textures (one flat
    colour each), refraction, caustics and depth of field. `aspect` is the picture's width over
    height, needed to turn the horizontal field of view into a vertical one."""
    with open(path) as f:
        S = json.load(f)
    cam = S['camera']
    vfov = 2 * np.degrees(np.arctan(np.tan(np.radians(cam.get('hfov', 50)) / 2) / aspect))
    target = np.asarray(cam['target'], F32)
    L0 = S['lights'][0]
    if 'corner' in L0:
        lc = np.asarray(L0['corner'], F32) + 0.5 * np.asarray(L0['u'], F32) + 0.5 * np.asarray(L0['v'], F32)
    else:
        lc = np.asarray(L0['center'], F32)
    sc = Scene(Camera(cam['position'], target, fov=vfov, up=cam.get('up', (0, 1, 0))), key=lc - target,
               key_color=np.clip(np.asarray(L0.get('color', (1, 1, 1)), F32), 0, 1), shadow=shadow, soft=5.0,
               backdrop=np.clip(np.asarray(S.get('environment', (0.02, 0.02, 0.02)), F32) * 4 + 0.03, 0, 1),
               sky=(0.45, 0.5, 0.6), bounce=(0.5, 0.36, 0.26))
    mats = S.get('materials', {})

    def mat(m):
        m = mats[m] if isinstance(m, str) else m
        ty = m.get('type', 'diffuse')
        if ty == 'glass':
            sg = np.asarray(m.get('sigma', (0.02, 0.02, 0.02)), F32)
            if sg.max() < 0.3:
                return dict(color=np.exp(-sg * 6.0), kind='glass', gloss=1.0)
            return dict(color=np.clip(np.exp(-sg * 5.0) * 0.7 + 0.02, 0, 1), kind='matte', gloss=0.8)      # a liquid: its colour, glossy
        if ty == 'metal':
            return dict(color=m.get('f0', (0.8, 0.8, 0.8)), kind='metal', f0=m.get('f0'), gloss=1.0)
        c = rgb(_TEX.get(m.get('tex'), '#808080')) if 'tex' in m else np.asarray(m.get('color', (0.5, 0.5, 0.5)), F32)
        c = np.clip(c * m.get('tint', 1.0), 0, 1)
        return dict(color=c, kind='matte', gloss=0.7 if ty == 'coat' else 0.0)
    names = {}

    def nm(o, default):
        n = o.get('name', default).replace(' ', '_')
        names[n] = names.get(n, 0) + 1
        return n if names[n] == 1 else f'{n}_{names[n]}'
    for i, pl in enumerate(S.get('planes', [])):
        n = np.asarray(pl['normal'], F32)
        sc.add(nm(pl, f'plane{i}'), plane(pl['point'], n), grain=(1, 0, 0) if abs(n[1]) > 0.7 else (0, 1, 0),
               pattern=mottle(40.0, 0.06), **mat(pl['material']))
    unmapped = []
    last = None
    for i, o in enumerate(S.get('objects', [])):
        ty, rot = o['type'], o.get('rot')
        if ty == 'scatter':
            r = np.random.default_rng(o.get('seed', 0))
            parts = []
            for _ in range(o['count']):
                a, rad = r.uniform(0, 2 * np.pi), r.uniform(*o['ring'])
                rr = r.uniform(*o['radius'])
                sx, sz = o.get('scale', (1, 1))
                parts.append(sphere((o['center'][0] + np.cos(a) * rad * sx, rr, o['center'][1] + np.sin(a) * rad * sz), rr))
            sc.add(nm(o, 'scatter'), parts, **mat(o['material']))
            continue
        fill = None
        if ty == 'sphere':
            pr = sphere(o['center'], o['radius'])
        elif ty == 'ellipsoid':
            pr = ellipsoid(o['center'], o['radii'], rot)
        elif ty == 'cylinder':
            pr = cylinder(o['base'], o['height'], o['radius'], rot)
        elif ty == 'box':
            pr = box(o['center'], np.asarray(o['size'], F32) / 2, rot, r=0.05)
        elif ty == 'disk':
            pr = cylinder(o['center'], 0.05, o['radius'], rot)
        elif ty == 'tube':
            pr = cylinder(o['base'], o['height'], o['r_outer'], rot)
            if 'fill' in o:
                fill = (cylinder(np.asarray(o['base'], F32) + (0, o.get('bottom', 0.3), 0), o['fill']['level'] - o.get('bottom', 0.3),
                                 o['r_inner'] * 0.98, rot), o['fill']['material'])
        elif ty == 'bowl':
            c, ro = np.asarray(o['center'], F32), o['r_outer']
            pr = sphere(c, ro)
            if 'fill' in o:
                lv = o['fill']['level']
                fill = (ellipsoid(c + (0, (lv - ro) / 2, 0), (np.sqrt(max(ro * ro - lv * lv, 1e-3)) * 0.97, (ro + lv) / 2 * 0.97,
                                                              np.sqrt(max(ro * ro - lv * lv, 1e-3)) * 0.97)), o['fill']['material'])
        elif ty == 'prism':
            s_ = o['side']
            pr = box(np.asarray(o['center'], F32) + (0, s_ * 0.43, 0), (s_ / 2, s_ * 0.43, o['length'] / 2), rot)
            unmapped.append('prism (painted as a box)')
        elif ty == 'gem':
            pr = ellipsoid(o['center'], (o['radius'], o['radius'] * 0.6, o['radius']), None)
            unmapped.append('gem (painted as an ellipsoid)')
        else:
            unmapped.append(ty)
            continue
        m = mat(o['material'])
        mname = o['material'] if isinstance(o['material'], str) else ty
        # small unnamed parts of the same material join the object before them (a pomegranate's crown)
        if 'name' not in o and last is not None and last[1] == mname and m['kind'] != 'glass':
            last[0].parts.append(pr)
            cs = np.stack([q.c for q in last[0].parts])
            c_ = cs.mean(0)
            last[0].bound = (c_, float(max(np.linalg.norm(q.c - c_) + q.radius for q in last[0].parts)))
            continue
        if ty in ('tube', 'bowl') and m['kind'] == 'glass':
            m['ior'] = 1.0                                   # a vessel: thin walls
        th = sc.add(nm(o, f'{mname}'), pr, **m)
        last = (th, mname)
        if fill:
            fm = mat(fill[1])
            sc.add(th.name + '.fill', fill[0], color=fm['color'], kind='matte', gloss=0.3)
    sc.unmapped = sorted(set(unmapped))
    return sc
