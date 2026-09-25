"""
pathtracer.py - a Monte Carlo path tracer in pure numpy. No image model, no render engine.

Scenes are JSON files (see scenes/*.json and the schema in references). The script renders one
scene into a resumable accumulator: every run ADDS samples, so chain runs until the image is clean.

Light transport : unidirectional path tracing with next-event estimation (light sampling) and
                  multiple importance sampling against BSDF sampling on diffuse, clear-coat and
                  rough-metal surfaces; deterministic "specular next-event" on glass and mirrors
                  (the reflected / refracted ray is tested against the lights directly, which
                  removes most speckle from window reflections); optional caustic photon map for
                  sharp caustics; Russian roulette.
Materials       : diffuse (procedural textures) - clear coat over pigment (GGX lobe) - GGX metal
                  with coloured F0 - dielectric with exact Fresnel, Beer-Lambert absorption and an
                  optional outside medium (liquid touching glass, coloured core inside a marble)
Geometry        : sphere, ellipsoid, cylinder, tube (drinking glass, optional liquid fill), box,
                  disk (all but sphere take a rotation), infinite planes, parallelogram lights
Camera          : thin lens with depth of field, jittered anti-aliasing
Sampling        : uniform passes first, then adaptive passes that put samples where the estimated
                  display-space error is largest (Neyman allocation from per-pixel variance)
Output          : accumulator with radiance sum, per-pixel sample count, luminance variance and
                  first-hit albedo / normal / depth / specular mask (for denoising in post)

usage:
  python pathtracer.py --scene scenes/pomegranate.json --size 900x300 --spp 8 --tag draft
  python post_pathtrace.py --tag draft --out draft.png
  (legacy form still works: python pathtracer.py 900 300 8 draft  -> pomegranate scene)
"""
import os, sys, json, math, time, argparse, hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EPS = 1e-3

# ------------------------------------------------------------------ noise (fixed seed: textures are stable)
_r = np.random.default_rng(11)
PERM = np.concatenate([_r.permutation(4096)] * 2).astype(np.int64); RND = _r.random(4096)
def _h2(ix, iy): return RND[PERM[(PERM[ix & 4095] + iy) & 4095]]
def vnoise2(x, y):
    ix, iy = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64); fx, fy = x - ix, y - iy
    ux, uy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    a, b, c, d = _h2(ix, iy), _h2(ix + 1, iy), _h2(ix, iy + 1), _h2(ix + 1, iy + 1)
    return (a + (b - a) * ux) * (1 - uy) + (c + (d - c) * ux) * uy
def fbm2(x, y, o=5):
    s, a, n = 0.0, 0.5, 0.0
    for _ in range(o):
        s = s + a * vnoise2(x, y); n += a; x, y = x * 2.03 + 17.1, y * 2.01 - 7.3; a *= 0.5
    return s / n

def v3(x): return np.array(x, float)
def rgb(x, default=None):
    if x is None: return None if default is None else np.array(default, float)
    x = np.array(x, float)
    return np.full(3, float(x)) if x.ndim == 0 else x

def rot_matrix(deg):
    """local -> world rotation from [rx, ry, rz] degrees (applied x, then y, then z), or from a list of
    [axis, degrees] steps applied in order about the world axes, e.g. [["z", -40.75], ["y", 30]]."""
    if deg is not None and len(deg) and isinstance(deg[0], (list, tuple)):
        R = np.eye(3)
        for ax, a in deg:
            v = [0, 0, 0]; v["xyz".index(ax)] = a; R = rot_matrix(v) @ R
        return R
    rx, ry, rz = np.radians(deg if deg is not None else [0, 0, 0])
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

# ------------------------------------------------------------------ materials
DIFF, COAT, METAL, GLASS = 0, 1, 2, 3
KINDS = {"diffuse": DIFF, "coat": COAT, "metal": METAL, "glass": GLASS}
MIRROR_ALPHA = 2e-3            # GGX alpha below this is treated as a perfect mirror

class Material:
    def __init__(self, name, d):
        self.name = name; self.p = d
        self.kind = KINDS[d.get("type", "diffuse")]
        self.color = rgb(d.get("color"), [0.5, 0.5, 0.5])
        self.tex = d.get("tex")
        self.tint = rgb(d.get("tint", 1.0))
        self.ior = float(d.get("ior", 1.5))
        self.sigma = rgb(d.get("sigma"), [0, 0, 0])
        self.out_ior = float(d.get("outside_ior", 1.0))
        self.out_sigma = rgb(d.get("outside_sigma"), [0, 0, 0])
        self.f0 = rgb(d.get("f0"), [0.9, 0.9, 0.9])
        self.alpha = float(d.get("rough", 0.0)) if self.kind == METAL else float(d.get("coat_rough", 0.06))
        ci = float(d.get("ior", 1.5))
        self.coat_f0 = float(d.get("coat_f0", ((ci - 1) / (ci + 1)) ** 2))
        # dispersion from an Abbe number V (lower = more rainbow; crown glass 59, flint 30, diamond 55 at n=2.42)
        V = d.get("abbe"); self.disp = 0.0 if V is None else (self.ior - 1) / (float(V) * (1 / 0.4861 ** 2 - 1 / 0.6563 ** 2))

# ------------------------------------------------------------------ primitives
class Prim:
    """Every primitive intersects in its own local frame: q = (p - center) @ R."""
    kind = "?"
    def __init__(self, d, mat):
        self.mat = mat; self.d = d
        self.R = rot_matrix(d.get("rot")); self.rotated = d.get("rot") is not None
    def to_local(self, o, d): return (o - self.c) @ self.R, d @ self.R
    def to_world_dir(self, v): return v @ self.R.T
    def local_point(self, p): return (p - self.c) @ self.R

def _pick(t0, t1):
    return np.where(t0 > EPS, t0, np.where(t1 > EPS, t1, np.inf))

class Sphere(Prim):
    kind = "sphere"
    def __init__(self, d, mat):
        super().__init__(d, mat); self.c = v3(d["center"]); self.r = float(d["radius"])
        self.bc, self.br = self.c, self.r; self.ry = self.r
    def normal(self, p): return (p - self.c) / self.r

class Ellipsoid(Prim):
    kind = "ellipsoid"
    def __init__(self, d, mat):
        super().__init__(d, mat); self.c = v3(d["center"]); self.rad = v3(d["radii"])
        self.bc, self.br = self.c, float(self.rad.max()); self.ry = self.rad[1]
    def hit(self, o, d):
        ol, dl = self.to_local(o, d)
        oo = ol / self.rad; dd = dl / self.rad
        a = np.sum(dd * dd, 1); b = np.sum(oo * dd, 1); c = np.sum(oo * oo, 1) - 1
        disc = b * b - a * c; sq = np.sqrt(np.maximum(disc, 0))
        t = _pick((-b - sq) / a, (-b + sq) / a); t[disc < 0] = np.inf
        return t
    def normal(self, p):
        q = self.local_point(p); return self.to_world_dir(q / self.rad ** 2)

class Cylinder(Prim):
    """capped cylinder: base = centre of the bottom disk, local axis y, 0..height."""
    kind = "cylinder"
    def __init__(self, d, mat):
        super().__init__(d, mat); self.c = v3(d["base"]); self.r = float(d["radius"]); self.h = float(d["height"])
        self.bc = self.c + self.R @ np.array([0, self.h / 2, 0]); self.br = math.hypot(self.r, self.h / 2)
        self.ry = self.h / 2
    def hit(self, o, d):
        ol, dl = self.to_local(o, d); r, h = self.r, self.h
        ox, oz = ol[:, 0], ol[:, 2]
        a = dl[:, 0] ** 2 + dl[:, 2] ** 2; b = ox * dl[:, 0] + oz * dl[:, 2]; c = ox * ox + oz * oz - r * r
        disc = b * b - a * c; sq = np.sqrt(np.maximum(disc, 0)); a_ = np.maximum(a, 1e-12)
        best = np.full(len(o), np.inf)
        for tt in ((-b - sq) / a_, (-b + sq) / a_):
            yy = ol[:, 1] + tt * dl[:, 1]
            best = np.minimum(best, np.where((disc >= 0) & (tt > EPS) & (yy >= 0) & (yy <= h), tt, np.inf))
        dy = np.where(np.abs(dl[:, 1]) < 1e-9, 1e-9, dl[:, 1])
        for yc in (0.0, h):
            tt = (yc - ol[:, 1]) / dy; xx = ox + tt * dl[:, 0]; zz = oz + tt * dl[:, 2]
            best = np.minimum(best, np.where((tt > EPS) & (xx * xx + zz * zz <= r * r), tt, np.inf))
        return best
    def normal(self, p):
        q = self.local_point(p)
        n = np.stack([q[:, 0], np.zeros(len(q)), q[:, 2]], 1)
        n[np.abs(q[:, 1] - self.h) < 1e-3] = [0, 1, 0]; n[np.abs(q[:, 1]) < 1e-3] = [0, -1, 0]
        return self.to_world_dir(n)

class Tube(Prim):
    """drinking glass: outer radius, inner radius, solid base of thickness `bottom`, open top.
    Optional fill: {"level": h, "material": m} puts a liquid inside that touches the glass."""
    kind = "tube"
    def __init__(self, d, mat):
        super().__init__(d, mat); self.c = v3(d["base"]); self.ro = float(d["r_outer"]); self.ri = float(d["r_inner"])
        self.h = float(d["height"]); self.yb = float(d["bottom"])
        self.bc = self.c + self.R @ np.array([0, self.h / 2, 0]); self.br = math.hypot(self.ro, self.h / 2)
        self.ry = self.h / 2; self.fill_level = None
    def hit(self, o, d):
        ol, dl = self.to_local(o, d); ro, ri, yb, h = self.ro, self.ri, self.yb, self.h
        ox, oz = ol[:, 0], ol[:, 2]
        a = np.maximum(dl[:, 0] ** 2 + dl[:, 2] ** 2, 1e-12); b = ox * dl[:, 0] + oz * dl[:, 2]
        best = np.full(len(o), np.inf)
        for r, ylo in ((ro, 0.0), (ri, yb)):
            c = ox * ox + oz * oz - r * r; disc = b * b - a * c; sq = np.sqrt(np.maximum(disc, 0))
            for tt in ((-b - sq) / a, (-b + sq) / a):
                yy = ol[:, 1] + tt * dl[:, 1]
                best = np.minimum(best, np.where((disc >= 0) & (tt > EPS) & (yy >= ylo) & (yy <= h), tt, np.inf))
        dy = np.where(np.abs(dl[:, 1]) < 1e-9, 1e-9, dl[:, 1])
        for yc, rlo, rhi in ((h, ri, ro), (0.0, 0, ro), (yb, 0, ri)):
            tt = (yc - ol[:, 1]) / dy; xx = ox + tt * dl[:, 0]; zz = oz + tt * dl[:, 2]; rr = xx * xx + zz * zz
            best = np.minimum(best, np.where((tt > EPS) & (rr <= rhi * rhi) & (rr >= rlo * rlo), tt, np.inf))
        return best
    def _inner(self, q):
        rho = np.hypot(q[:, 0], q[:, 2]) + 1e-9
        wall = (np.abs(rho - self.ri) < np.abs(rho - self.ro)) & (q[:, 1] > self.yb + 2e-3)
        base = (np.abs(q[:, 1] - self.yb) < 2e-3) & (rho < self.ri)
        return wall | base, rho
    def normal(self, p):
        q = self.local_point(p); inner, rho = self._inner(q)
        radial = np.stack([q[:, 0] / rho, np.zeros(len(q)), q[:, 2] / rho], 1)
        nn = radial.copy()
        wall_in = np.abs(rho - self.ri) < np.abs(rho - self.ro); nn[wall_in] = -radial[wall_in]
        nn[np.abs(q[:, 1] - self.h) < 2e-3] = [0, 1, 0]
        nn[np.abs(q[:, 1]) < 2e-3] = [0, -1, 0]
        nn[(np.abs(q[:, 1] - self.yb) < 2e-3) & (rho < self.ri)] = [0, 1, 0]
        return self.to_world_dir(nn)
    def in_liquid(self, p):
        """surface points that touch the liquid fill (inner wall below the level, inner base)."""
        if self.fill_level is None: return np.zeros(len(p), bool)
        q = self.local_point(p); inner, _ = self._inner(q)
        return inner & (q[:, 1] <= self.fill_level + 1e-6)

class Bowl(Prim):
    """glass bowl: spherical shell (outer radius r_outer, inner r_inner) around `center`, cut open
    above local height `cut` (relative to the centre; a positive cut gives a balloon glass that narrows
    at the top). Optional fill: {"level": h (relative to centre), "material": m}."""
    kind = "bowl"
    def __init__(self, d, mat):
        super().__init__(d, mat); self.c = v3(d["center"]); self.ro = float(d["r_outer"]); self.ri = float(d["r_inner"])
        self.cut = float(d["cut"]); self.bc, self.br = self.c, self.ro; self.ry = self.ro; self.fill_level = None
        self.rim_lo = math.sqrt(max(self.ri ** 2 - self.cut ** 2, 0)); self.rim_hi = math.sqrt(max(self.ro ** 2 - self.cut ** 2, 0))
    def hit(self, o, d):
        ol, dl = self.to_local(o, d)
        b = np.sum(ol * dl, 1); oo = np.sum(ol * ol, 1)
        best = np.full(len(o), np.inf)
        for r in (self.ro, self.ri):
            disc = b * b - (oo - r * r); sq = np.sqrt(np.maximum(disc, 0))
            for tt in (-b - sq, -b + sq):
                yy = ol[:, 1] + tt * dl[:, 1]
                best = np.minimum(best, np.where((disc >= 0) & (tt > EPS) & (yy <= self.cut), tt, np.inf))
        dy = np.where(np.abs(dl[:, 1]) < 1e-9, 1e-9, dl[:, 1]); tt = (self.cut - ol[:, 1]) / dy
        rr = (ol[:, 0] + tt * dl[:, 0]) ** 2 + (ol[:, 2] + tt * dl[:, 2]) ** 2
        best = np.minimum(best, np.where((tt > EPS) & (rr >= self.rim_lo ** 2) & (rr <= self.rim_hi ** 2), tt, np.inf))
        return best
    def _inner(self, q):
        rho = np.linalg.norm(q, axis=1) + 1e-12
        return (np.abs(rho - self.ri) < np.abs(rho - self.ro)) & (q[:, 1] < self.cut - 2e-3), rho
    def normal(self, p):
        q = self.local_point(p); inner, rho = self._inner(q)
        n = q / rho[:, None]; n[inner] = -n[inner]
        n[np.abs(q[:, 1] - self.cut) < 2e-3] = [0, 1, 0]
        return self.to_world_dir(n)
    def in_liquid(self, p):
        if self.fill_level is None: return np.zeros(len(p), bool)
        q = self.local_point(p); inner, _ = self._inner(q)
        return inner & (q[:, 1] <= self.fill_level + 1e-6)

class Box(Prim):
    kind = "box"
    def __init__(self, d, mat):
        super().__init__(d, mat); self.c = v3(d["center"]); self.hs = v3(d["size"]) / 2
        self.bc, self.br = self.c, float(np.linalg.norm(self.hs)); self.ry = self.hs[1]
    def hit(self, o, d):
        ol, dl = self.to_local(o, d)
        inv = 1.0 / np.where(np.abs(dl) < 1e-12, 1e-12, dl)
        t1 = (-self.hs - ol) * inv; t2 = (self.hs - ol) * inv
        tn = np.minimum(t1, t2).max(1); tf = np.maximum(t1, t2).min(1)
        ok = tf >= np.maximum(tn, 0)
        t = np.where(tn > EPS, tn, np.where(tf > EPS, tf, np.inf)); t[~ok] = np.inf
        return t
    def normal(self, p):
        q = self.local_point(p) / self.hs; ax = np.abs(q).argmax(1)
        n = np.zeros_like(q); n[np.arange(len(q)), ax] = np.sign(q[np.arange(len(q)), ax])
        return self.to_world_dir(n)

class Convex(Prim):
    """convex polyhedron = intersection of half-spaces n.q <= h in the local frame.
    Built by the generators below: "prism" (triangular, lying on a face) and "gem" (round brilliant)."""
    kind = "convex"
    def __init__(self, d, mat, planes):
        super().__init__(d, mat); self.c = v3(d["center"])
        P = np.array(planes, float); self.Nn = P[:, :3] / np.linalg.norm(P[:, :3], axis=1, keepdims=True)
        self.Hh = P[:, 3] / np.linalg.norm(P[:, :3], axis=1)
        vs = []                                                  # vertices -> bounding sphere
        n = len(P)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    A = self.Nn[[i, j, k]]
                    if abs(np.linalg.det(A)) < 1e-9: continue
                    x = np.linalg.solve(A, self.Hh[[i, j, k]])
                    if np.all(self.Nn @ x <= self.Hh + 1e-7): vs.append(x)
        vs = np.array(vs); mid = (vs.max(0) + vs.min(0)) / 2
        self.bc = self.c + self.R @ mid; self.br = float(np.linalg.norm(vs - mid, axis=1).max()) * 1.001
        self.ry = float(vs[:, 1].max() - vs[:, 1].min()) / 2
    def hit(self, o, d):
        ol, dl = self.to_local(o, d)
        den = dl @ self.Nn.T; num = self.Hh[None] - ol @ self.Nn.T
        with np.errstate(divide="ignore", invalid="ignore"):
            tp = num / den
        enter = den < -1e-12; exit_ = den > 1e-12
        tn = np.where(enter, tp, -np.inf).max(1); tf = np.where(exit_, tp, np.inf).min(1)
        par_out = ((~enter & ~exit_) & (num < 0)).any(1)
        ok = (tn <= tf) & ~par_out
        t = np.where(tn > EPS, tn, np.where(tf > EPS, tf, np.inf)); t[~ok] = np.inf
        return t
    def normal(self, p):
        q = self.local_point(p); k = (q @ self.Nn.T - self.Hh[None]).argmax(1)
        return self.to_world_dir(self.Nn[k])

def gen_prism(d, mat):
    """triangular prism lying on one face: equilateral side `side`, length `length` along local z;
    `center` = centre of the bottom face."""
    s, L = float(d["side"]), float(d["length"]); c30, s30 = math.sqrt(3) / 2, 0.5
    planes = [[0, -1, 0, 0], [c30, s30, 0, c30 * s / 2], [-c30, s30, 0, c30 * s / 2], [0, 0, 1, L / 2], [0, 0, -1, L / 2]]
    return Convex(d, mat, planes)

def gen_gem(d, mat):
    """round-brilliant cut: girdle radius `radius`; standard proportions (table 57%, crown 34.5 deg,
    pavilion 40.75 deg). `center` is the girdle centre; rot [180,0,0] stands it on its table."""
    R = float(d["radius"]); tr = float(d.get("table", 0.57)) * R; ca = math.radians(float(d.get("crown_angle", 34.5)))
    pa = math.radians(float(d.get("pavilion_angle", 40.75))); g = float(d.get("girdle", 0.03)) * R
    planes = [[0, 1, 0, g / 2 + (R - tr) * math.tan(ca)]]                                     # table
    def facet(az, ang, rad, y, up=True):
        n = np.array([math.sin(ang) * math.cos(az), math.cos(ang) if up else -math.cos(ang), math.sin(ang) * math.sin(az)])
        p0 = np.array([rad * math.cos(az), y, rad * math.sin(az)])
        return list(n) + [float(n @ p0)]
    for k in range(8):
        a0, a1 = k * math.pi / 4, k * math.pi / 4 + math.pi / 8
        planes.append(facet(a0, ca, R, g / 2))                                                 # bezel
        planes.append(facet(a1, ca + math.radians(6), R, g / 2))                               # upper girdle
        planes.append(facet(a1, ca - math.radians(12), tr / math.cos(math.pi / 8), g / 2 + (R - tr) * math.tan(ca)))  # star
        planes.append(facet(a0, pa, R, -g / 2, up=False))                                     # pavilion main
        planes.append(facet(a1, pa + math.radians(1.5), R, -g / 2, up=False))                 # lower girdle
    for k in range(16):                                                                        # girdle
        a = k * math.pi / 8; planes.append([math.cos(a), 0, math.sin(a), R])
    return Convex(d, mat, planes)

class Disk(Prim):
    """flat disk, normal = local +y. Used for liquid surfaces; also usable as a coaster."""
    kind = "disk"
    def __init__(self, d, mat):
        super().__init__(d, mat); self.c = v3(d["center"]); self.r = float(d["radius"])
        self.bc, self.br = self.c, self.r; self.ry = self.r
    def hit(self, o, d):
        ol, dl = self.to_local(o, d)
        dy = np.where(np.abs(dl[:, 1]) < 1e-12, 1e-12, dl[:, 1]); t = -ol[:, 1] / dy
        xx = ol[:, 0] + t * dl[:, 0]; zz = ol[:, 2] + t * dl[:, 2]
        return np.where((t > EPS) & (xx * xx + zz * zz <= self.r ** 2), t, np.inf)
    def normal(self, p): return np.tile(self.R @ np.array([0, 1.0, 0]), (len(p), 1))

class Plane(Prim):
    kind = "plane"
    def __init__(self, d, mat):
        self.mat = mat; self.d = d; self.c = v3(d["point"]); self.n = v3(d["normal"]); self.n /= np.linalg.norm(self.n)
        a = np.abs(self.n)                                     # texture axes: (x,z) floor, (x,y) wall, (z,y) side wall
        if a[1] >= a[0] and a[1] >= a[2]: self.U, self.V = v3([1, 0, 0]), v3([0, 0, 1])
        elif a[2] >= a[0]: self.U, self.V = v3([1, 0, 0]), v3([0, 1, 0])
        else: self.U, self.V = v3([0, 0, 1]), v3([0, 1, 0])
        self.R = np.stack([self.U, self.V, self.n], 1); self.ry = 1.0
    def hit(self, o, d):
        dn = d @ self.n; dn = np.where(np.abs(dn) < 1e-9, 1e-9, dn)
        t = ((self.c - o) @ self.n) / dn
        return np.where(t > EPS, t, np.inf)
    def normal(self, p): return np.tile(self.n, (len(p), 1))
    def local_point(self, p): return np.stack([p @ self.U, p @ self.V, (p - self.c) @ self.n], 1)
    def to_world_dir(self, v): return v @ self.R.T

class RectLight:
    """parallelogram corner + s*u + t*v, emitting on the side of normalize(u x v)."""
    def __init__(self, d):
        if "aim" in d:                  # alternative: centre, a point to face, size [w, h]
            c = v3(d["center"]); f = v3(d["aim"]) - c; f /= np.linalg.norm(f)
            u = np.cross(f, [0, 1, 0]); u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-6 else v3([1, 0, 0])
            v = np.cross(u, f); w, h = d.get("size", [10, 10])
            # emit side is normalize(u x v) = -f ... flip so the light faces the aim point
            self.u, self.v = v * h, u * w
            if np.dot(np.cross(self.u, self.v), f) < 0: self.u, self.v = u * w, v * h
            self.corner = c - self.u / 2 - self.v / 2
        else:
            self.corner = v3(d["corner"]); self.u = v3(d["u"]); self.v = v3(d["v"])
        cr = np.cross(self.u, self.v); self.area = float(np.linalg.norm(cr)); self.n = cr / self.area
        self.emit = rgb(d.get("color", [1, 1, 1])) * float(d.get("intensity", 1.0))
        self.spot = float(d.get("spot", 0.0))          # emission * cos(angle from the normal)^spot: a pool of light
        # solve for (s, t) in the plane
        self.uu, self.vv, self.uv = self.u @ self.u, self.v @ self.v, self.u @ self.v
        self.det = self.uu * self.vv - self.uv ** 2
    def hit(self, o, d):
        dn = d @ self.n; dn = np.where(np.abs(dn) < 1e-9, 1e-9, dn)
        t = ((self.corner - o) @ self.n) / dn
        q = o + d * t[:, None] - self.corner
        qu, qv = q @ self.u, q @ self.v
        s = (qu * self.vv - qv * self.uv) / self.det; w = (qv * self.uu - qu * self.uv) / self.det
        return np.where((t > EPS) & (s >= 0) & (s <= 1) & (w >= 0) & (w <= 1), t, np.inf)

PRIMS = {"sphere": Sphere, "ellipsoid": Ellipsoid, "cylinder": Cylinder, "tube": Tube, "bowl": Bowl, "box": Box, "disk": Disk,
         "prism": gen_prism, "gem": gen_gem}

# ------------------------------------------------------------------ scene loading
def expand_objects(objs):
    """generators: {"type":"scatter", ...} places spheres resting on the table."""
    out = []
    for o in objs:
        if o.get("type") == "scatter":
            rs = np.random.default_rng(o.get("seed", 0)); cx, cz = o["center"]; sx, sz = o.get("scale", [1, 1])
            r0, r1 = o.get("ring", [0, 5]); a0, a1 = o["radius"]; y0 = o.get("y", 0.0)
            for _ in range(o["count"]):
                a = rs.uniform(0, 2 * np.pi); rr = rs.uniform(r0, r1); r_ = rs.uniform(a0, a1)
                out.append(dict(type="sphere", center=[cx + rr * math.cos(a) * sx, y0 + r_, cz + rr * math.sin(a) * sz],
                                radius=r_, material=o["material"]))
        else:
            out.append(o)
    return out

class Scene:
    def __init__(self, path):
        self.path = path
        with open(path) as f: self.text = f.read()
        S = json.loads(self.text); self.S = S
        self.mats = {}
        for k, v in S.get("materials", {}).items(): self.mats[k] = Material(k, v)
        def mat(m):
            if isinstance(m, str): return self.mats[m]
            nm = f"_inline{len(self.mats)}"; self.mats[nm] = Material(nm, m); return self.mats[nm]
        self.objs = []
        for d in expand_objects(S.get("objects", [])):
            if d["type"] not in PRIMS: raise ValueError(f"unknown primitive {d['type']}")
            ob = PRIMS[d["type"]](d, mat(d["material"])); self.objs.append(ob)
            if d["type"] in ("tube", "bowl") and "fill" in d:        # liquid touching the glass: a surface disk
                fl = d["fill"]; lvl = float(fl["level"]); ob.fill_level = lvl
                fm = mat(fl["material"]); ob.fill_mat = fm
                rad = ob.ri if d["type"] == "tube" else math.sqrt(max(ob.ri ** 2 - lvl ** 2, 0))
                disk = Disk(dict(center=list(ob.c + ob.R @ np.array([0, lvl, 0])), radius=rad - 1e-4, rot=d.get("rot")), fm)
                self.objs.append(disk)
        for d in S.get("planes", []): self.objs.append(Plane(d, mat(d["material"])))
        self.lights = [RectLight(d) for d in S["lights"]]
        pw = np.array([L.area * L.emit.mean() * 2 / (L.spot + 2) for L in self.lights]); self.lsel = pw / pw.sum()
        self.lcdf = np.cumsum(self.lsel)
        self.env = rgb(S.get("environment", [0, 0, 0]))
        R = S.get("render", {})
        self.max_depth = int(R.get("max_depth", 8)); self.clamp = float(R.get("clamp", 0) or 0)
        self.caustics = R.get("caustics")
        self.K = len(self.objs); self.LBASE = self.K
        # per-object material tables (fast lookup by object id)
        ms = [o.mat for o in self.objs]
        self.KIND = np.array([m.kind for m in ms]); self.IOR = np.array([m.ior for m in ms])
        self.SIG = np.array([m.sigma for m in ms]); self.OIOR = np.array([m.out_ior for m in ms])
        self.OSIG = np.array([m.out_sigma for m in ms]); self.F0 = np.array([m.f0 for m in ms])
        self.ALPHA = np.array([m.alpha for m in ms]); self.CF0 = np.array([m.coat_f0 for m in ms])
        self.DISP = np.array([m.disp for m in ms]); self.any_disp = bool((self.DISP > 0).any())
        self.SPECULAR = ((self.KIND == GLASS) | ((self.KIND == METAL) & (self.ALPHA < MIRROR_ALPHA)))
        self.SPCMASK = (self.KIND == GLASS) | (self.KIND == METAL)
        # surfaces photons bounce off (specular + glossy metal); camera paths leave those caustics to the photon map
        ga = float((self.caustics or {}).get("glossy_alpha", 0.2))
        self.PMSURF = self.SPECULAR | ((self.KIND == METAL) & (self.ALPHA <= ga))
        # bounded objects (everything but planes), grouped into clusters of bounding spheres
        bidx = [i for i, o in enumerate(self.objs) if o.kind != "plane"]
        self.groups = [Group(self, g) for g in cluster([(self.objs[i].bc, self.objs[i].br) for i in bidx], bidx)]
        self.GC = np.array([g.c for g in self.groups]).reshape(-1, 3); self.GR = np.array([g.r for g in self.groups])
        self.nbound = len(bidx)
        self.pidx = [i for i, o in enumerate(self.objs) if o.kind == "plane"]
        self.tubes_fill = [i for i, o in enumerate(self.objs) if o.kind in ("tube", "bowl") and o.fill_level is not None]
        c = S["camera"]
        self.cam = v3(c["position"]); tgt = v3(c["target"])
        fwd = tgt - self.cam; dist = np.linalg.norm(fwd); self.fwd = fwd / dist
        upv = v3(c.get("up", [0, 1, 0]))
        self.right = np.cross(upv, self.fwd); self.right /= np.linalg.norm(self.right); self.up = np.cross(self.fwd, self.right)
        self.hfov = math.radians(float(c.get("hfov", 50))); self.aperture = float(c.get("aperture", 0.0))
        f = c.get("focus", "target")
        self.focus = dist if f == "target" else (float(np.dot(v3(f) - self.cam, self.fwd)) if isinstance(f, list) else float(f))
        self.post = S.get("post", {})
        self.hash = hashlib.sha1(self.text.encode()).hexdigest()[:12]

# ------------------------------------------------------------------ intersection
def sphere_test(o, d, C, R):
    """batched ray / sphere test via matmul (no N x K x 3 temporaries): returns t0, t1, miss."""
    b = (np.sum(o * d, 1)[:, None] - d @ C.T)
    c = np.sum(o * o, 1)[:, None] - 2 * (o @ C.T) + np.sum(C ** 2, 1)[None] - R[None] ** 2
    disc = b * b - c; del c
    sq = np.sqrt(np.maximum(disc, 0)); t0 = -b - sq; t1 = -b + sq
    return t0, t1, (disc < 0) | (t1 <= EPS)

def cluster(bounds, ids):
    """greedy agglomeration of bounding spheres: merge the pair whose union grows least, while the
    union stays under 1.8x the larger member and the two are within 6x in size (a huge object must not
    swallow small ones, or every ray would test them all)."""
    cl = [([i], np.array(c, float), float(r)) for (c, r), i in zip(bounds, ids)]
    def union(a, b):
        (ca, ra), (cb, rb) = (a[1], a[2]), (b[1], b[2]); dd = np.linalg.norm(cb - ca)
        if dd + rb <= ra: return ca, ra
        if dd + ra <= rb: return cb, rb
        r = (dd + ra + rb) / 2; return ca + (cb - ca) * ((r - ra) / max(dd, 1e-12)), r
    while len(cl) > 1:
        best = None
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                c, r = union(cl[i], cl[j]); cost = r - max(cl[i][2], cl[j][2])
                big, small = max(cl[i][2], cl[j][2]), min(cl[i][2], cl[j][2])
                if r <= 1.8 * big and big <= 6 * small and (best is None or cost < best[0]): best = (cost, i, j, c, r)
        if best is None: break
        _, i, j, c, r = best; merged = (cl[i][0] + cl[j][0], c, r)
        cl = [x for k, x in enumerate(cl) if k not in (i, j)] + [merged]
    return [(m, c, r) for m, c, r in cl]

class Group:
    def __init__(self, sc, g):
        self.ids = np.array(g[0], int); self.c, self.r = g[1], g[2]
        self.BC = np.array([sc.objs[i].bc for i in self.ids]); self.BR = np.array([sc.objs[i].br for i in self.ids])
        self.is_sph = np.array([sc.objs[i].kind == "sphere" for i in self.ids], bool)
        self.single = len(self.ids) == 1

def intersect(sc, o, d, tmax=None, lights=True):
    N = len(o); best = np.full(N, np.inf) if tmax is None else tmax.copy(); oid = np.full(N, -1)
    if sc.groups:
        g0, g1, gmiss = sphere_test(o, d, sc.GC, sc.GR)
        for gi, G in enumerate(sc.groups):
            idx = np.nonzero(~gmiss[:, gi] & (g0[:, gi] < best))[0]
            if len(idx) == 0: continue
            oo, dd, bb = o[idx], d[idx], best[idx]
            if G.single:
                ob = sc.objs[G.ids[0]]
                t = _pick(g0[idx, gi], g1[idx, gi]) if ob.kind == "sphere" else ob.hit(oo, dd)
                m = t < bb; best[idx[m]] = t[m]; oid[idx[m]] = G.ids[0]
                continue
            t0, t1, miss = sphere_test(oo, dd, G.BC, G.BR)
            ob_ = np.full(len(idx), -1)
            if G.is_sph.any():
                ts = np.where(t0[:, G.is_sph] > EPS, t0[:, G.is_sph], t1[:, G.is_sph]); ts[miss[:, G.is_sph]] = np.inf
                k = ts.argmin(1); tk = ts[np.arange(len(idx)), k]
                m = tk < bb; bb[m] = tk[m]; ob_[m] = G.ids[G.is_sph][k[m]]
            for j in np.where(~G.is_sph)[0]:
                jj = np.nonzero(~miss[:, j] & (t0[:, j] < bb))[0]
                if len(jj) == 0: continue
                t = sc.objs[G.ids[j]].hit(oo[jj], dd[jj])
                m = t < bb[jj]; bb[jj[m]] = t[m]; ob_[jj[m]] = G.ids[j]
            m = ob_ >= 0; best[idx[m]] = bb[m]; oid[idx[m]] = ob_[m]
    for i in sc.pidx:
        t = sc.objs[i].hit(o, d); m = t < best; best[m] = t[m]; oid[m] = i
    if lights:
        for j, L in enumerate(sc.lights):
            t = L.hit(o, d); m = t < best; best[m] = t[m]; oid[m] = sc.LBASE + j
    return best, oid

def geo_normals(sc, p, oid):
    n = np.zeros_like(p)
    for i in np.unique(oid):
        if i < 0 or i >= sc.K: continue
        m = oid == i; n[m] = sc.objs[i].normal(p[m])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)

# ------------------------------------------------------------------ textures: f(q_local, n_world, obj, mat) -> (albedo, bumped normal)
def tex_walnut(q, n, ob, mt):
    x, z = q[:, 0] * mt.p.get("scale", 1.0), q[:, 1] * mt.p.get("scale", 1.0)
    warp = fbm2(x * 0.05, z * 0.35, 4) * 6
    grain = 0.5 + 0.5 * np.sin((z + warp) * (1.3 + 0.8 * fbm2(x * 0.01, z * 0.05, 2)) + fbm2(x * 0.03, z * 0.6, 4) * 9)
    fine = fbm2(x * 0.6, z * 9, 3)
    base = rgb(mt.p.get("base"), [0.20, 0.105, 0.055]); dark = rgb(mt.p.get("dark"), [0.09, 0.045, 0.025])
    g = (grain ** 3 * 0.7 + fine * 0.3)[:, None]
    return base * (1 - g) + dark * g, n

def tex_plaster(q, n, ob, mt):
    v = fbm2(q[:, 0] * 0.06, q[:, 1] * 0.06, 5)[:, None]
    return rgb(mt.p.get("base"), [0.10, 0.095, 0.075]) * (0.7 + 0.6 * v), n

def tex_pom(q, n, ob, mt):
    ang = np.arctan2(q[:, 2], q[:, 0]); lat = q[:, 1] / ob.ry
    blot = fbm2(ang * 2.2 + 3, lat * 3.1, 5)
    crimson = np.array([0.46, 0.035, 0.03]); deep = np.array([0.22, 0.012, 0.018]); ochre = np.array([0.55, 0.30, 0.10])
    t1 = np.clip((blot - 0.35) * 2.2, 0, 1)[:, None]
    c = crimson * (1 - t1) + deep * t1
    pale = np.clip((fbm2(ang * 5 + 9, lat * 7, 4) - 0.62) * 4, 0, 1)[:, None] * np.clip(0.5 - lat, 0, 1)[:, None]
    c = c * (1 - pale) + ochre * pale
    c = c * (0.65 + 0.35 * np.clip(1.2 - np.abs(lat), 0, 1))[:, None]
    lobe = np.cos(ang * 6) * 0.06 * (1 - lat ** 2)                      # six soft lobes + fine pitting
    tang = ob.to_world_dir(np.stack([-np.sin(ang), np.zeros_like(ang), np.cos(ang)], 1))
    pit = (fbm2(ang * 40, lat * 40, 2) - 0.5) * 0.08
    nb = n + tang * (-np.sin(ang * 6) * 0.35)[:, None] * 0.25 + n * lobe[:, None] + pit[:, None] * tang
    return c, nb

def tex_apricot(q, n, ob, mt):
    ang = np.arctan2(q[:, 2], q[:, 0]); lat = q[:, 1] / ob.ry
    blush = np.clip(fbm2(ang * 1.5, lat * 2, 4) * 1.6 - 0.45, 0, 1)[:, None]
    return np.array([0.80, 0.42, 0.08]) * (1 - blush) + np.array([0.62, 0.16, 0.05]) * blush, \
        n + (fbm2(ang * 60, lat * 60, 2) - 0.5)[:, None] * 0.05

def tex_crown(q, n, ob, mt):
    return np.array([0.20, 0.08, 0.04]) * (0.6 + 0.8 * fbm2(q[:, 0] * 4, q[:, 2] * 4 + q[:, 1] * 3, 3))[:, None], n

def tex_marble(q, n, ob, mt):
    """white stone with soft grey veins; params base, vein, scale."""
    s = mt.p.get("scale", 1.0); x, z = q[:, 0] * s, q[:, 1] * s
    turb = fbm2(x * 0.08, z * 0.08, 6) * 7 + fbm2(x * 0.3 + 5, z * 0.3, 4) * 1.5
    v = np.abs(np.sin(x * 0.11 + z * 0.05 + turb)) ** 0.35
    vein = (1 - v)[:, None]
    cloud = (0.9 + 0.2 * fbm2(x * 0.04 + 9, z * 0.04, 4))[:, None]
    base = rgb(mt.p.get("base"), [0.78, 0.76, 0.72]); vc = rgb(mt.p.get("vein"), [0.35, 0.34, 0.33])
    return (base * (1 - vein) + vc * vein) * cloud, n

def tex_oak(q, n, ob, mt):
    """dark planked wood: grain along u, plank seams every `plank` units along v."""
    s = mt.p.get("scale", 1.0); x, z = q[:, 0] * s, q[:, 1] * s
    pl = mt.p.get("plank", 22.0); k = np.floor(z / pl); zz = z - k * pl
    off = _h2(k.astype(np.int64) + 50, np.zeros_like(k, dtype=np.int64)) * 300
    xx = x + off
    warp = fbm2(xx * 0.02, zz * 0.2 + k * 3, 4) * 5
    ring = 0.5 + 0.5 * np.sin((zz + warp) * 1.7 + fbm2(xx * 0.05, zz * 0.9, 3) * 6)
    fine = fbm2(xx * 0.3, zz * 12, 3)
    g = (ring ** 4 * 0.55 + fine * 0.45)[:, None]
    tone = (0.8 + 0.4 * _h2(k.astype(np.int64) + 7, np.ones_like(k, dtype=np.int64) * 3))[:, None]
    base = rgb(mt.p.get("base"), [0.16, 0.085, 0.045]); dark = rgb(mt.p.get("dark"), [0.05, 0.025, 0.014])
    alb = (base * (1 - g) + dark * g) * tone
    seam = np.exp(-(np.minimum(zz, pl - zz) / 0.12) ** 2)[:, None]
    alb = alb * (1 - 0.7 * seam)
    return alb, n

def tex_linen(q, n, ob, mt):
    s = mt.p.get("scale", 1.0); x, z = q[:, 0] * s, q[:, 1] * s
    w = 0.5 + 0.25 * np.sin(x * 25) * np.sin(z * 25) + 0.25 * fbm2(x * 3, z * 0.4, 3)
    base = rgb(mt.p.get("base"), [0.7, 0.66, 0.6])
    return base * (0.8 + 0.25 * w)[:, None], n

TEXTURES = {"walnut": tex_walnut, "plaster": tex_plaster, "pom": tex_pom, "apricot": tex_apricot,
            "crown": tex_crown, "marble": tex_marble, "oak": tex_oak, "linen": tex_linen}

def albedo_and_bump(sc, p, n, oid):
    alb = np.full((len(p), 3), 0.5); nb = n.copy()
    for i in np.unique(oid):
        m = oid == i; ob = sc.objs[i]; mt = ob.mat
        if mt.tex:
            a, b = TEXTURES[mt.tex](ob.local_point(p[m]), n[m], ob, mt)
            alb[m] = a * mt.tint; nb[m] = b
        else:
            alb[m] = mt.color * mt.tint
    return alb, nb / np.linalg.norm(nb, axis=1, keepdims=True)

# ------------------------------------------------------------------ sampling helpers
def onb(n):
    s = np.where(n[:, 2] >= 0, 1.0, -1.0)
    a = -1 / (s + n[:, 2]); b = n[:, 0] * n[:, 1] * a
    t = np.stack([1 + s * n[:, 0] ** 2 * a, s * b, -s * n[:, 0]], 1)
    bt = np.stack([b, s + n[:, 1] ** 2 * a, -n[:, 1]], 1)
    return t, bt

def cosine_dir(n, rng):
    u1, u2 = rng.random(len(n)), rng.random(len(n))
    r = np.sqrt(u1); phi = 2 * np.pi * u2; t, b = onb(n)
    return t * (r * np.cos(phi))[:, None] + b * (r * np.sin(phi))[:, None] + n * np.sqrt(1 - u1)[:, None]

def reflect(d, n): return d - 2 * np.sum(d * n, 1, keepdims=True) * n
def dot(a, b): return np.sum(a * b, 1)

def ggx_D(ch, a):
    a2 = a * a; den = ch * ch * (a2 - 1) + 1
    return a2 / (np.pi * den * den)
def ggx_G1(cv, a):
    a2 = a * a; return 2 * cv / (cv + np.sqrt(a2 + (1 - a2) * cv * cv))
def ggx_sample_h(n, a, rng):
    u1, u2 = rng.random(len(n)), rng.random(len(n))
    ct = 1 / np.sqrt(1 + a * a * u1 / (1 - u1 + 1e-12)); st = np.sqrt(np.maximum(0, 1 - ct * ct)); ph = 2 * np.pi * u2
    t, b = onb(n)
    return t * (st * np.cos(ph))[:, None] + b * (st * np.sin(ph))[:, None] + n * ct[:, None]

def bsdf_eval(kind, alb, f0, cf0, a, n, wo, wi, P_s):
    """f*cos(wi) and the sampling pdf (solid angle) for diffuse / coat / rough metal."""
    ci = np.maximum(dot(n, wi), 0); co = np.maximum(dot(n, wo), 1e-6)
    h = wo + wi; h /= np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-12)
    ch = np.clip(dot(n, h), 0, 1); oh = np.maximum(dot(wo, h), 1e-6)
    D = ggx_D(ch, a); G = ggx_G1(ci, a) * ggx_G1(co, a)
    pdf_spec = D * ch / (4 * oh)
    f = np.zeros((len(n), 3)); pdf = np.zeros(len(n))
    m = kind == DIFF
    f[m] = alb[m] / np.pi * ci[m, None]; pdf[m] = ci[m] / np.pi
    m = kind == METAL
    F = f0[m] + (1 - f0[m]) * ((1 - oh[m]) ** 5)[:, None]
    f[m] = F * (D[m] * G[m] / (4 * co[m]))[:, None]; pdf[m] = pdf_spec[m]
    m = kind == COAT
    Fh = cf0[m] + (1 - cf0[m]) * (1 - oh[m]) ** 5; Fo = cf0[m] + (1 - cf0[m]) * (1 - co[m]) ** 5
    f[m] = (Fh * D[m] * G[m] / (4 * co[m]))[:, None] + ((1 - Fo) * ci[m] / np.pi)[:, None] * alb[m]
    pdf[m] = P_s[m] * pdf_spec[m] + (1 - P_s[m]) * ci[m] / np.pi
    f[ci <= 0] = 0
    return f, pdf

def fresnel_dielectric(cosi, eta):
    """exact unpolarised Fresnel; eta = n1/n2; returns (F, cos_t, tir)."""
    k = 1 - eta * eta * (1 - cosi * cosi); tir = k < 0; ct = np.sqrt(np.maximum(k, 0))
    rs = (eta * cosi - ct) / (eta * cosi + ct + 1e-12); rp = (eta * ct - cosi) / (eta * ct + cosi + 1e-12)
    F = np.where(tir, 1.0, 0.5 * (rs * rs + rp * rp))
    return F, ct, tir

def light_hit_emit(sc, lid, d):
    """emitted radiance of light index lid seen along d (front side only), and cos at the light."""
    out = np.zeros((len(d), 3)); cosl = np.zeros(len(d))
    for j, L in enumerate(sc.lights):
        m = lid == j
        if not m.any(): continue
        c = -(d[m] @ L.n); cosl[m] = c; out[m] = L.emit * ((c > 0) * np.maximum(c, 0) ** L.spot)[:, None]
    return out, cosl

def sample_lights(sc, p, rng):
    M = len(p); j = np.searchsorted(sc.lcdf, rng.random(M) * sc.lcdf[-1]).clip(0, len(sc.lights) - 1)
    u1, u2 = rng.random(M), rng.random(M)
    lp = np.zeros((M, 3)); ln = np.zeros((M, 3)); A = np.zeros(M); Le = np.zeros((M, 3)); sk = np.zeros(M)
    for k, L in enumerate(sc.lights):
        m = j == k
        lp[m] = L.corner + L.u * u1[m, None] + L.v * u2[m, None]; ln[m] = L.n; A[m] = L.area; Le[m] = L.emit; sk[m] = L.spot
    return lp, ln, A, Le, sc.lsel[j], sk

def light_pdf(sc, lid, t, cosl):
    A = np.array([L.area for L in sc.lights])[lid]; P = sc.lsel[lid]
    return P * t * t / (A * np.maximum(cosl, 1e-6))

def clampc(c, dif, clamp):
    if not clamp: return c
    mx = c.max(1); s = np.where(dif & (mx > clamp), clamp / np.maximum(mx, 1e-12), 1.0)
    return c * s[:, None]

# ------------------------------------------------------------------ spectral helpers (dispersion)
def _cie_xyz(l):
    g = lambda l, mu, s1, s2: np.exp(-0.5 * ((l - mu) / np.where(l < mu, s1, s2)) ** 2)
    x = 1.056 * g(l, 599.8, 37.9, 31.0) + 0.362 * g(l, 442.0, 16.0, 26.7) - 0.065 * g(l, 501.1, 20.4, 26.2)
    y = 0.821 * g(l, 568.8, 46.9, 40.5) + 0.286 * g(l, 530.9, 16.3, 31.1)
    z = 1.217 * g(l, 437.0, 11.8, 36.0) + 0.681 * g(l, 459.0, 26.0, 13.8)
    return np.stack([x, y, z], -1)
LAM0, LAM1 = 380.0, 720.0
_L = np.linspace(LAM0, LAM1, 341)
_RGB = np.maximum(_cie_xyz(_L) @ np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]]).T, 0)
_RGB /= _RGB.mean(0)                    # white-balanced: the average weight over wavelengths is (1, 1, 1)
def spectral_weight(lam): return np.stack([np.interp(lam, _L, _RGB[:, c]) for c in range(3)], 1)

def pick_wavelengths(sc, oid, lam, rng):
    """paths meeting dispersive glass for the first time get a wavelength; returns (lam, rgb weight or None)."""
    need = (sc.DISP[oid] > 0) & np.isnan(lam)
    if not need.any(): return lam, None
    lam = lam.copy(); w = np.ones((len(lam), 3))
    lam[need] = LAM0 + (LAM1 - LAM0) * rng.random(need.sum()); w[need] = spectral_weight(lam[need])
    return lam, w

# ------------------------------------------------------------------ dielectric event (shared by camera paths and photons)
def dielectric(sc, d, ng, nb, t, oid, into, p, lam=None):
    ior_in = sc.IOR[oid]; ior_out = sc.OIOR[oid].copy(); sig_out = sc.OSIG[oid].copy()
    if lam is not None and sc.any_disp:                          # Cauchy: n(l) = n_d + B (1/l^2 - 1/l_d^2), l in micrometres
        lm = np.where(np.isnan(lam), 587.6, lam) / 1000
        ior_in = ior_in + sc.DISP[oid] * (1 / lm ** 2 - 1 / 0.5876 ** 2)
    for i in sc.tubes_fill:                                      # glass wall touching a liquid fill
        m = oid == i
        if m.any():
            ob = sc.objs[i]; lq = ob.in_liquid(p[m]); idx = np.where(m)[0][lq]
            ior_out[idx] = ob.fill_mat.ior; sig_out[idx] = ob.fill_mat.sigma
    sig = np.where(into[:, None], sig_out, sc.SIG[oid])
    atten = np.exp(-sig * t[:, None])                            # Beer-Lambert on the segment just travelled
    nf = np.where(into[:, None], nb, -nb)
    eta = np.where(into, ior_out / ior_in, ior_in / ior_out)
    cosi = np.clip(-dot(d, nf), 0, 1)
    Fr, ct, tir = fresnel_dielectric(cosi, eta)
    rdir = reflect(d, nf)
    tdir = eta[:, None] * d + (eta * cosi - ct)[:, None] * nf
    tdir /= np.maximum(np.linalg.norm(tdir, axis=1, keepdims=True), 1e-12)
    return atten, Fr, rdir, tdir, tir

def spec_connect(sc, p, ng, dirs, w, excl, dif):
    """deterministic one-hop check: does this specular direction reach a light? returns radiance."""
    out = np.zeros((len(p), 3))
    cand = np.zeros(len(p), bool); tl = np.full(len(p), np.inf)
    for L in sc.lights:
        t = L.hit(p, dirs); c = (-(dirs @ L.n) > 0) & (t < tl)
        tl = np.where(c, t, tl); cand |= c
    cand &= ~excl & (w.max(1) > 0)
    if not cand.any(): return out
    idx = np.where(cand)[0]
    side = np.sign(dot(dirs[idx], ng[idx]))[:, None]
    t, oid = intersect(sc, p[idx] + ng[idx] * side * EPS * 5, dirs[idx])
    lit = oid >= sc.LBASE
    if lit.any():
        ii = idx[lit]; Le, cl = light_hit_emit(sc, oid[lit] - sc.LBASE, dirs[ii])
        out[ii] = clampc(w[ii] * Le, dif[ii], sc.clamp)
    return out

# ------------------------------------------------------------------ caustic photon map
class PhotonMap:
    def __init__(self, sc, rng, n_photons, k, rmax):
        from scipy.spatial import cKDTree
        self.k, self.rmax = k, rmax; self.ok = False
        tgt = [i for i, o in enumerate(sc.objs) if sc.PMSURF[i] and o.kind != "plane"]
        if not tgt: return
        TC = np.array([sc.objs[i].bc for i in tgt]); TR = np.array([sc.objs[i].br for i in tgt]) * 1.02
        N = n_photons
        lp, ln, A, Le, P, sk = sample_lights(sc, np.zeros((N, 3)), rng)
        # choose a target by the solid angle of its bounding sphere, sample a direction in its cone
        v = TC[None] - lp[:, None]; dist = np.linalg.norm(v, axis=2)
        sina = np.clip(TR[None] / dist, 0, 0.9999); cosa = np.sqrt(1 - sina ** 2)
        front = np.einsum("nkj,nj->nk", v, ln) > 0
        om = 2 * np.pi * (1 - cosa) * front; tot = om.sum(1)
        good = tot > 0
        cdf = np.cumsum(om, 1) / np.maximum(tot, 1e-30)[:, None]
        k_ = (rng.random(N)[:, None] > cdf).sum(1).clip(0, len(tgt) - 1)
        ax = v[np.arange(N), k_] / dist[np.arange(N), k_][:, None]
        ca = cosa[np.arange(N), k_]
        u1, u2 = rng.random(N), rng.random(N)
        ct = 1 - u1 * (1 - ca); st = np.sqrt(np.maximum(0, 1 - ct * ct)); ph = 2 * np.pi * u2
        t_, b_ = onb(ax)
        d = t_ * (st * np.cos(ph))[:, None] + b_ * (st * np.sin(ph))[:, None] + ax * ct[:, None]
        # mixture pdf over all cones that contain d
        cosk = np.einsum("nkj,nj->nk", v, d) / dist
        inside = (cosk >= cosa) & front
        pdf = (inside * (om / np.maximum(tot, 1e-30)[:, None]) / np.maximum(om, 1e-30)).sum(1)
        cosl = dot(d, ln)
        pw = Le * (A * np.maximum(cosl, 0) ** (1 + sk) / (P * np.maximum(pdf, 1e-30) * N))[:, None]
        pw[~good | (cosl <= 0)] = 0
        o = lp + ln * EPS * 5
        keep = pw.max(1) > 0; o, d, pw = o[keep], d[keep], pw[keep]
        nspec = np.zeros(len(o), int); lam = np.full(len(o), np.nan)
        P_pos, P_pw, P_n = [], [], []
        for depth in range(sc.max_depth):
            if len(o) == 0: break
            t, oid = intersect(sc, o, d, lights=False)
            hit = oid >= 0
            o, d, pw, nspec, t, oid, lam = o[hit], d[hit], pw[hit], nspec[hit], t[hit], oid[hit], lam[hit]
            if len(o) == 0: break
            p = o + d * t[:, None]; ng = geo_normals(sc, p, oid); kind = sc.KIND[oid]; spec = sc.PMSURF[oid]
            dep = ~spec & ((kind == DIFF) | (kind == COAT)) & (nspec > 0)
            if dep.any():
                into = dot(d[dep], ng[dep]) < 0
                P_pos.append(p[dep]); P_pw.append(pw[dep]); P_n.append(np.where(into[:, None], ng[dep], -ng[dep]))
            s = spec
            o, d, pw, nspec, t, oid, p, ng, lam = o[s], d[s], pw[s], nspec[s], t[s], oid[s], p[s], ng[s], lam[s]
            if len(o) == 0: break
            kind = sc.KIND[oid]; into = dot(d, ng) < 0
            nd = np.zeros_like(d)
            g = kind == GLASS
            if g.any():
                lam, sw = pick_wavelengths(sc, oid, lam, rng)
                if sw is not None: pw *= sw
                atten, Fr, rdir, tdir, tir = dielectric(sc, d[g], ng[g], ng[g], t[g], oid[g], into[g], p[g], lam[g])
                pw[g] *= atten; refl = rng.random(g.sum()) < Fr
                nd[g] = np.where(refl[:, None], rdir, tdir)
            m = kind == METAL
            if m.any():
                nf = np.where(into[m, None], ng[m], -ng[m]); a = sc.ALPHA[oid[m]]; f0 = sc.F0[oid[m]]
                mir = a < MIRROR_ALPHA
                h = np.where(mir[:, None], nf, ggx_sample_h(nf, np.maximum(a, 1e-3), rng))
                wi = reflect(d[m], h); wo = -d[m]
                oh = np.clip(dot(wo, h), 1e-6, 1); co = np.clip(dot(wo, nf), 1e-6, 1); ci = dot(wi, nf); ch = np.clip(dot(h, nf), 1e-6, 1)
                F = f0 + (1 - f0) * ((1 - oh) ** 5)[:, None]
                g = np.where(mir, 1.0, ggx_G1(np.clip(ci, 1e-6, 1), a) * ggx_G1(co, a) * oh / (co * ch))
                pw[m] *= F * (g * (ci > 0))[:, None]; nd[m] = wi
            side = np.sign(dot(nd, ng))[:, None]
            o = p + ng * side * EPS * 5; d = nd; nspec = nspec + 1
            if depth >= 3:
                q = np.clip(pw.max(1) / np.maximum(pw.max(), 1e-30) * 4, 0.05, 1); sv = rng.random(len(o)) < q
                pw = pw / q[:, None]; o, d, pw, nspec, lam = o[sv], d[sv], pw[sv], nspec[sv], lam[sv]
        if not P_pos: return
        self.pos = np.concatenate(P_pos); self.pw = np.concatenate(P_pw); self.n = np.concatenate(P_n)
        self.tree = cKDTree(self.pos); self.ok = True; self.count = len(self.pos)

    def gather(self, p, n):
        """irradiance-like sum: returns flux density (per area) at points p with facing normals n."""
        out = np.zeros((len(p), 3))
        if not self.ok or len(p) == 0: return out
        k = min(self.k, self.count)
        dd, ii = self.tree.query(p, k=k, distance_upper_bound=self.rmax)
        if k == 1: dd, ii = dd[:, None], ii[:, None]
        valid = ii < self.count; iic = np.where(valid, ii, 0)
        r = np.where(valid.all(1), dd[:, -1], self.rmax); r = np.maximum(r, 1e-4)
        nok = np.einsum("nkj,nj->nk", self.n[iic], n) > 0.5
        w = valid & nok
        # cone filter (weight 1 - d/r, normalised by 1 - 2/3) keeps edges crisp
        kern = np.clip(1 - np.where(valid, dd, 0) / r[:, None], 0, 1) * w
        flux = np.einsum("nk,nkj->nj", kern, self.pw[iic])
        return flux / ((1 - 2 / 3) * np.pi * r * r)[:, None]

# ------------------------------------------------------------------ path tracing
def trace(sc, o, d, rng, want_aux, mis=True, pmap=None):
    N = len(o)
    L = np.zeros((N, 3)); T = np.ones((N, 3)); pid = np.arange(N)
    prev_pdf = np.zeros(N)                   # pdf of the BSDF sample that produced d (0 = camera/specular: full weight, -1 = NEE only)
    skip = np.zeros(N, bool)                 # emission along d already counted by a specular connection
    dif = np.zeros(N, bool)                  # a diffuse vertex happened (firefly clamp applies)
    cst = np.zeros(N, np.int8)               # caustic state: 0 none, 1 first diffuse, 2 diffuse+specular, 3 other
    lam = np.full(N, np.nan)                 # wavelength (nm), chosen at the first dispersive glass
    use_pm = pmap is not None and pmap.ok
    aux = None
    for depth in range(sc.max_depth):
        if len(pid) == 0: break
        t, oid = intersect(sc, o, d)
        miss = oid == -1
        L[pid[miss]] += T[miss] * sc.env
        hl = oid >= sc.LBASE
        if hl.any():
            lid = oid[hl] - sc.LBASE; Le, cl = light_hit_emit(sc, lid, d[hl])
            pb = prev_pdf[hl]; pl = light_pdf(sc, lid, t[hl], cl)
            w = np.where(pb == 0, 1.0, np.where(pb < 0, 0.0, pb * pb / (pb * pb + pl * pl)))
            w[skip[hl]] = 0
            if use_pm: w[(cst[hl] == 2)] = 0
            L[pid[hl]] += clampc(T[hl] * Le * w[:, None], dif[hl], sc.clamp)
        keep = ~miss & ~hl
        o, d, t, oid, T, pid, dif, cst, lam = o[keep], d[keep], t[keep], oid[keep], T[keep], pid[keep], dif[keep], cst[keep], lam[keep]
        if len(pid) == 0: break
        p = o + d * t[:, None]
        ng = geo_normals(sc, p, oid)
        alb, nb = albedo_and_bump(sc, p, ng, oid)
        kind = sc.KIND[oid]; spec = sc.SPECULAR[oid]
        if depth == 0 and want_aux:
            aux = dict(alb=np.zeros((N, 3)), nrm=np.zeros((N, 3)), dep=np.full(N, 1e3), spc=np.zeros(N))
            a_ = alb.copy(); a_[(kind == GLASS) | (kind == METAL)] = 1.0
            aux["alb"][pid] = a_; aux["nrm"][pid] = ng; aux["dep"][pid] = t; aux["spc"][pid] = sc.SPCMASK[oid]
        n_ = len(pid)
        into = dot(d, ng) < 0
        nf = np.where(into[:, None], nb, -nb)
        wo = -d
        nd = np.zeros_like(d); npdf = np.zeros(n_); nskip = np.zeros(n_, bool); ncst = cst.copy(); ndif = dif.copy()
        alive = np.ones(n_, bool)
        # ---------------- non-specular: diffuse, coat, rough metal
        ns = ~spec
        if ns.any():
            idx = np.where(ns)[0]; k_ = kind[idx]
            f0 = sc.F0[oid[idx]]; cf0 = sc.CF0[oid[idx]]; a = np.maximum(sc.ALPHA[oid[idx]], 1e-3)
            co = np.clip(dot(nf[idx], wo[idx]), 1e-6, 1)
            Fo = cf0 + (1 - cf0) * (1 - co) ** 5
            P_s = np.where(k_ == COAT, np.clip(2 * Fo + 0.1, 0.1, 0.6), np.where(k_ == METAL, 1.0, 0.0))
            # caustic photon gather at the first diffuse vertex
            if use_pm:
                g = ((k_ == DIFF) | (k_ == COAT)) & (cst[idx] == 0)
                if g.any():
                    gi = idx[g]; E = pmap.gather(p[gi], np.where(into[gi, None], ng[gi], -ng[gi]))
                    fd = alb[gi] / np.pi * np.where(k_[g] == COAT, 1 - Fo[g], 1.0)[:, None]
                    L[pid[gi]] += T[gi] * fd * E
            # next-event estimation
            do_nee = np.ones(len(idx), bool) if mis else (k_ != METAL)
            glossy_pm = sc.PMSURF[oid[idx]] if use_pm else np.zeros(len(idx), bool)
            if use_pm: do_nee &= ~(glossy_pm & ((cst[idx] == 1) | (cst[idx] == 2)))
            if do_nee.any():
                ii = idx[do_nee]
                lp, ln, A, Le, P, sk = sample_lights(sc, p[ii], rng)
                wv = lp - p[ii]; dist = np.linalg.norm(wv, axis=1); wv /= dist[:, None]
                cosl = -dot(wv, ln); ok = (cosl > 0) & (dot(wv, nf[ii]) > 0)
                Le = Le * (np.maximum(cosl, 0) ** sk)[:, None]
                if mis: fe, pb = bsdf_eval(k_[do_nee], alb[ii], f0[do_nee], cf0[do_nee], a[do_nee], nf[ii], wo[ii], wv, P_s[do_nee])
                else:   # basic mode: NEE carries only the diffuse part
                    kk = np.where(k_[do_nee] == METAL, METAL, DIFF)
                    fe, pb = bsdf_eval(kk, alb[ii], f0[do_nee], cf0[do_nee], a[do_nee], nf[ii], wo[ii], wv, P_s[do_nee])
                    fe = fe * np.where(k_[do_nee] == COAT, 1 - Fo[do_nee], 1.0)[:, None]
                pl = P * dist * dist / (A * np.maximum(cosl, 1e-6))
                wl = pl * pl / (pl * pl + pb * pb) if mis else np.ones(len(ii))
                ok &= fe.max(1) > 0
                if ok.any():
                    jj = np.where(ok)[0]
                    _, hid = intersect(sc, p[ii[jj]] + nf[ii[jj]] * EPS * 5, wv[jj], tmax=dist[jj] - 0.01, lights=False)
                    vis = hid == -1; jj = jj[vis]
                    L[pid[ii[jj]]] += clampc(T[ii[jj]] * fe[jj] * Le[jj] * (wl[jj] / pl[jj])[:, None], dif[ii[jj]], sc.clamp)
            # BSDF sampling
            pick_spec = rng.random(len(idx)) < P_s
            wi = np.zeros((len(idx), 3))
            if pick_spec.any():
                h = ggx_sample_h(nf[idx[pick_spec]], a[pick_spec], rng); wi[pick_spec] = reflect(d[idx[pick_spec]], h)
            if (~pick_spec).any(): wi[~pick_spec] = cosine_dir(nf[idx[~pick_spec]], rng)
            if mis:
                fs, pdf = bsdf_eval(k_, alb[idx], f0, cf0, a, nf[idx], wo[idx], wi, P_s)
                thr = fs / np.maximum(pdf, 1e-12)[:, None]; npdf[idx] = pdf
            else:
                # basic mode: each lobe estimated only by its own sampling (diffuse by NEE, glossy by BSDF)
                fs, pdf = bsdf_eval(np.where(k_ == COAT, np.where(pick_spec, COAT, DIFF), k_), alb[idx], f0, cf0, a,
                                    nf[idx], wo[idx], wi, np.where(k_ == COAT, 1.0, P_s))
                ls = np.where(pick_spec, P_s, 1 - P_s)
                # coat: spec lobe alone (strip the diffuse part) / diffuse lobe alone
                ci = np.maximum(dot(nf[idx], wi), 0)
                dpart = ((1 - Fo) * ci / np.pi)[:, None] * alb[idx]
                fs = np.where(((k_ == COAT) & pick_spec)[:, None], np.maximum(fs - dpart, 0),
                              np.where(((k_ == COAT) & ~pick_spec)[:, None], dpart, fs))
                pdf = np.where((k_ == COAT) & ~pick_spec, ci / np.pi, pdf)
                thr = fs / np.maximum(pdf * ls, 1e-12)[:, None]
                npdf[idx] = np.where(pick_spec & (k_ != DIFF), 0.0, -1.0)
            bad = (dot(wi, nf[idx]) <= 0) | (pdf <= 0)
            thr[bad] = 0; alive[idx[bad]] = False
            T[idx] *= thr; nd[idx] = wi
            isd = (k_ == DIFF) | ((k_ == COAT) & ~pick_spec)
            c0 = cst[idx]
            ncst[idx] = np.where(isd, np.where(c0 == 0, 1, 3),
                                 np.where(c0 == 0, 0, np.where(glossy_pm & ((c0 == 1) | (c0 == 2)), 2, 3)))
            ndif[idx] |= isd
        # ---------------- specular: glass and mirror metal
        sp = spec
        if sp.any():
            idx = np.where(sp)[0]
            excl = np.zeros(len(idx), bool)
            if use_pm: excl = (cst[idx] == 1) | (cst[idx] == 2)
            g = kind[idx] == GLASS
            if g.any():
                gi = idx[g]
                if sc.any_disp:
                    lg, sw = pick_wavelengths(sc, oid[gi], lam[gi], rng); lam[gi] = lg
                    if sw is not None: T[gi] *= sw
                atten, Fr, rdir, tdir, tir = dielectric(sc, d[gi], ng[gi], nb[gi], t[gi], oid[gi], into[gi], p[gi], lam[gi])
                T[gi] *= atten
                if mis:
                    L[pid[gi]] += spec_connect(sc, p[gi], ng[gi], rdir, T[gi] * Fr[:, None], excl[g], dif[gi])
                    L[pid[gi]] += spec_connect(sc, p[gi], ng[gi], tdir, T[gi] * (1 - Fr)[:, None], excl[g] | tir, dif[gi])
                # near the camera, reflect at least 20% of the time (and weight it): reflections of lit
                # surfaces in glass are otherwise sparse bright speckle; deeper in the path choose by Fresnel
                pr = np.where(tir, 1.0, np.clip(Fr, 0.2, 0.8)) if (mis and depth <= 1) else Fr
                refl = rng.random(len(gi)) < pr
                T[gi] *= np.where(refl, Fr / np.maximum(pr, 1e-9), (1 - Fr) / np.maximum(1 - pr, 1e-9))[:, None]
                nd[gi] = np.where(refl[:, None], rdir, tdir)
            m = kind[idx] == METAL
            if m.any():
                mi = idx[m]
                ci = np.clip(-dot(d[mi], nf[mi]), 0, 1)[:, None]
                F = sc.F0[oid[mi]] + (1 - sc.F0[oid[mi]]) * (1 - ci) ** 5
                rdir = reflect(d[mi], nf[mi])
                if mis: L[pid[mi]] += spec_connect(sc, p[mi], ng[mi], rdir, T[mi] * F, excl[m], dif[mi])
                T[mi] *= F; nd[mi] = rdir
            nskip[idx] = mis
            npdf[idx] = 0
            c0 = cst[idx]; ncst[idx] = np.where(c0 == 1, 2, c0)
        nd /= np.maximum(np.linalg.norm(nd, axis=1, keepdims=True), 1e-12)
        side = np.sign(dot(nd, ng))[:, None]
        o = p + ng * side * EPS * 5; d = nd; prev_pdf = npdf; skip = nskip; cst = ncst; dif = ndif
        if depth >= 3:
            q = np.clip(T.max(1), 0.05, 0.95); surv = rng.random(n_) < q
            T[surv] /= q[surv, None]; alive &= surv
        o, d, T, pid, prev_pdf, skip, dif, cst, lam = o[alive], d[alive], T[alive], pid[alive], prev_pdf[alive], skip[alive], dif[alive], cst[alive], lam[alive]
    return L, aux

def camera_rays(sc, W, H, pix, rng):
    ys, xs = pix // W, pix % W; M = len(pix)
    fl = 0.5 * W / math.tan(sc.hfov / 2)
    sx = (xs + rng.random(M) - W / 2) / fl; sy = -(ys + rng.random(M) - H / 2) / fl
    d = sc.fwd[None] + sx[:, None] * sc.right[None] + sy[:, None] * sc.up[None]; d /= np.linalg.norm(d, axis=1, keepdims=True)
    fp = sc.cam + d * (sc.focus / (d @ sc.fwd))[:, None]
    r = sc.aperture * np.sqrt(rng.random(M)); ph = 2 * np.pi * rng.random(M)
    o = sc.cam + sc.right * (r * np.cos(ph))[:, None] + sc.up * (r * np.sin(ph))[:, None]
    d = fp - o; d /= np.linalg.norm(d, axis=1, keepdims=True)
    return o, d

def render_samples(sc, W, H, pix, seed, want_aux, mis, chunk, pmap_cfg):
    rng = np.random.default_rng(seed)
    pmap = None
    if pmap_cfg:
        pmap = PhotonMap(sc, rng, int(pmap_cfg.get("photons", 200000)), int(pmap_cfg.get("k", 48)), float(pmap_cfg.get("rmax", 1.0)))
    o, d = camera_rays(sc, W, H, pix, rng)
    M = len(pix); Lall = np.zeros((M, 3)); auxall = None
    for c0 in range(0, M, chunk):
        Lc, ac = trace(sc, o[c0:c0 + chunk], d[c0:c0 + chunk], rng, want_aux, mis, pmap)
        Lall[c0:c0 + chunk] = Lc
        if ac is not None:
            if auxall is None: auxall = {k: np.zeros((M,) + v.shape[1:]) for k, v in ac.items()}
            for k, v in ac.items(): auxall[k][c0:c0 + chunk] = v
    return Lall, auxall

# ------------------------------------------------------------------ adaptive allocation
def display_slope(Y, exposure):
    """d(display value)/d(linear luminance) for ACES + gamma 2.2, used to weigh noise the way it is seen."""
    x = np.maximum(Y * exposure, 1e-4)
    aces = lambda v: np.clip((v * (2.51 * v + 0.03)) / (v * (2.43 * v + 0.59) + 0.14), 1e-6, 1)
    e = 1e-3 * np.maximum(x, 0.01)
    return np.abs(aces(x + e) ** (1 / 2.2) - aces(x) ** (1 / 2.2)) / e * exposure

def allocate(acc, budget, rng, exposure, diffuse_weight, mask):
    """Neyman allocation: target n_i proportional to sigma_i * slope_i; returns pixel indices (repeats allowed)."""
    from scipy.ndimage import gaussian_filter, maximum_filter
    cnt = np.maximum(acc["cnt"], 1); Y = (acc["sum"] @ np.array([0.2126, 0.7152, 0.0722])) / cnt
    var = np.maximum(acc["sy2"] / cnt - Y * Y, 0) * cnt / np.maximum(cnt - 1, 1)
    sig = np.sqrt(var)
    sig = np.maximum(sig, gaussian_filter(maximum_filter(sig, 3), 1.0))      # spread: few-sample pixels can miss rare paths
    w = sig * display_slope(Y, exposure)
    w *= np.where(acc["spc"] > 0.5, 1.0, diffuse_weight)                        # the denoiser cleans diffuse regions
    w = np.where(mask, w, 0)
    w = w + 0.05 * w[mask].mean()
    w[~mask] = 0
    total = acc["cnt"][mask].sum() + budget
    npix = mask.sum(); capv = 32 * total / npix                                 # no pixel's target above 32x the mean
    t = total * w / w.sum()
    for _ in range(12):                                                         # water-fill the targets under the cap
        over = t > capv
        if not over.any(): break
        excess = (t[over] - capv).sum(); t[over] = capv
        free = ~over & mask
        if not free.any(): break
        t[free] += excess * w[free] / w[free].sum()
    extra = np.maximum(t - acc["cnt"], 0) * mask
    if extra.sum() <= 0: extra = mask.astype(float)                            # (cannot happen: sum(t) > sum(cnt))
    extra *= budget / extra.sum()
    extra = np.minimum(extra, 16 * budget / npix)                               # at most 16x the mean in one pass
    extra *= budget / extra.sum()
    base = np.floor(extra).astype(np.int64); base += (rng.random(extra.shape) < (extra - base))
    return np.repeat(np.arange(extra.size), base.ravel())

# ------------------------------------------------------------------ main
def parse_args(argv):
    if len(argv) >= 3 and argv[0].isdigit() and argv[1].isdigit():         # legacy: W H SPP [tag]
        new = ["--size", f"{argv[0]}x{argv[1]}"]
        if len(argv) > 2: new += ["--spp", argv[2]]
        if len(argv) > 3: new += ["--tag", argv[3]]
        argv = new
    ap = argparse.ArgumentParser(description="numpy path tracer for JSON still-life scenes (resumable)")
    ap.add_argument("--scene", default=os.path.join(HERE, "..", "scenes", "pomegranate.json"), help="scene JSON file")
    ap.add_argument("--size", default="600x200", help="WxH image size")
    ap.add_argument("--spp", type=float, default=8, help="samples per pixel to ADD in this run (average over the frame)")
    ap.add_argument("--tag", default=None, help="accumulator name (default: scene name + size)")
    ap.add_argument("--work", default=os.environ.get("PT_WORK", "./pt_work"), help="folder for accumulators (env PT_WORK)")
    ap.add_argument("--adaptive-after", type=float, default=8, help="uniform passes until this mean spp, then adaptive (0 = adaptive at once after 2 spp, -1 = never)")
    ap.add_argument("--diffuse-weight", type=float, default=0.4, help="adaptive: relative priority of diffuse first-hit pixels (the denoiser cleans those)")
    ap.add_argument("--basic", action="store_true", help="basic sampling: no MIS, no specular light connections (for comparisons)")
    ap.add_argument("--no-caustics", action="store_true", help="ignore the scene's caustic photon map")
    ap.add_argument("--clamp", type=float, default=None, help="override the scene's firefly clamp (0 = off)")
    ap.add_argument("--crop", default=None, help="x0:x1,y0:y1 render only this pixel window")
    ap.add_argument("--time", type=float, default=None, help="stop after this many seconds (checked between passes)")
    ap.add_argument("--chunk", type=int, default=None, help="rays per batch (memory); default from object count")
    ap.add_argument("--exposure", type=float, default=None, help="exposure used to weigh noise for adaptive sampling")
    return ap.parse_args(argv)

def main(argv=None):
    a = parse_args(sys.argv[1:] if argv is None else argv)
    sc = Scene(a.scene)
    if a.clamp is not None: sc.clamp = a.clamp
    W, H = map(int, a.size.lower().split("x"))
    tag = a.tag or f"{os.path.splitext(os.path.basename(a.scene))[0]}_{W}x{H}"
    os.makedirs(a.work, exist_ok=True); path = os.path.join(a.work, f"acc_{tag}.npz")
    mask = np.ones((H, W), bool)
    if a.crop:
        xs, ys = a.crop.split(","); x0, x1 = map(int, xs.split(":")); y0, y1 = map(int, ys.split(":"))
        mask[:] = False; mask[y0:y1, x0:x1] = True
    chunk = a.chunk or int(np.clip(5e6 / max(len(sc.groups), 1), 40000, 200000))
    pm_cfg = None if (a.no_caustics or a.basic) else sc.caustics
    exposure = a.exposure or float(sc.post.get("exposure", 1.0))
    t0 = time.time()
    if os.path.exists(path):
        z = np.load(path)
        if z["sum"].shape[:2] != (H, W): sys.exit(f"{path} is {z['sum'].shape[1]}x{z['sum'].shape[0]}, not {W}x{H}; use another --tag")
        acc = {k: z[k] for k in z.files if k not in ("scene", "hash")}
        if str(z["hash"]) != sc.hash: print("warning: scene file changed since this accumulator was started", flush=True)
    else:
        acc = dict(sum=np.zeros((H, W, 3)), sy2=np.zeros((H, W)), cnt=np.zeros((H, W), np.int64), passes=np.array(0))
    npix = int(mask.sum()); goal = a.spp * npix; done = 0; rng = np.random.default_rng(int(acc["passes"]) * 7919 + 17)
    while done < goal - 0.5:
        passes = int(acc["passes"]); mean_spp = acc["cnt"][mask].mean()
        budget = int(min(npix, goal - done))
        adaptive = a.adaptive_after >= 0 and "alb" in acc and mean_spp >= max(a.adaptive_after, 2) and not a.basic
        if adaptive:
            pix = allocate(acc, budget, rng, exposure, a.diffuse_weight, mask)
            if len(pix) < budget // 2: adaptive = False                           # safety: never run near-empty passes
        if not adaptive:
            allp = np.flatnonzero(mask.ravel())
            pix = allp if budget >= npix else rng.choice(allp, budget, replace=False)
        want_aux = "alb" not in acc
        L, aux = render_samples(sc, W, H, pix, 1000 + passes, want_aux, not a.basic, chunk, pm_cfg)
        if want_aux:
            for k, v in aux.items():
                full = np.zeros((H * W,) + v.shape[1:]); full[pix] = v
                if k == "dep": full[np.setdiff1d(np.arange(H * W), pix)] = 1e3
                acc[k] = full.reshape((H, W) + v.shape[1:])
        Y = L @ np.array([0.2126, 0.7152, 0.0722])
        for c in range(3): acc["sum"][..., c] += np.bincount(pix, L[:, c], H * W).reshape(H, W)
        acc["sy2"] += np.bincount(pix, Y * Y, H * W).reshape(H, W)
        acc["cnt"] += np.bincount(pix, None, H * W).reshape(H, W)
        acc["passes"] = np.array(passes + 1); done += len(pix)
        np.savez(path, **acc, scene=sc.text, hash=sc.hash)
        print(f"pass {passes + 1} {'adaptive' if adaptive else 'uniform '}  mean spp {acc['cnt'][mask].mean():.1f}"
              f"  max {acc['cnt'][mask].max()}  {time.time() - t0:.0f}s", flush=True)
        if a.time and time.time() - t0 > a.time: break
    print(path)

if __name__ == "__main__":
    main()
