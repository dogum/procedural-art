"""
terrain_art — procedural landscape engraving from real elevation data.

Pipeline
  1. fetch_dem()   AWS Terrain Tiles (terrarium PNG, SRTM-derived, global, no key) -> .npz
  2. terrain()     resample DEM into a view-aligned grid (rows run away from the viewer),
                   NORMALISED to a reference mountain (4.3 km tall, 46 km half-width) so every
                   style constant below works for any peak on Earth.
  3. lighting()    slope shading + snow mask
  4. Scene         perspective camera, per-row screen projection, floating horizon (hidden
                   lines), and a per-pixel G-buffer (which terrain row owns each pixel)
  5. STYLES        survey | topo | nocturne | stipple | riso | woodcut

All screen constants are authored in a 3000x1000 reference frame and scaled to the output size.
"""
from __future__ import annotations
import math, os, re, sys
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter, zoom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import font_manager
from PIL import Image

REF_H, REF_L = 4.3, 46.0            # reference mountain: height above plain (km), half-width (km)
U0, U1, V0, V1, ZCAM = -46.0, 46.0, -35.0, 30.0, -95.0

# ------------------------------------------------------------------ data
from dem import fetch_dem, load_dem, ensure_dem  # re-exported for render/animate/build_studio (dem.py is shared with procedural-realism)
_REEXPORTS = (fetch_dem, load_dem, ensure_dem)


def fbm(shape, octaves=6, base=4, persistence=0.5, seed=0):
    r = np.random.default_rng(seed); out = np.zeros(shape); amp = 1.0; tot = 0
    for o in range(octaves):
        n = base * 2 ** o
        g = r.standard_normal((n + 3, n + 3))
        out += amp * zoom(g, (shape[0] / n, shape[1] / n), order=3)[:shape[0], :shape[1]]
        tot += amp; amp *= persistence
    return out / tot


# ------------------------------------------------------------------ terrain
class Terrain:
    """Normalised, view-aligned heightfield. h is in reference-km (REF_H == summit height)."""


def bearing_to_vec(bearing_deg):
    b = math.radians(bearing_deg)
    return math.sin(b), math.cos(b)          # (east, north) unit vector


def _smooth01(x, lo, hi):
    """0 below lo, 1 above hi, smoothstep in between."""
    t = float(np.clip((x - lo) / (hi - lo), 0, 1))
    return t * t * (3 - 2 * t)


def analyse_shape(dem, summit, peak_m, view_dir, fetched_r, kx=None):
    """
    Measure what kind of landform surrounds the summit and suggest framing defaults.

    Everything is measured on azimuthal rings around the (refined) summit:
      base0   legacy plain: 20th percentile on the 0.35..1 x 40 km ring (what cones were tuned on)
      r_half  radius where the ring median has dropped half-way from the peak to base0 (footprint)
      steep   (peak - base0) / r_half                       cone ~0.8, horn ~1.5, shield ~0.35
      around  90th pct of the 2..6 x r_half ring, as a fraction of the peak's height above base0
      front   the same, but only in the 90-degree sector facing the viewer
              (high ground between peak and viewer: cone ~0.3, Matterhorn ~0.6, Tetons from the valley ~0.25)
      rough   std of the 1 km high-pass on the 1..4 x r_half ring / height   (cone ~0.012, alps ~0.05)
      above   95th pct of the 3..20 km ring relative to the peak (> 0 = the target is below the
              surrounding high ground, as for a butte inside a canyon)
    Scores in 0..1: horn (steep and crowded in front), rough (alpine texture), canyon (target below plateau),
    shield (broad gentle dome).
    Suggested values: base_m, top_m (height that maps to the reference 4.3), extent_factor (x the
    legacy 10.5 x height rule), relief, spacing (x 0.375), smooth (extra, normalised units), camh, snowline_m.
    Cones score 0 on everything, so their defaults are exactly the legacy ones.
    """
    elev, lats, lons = dem
    la, lo = summit
    kx, ky = kx or 111.32 * math.cos(math.radians(la)), 110.57

    def ij(lat, lon):
        return (np.interp(lat, lats[::-1], np.arange(len(lats))[::-1].astype(float)),
                (lon - lons[0]) / (lons[1] - lons[0]))
    rr = min(fetched_r * 0.9, 40)
    rs = np.linspace(0, rr, 121); ang = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    A, Rr = np.meshgrid(ang, rs)
    E = np.clip(map_coordinates(elev, ij(la + Rr * np.cos(A) / ky, lo + Rr * np.sin(A) / kx), order=1, mode="nearest"), 0, None)

    def ring(r0, r1, sector=None):
        m = (Rr >= r0) & (Rr <= r1)
        if sector is not None: m &= sector
        return E[m] if m.any() else E[-1]
    # legacy base (same sampling density as before: 12 radii x 180 angles on 0.35..1 x rr)
    Al, Rl = np.meshgrid(np.linspace(0, 2 * np.pi, 180), np.linspace(rr * 0.35, rr, 12))
    base0 = float(np.percentile(map_coordinates(elev, ij(la + Rl * np.cos(Al) / ky, lo + Rl * np.sin(Al) / kx), order=1, mode="nearest"), 20))
    base0 = max(base0, 0.0)
    H = max(peak_m - base0, 300.0)
    med = np.median(E, 1)
    half = base0 + 0.5 * H
    k = int(np.argmax(med <= half)) if (med <= half).any() else len(rs) - 1
    r_half = float(np.interp(half, [med[k], med[k - 1]], [rs[k], rs[k - 1]])) if k > 0 else float(rs[1])
    r_half = max(r_half, 0.3)
    steep = H / 1000 / r_half
    around = (np.percentile(ring(2 * r_half, 6 * r_half), 90) - base0) / H
    dr = rs[1] - rs[0]
    hp_ = E - gaussian_filter(E, (0.8 / dr, 3), mode=("nearest", "wrap"))
    band = (rs > r_half) & (rs < 4 * r_half)
    rough = float(np.std(hp_[band])) / H if band.any() else 0.0
    above = (np.percentile(ring(3, min(20, rr)), 95) - peak_m) / H

    rough_s = _smooth01(rough, 0.02, 0.04)
    canyon = _smooth01(above, 0.1, 0.25)
    shield = _smooth01(0.5 - steep, 0.0, 0.12)      # broad, gentle dome (Mauna Kea): its far flanks are dense flat rows
    # local base in the sector facing the viewer (what actually sits in front of the peak)
    vb = math.atan2(-view_dir[0], -view_dir[1])                    # compass bearing summit -> viewer
    dang = np.angle(np.exp(1j * (A - vb)))
    base_view = float(np.percentile(ring(r_half, 3 * r_half, np.abs(dang) < math.radians(70)), 20))
    # high ground in the 90-degree sector between the peak and the viewer
    front = float((np.percentile(ring(2 * r_half, 6 * r_half, np.abs(dang) < math.radians(45)), 90) - base0) / H)
    # horn: a steep peak with high ground between it and the viewer (Matterhorn from Zermatt).
    # A steep peak rising out of a low foreground (Grand Teton from the valley) is left alone.
    horn = _smooth01(steep, 1.0, 1.4) * _smooth01(front, 0.35, 0.55)
    out = dict(base0=round(base0), r_half_km=round(r_half, 2), steep=round(steep, 2), around=round(float(around), 2),
               rough=round(rough, 3), above=round(float(above), 2), front=round(front, 2),
               horn=round(horn, 2), rough_score=round(rough_s, 2), canyon=round(canyon, 2), shield=round(shield, 2))
    if canyon > 0:
        # canyon / plateau: draw from the floor up to the rim, look down from a little above the rim
        floor = float(np.percentile(ring(0, min(12, rr)), 3))
        rim = float(np.percentile(ring(3, min(20, rr)), 90))
        out.update(kind="canyon", base_m=floor, top_m=rim, extent_factor=1.0, snowline_m=round(rim + 1500),
                   relief=1.0, spacing=1.0, smooth=0.0, camh=6.2, zoom=1.0)
        return out
    base = base0 + horn * max(base_view - base0, 0)
    kind = ("horn" if horn >= 0.5 else "range" if steep >= 1.2 else "massif" if rough_s >= 0.5
            else "shield" if steep < 0.5 else "cone")
    out.update(kind=kind,
               base_m=base, top_m=peak_m,
               extent_factor=round((1 - 0.4 * horn) * (1 - 0.1 * rough_s), 3),
               relief=round((1 - 0.08 * rough_s * (1 - horn)) * (1 + 0.1 * horn), 3), spacing=round(1 + 0.12 * rough_s, 3),
               smooth=round(0.06 * rough_s, 3), camh=round(3.4 - 1.4 * horn, 2) if horn > 0.05 else None,
               snowline_m=None, zoom=1.0)
    return out


_WATER_CACHE = {}


def water_mask(dem, sea_level=0.0, flat_tol=0.01, min_area_km2=0.5, window=None):
    """
    Boolean mask of open water on the DEM grid.
      sea    elevation <= sea_level (the tiles clip ocean bathymetry to 0, and below-zero values are sea floor)
      lakes  patches whose 3x3 elevation range is below flat_tol metres (DEM lakes are exactly flat)
    Connected pieces smaller than min_area_km2 are dropped. Dry land below sea level (Dead Sea shore,
    Death Valley) reads as water; terrain(..., water=False) turns the layer off.
    window=(i0, i1, j0, j1) works on that block of the DEM only (row/column slices). The last result is cached.
    """
    from scipy import ndimage as ndi
    elev, lats, lons = dem
    key = (id(elev), elev.shape, sea_level, flat_tol, min_area_km2, window)
    if key in _WATER_CACHE: return _WATER_CACHE[key]
    if window is not None:
        i0, i1, j0, j1 = window
        elev, lats, lons = elev[i0:i1, j0:j1], lats[i0:i1], lons[j0:j1]
    sea = elev <= sea_level
    rng_ = ndi.maximum_filter(elev, 3) - ndi.minimum_filter(elev, 3)
    lake = ndi.binary_dilation((rng_ < flat_tol) & ~sea)       # the 3x3 test shrinks a lake by one pixel
    m = sea | lake
    if not m.any():
        _WATER_CACHE.clear(); _WATER_CACHE[key] = None; return None
    lab, n = ndi.label(m)
    kx = 111.32 * math.cos(math.radians(float(np.mean(lats))))
    cell_km2 = abs(lats[0] - lats[-1]) / (len(lats) - 1) * 110.57 * abs(lons[1] - lons[0]) * kx
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    keep = sizes * cell_km2 >= min_area_km2
    out = keep[lab] if keep.any() else None
    _WATER_CACHE.clear(); _WATER_CACHE[key] = out          # keep one DEM's mask (orbit frames reuse it)
    return out


def terrain(dem, summit, view_from=None, facing=None, extent_km=None, base_m=None, yaw=0.0,
            NX=1400, NZ=520, smooth=0.6, shape_aware=True, water=True):
    """
    summit    (lat, lon) of the main peak — refined to the true DEM maximum within ~1.5 km
    view_from (lat, lon) of the viewer (e.g. a city); or facing = compass bearing the camera looks toward
    extent_km half-width of the scene. Default scales with the mountain's height (and its shape).
    base_m    elevation of the 'plain' (drawn at zero). Default: 20th percentile around the peak,
              raised towards the local base for horns, dropped to the floor for canyons.
    shape_aware  False restores the pre-shape-analysis auto values (cone rules for everything).
    """
    elev, lats, lons = dem
    la, lo = summit
    kx, ky = 111.32 * math.cos(math.radians(la)), 110.57

    def ij(lat, lon):
        return (np.interp(lat, lats[::-1], np.arange(len(lats))[::-1].astype(float)),
                (lon - lons[0]) / (lons[1] - lons[0]))
    # refine summit
    dd = np.linspace(-1.5, 1.5, 61); EE, NN = np.meshgrid(dd, dd)
    i_, j_ = ij(la + NN / ky, lo + EE / kx)
    e = map_coordinates(elev, [i_, j_], order=1, mode="nearest")
    a = np.unravel_index(e.argmax(), e.shape)
    la, lo = la + NN[a] / ky, lo + EE[a] / kx
    peak_m = float(e.max())
    fetched_r = min((lats[0] - lats[-1]) * ky, (lons[-1] - lons[0]) * kx) / 2
    if view_from is not None:
        E, N = (view_from[1] - lo) * kx, (view_from[0] - la) * ky
        r = math.hypot(E, N); dE, dN = -E / r, -N / r
    else:
        dE, dN = bearing_to_vec(facing if facing is not None else 0.0)
    shape = analyse_shape(dem, (la, lo), peak_m, (dE, dN), fetched_r, kx) if shape_aware else None
    if base_m is None:
        if shape is not None:
            base_m = float(shape["base_m"])
        else:
            rr = min(fetched_r * 0.9, 40)
            ang = np.linspace(0, 2 * np.pi, 180); rad = np.linspace(rr * 0.35, rr, 12)
            A, Rr = np.meshgrid(ang, rad)
            ii, jj = ij(la + Rr * np.cos(A) / ky, lo + Rr * np.sin(A) / kx)
            base_m = float(np.percentile(map_coordinates(elev, [ii, jj], order=1, mode="nearest"), 20))
    base_m = max(base_m, 0.0)
    top_m = shape["top_m"] if shape is not None else peak_m
    hp = max((top_m - base_m) / 1000, 0.3)
    if extent_km is None:
        extent_km = float(np.clip(10.5 * hp * (shape["extent_factor"] if shape else 1.0), 8, 60))
    extent_km = min(extent_km, fetched_r / 1.3)
    k = extent_km / REF_L                    # real km per normalised unit
    hs = REF_H / hp                          # height normalisation
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    dE, dN = dE * c - dN * s, dE * s + dN * c
    rE, rN = dN, -dE
    v0 = V0
    if view_from is not None:
        # start the terrain just in front of the real viewpoint: close viewpoints (Matterhorn from
        # Zermatt, 8 km) would otherwise pull in ridges from beside and behind the viewer
        D = math.hypot(view_from[0] - la, (view_from[1] - lo) * kx / ky) * ky
        v0 = max(V0, -0.8 * D / k)
    NZ = max(60, int(round(NZ * (V1 - v0) / (V1 - V0))))
    us = np.linspace(U0, U1, NX); vs = np.linspace(v0, V1, NZ)
    U, V = np.meshgrid(us, vs)
    lat = la + (U * rN + V * dN) * k / ky; lon = lo + (U * rE + V * dE) * k / kx
    ii, jj = ij(lat, lon)
    ev = map_coordinates(elev, [ii, jj], order=1, mode="nearest")
    sig = smooth
    if shape is not None and shape["smooth"] > 0:        # extra smoothing for rough terrain, in normalised units
        sig = (math.hypot(smooth, shape["smooth"] / (vs[1] - vs[0])), math.hypot(smooth, shape["smooth"] / (us[1] - us[0])))
    h = gaussian_filter(np.clip((ev - base_m) / 1000, 0, None) * hs, sig)

    t = Terrain()
    t.us, t.vs, t.h = us, vs, h
    # water layer (sea and lakes): None when no water is in the scene, so dry views render exactly as before
    t.water = None
    wm = None
    if water:                                  # only the block of DEM under the view grid (fast, and orbit-safe)
        i0, j0 = max(int(ii.min()) - 3, 0), max(int(jj.min()) - 3, 0)
        i1, j1 = min(int(ii.max()) + 4, elev.shape[0]), min(int(jj.max()) + 4, elev.shape[1])
        if i1 - i0 > 3 and j1 - j0 > 3: wm = water_mask(dem, window=(i0, i1, j0, j1))
    if wm is not None:
        wg = map_coordinates(wm.astype(np.float32), [ii - i0, jj - j0], order=1, mode="nearest")
        if (wg > 0.5).sum() >= 0.002 * wg.size:
            t.water = np.clip(gaussian_filter(wg, sig), 0, 1)
    t.k, t.hs, t.base_m, t.peak_m, t.extent_km, t.summit = k, hs, base_m, peak_m, extent_km, (la, lo)
    t.top_m, t.shape = top_m, (shape or {})

    def to_uv(p):
        pe, pn = (p[1] - lo) * kx / k, (p[0] - la) * ky / k
        return pe * rE + pn * rN, pe * dE + pn * dN
    t.to_uv = to_uv
    t.m_to_h = lambda m: (m - base_m) / 1000 * hs   # real metres -> normalised height
    return t


def lighting(t, light=(-0.75, -0.25, 0.6), snowline_m=None):
    """light: (right, away, up) vector in view space. Snowline in real metres (default: 80% of peak)."""
    h, us, vs = t.h, t.us, t.vs
    gz, gx = np.gradient(h, vs[1] - vs[0], us[1] - us[0])
    n = np.dstack([-gx, -gz, np.ones_like(h)]); n /= np.linalg.norm(n, axis=2, keepdims=True)
    L = np.array(light, float); L /= np.linalg.norm(L)
    t.shade = np.clip(n @ L, 0, 1)
    if snowline_m is None:
        snowline_m = getattr(t, "shape", {}).get("snowline_m") or t.base_m + 0.8 * (t.peak_m - t.base_m)
    sl = t.m_to_h(snowline_m)
    t.snow = np.clip((h - sl) / 0.45, 0, 1) * np.clip(1 - np.hypot(gx, gz) / 3.0, 0, 1)
    return t


# ------------------------------------------------------------------ scene
class Scene:
    def __init__(self, t, W=3000, H=1000, camh=3.4, fx=4500, fy=10200, cx=2150, hy=290, hmul=1.0, align="right"):
        self.t, self.W, self.H = t, W, H
        self.s = W / 3000
        self.oy = (H / self.s - 1000) * 0.55                  # re-centre for non-3:1 frames
        if align == "center": cx -= 650
        self.fx, self.fy, self.cx, self.hy, self.camh, self.hmul = fx, fy, cx, hy + self.oy, camh, hmul
        self.fadeX = cx - 1150 if align == "right" else -1e9
        self.us, self.vs, self.h = t.us, t.vs, t.h
        self.v0 = float(t.vs[0]); self.nf = min(16.0, 0.5 * (0 - self.v0))   # near-edge fade width
        self._rows()

    def proj(self, u, v, hh):
        d = v - ZCAM
        return self.s * (self.cx + self.fx * u / d), self.s * (self.hy - self.fy * (hh * self.hmul - self.camh) / d)

    def _rows(self):
        NZ = len(self.vs); px = np.arange(self.W)
        Y = np.full((NZ, self.W), np.inf, np.float32)
        for i in range(NZ):
            sx, sy = self.proj(self.us, self.vs[i], self.h[i])
            m = (px >= sx[0]) & (px <= sx[-1])
            Y[i, m] = np.interp(px[m], sx, sy)
        self.Y = Y
        self.CM = np.minimum.accumulate(Y, axis=0)
        self.silhouette = self.CM[-1]

    def u_at(self, i, px):
        return (px / self.s - self.cx) * (self.vs[i] - ZCAM) / self.fx

    def sample(self, field, i, px):
        j = (self.u_at(i, px) - self.us[0]) / (self.us[1] - self.us[0])
        return np.interp(j, np.arange(len(self.us)), field[i])

    fade_pow = 1.0     # >1 softens the start of the left fade (set for horns/canyons, whose flat rows are dense)
    thin_dense = 0.0   # 0..1: fade ridgelines that crowd closer than ~3 px on screen (set for shields)

    def xfade(self, px, width=450, extra=0):
        return np.clip((px / self.s - self.fadeX - extra) / width, 0, 1) ** self.fade_pow

    def edge_fade(self, u, v):
        return np.clip((46 - np.abs(u)) / 14, 0, 1) * np.clip((v - self.v0) / self.nf, 0, 1) * np.clip((V1 - v) / 8, 0, 1)

    def visible(self, u, v, hh, tol=1.5):
        sx, sy = self.proj(u, v, hh)
        i = np.clip(((v - self.v0) / (self.vs[1] - self.vs[0])).astype(int) - 1, 0, len(self.vs) - 1)
        px = np.clip(np.round(sx).astype(int), 0, self.W - 1)
        return ((sx >= 0) & (sx < self.W)) & (sy <= self.CM[i, px] + tol * self.s), sx, sy

    def ridgelines(self, spacing, alpha_fn, width_fn, color, min_alpha=0.03, over_water=False):
        """over_water=False fades the lines out over sea and lakes (t.water), which get their own layer."""
        step = max(1, int(round(spacing / (self.vs[1] - self.vs[0]))))
        segs, cols, wids = [], [], []
        px = np.arange(self.W); color = np.array(color, float)
        wat = getattr(self.t, "water", None) if not over_water else None
        for i in range(0, len(self.vs), step):
            y = self.Y[i]; cov = np.isfinite(y)
            prev = self.CM[i - 1] if i else np.full(self.W, np.inf)
            vis = cov & (y < prev - 0.3 * self.s)
            if vis.sum() < 2: continue
            ctx = dict(i=i, px=px, depth=(self.vs[i] - self.v0) / (V1 - self.v0), u=self.u_at(i, px), v=self.vs[i])
            a = np.where(vis, alpha_fn(ctx), 0)
            if wat is not None: a = a * (1 - self.sample(wat, i, px))
            if self.thin_dense > 0 and i >= step:
                # rows that pile up on screen (the far flanks of a broad shield) are thinned instead of stacking into a slab
                gap = np.where(np.isfinite(self.Y[i - step]) & cov, self.Y[i - step] - y, 99)
                a = a * (1 - self.thin_dense * (1 - np.clip(gap / (3.0 * self.s), 0.12, 1)))
            pts = np.stack([px, np.where(cov, y, 0)], 1)
            sgm = np.stack([pts[:-1], pts[1:]], 1)
            keep = vis[:-1] & vis[1:] & (a[:-1] > min_alpha)
            if not keep.any(): continue
            c = np.zeros((keep.sum(), 4)); c[:, :3] = color; c[:, 3] = np.clip(a[:-1][keep], 0, 1)
            segs.append(sgm[keep]); cols.append(c)
            wids.append(np.broadcast_to(width_fn(ctx), (self.W,))[:-1][keep] * self.s)
        return segs, cols, wids

    def gbuffer(self, fields):
        NZ = len(self.vs); ys = np.arange(self.H, dtype=np.float32)
        owner = np.full((self.H, self.W), NZ, np.int32)
        for c in range(self.W):
            cm = self.CM[:, c]
            if np.isfinite(cm[-1]): owner[:, c] = np.searchsorted(-cm, -ys, side="left")
        below_near_edge = (owner == 0) & (ys[:, None] > self.Y[0][None, :] + 1)   # under the grid's front edge
        owner[below_near_edge] = NZ
        mask = owner < NZ; oi = np.clip(owner, 0, NZ - 1)
        pxg = np.broadcast_to(np.arange(self.W), (self.H, self.W))
        u = (pxg / self.s - self.cx) * (self.vs[oi] - ZCAM) / self.fx
        j = (u - self.us[0]) / (self.us[1] - self.us[0])
        vv = self.vs[oi]
        out = {"mask": mask, "row": oi, "u": u,
               "edge": mask * np.clip((46 - np.abs(u)) / 14, 0, 1) * np.clip((vv - self.v0) / self.nf, 0, 1) * np.clip((V1 - vv) / 8, 0, 1)}
        for k_, f in fields.items():
            out[k_] = np.where(mask, map_coordinates(f, [oi.astype(float), j], order=1, mode="nearest"), 0)
        return out

    def sky_mask(self, soften=1.0, left_fade=True, horizon_pad=40, fade_len=250):
        yy = np.arange(self.H)[:, None]; xx = np.arange(self.W)[None, :]
        sil = np.where(np.isfinite(self.silhouette), self.silhouette, self.H + 10)[None, :]
        sky = gaussian_filter((yy < sil - 1).astype(float), soften)
        if left_fade:
            f = self.xfade(xx)
            sky = 1 - (1 - sky) * f
            sky = sky * (1 - np.clip((yy / self.s - (self.hy + horizon_pad)) / fade_len, 0, 1) * (1 - f))
        return sky

    def peaks_screen(self, latlons):
        out = []
        t = self.t
        for p in latlons:
            u, v = t.to_uv(p)
            i = np.argmin(np.abs(self.vs - v)); j = np.argmin(np.abs(self.us - u))
            win = self.h[max(i - 15, 0):i + 15, max(j - 15, 0):j + 15]
            if win.size == 0: out.append(None); continue
            a, b = np.unravel_index(win.argmax(), win.shape)
            ii, jj = a + max(i - 15, 0), b + max(j - 15, 0)
            x, y = self.proj(self.us[jj], self.vs[ii], self.h[ii, jj])
            out.append(np.array([x, y]) if 0 <= x < self.W else None)
        return out


# ------------------------------------------------------------------ water layer (sea and lakes)
def iso_lines(sc, field, levels, tol=2.0, min_fade=0.03):
    """Visible iso-lines of a grid field (the coastline is t.water at 0.5), drawn on the terrain surface.
    Returns [(segments (N,2,2) output px, fade (N,), level)]."""
    import contourpy
    t = sc.t; out = []
    gen = contourpy.contour_generator(t.us, t.vs, field)
    du, dv = t.us[1] - t.us[0], t.vs[1] - t.vs[0]
    for lev in levels:
        for line in gen.lines(lev):
            u, v = line[:, 0], line[:, 1]
            hh = map_coordinates(t.h, [(v - t.vs[0]) / dv, (u - t.us[0]) / du], order=1, mode="nearest")
            vis, sx, sy = sc.visible(u, v, hh, tol=tol)
            fade = sc.xfade(sx) * sc.edge_fade(u, v)
            P = np.stack([sx, sy], 1); S = np.stack([P[:-1], P[1:]], 1)
            keep = vis[:-1] & vis[1:] & (fade[:-1] > min_fade) & (np.hypot(*(P[1:] - P[:-1]).T) < 25 * sc.s)
            if keep.any(): out.append((S[keep], fade[:-1][keep], lev))
    return out


def wave_marks(sc, G, seed=0, density=1.8, min_len=12, max_len=46):
    """
    Sparse horizontal wave strokes on visible water, spaced in screen space: a flat sea seen at a grazing
    angle is otherwise a dense block of rows. Strokes are longer and further apart towards the viewer.
    G needs a "water" field. Returns (segments (N,2,2), alpha (N,), nearness 0..1 (N,)).
    """
    s = sc.s; H, W = G["mask"].shape
    w = G["water"] * G["edge"] * sc.xfade(np.arange(W))[None, :]
    near = np.clip((V1 - sc.vs[G["row"]]) / (V1 - sc.v0), 0, 1) ** 2
    L = (min_len + (max_len - min_len) * near) * s
    gap = (7 + 16 * near) * s
    rng = np.random.default_rng(seed + 77)
    p = np.where(w > 0.6, density * 0.6 / (L * gap), 0)
    ys, xs = np.nonzero(rng.random((H, W)) < p)
    if not len(xs): return np.zeros((0, 2, 2)), np.zeros(0), np.zeros(0)
    half = L[ys, xs] * rng.uniform(0.3, 0.7, len(xs))
    x0 = np.clip(xs - half, 0, W - 1); x1 = np.clip(xs + half, 0, W - 1)
    ok = (w[ys, x0.astype(int)] > 0.5) & (w[ys, x1.astype(int)] > 0.5)
    ys, xs, x0, x1 = ys[ok], xs[ok], x0[ok], x1[ok]
    yf = ys + 0.5
    segs = np.stack([np.stack([x0, yf], 1), np.stack([x1, yf], 1)], 1)
    nr = near[ys, xs]
    return segs, w[ys, xs] * (0.35 + 0.65 * nr), nr


# ------------------------------------------------------------------ backgrounds & drawing helpers
def parchment(W, H, seed=21, tint=(0.945, 0.905, 0.815), vignette=0.13, grain=0.012):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    p = np.ones((H, W, 3)) * np.array(tint)
    lw, lh = max(W // 4, 8), max(H // 4, 8)
    blot = zoom(fbm((lh, lw), 6, 3, 0.6, seed), (H / lh, W / lw), order=1)[:H, :W]
    p += blot[..., None] * np.array([0.035, 0.04, 0.05])
    p += rng.standard_normal((H, W))[..., None] * grain
    p += gaussian_filter(rng.standard_normal((H, W)), (1.2, 3))[..., None] * 0.02
    v = ((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H / 2) / (H * 0.75)) ** 2
    return np.clip(p * (1 - vignette * np.clip(v, 0, 1.4))[..., None], 0, 1)


def hexrgb(h):
    h = h.lstrip("#"); return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


# Fonts: tried in order; the first file that exists wins. Debian/Ubuntu paths come first (the look the
# examples were made with), then other Linux layouts, macOS and Windows. Family names are the fallback.
_FONT_DIRS_CJK = [
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSerifCJK-Regular.ttc", "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/google-noto-cjk/NotoSerifCJK-Regular.ttc", "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto-cjk/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc", "/System/Library/Fonts/ヒラギノ明朝 ProN.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc", "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/Library/Fonts/Arial Unicode.ttf", "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "C:/Windows/Fonts/YuGothM.ttc", "C:/Windows/Fonts/msgothic.ttc", "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/malgun.ttf",
]
_FAMILIES_CJK = ["Noto Serif CJK JP", "Noto Sans CJK JP", "Source Han Serif", "Source Han Sans", "Hiragino Mincho ProN",
                 "Hiragino Sans", "Yu Gothic", "MS Gothic", "Microsoft YaHei", "Malgun Gothic", "Arial Unicode MS"]
_FONT_DIRS_LATIN = {True: ["/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", "/usr/share/fonts/dejavu/DejaVuSerif.ttf"],
                    False: ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"]}
_warned = set()


def _find_family(names):
    for f in names:
        try:
            return font_manager.findfont(font_manager.FontProperties(family=f), fallback_to_default=False)
        except Exception:
            continue
    return None


def font_for(text, serif=True):
    """Pick an installed font that can render the script used in `text` (Linux, macOS or Windows)."""
    home = os.path.expanduser("~")
    if re.search(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]", text):
        cands = _FONT_DIRS_CJK + [os.path.join(home, "Library/Fonts", n) for n in ("NotoSansCJKjp-Regular.otf", "NotoSerifCJKjp-Regular.otf")]
        fams = _FAMILIES_CJK
    else:
        cands = _FONT_DIRS_LATIN[serif]
        fams = ["DejaVu Serif"] if serif else ["DejaVu Sans"]            # matplotlib ships DejaVu itself
    for c in cands:
        if os.path.exists(c): return font_manager.FontProperties(fname=c)
    f = _find_family(fams)
    if f: return font_manager.FontProperties(fname=f)
    if fams is _FAMILIES_CJK and "cjk" not in _warned:
        _warned.add("cjk")
        print("warning: no CJK font found (looked for Noto CJK, Hiragino, Yu Gothic, MS Gothic ...); "
              "Chinese/Japanese/Korean labels may render as boxes. Install fonts-noto-cjk or pass Latin labels.",
              file=sys.stderr, flush=True)
    return font_manager.FontProperties(family="serif" if serif else "sans-serif")


def figure(bg):
    H, W = bg.shape[:2]
    fig = plt.figure(figsize=(W / 200, H / 200), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    ax.imshow(bg, extent=(0, W, H, 0), interpolation="bilinear")
    return fig, ax


def add_lines(ax, segs, cols, wids, scale=0.72, cap="round"):
    for sg, c, w in zip(segs, cols, wids):
        ax.add_collection(LineCollection(sg, colors=c, linewidths=np.asarray(w) * scale, capstyle=cap))


def fig_to_array(fig):
    fig.canvas.draw()
    a = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(float) / 255
    plt.close(fig); return a


def draw_sun(img, sc, c, r, color, alpha=0.42, glow=0.12, hatch=9):
    H, W = img.shape[:2]; yy, xx = np.mgrid[0:H, 0:W]; s = sc.s
    d = np.hypot(xx - c[0], yy - c[1]); above = sc.sky_mask()
    m = (np.clip((r - d) / 2, 0, 1) * alpha + np.exp(-((d - r).clip(0) / (140 * s)) ** 2) * glow) * above
    img = img * (1 - m[..., None]) + np.array(color) * m[..., None]
    if hatch:
        band = (np.mod(yy - c[1], hatch * s) < 1.2 * s) & (d < r) & (above > 0.5)
        img[band] *= 0.93
    return np.clip(img, 0, 1)


def draw_labels(ax, sc, peaks, pts, color, alpha=0.85):
    s = sc.s
    sil = np.where(np.isfinite(sc.silhouette), sc.silhouette, sc.H + 10)
    for k, (pk, p) in enumerate(zip(peaks, pts)):
        if p is None or not pk.get("name"): continue
        if sc.xfade(p[0]) < 0.35: continue                              # peak sits in the faded-out left zone
        side = 1 if k % 2 == 0 else -1
        ha = "left" if side > 0 else "right"
        # lift the label above the skyline when terrain would run through the text (canyon rims, ranges)
        w = 0.62 * 36 * s * max(len(pk["name"]), 0.75 * len(pk.get("sub", "")))
        x0, x1 = (p[0] + 20 * s, p[0] + 40 * s + w) if side > 0 else (p[0] - 40 * s - w, p[0] - 20 * s)
        span = sil[int(np.clip(x0, 0, sc.W - 1)):int(np.clip(x1, 1, sc.W))]
        lift = max(0.0, (p[1] - 30 * s) - (span.min() - 14 * s)) if span.size else 0.0
        lift = min(lift, max(0.0, p[1] - 110 * s))                     # never off the top of the frame
        ax.text(p[0] + 30 * s * side, p[1] - 70 * s - lift, pk["name"], fontsize=13 * s, ha=ha, color=color, alpha=alpha, fontproperties=font_for(pk["name"]))
        if pk.get("sub"):
            ax.text(p[0] + 30 * s * side, p[1] - 40 * s - lift, pk["sub"], fontsize=8.5 * s, ha=ha, color=color, alpha=alpha, fontproperties=font_for(pk["sub"]))
        ax.plot([p[0] + 12 * s * side, p[0] + 26 * s * side], [p[1] - 12 * s, p[1] - 38 * s - lift], color=color, lw=0.6 * s, alpha=0.6)


def footnote(ax, sc, text, color, alpha=0.6):
    if text:
        ax.text(sc.W - 60 * sc.s, sc.H - 55 * sc.s, text, fontsize=7.5 * sc.s, ha="right", color=color, alpha=alpha, fontproperties=font_for(text))


def sun_anchor(pts, s):
    pts = [p for p in pts if p is not None]
    if len(pts) >= 2: return (pts[0] + pts[1]) / 2
    if pts: return pts[0] + np.array([-420 * s, 120 * s])
    return np.array([1800 * s, 300 * s])


def label_spans(sc, peaks, pts):
    """x-intervals (output px) that peak labels occupy, so a relocated sun can keep clear of them."""
    s, out = sc.s, []
    for k, (pk, p) in enumerate(zip(peaks, pts)):
        if p is None or not pk.get("name"): continue
        out.append((p[0] - 60 * s, p[0] + 360 * s) if k % 2 == 0 else (p[0] - 360 * s, p[0] + 60 * s))
    return out


def sun_spot(sc, c, r, peak=None, target=0.72, lo=0.4, hi=1.0, jag_max=0.22, avoid=()):
    """
    Check a sun/moon centre `c` (output px, radius r) against the skyline and move it if it is buried.
    The legacy placements (saddle between two peaks, or down-left of a lone peak) are kept whenever
    at least `lo` of the disc is visible and the skyline across it is smooth, so cones render
    exactly as before. Otherwise candidates along the skyline are scored: visible fraction close to
    `target`, a calm skyline under the disc, away from the summit label, close to the legacy spot.
    """
    s, W, H = sc.s, sc.W, sc.H
    sil = np.where(np.isfinite(sc.silhouette), sc.silhouette, H + 10)
    fade = sc.xfade(np.arange(W))
    ang = np.linspace(0, 2 * np.pi, 48, endpoint=False); rad = np.sqrt(np.linspace(0.02, 1, 10))
    A, Rr = np.meshgrid(ang, rad); ox, oy = (Rr * np.cos(A)).ravel(), (Rr * np.sin(A)).ravel()

    def score(x, y):
        px = np.clip((x + ox * r).astype(int), 0, W - 1); py = y + oy * r
        vis = float(np.mean((py < sil[px]) | (fade[px] < 0.05)))
        span = sil[int(np.clip(x - r, 0, W - 1)):int(np.clip(x + r, 1, W))]
        span = span[span < H]
        jag = float(np.std(np.diff(span)) * 6 / r) if span.size > 3 else 0.0      # skyline wiggle under the disc
        return vis, jag
    c = np.asarray(c, float)
    vis, jag = score(*c)
    if lo <= vis <= hi and jag <= jag_max and c[1] - r > -0.2 * r:
        return c
    best, bs = c, -1e9
    x0 = max((sc.fadeX + 250) * s, r + 30 * s)
    for x in np.arange(x0, W - r - 30 * s, r / 3):
        if peak is not None and abs(x - peak[0]) < r + 110 * s: continue
        if any(x + r > a0 and x - r < a1 for a0, a1 in avoid): continue
        seg = sil[int(max(x - r, 0)):int(min(x + r, W))]
        seg = seg[seg < H]
        if not seg.size: continue
        yb = float(np.median(seg))
        for f in (-1.3, -0.9, -0.5, -0.2, 0.1):
            y = yb + f * r
            if y - r < 0.05 * H or y + r > 0.92 * H: continue
            v, j = score(x, y)
            sc_ = -3 * abs(v - target) - 1.5 * max(j - 0.1, 0) - 0.6 * abs(x - c[0]) / W - 0.4 * abs(y - c[1]) / H
            if v < lo: sc_ -= 2
            if sc_ > bs: best, bs = np.array([x, y]), sc_
    return best


def save_png(arr_or_fig, path, preview=None, size=None):
    if isinstance(arr_or_fig, np.ndarray):
        im = Image.fromarray((np.clip(arr_or_fig, 0, 1) * 255).astype(np.uint8))
    else:
        im = Image.fromarray((fig_to_array(arr_or_fig) * 255).astype(np.uint8))
    if size and im.size != size: im = im.resize(size, Image.LANCZOS)
    if path.lower().endswith((".jpg", ".jpeg")): im.save(path, quality=93)
    else: im.save(path, optimize=True)
    if preview: im.resize((1200, int(1200 * im.size[1] / im.size[0])), Image.LANCZOS).save(preview)
    return path
