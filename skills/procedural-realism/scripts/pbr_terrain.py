#!/usr/bin/env python3
"""
pbr_terrain.py — photographic-style render of any real mountain, numpy only.

  terrain  real elevation (AWS Terrain Tiles) + procedural sub-pixel erosion detail + Earth curvature
  march    adaptive ray march over the heightfield with bisection refinement
  sky      single-scattering Rayleigh + Mie atmosphere: sky colour, sun colour, aerial perspective
           (--sky physical adds Earth shadow, ozone and a sky-derived ambient: needed for twilight)
  light    sun with soft terrain shadows, sky ambient with height-based occlusion, ground bounce
  surface  procedural fields / steppe / rock / snow by height, slope and noise
  water    optional lake/sea plane: multi-octave filtered wave normals, Fresnel reflection traced
           back into the terrain + sky, refraction into a tinted shallow bed, sun glint
  near     optional instanced trees (conifer / deciduous / poplar) and villages (houses with
           pitched roofs), ray-cast exactly inside their screen bounding boxes, with shadows
  extras   patchy valley mist (optional), volumetric lenticular cloud over the summit (optional,
           classic or multiple-scattering model), cumulus / stratus layer (optional)
  night    optional city: street-grid emitters + lit windows + sky glow, written as a light list
           that post_terrain.py splats with depth test and thin-lens bokeh
  post     `post_terrain.py`: lights, depth of field, bloom, ACES filmic, vignette, grain

Rendering is split into row bands saved as .npy, so long renders survive time-limited shells:
  python pbr_terrain.py --peak 39.7019,44.2986 --from 40.1792,44.4991 --size 1500x500 --rows 0:250
  python pbr_terrain.py ... --rows 250:500          # continue; existing bands are skipped
  python post_terrain.py --bands <workdir>/bands_1500x500 --out ararat.png
Missing rows (an interrupted run, or a --rows chunk that ended mid-band) are found and filled on the next run.
Water, trees, villages, city, cloud layer, moonlight and depth of field are all off by default.
"""
import argparse, math, os, re, sys, time, json, hashlib
sys.dont_write_bytecode = True
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter, label, distance_transform_edt, maximum_filter, minimum_filter

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--peak", required=True, help="summit lat,lon")
ap.add_argument("--from", dest="view_from", required=True, help="viewpoint lat,lon (the camera sits on the line from here to the peak; see --cam-offset)")
ap.add_argument("--size", default="1500x500"); ap.add_argument("--rows", default=None, help="a:b band range to render now")
ap.add_argument("--work", default="./pbr_work"); ap.add_argument("--dem", help="cached .npz (elev, lats, lons)")
ap.add_argument("--tag", default="", help="suffix for the band folder, so variants of one view don't collide")
ap.add_argument("--cam-height", type=float, default=0.30, help="km above ground")
ap.add_argument("--cam-offset", type=float, default=None, help="km to move the camera toward (+) / away (-) from the peak; default auto-frames")
ap.add_argument("--hfov", type=float, default=34.0, help="horizontal field of view, degrees")
ap.add_argument("--layout", default="right", choices=["right", "center", "left"], help="where the summit sits in frame")
ap.add_argument("--pitch", type=float, default=0.0, help="extra camera pitch, degrees (+ = up)")
ap.add_argument("--horizon", type=float, default=None, help="put the true horizon at this fraction of the height from the top (overrides the auto pitch; --pitch still adds)")
ap.add_argument("--second", default=None, help="lat,lon of a second peak to frame together with the main one")
ap.add_argument("--sun-az", type=float, default=None, help="compass azimuth; default lights the faces the camera sees")
ap.add_argument("--sun-el", type=float, default=None, help="degrees above horizon (5-15 = golden, 30+ = midday; default 10, or set by --time)")
ap.add_argument("--time", default=None, choices=["day", "golden", "blue", "night"],
                help="shortcut: day 35°, golden 6°, blue -4.5°, night -14° sun; blue/night switch on --sky physical")
ap.add_argument("--sky", default=None, choices=["classic", "physical"],
                help="classic = plain daylight sky; physical adds Earth shadow, ozone, twilight and a sky-derived ambient")
ap.add_argument("--snowline", type=float, default=None, help="metres; default 68%% of the way from plain to peak")
ap.add_argument("--base", type=float, default=None, help="plain elevation, metres (default: auto)")
ap.add_argument("--haze", type=float, default=0.22, help="ground aerosol (0.1 crisp winter ... 1.5 dusty summer)")
ap.add_argument("--air", type=float, default=0.5, help="Rayleigh density multiplier (1 = sea-level standard)")
ap.add_argument("--mist", type=float, default=3.2, help="valley mist density (0 = off)")
ap.add_argument("--cloud", type=int, default=1, help="lenticular cloud cap over the summit (1/0)")
ap.add_argument("--cloud-model", default="classic", choices=["classic", "ms"],
                help="lenticular lighting: classic single scatter, or ms = octave multiple scattering + dual-lobe phase")
ap.add_argument("--cloud-layer", default="none", choices=["none", "cumulus", "stratus"], help="extra cloud layer")
ap.add_argument("--cloud-cover", type=float, default=0.45, help="cloud layer coverage 0..1")
ap.add_argument("--biome", default="arid", choices=["arid", "temperate"], help="ground palette: arid steppe/fields (default) or temperate forest/meadow")
ap.add_argument("--treeline", type=float, default=None, help="temperate forest limit, metres (default 55%% of the way up)")
ap.add_argument("--season", default="summer", choices=["summer", "autumn", "spring"], help="temperate foliage colour")
ap.add_argument("--radius", type=float, default=45.0, help="terrain radius around the summit, km")
# water
ap.add_argument("--water-level", default=None, help="lake/sea surface in metres, or 'auto' (largest flat basin near the camera)")
ap.add_argument("--water-seed", default=None, help="lat,lon inside the lake (default: the auto basin, or the lowest point in view)")
ap.add_argument("--waves", type=float, default=1.0, help="wave strength: 0 mirror, 0.5 calm lake, 1 breeze, 3 choppy")
ap.add_argument("--wind", type=float, default=None, help="wave travel azimuth, degrees (default: across the view)")
ap.add_argument("--water-depth", type=float, default=15.0, help="max depth of the synthetic bed, metres")
ap.add_argument("--water-spp", type=int, default=3, help="reflection rays per water pixel")
ap.add_argument("--water-color", default="lake", choices=["lake", "clear", "sea"], help="absorption / scattering preset")
# near field
ap.add_argument("--trees", type=float, default=0.0, help="tree density 0..1 (0 = off)")
ap.add_argument("--tree-mix", default="auto", help="conifer,deciduous,poplar weights, e.g. 0.2,0.6,0.2 (auto from elevation)")
ap.add_argument("--villages", type=float, default=0.0, help="village density 0..1 (0 = off)")
ap.add_argument("--near-km", type=float, default=6.0, help="trees and houses are placed only this far from the camera")
ap.add_argument("--seed", type=int, default=1, help="seed for trees / villages / city layout")
# night
ap.add_argument("--city", type=float, default=0.0, help="night city density 0..1: street lights, lit windows, sky glow (0 = off)")
ap.add_argument("--city-km", type=float, default=12.0, help="city lights extend this far from the camera")
ap.add_argument("--geom-cache", action="store_true", help="cache sun-independent geometry per band (speeds up sun-angle timelapses)")
ap.add_argument("--near-lamps", type=int, default=0, help="with --city: a string of this many lamps 3.5-6 m in front of the lens (shows bokeh)")
ap.add_argument("--aperture", type=float, default=0.0, help="thin-lens aperture diameter in mm for depth of field / bokeh (0 = pinhole)")
ap.add_argument("--focus", type=float, default=None, help="focus distance km (default: the peak)")
ap.add_argument("--moon", type=float, default=None, help="with --time night: moonlight, 1 = full moon (default), 0.3 = crescent, 0 = starlight only")
ap.add_argument("--moon-az", type=float, default=None, help="moon azimuth, degrees (default: lights the faces the camera sees)")
ap.add_argument("--moon-el", type=float, default=28.0, help="moon elevation, degrees")
a = ap.parse_args()

TIME_EL = {"day": 35.0, "golden": 6.0, "blue": -4.5, "night": -14.0}
if a.sun_el is None: a.sun_el = TIME_EL[a.time] if a.time else 10.0
if a.sky is None: a.sky = "physical" if (a.time in ("blue", "night") or a.sun_el < 1.5 or a.city > 0) else "classic"
PHYS = a.sky == "physical"

W, H = map(int, a.size.lower().split("x"))
os.makedirs(a.work, exist_ok=True)
BANDS = os.path.join(a.work, f"bands_{W}x{H}" + (f"_{a.tag}" if a.tag else "")); os.makedirs(BANDS, exist_ok=True)
AUX = os.path.join(BANDS, "aux")
LA0, LO0 = map(float, a.peak.split(",")); VLA, VLO = map(float, a.view_from.split(","))
KX, KY = 111.32 * math.cos(math.radians(LA0)), 110.57
RE = 6371.0 * 7 / 6

# every argument that changes pixels is recorded; resuming with different ones would mix two pictures.
# --aperture / --focus are post effects (post_terrain.py can override them); only "is there a lens" matters here.
_pkeys = {k: v for k, v in vars(a).items() if k not in ("rows", "work", "dem", "tag", "geom_cache", "aperture", "focus")}
_pkeys["depth_aux"] = a.aperture > 0
_phash = hashlib.md5(json.dumps(_pkeys, sort_keys=True).encode()).hexdigest()[:10]
_pfile = os.path.join(BANDS, "params.json"); _rfile = os.path.join(BANDS, "refused.json")
if os.path.exists(_pfile):
    old = json.load(open(_pfile))
    if old.get("hash") != _phash and any(f.endswith(".npy") for f in os.listdir(BANDS)):
        diff = {k: (old["args"].get(k), v) for k, v in _pkeys.items() if old["args"].get(k) != v}
        json.dump({"refused_args": _pkeys, "diff": {k: list(v) for k, v in diff.items()}}, open(_rfile, "w"), indent=1)
        sys.exit(f"{BANDS} holds bands rendered with other settings {diff}; use --tag or delete the folder "
                 "(post_terrain.py will refuse this folder until a run with matching settings)")
json.dump({"hash": _phash, "args": _pkeys}, open(_pfile, "w"), indent=1)
if os.path.exists(_rfile): os.remove(_rfile)

# ------------------------------------------------------------------ elevation data
from dem import fetch_dem, load_dem  # shared with procedural-terrain-art

VIEW = np.array([(VLO - LO0) * KX, (VLA - LA0) * KY])                 # viewpoint, km E/N of summit
DIST = float(np.hypot(*VIEW))
HF_CACHE = os.path.join(a.work, "hf.npz")
RES = 0.04
NEED = max(a.radius, DIST + 8, 55)
# the heightfield cache belongs to one place: a different peak, radius or DEM file rebuilds it
SITE = dict(peak=[round(LA0, 5), round(LO0, 5)], need=round(NEED, 3), dem=os.path.abspath(a.dem) if a.dem else None)
META = None
if os.path.exists(HF_CACHE):
    z_ = np.load(HF_CACHE)
    _m = json.loads(str(z_["meta"]))
    if _m.get("site") == SITE:
        HF, GE, GN, META = z_["hf"], z_["ge"], z_["gn"], _m
    else:
        print("heightfield cache in --work is for another place or radius; rebuilding it", flush=True)
if META is None:
    need = NEED
    dem_path = a.dem or os.path.join(a.work, "dem.npz")
    if not a.dem and os.path.exists(dem_path):
        # a default dem.npz left by another mountain: refetch if it doesn't cover this one
        _d = np.load(dem_path)
        _la, _lo = _d["lats"], _d["lons"]
        _half = min((min(_la[0] - LA0, LA0 - _la[-1])) * KY, (min(LO0 - _lo[0], _lo[-1] - LO0)) * KX)
        if _half < need * 0.9:
            print("dem.npz in --work doesn't cover this peak; fetching a new one", flush=True); os.remove(dem_path)
    if not os.path.exists(dem_path):
        print(f"fetching elevation (radius {need * 1.05:.0f} km)…", flush=True); fetch_dem(LA0, LO0, need * 1.05, dem_path)
    e, la, lo = load_dem(dem_path)
    Rg = min(need, (la[0] - la[-1]) * KY / 2 * 0.98, (lo[-1] - lo[0]) * KX / 2 * 0.98)
    e0, e1, n0, n1 = -Rg, Rg, -Rg, Rg
    GE, GN = np.arange(e0, e1, RES), np.arange(n0, n1, RES)
    EE, NN = np.meshgrid(GE, GN)
    ii = np.interp(LA0 + NN / KY, la[::-1], np.arange(len(la))[::-1].astype(float))
    jj = (LO0 + EE / KX - lo[0]) / (lo[1] - lo[0])
    HF = gaussian_filter(np.maximum(map_coordinates(e, [ii, jj], order=1, mode="nearest"), 0) / 1000.0, 0.7).astype(np.float32)
    ci, cj = np.searchsorted(GN, 0), np.searchsorted(GE, 0)
    win = HF[ci - 40:ci + 40, cj - 40:cj + 40]
    peak = float(win.max())
    ring = HF[np.hypot(EE, NN) > min(a.radius, 40) * 0.5]
    base = float(np.percentile(ring, 20))
    META = dict(peak=peak, base=base, site=SITE)
    np.savez(HF_CACHE, hf=HF, ge=GE, gn=GN, meta=json.dumps(META))

PEAK = META["peak"]; BASE = (a.base / 1000) if a.base is not None else META["base"]
REL = PEAK - BASE
SNOW = (a.snowline / 1000) if a.snowline is not None else BASE + 0.68 * REL

def gsamp(G, x, y, order=1):
    return map_coordinates(G, [(y - GN[0]) / RES, (x - GE[0]) / RES], order=order, mode="nearest")
def to_xy(latlon):
    la_, lo_ = map(float, latlon.split(",")); return np.array([(lo_ - LO0) * KX, (la_ - LA0) * KY])

toPeak = -VIEW / DIST
OFFSET = a.cam_offset if a.cam_offset is not None else DIST - REL / 0.095
OFFSET = max(OFFSET, DIST - (min(-GE[0], GE[-1], -GN[0], GN[-1]) - 3))     # stay on the terrain grid
CAM_XY = VIEW + toPeak * OFFSET

# ------------------------------------------------------------------ water body (optional)
WATER = a.water_level is not None
if WATER:
    ci, cj = (CAM_XY[1] - GN[0]) / RES, (CAM_XY[0] - GE[0]) / RES
    yy_, xx_ = np.mgrid[0:HF.shape[0], 0:HF.shape[1]]
    near = np.hypot(yy_ - ci, xx_ - cj) * RES < 12
    if a.water_level == "auto":
        # A lake in the DEM is flat from shore to shore (within a metre or two). Irrigated plains pass the local
        # flatness test too, but over their extent they rise by tens of metres, and flooding them spills far beyond
        # the flat patch. Both are checked, so a dry plain is not turned into a lake.
        rng_ = maximum_filter(HF, 3) - minimum_filter(HF, 3)
        flat_all = rng_ < 0.0012
        flat = flat_all & near
        lab, nl = label(flat); sz = np.bincount(lab.ravel()); sz[0] = 0
        why = None
        if a.water_seed:
            s = to_xy(a.water_seed); si, sj = int(round((s[1] - GN[0]) / RES)), int(round((s[0] - GE[0]) / RES))
            r_ = int(0.5 / RES); w_ = lab[max(si - r_, 0):si + r_ + 1, max(sj - r_, 0):sj + r_ + 1]
            if 0 <= si < lab.shape[0] and 0 <= sj < lab.shape[1] and lab[si, sj] > 0:
                k = int(lab[si, sj])
            elif w_.size and w_.max() > 0:                               # snap to flat water within 500 m
                wi_, wj_ = np.nonzero(w_); c_ = np.argmin(np.hypot(wi_ - min(si, r_), wj_ - min(sj, r_)))
                k = int(w_[wi_[c_], wj_[c_]])
            else:
                k = 0; why = f"--water-seed {a.water_seed} is not on a flat water surface in the elevation data (none within 500 m)"
        else:
            k = int(sz.argmax())
        if why is None and sz[k] * RES * RES < 0.1:
            why = "no flat basin of 0.1 km² within 12 km of the camera"
        if why is None:
            WL = float(np.median(HF[lab == k])) + 0.0012
            seed_ij = np.argwhere(lab == k)[0]
            flab, _ = label(HF <= WL); fl_ = flab == flab[tuple(seed_ij)]
            alab, _ = label(flat_all); fa_ = alab == alab[tuple(seed_ij)]             # the whole flat patch, not just its near part
            f_km2, a_km2 = fl_.sum() * RES * RES, fa_.sum() * RES * RES
            hv_ = HF[fa_]; spread = float(np.percentile(hv_, 95) - np.percentile(hv_, 5)) * 1000
            if spread > 3.0:
                why = (f"the flattest patch near the camera ({a_km2:.0f} km² around {WL * 1000:.0f} m) rises {spread:.0f} m "
                       "across its extent; a lake surface in the elevation data is level to within about 2 m")
            elif f_km2 > 3 * a_km2:
                why = (f"a lake at {WL * 1000:.0f} m would flood {f_km2:.0f} km², far beyond the {a_km2:.1f} km² flat patch; "
                       "the basin isn't closed")
            del flab, fl_, alab, fa_, hv_
        if why:
            print(f"--water-level auto: no lake found ({why}). Rendering without water. For a specific lake or the "
                  "sea give --water-level <metres> --water-seed lat,lon.", flush=True)
            WATER = False
    else:
        WL = float(a.water_level) / 1000
        if a.water_seed:
            s = to_xy(a.water_seed); seed_ij = (int(round((s[1] - GN[0]) / RES)), int(round((s[0] - GE[0]) / RES)))
        else:
            seed_ij = np.unravel_index(np.argmin(np.where(near, HF, 99)), HF.shape)
if WATER:
    lab, _ = label(HF <= WL)
    k = lab[tuple(seed_ij)]
    if k == 0: sys.exit(f"water seed is above the water level {WL * 1000:.0f} m")
    WMASK = lab == k
    WDEPTH = (a.water_depth / 1000) * (1 - np.exp(-distance_transform_edt(WMASK) * RES / 0.35))
    HF = np.where(WMASK, np.float32(WL), HF).astype(np.float32)
    WM = gaussian_filter(WMASK.astype(np.float32), 0.6)
    WDEPTH = gaussian_filter(WDEPTH.astype(np.float32), 1.0)
    print(f"water: level {WL * 1000:.0f} m, {WMASK.sum() * RES * RES:.1f} km²"
          + (f" (auto: level to {spread:.1f} m over {a_km2:.1f} km²)" if a.water_level == "auto" else ""), flush=True)
if a.water_level is not None:
    del yy_, xx_, near

def hf(x, y):
    return map_coordinates(HF, [(y - GN[0]) / RES, (x - GE[0]) / RES], order=1, mode="nearest")
HFB = gaussian_filter(HF, 25)

# ------------------------------------------------------------------ noise
_r = np.random.default_rng(7)
PERM = np.concatenate([_r.permutation(4096)] * 2).astype(np.int64); RND = _r.random(4096)
def _h2(ix, iy): return RND[PERM[(PERM[ix & 4095] + iy) & 4095]]
def vnoise2(x, y):
    ix, iy = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64); fx, fy = x - ix, y - iy
    ux, uy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    A, B, C, D = _h2(ix, iy), _h2(ix + 1, iy), _h2(ix, iy + 1), _h2(ix + 1, iy + 1)
    return (A + (B - A) * ux) * (1 - uy) + (C + (D - C) * ux) * uy
def fbm2(x, y, o=5):
    s, amp, n = 0.0, 0.5, 0.0
    for _ in range(o):
        s = s + amp * vnoise2(x, y); n += amp; x, y = x * 2.03 + 17.3, y * 2.03 - 9.1; amp *= 0.5
    return s / n
def _h3(ix, iy, iz): return RND[PERM[(PERM[(PERM[ix & 4095] + iy) & 4095] + iz) & 4095]]
def vnoise3(x, y, z):
    ix, iy, iz = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64), np.floor(z).astype(np.int64)
    fx, fy, fz = x - ix, y - iy, z - iz
    ux, uy, uz = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy), fz * fz * (3 - 2 * fz)
    L = lambda dx, dy, dz: _h3(ix + dx, iy + dy, iz + dz)
    x00 = L(0,0,0) + (L(1,0,0) - L(0,0,0)) * ux; x10 = L(0,1,0) + (L(1,1,0) - L(0,1,0)) * ux
    x01 = L(0,0,1) + (L(1,0,1) - L(0,0,1)) * ux; x11 = L(0,1,1) + (L(1,1,1) - L(0,1,1)) * ux
    return (x00 + (x10 - x00) * uy) * (1 - uz) + (x01 + (x11 - x01) * uy) * uz
def fbm3(x, y, z, o=4):
    s, amp, n = 0.0, 0.5, 0.0
    for _ in range(o):
        s = s + amp * vnoise3(x, y, z); n += amp; x, y, z = x * 2.02 + 5.2, y * 2.03 - 1.7, z * 2.01 + 3.3; amp *= 0.5
    return s / n
def hash2(i, j, salt=0):
    """deterministic uniform [0,1) per integer cell"""
    h = (np.asarray(i, np.int64) * 73856093) ^ (np.asarray(j, np.int64) * 19349663) ^ (salt * 83492791 + a.seed * 2654435761)
    h = (h ^ (h >> 13)) * 1274126177
    return ((h ^ (h >> 16)) & 0xFFFFFF) / float(0x1000000)

# ------------------------------------------------------------------ camera & sun
CAM = np.array([CAM_XY[0], CAM_XY[1], float(hf(np.array([CAM_XY[0]]), np.array([CAM_XY[1]]))[0]) + a.cam_height])
if a.second:
    sla, slo = map(float, a.second.split(",")); S2 = np.array([(slo - LO0) * KX, (sla - LA0) * KY])
    tgt_xy = S2 / 2
else:
    tgt_xy = np.zeros(2)
look_at = np.array([tgt_xy[0], tgt_xy[1], BASE + 0.35 * REL])
fwd = look_at - CAM; fwd /= np.linalg.norm(fwd)
# layout: rotate so the target sits at 1/3 or 2/3 of the frame
yaw = {"right": 1, "center": 0, "left": -1}[a.layout] * math.radians(a.hfov) * 0.2
c_, s_ = math.cos(yaw), math.sin(yaw)
fl = 0.5 * W / math.tan(math.radians(a.hfov) / 2)
if a.horizon is not None:
    hz = math.hypot(fwd[0], fwd[1]); fwd = np.array([fwd[0] / hz, fwd[1] / hz, (a.horizon * H - 0.52 * H) / fl])
fwd = np.array([fwd[0] * c_ - fwd[1] * s_, fwd[0] * s_ + fwd[1] * c_, fwd[2] + math.radians(a.pitch)]); fwd /= np.linalg.norm(fwd)
right = np.cross(fwd, [0, 0, 1]); right /= np.linalg.norm(right); up = np.cross(right, fwd)
cam_bearing = math.degrees(math.atan2(VIEW[0], VIEW[1])) % 360            # from summit to camera
SUN_AZ = math.radians(a.sun_az if a.sun_az is not None else (cam_bearing + 80) % 360)
SUN_EL = math.radians(a.sun_el)
SUN = np.array([math.sin(SUN_AZ) * math.cos(SUN_EL), math.cos(SUN_AZ) * math.cos(SUN_EL), math.sin(SUN_EL)])
# --time night: a full moon lights terrain and sky (physical sky only); stars go to a separate layer that post
# adds after bloom, so they stay points
NIGHT = PHYS and SUN_EL < 0.05 and a.time == "night"
MOON_ON = NIGHT and (a.moon is None or a.moon > 0)
if MOON_ON:
    MOON_AZ = math.radians(a.moon_az if a.moon_az is not None else (cam_bearing + 80) % 360); MOON_EL = math.radians(a.moon_el)
    MOON = np.array([math.sin(MOON_AZ) * math.cos(MOON_EL), math.cos(MOON_AZ) * math.cos(MOON_EL), math.sin(MOON_EL)])
    MOON_TINT = np.array([0.82, 0.93, 1.15])                                  # rendered cool, the way night photos read
PIXA = 1.0 / fl                                                            # radians per pixel
GEOM_DIR = None
if a.geom_cache:                                                           # keyed by everything that moves geometry
    _gk = {k: v for k, v in _pkeys.items() if k in ("peak", "view_from", "size", "cam_height", "cam_offset", "hfov", "layout", "pitch",
           "horizon", "second", "base", "radius", "water_level", "water_seed", "trees", "tree_mix", "villages", "near_km", "seed",
           "city", "city_km", "biome", "treeline", "season")}
    GEOM_DIR = os.path.join(a.work, "geom_" + hashlib.md5(json.dumps(_gk, sort_keys=True).encode()).hexdigest()[:10])

def terrain_z(x, y):
    return hf(x, y) - ((x - CAM[0]) ** 2 + (y - CAM[1]) ** 2) / (2 * RE)

def detail(x, y, sw):
    r = 1 - np.abs(2 * fbm2(x * 9.0, y * 9.0, 4) - 1)
    return (r ** 2 - 0.35) * 0.018 * sw + (fbm2(x * 40, y * 40, 3) - 0.5) * 0.004

# ------------------------------------------------------------------ atmosphere
RP, RA, HR, HM = 6360.0, 6420.0, 8.0, 1.2
BR = np.array([5.8e-3, 13.5e-3, 33.1e-3]) * a.air
BM = np.array([21e-3] * 3); BMe = BM * 1.1
BO = np.array([0.650e-3, 1.881e-3, 0.085e-3])                             # ozone absorption, per km at peak
SUN_I, G = 22.0, 0.78
FOG, FOG_Z, FOG_H = a.mist, BASE + 0.05, 0.06

def airmass(cz):
    zd = np.degrees(np.arccos(np.clip(cz, -1, 1)))
    return 1.0 / (np.clip(cz, 0, 1) + 0.50572 * np.clip(96.07995 - zd, 0.05, None) ** -1.6364)
def sun_T_classic(alt):
    m = airmass(math.sin(SUN_EL))
    return np.exp(-(BR[None] * (HR * np.exp(-alt / HR) * m)[..., None] + BMe[None] * (HM * np.exp(-alt / HM) * m * a.haze)[..., None]))

# physical: transmittance to the sun through a spherical atmosphere, with the Earth's shadow and ozone
LUT_H = np.linspace(0, RA - RP, 61); LUT_MU = np.linspace(-0.45, 1.0, 581)
def _build_sun_lut():
    r = (RP + LUT_H)[:, None, None]; mu = LUT_MU[None, :, None]
    t_top = -r * mu + np.sqrt(np.maximum(r * r * (mu * mu - 1) + RA * RA, 0))
    s = t_top * np.linspace(0, 1, 160)[None, None] ** 2
    alt = np.maximum(np.sqrt(r * r + s * s + 2 * r * s * mu) - RP, 0)
    ds = np.diff(s, axis=2)
    trap = lambda f: np.sum((f[..., 1:] + f[..., :-1]) * 0.5 * ds, 2)
    odR, odM = trap(np.exp(-alt / HR)), trap(np.exp(-alt / HM))
    odO = trap(np.clip(1 - np.abs(alt - 25) / 15, 0, 1))
    T = np.exp(-(BR * odR[..., None] + BMe * a.haze * odM[..., None] + BO * odO[..., None]))
    mu_h = -np.sqrt(np.maximum(1 - RP ** 2 / (RP + LUT_H) ** 2, 0))[:, None]
    f = np.clip((LUT_MU[None] - mu_h) / 0.0094 + 0.5, 0, 1); f = f * f * (3 - 2 * f)       # sun disc sinking
    return (T * f[..., None]).astype(np.float32)
if PHYS: SUN_LUT = _build_sun_lut()
def sun_T_lut(alt, mu_s):
    ia = np.clip(alt / (LUT_H[1] - LUT_H[0]), 0, len(LUT_H) - 1)
    im = np.clip((mu_s - LUT_MU[0]) / (LUT_MU[1] - LUT_MU[0]), 0, len(LUT_MU) - 1)
    return np.stack([map_coordinates(SUN_LUT[..., c], [ia.ravel(), im.ravel()], order=1, mode="nearest").reshape(np.shape(alt))
                     for c in range(3)], -1)
def sun_T(alt):
    if PHYS: return sun_T_lut(np.asarray(alt, float), np.full(np.shape(alt), math.sin(SUN_EL)))
    return sun_T_classic(alt)

def ph_r(mu): return 3 / (16 * np.pi) * (1 + mu * mu)
def ph_m(mu): return 3 / (8 * np.pi) * ((1 - G * G) * (1 + mu * mu)) / ((2 + G * G) * (1 + G * G - 2 * G * mu) ** 1.5)
def hg(mu, g): return (1 - g * g) / (4 * np.pi * (1 + g * g - 2 * g * mu) ** 1.5)

NIGHT_SKY = np.array([0.55, 0.75, 1.25]) * 4e-5                           # airglow + starlight floor (physical sky)
GLOW = None                                                                # city sky-glow field, set up by the city block

def atmosphere(dirs, tmax, n=28, O=None, light=None):
    """in-scattered light and transmittance along rays from O (default the camera) up to tmax.
    light=(direction, intensity) scatters a second source (the moon) instead of the sun, without the night floor."""
    LD, LI = (SUN, SUN_I) if light is None else light[:2]
    N = len(dirs); mu = dirs @ LD
    if O is None:
        oz = RP + CAM[2]; ox = oy = 0.0; b = dirs[:, 2] * oz; cc = oz * oz - RA * RA; O_ = CAM[None]
    else:
        ox, oy, oz = (O[:, 0] - CAM[0])[:, None], (O[:, 1] - CAM[1])[:, None], (RP + O[:, 2])[:, None]
        b = dirs[:, 0] * ox[:, 0] + dirs[:, 1] * oy[:, 0] + dirs[:, 2] * oz[:, 0]
        cc = (ox * ox + oy * oy + oz * oz)[:, 0] - RA * RA; O_ = O
    t_exit = -b + np.sqrt(np.maximum(b * b - cc, 0))
    T_ = np.minimum(tmax, t_exit); u = (np.arange(n) + 0.5) / n
    ts = T_[:, None] * u[None] ** 2.2
    dt = np.diff(np.concatenate([np.zeros((N, 1)), ts], 1), axis=1)
    px, py, pz = ox + ts * dirs[:, 0:1], oy + ts * dirs[:, 1:2], oz + ts * dirs[:, 2:3]
    rad = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
    alt = np.maximum(rad - RP, 0)
    rhoR = np.exp(-alt / HR)
    rhoF = np.zeros_like(alt)
    if FOG > 0:
        sx = O_[:, 0:1] + ts * dirs[:, 0:1]; sy = O_[:, 1:2] + ts * dirs[:, 1:2]; near = ts < 70
        fz = np.full(ts.shape, FOG_Z)
        fz[near] = FOG_Z + 0.16 * (fbm2(sx[near] * 0.55, sy[near] * 0.55, 3) - 0.5) * 2 + 0.05 * (fbm2(sx[near] * 3.1, sy[near] * 3.1, 2) - 0.5)
        rhoF = FOG * np.exp(-np.maximum(alt - fz, 0) / FOG_H) * near
    rhoM = np.exp(-alt / HM) * a.haze + rhoF
    odR = np.cumsum(rhoR * dt, 1); odM = np.cumsum(rhoM * dt, 1)
    if PHYS:
        odO = np.cumsum(np.clip(1 - np.abs(alt - 25) / 15, 0, 1) * dt, 1)
        Tv = np.exp(-(BR[None, None] * odR[..., None] + BMe[None, None] * odM[..., None] + BO[None, None] * odO[..., None]))
        mu_s = (px * LD[0] + py * LD[1] + pz * LD[2]) / rad
        Ts = sun_T_lut(alt, mu_s)
    else:
        Tv = np.exp(-(BR[None, None] * odR[..., None] + BMe[None, None] * odM[..., None]))
        Ts = sun_T_classic(alt)
    scat = (BR * ph_r(mu)[:, None])[:, None] * rhoR[..., None] + (BM * ph_m(mu)[:, None])[:, None] * (rhoM - rhoF)[..., None] \
         + (BM * (0.35 * ph_m(mu) + 0.65 * 0.25 / np.pi)[:, None])[:, None] * rhoF[..., None] * 1.6
    Lin = np.sum(Tv * Ts * scat * dt[..., None], 1) * LI
    if light is not None:                                                  # moon: scattering only
        ms = (BR[None, None] * rhoR[..., None] + BM[None, None] * rhoM[..., None]) * MS_SRC(alt, mu_s) * (LI / SUN_I)
        return (Lin + np.sum(Tv * ms * dt[..., None], 1)) * light[2], Tv[:, -1]
    if PHYS:
        # crude multiple scattering: an isotropic source proportional to the local sunlit sky brightness
        ms = (BR[None, None] * rhoR[..., None] + BM[None, None] * rhoM[..., None]) * MS_SRC(alt, mu_s)
        Lin = Lin + np.sum(Tv * ms * dt[..., None], 1) + NIGHT_SKY * (1 - Tv[:, -1])
    if MOON_ON:
        Lin = Lin + atmosphere(dirs, tmax, n, O, light=(MOON, MOON_I, MOON_TINT))[0]
    if GLOW is not None:
        sx = O_[:, 0:1] + ts * dirs[:, 0:1]; sy = O_[:, 1:2] + ts * dirs[:, 1:2]
        g = GLOW(sx, sy, alt)
        Lin = Lin + np.sum(Tv * (BM[None, None] * rhoM[..., None] * 0.25 / np.pi + BR[None, None] * rhoR[..., None] * 0.08) * g * dt[..., None], 1)
    return Lin, Tv[:, -1]

MOON_I = SUN_I * 1e-3 * (a.moon if a.moon is not None else 1.0)           # not physical (1/400000): tuned against NIGHT_SKY and city lights

def MS_SRC(alt, mu_s):
    """isotropic multiple-scattering source (radiance units / km), fitted by eye to keep twilight skies from going black"""
    day = sun_T_lut(alt, mu_s) * np.array([0.8, 0.95, 1.15])
    u = np.clip((mu_s + 0.20) / 0.24, 0, 1); u = u * u * (3 - 2 * u)      # sunlit sky overhead lasts until ~ -11 deg
    twi = (u * np.exp(-alt / 8.0))[..., None] * np.array([0.30, 0.50, 1.0]) * 0.30
    return (day + twi) * SUN_I * 0.02

# ------------------------------------------------------------------ lenticular cloud
CL_C = np.array([0.0, 0.3, PEAK + 0.4]); CL_R = np.array([0.7, 0.58, 0.12]) * max(REL, 1.5) + np.array([0, 0, 0.1])
def cloud_density(p):
    q = (p - CL_C) / CL_R; r = np.sqrt(q[:, 0] ** 2 + q[:, 1] ** 2)
    lens = np.clip(1 - r ** 2, 0, 1) ** 0.7; z = q[:, 2]
    plates = np.clip(1 - np.abs((np.mod(z * 1.5 + 0.5 + 0.25 * r, 1.0) - 0.5)) * 3.2, 0, 1) * (np.abs(z) < lens)
    n = fbm3(p[:, 0] * 2.6, p[:, 1] * 2.6, p[:, 2] * 5.0, 4)
    return np.clip(plates * (0.3 + 1.3 * n) - 0.62 + 0.4 * lens, 0, None) * np.clip(lens - np.abs(z), 0, 1) * 9.0
def march_cloud(dirs, tmax, n=56, O=None):
    N = len(dirs); col = np.zeros((N, 3)); tr = np.ones(N)
    Oc = CAM[None] if O is None else O
    o = (Oc - CL_C) / CL_R; d = dirs / CL_R
    A = np.sum(d * d, 1); B = 2 * np.sum(o * d, 1); C = np.sum(o * o, 1) - 1; disc = B * B - 4 * A * C
    sq = np.sqrt(np.maximum(disc, 0)); t0 = np.maximum((-B - sq) / (2 * A), 0); t1 = np.minimum((-B + sq) / (2 * A), tmax)
    idx = np.where((disc > 0) & (t1 > t0))[0]
    if not len(idx): return col, tr
    Oi = Oc if O is None else Oc[idx]
    dd, a0, a1 = dirs[idx], t0[idx], t1[idx]; dtv = (a1 - a0) / n
    Tr = np.ones(len(idx)); Cc = np.zeros((len(idx), 3))
    sun_col = sun_T(np.array([CL_C[2]]))[0] * SUN_I * 0.16; amb = np.array([0.35, 0.45, 0.65]) * 0.16 * AMB_SCALE
    mu = dd @ SUN
    if a.cloud_model == "ms":
        ph0 = 0.7 * hg(mu, 0.75) + 0.3 * hg(mu, -0.3)                    # dual lobe: forward + back
        ph0 = (ph0 + 0.25 * hg(mu, 0.97)) * 4 * np.pi * 0.5               # sharp silver-lining lobe near the sun
    else:
        ph = 0.6 * ph_m(mu) * np.pi + 0.4
    for k in range(n):
        p = Oi + dd * (a0 + (k + 0.5) * dtv)[:, None]; den = cloud_density(p); m = den > 1e-3
        if not m.any(): continue
        odl = sum(cloud_density(p[m] + SUN * 0.18 * j) * 0.18 for j in range(1, 5))
        hgt = (0.6 + 0.4 * (p[m, 2] - CL_C[2]) / CL_R[2])[:, None]
        if a.cloud_model == "ms":
            # Wrenninge-style octaves: each successive order sees less extinction and a flatter phase
            sun_term = np.zeros((m.sum(), 3)); aa, bb, cc_ = 1.0, 1.0, 1.0
            for _o in range(4):
                pho = ph0[m] * cc_ + (1 - cc_) * 1.0
                sun_term += aa * np.exp(-odl * 1.2 * bb)[:, None] * pho[:, None]
                aa, bb, cc_ = aa * 0.4, bb * 0.7, cc_ * 0.5
            powder = 1 - 0.5 * np.exp(-den[m] * 0.6)                      # darker fringes facing the sun
            light = sun_term * (sun_col / 1.25)[None] * powder[:, None] + amb[None] * hgt ** 1.5 * 1.1
        else:
            light = np.exp(-odl * 1.2)[:, None] * sun_col[None] * ph[m, None] + amb[None] * hgt
        st = np.exp(-den[m] * dtv[m] * 1.4); Cc[m] += (Tr[m] * (1 - st))[:, None] * light; Tr[m] *= st
    col[idx] = Cc; tr[idx] = Tr
    return col, tr

# ------------------------------------------------------------------ ambient light (sky-derived when physical)
SKYC0 = np.array([0.30, 0.40, 0.62]) * 0.075; BOUNCE0 = np.array([0.20, 0.15, 0.10]) * 0.018
AMB_SCALE = np.ones(3); SKYC, BOUNCE = SKYC0, BOUNCE0
if PHYS:
    k_ = np.arange(160) + 0.5; zc = 1 - k_ / 160; ph_ = np.pi * (1 + 5 ** 0.5) * k_
    hd = np.stack([np.sqrt(1 - zc ** 2) * np.cos(ph_), np.sqrt(1 - zc ** 2) * np.sin(ph_), zc], 1)
    Lh, _ = atmosphere(hd, np.full(160, 1e5), O=np.repeat([[CAM[0], CAM[1], BASE + 0.03]], 160, 0))
    E_sky = (Lh * hd[:, 2:3]).sum(0) * 2 * np.pi / 160
    SKYC = E_sky / np.pi * 1.35                                            # 1.35: light from beyond single scatter
    sun_h = sun_T(np.array([BASE]))[0] * SUN_I * 0.17 * max(math.sin(SUN_EL), 0)
    BOUNCE = BOUNCE0 * float((sun_h + SKYC).mean() / 0.42)
    AMB_SCALE = SKYC / SKYC0
    print("sky ambient", np.round(SKYC, 5), "(classic", np.round(SKYC0, 4), ")", flush=True)

# ------------------------------------------------------------------ ray march over the heightfield
def march(O, dirs, t0=0.05, iters=900, tfar=220.0):
    N = len(dirs); single = O.shape[0] == 1
    t = np.full(N, t0); prev = t.copy(); alive = np.ones(N, bool); hit = np.zeros(N, bool)
    ZTOP = PEAK + 0.2
    for _ in range(iters):
        ia = np.where(alive)[0]
        if not len(ia): break
        p = (O if single else O[ia]) + dirs[ia] * t[ia, None]; dh = p[:, 2] - terrain_z(p[:, 0], p[:, 1])
        h_ = dh < 0.0004 + 0.00025 * t[ia]; hit[ia[h_]] = True; alive[ia[h_]] = False
        gone = (t[ia] > tfar) | ((p[:, 2] > ZTOP) & (dirs[ia, 2] > 0)); alive[ia[gone & ~h_]] = False
        mv = ~h_ & ~gone; prev[ia[mv]] = t[ia[mv]]
        t[ia[mv]] += np.maximum(dh[mv] * 0.45, 0.004 + 0.0022 * t[ia[mv]])
    hi_ = np.where(hit)[0]; lo, hi = prev[hi_].copy(), t[hi_].copy()
    for _ in range(8):
        mid = (lo + hi) / 2; p = (O if single else O[hi_]) + dirs[hi_] * mid[:, None]
        below = p[:, 2] - terrain_z(p[:, 0], p[:, 1]) < 0; hi = np.where(below, mid, hi); lo = np.where(below, lo, mid)
    t[hi_] = hi
    return t, hit

def soft_shadow(P, n, steps=70, L=None):
    sh = np.ones(len(P)); ts = np.full(len(P), 0.03)
    if L is None:
        if SUN_EL < -0.1: return np.zeros(len(P))                          # sun far below every horizon
        L = SUN
    for _ in range(steps):
        q = P + L * ts[:, None] + n * 0.002; dq = q[:, 2] - terrain_z(q[:, 0], q[:, 1])
        sh = np.minimum(sh, np.clip(10 * dq / ts, 0, 1)); ts += np.clip(dq * 0.6, 0.03, 1.5)
        if ts.min() > 30: break
    return sh * sh * (3 - 2 * sh)

# ------------------------------------------------------------------ terrain surface
def terrain_normal(x, y, dist):
    e = 0.03
    hx = (hf(x + e, y) - hf(x - e, y)) / (2 * e); hy = (hf(x, y + e) - hf(x, y - e)) / (2 * e)
    slope = np.hypot(hx, hy); sw = np.clip(slope / 0.6, 0, 1); dd = 0.006
    dxn = (detail(x + dd, y, sw) - detail(x - dd, y, sw)) / (2 * dd); dyn = (detail(x, y + dd, sw) - detail(x, y - dd, sw)) / (2 * dd)
    fade = np.clip(1 - dist / 90, 0.2, 1)
    n = np.stack([-(hx + dxn * fade), -(hy + dyn * fade), np.ones_like(x)], 1); n /= np.linalg.norm(n, axis=1, keepdims=True)
    return n, slope

def terrain_albedo(x, y, n, slope, dist):
    z = hf(x, y); nz = fbm2(x * 3, y * 3, 5)
    steppe = np.array([0.19, 0.15, 0.105]) * (0.8 + 0.4 * nz[:, None])
    xr, yr = x * 0.82 + y * 0.57, -x * 0.57 + y * 0.82; warp = fbm2(x * 1.3, y * 1.3, 3) * 0.6
    fcell = vnoise2(np.floor(xr / (0.45 + 0.3 * warp) + warp) * 1.37, np.floor(yr / (0.3 + 0.15 * warp) - warp) * 2.11)
    fields = np.where(fcell[:, None] > 0.62, np.array([0.21, 0.25, 0.12]), np.where(fcell[:, None] > 0.32, np.array([0.40, 0.34, 0.21]), np.array([0.31, 0.24, 0.16])))
    fd = np.clip(1 - dist / 18, 0, 1)[:, None] ** 1.3
    fields = fields * fd + np.array([0.31, 0.27, 0.17]) * (1 - fd)
    low = np.clip((BASE + 0.2 - z) / 0.2, 0, 1)[:, None]; base_c = steppe * (1 - low) + fields * low
    if a.biome == "temperate": base_c = temperate_ground(x, y, z, slope, dist, fields)
    rock = np.array([0.085, 0.078, 0.075]) * (0.8 + 0.5 * fbm2(x * 20, y * 20, 3)[:, None])
    rw = np.clip((slope - 0.35) / 0.35, 0, 1) * np.clip((z - BASE - 0.2 * REL) / (0.2 * REL), 0, 1)
    rw = np.clip(rw + np.clip((z - BASE - 0.5 * REL) / (0.2 * REL), 0, 1) * 0.7, 0, 1)[:, None]
    alb = base_c * (1 - rw) + rock * rw
    sl = SNOW + 0.35 * (fbm2(x * 2.2, y * 2.2, 4) - 0.5) - 0.25 * np.clip(n[:, 2] - 0.7, 0, 1)
    sn = np.clip((z - sl) / 0.12, 0, 1) * np.clip((n[:, 2] - 0.45) / 0.2, 0, 1)
    gully = np.clip((z - SNOW + 0.5) / 0.3, 0, 1) * np.clip(fbm2(x * 25, y * 25, 3) - 0.62, 0, 1) * 6
    sn = np.clip(sn + gully * (n[:, 2] > 0.5), 0, 1)[:, None]
    alb = alb * (1 - sn) + np.array([0.86, 0.88, 0.92]) * sn
    if TREES_ON or VILLAGES_ON or CITY_ON:                                 # forest floor, village yards, streets
        alb = near_ground(x, y, z, slope, alb, sn, dist)
    return alb, z, sn

TREELINE = (a.treeline / 1000) if a.treeline is not None else BASE + 0.55 * REL
FOLIAGE = {"summer": np.array([0.050, 0.075, 0.032]), "autumn": np.array([0.11, 0.075, 0.035]), "spring": np.array([0.07, 0.10, 0.04])}[a.season]
def temperate_open(x, y, z, slope):
    """fraction of open land (meadow / fields) vs forest in the temperate biome"""
    n1 = fbm2(x * 1.7 + 11, y * 1.7 - 4, 5)
    flat = np.clip((0.18 - slope) / 0.12, 0, 1) * np.clip((BASE + 0.35 - z) / 0.25, 0, 1)
    return n1, np.clip(flat * 1.3 * (n1 > 0.45) + np.clip((n1 - 0.62) * 4, 0, 1), 0, 1)

def temperate_ground(x, y, z, slope, dist, fields):
    """forest below the treeline, broken by meadows and fields in the flats; dark scoria/scree above"""
    n1, open1 = temperate_open(x, y, z, slope); n2 = fbm2(x * 7, y * 7, 3)
    forest = FOLIAGE * (0.65 + 0.7 * n2[:, None]) * np.array([1.0, 1.0 - 0.15 * (a.season == "autumn"), 1.0])
    if a.season == "autumn":
        mix = np.clip((fbm2(x * 4 - 3, y * 4 + 8, 3) - 0.45) * 3, 0, 1)[:, None]
        forest = forest * (1 - mix) + np.array([0.14, 0.06, 0.025]) * (0.7 + 0.6 * n2[:, None]) * mix
    meadow = np.array([0.10, 0.12, 0.055]) * (0.8 + 0.4 * n2[:, None])
    open_ = open1[:, None]
    fl_ = fields * np.array([0.55, 0.75, 0.55])
    low_c = forest * (1 - open_) + (meadow * 0.5 + fl_ * 0.5) * open_
    scoria = np.array([0.13, 0.085, 0.07]) * (0.75 + 0.5 * fbm2(x * 12, y * 12, 3)[:, None])
    tl = TREELINE + 0.25 * (n1 - 0.5)
    above = np.clip((z - tl) / 0.25, 0, 1)[:, None]
    shrub = np.clip((z - tl + 0.25) / 0.25, 0, 1)[:, None] * (1 - above)
    return low_c * (1 - above - shrub) + (meadow * 0.7 + scoria * 0.3) * shrub + scoria * above

def shade_terrain(P, D, dist, extra_shadow=None):
    x, y = P[:, 0], P[:, 1]
    n, slope = terrain_normal(x, y, dist)
    alb, z, sn = terrain_albedo(x, y, n, slope, dist)
    sh = soft_shadow(P, n)
    if extra_shadow is not None: sh = sh * extra_shadow(P, n)
    big = map_coordinates(HFB, [(y - GN[0]) / RES, (x - GE[0]) / RES], order=1, mode="nearest")
    ao = np.clip(1 - (big - z) * 2.2, 0.35, 1)
    sunc = sun_T(np.maximum(z, 0)) * SUN_I * 0.17; skyc = SKYC
    ndl = np.clip(n @ SUN, 0, 1)
    col = alb * (sunc * (ndl * sh)[:, None] + skyc[None] * (ao * (0.55 + 0.45 * n[:, 2]))[:, None] + BOUNCE * ao[:, None])
    hv = SUN - D; hv /= np.linalg.norm(hv, axis=1, keepdims=True)
    col += (np.clip(np.sum(n * hv, 1), 0, 1) ** 60)[:, None] * sunc * sh[:, None] * sn * 0.25
    if CITY_ON: col += alb * city_ground_light(x, y)
    if MOON_ON:
        shm = soft_shadow(P, n, L=MOON)
        moonc = sun_T_lut(np.maximum(z, 0), np.full(len(z), math.sin(MOON_EL))) * MOON_I * 0.17 * MOON_TINT
        col += alb * moonc * (np.clip(n @ MOON, 0, 1) * shm)[:, None]
    return col

# ------------------------------------------------------------------ water
if WATER:
    _wr = np.random.default_rng(11)
    NWV = 36
    WV_LAM = 0.03e-3 * (180 ** (np.arange(NWV) / (NWV - 1)))                # 3 cm ... 5.4 m, in km
    wind = math.radians(a.wind) if a.wind is not None else math.atan2(right[0], right[1])
    WV_ANG = wind + _wr.normal(0, 0.55, NWV) * (1 + 0.6 * (np.arange(NWV) < NWV // 2))
    WV_DIR = np.stack([np.sin(WV_ANG), np.cos(WV_ANG)], 1)
    WV_PH = _wr.random(NWV) * 2 * np.pi
    WV_S = np.full(NWV, 0.045 * a.waves * math.sqrt(2.0 / NWV))            # slope amplitude per component
    WV_S[NWV - 8:] *= np.linspace(1, 1.6, 8)                               # a little swell
    WSIG, WSCAT, WBED = {"lake": (np.array([420., 95., 80.]), np.array([0.010, 0.030, 0.026]), np.array([0.20, 0.18, 0.12])),
                         "clear": (np.array([380., 55., 30.]), np.array([0.004, 0.016, 0.024]), np.array([0.32, 0.29, 0.21])),
                         "sea": (np.array([400., 80., 45.]), np.array([0.004, 0.018, 0.030]), np.array([0.25, 0.23, 0.17]))}[a.water_color]

def water_frac(x, y): return map_coordinates(WM, [(y - GN[0]) / RES, (x - GE[0]) / RES], order=1, mode="nearest")

def plane_t(O, D, zlev):
    """ray / curved water surface z = zlev - r^2/2RE (r from the camera)"""
    ox, oy = O[:, 0] - CAM[0], O[:, 1] - CAM[1]
    A = (D[:, 0] ** 2 + D[:, 1] ** 2) / (2 * RE); B = D[:, 2] + (ox * D[:, 0] + oy * D[:, 1]) / RE
    C = O[:, 2] - zlev + (ox ** 2 + oy ** 2) / (2 * RE)
    disc = np.maximum(B * B - 4 * A * C, 0)
    return 2 * C / (-B + np.sqrt(disc))

def wave_slopes(x, y, D, dist, rng):
    """resolved slope field (km/km) + unresolved slope variance, filtered by the pixel footprint"""
    X, Y = x * 1000, y * 1000                                               # metres
    u_ = X * math.sin(wind) + Y * math.cos(wind); v_ = X * math.cos(wind) - Y * math.sin(wind)
    gn = fbm2(u_ * 0.0022 + 3.1, v_ * 0.011 - 2.2, 4)                       # wind streaks along the wind
    gust = 0.15 + 2.2 * np.clip((gn - 0.48) * 3.0, 0, 1) ** 1.5
    vh = D[:, :2] / np.maximum(np.linalg.norm(D[:, :2], axis=1, keepdims=True), 1e-6)
    fa = PIXA * dist * 1000 / np.maximum(np.abs(D[:, 2]), 0.004); fc = PIXA * dist * 1000
    sx = np.zeros_like(X); sy = np.zeros_like(X); var = np.zeros_like(X)
    for k in range(NWV):
        cos_k = np.abs(WV_DIR[k, 0] * vh[:, 0] + WV_DIR[k, 1] * vh[:, 1])
        fp = np.sqrt((fa * cos_k) ** 2 + (fc * (1 - cos_k ** 2) ** 0.5) ** 2) + 1e-6
        r_ = WV_LAM[k] * 1000 / fp
        w = np.clip((r_ - 1.5) / 2.5, 0, 1); w = w * w * (3 - 2 * w)
        ph = (X * WV_DIR[k, 0] + Y * WV_DIR[k, 1]) * (2 * np.pi / (WV_LAM[k] * 1000)) + WV_PH[k]
        c = np.cos(ph) * WV_S[k] * w
        sx += c * WV_DIR[k, 0]; sy += c * WV_DIR[k, 1]
        var += (1 - w * w) * WV_S[k] ** 2 / 2
    return sx * gust, sy * gust, var * gust ** 2 + 2e-6

def fresnel_w(c):
    return 0.02 + 0.98 * np.clip(1 - c, 0, 1) ** 5

def trace_secondary(O, R, dist0):
    """radiance arriving at O from direction R (terrain or sky, with atmosphere and cloud), no sun disc"""
    t2, hit2 = march(O, R, t0=0.002)
    col = np.zeros((len(R), 3)); tmax = np.where(hit2, t2, 1e5)
    h = np.where(hit2)[0]
    if len(h):
        P2 = O[h] + R[h] * t2[h, None]
        col[h] = shade_terrain(P2, R[h], dist0[h] + t2[h])
    Lin, Tv = atmosphere(R, tmax, O=O); col = col * Tv + Lin
    if a.cloud:
        cc, ct = march_cloud(R, tmax, O=O)
        dcl = np.linalg.norm(CL_C[None] - O, axis=1)
        Lc, Tc = atmosphere(R, np.minimum(tmax, dcl), O=O)
        col = Lc + Tc * cc + ct[:, None] * (col - Lc)
    if CLOUD_LAYER_ON:
        col = apply_cloud_layer(R, tmax, col, O=O)
    return col, tmax

def shade_water(P, D, dist, rng):
    Nw = len(P); x, y = P[:, 0], P[:, 1]
    sh = soft_shadow(P, np.tile([0, 0, 1.0], (Nw, 1)))
    sunc = sun_T(np.array([WL]))[0] * SUN_I * 0.17
    depth = np.maximum(map_coordinates(WDEPTH, [(y - GN[0]) / RES, (x - GE[0]) / RES], order=1, mode="nearest"), 0.0003)
    bed = WBED * (0.7 + 0.6 * fbm2(x * 400, y * 400, 3))[:, None]
    ns = max(1, a.water_spp); acc = np.zeros((Nw, 3)); spread = np.zeros(Nw)
    Fsum = np.zeros(Nw); Rsum = np.zeros((Nw, 3)); Tref = np.full(Nw, 1e5)
    fp = PIXA * dist                                                         # pixel footprint across, km
    for s in range(ns):
        ju, jv = (s + rng.random(Nw)) / ns - 0.5, rng.random(Nw) - 0.5    # stratified sub-pixel jitter
        along = D[:, :2] / np.maximum(np.linalg.norm(D[:, :2], axis=1, keepdims=True), 1e-6)
        xs = x + along[:, 0] * ju * fp / np.maximum(np.abs(D[:, 2]), 0.004) - along[:, 1] * jv * fp
        ys = y + along[:, 1] * ju * fp / np.maximum(np.abs(D[:, 2]), 0.004) + along[:, 0] * jv * fp
        sx, sy, var = wave_slopes(xs, ys, D, dist, rng)
        sig = np.sqrt(var)
        gx, gy = sx + rng.normal(0, 1, Nw) * sig, sy + rng.normal(0, 1, Nw) * sig
        n = np.stack([-gx, -gy, np.ones(Nw)], 1); n /= np.linalg.norm(n, axis=1, keepdims=True)
        cosi = np.clip(-(D * n).sum(1), 0.0, 1)
        R = D + 2 * cosi[:, None] * n
        R[:, 2] = np.maximum(R[:, 2], np.abs(R[:, 2]) * 0.3 + 0.0015); R /= np.linalg.norm(R, axis=1, keepdims=True)
        F = fresnel_w(cosi)
        refl, trefl = trace_secondary(P + np.array([0, 0, 0.0006]), R, dist)
        Fsum += F / ns; Rsum += F[:, None] * refl / ns; Tref = np.minimum(Tref, trefl)
        # refraction into the water: absorbed bed + in-scatter
        eta = 1 / 1.333; k = np.clip(1 - eta * eta * (1 - cosi ** 2), 0, 1); cost = np.sqrt(k)
        ssz = max(math.sin(SUN_EL), 0.02); cs = math.sqrt(max(1 - eta * eta * (1 - ssz ** 2), 0))
        Edn = sunc * (sh * ssz)[:, None] * (1 - fresnel_w(ssz)) + SKYC[None] * 0.9    # irradiance just below, /pi
        path = depth[:, None] * (1 / cs + 1 / np.maximum(cost, 0.05))[:, None]
        Tw = np.exp(-WSIG[None] * path)
        L_bed = bed * Edn * Tw
        L_in = WSCAT[None] * Edn * (1 - Tw) * 3.0
        trans = (L_bed + L_in) * (1 - F)[:, None]
        # sun glint: Beckmann lobe around the resolved normal, width from the unresolved slopes
        nr = np.stack([-sx, -sy, np.ones(Nw)], 1); nr /= np.linalg.norm(nr, axis=1, keepdims=True)
        hv = SUN - D; hv /= np.linalg.norm(hv, axis=1, keepdims=True)
        ch = np.clip((nr * hv).sum(1), 1e-3, 1); a2 = 2 * var + 1.2e-5
        Dm = np.exp(-(1 - ch * ch) / (ch * ch * a2)) / (np.pi * a2 * ch ** 4)
        co = np.clip(-(D * nr).sum(1), 0.02, 1)
        glint = (np.pi * sunc) * (fresnel_w((hv * SUN).sum(1)) * Dm / (4 * co) * sh)[:, None] * (SUN_EL > 0)
        acc += F[:, None] * refl + trans + np.minimum(glint, 400)
        spread += 2 * sig * fl / ns                                          # vertical smear of the reflection, px
    return acc / ns, spread, Fsum, Rsum, Tref

# ------------------------------------------------------------------ near field: trees and houses (stubs filled below)
TREES_ON = a.trees > 0; VILLAGES_ON = a.villages > 0; CITY_ON = a.city > 0; CLOUD_LAYER_ON = a.cloud_layer != "none"
def near_ground(x, y, z, slope, alb, sn, dist): return alb
def city_ground_light(x, y): return 0.0
def apply_cloud_layer(dirs, tmax, col, O=None, extra=None): return col

exec_extras = os.path.join(os.path.dirname(os.path.abspath(__file__)), "terrain_near.py")
if TREES_ON or VILLAGES_ON or CITY_ON or CLOUD_LAYER_ON:
    exec(compile(open(exec_extras).read(), exec_extras, "exec"))

# ------------------------------------------------------------------ render
def render_rows(r0, r1):
    ys, xs = np.mgrid[r0:r1, 0:W]
    px = (xs + 0.5 - W / 2) / fl; py = -(ys + 0.5 - H / 2 - H * 0.02) / fl
    dirs = fwd[None] + px.ravel()[:, None] * right[None] + py.ravel()[:, None] * up[None]
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True); N = len(dirs)
    gfile = os.path.join(GEOM_DIR, f"{r0:05d}_{r1:05d}.npz") if GEOM_DIR else None
    G_ = dict(np.load(gfile)) if gfile and os.path.exists(gfile) else None
    if G_ is not None: t, hit = G_["t"], G_["hit"]
    else: t, hit = march(CAM[None], dirs)
    rng = np.random.default_rng(1000 + r0)
    t_geom = t.copy()
    img = np.zeros((N, 3))
    wat = np.zeros(N, bool)
    if WATER and hit.any():
        hi_ = np.where(hit)[0]; P = CAM + dirs[hi_] * t[hi_, None]
        wf = water_frac(P[:, 0], P[:, 1])
        isw = (wf > 0.5) & (hf(P[:, 0], P[:, 1]) <= WL + 0.0003)
        wat[hi_[isw]] = True
        tw = plane_t(CAM[None].repeat(isw.sum(), 0), dirs[hi_[isw]], WL)
        t[hi_[isw]] = np.where(np.isfinite(tw) & (tw > 0), tw, t[hi_[isw]])
    tmax = np.where(hit, t, 1e5)
    inst = None
    vis = None
    if TREES_ON or VILLAGES_ON or CITY_ON:
        vis = (G_["vh"], G_["vi"], G_["vt"]) if G_ is not None else near_vis(r0, r1, tmax)
        inst = near_field(r0, r1, dirs, tmax, vis=vis)                   # (ids, t, colour, coverage) of instance hits
    if gfile and G_ is None:
        os.makedirs(GEOM_DIR, exist_ok=True)
        ex_ = dict(vh=vis[0], vi=vis[1], vt=vis[2]) if vis is not None else {}
        np.savez(gfile, t=t_geom, hit=hit, **ex_)
    ter = np.where(hit & ~wat)[0]
    if len(ter):
        P = CAM + dirs[ter] * t[ter, None]
        ex = inst_shadow if (TREES_ON or VILLAGES_ON or CITY_ON) else None
        img[ter] = shade_terrain(P, dirs[ter], t[ter], extra_shadow=ex)
    wi = np.where(wat)[0]
    if len(wi):
        P = CAM + dirs[wi] * t[wi, None]
        img[wi], wsp, Fw, Rw, Trw = shade_water(P, dirs[wi], t[wi], rng)
        wblur = np.zeros(N, np.float32); wblur[wi] = np.maximum(wsp, 0.01)
        if TREES_ON or VILLAGES_ON or CITY_ON:                             # trees and houses seen in the water
            mids, mcov, mcol = near_field_mirror(r0, r1, dirs, wi, t[wi], Trw)
            if len(mids):
                loc = np.searchsorted(wi, mids)
                img[mids] += (Fw[loc] * mcov)[:, None] * (mcol - Rw[loc] / np.maximum(Fw[loc], 1e-6)[:, None])
    sky = ~hit; disc = np.clip((dirs @ SUN - math.cos(math.radians(0.27))) / 2e-5, 0, 1)
    img[sky] = disc[sky, None] * sun_T(np.array([CAM[2]]))[0] * 2500
    star = None
    if NIGHT:                                                                # kept apart from img: post adds it after bloom
        star = np.zeros((N, 3)); star[sky] = stars(dirs[sky])
        if MOON_ON:
            mdisc = np.clip((dirs @ MOON - math.cos(math.radians(0.26))) / 2e-5, 0, 1)
            img[sky] += mdisc[sky, None] * sun_T_lut(np.array([CAM[2]]), np.array([math.sin(MOON_EL)]))[0] * MOON_I / SUN_I * 2500
    Lin, Tv = atmosphere(dirs, tmax); img = img * Tv + Lin
    if star is not None: star *= Tv
    if inst is not None and len(inst[0]):
        ids, ti, ci, cov = inst                                              # anti-aliased: blend by sub-pixel coverage
        Li, Ti = atmosphere(dirs[ids], ti)
        img[ids] = cov[:, None] * (ci * Ti + Li) + (1 - cov[:, None]) * img[ids]
        tmax[ids] = np.where(cov > 0.5, ti, tmax[ids])
        if star is not None: star[ids] *= (1 - cov[:, None])
    if a.cloud:
        cc, ct = march_cloud(dirs, tmax); Lc, Tc = atmosphere(dirs, np.minimum(tmax, np.linalg.norm(CL_C - CAM)))
        img = Lc + Tc * cc + ct[:, None] * (img - Lc)
        if star is not None: star *= ct[:, None]
    if CLOUD_LAYER_ON:
        img = apply_cloud_layer(dirs, tmax, img, extra=star)
    if star is not None:
        os.makedirs(AUX, exist_ok=True)
        np.save(os.path.join(AUX, f"{r0:05d}_stars.npy"), star.reshape(r1 - r0, W, 3).astype(np.float32))
    if CITY_ON or a.aperture > 0 or WATER:
        os.makedirs(AUX, exist_ok=True)
        np.save(os.path.join(AUX, f"{r0:05d}_depth.npy"), tmax.reshape(r1 - r0, W).astype(np.float32))
        if WATER:
            if not len(wi): wblur = np.zeros(N, np.float32)
            if inst is not None and len(inst[0]): wblur[inst[0]] = 0
            np.save(os.path.join(AUX, f"{r0:05d}_wblur.npy"), wblur.reshape(r1 - r0, W))
    return img.reshape(r1 - r0, W, 3)

def stars(d):
    """a sparse procedural star field (night only)"""
    az = np.arctan2(d[:, 0], d[:, 1]); el = np.arcsin(np.clip(d[:, 2], -1, 1))
    u, v = az * fl, el * fl
    iu, iv = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    h = hash2(iu, iv, 77); b = np.clip((h - 0.9965) / 0.0035, 0, 1) ** 3
    return b[:, None] * np.array([0.9, 0.95, 1.1]) * 0.02 * np.clip(d[:, 2] * 20, 0, 1)[:, None]

def camera_info():
    return dict(W=W, H=H, fl=fl, cam=CAM.tolist(), fwd=fwd.tolist(), right=right.tolist(), up=up.tolist(),
                peak_dist=float(np.hypot(*CAM[:2])), aperture_mm=a.aperture, night=bool(NIGHT),
                focus_km=a.focus if a.focus is not None else float(np.hypot(*(CAM[:2] - tgt_xy))))

def band_coverage():
    """rows covered by the band files on disk: {start row: row count} and a boolean mask over H"""
    have = {}
    for f in os.listdir(BANDS):
        if re.fullmatch(r"\d{5}\.npy", f):
            have[int(f[:5])] = np.load(os.path.join(BANDS, f), mmap_mode="r").shape[0]
    cov = np.zeros(H, bool)
    for s0, n in have.items(): cov[s0:s0 + n] = True
    return have, cov

def spans(mask):
    """[(start, end), ...] of the True runs in a boolean row mask"""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0].tolist(), np.where(d == -1)[0].tolist()))

if __name__ == "__main__":
    r0, r1 = (map(int, a.rows.split(":")) if a.rows else (0, H))
    r0, r1 = max(0, r0), min(r1, H)
    info = dict(peak_m=round(PEAK * 1000), base_m=round(BASE * 1000), snowline_m=round(SNOW * 1000),
                cam_dist_km=round(DIST - OFFSET, 1), sun_az=round(math.degrees(SUN_AZ)))
    if a.sky != "classic" or a.time: info.update(sky=a.sky, sun_el=a.sun_el)
    print(json.dumps(info), flush=True)
    json.dump(camera_info(), open(os.path.join(BANDS, "camera.json"), "w"))
    if CITY_ON: write_lights()
    # bands sit on a fixed grid of `step` rows; missing rows inside [r0, r1) are rendered in pieces that stop at
    # grid lines and at rows already on disk, so chunks of any size and full runs can be mixed without gaps
    step = max(8, 60000 // W); t0 = time.time()
    _, cov = band_coverage()
    want = np.zeros(H, bool); want[r0:r1] = True
    for m0, m1 in spans(want & ~cov):
        for r in range(m0, m1):
            if r != m0 and r % step: continue
            e = min((r // step + 1) * step, m1)
            np.save(os.path.join(BANDS, f"{r:05d}.npy"), render_rows(r, e).astype(np.float32))
            print(f"rows {r}-{e}  {time.time() - t0:.0f}s", flush=True)
    _, cov = band_coverage()
    done = int(cov.sum())
    if done >= H:
        print(f"{done}/{H} rows rendered — run post_terrain.py --bands {BANDS}")
    else:
        miss = spans(~cov)
        print(f"{done}/{H} rows rendered; missing rows " + ", ".join(f"{m0}-{m1}" for m0, m1 in miss)
              + f": run again without --rows (or with --rows {miss[0][0]}:{miss[-1][1]}) to fill them")
