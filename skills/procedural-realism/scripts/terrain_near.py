# terrain_near.py — near-field detail for pbr_terrain.py. Not a standalone script: pbr_terrain.py runs
# this file inside its own namespace when --trees, --villages, --city or --cloud-layer is used, so it
# reads the renderer's globals (a, CAM, hf, terrain_z, fwd, right, up, fl, W, H, SUN, SKYC, ...).
#
#   trees     conifer (whorled tiers), deciduous (lobed crown with limbs), poplar (flame) on jittered grids,
#             placed by slope / elevation / water / forest-noise masks; poplar rows on field edges.
#             Crowns are an envelope carved by 3-octave noise into leaf clumps with gaps; the carving
#             fades with the pixels per metre of each tree (level of detail), so far trees stay solid
#   houses    convex boxes with gable roofs on a grid aligned with the field pattern; flat-roofed
#             apartment blocks when --city is on. Close up: stone / plaster / board walls, tile / tin /
#             slate roofs, windows with frames, sills and shutters, doors, plinths and weathering
#   clutter   dry-stone walls, hedges and wooden fences on field edges, dirt tracks between houses,
#             crop rows and grass texture on the ground, all only within a few hundred metres
#   render    each instance is projected to its screen bounding box; only those pixels cast rays
#             against it (2x2 sub-samples for anti-aliasing), so cost scales with covered pixels
#   shadows   sun rays walk the instance grids (closed-form shapes) for terrain and instance hits;
#             crowns let light through by their chord length, and light inside a crown falls off
#             with the depth of foliage toward the sun (self-shadowing, backlit glow at the edges)
#   city      street-grid emitters, lit windows and a sky-glow field for post_terrain.py to splat
#   clouds    optional cumulus / stratus layer (raymarched slab, octave multiple scattering)

NEAR = a.near_km
T_CON, T_DEC, T_POP, T_HOUSE, T_BLOCK, T_WALL, T_FENCE = 0, 1, 2, 3, 4, 5, 6
ROT_C, ROT_S = 0.82, 0.57                                   # field grid orientation (see terrain_albedo)
FH = fwd[:2] / np.linalg.norm(fwd[:2])
CLUT_KM = min(NEAR, 0.6)                                    # walls, fences, tracks, crop rows: only this close

def _in_view(x, y, margin=7.0, rmax=None):
    vx, vy = x - CAM[0], y - CAM[1]; d = np.hypot(vx, vy)
    c = (vx * FH[0] + vy * FH[1]) / np.maximum(d, 1e-9)
    return ((c > math.cos(math.radians(a.hfov / 2 + margin))) | (d < 0.05)) & (d < (rmax or NEAR)), d

def _slope(x, y):
    e = 0.03
    return np.hypot((hf(x + e, y) - hf(x - e, y)) / (2 * e), (hf(x, y + e) - hf(x, y - e)) / (2 * e))

def _water(x, y):
    return water_frac(x, y) if WATER else np.zeros_like(x)

# ------------------------------------------------------------------ real streets (OpenStreetMap) for --city
# osm_roads.py fetches the highways in the view wedge once (cached in <work>/osm) and rasterises them to an 8 m
# grid: distance to the nearest centre line, its class, width and direction. The city density, building frontage,
# lamps, traffic, ground light and sky glow all follow it. Without network data: the synthetic street grid.
ROADS = None
if CITY_ON and a.streets == "osm":
    _res = None
    try:
        import osm_roads
        _fa = math.atan2(FH[1], FH[0]); _ang = np.radians(np.linspace(-a.hfov / 2 - 8, a.hfov / 2 + 8, 9))
        _px = np.concatenate([[CAM[0]], CAM[0] + (a.city_km + 0.5) * np.cos(_fa + _ang)])
        _py = np.concatenate([[CAM[1]], CAM[1] + (a.city_km + 0.5) * np.sin(_fa + _ang)])
        _bb = tuple(round(float(v), 3) for v in (LA0 + (_py.min() - 0.5) / KY, LO0 + (_px.min() - 0.5) / KX,
                                          LA0 + (_py.max() + 0.5) / KY, LO0 + (_px.max() + 0.5) / KX))
        print(f"streets: OpenStreetMap highways in {_bb} (cached in {os.path.join(a.work, 'osm')})", flush=True)
        _res = osm_roads.road_field(_bb, (LA0, LO0), os.path.join(a.work, "osm"), res=8.0, log=lambda m: print(m, flush=True))
    except Exception as ex:                                                # no network, no module: synthetic grid
        print(f"streets: OpenStreetMap unavailable ({str(ex)[:120]})", flush=True)
    if _res is None:
        print("streets: using the synthetic street grid", flush=True)
    else:
        _RD, _sg, RG = _res
        RSEG = {k: (v / 1000.0 if k in ("x0", "y0", "x1", "y1") else v) for k, v in _sg.items()}   # km
        RWAY = dict(id=np.array([r["id"] for r in _RD["roads"]], np.int64), lit=np.array([r["lit"] != "no" for r in _RD["roads"]]),
                    bridge=np.array([r["bridge"] for r in _RD["roads"]]), oneway=np.array([r["oneway"] for r in _RD["roads"]]),
                    service=np.array([(r["service"] or "") in ("driveway", "parking_aisle", "drive-through") for r in _RD["roads"]]))
        # urban density from road length per area: ~15 km of street per km² reads as dense city, parks and fields as 0
        _line = (RG["dist"] < 0.5 * RG["res"]) * np.where(RG["cls"] == 6, 0.6, 1.0)
        _dens = gaussian_filter(_line.astype(np.float32), 90.0 / RG["res"]) / 0.12
        RG["urb"] = np.clip((_dens - 0.2) / 0.8, 0, 1).astype(np.float32)
        ROADS = dict(tag=hashlib.md5(json.dumps(_bb).encode() + RG["dist"][::97, ::89].tobytes()).hexdigest()[:10])
        print(f"streets: {len(_RD['roads'])} OSM roads, {len(RSEG['x0'])} segments (Road data © OpenStreetMap contributors, ODbL)", flush=True)

def _rg(key, x, y, order=0):
    """sample the road grid at km coordinates (outside: far from any road)"""
    fx = (np.asarray(x) * 1000 - RG["x0"]) / RG["res"]; fy = (np.asarray(y) * 1000 - RG["y0"]) / RG["res"]
    cv = {"dist": 1e4, "cls": -1}.get(key, 0.0)
    return map_coordinates(RG[key], [fy - 0.5, fx - 0.5], order=order, mode="constant", cval=cv)

def road_edge(x, y):
    """metres from the edge of the nearest carriageway (negative on the road)"""
    return _rg("dist", x, y, 1) - 0.5 * _rg("width", x, y)

def _road_normal(x, y):
    e = 0.008
    gx = _rg("dist", x + e, y, 1) - _rg("dist", x - e, y, 1); gy = _rg("dist", x, y + e, 1) - _rg("dist", x, y - e, 1)
    n = np.hypot(gx, gy) + 1e-9
    return gx / n, gy / n

def village_field(x, y):
    """0..1: where settlements are (flat, low, dry ground; patches 1-3 km)"""
    v = fbm2(x * 0.42 + 40.0, y * 0.42 - 17.0, 4)
    thr = 0.66 - 0.22 * max(a.villages, a.city * 0.6)
    return np.clip((v - thr) * 7, 0, 1)

def city_field(x, y):
    """0..1 urban density for the night city: dense core near the camera, thinning with distance"""
    if not CITY_ON: return np.zeros_like(x)
    d = np.hypot(x - CAM[0], y - CAM[1])
    if ROADS is not None:                                                  # real streets: density from road length
        return _rg("urb", x, y, 1) * (d < a.city_km)
    n = fbm2(x * 0.5 + 7.0, y * 0.5 + 3.0, 4)
    core = np.clip(1 - d / a.city_km, 0, 1) ** 0.6
    return np.clip((n * 0.9 + core * 0.8 - 0.55 + 0.35 * a.city) * 2.5, 0, 1) * (d < a.city_km)

def _ground_ok(x, y, z, sl):
    ok = (_water(x, y) < 0.02) & (sl < 0.9)
    if WATER: ok &= z > WL + 0.0015
    return ok

def tree_density(x, y, z, sl):
    if a.trees <= 0: return np.zeros_like(x)
    if a.biome == "temperate":
        _, op = temperate_open(x, y, z, sl)
        tl = TREELINE + 0.25 * (fbm2(x * 1.7 + 11, y * 1.7 - 4, 5) - 0.5)
        p = (1 - op) * np.clip((tl - z) / 0.12, 0, 1) * 0.92 + 0.05
        p = p + 0.35 * village_field(x, y) * (1 - p)                      # garden trees in villages
    else:
        vf = village_field(x, y)
        p = np.clip(vf * 1.6, 0, 1) * 0.5 + 0.02 * np.clip(fbm2(x * 3 + 5, y * 3, 3) * 2 - 0.6, 0, 1)
        p = p * np.clip((BASE + 0.5 - z) / 0.2, 0, 1) + 0.12 * np.clip((sl - 0.25) * 2, 0, 1) * np.clip((BASE + 0.9 - z) / 0.3, 0, 1)
    p = p + 0.45 * city_field(x, y) * (1 - p) * (a.trees > 0)            # city streets and parks are green
    return np.clip(p * a.trees, 0, 1)

# ------------------------------------------------------------------ ground detail near the camera
def _stripe_amp(ext, period):
    """contrast left in a stripe pattern of this period (m) after averaging over a pixel footprint ext (m)"""
    return np.exp(-0.5 * (np.pi * ext / period) ** 2)

def _field_geom(x, y):
    """field cell value and row direction, the same cell maths as terrain_albedo's field colours"""
    xr, yr = x * ROT_C + y * ROT_S, -x * ROT_S + y * ROT_C; warp = fbm2(x * 1.3, y * 1.3, 3) * 0.6
    ci, cj = np.floor(xr / (0.45 + 0.3 * warp) + warp), np.floor(yr / (0.3 + 0.15 * warp) - warp)
    fcell = vnoise2(ci * 1.37, cj * 2.11)
    along_u = hash2(ci.astype(np.int64), cj.astype(np.int64), 44) < 0.5    # rows run along u (vary in yr) or along v
    return xr, yr, fcell, along_u, ci, cj

ORCH_KM = min(NEAR, 1.5)                                    # orchards planted in rows only this close
def orchard_field(x, y, z=None):
    """0/1: field cells at village margins planted as orchards (arid biome only)"""
    if a.biome == "temperate" or a.trees <= 0: return np.zeros(len(x))
    xr, yr, fcell, along_u, ci, cj = _field_geom(x, y)
    vf = village_field(x, y)
    z = hf(x, y) if z is None else z
    pick = hash2(ci.astype(np.int64), cj.astype(np.int64), 70) < 0.45 * min(1.0, a.trees * 1.25)
    return (pick & (vf > 0.03) & (vf < 0.8) & (z < BASE + 0.2)).astype(float)

def near_ground(x, y, z, slope, alb, sn, dist):
    fade = np.clip(1 - dist / (NEAR * 1.3), 0, 1)[:, None]
    td = tree_density(x, y, z, slope)[:, None] * fade * (1 - sn)
    alb = alb * (1 - 0.45 * td) + np.array([0.05, 0.062, 0.032]) * 0.45 * td             # shaded floor under canopy
    vfr = np.zeros(len(x))
    if VILLAGES_ON or CITY_ON:
        vfr = np.maximum(village_field(x, y) * (a.villages > 0), city_field(x, y)) * np.clip(1 - dist / (max(NEAR, a.city_km) * 1.5), 0, 1)
        vf = vfr[:, None]
        yard = np.array([0.24, 0.215, 0.19]) * (0.8 + 0.4 * fbm2(x * 60, y * 60, 2))[:, None]
        alb = alb * (1 - 0.6 * vf) + yard * 0.6 * vf
        alb = _tracks(x, y, dist, alb, vfr)
        if ROADS is not None:                                             # asphalt and pavements of the real streets
            fp = np.maximum(dist * 1000 / fl, 0.5)                          # metres per pixel
            re_ = road_edge(x, y); inc = (dist < a.city_km)[:, None]
            tar = np.clip(0.5 - re_ / fp, 0, 1)[:, None] * inc
            pav = np.clip(0.5 - (re_ - 2.5) / fp, 0, 1)[:, None] * inc * (1 - tar)
            alb = alb * (1 - tar - 0.7 * pav) + tar * np.array([0.075, 0.075, 0.08]) + 0.7 * pav * np.array([0.2, 0.19, 0.18])
    ko = np.where(dist < ORCH_KM * 1.05)[0]
    if len(ko):                                                         # irrigated grass under the orchard rows
        of = orchard_field(x[ko], y[ko], z[ko]) * np.clip((ORCH_KM * 1.05 - dist[ko]) / 0.15, 0, 1)
        grass = np.array([0.075, 0.095, 0.045]) * (0.8 + 0.4 * fbm2(x[ko] * 90, y[ko] * 90, 2))[:, None]
        alb[ko] = alb[ko] * (1 - 0.8 * of[:, None]) + grass * 0.8 * of[:, None]
    k = np.where((dist < CLUT_KM * 1.6) & (sn[:, 0] < 0.5))[0]
    if len(k):
        alb[k] = _ground_texture(x[k], y[k], z[k], dist[k], alb[k], vfr[k], td[k, 0])
    return alb

def _footprint(x, y, z, dist):
    """pixel footprint on flat ground in metres: across the view, along the view, and the view direction"""
    d = np.maximum(dist, 1e-4)
    mpp = d * 1000 / fl
    graz = np.clip((CAM[2] - z) / d, 0.012, 1)
    vx, vy = (x - CAM[0]) / d, (y - CAM[1]) / d
    return mpp, mpp / graz, vx, vy

def _ext(nx, ny, fc, fa, vx, vy):
    """footprint extent along the unit direction (nx, ny)"""
    return np.hypot(fa * (nx * vx + ny * vy), fc * (-nx * vy + ny * vx))

SUN_H = np.array([SUN[0], SUN[1]]) / max(math.hypot(SUN[0], SUN[1]), 1e-6)   # sun direction on the ground
RAKE = math.cos(SUN_EL) / max(math.sin(SUN_EL), 0.05)                           # how much a tilt of the ground shows
def _rake(tilt, nx, ny):
    """brightness factor of ground tilted by `tilt` (slope, m/m) toward the horizontal direction (nx, ny):
    micro-relief shading, done in the albedo so the low sun rakes rows, ruts and tufts"""
    return np.clip(1 + tilt * (nx * SUN_H[0] + ny * SUN_H[1]) * RAKE, 0.25, 2.2)

def _ground_texture(x, y, z, dist, alb, vf, td):
    fc, fa, vx, vy = _footprint(x, y, z, dist)
    X, Y = x * 1000, y * 1000                                                # metres
    xr, yr, fcell, along_u, ci, cj = _field_geom(x, y)
    if a.biome == "temperate":
        _, op = temperate_open(x, y, z, _slope(x, y)); fieldw = op * np.clip((BASE + 0.2 - z) / 0.2, 0, 1)
    else:
        fieldw = np.clip((BASE + 0.2 - z) / 0.2, 0, 1)
    fieldw = fieldw * (1 - np.clip(vf * 2.5, 0, 1)) * (1 - np.clip(td * 3, 0, 1))
    iso = np.sqrt(fc * fa)
    green, wheat = fcell > 0.62, (fcell > 0.32) & (fcell <= 0.62)
    plough = ~green & ~wheat
    fh = hash2(ci.astype(np.int64), cj.astype(np.int64), 143)                  # per field: crop stage
    # colour at several scales: per field, 40 m drifts, 8 m patches
    n40 = vnoise2(X / 40.0 + 3.3, Y / 40.0 - 1.1); n8 = vnoise2(X / 8.0 - 7.0, Y / 8.0 + 2.0)
    if a.biome != "temperate":
        # the Ararat plain in early summer: wheat from green to gold, lucerne and grass fields, brown fallow
        ripe = np.clip(fh * 0.8 + (n40 - 0.5) * 0.6 + (n8 - 0.5) * 0.25, 0, 1)[:, None]
        wcol = np.array([0.15, 0.20, 0.07]) * (1 - ripe) + np.array([0.40, 0.34, 0.15]) * ripe
        gcol = np.array([0.10, 0.16, 0.05]) * (0.8 + 0.4 * n40[:, None]) + np.array([0.04, 0.03, 0.0]) * n8[:, None]
        pcol = np.array([0.24, 0.18, 0.125]) * (0.85 + 0.3 * n8[:, None])
        pcol = np.where((fh > 0.55)[:, None], pcol * 0.6 + np.array([0.12, 0.15, 0.06]) * 0.4, pcol)   # young sprouts
        crop = np.where(green[:, None], gcol, np.where(wheat[:, None], wcol, pcol))
        near_w = (np.clip((0.95 - dist) / 0.45, 0, 1) * fieldw)[:, None]
        alb = alb * (1 - near_w) + crop * near_w
        # grass verges and field margins: green, darker where damp
        mar = np.clip(1 - _edge_dist(x, y) / 2.5, 0, 1) * np.clip((0.95 - dist) / 0.45, 0, 1)
        alb = alb * (1 - 0.7 * mar[:, None]) + np.array([0.11, 0.15, 0.055]) * (0.8 + 0.4 * n8)[:, None] * 0.7 * mar[:, None]
    else:
        alb = alb * (1 + 0.25 * (n40 - 0.5) + 0.15 * (n8 - 0.5))[:, None]
    # crop rows: stripes across the row direction, filtered by the pixel footprint, with raking light on the ridges
    s = np.where(along_u, yr, xr) * 1000
    nx = np.where(along_u, -ROT_S, ROT_C); ny = np.where(along_u, ROT_C, ROT_S)
    ext = _ext(nx, ny, fc, fa, vx, vy)
    period = np.where(green, 0.75, np.where(wheat, 0.2, 0.9))
    amp = _stripe_amp(ext, period) * fieldw
    ph = 2 * np.pi * s / period; st = 0.5 + 0.5 * np.cos(ph)
    soil = np.array([0.19, 0.145, 0.10])
    w_g = (amp * 0.15 * (1 - st) ** 1.5)[:, None] * green[:, None]
    out = alb * (1 - w_g) + soil * w_g
    tilt = -np.sin(ph) * np.where(plough, 0.35, np.where(green, 0.06, 0.12)) * amp            # ridge flanks
    out = out * _rake(tilt, nx, ny)[:, None]
    out = out * (1 + (amp * wheat * 0.18 * (st - 0.5) * 2))[:, None]
    tram = np.abs(np.mod(s / 20.0, 1.0) - 0.5) * 20.0                                      # tractor tramlines in wheat
    tr_ = ((np.abs(tram - 0.9) < 0.25) & wheat) * _stripe_amp(ext, 1.2) * fieldw
    out = out * (1 - 0.3 * tr_)[:, None] + np.array([0.26, 0.21, 0.15]) * (0.3 * tr_)[:, None]
    # clods on fallow
    cl = vnoise2(X / 0.25 + 1.0, Y / 0.25 - 3.0); cl2 = vnoise2((X + SUN_H[0] * 0.08) / 0.25 + 1.0, (Y + SUN_H[1] * 0.08) / 0.25 - 3.0)
    ca = _stripe_amp(iso * 1.4, 0.5) * plough * fieldw
    out = out * (1 + ca * np.clip((cl2 - cl) * 5.0 * RAKE * 0.3, -0.6, 0.6))[:, None]
    # grass and crop seen from a low camera: blades and tufts resolved across the view, smeared along it. The pattern
    # lives in camera-polar coordinates (arc length across, pixel rows down), the way grass reads from eye height
    dm = np.maximum(np.hypot(X - CAM[0] * 1000, Y - CAM[1] * 1000), 0.5)
    th = np.arctan2(Y - CAM[1] * 1000, X - CAM[0] * 1000) * dm                 # metres across the view
    vrow = np.maximum(CAM[2] - z, 0.0005) * 1000 * fl / dm                     # pixel rows below the horizon
    b1 = vnoise2(th / 0.035 + 17.0, vrow / 2.5) - 0.5                         # blades: short upright strokes
    b2 = vnoise2(th / 0.14 - 3.0, vrow / 6.0 + 9.0) - 0.5                     # tufts
    g2 = vnoise2(X / 1.4 - 11.0, Y / 1.4 + 5.0)                                # patches of dry grass and bare soil
    g3 = vnoise2(X / 5.0 + 2.0, Y / 5.0 - 6.0)
    ab1 = _stripe_amp(fc * 1.3, 0.07); ab2 = _stripe_amp(fc * 1.3, 0.28); a2 = _stripe_amp(iso * 1.4, 2.8)
    gw = 1 - 0.35 * fieldw * plough
    out = out * (1 + gw * (ab1 * 0.5 * b1 + ab2 * 0.4 * b2 + a2 * 0.18 * (g2 - 0.5)))[:, None]
    # standing grass: strokes that stand up in the image. A stroke of height h seen from a camera Hc above the ground
    # covers a ground depth of about d h / Hc, so the stroke rows are spaced in log(distance)
    Hc = np.maximum(CAM[2] - z, 0.0005) * 1000
    ldm = np.log(dm)
    grassy_ = 1 - 0.8 * fieldw * plough.astype(float)
    rx_, ry_ = X - CAM[0] * 1000, Y - CAM[1] * 1000
    angv = np.arctan2(rx_ * FH[1] - ry_ * FH[0], rx_ * FH[0] + ry_ * FH[1])  # bearing from the view axis
    band = np.floor(ldm / 0.4); dmb = np.exp((band + 0.5) * 0.4)             # columns stay put within a distance band
    for w_, h_, salt, strength in ((0.15, 0.3, 160, 0.55), (0.04, 0.35, 170, 0.45)):
        ci_ = np.floor(angv * dmb / w_) + band * 100003; off = hash2(ci_.astype(np.int64), 0, salt)
        lv = ldm / (h_ / Hc) + off
        cj_ = np.floor(lv); fv = lv - cj_                                    # 0 at the base (near end), 1 at the tip
        hh = hash2(ci_.astype(np.int64), cj_.astype(np.int64), salt + 1)
        lean = (hash2(ci_.astype(np.int64), cj_.astype(np.int64), salt + 2) - 0.5) * 0.5
        fx = angv * dmb / w_ - (ci_ - band * 100003) - 0.5 - lean * fv
        blade = (np.abs(fx) < 0.42 * (1 - fv) ** 0.6) & (hh < 0.75)
        vis = np.clip((w_ / np.maximum(fc, 1e-5) - 1.0) / 2.0, 0, 1) * grassy_  # fades out below ~1 px wide
        tone = np.where(blade, 0.8 + 0.45 * fv * (0.6 + 0.8 * hh), 0.72)      # tips catch the light, gaps are dark
        out = out * (1 + (tone - 0.85) * strength * 1.6 * vis)[:, None]
        dryb = blade & (hh < 0.08) & (vis > 0)
        out = np.where(dryb[:, None], out * (1 - 0.35 * vis[:, None]) + np.array([0.30, 0.27, 0.14]) * 0.35 * vis[:, None], out)
    # patches in grass and lucerne: clover (dark), seed heads (pale gold), a few metres across
    if a.biome != "temperate":
        gp = vnoise2(X / 2.2 + 9.0, Y / 2.2 - 4.0); a22 = _stripe_amp(iso * 1.4, 2.2)
        grassy = (green | ~(wheat | plough) | (fieldw < 0.5)).astype(float)
        seed_ = np.clip((gp - 0.62) * 4, 0, 1) * a22 * grassy
        clov = np.clip((0.38 - gp) * 4, 0, 1) * a22 * grassy
        out = out * (1 - 0.45 * seed_[:, None]) + np.array([0.30, 0.27, 0.13]) * (1 + 0.5 * b1 * ab1)[:, None] * 0.45 * seed_[:, None]
        out = out * (1 - 0.3 * clov)[:, None]
        # wild flowers in grass and verges: poppies, yellow and white specks, sparse
        fcs = 0.3; fi_, fj_ = np.floor(X / fcs), np.floor(Y / fcs)
        fx_ = X / fcs - fi_ - 0.5; fy_ = Y / fcs - fj_ - 0.5
        hf_ = hash2(fi_.astype(np.int64), fj_.astype(np.int64), 144)
        fl_ = (hf_ < 0.05 * (0.4 + gp)) * np.clip(1 - np.hypot(fx_, fy_) * fcs / 0.035, 0, 1) * _stripe_amp(iso * 2.0, 0.07) * grassy
        fc_ = np.where((hf_ < 0.02)[:, None], np.array([0.55, 0.06, 0.03]), np.where((hf_ < 0.035)[:, None], np.array([0.6, 0.5, 0.08]), np.array([0.6, 0.6, 0.55])))
        fl_ = np.clip(fl_ * 3, 0, 1)
        out = out * (1 - fl_[:, None]) + fc_ * fl_[:, None]
    # wheat heads catch the light: warm speckle in ripening fields
    head = np.clip(b1 * 4, 0, 1) * ab1 * wheat * fieldw
    out = out + np.array([0.10, 0.08, 0.03]) * head[:, None] * 0.6
    dry = np.clip((g2 - 0.55) * 2.5, 0, 1) * a2 * (1 - fieldw) * 0.45
    out = out * (1 - dry[:, None]) + np.array([0.34, 0.29, 0.16]) * (1 + 0.4 * b1 * ab1)[:, None] * dry[:, None]
    bare = np.clip((g3 - 0.64) * 4, 0, 1) * _stripe_amp(iso, 10.0) * (1 - fieldw) * 0.6
    out = out * (1 - bare[:, None]) + np.array([0.26, 0.21, 0.16]) * (1 + 0.3 * b2 * ab2)[:, None] * bare[:, None]
    g1 = 0.5 + b2
    # tufts and weeds: a few per square metre, lit on the sun side, shadowed behind
    tc = 0.45; ti, tj = np.floor(X / tc), np.floor(Y / tc)
    tx = (X / tc - ti - 0.5 - 0.5 * (hash2(ti.astype(np.int64), tj.astype(np.int64), 140) - 0.5)) * tc
    ty = (Y / tc - tj - 0.5 - 0.5 * (hash2(ti.astype(np.int64), tj.astype(np.int64), 141) - 0.5)) * tc
    has = hash2(ti.astype(np.int64), tj.astype(np.int64), 142) < np.where(plough, 0.06, 0.16)
    rt_ = np.hypot(tx, ty); tuft = has * np.clip(1 - rt_ / 0.14, 0, 1)
    ta = _stripe_amp(iso * 1.5, 0.35)
    side = (tx * SUN_H[0] + ty * SUN_H[1]) / np.maximum(rt_, 1e-3)              # +1 sun side, -1 shadow side
    shad = has * np.clip(1 - np.hypot(tx + SUN_H[0] * 0.12, ty + SUN_H[1] * 0.12) / 0.16, 0, 1) * (1 - tuft)
    tcol = np.where(wheat[:, None], np.array([0.26, 0.25, 0.10]), np.array([0.08, 0.11, 0.035]))
    out = out * (1 - (np.clip(tuft * 2, 0, 1) * ta * 0.7)[:, None]) + tcol * (1 + 0.5 * side)[:, None] * (np.clip(tuft * 2, 0, 1) * ta * 0.7)[:, None]
    out = out * (1 - 0.45 * np.clip(shad * 2, 0, 1) * ta)[:, None]
    # dirt tracks along some field edges: two ruts with a grass strip between, lit and shadowed rut walls
    fe, rt_tilt, tn = _edge_track(x, y, fc, fa, vx, vy)
    lane = np.array([0.30, 0.25, 0.19]) * (0.85 + 0.3 * g1)[:, None] * _rake(rt_tilt, tn[:, 0], tn[:, 1])[:, None]
    out = out * (1 - fe[:, None]) + lane * fe[:, None]
    ld_, ln_ = lane_dist(x, y)
    ext = _ext(ln_[:, 0], ln_[:, 1], fc, fa, vx, vy)
    body = np.clip((LANE_W + 0.4 - ld_) / 0.5, 0, 1)
    mid = np.clip(1 - ld_ / (0.25 + 0.2 * vnoise2(X / 1.5, Y / 1.5 + 3)), 0, 1) * _stripe_amp(ext, 0.6)
    wob = 0.12 * (vnoise2(X / 2.5 + 4, Y / 2.5) - 0.5)                        # ruts wander a little
    rd = ld_ - 0.85 - wob
    rutw = np.clip(1 - np.abs(rd) / 0.38, 0, 1) ** 2                         # soft rut profile
    tl = -np.sin(np.clip(rd / 0.38, -1, 1) * np.pi) * 0.1 * _stripe_amp(ext, 0.6) * body
    rut = rutw > 0.3
    pud = np.clip((vnoise2(X / 3.0, Y / 3.0) - 0.8) * 6, 0, 1) * rut * body * _stripe_amp(iso, 1.0)
    lc = np.array([0.36, 0.29, 0.20]) * (0.85 + 0.3 * g1)[:, None] * _rake(tl, ln_[:, 0], ln_[:, 1])[:, None] * (1 - 0.35 * pud)[:, None]
    grass = np.array([0.10, 0.13, 0.05]) * (0.8 + 0.4 * g1)[:, None]
    lc = lc * (1 - 0.18 * rutw * _stripe_amp(ext, 0.6))[:, None]            # ruts: packed, a little darker
    lc = lc * (1 - mid[:, None]) + grass * mid[:, None]
    verge = np.clip((LANE_W + 1.6 - ld_) / 0.6, 0, 1) * (1 - body)         # weedy verge
    out = out * (1 - body[:, None]) + lc * body[:, None]
    out = out * (1 - 0.5 * verge[:, None]) + np.array([0.13, 0.15, 0.06]) * (0.7 + 0.6 * g1)[:, None] * 0.5 * verge[:, None]
    return np.clip(out, 0, 1)

def _edge_dist(x, y):
    """metres to the nearest field edge"""
    u, v = x * ROT_C + y * ROT_S, -x * ROT_S + y * ROT_C
    w = fbm2(x * 1.3, y * 1.3, 3) * 0.6
    su, sv = 0.45 + 0.3 * w, 0.3 + 0.15 * w
    gu, gv = u / su + w, v / sv - w
    return np.minimum(np.abs(gu - np.round(gu)) * su, np.abs(gv - np.round(gv)) * sv) * 1000

LANE_W = 1.7                                                                # half width of a country lane, m
def lane_dist(x, y):
    """metres to the centre line of the winding dirt lanes that link villages across the fields (iso-lines of a
    1 km noise), and the unit direction across the lane"""
    f = lambda x_, y_: fbm2(x_ * 0.9 + 4.0, y_ * 0.9 - 0.75, 3)
    e = 0.004
    n0 = f(x, y); gx = (f(x + e, y) - f(x - e, y)) / (2 * e); gy = (f(x, y + e) - f(x, y - e)) / (2 * e)
    gl = np.maximum(np.hypot(gx, gy), 1e-6)
    return np.abs(n0 - 0.5) / gl * 1000, np.stack([gx / gl, gy / gl], 1) * np.sign(n0 - 0.5)[:, None]

TRACK_P = 0.3                                                               # share of field edges with a farm track
def _edge_track(x, y, fc, fa, vx, vy):
    """cover 0..1 of an earth farm track along a field edge, plus the micro tilt of its ruts and the tilt direction"""
    u, v = x * ROT_C + y * ROT_S, -x * ROT_S + y * ROT_C
    w = fbm2(x * 1.3, y * 1.3, 3) * 0.6
    su, sv = 0.45 + 0.3 * w, 0.3 + 0.15 * w
    gu, gv = u / su + w, v / sv - w
    eu, ev = (gu - np.round(gu)) * su * 1000, (gv - np.round(gv)) * sv * 1000
    rid_u = (np.round(gu) * 7919 + np.floor(gv)).astype(np.int64); rid_v = (np.round(gv) * 104729 + np.floor(gu) + 5e5).astype(np.int64)
    on_u = (hash2(rid_u, 0, 45) < TRACK_P) & (np.abs(eu) < 5.0); on_v = (hash2(rid_v, 0, 45) < TRACK_P) & (np.abs(ev) < 5.0)
    e = np.where(on_u, eu, np.where(on_v, ev, 99.0)) - 2.6                  # signed metres from the track centre line
    tn = np.where(on_u[:, None], np.array([ROT_C, ROT_S]), np.array([-ROT_S, ROT_C]))   # across the track
    ext = np.where(on_u, _ext(ROT_C, ROT_S, fc, fa, vx, vy), _ext(-ROT_S, ROT_C, fc, fa, vx, vy))
    ae = np.abs(e)
    body = np.clip((1.9 - ae) / 0.5, 0, 1)                                   # 3 m of beaten earth, soft edges
    mid = np.clip(1 - ae / 0.35, 0, 1) * _stripe_amp(ext, 0.7)               # grass strip in the middle
    rut = np.clip(1 - np.abs(ae - 0.85) / 0.3, 0, 1)                          # two ruts
    ra = _stripe_amp(ext, 0.6)
    tilt = np.sign(e) * np.where(np.abs(ae - 0.85) < 0.3, -np.sign(ae - 0.85) * 0.35, 0) * ra * body
    cov = np.clip(body * (1 - 0.8 * mid) * (0.85 + 0.15 * rut), 0, 1)
    fade = np.clip((CLUT_KM * 1.6 - np.hypot(x - CAM[0], y - CAM[1])) / 0.2, 0, 1)
    return cov * fade, tilt, tn

def _tracks(x, y, dist, alb, vf):
    """village lanes on the street lines of the house grid (every 5 x 4 plots), dust with wheel ruts"""
    m = np.where(vf > 0.02)[0]
    if not len(m): return alb
    xm, ym = x[m], y[m]
    c = 26.0
    u, v = (xm * ROT_C + ym * ROT_S) * 1000, (-xm * ROT_S + ym * ROT_C) * 1000
    du = np.abs(u - (np.round((u / c - 0.5) / 5) * 5 + 0.5) * c); dv = np.abs(v - (np.round((v / c - 0.5) / 4) * 4 + 0.5) * c)
    d = np.minimum(du, dv); alongv = du < dv
    road = np.clip((2.6 - d) / 0.7, 0, 1) * np.clip(vf[m] * 3, 0, 1)
    fc, fa, vx, vy = _footprint(xm, ym, hf(xm, ym), dist[m])
    ext = np.where(alongv, _ext(ROT_C, ROT_S, fc, fa, vx, vy), _ext(-ROT_S, ROT_C, fc, fa, vx, vy))
    ra = _stripe_amp(ext, 0.9)
    if a.biome == "temperate":
        col = np.array([0.11, 0.11, 0.115]) * (0.9 + 0.2 * vnoise2(u / 3.0, v / 3.0))[:, None]      # asphalt
        col = col * (1 - 0.2 * ra * (np.abs(d - 1.0) < 0.3))[:, None]
    else:
        col = np.array([0.30, 0.26, 0.21]) * (0.85 + 0.3 * vnoise2(u / 2.0, v / 2.0))[:, None]      # dust and gravel
        ruts = np.clip(1 - np.abs(np.abs(d) - 0.85) / 0.3, 0, 1) * ra
        col = col * (1 - 0.3 * ruts)[:, None]
        mid = np.clip(1 - d / 0.35, 0, 1) * ra
        col = col * (1 - mid[:, None] * 0.4) + np.array([0.12, 0.13, 0.07]) * mid[:, None] * 0.4
    out = alb.copy(); w_ = road[:, None]
    out[m] = alb[m] * (1 - w_) + col * w_
    return out

# ------------------------------------------------------------------ instance generation
_near_keys = {k: getattr(a, k) for k in ("peak", "view_from", "cam_height", "cam_offset", "hfov", "layout", "pitch", "horizon", "second",
                                          "trees", "tree_mix", "villages", "near_km", "seed", "biome", "treeline", "season",
                                          "water_level", "water_seed", "city", "city_km")}
_near_keys["code"] = hashlib.md5(open(exec_extras, "rb").read()).hexdigest()[:8]   # placement code changes rebuild the cache
if ROADS is not None: _near_keys["streets"] = ROADS["tag"]
_near_cache = os.path.join(a.work, "near_" + hashlib.md5(json.dumps(_near_keys, sort_keys=True).encode()).hexdigest()[:10] + ".npz")

def _grid_cells(c, rot, rmax):
    cr, sr = rot
    uc, vc = CAM[0] * cr + CAM[1] * sr, -CAM[0] * sr + CAM[1] * cr
    i = np.arange(math.floor((uc - rmax) / c), math.ceil((uc + rmax) / c))
    j = np.arange(math.floor((vc - rmax) / c), math.ceil((vc + rmax) / c))
    out = []
    for i_blk in np.array_split(i, max(1, len(i) // 400)):
        II, JJ = np.meshgrid(i_blk, j, indexing="ij"); II, JJ = II.ravel(), JJ.ravel()
        u, v = (II + 0.5) * c, (JJ + 0.5) * c
        x, y = u * cr - v * sr, u * sr + v * cr
        ok, _ = _in_view(x, y, rmax=rmax)
        out.append((II[ok], JJ[ok]))
    return np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out]), (i[0], j[0], len(i), len(j))

def _mix_weights(z):
    if a.tree_mix != "auto":
        w = np.array([float(v) for v in a.tree_mix.split(",")]); w = w / w.sum()
        return np.tile(w, (len(z), 1))
    if a.biome == "temperate":
        pc = np.clip(0.2 + 0.8 * (z - BASE) / max(TREELINE - BASE, 0.3), 0.15, 0.92)
        return np.stack([pc, 1 - pc, np.zeros_like(pc)], 1)
    return np.tile([0.02, 0.98, 0.0], (len(z), 1))

# house materials: wall 0 plaster, 1 stone blocks, 2 boards; roof 0 tiles, 1 corrugated tin, 2 slate / sheets
if a.biome == "temperate":
    WALLS = [([0.38, 0.37, 0.35], 0), ([0.39, 0.35, 0.29], 0), ([0.33, 0.32, 0.30], 0), ([0.15, 0.105, 0.075], 2),
             ([0.27, 0.26, 0.24], 2), ([0.35, 0.31, 0.26], 0)]
    ROOFS = [([0.10, 0.11, 0.13], 0), ([0.12, 0.125, 0.13], 2), ([0.20, 0.12, 0.08], 1), ([0.08, 0.13, 0.20], 1),
             ([0.27, 0.10, 0.07], 1), ([0.14, 0.15, 0.16], 0)]
else:                                   # Armenian plain: tuff and basalt, plastered walls; tin, tile and sheet roofs
    WALLS = [([0.30, 0.18, 0.14], 1), ([0.33, 0.24, 0.17], 1), ([0.30, 0.26, 0.21], 1), ([0.16, 0.15, 0.15], 1),
             ([0.31, 0.27, 0.21], 0), ([0.33, 0.26, 0.18], 0), ([0.28, 0.265, 0.24], 0), ([0.31, 0.24, 0.21], 0)]
    ROOFS = [([0.25, 0.14, 0.08], 1), ([0.26, 0.13, 0.07], 1), ([0.24, 0.25, 0.26], 1), ([0.33, 0.14, 0.08], 0),
             ([0.35, 0.16, 0.09], 0), ([0.21, 0.21, 0.21], 2), ([0.30, 0.10, 0.07], 1), ([0.10, 0.17, 0.12], 1)]
WALL_C = np.array([w[0] for w in WALLS]); WALL_M = np.array([w[1] for w in WALLS])
ROOF_C = np.array([r[0] for r in ROOFS]); ROOF_M = np.array([r[1] for r in ROOFS])

def build_instances():
    L = {k: [] for k in ("x", "y", "tp", "h", "r", "a", "b", "hw", "hr", "yaw", "seed", "col", "mat")}
    grids = []
    def add(x, y, tp, h, r, a_, b_, hw, hr, yaw, seed, col, mat=None):
        mat = np.zeros(len(np.atleast_1d(x)), np.int16) if mat is None else mat
        for k, v in zip(L, (x, y, tp, h, r, a_, b_, hw, hr, yaw, seed, col, mat)): L[k].append(np.asarray(v))
    n0 = lambda: sum(len(v) for v in L["x"])
    rot0 = (1.0, 0.0); rotF = (ROT_C, ROT_S)
    # --- houses and apartment blocks (placed first; trees avoid them)
    house_ids = None
    if VILLAGES_ON or CITY_ON:
        CHc = 0.026
        I, J, gi = _grid_cells(CHc, rotF, max(NEAR, a.city_km if CITY_ON else 0))
        hu, hv = hash2(I, J, 1), hash2(I, J, 2)
        u, v = (I + 0.5) * CHc, (J + 0.5) * CHc
        x, y = u * ROT_C - v * ROT_S, u * ROT_S + v * ROT_C
        z = hf(x, y); sl = _slope(x, y)
        vf = village_field(x, y) * (a.villages > 0); cf_ = city_field(x, y)
        street = (np.mod(I, 5) == 0) | (np.mod(J, 4) == 0)                   # keep streets free
        occ = np.maximum(np.minimum(vf * 1.2, 0.97), cf_ * 0.5 * np.clip((fbm2(x * 3.0 + 9, y * 3.0 - 4, 3) - 0.3) * 3, 0.15, 1))   # parks, gaps
        if ROADS is not None:
            # real streets: buildings line the roads; plots deep inside big blocks are courtyards and gardens
            osm = cf_ > 0.02
            redge = road_edge(x, y)
            street &= ~osm
            occ = np.where(osm, np.maximum(occ, cf_ * np.where(redge < 32, 0.85, 0.3)) * (redge > 1.0), occ)
        keep = (hash2(I, J, 3) < occ) & _ground_ok(x, y, z, sl) & (sl < 0.35) & ~(street & (cf_ > 0.3))
        keep &= ~street | (cf_ > 0.3) | (hash2(I, J, 18) < 0.08 / 0.62)       # village lanes run along the street lines
        d_ = np.hypot(x - CAM[0], y - CAM[1]); keep &= d_ > 0.03
        x, y, z, hu, hv, I, J, cf_ = x[keep], y[keep], z[keep], hu[keep], hv[keep], I[keep], J[keep], cf_[keep]
        pbig = 0.1 + 0.28 * cf_
        if ROADS is not None:                                                 # apartment blocks front the avenues
            rcl, redge = _rg("cls", x, y), road_edge(x, y)
            front = (cf_ > 0.02) & (redge < 32)
            pbig = pbig + 0.2 * (front & (rcl >= 0) & (rcl <= 4)) - 0.15 * (front & (rcl >= 5))
        big = (cf_ > 0.35) & (hash2(I, J, 4) < pbig)
        h1, h2, h3 = hash2(I, J, 5), hash2(I, J, 6), hash2(I, J, 7)
        A_ = np.where(big, 7 + 5 * h1, 4.0 + 3.2 * h1); B_ = np.where(big, 5.5 + 4 * h2, 3.4 + 1.8 * h2)
        floors = np.where(big, 3 + np.floor(h3 ** 2.2 * 9 * (0.3 + cf_)), 1 + (h3 > 0.55))
        if ROADS is not None:
            # Yerevan: 4-9 storey tuff and concrete blocks, a few taller ones on the avenues
            hb = hash2(I, J, 19)
            floors = np.where(big & (cf_ > 0.02), 4 + np.floor(h3 ** 1.6 * 6) + front * (rcl >= 0) * (rcl <= 3) * (hb < 0.35) * np.floor(1 + 5 * hb / 0.35), floors)
        HW = floors * 3.0 + 0.4; HR = np.where(big, 0.0, B_ * np.tan(np.radians(18 + 16 * h2)))
        yaw = math.atan2(ROT_S, ROT_C) + (hash2(I, J, 8) - 0.5) * np.where(cf_ > 0.3, 0.35, 0.12) + (hash2(I, J, 9) > 0.7) * np.pi / 2
        jit = CHc * 1000 / 2 - np.maximum(A_, B_) - 1.0
        x = x + (hu - 0.5) * 2 * np.maximum(jit, 0) / 1000 * 0.8; y = y + (hv - 0.5) * 2 * np.maximum(jit, 0) / 1000 * 0.8
        if ROADS is not None:
            # facades parallel to the nearest street, set back 3-7 m (8-14 m on avenues) from the kerb
            rd, rw, ra = _rg("dist", x, y, 1), _rg("width", x, y), _rg("ang", x, y)
            nx_, ny_ = _road_normal(x, y)
            setb = np.where((rcl >= 0) & (rcl <= 3), 8 + 6 * hash2(I, J, 20), 3 + 4 * hash2(I, J, 20))
            mv = np.clip(0.5 * rw + setb + B_ - rd, -10, 10) * front
            x = x + mv * nx_ / 1000; y = y + mv * ny_ / 1000
            yaw = np.where(front, ra + (hash2(I, J, 21) - 0.5) * 0.06, yaw)
            clear = road_edge(x, y) > 0.8 * B_                                  # still off the carriageway
            if not clear.all():
                k_ = np.where(clear)[0]
                x, y, z, I, J, cf_, big, A_, B_, HW, HR, yaw, hu, hv = (v[k_] for v in (x, y, z, I, J, cf_, big, A_, B_, HW, HR, yaw, hu, hv))
        wi = (hash2(I, J, 10) * len(WALLS)).astype(int); ri = (hash2(I, J, 12) * len(ROOFS)).astype(int)
        wc = WALL_C[wi] * (0.88 + 0.24 * hash2(I, J, 11))[:, None]
        # apartment blocks: weathered tuff, concrete and painted plaster, darker and more varied than village walls
        blockw = np.array([[0.30, 0.20, 0.17], [0.27, 0.25, 0.22], [0.33, 0.29, 0.23], [0.22, 0.21, 0.20], [0.36, 0.33, 0.30]])
        bw = blockw[(hash2(I, J, 15) * 5).astype(int)] * (0.65 + 0.5 * hash2(I, J, 16))[:, None]
        wc = np.where(big[:, None], bw, wc)
        rc = ROOF_C[ri] * (0.85 + 0.3 * hash2(I, J, 13))[:, None]
        rc = np.where(big[:, None], np.array([0.20, 0.19, 0.18]) * (0.7 + 0.6 * hash2(I, J, 17))[:, None], rc)
        col = np.concatenate([wc, rc], 1)
        mat = (WALL_M[wi] * 10 + ROOF_M[ri]).astype(np.int16)
        start = n0()
        add(x, y, np.where(big, T_BLOCK, T_HOUSE), HW + HR, np.hypot(A_, B_), A_, B_, HW, HR, yaw, hash2(I, J, 14), col, mat)
        ids = np.full((gi[2], gi[3]), -1, np.int32); ids[I - gi[0], J - gi[1]] = start + np.arange(len(x))
        grids.append(dict(c=CHc, rot=rotF, i0=gi[0], j0=gi[1], ids=ids))
        house_ids = grids[-1]
        print(f"near field: {len(x)} buildings", flush=True)
    def in_house_plot(x, y):
        if house_ids is None: return np.zeros(len(x), bool)
        u, v = x * ROT_C + y * ROT_S, -x * ROT_S + y * ROT_C
        hi = np.floor(u / house_ids["c"]).astype(int) - house_ids["i0"]; hj = np.floor(v / house_ids["c"]).astype(int) - house_ids["j0"]
        inb = (hi >= 0) & (hj >= 0) & (hi < house_ids["ids"].shape[0]) & (hj < house_ids["ids"].shape[1])
        occ = np.zeros(len(x), bool); occ[inb] = house_ids["ids"][hi[inb], hj[inb]] >= 0
        return occ
    # --- trees on a jittered grid
    if TREES_ON:
        CT = 0.0085
        I, J, gi = _grid_cells(CT, rot0, NEAR)
        x = (I + 0.1 + 0.8 * hash2(I, J, 21)) * CT; y = (J + 0.1 + 0.8 * hash2(I, J, 22)) * CT
        z = hf(x, y); sl = _slope(x, y); d_ = np.hypot(x - CAM[0], y - CAM[1])
        p = tree_density(x, y, z, sl) * np.clip((NEAR - d_) / (0.3 * NEAR), 0, 1)
        keep = (hash2(I, J, 23) < p) & _ground_ok(x, y, z, sl) & (d_ > 0.025)
        keep &= ~in_house_plot(x, y)                                       # not inside a building plot
        if ROADS is not None: keep &= road_edge(x, y) > 1.5                 # street trees, not trees on the street
        keep &= ~((orchard_field(x, y, z) > 0) & (d_ < ORCH_KM))            # orchards are planted in rows (below)
        x, y, z, I, J, sl = x[keep], y[keep], z[keep], I[keep], J[keep], sl[keep]
        wmix = _mix_weights(z); r_ = hash2(I, J, 24)
        tp = np.where(r_ < wmix[:, 0], T_CON, np.where(r_ < wmix[:, 0] + wmix[:, 1], T_DEC, T_POP))
        h1, h2 = hash2(I, J, 25), hash2(I, J, 26)
        size = 0.75 + 0.5 * fbm2(x * 5 + 2, y * 5 - 1, 2)
        orchard = (a.biome != "temperate") & (village_field(x, y) > 0.05)
        Ht = np.select([tp == T_CON, tp == T_POP, orchard], [13 + 13 * h1, 16 + 11 * h1, 4.5 + 3 * h1], 8 + 8 * h1) * size
        R = np.select([tp == T_CON, tp == T_POP, orchard], [Ht * (0.17 + 0.07 * h2), 1.3 + 0.8 * h2, 2.0 + 1.4 * h2], (2.6 + 2.2 * h2) * size)
        Ht = np.where(tp == T_DEC, np.minimum(Ht, 2.5 + 3.3 * R), Ht)       # no lollipops: a tall tree gets a big crown
        col = tree_colours(tp, hash2(I, J, 27), hash2(I, J, 28))
        start = n0()
        add(x, y, tp, Ht, R, R, R, Ht, 0 * Ht, 0 * Ht, hash2(I, J, 29), np.concatenate([col, col], 1))
        ids = np.full((gi[2], gi[3]), -1, np.int32); ids[I - gi[0], J - gi[1]] = start + np.arange(len(x))
        grids.append(dict(c=CT, rot=rot0, i0=gi[0], j0=gi[1], ids=ids))
        print(f"near field: {len(x)} trees", flush=True)
        # --- orchards: fruit trees on a 5.5 m grid aligned with the fields, clear of the field edges
        if a.biome != "temperate" and ORCH_KM > 0.05:
            CO = 0.0055
            I, J, gi = _grid_cells(CO, rotF, ORCH_KM)
            u, v = (I + 0.5) * CO, (J + 0.5) * CO
            x, y = u * ROT_C - v * ROT_S, u * ROT_S + v * ROT_C
            x = x + (hash2(I, J, 71) - 0.5) * 0.0005; y = y + (hash2(I, J, 72) - 0.5) * 0.0005
            z = hf(x, y); sl = _slope(x, y); d_ = np.hypot(x - CAM[0], y - CAM[1])
            _, _, onu, onv, _, _, _, _ = _edge_cells(I, J, 0.009)            # keep a lane along the field edges
            keep = (orchard_field(x, y, z) > 0) & ~onu & ~onv & _ground_ok(x, y, z, sl) & (sl < 0.2) & (d_ > 0.02)
            keep &= (hash2(I, J, 73) < 0.94) & ~in_house_plot(x, y)
            x, y, I, J = x[keep], y[keep], I[keep], J[keep]
            h1, h2 = hash2(I, J, 74), hash2(I, J, 75)
            Ht = 4.0 + 1.6 * h1; R = 2.1 + 0.6 * h2
            col = tree_colours(np.full(len(x), T_DEC), 0.35 + 0.4 * hash2(I, J, 76), hash2(I, J, 77)) * np.array([1.1, 1.08, 0.95])
            start = n0()
            add(x, y, np.full(len(x), T_DEC), Ht, R, R, R, Ht, 0 * Ht, 0 * Ht, hash2(I, J, 78), np.concatenate([col, col], 1))
            ids = np.full((gi[2], gi[3]), -1, np.int32); ids[I - gi[0], J - gi[1]] = start + np.arange(len(x))
            grids.append(dict(c=CO, rot=rotF, i0=gi[0], j0=gi[1], ids=ids))
            print(f"near field: {len(x)} orchard trees in rows", flush=True)
        # --- poplar rows along field edges (arid valleys; or when the mix asks for poplars)
        want_rows = a.biome != "temperate" or (a.tree_mix != "auto" and _mix_weights(np.zeros(1))[0, 2] > 0)
        if want_rows:
            CP = 0.005
            I, J, gi = _grid_cells(CP, rotF, NEAR)
            x, y, onu, onv, gu, gv, su, sv = _edge_cells(I, J, CP)
            rowid = np.where(onu, np.round(gu) * 7919 + np.floor(gv), np.round(gv) * 104729 + np.floor(gu) + 5e5).astype(np.int64)
            pick = hash2(rowid, 0, 31) < (0.05 + 0.3 * village_field(x, y)) * min(1.0, a.trees * 1.25)
            z = hf(x, y); sl = _slope(x, y); d_ = np.hypot(x - CAM[0], y - CAM[1])
            keep = (onu ^ onv) & pick & _ground_ok(x, y, z, sl) & (z < BASE + 0.45) & (sl < 0.2) & (d_ > 0.03) & (hash2(I, J, 32) < 0.9)
            keep &= (d_ < NEAR * (0.7 + 0.3 * hash2(I, J, 33)))
            x, y = _snap_edge(x, y, onu, gu, gv, su, sv)
            x, y, I, J = x[keep], y[keep], I[keep], J[keep]
            h1, h2 = hash2(I, J, 34), hash2(I, J, 35)
            Ht = 14 + 10 * h1; R = 1.4 + 0.8 * h2
            col = tree_colours(np.full(len(x), T_POP), hash2(I, J, 36), hash2(I, J, 37))
            start = n0()
            add(x, y, np.full(len(x), T_POP), Ht, R, R, R, Ht, 0 * Ht, 0 * Ht, hash2(I, J, 38), np.concatenate([col, col], 1))
            ids = np.full((gi[2], gi[3]), -1, np.int32); ids[I - gi[0], J - gi[1]] = start + np.arange(len(x))
            grids.append(dict(c=CP, rot=rotF, i0=gi[0], j0=gi[1], ids=ids))
            print(f"near field: {len(x)} poplars in rows", flush=True)
    # --- field-edge walls, hedges and fences, close to the camera only: overlapping 3.2 m segments on a 2.5 m grid
    # along the whole field edge, so a run is continuous apart from its gates
    if (TREES_ON or VILLAGES_ON) and CLUT_KM > 0.05:
        CP = 0.0025
        I, J, gi = _grid_cells(CP, rotF, CLUT_KM)
        x, y, onu, onv, gu, gv, su, sv = _edge_cells(I, J, CP)
        (pu_, xu_, yu_), (pv_, xv_, yv_) = _edge_perp(x, y, gu, gv)       # select by true distance to the curved edge
        onu, onv = pu_ < CP * 0.6, pv_ < CP * 0.6
        rowid = np.where(onu, np.round(gu) * 7919 + np.floor(gv), np.round(gv) * 104729 + np.floor(gu) + 5e5).astype(np.int64)
        kind = hash2(rowid, 0, 41)
        pick = kind < 0.55
        z = hf(x, y); sl = _slope(x, y); d_ = np.hypot(x - CAM[0], y - CAM[1])
        along = np.where(onu, gv * sv, gu * su) * 1000                    # metres along the edge
        gate = np.mod(along + hash2(rowid, 0, 42) * 60, 60.0) < 4.0          # a 4 m gate every 60 m
        keep = (onu ^ onv) & pick & ~gate & _ground_ok(x, y, z, sl) & (sl < 0.25) & (d_ > 0.012)
        x, y = np.where(onu, xu_, xv_), np.where(onu, yu_, yv_)             # onto the edge along the gradient
        keep &= ~in_house_plot(x, y)
        yaw = _edge_yaw(x, y, onu)
        # where the field warp folds an edge line into a small loop there is no real boundary: no wall
        base_ = np.where(onu, math.atan2(ROT_C, -ROT_S), math.atan2(ROT_S, ROT_C))
        for s_ in (0.0, -0.015, 0.015):                                    # straight for 15 m either way, or no wall
            y_s = yaw if s_ == 0 else _edge_yaw(x + np.cos(yaw) * s_, y + np.sin(yaw) * s_, onu)
            keep &= np.abs(np.sin(y_s - base_)) < 0.42
        x, y, I, J, kind, onu, yaw = x[keep], y[keep], I[keep], J[keep], kind[keep], onu[keep], yaw[keep]
        # wall material: 0 dry stone, 1 hedge (walls); fences are wooden post-and-rail
        if a.biome == "temperate":
            tp = np.where(kind < 0.3, T_FENCE, T_WALL); mat = np.where(kind < 0.3, 0, 1)
        else:
            tp = np.where(kind < 0.2, T_FENCE, T_WALL); mat = np.where(kind < 0.2, 0, np.where(kind < 0.47, 0, 1))
        h1 = hash2(rowid[keep], 0, 43)                                     # height, width and tint set per run, so
        hgt = np.select([tp == T_FENCE, mat == 1], [1.15 + 0.2 * h1, 1.3 + 0.9 * hash2(rowid[keep], 0, 46)], 0.9 + 0.3 * h1)
        bw = np.select([tp == T_FENCE, mat == 1], [0.06, 0.45 + 0.25 * h1], 0.28 + 0.08 * h1)   # overlapping segments match
        A_ = np.full(len(x), CP * 1000 * 0.85)                               # 4.3 m segments every ~2.5 m: no gaps on curves
        if a.biome == "temperate":
            cw = np.where((mat == 1)[:, None], FOLIAGE * 1.5, np.array([0.20, 0.19, 0.18]))
        else:
            cw = np.where((mat == 1)[:, None], FOLIAGE * np.array([1.5, 1.45, 1.2]), np.array([0.12, 0.112, 0.108]))
        cw = np.where((tp == T_FENCE)[:, None], np.array([0.24, 0.215, 0.19]), cw) * (0.85 + 0.3 * hash2(rowid[keep], 0, 47))[:, None]
        start = n0()
        add(x, y, tp, hgt, np.hypot(A_, bw), A_, bw, hgt, 0 * hgt, yaw, hash2(rowid[keep], 0, 48), np.concatenate([cw, cw], 1), mat.astype(np.int16))
        ids = np.full((gi[2], gi[3]), -1, np.int32); ids[I - gi[0], J - gi[1]] = start + np.arange(len(x))
        grids.append(dict(c=CP, rot=rotF, i0=gi[0], j0=gi[1], ids=ids, walls=True))
        print(f"near field: {len(x)} wall, hedge and fence segments", flush=True)
    # --- yard walls and fences on the plot boundaries next to village houses, close to the camera only
    if VILLAGES_ON and house_ids is not None and CLUT_KM > 0.05:
        CY = 0.026 / 5
        I, J, gi = _grid_cells(CY, rotF, CLUT_KM)
        onI, onJ = np.mod(I, 5) == 0, (np.mod(J, 5) == 0) & (np.mod(I, 5) != 0)
        u, v = (I + 0.5) * CY, (J + 0.5) * CY
        u = np.where(onI, I * CY, u); v = np.where(onJ, J * CY, v)                 # onto the plot boundary line
        x, y = u * ROT_C - v * ROT_S, u * ROT_S + v * ROT_C
        hc = house_ids["c"]; ids_h = house_ids["ids"]
        def occ_at(uu, vv):
            hi = np.floor(uu / hc).astype(int) - house_ids["i0"]; hj = np.floor(vv / hc).astype(int) - house_ids["j0"]
            inb = (hi >= 0) & (hj >= 0) & (hi < ids_h.shape[0]) & (hj < ids_h.shape[1])
            o = np.zeros(len(uu), bool); o[inb] = ids_h[hi[inb], hj[inb]] >= 0
            return o
        e = 0.001
        nextto = np.where(onI, occ_at(u - e, v) | occ_at(u + e, v), occ_at(u, v - e) | occ_at(u, v + e))
        bid = np.where(onI, I * 100003 + np.floor(J / 5), J * 100019 + np.floor(I / 5) + 7e6).astype(np.int64)  # one plot side
        z = hf(x, y); sl = _slope(x, y); d_ = np.hypot(x - CAM[0], y - CAM[1])
        pos = np.where(onI, np.mod(J, 5), np.mod(I, 5))                        # 5 cells per plot side; one is the gate
        gate = pos == (hash2(bid, 0, 52) * 5).astype(int)
        keep = (onI | onJ) & nextto & (hash2(bid, 0, 51) < 0.6) & ~gate & _ground_ok(x, y, z, sl) & (d_ > 0.012)
        x, y, I, J, onI, bid = x[keep], y[keep], I[keep], J[keep], onI[keep], bid[keep]
        yaw = np.where(onI, math.atan2(ROT_C, -ROT_S), math.atan2(ROT_S, ROT_C))
        kind = hash2(bid, 0, 53)
        if a.biome == "temperate":
            tp = np.where(kind < 0.5, T_FENCE, T_WALL); mat = np.where(kind < 0.5, 0, 1)
        else:
            tp = np.where(kind < 0.35, T_FENCE, T_WALL); mat = np.zeros(len(x), int)
        h1 = hash2(I, J, 54)
        hgt = np.select([tp == T_FENCE, mat == 1], [1.2 + 0.25 * hash2(bid, 0, 55), 1.4 + 0.5 * hash2(bid, 0, 56)], 0.95 + 0.25 * hash2(bid, 0, 57))
        bw = np.select([tp == T_FENCE, mat == 1], [0.06, 0.5], 0.25 + 0.05 * h1)
        A_ = np.full(len(x), CY * 1000 / 2 + 0.12)
        # muted grey, ochre and tuff-rose field stone
        stonec = np.array([[0.125, 0.12, 0.112], [0.135, 0.122, 0.10], [0.135, 0.115, 0.105], [0.11, 0.106, 0.10]])[(hash2(bid, 0, 58) * 4).astype(int)]
        cw = np.where((mat == 1)[:, None], FOLIAGE * 1.5, stonec)
        cw = np.where((tp == T_FENCE)[:, None], np.array([0.24, 0.215, 0.19]), cw) * (0.85 + 0.3 * hash2(bid, 0, 59))[:, None]
        start = n0()
        add(x, y, tp, hgt, np.hypot(A_, bw), A_, bw, hgt, 0 * hgt, yaw, hash2(I, J, 60), np.concatenate([cw, cw], 1), mat.astype(np.int16))
        ids = np.full((gi[2], gi[3]), -1, np.int32); ids[I - gi[0], J - gi[1]] = start + np.arange(len(x))
        grids.append(dict(c=CY, rot=rotF, i0=gi[0], j0=gi[1], ids=ids, walls=True))
        print(f"near field: {len(x)} yard wall and fence segments", flush=True)
    if not L["x"]:
        return {k: np.zeros(0) for k in L}, grids
    inst = {k: np.concatenate(v) for k, v in L.items()}
    inst["col"] = inst["col"].reshape(-1, 6)
    inst["tp"] = inst["tp"].astype(np.int8)
    inst["z"] = terrain_z(inst["x"], inst["y"])
    if a.cam_height < 0.03:                                               # keep the lens out of the foliage
        d_ = np.hypot(inst["x"] - CAM[0], inst["y"] - CAM[1]) * 1000
        clear = np.where(inst["tp"] <= T_POP, np.maximum(inst["r"] + 6, 2.2 * inst["h"]), inst["r"] + 6)
        inst["h"] = np.where(d_ < clear, 0, inst["h"])
    # nothing stands on the country lanes (they are drawn within about a kilometre)
    near_ = np.where(np.hypot(inst["x"] - CAM[0], inst["y"] - CAM[1]) < CLUT_KM * 1.6)[0]
    if len(near_):
        ld_, _ = lane_dist(inst["x"][near_], inst["y"][near_])
        tpn = inst["tp"][near_]
        clr = np.select([tpn <= T_POP, (tpn == T_HOUSE) | (tpn == T_BLOCK)], [0.5 * inst["r"][near_] + 0.6, 0.8 * inst["r"][near_] + 1.5], 0.6)
        inst["h"][near_[ld_ < LANE_W + clr]] = 0
    gone = np.where(inst["h"] <= 0)[0]
    for g in grids:                                                       # hidden instances cast no shadows either
        g["ids"][np.isin(g["ids"], gone)] = -1
    return inst, grids

def _edge_cells(I, J, c):
    """cells of a rotated grid that sit on a field-edge line (the same cell maths as the field colours)"""
    u, v = (I + 0.5) * c, (J + 0.5) * c
    x, y = u * ROT_C - v * ROT_S, u * ROT_S + v * ROT_C
    w = fbm2(x * 1.3, y * 1.3, 3) * 0.6
    su, sv = 0.45 + 0.3 * w, 0.3 + 0.15 * w
    gu, gv = u / su + w, v / sv - w
    du, dv = np.abs(gu - np.round(gu)) * su, np.abs(gv - np.round(gv)) * sv
    return x, y, du < c / 2, dv < c / 2, gu, gv, su, sv

def _edge_yaw(x, y, onu):
    """direction of the field-edge line through (x, y): perpendicular to the gradient of the field coordinate"""
    def g(x_, y_):
        u, v = x_ * ROT_C + y_ * ROT_S, -x_ * ROT_S + y_ * ROT_C
        w = fbm2(x_ * 1.3, y_ * 1.3, 3) * 0.6
        return np.where(onu, u / (0.45 + 0.3 * w) + w, v / (0.3 + 0.15 * w) - w)
    e = 0.002
    gx = (g(x + e, y) - g(x - e, y)) / (2 * e); gy = (g(x, y + e) - g(x, y - e)) / (2 * e)
    return np.arctan2(gx, -gy)                                            # tangent = (-gy, gx)

def _edge_perp(x, y, gu, gv):
    """true perpendicular distance (km) from each point to the nearest u-edge and v-edge line, and the point moved onto
    each line along the gradient: edges curve, so a distance measured along a grid axis breaks runs into dashes"""
    def g(x_, y_):
        u, v = x_ * ROT_C + y_ * ROT_S, -x_ * ROT_S + y_ * ROT_C
        w = fbm2(x_ * 1.3, y_ * 1.3, 3) * 0.6
        return u / (0.45 + 0.3 * w) + w, v / (0.3 + 0.15 * w) - w
    e = 0.002
    (uxp, vxp), (uxm, vxm) = g(x + e, y), g(x - e, y); (uyp, vyp), (uym, vym) = g(x, y + e), g(x, y - e)
    out = []
    for f0, gx, gy in ((gu, (uxp - uxm) / (2 * e), (uyp - uym) / (2 * e)), (gv, (vxp - vxm) / (2 * e), (vyp - vym) / (2 * e))):
        g2 = np.maximum(gx * gx + gy * gy, 1e-9); df = f0 - np.round(f0)
        out.append((np.abs(df) / np.sqrt(g2), x - df * gx / g2, y - df * gy / g2))
    return out

def _snap_edge(x, y, onu, gu, gv, su, sv):
    """move a cell centre onto its field edge so rows and walls are straight"""
    xs = np.where(onu, x - (gu - np.round(gu)) * su * ROT_C, x + (gv - np.round(gv)) * sv * ROT_S)
    ys = np.where(onu, y - (gu - np.round(gu)) * su * ROT_S, y - (gv - np.round(gv)) * sv * ROT_C)
    return xs, ys

def tree_colours(tp, h1, h2):
    c = np.tile(np.array([0.028, 0.048, 0.030]), (len(tp), 1))              # conifers: dark blue-green
    c[tp == T_DEC] = FOLIAGE * 1.1; c[tp == T_POP] = FOLIAGE * np.array([1.05, 1.12, 1.05])
    warm, cool = np.array([1.16, 1.07, 0.74]), np.array([0.88, 0.98, 1.12])  # yellow-green ... blue-green
    k = np.where(tp == T_CON, 0.4, 1.0)[:, None]
    tint = 1 + k * ((warm - 1) * (1 - h2)[:, None] + (cool - 1) * h2[:, None])
    c = c * (0.75 + 0.5 * h1)[:, None] * tint
    if a.season == "autumn":
        aut = np.array([[0.20, 0.09, 0.025], [0.24, 0.17, 0.03], [0.15, 0.05, 0.02]])[np.minimum((h2 * 3).astype(int), 2)]
        m = (tp != T_CON) & (h2 > 0.2)
        c[m] = aut[m] * (0.7 + 0.6 * h1[m, None])
    return c

if os.path.exists(_near_cache):
    _z = np.load(_near_cache, allow_pickle=True); INST = _z["inst"].item(); GRIDS = list(_z["grids"])
else:
    INST, GRIDS = build_instances()
    np.savez(_near_cache, inst=np.array(INST, dtype=object), grids=np.array(GRIDS, dtype=object))
NI = len(INST["x"])
HMAX = float(INST["h"].max()) if NI else 0.0
IS_TREE = INST["tp"] <= T_POP if NI else np.zeros(0, bool)
INST_FRONT = np.ones(NI)                                                  # lit-window share: 1 on a street, 0.3 in the block
if ROADS is not None and NI:
    _b = (INST["tp"] == T_HOUSE) | (INST["tp"] == T_BLOCK)
    INST_FRONT[_b] = np.where(road_edge(INST["x"][_b], INST["y"][_b]) < 32, 1.0, 0.2)

# screen bounding boxes
def _project(P):
    v = P - CAM; zc = v @ fwd
    xs = (v @ right) / np.maximum(zc, 1e-6) * fl + W / 2 - 0.5
    ys = -(v @ up) / np.maximum(zc, 1e-6) * fl + H / 2 + H * 0.02 - 0.5
    return xs, ys, zc
def _bound_r(): return np.where(IS_TREE, INST["r"] * 1.35 + 0.3, INST["r"] + 0.9) / 1000
def _bound_top(): return np.where(IS_TREE, INST["h"] * 1.03 + 0.3, INST["h"] * 1.03 + (INST["tp"] == T_HOUSE) * 1.0) / 1000   # + chimney
if NI:
    _rb = _bound_r(); _tb = _bound_top()
    _corn = [(sx_, sy_, sz_) for sx_ in (-1, 1) for sy_ in (-1, 1) for sz_ in (0, 1)]
    XS, YS, ZC = [], [], []
    for sx_, sy_, sz_ in _corn:
        P = np.stack([INST["x"] + sx_ * _rb, INST["y"] + sy_ * _rb, INST["z"] + (_tb if sz_ else -0.002)], 1)
        xs_, ys_, zc_ = _project(P); XS.append(xs_); YS.append(ys_); ZC.append(zc_)
    XS, YS, ZC = np.array(XS), np.array(YS), np.array(ZC)
    BB = np.stack([np.floor(XS.min(0)), np.ceil(XS.max(0)) + 1, np.floor(YS.min(0)), np.ceil(YS.max(0)) + 1], 1).astype(np.int64)
    VIS = (ZC.min(0) > 0.004) & (BB[:, 0] < W) & (BB[:, 1] > 0) & (BB[:, 2] < H) & (BB[:, 3] > 0) & (INST["h"] > 0)
    BB[:, 0] = np.clip(BB[:, 0], 0, W); BB[:, 1] = np.clip(BB[:, 1], 0, W); BB[:, 2] = np.clip(BB[:, 2], 0, H); BB[:, 3] = np.clip(BB[:, 3], 0, H)
    DMIN = np.hypot(INST["x"] - CAM[0], INST["y"] - CAM[1]) - _rb
    DI_M = np.maximum(np.hypot(INST["x"] - CAM[0], INST["y"] - CAM[1]) * 1000, 1.0)   # metres to each instance
    SPX = INST["r"] / DI_M * fl                                                      # radius in px
    # level of detail: crown clumps (~1.6 m) and limbs appear once they span a few pixels; far trees stay solid
    LODT = np.where(IS_TREE, np.clip((fl / DI_M * 1.6 - 2.5) / 6.0, 0, 1.0), 0.0)
    _leaf = np.clip((fl / DI_M * 0.095 - 1.5) / 1.5, 0, 1)                    # single leaves once a leaf spans 1.5 px
    LODT = np.where(IS_TREE & (_leaf > 0), 1 + _leaf, LODT)                       # > 1: leaf level
    VIS &= ~((INST["tp"] >= T_WALL) & (INST["h"] / DI_M * fl < 3.0))              # walls under 3 px tall drop out
    LODT = np.maximum(LODT, np.where(IS_TREE, 0.4 * np.clip(SPX / 2.5 - 0.5, 0, 1), 0))  # lumpy crowns once a crown spans a few px
    print(f"near field: {int(VIS.sum())} instances in view, {int(((BB[:, 1] - BB[:, 0]) * (BB[:, 3] - BB[:, 2]))[VIS].sum())} bbox px", flush=True)

if NI and WATER:                                                      # bounding boxes of the mirror images
    _rb2 = _bound_r()
    XS, YS = [], []
    for sx_, sy_, sz_ in _corn:
        x_, y_ = INST["x"] + sx_ * _rb2, INST["y"] + sy_ * _rb2
        zw = WL - ((x_ - CAM[0]) ** 2 + (y_ - CAM[1]) ** 2) / (2 * RE)
        zt = INST["z"] + (_tb if sz_ else -0.002)
        xs_, ys_, _ = _project(np.stack([x_, y_, 2 * zw - zt], 1)); XS.append(xs_); YS.append(ys_)
    XS, YS = np.array(XS), np.array(YS)
    MBB = np.stack([np.floor(XS.min(0)) - 1, np.ceil(XS.max(0)) + 2, np.floor(YS.min(0)) - 1, np.ceil(YS.max(0)) + 2], 1).astype(np.int64)
    MVIS = (ZC.min(0) > 0.004) & (MBB[:, 0] < W) & (MBB[:, 1] > 0) & (MBB[:, 2] < H) & (MBB[:, 3] > 0) & (INST["h"] > 0)
    MVIS &= INST["z"] < WL + 0.08                                           # only things near the water line reflect in view
    for k_ in range(4): MBB[:, k_] = np.clip(MBB[:, k_], 0, W if k_ < 2 else H)

# ------------------------------------------------------------------ shapes (local metres, z up from the base)
def _local(O, D, ids):
    """ray origin/direction in each instance's local frame (metres; houses rotated by yaw)"""
    o = (O - np.stack([INST["x"][ids], INST["y"][ids], INST["z"][ids]], 1)) * 1000
    yaw = INST["yaw"][ids]; c, s = np.cos(yaw), np.sin(yaw)
    ox = o[:, 0] * c + o[:, 1] * s; oy = -o[:, 0] * s + o[:, 1] * c
    dx = D[:, 0] * c + D[:, 1] * s; dy = -D[:, 0] * s + D[:, 1] * c
    return np.stack([ox, oy, o[:, 2]], 1), np.stack([dx, dy, D[:, 2]], 1)

_nr = np.random.default_rng(5)
NTEX = gaussian_filter(_nr.random((48, 48, 48)).astype(np.float32), 1.6, mode="wrap")
NTEX = ((np.argsort(np.argsort(NTEX.ravel())) + 0.5) / NTEX.size).reshape(NTEX.shape).astype(np.float32)   # uniform 0..1
def tnoise(x, y, z):
    """cheap tileable 3-D value noise (trilinear lookup of a smoothed random volume); period 48"""
    x, y, z = np.broadcast_arrays(x, y, z)
    return map_coordinates(NTEX, [x, y, z], order=1, mode="grid-wrap")
# crown noise: lobes, clumps and sprays baked into one volume, so one lookup gives all three octaves
_cr = np.random.default_rng(9)
def _uniform(v): return ((np.argsort(np.argsort(v.ravel())) + 0.5) / v.size).reshape(v.shape).astype(np.float32)
# crown noise: two channels (lobes ~1.7 m, clumps ~0.6 m at CK cells per metre), so each scale gets its own depth
CTEX = np.stack([_uniform(gaussian_filter(_cr.random((64, 64, 64)), s_, mode="wrap")) for s_ in (4.0, 1.5)], -1)
CK = 2.4                                                                    # crown-noise cells per metre
CARVE_M = 1.6                                                               # metres per unit of crown noise
LEAF_LOD = False                                   # single leaves are shading only (a carve at leaf scale leaves films)

def _crown_rv(R, Ht):
    """vertical semi-axis of a deciduous crown: at least 36% of the tree height, so tall trees are not lollipops"""
    return np.minimum(np.maximum(R * 0.95, 0.36 * Ht), (Ht - 1.2) / 2)

def _surf_point(q, tp, Ht, R):
    """q projected onto the crown's reference surface along the direction from the crown centre (deciduous) or
    from the trunk axis (conifer, poplar). Noise sampled there depends on direction only, so a crown displaced by
    it stays star-shaped: no clump can float free of the rest"""
    Rv = _crown_rv(R, Ht); zc = Ht * 1.02 - 1.26 * Rv
    dx, dy, dz = q[:, 0] / R, q[:, 1] / R, (q[:, 2] - zc) / Rv
    ln = np.maximum(np.sqrt(dx * dx + dy * dy + dz * dz), 1e-6)
    rr = np.maximum(np.hypot(q[:, 0], q[:, 1]), 1e-6)
    dec = tp == T_DEC
    return (np.where(dec, dx / ln * R, q[:, 0] / rr * R), np.where(dec, dy / ln * R, q[:, 1] / rr * R),
            np.where(dec, zc + dz / ln * Rv, q[:, 2]))

def _crown_noise(q, tp, Ht, R, seed, m):
    """crown noise on the reference surface: (lobes, clumps), each uniform 0..1"""
    n = np.full((len(q), 2), 0.5)
    if m.any():
        tpm, Rm = tp[m], R[m]
        px, py, pz = _surf_point(q[m], tpm, Ht[m], Rm)
        k = CK * np.clip(3.5 / Rm, 0.6, 1.6) ** 0.5
        sz = np.where(tpm == T_POP, 0.35, np.where(tpm == T_CON, 0.8, 1.0))
        so = seed[m] * 211.0
        c = [px * k + so, py * k - so * 0.7, pz * k * sz + so * 0.3]
        n[m, 0] = map_coordinates(CTEX[..., 0], c, order=1, mode="grid-wrap")
        n[m, 1] = map_coordinates(CTEX[..., 1], c, order=1, mode="grid-wrap")
    return n

def _crown_ellipsoid(tp, Ht, R):
    """centre height and radii of an ellipsoid standing in for the crown (shadows, light inside the crown)"""
    Rv_d = _crown_rv(R, Ht)
    h0 = Ht * 0.1
    zc = np.select([tp == T_DEC, tp == T_CON], [Ht * 1.02 - 1.26 * Rv_d, h0 + (Ht - h0) * 0.38], Ht * 0.53)
    Rh = np.select([tp == T_DEC, tp == T_CON], [R, R * 0.8], R)
    Rv = np.select([tp == T_DEC, tp == T_CON], [Rv_d, (Ht - h0) * 0.62], Ht * 0.47)
    return zc, Rh, Rv

def _capsule(p, a0, b0, r0, r1):
    ba = b0 - a0; pa = p - a0
    h = np.clip((pa * ba).sum(1) / np.maximum((ba * ba).sum(1), 1e-9), 0, 1)
    return np.linalg.norm(pa - ba * h[:, None], axis=1) - (r0 + (r1 - r0) * h)

NCLUMP = 14
def _smin(a_, b_, k):
    h = np.clip(0.5 + 0.5 * (b_ - a_) / k, 0, 1)
    return b_ * (1 - h) + a_ * h - k * h * (1 - h)

_CLT = {}
def _clump_table():
    """clump centres (local metres) and radii for every deciduous instance, computed once"""
    if "c" not in _CLT:
        R, Ht, sd = INST["r"], INST["h"], INST["seed"]
        Rv = _crown_rv(R, Ht); zc = Ht * 1.02 - 1.26 * Rv; mn = np.minimum(R, Rv)
        si = (sd * 1e5).astype(np.int64)
        C = np.zeros((len(R), NCLUMP, 3), np.float32); RC = np.zeros((len(R), NCLUMP), np.float32)
        for k in range(NCLUMP):
            h1, h2, h3 = hash2(si, k, 150), hash2(si, k, 151), hash2(si, k, 152)
            cz = np.clip(1 - (k + 0.5) / NCLUMP * 1.85 + (h1 - 0.5) * 0.15, -0.85, 1)
            ph = k * 2.39996 + sd * 40 + (h2 - 0.5) * 0.8
            rho = np.sqrt(np.maximum(1 - cz * cz, 0)); reach = 0.64 + 0.14 * h3
            C[:, k] = np.stack([np.cos(ph) * rho * reach * R, np.sin(ph) * rho * reach * R, zc + cz * reach * Rv], 1)
            RC[:, k] = mn * (0.3 + 0.1 * h1)
        _CLT["c"], _CLT["r"] = C, RC
    return _CLT["c"], _CLT["r"]

def _crown_clumps(q, ids, R, sd, mn, zc, Rv):
    """deciduous crown as a smooth union of a core and NCLUMP leaf clumps; branches from the trunk top to each clump.
    returns (foliage field, wood field) in metres"""
    CT, RT_ = _clump_table()
    N = len(q); f = np.empty(N); fw = np.empty(N)
    for c0 in range(0, N, 60000):
        sl_ = slice(c0, min(N, c0 + 60000)); qq = q[sl_].astype(np.float32); ii = ids[sl_]
        C = CT[ii]; RC = RT_[ii]; Rq = R[sl_]; zq = zc[sl_]; Rvq = Rv[sl_]; mq = mn[sl_]
        rt = (0.12 + 0.04 * Rq).astype(np.float32)
        core = (np.sqrt((qq[:, 0] / (0.6 * Rq)) ** 2 + (qq[:, 1] / (0.6 * Rq)) ** 2 + ((qq[:, 2] - zq) / (0.6 * Rvq)) ** 2) - 1) * 0.6 * mq * 0.85
        dv = qq[:, None, :] - C
        D = np.sqrt((dv * dv).sum(2)) - RC                                   # (n, NCLUMP)
        kk = 0.25
        m0 = np.minimum(core, D.min(1))
        fs = m0 - kk * np.log(np.exp(-(core - m0) / kk) + np.exp(-(D - m0[:, None]) / kk).sum(1))   # smooth union
        kb = D.argmin(1); r_ = np.arange(len(kb))
        cb = C[r_, kb]; rb_ = RC[r_, kb]; dbest = D[r_, kb]
        # branches: capsules from the trunk top to 82% of the way to each clump
        root = np.stack([np.zeros(len(qq)), np.zeros(len(qq)), zq - 0.7 * Rvq], 1).astype(np.float32)
        ba = (C - root[:, None, :]) * 0.82; pa = qq[:, None, :] - root[:, None, :]
        h = np.clip((pa * ba).sum(2) / np.maximum((ba * ba).sum(2), 1e-6), 0, 1)
        e_ = pa - ba * h[..., None]
        fwq = (np.sqrt((e_ * e_).sum(2)) - rt[:, None] * (0.55 - 0.37 * h)).min(1)
        # leaf sprays: bumps on the nearest clump, sampled by direction from its centre, so they stay attached to it
        u = qq - cb; u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-6)
        ps = u * rb_[:, None] * 4.5 + (kb * 13.1 + sd[sl_] * 90)[:, None]
        nsp = map_coordinates(NTEX, [ps[:, 0], ps[:, 1], ps[:, 2]], order=1, mode="grid-wrap")
        ps2 = u * rb_[:, None] * 11.0 + (kb * 7.7 - sd[sl_] * 50)[:, None]
        nsp2 = map_coordinates(NTEX, [ps2[:, 0], ps2[:, 1], ps2[:, 2]], order=1, mode="grid-wrap")
        on = np.clip(1 - np.abs(dbest - fs) / 0.3, 0, 1)
        f[sl_] = fs - on * ((nsp - 0.5) * 2 * 0.4 + (nsp2 - 0.5) * 2 * 0.15)
        fw[sl_] = fwq
    return f, fw

def tree_field(q, tp, Ht, R, seed, lod, ids=None):
    """signed-ish distance (m): < 0 inside foliage or wood. lod 0..1 blends from a smooth crown to clumps with gaps.
    returns (field, wood flag, crown noise, envelope field)"""
    rr = np.hypot(q[:, 0], q[:, 1]); z = q[:, 2]; N = len(rr)
    if lod is None: lod = np.zeros(N)
    lodf = lod; lod = np.minimum(lod, 1.0)
    nn = _crown_noise(q, tp, Ht, R, seed, lod > 0)
    n = nn[:, 1]
    # displacement in metres, each scale no deeper than it is wide, so dents stay dents and never become crevices
    disp = lod * ((nn[:, 0] - 0.5) * 2 * np.minimum(0.16 * R, 0.9) + (nn[:, 1] - 0.5) * 2 * 0.3)
    dn = disp / np.maximum(R, 0.5)                                           # as a fraction of the crown radius
    fb = np.full(N, 1e3); shell = np.ones(N)
    m = tp == T_DEC
    wood_b = np.full(N, 1e3)
    if m.any():
        Rm, Hm = R[m], Ht[m]
        Rv = _crown_rv(Rm, Hm); zc = Hm * 1.02 - 1.26 * Rv
        zz = (z[m] - zc) / Rv; zz = np.where(zz < 0, zz * 1.05, zz)          # flatter underside
        mn = np.minimum(Rm, Rv)
        fe = (np.sqrt((rr[m] / Rm) ** 2 + zz ** 2) - 1) * mn * 0.85
        fb[m] = fe - disp[m]
        shell[m] = 0.7 + 0.25 * mn
        # resolved crowns: clumps at the ends of branches around a dense core. Every clump hangs on a branch
        # from the top of the trunk, so nothing floats, and sky shows between clumps at the crown edge
        w = np.clip((lod[m] - 0.45) / 0.35, 0, 1) * (ids is not None)
        k_ = np.where((w > 0) & (fe < 0.8))[0]
        if len(k_):
            mi = np.where(m)[0][k_]
            fc_, fw_ = _crown_clumps(q[mi], ids[mi], R[mi], seed[mi], mn[k_], zc[k_], Rv[k_])
            fcl = fc_ - lod[mi] * (nn[mi, 1] - 0.5) * 2 * 0.22
            # ragged clump surfaces: dents only (they cannot cut a clump loose), two octaves, ~0.6 m and ~0.25 m
            qq = q[mi]; so = seed[mi] * 77.0
            d1 = map_coordinates(NTEX, [qq[:, 0] * 4.0 + so, qq[:, 1] * 4.0 - so, qq[:, 2] * 4.0], order=1, mode="grid-wrap")
            d2 = map_coordinates(NTEX, [qq[:, 0] * 10.0 - so, qq[:, 1] * 10.0 + so, qq[:, 2] * 10.0 + 5], order=1, mode="grid-wrap")
            near_s = np.clip(1 - np.abs(fcl) / 0.8, 0, 1)
            fcl = fcl + near_s * (0.3 * np.clip(0.55 - d1, 0, 1)) * lod[mi]
            fb[mi] = fb[mi] * (1 - w[k_]) + fcl * w[k_]
            wood_b[mi] = fw_ + (1 - w[k_]) * 1e3
    m = tp == T_CON
    if m.any():
        Rm, Hm, sd = R[m], Ht[m], seed[m]
        h0 = Hm * 0.1; u = (z[m] - h0) / (Hm - h0); uc = np.clip(u, 0, 1)
        nt = 6 + np.floor(np.mod(sd * 13.7, 1.0) * 6)                      # whorls of branches
        tt = uc * nt + sd * 3; kk = np.floor(tt); s = tt - kk
        whorl = (0.5 + 0.5 * (1 - s) ** 1.6) * (0.8 + 0.2 * np.clip(s / 0.08, 0, 1))   # widest at the drooping tips
        l_ = lod[m]
        nb = 5 + np.floor(np.mod(sd * 7.7, 1.0) * 3)                       # branches per whorl
        br = (0.5 + 0.5 * np.cos(nb * np.arctan2(q[m, 1], q[m, 0]) + kk * 2.39 + sd * 40)) ** 0.7
        tj = 0.82 + 0.3 * hash2(kk.astype(np.int64), (sd * 1e4).astype(np.int64), 120)   # whorls differ in reach
        prof = Rm * (1 - uc) ** 0.9 * (0.55 + 0.45 * whorl) * (1 + l_ * ((tj - 1) + 0.3 * (br - 0.7) + dn[m] * 0.5))
        fb[m] = np.where((u >= 0) & (u <= 1), rr[m] - prof, np.maximum(-u, u - 1) * (Hm - h0) + 0.2)
        shell[m] = 0.35 + 0.18 * Rm
    m = tp == T_POP
    if m.any():
        Rm, Hm = R[m], Ht[m]
        Rv = Hm * 0.47; zc = Hm - Rv; v = (z[m] - zc) / Rv
        prof = Rm * np.sqrt(np.clip(1 - v * v, 0, 1)) * (1 - 0.3 * v - 0.35 * np.clip(v, 0, 1) ** 2) * (1 + dn[m] * 0.8)
        fb[m] = np.where(np.abs(v) < 1, rr[m] - prof, np.abs(v) * Rv - Rv + 0.2)
        shell[m] = 0.3 + 0.25 * Rm
    f = fb.copy()
    # twigs and leaf sprays: a finer carve of the outer half metre, once 0.4 m spans a few pixels
    l2 = np.clip((lod - 0.55) / 0.45, 0, 1)
    m2 = np.where((l2 > 0) & (f > -0.45) & (f < 0.3))[0]
    if len(m2):
        so = seed[m2] * 131.0
        px, py, pz = _surf_point(q[m2], tp[m2], Ht[m2], R[m2])               # sprays: dents, never holes or films
        n2 = map_coordinates(NTEX, [px * 9.0 + so, py * 9.0 - so, pz * 9.0 * np.where(tp[m2] == T_POP, 0.5, 1.0)],
                             order=1, mode="grid-wrap")
        f[m2] = f[m2] + np.clip(0.62 - n2, 0, 1) * 0.15 * l2[m2]
    # single leaves and the gaps between them, for trees within a few tens of metres at full size
    l3 = np.clip((lodf - 1.0) / 1.0, 0, 1) if LEAF_LOD else np.zeros(N)
    m3 = np.where((l3 > 0) & (f > -0.3) & (f < 0.12))[0]
    if len(m3):
        so = seed[m3] * 57.0
        n3 = map_coordinates(NTEX, [q[m3, 0] * 26.0 - so, q[m3, 1] * 26.0 + so, q[m3, 2] * 26.0 + so * 0.5], order=1, mode="grid-wrap")
        thr3 = 0.55 * l3[m3] * np.clip(1 + f[m3] / 0.3, 0, 1)
        f[m3] = np.maximum(f[m3], (thr3 - n3) * 0.14)
    # wood: trunk, and a few limbs inside deciduous crowns (seen through the gaps)
    rt = 0.12 + 0.04 * R
    zcd = np.where(tp == T_DEC, Ht * 1.02 - 1.26 * _crown_rv(R, Ht), 0)
    Rvd = _crown_rv(R, Ht)
    ztop = np.select([tp == T_DEC, tp == T_CON], [zcd - 0.7 * Rvd, Ht * 0.8], Ht * 0.55)
    flare = 1 + 0.3 * np.exp(-np.maximum(z + 0.3, 0) / 0.3)                  # root flare
    ft = np.maximum(rr - rt * flare * (1 - 0.35 * np.clip(z / np.maximum(ztop, 1), 0, 1)), np.maximum(-1.5 - z, z - ztop))
    ft = np.minimum(ft, wood_b)
    return np.minimum(f, ft), ft < f, n, fb

def isect_trees(O, D, ids, K=16, noisy=None, want_normal=False):
    """first hit t (m, local = world scale) of rays vs trees ids; inf = miss. noisy: level of detail per ray (0..1)"""
    q0, d = _local(O, D, ids)
    tp, Ht, R, sd = INST["tp"][ids], INST["h"][ids], INST["r"][ids], INST["seed"][ids]
    lod = noisy if noisy is not None else np.zeros(len(ids))
    rb = R * 1.35 + 0.3
    A = d[:, 0] ** 2 + d[:, 1] ** 2; B = 2 * (q0[:, 0] * d[:, 0] + q0[:, 1] * d[:, 1]); C = q0[:, 0] ** 2 + q0[:, 1] ** 2 - rb ** 2
    disc = B * B - 4 * A * C; sq = np.sqrt(np.maximum(disc, 0)); A_ = np.maximum(A, 1e-12)
    t0 = np.where(A > 1e-12, (-B - sq) / (2 * A_), -1e9); t1 = np.where(A > 1e-12, (-B + sq) / (2 * A_), np.where(C < 0, 1e9, -1e9))
    dz = np.where(np.abs(d[:, 2]) < 1e-9, 1e-9, d[:, 2])
    za, zb = (-1.6 - q0[:, 2]) / dz, (Ht * 1.03 + 0.3 - q0[:, 2]) / dz
    ta = np.maximum.reduce([t0, np.minimum(za, zb), np.zeros_like(t0)]); tb = np.minimum(t1, np.maximum(za, zb))
    ok = (disc > 0) & (tb > ta)
    dec = np.where(ok & (tp == T_DEC))[0]
    if len(dec):                                  # deciduous: crown ellipsoid and trunk cylinder bound the march tightly
        Rd, Hd = R[dec], Ht[dec]; Rv = _crown_rv(Rd, Hd); zc = Hd * 1.02 - 1.26 * Rv
        eh, ev = Rd * 1.15 + 0.45, Rv * 1.15 + 0.45
        qx, qy, qz = q0[dec, 0] / eh, q0[dec, 1] / eh, (q0[dec, 2] - zc) / ev
        dx, dy, dz_ = d[dec, 0] / eh, d[dec, 1] / eh, d[dec, 2] / ev
        A2 = dx * dx + dy * dy + dz_ * dz_; B2 = 2 * (qx * dx + qy * dy + qz * dz_); C2 = qx * qx + qy * qy + qz * qz - 1
        D2 = B2 * B2 - 4 * A2 * C2; s2 = np.sqrt(np.maximum(D2, 0))
        te0 = np.where(D2 > 0, (-B2 - s2) / (2 * A2), np.inf); te1 = np.where(D2 > 0, (-B2 + s2) / (2 * A2), -np.inf)
        rt = (0.12 + 0.04 * Rd) * 1.5 + 0.05; ztop = zc - 0.7 * Rv + 0.5
        Ac = d[dec, 0] ** 2 + d[dec, 1] ** 2; Bc = 2 * (q0[dec, 0] * d[dec, 0] + q0[dec, 1] * d[dec, 1]); Cc = q0[dec, 0] ** 2 + q0[dec, 1] ** 2 - rt ** 2
        Dc = Bc * Bc - 4 * Ac * Cc; sc = np.sqrt(np.maximum(Dc, 0)); Ac_ = np.maximum(Ac, 1e-12)
        tc0 = np.where(Dc > 0, (-Bc - sc) / (2 * Ac_), np.inf); tc1 = np.where(Dc > 0, (-Bc + sc) / (2 * Ac_), -np.inf)
        zc0, zc1 = (-1.6 - q0[dec, 2]) / dz[dec], (ztop - q0[dec, 2]) / dz[dec]
        tc0 = np.maximum(tc0, np.minimum(zc0, zc1)); tc1 = np.minimum(tc1, np.maximum(zc0, zc1))
        tc_ok = tc1 > tc0; te_ok = te1 > te0
        lo_ = np.minimum(np.where(te_ok, te0, np.inf), np.where(tc_ok, tc0, np.inf))
        hi_ = np.maximum(np.where(te_ok, te1, -np.inf), np.where(tc_ok, tc1, -np.inf))
        ta[dec] = np.maximum(ta[dec], lo_); tb[dec] = np.minimum(tb[dec], hi_)
        ok[dec] &= tb[dec] > ta[dec]
    t_hit = np.full(len(ids), np.inf)
    idx = np.where(ok)[0]
    if not len(idx): return t_hit
    ta, tb = ta[idx], tb[idx]; dtmin = np.maximum((tb - ta) / (K + 6), 0.1)
    ld = lod[idx]
    found = np.zeros(len(idx), bool); alive = np.ones(len(idx), bool)
    tcur = ta.copy(); lo = ta.copy(); hi = tb.copy(); last = np.zeros(len(idx)); lastf = np.full(len(idx), 1e3)
    for k in range(K + 14 + 24 * int((ld >= 0.5).any())):   # distance-guided march: big steps far from the crown
        live = np.where(alive & ((k < K + 14) | (ld >= 0.5)))[0]
        if not len(live): break
        j = idx[live]; tt = tcur[live]
        f, _, _, fb = tree_field(q0[j] + d[j] * tt[:, None], tp[j], Ht[j], R[j], sd[j], ld[live], ids[j])
        inside = f < 0
        L_in = live[inside]
        hi[L_in] = tt[inside]; lo[L_in] = np.maximum(tt[inside] - last[L_in], ta[L_in]); found[L_in] = True; alive[L_in] = False
        L_out = live[~inside]; fo = f[~inside]
        res = ld[L_out] >= 0.5                                           # resolved crowns: plain sphere tracing
        stp = np.where(res, np.maximum(0.45 * fo, 0.02),
                       np.maximum(np.where(fb[~inside] < 0, np.minimum(0.55 * fo, 0.7), 0.45 * fo), dtmin[L_out]))
        last[L_out] = stp; lastf[L_out] = fo
        tcur[L_out] += stp; alive[L_out[tcur[L_out] > tb[L_out]]] = False
    close = alive & (lastf < 0.03) & (ld < 0.5)       # out of steps right at a surface: count it as a hit
    found |= close; lo[close] = tcur[close] - last[close]; hi[close] = tcur[close] + 0.15
    f_ = np.where(found)[0]
    lo, hi = lo[f_], hi[f_]
    for _ in range(4):
        mid = (lo + hi) / 2; j = idx[f_]
        f, _, _, _ = tree_field(q0[j] + d[j] * mid[:, None], tp[j], Ht[j], R[j], sd[j], ld[f_], ids[j])
        hi = np.where(f < 0, mid, hi); lo = np.where(f < 0, lo, mid)
    t_hit[idx[f_]] = hi
    return t_hit

def _house_planes(ids):
    A, B, HW, HR = INST["a"][ids], INST["b"][ids], INST["hw"][ids], INST["hr"][ids]
    z0 = -2.5 * np.ones_like(A); one = np.ones_like(A); zero = np.zeros_like(A)
    sl = HR / B
    # (nx, ny, nz, d) with n.p <= d inside
    return [(one, zero, zero, A), (-one, zero, zero, A), (zero, one, zero, B), (zero, -one, zero, B), (zero, zero, -one, -z0),
            (zero, sl, one, HW + HR), (zero, -sl, one, HW + HR)]

def _convex(q, d, planes):
    tn = np.zeros(len(q)); tf = np.full(len(q), np.inf); which = np.zeros(len(q), np.int8); miss = np.zeros(len(q), bool)
    for k, (nx, ny, nz, dd) in enumerate(planes):
        den = nx * d[:, 0] + ny * d[:, 1] + nz * d[:, 2]; num = dd - (nx * q[:, 0] + ny * q[:, 1] + nz * q[:, 2])
        with np.errstate(divide="ignore", invalid="ignore"):
            t = num / den
        ent = den < -1e-12; ext = den > 1e-12
        upd = ent & (t > tn); tn = np.where(upd, t, tn); which = np.where(upd, k, which)
        tf = np.where(ext, np.minimum(tf, t), tf)
        miss |= (np.abs(den) <= 1e-12) & (num < 0)
    hit = ~miss & (tn <= tf) & (tf > 0) & (tn > 0)
    return np.where(hit, tn, np.inf), which

def _roof_slab(ids, sgn):
    A, B, HW, HR = INST["a"][ids], INST["b"][ids], INST["hw"][ids], INST["hr"][ids]
    ov, th = 0.55, 0.28; one = np.ones_like(A); zero = np.zeros_like(A); sl = HR / B
    top = HW + HR + 0.06
    return [(one, zero, zero, A + ov), (-one, zero, zero, A + ov), (zero, -sgn * one, zero, zero), (zero, sgn * one, zero, B + ov),
            (zero, sgn * sl, one, top), (zero, -sgn * sl, -one, -(top - th))]

def isect_houses(O, D, ids, want=False):
    """houses = walls/gables (convex) + two overhanging roof slabs; blocks = one box"""
    q, d = _local(O, D, ids)
    body = _house_planes(ids)
    t, which = _convex(q, d, body)
    kind = np.zeros(len(ids), np.int8)                                    # 0 body, 1 roof slab
    pitched = INST["hr"][ids] > 0
    nl = None
    if want:
        nl = np.zeros((len(ids), 3))
        for w_, (nx, ny, nz, _) in enumerate(body):
            m = which == w_; nl[m] = np.stack([nx[m], ny[m], nz[m]], 1)
    if pitched.any():
        k = np.where(pitched)[0]
        for sgn in (1.0, -1.0):
            pl = _roof_slab(ids[k], sgn)
            ts, ws = _convex(q[k], d[k], pl)
            better = ts < t[k]
            t[k[better]] = ts[better]; kind[k[better]] = 1
            if want:
                for w_, (nx, ny, nz, _) in enumerate(pl):
                    m = better & (ws == w_)
                    nl[k[m]] = np.stack([nx[m], ny[m], nz[m]], 1)
        # chimney: a box through one roof slope, rising 0.9 m above the ridge (village houses only)
        kc = k[INST["tp"][ids[k]] == T_HOUSE]
        if len(kc):
            ic = ids[kc]; A, B, HW, HR = INST["a"][ic], INST["b"][ic], INST["hw"][ic], INST["hr"][ic]
            cx = (hash2(ic, 0, 120) - 0.5) * 1.2 * A; cy = (0.25 + 0.2 * hash2(ic, 0, 121)) * B * np.where(hash2(ic, 0, 122) < 0.5, 1, -1)
            one = np.ones(len(kc)); zero = np.zeros(len(kc)); hw_ = 0.3
            pl = [(one, zero, zero, cx + hw_), (-one, zero, zero, -(cx - hw_)), (zero, one, zero, cy + hw_), (zero, -one, zero, -(cy - hw_)),
                  (zero, zero, -one, -HW), (zero, zero, one, HW + HR + 0.9)]
            tc, wc = _convex(q[kc], d[kc], pl)
            better = tc < t[kc]
            t[kc[better]] = tc[better]; kind[kc[better]] = 2
            if want:
                for w_, (nx, ny, nz, _) in enumerate(pl):
                    m = better & (wc == w_)
                    nl[kc[m]] = np.stack([nx[m], ny[m], nz[m]], 1)
    if not want: return t
    nl /= np.maximum(np.linalg.norm(nl, axis=1, keepdims=True), 1e-9)
    return t, which, kind, nl, q, d

def isect_walls(O, D, ids, want=False, cut=True):
    """field walls, hedges (boxes with a ragged top) and post-and-rail fences (a thin box with cut-outs)"""
    q, d = _local(O, D, ids)
    A, B, HW = INST["a"][ids], INST["b"][ids], INST["hw"][ids]
    one = np.ones_like(A); zero = np.zeros_like(A)
    t, which = _convex(q, d, [(one, zero, zero, A), (-one, zero, zero, A), (zero, one, zero, B), (zero, -one, zero, B),
                              (zero, zero, -one, 0.6 * one), (zero, zero, one, HW)])
    if cut:
        p = q + d * np.where(np.isfinite(t), t, 0)[:, None]
        fence = INST["tp"][ids] == T_FENCE; sd = INST["seed"][ids]
        post = np.mod(p[:, 0] + 50.0 + sd * 2.4, 2.4) < 0.12
        rail = (np.abs(p[:, 2] - HW + 0.1) < 0.055) | (np.abs(p[:, 2] - HW * 0.5) < 0.05)
        hedge = INST["mat"][ids] == 1
        # hedges are ragged; stone walls have a level capstone course that steps a few cm from stone to stone
        top = np.where(hedge, HW - 0.35 * tnoise(p[:, 0] * 1.6 + sd * 90, p[:, 1] * 0.5 + 7.0, sd * 30) ** 1.2,
                       HW - 0.05 * hash2(np.floor(p[:, 0] / 0.45 + sd * 7).astype(np.int64), (sd * 1e4).astype(np.int64), 92))
        bad = np.where(fence, ~(post | rail), p[:, 2] > top)
        t = np.where(bad, np.inf, t)
    if not want: return t
    nl = np.zeros((len(ids), 3))
    for w_, n_ in enumerate(([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, -1], [0, 0, 1])):
        nl[which == w_] = n_
    return t, which, nl, q, d

def isect_any(O, D, ids, noisy=None):
    t = np.full(len(ids), np.inf)
    tpi = INST["tp"][ids]
    tr = tpi <= T_POP; hs = (tpi == T_HOUSE) | (tpi == T_BLOCK); wl = tpi >= T_WALL
    one = len(O) == 1
    if tr.any(): t[tr] = isect_trees(O if one else O[tr], D[tr], ids[tr], noisy=noisy[tr] if noisy is not None else None)
    if hs.any(): t[hs] = isect_houses(O if one else O[hs], D[hs], ids[hs])
    if wl.any(): t[wl] = isect_walls(O if one else O[wl], D[wl], ids[wl])
    return t

# ------------------------------------------------------------------ sun shadows cast by instances
LAM_SH = 1.1                                                                # m of crown that dims the sun by 1/e
def _crown_T(P, ids, L):
    """transmittance of rays P + s*L through simplified crowns (ellipsoids): exp(-chord / LAM_SH)"""
    q, d = _local(P, np.tile(L, (len(P), 1)), ids)
    zc, Rh, Rv = _crown_ellipsoid(INST["tp"][ids], INST["h"][ids], INST["r"][ids])
    Rh = Rh * 0.95
    qx, qy, qz = q[:, 0] / Rh, q[:, 1] / Rh, (q[:, 2] - zc) / Rv
    dx, dy, dz = d[:, 0] / Rh, d[:, 1] / Rh, d[:, 2] / Rv
    A = dx * dx + dy * dy + dz * dz; B = 2 * (qx * dx + qy * dy + qz * dz); C = qx * qx + qy * qy + qz * qz - 1
    disc = B * B - 4 * A * C; sq = np.sqrt(np.maximum(disc, 0))
    t0 = np.maximum((-B - sq) / (2 * A), 0.3); t1 = (-B + sq) / (2 * A)
    chord = np.where(disc > 0, np.maximum(t1 - t0, 0), 0)
    lam = np.full(len(P), LAM_SH)
    m = chord > 0
    if m.any():                                                       # sun flecks: gaps in the crown vary the path
        lam[m] = LAM_SH * (0.35 + 1.3 * tnoise(P[m, 0] * 2200.0, P[m, 1] * 2200.0, P[m, 2] * 2200.0) ** 2)
    return np.exp(-chord / lam)

def inst_shadow(P, n, skip=None):
    """1 = lit, 0 = in the shadow of a tree or building (terrain shadow is separate). Crowns let some light through.
    skip: per point, an instance to ignore (a tree lights its own crown in shade_instances)"""
    out = np.ones(len(P))
    if not NI or SUN_EL <= 0.005: return out
    Pm = P + n * 0.0004
    d = np.hypot(Pm[:, 0] - CAM[0], Pm[:, 1] - CAM[1])
    sel = np.where(d < NEAR * 1.02 + 0.2)[0]
    if not len(sel): return out
    Lh = min(HMAX / 1000 / math.tan(SUN_EL), 0.35)                      # horizontal reach of the tallest shadow
    ch = math.cos(SUN_EL); T = np.ones(len(sel))
    sk = skip[sel] if skip is not None else np.full(len(sel), -1)
    for g in GRIDS:
        c = g["c"]; cr, sr = g["rot"]; ids_g = g["ids"]
        walls = g.get("walls", False)
        hg = g.get("hmax")
        if hg is None:
            iv = ids_g[ids_g >= 0]; hg = g["hmax"] = float(INST["h"][iv].max()) / 1000 if len(iv) else 0.0
        reach = min(Lh, hg / math.tan(SUN_EL))                           # each grid's tallest shadow
        steps = int(math.ceil(reach / (c / 2.5)))
        seen = np.full(len(sel), -1, np.int64)
        for k in range(steps + 1):
            live = np.where(T > 0.02)[0]
            if not len(live): break
            s = (k * c / 2.5) / ch
            Q = Pm[sel[live]] + SUN * s
            if k > 2 and (Q[:, 2] - terrain_z(Q[:, 0], Q[:, 1]) > hg).all(): break
            u, v = Q[:, 0] * cr + Q[:, 1] * sr, -Q[:, 0] * sr + Q[:, 1] * cr
            i = np.floor(u / c).astype(np.int64) - g["i0"]; j = np.floor(v / c).astype(np.int64) - g["j0"]
            inb = (i >= 0) & (j >= 0) & (i < ids_g.shape[0]) & (j < ids_g.shape[1])
            iid = np.full(len(live), -1, np.int64); iid[inb] = ids_g[i[inb], j[inb]]
            m = (iid >= 0) & (iid != seen[live]) & (iid != sk[live])
            if not m.any(): continue
            seen[live[m]] = iid[m]
            L_ = live[m]; ii = iid[m]; Pq = Pm[sel[L_]]
            tpi = INST["tp"][ii]
            tr = tpi <= T_POP; hs = (tpi == T_HOUSE) | (tpi == T_BLOCK); wl = tpi == T_WALL
            tv = np.ones(len(ii))
            if tr.any(): tv[tr] = _crown_T(Pq[tr], ii[tr], SUN)
            if hs.any(): tv[hs] = np.where(np.isfinite(isect_houses(Pq[hs], np.tile(SUN, (hs.sum(), 1)), ii[hs])), 0.0, 1.0)
            if wl.any(): tv[wl] = np.where(np.isfinite(isect_walls(Pq[wl], np.tile(SUN, (wl.sum(), 1)), ii[wl], cut=False)), 0.0, 1.0)
            T[L_] *= tv
    out[sel] = np.where(T > 0.02, T, 0.0)
    return out

# ------------------------------------------------------------------ primary visibility of instances
SUB = np.array([[-0.25, -0.25], [0.25, -0.25], [-0.25, 0.25], [0.25, 0.25]])
SUB2 = np.array([[-0.2, -0.25], [0.2, 0.25]])
def near_vis(r0, r1, tmax):
    """sun-independent part: nearest instance per pixel and sub-sample (cacheable across timelapse frames)"""
    empty = (np.zeros(0, np.int64), np.zeros((0, 4), np.int64), np.zeros((0, 4)))
    if not NI: return empty
    sel = np.where(VIS & (BB[:, 2] < r1) & (BB[:, 3] > r0))[0]
    if not len(sel): return empty
    x0, x1 = BB[sel, 0], BB[sel, 1]; y0, y1 = np.maximum(BB[sel, 2], r0), np.minimum(BB[sel, 3], r1)
    nx, ny = x1 - x0, y1 - y0; cnt = nx * ny
    best_t = np.full(((r1 - r0) * W, 4), np.inf); best_i = np.full(((r1 - r0) * W, 4), -1, np.int64)
    # chunk instances so the pair arrays stay small
    # nearest instances first, in small chunks, so pairs behind a hit already found are skipped
    order = np.argsort(DMIN[sel], kind="stable"); csum = np.cumsum(cnt[order]); chunk = 80000
    starts = np.searchsorted(csum, np.arange(0, csum[-1], chunk), side="right")
    starts = np.unique(np.concatenate([[0], starts, [len(sel)]]))
    for s0, s1 in zip(starts[:-1], starts[1:]):
        ks = order[s0:s1]
        if not len(ks): continue
        c = cnt[ks]; tot = int(c.sum())
        if tot == 0: continue
        li = np.repeat(ks, c); off = np.arange(tot) - np.repeat(np.cumsum(c) - c, c)
        col = x0[li] + off % nx[li]; row = y0[li] + off // nx[li]
        pix = (row - r0) * W + col; iid = sel[li]
        keep = (DMIN[iid] < tmax[pix]) & (DMIN[iid] < best_t[pix].max(1))
        pix, iid, col, row = pix[keep], iid[keep], col[keep], row[keep]
        if not len(pix): continue
        lod = LODT[iid]
        for sI, (ox_, oy_) in enumerate(SUB):
            px_ = (col + 0.5 + ox_ - W / 2) / fl; py_ = -(row + 0.5 + oy_ - H / 2 - H * 0.02) / fl
            D = fwd[None] + px_[:, None] * right[None] + py_[:, None] * up[None]; D /= np.linalg.norm(D, axis=1, keepdims=True)
            t = isect_any(CAM[None], D, iid, lod) / 1000
            ok = np.isfinite(t) & (t < tmax[pix])
            if not ok.any(): continue
            # keep the nearest per pixel/sub-sample
            pp, tt, ii = pix[ok], t[ok], iid[ok]
            o = np.lexsort((tt, pp)); pp, tt, ii = pp[o], tt[o], ii[o]
            first = np.concatenate([[True], pp[1:] != pp[:-1]])
            pp, tt, ii = pp[first], tt[first], ii[first]
            better = tt < best_t[pp, sI]
            best_t[pp[better], sI] = tt[better]; best_i[pp[better], sI] = ii[better]
    hitpix = np.where((best_i >= 0).any(1))[0]
    return hitpix, best_i[hitpix], best_t[hitpix]

def near_field(r0, r1, dirs, tmax, vis=None):
    empty = (np.zeros(0, int), np.zeros(0), np.zeros((0, 3)), np.zeros(0))
    hitpix, BI, BT = vis if vis is not None else near_vis(r0, r1, tmax)
    if not len(hitpix): return empty
    cov = (BI >= 0).mean(1)
    colsum = np.zeros((len(hitpix), 3)); tsum = np.zeros(len(hitpix))
    ys_, xs_ = hitpix // W + r0, hitpix % W
    for sI, (ox_, oy_) in enumerate(SUB):
        m = BI[:, sI] >= 0
        if not m.any(): continue
        ii = BI[m, sI]; tt = BT[m, sI]
        px_ = (xs_[m] + 0.5 + ox_ - W / 2) / fl; py_ = -(ys_[m] + 0.5 + oy_ - H / 2 - H * 0.02) / fl
        D = fwd[None] + px_[:, None] * right[None] + py_[:, None] * up[None]; D /= np.linalg.norm(D, axis=1, keepdims=True)
        colsum[m] += shade_instances(CAM + D * tt[:, None], D, ii, tt)
        tsum[m] += tt
    nh = np.maximum((BI >= 0).sum(1), 1)
    return hitpix, tsum / nh, colsum / nh[:, None], cov

def _pairs(bb, sel, r0, r1):
    x0, x1 = bb[sel, 0], bb[sel, 1]; y0, y1 = np.maximum(bb[sel, 2], r0), np.minimum(bb[sel, 3], r1)
    nx, ny = np.maximum(x1 - x0, 0), np.maximum(y1 - y0, 0); c = nx * ny; tot = int(c.sum())
    li = np.repeat(np.arange(len(sel)), c); off = np.arange(tot) - np.repeat(np.cumsum(c) - c, c)
    col = x0[li] + off % nx[li]; row = y0[li] + off // nx[li]
    return (row - r0) * W + col, sel[li], col, row

def near_field_mirror(r0, r1, dirs, wi, tw, trefl):
    """instances seen in the water: rays from each water point along the flat-water mirror direction"""
    empty = (np.zeros(0, int), np.zeros(0), np.zeros((0, 3)))
    if not NI: return empty
    sel = np.where(MVIS & (MBB[:, 2] < r1) & (MBB[:, 3] > r0))[0]
    if not len(sel): return empty
    iswater = np.zeros((r1 - r0) * W, bool); iswater[wi] = True
    loc = np.full((r1 - r0) * W, -1); loc[wi] = np.arange(len(wi))
    best_t = np.full((len(wi), 2), np.inf); best_i = np.full((len(wi), 2), -1, np.int64)
    x0, x1 = MBB[sel, 0], MBB[sel, 1]; y0, y1 = np.maximum(MBB[sel, 2], r0), np.minimum(MBB[sel, 3], r1)
    cnt = np.maximum(x1 - x0, 0) * np.maximum(y1 - y0, 0); csum = np.cumsum(cnt)
    starts = np.unique(np.concatenate([[0], np.searchsorted(csum, np.arange(0, csum[-1], 250000), side="right"), [len(sel)]]))
    for c0, c1 in zip(starts[:-1], starts[1:]):                           # chunked to bound memory
        pix, iid, col, row = _pairs(MBB, sel[c0:c1], r0, r1)
        k = iswater[pix]; pix, iid, col, row = pix[k], iid[k], col[k], row[k]
        if not len(pix): continue
        lod = LODT[iid] * 0.5                                              # reflections are smeared: less detail
        for sI, (ox_, oy_) in enumerate(SUB2):
            px_ = (col + 0.5 + ox_ - W / 2) / fl; py_ = -(row + 0.5 + oy_ - H / 2 - H * 0.02) / fl
            D = fwd[None] + px_[:, None] * right[None] + py_[:, None] * up[None]; D /= np.linalg.norm(D, axis=1, keepdims=True)
            tp_ = plane_t(np.repeat(CAM[None], len(D), 0), D, WL)
            ok0 = np.isfinite(tp_) & (tp_ > 0)
            P = CAM + D * np.where(ok0, tp_, 0)[:, None]; Dm = D * np.array([1, 1, -1.0])
            s_ = np.full(len(D), np.inf)
            s_[ok0] = isect_any(P[ok0] + np.array([0, 0, 0.0003]), Dm[ok0], iid[ok0], lod[ok0]) / 1000
            lw = loc[pix]; ok = np.isfinite(s_) & (s_ < trefl[lw])
            if not ok.any(): continue
            pp, tt, ii = lw[ok], s_[ok], iid[ok]
            o = np.lexsort((tt, pp)); pp, tt, ii = pp[o], tt[o], ii[o]
            first = np.concatenate([[True], pp[1:] != pp[:-1]]); pp, tt, ii = pp[first], tt[first], ii[first]
            better = tt < best_t[pp, sI]; best_t[pp[better], sI] = tt[better]; best_i[pp[better], sI] = ii[better]
    hw = np.where((best_i >= 0).any(1))[0]
    if not len(hw): return empty
    cov = (best_i[hw] >= 0).mean(1); colsum = np.zeros((len(hw), 3))
    hp = wi[hw]; ys_, xs_ = hp // W + r0, hp % W
    for sI, (ox_, oy_) in enumerate(SUB2):
        m = best_i[hw, sI] >= 0
        if not m.any(): continue
        ii = best_i[hw[m], sI]; ss = best_t[hw[m], sI]
        px_ = (xs_[m] + 0.5 + ox_ - W / 2) / fl; py_ = -(ys_[m] + 0.5 + oy_ - H / 2 - H * 0.02) / fl
        D = fwd[None] + px_[:, None] * right[None] + py_[:, None] * up[None]; D /= np.linalg.norm(D, axis=1, keepdims=True)
        P = CAM + D * plane_t(np.repeat(CAM[None], len(D), 0), D, WL)[:, None] + np.array([0, 0, 0.0003]); Dm = D * np.array([1, 1, -1.0])
        c_ = shade_instances(P + Dm * ss[:, None], Dm, ii, ss, O=P, lod_k=0.5)
        Li, Ti = atmosphere(Dm, ss, O=P)
        colsum[m] += c_ * Ti + Li
    nh = np.maximum((best_i[hw] >= 0).sum(1), 1)
    return hp, cov, colsum / nh[:, None]

def _normal_trees(P, ids, lod):
    """gradient of the tree field by the tetrahedron trick (4 evaluations)"""
    q = (P - np.stack([INST["x"][ids], INST["y"][ids], INST["z"][ids]], 1)) * 1000
    tp, Ht, R, sd = INST["tp"][ids], INST["h"][ids], INST["r"][ids], INST["seed"][ids]
    e = 0.12 + 0.1 * (1 - np.minimum(lod, 1)) - 0.08 * np.clip(lod - 1, 0, 1)
    g = np.zeros_like(q)
    for kv in ((1, -1, -1), (-1, -1, 1), (-1, 1, -1), (1, 1, 1)):
        kv = np.array(kv, float)
        f, _, _, _ = tree_field(q + kv * e[:, None], tp, Ht, R, sd, lod, ids); g += kv * f[:, None]
    n = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)
    f0, wood, nv, fb = tree_field(q, tp, Ht, R, sd, lod, ids)
    return n, q, wood, nv, fb

def _ell_exit(q, L, zc, Rh, Rv):
    """distance (m) from q along L through the crown ellipsoid (0 when the ray misses it)"""
    qx, qy, qz = q[:, 0] / Rh, q[:, 1] / Rh, (q[:, 2] - zc) / Rv
    dx, dy, dz = L[0] / Rh, L[1] / Rh, L[2] / Rv
    A = dx * dx + dy * dy + dz * dz; B = 2 * (qx * dx + qy * dy + qz * dz); C = qx * qx + qy * qy + qz * qz - 1
    disc = B * B - 4 * A * C; sq = np.sqrt(np.maximum(disc, 0))
    t0 = np.maximum((-B - sq) / (2 * A), 0); t1 = (-B + sq) / (2 * A)
    return np.where(disc > 0, np.maximum(t1 - t0, 0), 0.0)

_eld = math.degrees(SUN_EL)
LR_WIN = float(np.clip((1.5 - _eld) / 5.0, 0, 1))                         # share of windows lit: 0 in daylight .. 1 at -3.5 deg
LR_ST = float(np.clip((1.0 - _eld) / 2.0, 0, 1))                          # street lamps switch on around sunset
NIGHT_WIN = LR_WIN > 0

def _fade(mpp, period):
    """0..1 contrast for a pattern of this period (m) at mpp metres per pixel"""
    return np.clip(period / np.maximum(mpp, 1e-6) / 2.5 - 1, 0, 1)

def _hash3(i, j, k, salt):
    return hash2(i * 7919 + k * 104729, j, salt)

def _leaf_cells(p, salt):
    """3-D cellular pattern in cell units: distance to the nearest and second-nearest jittered point, and the
    nearest point's cell (2x2x2 search around the nearest lattice corner)"""
    b = np.floor(p - 0.5).astype(np.int64)
    f1 = np.full(len(p), 9.0); f2 = np.full(len(p), 9.0); cid = np.zeros((len(p), 3), np.int64)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                c = b + np.array([dx, dy, dz])
                jx, jy, jz = _hash3(c[:, 0], c[:, 1], c[:, 2], salt), _hash3(c[:, 0], c[:, 1], c[:, 2], salt + 1), _hash3(c[:, 0], c[:, 1], c[:, 2], salt + 2)
                d = np.sqrt((c[:, 0] + 0.2 + 0.6 * jx - p[:, 0]) ** 2 + (c[:, 1] + 0.2 + 0.6 * jy - p[:, 1]) ** 2 + (c[:, 2] + 0.2 + 0.6 * jz - p[:, 2]) ** 2)
                nb = d < f1
                f2 = np.where(nb, f1, np.minimum(f2, d)); f1 = np.where(nb, d, f1); cid = np.where(nb[:, None], c, cid)
    return f1, f2, cid

def shade_trees(P, D, ii, sunc, lod_k=1.0):
    lodf = LODT[ii] * lod_k
    n, q, wood, nv, fb = _normal_trees(P, ii, lodf)
    lod = np.minimum(lodf, 1.0)
    tp, Ht, R = INST["tp"][ii], INST["h"][ii], INST["r"][ii]
    zc, Rh, Rv = _crown_ellipsoid(tp, Ht, R)
    radial = np.stack([q[:, 0], q[:, 1], (q[:, 2] - zc) * (Rh / Rv) ** 2 * 0.6 + 0.3 * R], 1)
    radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-9)
    n = np.where(wood[:, None], n, n * 0.62 + radial * 0.38); n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    # light inside the crown: depth of foliage toward the sun, and depth below the crown surface
    sp = 0.28 + 0.24 * np.mod(INST["seed"][ii] * 7.31, 1.0)
    lam = 0.7 + 1.6 * sp * (0.4 + 0.6 * lod)
    Ls = _ell_exit(q, SUN, zc, Rh, Rv)
    selfT = np.exp(-Ls / lam)
    rn = np.sqrt((q[:, 0] / Rh) ** 2 + (q[:, 1] / Rh) ** 2 + ((q[:, 2] - zc) / Rv) ** 2)
    depth = np.clip(1 - rn, 0, 1) * np.minimum(Rh, Rv)
    sh = soft_shadow(P, n, steps=40) * inst_shadow(P, n, skip=ii)
    mpp = DI_M[ii] / fl
    hol = np.clip((nv - 0.2) / 0.6, 0, 1) * lod + 0.3 * (1 - lod)          # clump tops bright, hollows dark
    big = tnoise(q[:, 0] * 0.5 + INST["seed"][ii] * 40, q[:, 1] * 0.5, q[:, 2] * 0.5)
    alb = INST["col"][ii, :3] * (0.8 + 0.4 * big)[:, None]
    alb = alb * (1 + np.array([0.08, 0.03, -0.1])[None, :] * ((big - 0.5) * 2)[:, None])  # clumps differ a little in hue
    leaf = tnoise(q[:, 0] * 6.3 + 3, q[:, 1] * 6.3, q[:, 2] * 6.3 - 7) - 0.5  # spray-scale speckle
    grain = tnoise(q[:, 0] * 21 - 5, q[:, 1] * 21 + 2, q[:, 2] * 21) - 0.5     # leaves
    alb = alb * (1 + 0.7 * leaf * _fade(mpp, 0.3) + 1.1 * grain * _fade(mpp, 0.09))[:, None]
    bark = np.array([0.075, 0.062, 0.052]) * (0.8 + 0.4 * tnoise(q[:, 0] * 3, q[:, 1] * 3, q[:, 2] * 0.6))[:, None]
    alb = np.where(wood[:, None], bark, alb)
    hgt = np.clip(q[:, 2] / Ht, 0, 1)
    ao = np.clip(0.2 + 0.8 * np.exp(-depth / 1.6), 0, 1) * (0.55 + 0.45 * hol) * (0.45 + 0.55 * hgt ** 0.5)
    # single leaves (close trees): cells of a 3-D pattern, each leaf tilted at random around the clump normal,
    # with dark gaps between leaves. Conifer and poplar cells are stretched along their branches.
    l3 = np.clip(lodf - 1.0, 0, 1) * (~wood)
    lv = np.where(l3 > 0)[0]
    gap = np.zeros(len(ii)); nleaf = n.copy()
    if len(lv):
        tpl = tp[lv]; cs = np.where(tpl == T_CON, 0.06, np.where(tpl == T_POP, 0.08, 0.095))
        ql_ = q[lv] / cs[:, None]
        az = np.where(tpl == T_POP, 0.6, 1.0)                               # poplar leaves stack up the shoots
        pq = np.stack([ql_[:, 0], ql_[:, 1], ql_[:, 2] * az], 1) + (INST["seed"][ii[lv]] * 97.0)[:, None]
        wq = q[lv] * 1.7                                                    # warp the lattice so leaves are irregular
        pq = pq + np.stack([tnoise(wq[:, 0], wq[:, 1], wq[:, 2] + o_) - 0.5 for o_ in (0.0, 17.0, 31.0)], 1) * 1.6
        f1, f2, cid = _leaf_cells(pq, 130)
        rv = np.stack([_hash3(cid[:, 0], cid[:, 1], cid[:, 2], s_) - 0.5 for s_ in (133, 134, 135)], 1)
        nl_ = n[lv] + rv * 1.6; nl_ /= np.maximum(np.linalg.norm(nl_, axis=1, keepdims=True), 1e-9)
        w3 = l3[lv][:, None]
        nleaf[lv] = n[lv] * (1 - w3) + nl_ * w3
        hole = _hash3(cid[:, 0], cid[:, 1], cid[:, 2], 137) < 0.38                       # no leaf here: a gap
        edge = np.clip((f1 - 0.42) / 0.1, 0, 1) * np.clip(1 - (f2 - f1) / 0.25, 0, 1)      # between leaves
        gap[lv] = np.maximum(hole * 0.75, edge * 0.8) * l3[lv]
        lt = 0.8 + 0.4 * _hash3(cid[:, 0], cid[:, 1], cid[:, 2], 136)
        alb[lv] = alb[lv] * (1 + (lt - 1) * l3[lv])[:, None]
    ndl = nleaf @ SUN
    wfar = np.clip((ndl + 0.3) / 1.3, 0, 1) ** 1.5                            # far trees: the old solid-crown shading
    wrap = np.where(wood, np.clip(ndl, 0, 1), (0.3 + 0.7 * np.clip((ndl + 0.35) / 1.35, 0, 1) ** 1.3) * lod + wfar * (1 - lod))
    l3a = np.clip(lodf - 1.0, 0, 1)
    wrap = wrap * (1 - 0.55 * l3a) + np.clip(ndl, 0, 1) * 0.55 * l3a        # single leaves: closer to plain cosine
    ao = ao * (1 - 0.75 * gap)
    direct = wrap * sh * (0.04 + 0.96 * selfT) * (0.6 + 0.4 * hol) * (1 - 0.85 * gap)
    # leaves transmit yellow-green light: bright crown edges when the sun is behind the tree
    fwdp = np.clip(D @ SUN, 0, 1)
    tr = (0.12 + 1.8 * fwdp ** 4) * np.exp(-Ls / (lam * 1.3)) * sh * (~wood)
    # light scattered leaf to leaf inside the crown keeps the shaded side from going black
    ms = 0.07 * lod * np.exp(-depth / 2.5) * sh * (~wood) * (0.5 + 0.5 * selfT)
    col = alb * (sunc * (direct + ms)[:, None] + SKYC[None] * (ao * (0.55 + 0.45 * n[:, 2]))[:, None] + BOUNCE * ao[:, None]) \
        + alb * np.array([1.5, 1.7, 0.55]) * sunc * (0.45 * tr)[:, None]
    return col

def _hn(ii, salt):
    return hash2(ii, 0, salt)

def _cells2(u, v, salt):
    """2-D jittered cells: distances to the nearest and second-nearest point, the nearest cell and its point"""
    bu, bv = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    f1 = np.full(len(u), 9.0); f2 = np.full(len(u), 9.0); cid = np.zeros((len(u), 2), np.int64); cxy = np.zeros((len(u), 2))
    for du in (-1, 0, 1):
        for dv in (-1, 0, 1):
            cu, cv = bu + du, bv + dv
            pu = cu + 0.15 + 0.7 * hash2(cu, cv, salt); pv = cv + 0.15 + 0.7 * hash2(cu, cv, salt + 1)
            d = np.hypot(pu - u, pv - v)
            nb = d < f1
            f2 = np.where(nb, f1, np.minimum(f2, d)); f1 = np.where(nb, d, f1)
            cid = np.where(nb[:, None], np.stack([cu, cv], 1), cid); cxy = np.where(nb[:, None], np.stack([pu, pv], 1), cxy)
    return f1, f2, cid, cxy

def ground_bounce(n, sunc):
    """light reflected up from sunlit ground onto walls and foliage (ground albedo ~0.15, half the lower hemisphere)"""
    return sunc * max(math.sin(SUN_EL), 0) * 0.15 * (0.5 - 0.5 * n[:, 2:3])

def shade_walls(Pk, Dk, ii, sunc, Ok):
    t, which, nl, q0, d0 = isect_walls(Ok, Dk, ii, want=True)
    ql = q0 + d0 * np.where(np.isfinite(t), t, 0)[:, None]
    yaw = INST["yaw"][ii]; c, s = np.cos(yaw), np.sin(yaw)
    n = np.stack([nl[:, 0] * c - nl[:, 1] * s, nl[:, 0] * s + nl[:, 1] * c, nl[:, 2]], 1)
    mpp = DI_M[ii] / fl
    base = INST["col"][ii, :3]
    fence = INST["tp"][ii] == T_FENCE; hedge = (INST["mat"][ii] == 1) & ~fence
    x, z = ql[:, 0], ql[:, 2]; sd = INST["seed"][ii]
    # dry stone: roughly level courses of 20-35 cm stones, a capstone course on top, shallow dark joints
    along = np.where(np.abs(nl[:, 0]) > 0.5, ql[:, 1], x)
    HWk = INST["hw"][ii]; so = (sd * 1e4).astype(np.int64)
    cap = z > HWk - 0.17
    zr = z / 0.2 + 0.7 * (tnoise(along * 0.7, sd * 30, 1.0) - 0.5) + 0.25 * (tnoise(along * 2.2, sd * 11, 3.0) - 0.5)   # courses wander
    row = np.floor(zr); rj = zr - row
    row = np.where(cap, 999, row); rj = np.where(cap, (z - HWk + 0.17) / 0.17, rj)
    slen = np.where(cap, 0.45, 0.24 + 0.1 * hash2(row.astype(np.int64), so, 89))   # stone length per course
    ar = along / slen + hash2(row.astype(np.int64), so, 88) * 3 + 0.55 * (tnoise(along * 2.5, z * 4, sd * 9) - 0.5) \
       + 0.35 * (rj - 0.5) * (hash2(np.floor(along / slen).astype(np.int64), row.astype(np.int64) + so, 87) - 0.5)   # slanted joints
    stn = np.floor(ar); sj = ar - stn
    hst = hash2(stn.astype(np.int64) * 7 + row.astype(np.int64), so, 90)
    tint = 0.8 + 0.4 * hst                                                     # low contrast between stones
    tint = tint * (0.92 + 0.16 * tnoise(along * 3 + sd * 5, z * 3, 2.0))
    jr = np.minimum(rj, 1 - rj) / 0.14; js = np.minimum(sj, 1 - sj) / 0.1
    joint = np.clip(1 - np.minimum(jr, js), 0, 1) ** 1.5 * _fade(mpp, 0.12)
    hs_ = hash2(stn.astype(np.int64) * 3 + row.astype(np.int64), so, 91)      # a few ochre and rose tuff stones
    hue = np.where((hs_ > 0.82)[:, None], np.array([1.07, 0.98, 0.9]), np.where((hs_ < 0.12)[:, None], np.array([1.06, 0.95, 0.94]), 1.0))
    stone = base * (1 + (tint - 1) * _fade(mpp, 0.3))[:, None] * (1 + (hue - 1) * _fade(mpp, 0.3)[:, None]) * (1 - 0.45 * joint)[:, None]
    stone = np.where(cap[:, None], stone * 1.06, stone)
    grassf = np.clip(1 - z / (0.08 + 0.18 * tnoise(along * 4 + sd, 3.0, sd * 2)), 0, 1) * _fade(mpp, 0.1)   # grass at the foot
    stone = stone * (1 - 0.7 * grassf[:, None]) + np.array([0.08, 0.1, 0.04]) * 0.7 * grassf[:, None]
    lich = np.clip((tnoise(along * 2 + sd * 9, z * 2, 3) - 0.72) * 5, 0, 1) * _fade(mpp, 0.3)
    stone = stone * (1 - 0.3 * lich[:, None]) + np.array([0.30, 0.28, 0.20]) * 0.3 * lich[:, None]
    hed = base * (0.7 + 0.6 * tnoise(ql[:, 0] * 2.5 + sd * 50, ql[:, 1] * 2.5, z * 2.5))[:, None]
    wood = base * (0.8 + 0.4 * tnoise(along * 0.3, z * 4, sd * 20))[:, None]
    alb = np.where(fence[:, None], wood, np.where(hedge[:, None], hed, stone))
    sh = inst_shadow(Pk, n) * soft_shadow(Pk, n, steps=24)
    # each stone bulges: its top faces the sky, its underside is in shadow
    side = (np.abs(nl[:, 2]) < 0.5) & ~fence & ~hedge
    bz = (0.5 - rj) * 0.22 * _fade(mpp, 0.15); ba = (0.5 - sj) * 0.12 * _fade(mpp, 0.15)
    ax_ = np.where(np.abs(nl[:, 0]) > 0.5, 1, 0)                            # local axis along the wall face
    t_along = np.stack([np.where(ax_ == 1, -s, c), np.where(ax_ == 1, c, s), np.zeros(len(c))], 1)
    nb = n + side[:, None] * (np.array([0, 0, 1.0]) * bz[:, None] + t_along * ba[:, None])
    nb /= np.maximum(np.linalg.norm(nb, axis=1, keepdims=True), 1e-9)
    ndl = np.clip(nb @ SUN, 0, 1)
    ndl = np.where(hedge, np.clip((n @ SUN + 0.4) / 1.4, 0, 1), ndl)
    n = nb
    ao = np.clip(0.55 + z / 2.5, 0.55, 1.0) * np.where(hedge, 0.75, 1.0)
    return alb * (sunc * (ndl * sh)[:, None] + SKYC[None] * (ao * (0.6 + 0.4 * n[:, 2]))[:, None] + BOUNCE * ao[:, None]
                  + ground_bounce(n, sunc) * ao[:, None])

def house_albedo(ql, which, kind, nl, ii, mpp):
    """albedo, specular weight and window mask for village houses (T_HOUSE), from local hit coordinates"""
    N = len(ii); A, B, HW = INST["a"][ii], INST["b"][ii], INST["hw"][ii]
    wallc, roofc = INST["col"][ii, :3], INST["col"][ii, 3:]
    wm, rm = INST["mat"][ii] // 10, INST["mat"][ii] % 10
    sd = INST["seed"][ii]
    xf = np.abs(nl[:, 0]) > 0.5                                              # gable-end walls
    along = np.where(xf, ql[:, 1], ql[:, 0]); upz = ql[:, 2]
    halfw = np.where(xf, B, A)
    roof = kind == 1
    top = roof & (nl[:, 2] > 0.5); eave = roof & ~top
    # ---- walls
    f33 = _fade(mpp, 0.33)
    row = np.floor(upz / 0.33); rj = np.mod(upz / 0.33, 1.0)
    bl = 0.5 + 0.25 * _hn(ii, 101)
    blk = np.floor(along / bl + row * 0.5); bj = np.mod(along / bl + row * 0.5, 1.0)
    btint = hash2(blk.astype(np.int64) * 131 + row.astype(np.int64), ii, 102)
    hue = np.where((btint > 0.9)[:, None], np.array([1.1, 0.92, 0.85]), np.where((btint < 0.07)[:, None], np.array([0.82, 0.84, 0.88]), 1.0))
    stone = wallc * (1 + (0.9 + 0.2 * btint - 1) * _fade(mpp, 0.5))[:, None] * (1 + (hue - 1) * _fade(mpp, 0.5)[:, None])
    mortar = ((rj < 0.07) | (bj < 0.05 * 0.62 / bl)) * f33
    stone = stone * (1 - 0.25 * mortar)[:, None] + np.array([0.36, 0.34, 0.30]) * 0.25 * mortar[:, None]
    n1 = tnoise(along * 1.1 + sd * 50, upz * 1.1, sd * 20)
    streak = np.clip((tnoise(along * 3.0 + sd * 70, upz * 0.25, sd * 10) - 0.55) * 3, 0, 1) * _fade(mpp, 0.3)
    plaster = wallc * (0.9 + 0.2 * n1 * _fade(mpp, 1.0))[:, None] * (1 - 0.22 * streak)[:, None]
    peel = np.clip((tnoise(along * 0.7 + sd * 11, upz * 0.7, sd * 5 + 3) - 0.74) * 8, 0, 1) * _fade(mpp, 0.6)
    plaster = plaster * (1 - peel[:, None]) + np.array([0.30, 0.22, 0.18]) * (0.8 + 0.2 * mortar)[:, None] * peel[:, None]
    bj2 = np.mod(along / 0.2, 1.0)
    boards = wallc * (0.85 + 0.3 * hash2(np.floor(along / 0.2).astype(np.int64), ii, 103))[:, None] * (1 - 0.35 * (bj2 < 0.08) * _fade(mpp, 0.2))[:, None]
    wall = np.where((wm == 1)[:, None], stone, np.where((wm == 2)[:, None], boards, plaster))
    # mottling at 3 m and 0.7 m, so no wall is a flat fill; slightly warmer where it is lighter
    mo = (tnoise(along * 0.33 + sd * 17, upz * 0.33, sd * 9 + 2) - 0.5) * 0.2 * _fade(mpp, 2.0) \
       + (tnoise(along * 1.4 - sd * 7, upz * 1.4, sd * 3 + 8) - 0.5) * 0.12 * _fade(mpp, 0.5)
    wall = wall * (1 + mo[:, None] * np.array([1.1, 1.0, 0.85]))
    # rain stains: grey-brown streaks running down from the eaves on the long walls
    rs = np.clip((tnoise(along * 2.2 + sd * 31, upz * 0.18, sd * 4) - 0.58) * 3.5, 0, 1) * np.clip((upz - HW * 0.35) / (HW * 0.5), 0, 1)
    wall = wall * (1 - 0.16 * rs * ~xf * _fade(mpp, 0.25))[:, None]
    # gutter and eave shadow line along the top of the long walls
    wall = wall * (1 - 0.45 * (~xf & (upz > HW - 0.16)) * _fade(mpp, 0.08))[:, None]
    # basalt plinth with its own coursed stones; damp and dust in the lowest half metre above it
    ph_ = 0.6 + 0.25 * _hn(ii, 116)
    plinth = upz < ph_
    pst = hash2(np.floor(along / 0.55 + np.floor(upz / 0.3) * 0.5).astype(np.int64), ii + np.floor(upz / 0.3).astype(np.int64) * 7, 117)
    pj = ((np.mod(upz / 0.3, 1.0) < 0.1) | (np.mod(along / 0.55 + np.floor(upz / 0.3) * 0.5, 1.0) < 0.06)) * _fade(mpp, 0.3)
    pcol = np.array([0.12, 0.115, 0.11]) * (0.8 + 0.4 * pst * _fade(mpp, 0.5))[:, None] * (1 - 0.5 * pj)[:, None]
    wall = np.where(plinth[:, None], pcol, wall)
    damp = np.clip(1 - (upz - ph_) / 0.6, 0, 1) * (upz >= ph_) * (0.5 + 0.5 * n1) * (wm != 1)
    wall = wall * (1 - 0.25 * damp)[:, None]
    # patched plaster: rectangles of fresher render
    pa = hash2(np.floor(along / 1.7 + sd * 3).astype(np.int64), np.floor(upz / 1.3).astype(np.int64) + ii * 3, 118) < 0.12
    wall = np.where((pa & (wm == 0))[:, None], wall * np.array([1.07, 1.06, 1.03]), wall)
    # ---- windows, sills, shutters, door
    ww = 0.9 + 0.35 * _hn(ii, 104); wh = 1.1 + 0.35 * _hn(ii, 105)
    ncol = np.maximum(np.floor((2 * halfw - 0.8) / (2.6 + 0.8 * _hn(ii, 115))), 1)   # windows spread evenly on each wall
    sp = 2 * halfw / ncol
    fi = np.floor((along + halfw) / sp); fu = (along + halfw) / sp - fi
    ffl = np.floor(upz / 3.0); fz = upz - ffl * 3.0
    xin = np.abs(fu - 0.5) * sp
    open_col = (hash2(fi.astype(np.int64) + ii * 37 + xf * 1000, 0, 106) > np.where(xf, 0.25, 0.18)) & (hash2(fi.astype(np.int64), ffl.astype(np.int64) + ii * 7, 114) > 0.08)
    open_col |= (ncol == 1) & xf                                             # a gable end with one column always has it
    inwall = (fi >= 0) & (fi < ncol) & (upz < HW - 0.5) & (upz > 0.6) & ~roof
    dlong = 2 + (_hn(ii, 107) < 0.5); dgab = (_hn(ii, 123) < 0.5).astype(int)   # a front door on a long wall,
    gdoor = _hn(ii, 124) < 0.65                                                 # and often one in a gable end
    dpos = np.where(xf, (_hn(ii, 125) - 0.5) * np.maximum(2 * B - 3.0, 0) * 0.6, (_hn(ii, 108) - 0.5) * np.maximum(2 * A - 3.0, 0))
    ondoorwall = ((which == dlong) | ((which == dgab) & gdoor)) & ~roof & (kind == 0)
    door_zone = ondoorwall & (np.abs(along - dpos) < 1.4) & (ffl == 0)
    wbox = (xin < ww / 2) & (fz > 0.9) & (fz < 0.9 + wh) & open_col & inwall & ~door_zone
    attic = xf & ~roof & (INST["hr"][ii] > 1.5) & (np.abs(along) < 0.32) & (upz > HW + 0.25) & (upz < HW + 0.95)
    wbox |= attic; fz = np.where(attic, upz - HW + 0.65, fz); xin = np.where(attic, np.abs(along) * ww / 0.64, xin)
    wh = np.where(attic, 0.7, wh)
    frame = wbox & ((xin > ww / 2 - 0.1) | (fz < 1.0) | (fz > 0.9 + wh - 0.1) | (np.abs(xin) < 0.035) | (np.abs(fz - 0.9 - wh * 0.62) < 0.035))
    glass = wbox & ~frame
    lintel = glass & (fz > 0.9 + wh - 0.08 - 0.25 * wh)
    sill = (xin < ww / 2 + 0.07) & (fz > 0.82) & (fz <= 0.9) & open_col & inwall & ~door_zone
    shut = (_hn(ii, 109) < 0.35) & (xin >= ww / 2) & (xin < ww) & (fz > 0.9) & (fz < 0.9 + wh) & open_col & inwall & ~door_zone
    door = ondoorwall & (np.abs(along - dpos) < 0.5) & (upz < 2.15) & (upz > -0.5)
    dframe = door & ((np.abs(along - dpos) > 0.42) | (upz > 2.07))
    pal = np.array([[0.40, 0.39, 0.36], [0.14, 0.09, 0.06], [0.10, 0.17, 0.22], [0.12, 0.18, 0.12], [0.30, 0.30, 0.29]])
    trim = pal[(_hn(ii, 110) * 5).astype(int)]
    doorc = np.array([[0.11, 0.065, 0.04], [0.05, 0.08, 0.11], [0.08, 0.06, 0.045], [0.06, 0.09, 0.07]])[(_hn(ii, 111) * 4).astype(int)]
    pan = (np.abs(np.abs(along - dpos) - 0.2) < 0.13) & (np.mod(upz, 1.0) > 0.12) & (np.mod(upz, 1.0) < 0.88)   # two panel rows
    doorc = doorc * np.where(pan, 0.8, 1.0)[:, None] * (0.9 + 0.2 * tnoise(along * 6, upz * 0.8, sd * 5))[:, None]
    fdet = _fade(mpp, 0.12)[:, None]
    curtain = (np.abs(xin) < ww * 0.3) & (_hn(ii, 119) < 0.5) & (fz < 0.9 + wh * 0.6)
    glassc = np.where(curtain[:, None], np.array([0.09, 0.085, 0.075]), np.array([0.018, 0.02, 0.024])) * np.where(lintel, 0.55, 1.0)[:, None]
    below = (xin < ww / 2 + 0.05) & (fz < 0.82) & (fz > 0.82 - 0.9 * tnoise(along * 3 + sd, fz, 5.0)) & open_col & inwall
    wall = wall * (1 - 0.12 * below * _fade(mpp, 0.15))[:, None]              # sills drip: streaks under windows
    alb = wall
    alb = np.where(shut[:, None], trim * (0.85 + 0.25 * (np.mod(fz / 0.07, 1.0) > 0.35))[:, None], alb)
    alb = np.where(sill[:, None], wall * (1 - fdet) + np.array([0.42, 0.40, 0.37]) * fdet, alb)
    alb = np.where(frame[:, None], wall * (1 - fdet) + trim * fdet, alb)
    alb = np.where(glass[:, None], glassc, alb)
    alb = np.where(door[:, None], np.where(dframe[:, None], trim, doorc), alb)
    # ---- roofs
    sy = np.abs(ql[:, 1]); sx = ql[:, 0]
    tile = rm == 0; tin = rm == 1; slate = rm == 2
    course = np.where(tile, 0.3, 0.27); cy = sy / course; ci = np.floor(cy); fcy = cy - ci
    bw = np.where(tile, 0.22, 0.36); cx = sx / bw + 0.5 * ci; cxi = np.floor(cx)
    ttint = 0.82 + 0.36 * hash2(cxi.astype(np.int64) * 71 + ci.astype(np.int64), ii, 112)
    rtile = (1 - 0.45 * (fcy < 0.14) * _fade(mpp, course)) * (1 + (ttint - 1) * _fade(mpp, 0.3))
    rtile = rtile * np.where(tile, 0.85 + 0.15 * np.sin(2 * np.pi * cx) * _fade(mpp, 0.22), 1 - 0.3 * (np.mod(cx, 1.0) < 0.06) * _fade(mpp, 0.36))
    sheet = np.floor(sx / 0.9); srow = np.floor(sy / 2.2)
    stint = 0.85 + 0.3 * hash2(sheet.astype(np.int64) * 13 + srow.astype(np.int64), ii, 113)
    rib = 0.84 + 0.16 * np.abs(np.sin(np.pi * sx / 0.076)) ** 0.5
    rtin = (1 + (rib - 1) * _fade(mpp, 0.076)) * (1 + (stint - 1) * _fade(mpp, 0.9))
    rust = np.clip((tnoise(sx * 0.8 + sd * 40, sy * 0.8, sd * 7) - 0.5) * 3, 0, 1) * _fade(mpp, 0.8)
    rustc = np.array([0.23, 0.11, 0.05])
    rpat = np.where(tile, rtile, np.where(tin, rtin, rtile))
    rcol = roofc * rpat[:, None]
    rcol = np.where(tin[:, None], rcol * (1 - 0.6 * rust[:, None]) + rustc * 0.6 * rust[:, None], rcol)
    stain = np.clip((tnoise(sx * 3 + sd * 20, sy * 0.35, sd * 3) - 0.55) * 3, 0, 1) * _fade(mpp, 0.3)
    moss = np.clip((tnoise(sx * 0.6 + sd * 5, sy * 0.6 + 9, sd * 13) - 0.7) * 4, 0, 1) * _fade(mpp, 0.5) * ~tin
    rcol = rcol * (1 - 0.2 * stain)[:, None]
    rcol = rcol * (1 - 0.5 * moss[:, None]) + np.array([0.08, 0.085, 0.05]) * 0.5 * moss[:, None]
    ridge = sy < 0.16
    rcol = np.where(ridge[:, None], roofc * 0.75, rcol)
    alb = np.where(top[:, None], rcol, alb)
    alb = np.where(eave[:, None], roofc * 0.45, alb)
    gutter = top & (sy > B + 0.43)                                           # galvanised gutter along the eaves
    alb = np.where(gutter[:, None], np.array([0.17, 0.17, 0.18]) * (0.8 + 0.4 * tnoise(sx * 2, sd * 9, 1.0))[:, None], alb)
    chim = kind == 2                                                          # chimney: plaster, sooty cap, dark flue
    ccol = wallc * 0.9 * (0.85 + 0.3 * tnoise(ql[:, 0] * 3, ql[:, 1] * 3, upz * 3))[:, None]
    ccol = np.where((upz > HW + INST["hr"][ii] + 0.72)[:, None], np.array([0.08, 0.075, 0.07]), ccol)
    ccol = np.where((nl[:, 2] > 0.5)[:, None], np.array([0.02, 0.02, 0.02]), ccol)
    alb = np.where(chim[:, None], ccol, alb)
    spec = np.where(glass, 0.6, np.where(top & tin, 0.2 * (1 - rust), np.where(top, 0.08, 0.0)))
    spec = np.where(gutter, 0.3, spec)
    # micro-relief the low sun picks out: corrugations, tile barrels and course steps (local normal offsets)
    nb = np.zeros((N, 3))
    upslope = np.sign(ql[:, 1])                                               # roof falls away from the ridge
    rib_n = np.cos(np.pi * sx / 0.076) * 0.35 * _fade(mpp, 0.05)
    bar_n = np.cos(2 * np.pi * cx) * 0.35 * _fade(mpp, 0.15)
    step_n = (0.5 - fcy) * 0.5 * _fade(mpp, course * 0.6)
    nb[:, 0] = np.where(top & tin, rib_n, np.where(top & tile, bar_n, 0.0))
    nb[:, 1] = np.where(top & ~tin, upslope * step_n, 0.0)
    nb[:, 2] = np.where(top & ~tin, -np.abs(step_n) * 0.3, 0.0)
    nb[gutter | chim] = 0
    # ambient occlusion under the eaves and at the foot of the walls
    ao = np.clip(0.55 + upz / 6, 0.55, 1.0) * np.where(~xf & ~roof, 1 - 0.4 * np.clip((upz - HW + 0.7) / 0.7, 0, 1), 1.0)
    return alb, spec, glass, ao, nb

def shade_instances(P, D, ids, dist, O=None, lod_k=1.0):
    col = np.zeros((len(ids), 3))
    z = P[:, 2] + ((P[:, 0] - CAM[0]) ** 2 + (P[:, 1] - CAM[1]) ** 2) / (2 * RE)
    sunc = sun_T(np.maximum(z, 0)) * SUN_I * 0.17
    tpi = INST["tp"][ids]
    tr = tpi <= T_POP
    if tr.any():
        k = np.where(tr)[0]
        col[k] = shade_trees(P[k], D[k], ids[k], sunc[k], lod_k)
    wl = tpi >= T_WALL
    if wl.any():
        k = np.where(wl)[0]
        col[k] = shade_walls(P[k], D[k], ids[k], sunc[k], CAM[None] if O is None else O[k])
    hs = (tpi == T_HOUSE) | (tpi == T_BLOCK)
    if hs.any():
        k = np.where(hs)[0]; ii = ids[k]
        Ok = CAM[None] if O is None else O[k]
        t, which, kind, nl, q0, d0 = isect_houses(Ok, D[k], ii, want=True)
        ql = q0 + d0 * np.where(np.isfinite(t), t, 0)[:, None]
        yaw = INST["yaw"][ii]; c, s = np.cos(yaw), np.sin(yaw)
        n = np.stack([nl[:, 0] * c - nl[:, 1] * s, nl[:, 0] * s + nl[:, 1] * c, nl[:, 2]], 1)
        roof = (kind == 1) | (((which == 5) | (which == 6)) & (INST["tp"][ii] == T_BLOCK))
        roof |= (INST["tp"][ii] == T_BLOCK) & (ql[:, 2] > INST["hw"][ii] - 0.05)
        eave = (kind == 1) & (nl[:, 2] < 0.5)                                # roof edges and undersides
        wallc, roofc = INST["col"][ii, :3], INST["col"][ii, 3:]
        # windows on walls: rows per floor, columns every ~3 m
        along = np.where(np.abs(nl[:, 0]) > 0.5, ql[:, 1], ql[:, 0])
        fu = np.mod(along / 3.1 + INST["seed"][ii] * 7, 1.0); fz = np.mod(ql[:, 2] / 3.0, 1.0)
        win = (~roof) & (fu > 0.3) & (fu < 0.68) & (fz > 0.32) & (fz < 0.78) & (ql[:, 2] > 0.5) & (ql[:, 2] < INST["hw"][ii] - 0.3)
        dirt = 0.85 + 0.3 * tnoise(ql[:, 0] * 0.8, ql[:, 1] * 0.8, ql[:, 2] * 0.8)
        alb = np.where(roof[:, None], roofc, wallc) * dirt[:, None]
        alb = np.where(eave[:, None], roofc * 0.6, alb)
        alb = np.where(win[:, None], np.array([0.045, 0.05, 0.06]), alb)
        spec_w = np.where(win, 0.6, np.where(roof, 0.12, 0.0))
        ao = np.clip(0.55 + ql[:, 2] / 6, 0.55, 1.0)
        hm = INST["tp"][ii] == T_HOUSE                                     # village houses: close-up materials
        if hm.any():
            h_ = np.where(hm)[0]
            a2, s2, w2, o2, nb2 = house_albedo(ql[h_], which[h_], kind[h_], nl[h_], ii[h_], DI_M[ii[h_]] / fl)
            alb[h_] = a2; spec_w[h_] = s2; win[h_] = w2; ao[h_] = o2
            nl2 = nl[h_] + nb2; nl2 /= np.maximum(np.linalg.norm(nl2, axis=1, keepdims=True), 1e-9)
            n[h_] = np.stack([nl2[:, 0] * c[h_] - nl2[:, 1] * s[h_], nl2[:, 0] * s[h_] + nl2[:, 1] * c[h_], nl2[:, 2]], 1)
        sh = soft_shadow(P[k], n, steps=40) * inst_shadow(P[k], n)
        ndl = np.clip(n @ SUN, 0, 1)
        col[k] = alb * (sunc[k] * (ndl * sh)[:, None] + SKYC[None] * (ao * (0.6 + 0.4 * n[:, 2]))[:, None] + BOUNCE * ao[:, None]
                        + ground_bounce(n, sunc[k]) * ao[:, None])
        hv = SUN - D[k]; hv /= np.linalg.norm(hv, axis=1, keepdims=True)
        glass_ = win & (INST["tp"][ii] == T_HOUSE)
        spec = np.clip((n * hv).sum(1), 0, 1) ** np.where(glass_, 400, 40) * spec_w
        col[k] += (spec * sh)[:, None] * sunc[k]
        cth = np.clip(-(D[k] * n).sum(1), 0, 1)                              # glass mirrors the sky
        col[k] += (glass_ * (0.04 + 0.96 * (1 - cth) ** 5) * 0.9)[:, None] * SKYC[None] * 0.9
        if NIGHT_WIN or CITY_ON:
            fi = np.floor(along / 3.1 + INST["seed"][ii] * 7); fj = np.floor(ql[:, 2] / 3.0)
            occ = (0.06 + 0.45 * hash2(ii, 0, 50) ** 1.5) * INST_FRONT[ii]      # some buildings busy, some asleep
            lit = win & (hash2(fi.astype(np.int64) + ii * 131, fj.astype(np.int64), 51) < occ * LR_WIN)
            hc = hash2(fi.astype(np.int64), ii, 52)
            pal = np.array([[1.0, 0.60, 0.28], [1.0, 0.80, 0.55], [0.80, 0.88, 1.0], [0.45, 0.55, 1.0]])
            wc = pal[np.searchsorted([0.6, 0.88, 0.97], hc)]                     # mostly warm: tungsten and warm LED
            br = 0.25 + 1.0 * hash2(fi.astype(np.int64), fj.astype(np.int64) + ii, 53) ** 2
            col[k] += lit[:, None] * wc * WIN_L * br[:, None]
    if CITY_ON:                                                          # street light washing up the lower walls
        zl = (P[:, 2] - INST["z"][ids]) * 1000
        up_ = 0.6 + 2.4 * np.exp(-np.maximum(zl, 0) / 7.0)
        col += INST["col"][ids, :3] * city_ground_light(P[:, 0], P[:, 1]) * up_[:, None]
    return col
WIN_L = 0.12

# ------------------------------------------------------------------ night city: emitters, ground light, sky glow
if CITY_ON:
    CITY_GRID = 0.2                                                      # km, resolution of the light-density map
    # lamp colours relative to the white sun: sodium, LED 4000 K, cold LED / metal halide
    SODIUM, LED, COLD = np.array([1.0, 0.5, 0.15]), np.array([1.0, 0.80, 0.58]), np.array([0.85, 0.9, 1.0])
    def _way_geom():
        """per segment: its start along its way (km), and per way the terrain height at both ends (bridges)"""
        L_ = np.hypot(RSEG["x1"] - RSEG["x0"], RSEG["y1"] - RSEG["y0"]); w_ = RSEG["way"]
        cs = np.cumsum(L_); first = np.r_[0, np.where(np.diff(w_) != 0)[0] + 1]
        base = np.repeat(cs[first] - L_[first], np.diff(np.r_[first, len(w_)]))
        last = np.r_[first[1:] - 1, len(w_) - 1]
        wl = np.zeros(int(w_.max()) + 1 if len(w_) else 0); z0 = wl.copy(); z1 = wl.copy()
        wl[w_[first]] = cs[last] - base[first]
        z0[w_[first]] = hf(RSEG["x0"][first], RSEG["y0"][first]); z1[w_[first]] = hf(RSEG["x1"][last], RSEG["y1"][last])
        return L_, cs - L_ - base, wl, z0, z1
    def _on_roads(sel, t, lat, hgt):
        """points on segments sel at t km along them, lat metres to the left, hgt metres up (bridges: deck line)"""
        dx, dy = RSEG["x1"][sel] - RSEG["x0"][sel], RSEG["y1"][sel] - RSEG["y0"][sel]; Ls = np.maximum(np.hypot(dx, dy), 1e-9)
        x = RSEG["x0"][sel] + dx * t / Ls - dy / Ls * lat / 1000; y = RSEG["y0"][sel] + dy * t / Ls + dx / Ls * lat / 1000
        w_ = RSEG["way"][sel]; z = terrain_z(x, y)
        br = RWAY["bridge"][w_]
        if br.any():                                                       # bridges span gorges: straight deck
            f = np.clip((_S0[sel] + t) / np.maximum(_WL[w_], 1e-6), 0, 1)
            deck = _Z0[w_] + (_Z1[w_] - _Z0[w_]) * f - ((x - CAM[0]) ** 2 + (y - CAM[1]) ** 2) / (2 * RE)
            z = np.where(br, np.maximum(z, deck), z)
        return np.stack([x, y, z + hgt / 1000], 1), dx / Ls, dy / Ls
    if ROADS is not None:
        _SL, _S0, _WL, _Z0, _Z1 = _way_geom()
        _mid_ok, _ = _in_view((RSEG["x0"] + RSEG["x1"]) / 2, (RSEG["y0"] + RSEG["y1"]) / 2, margin=6.0, rmax=a.city_km)
    def _osm_lamps():
        """poles along the real streets: spacing, sides, height, lamp type and power by road class"""
        cl = RSEG["cls"].astype(int); w_ = RSEG["way"]; hw = hash2(RWAY["id"][w_] % 2147483647, 0, 90)
        ok = _mid_ok & RWAY["lit"][w_] & ~(RWAY["service"][w_] & (hash2(w_, 0, 91) < 0.85))
        sp = np.array([40, 38, 32, 32, 32, 36, 42.0])[cl] / 1000
        ph = hw * sp
        k0 = np.ceil((_S0 - ph) / sp); k1 = np.ceil((_S0 + _SL - ph) / sp)
        n = np.where(ok, np.maximum(k1 - k0, 0), 0).astype(int)
        sel = np.repeat(np.arange(len(n)), n); m = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        k = k0[sel] + m; t = ph[sel] + k * sp[sel] - _S0[sel]
        c_ = cl[sel]; wid = RSEG["width"][sel]
        both = c_ <= 3
        side = np.where(c_ == 4, np.where(np.mod(k, 2) == 0, 1, -1), np.where(hw[sel] < 0.5, 1, -1))
        keep = (c_ != 6) | (hash2(sel, k.astype(np.int64), 92) < 0.55)
        sel, t, c_, wid, side, k = (v[keep] for v in (sel, t, c_, wid, side, k))
        both = c_ <= 3
        sel = np.r_[sel, sel[both]]; t = np.r_[t, t[both]]                  # avenues: poles on both kerbs
        side = np.r_[side, -side[both]]; c_ = np.r_[c_, c_[both]]; wid = np.r_[wid, wid[both]]; k = np.r_[k, k[both]]
        hgt = np.array([11, 11, 10, 10, 9, 8, 7.0])[c_]
        P, _, _ = _on_roads(sel, t, side * (0.5 * wid + 1.0), hgt)
        u = hash2(RWAY["id"][RSEG["way"][sel]] % 2147483647, 0, 93)             # one lamp type per street
        led = np.array([0.8, 0.8, 0.75, 0.55, 0.4, 0.25, 0.3])[c_]
        colr = np.where((u < led)[:, None], LED, SODIUM)
        colr = np.where((u > 0.96)[:, None], COLD, colr)
        power = np.array([3.0, 3.0, 2.7, 2.4, 2.0, 1.4, 0.9])[c_] * (0.9 + 0.2 * hash2(sel, k.astype(np.int64), 94))
        return P, colr * power[:, None]
    def _traffic():
        """cars on trunk..tertiary roads, driving on the right: head lights toward the lens, tail lights away;
        with --shutter, each car is a streak of points along the distance it covers during the exposure"""
        if a.traffic <= 0: return np.zeros((0, 3)), np.zeros((0, 3))
        cl = RSEG["cls"].astype(int); w_ = RSEG["way"]
        ok = _mid_ok & (cl >= 1) & (cl <= 4)
        gap = np.array([1, 32, 32, 45, 90, 1, 1.0])[cl] / 1000 / a.traffic
        ow = RWAY["oneway"][w_]
        nl = np.clip(np.round(RSEG["width"] / 3.3 / np.where(ow, 1, 2)), 1, 4).astype(int)
        seg_i, dr_, ln_ = [], [], []
        for d_ in (1, -1):
            okd = ok & ((d_ == 1) | ~ow)
            for l_ in range(4):
                s_ = np.where(okd & (nl > l_))[0]; seg_i.append(s_); dr_.append(np.full(len(s_), d_)); ln_.append(np.full(len(s_), l_))
        seg_i, dr_, ln_ = np.concatenate(seg_i), np.concatenate(dr_), np.concatenate(ln_)
        nc = np.floor(_SL[seg_i] / gap[seg_i] + hash2(seg_i, dr_ * 7 + ln_, 95)).astype(int)
        car = np.repeat(np.arange(len(seg_i)), nc); j = np.arange(nc.sum()) - np.repeat(np.cumsum(nc) - nc, nc)
        si, d_, l_ = seg_i[car], dr_[car], ln_[car]
        t = (j + hash2(si * 8 + l_, j * 2 + (d_ > 0), 96)) * gap[si]
        t = np.minimum(t, _SL[si])
        v = np.array([1, 22, 18, 15, 12, 1, 1.0])[cl[si]] * (0.7 + 0.5 * hash2(si, j, 97))     # m/s
        streak = v * a.shutter / 1000                                                       # km
        npt = np.clip(np.ceil(streak * 1000 / 1.5), 1, 12).astype(int)
        P_, C_ = [], []
        for side_off in (-0.75, 0.75):
            lat = -d_ * ((l_ + 0.5) * 3.3) + side_off                                   # right-hand traffic
            P0, ux, uy = _on_roads(si, t, lat, 0.75)
            tx, ty = ux * d_, uy * d_                                                    # travel direction
            vx, vy = CAM[0] - P0[:, 0], CAM[1] - P0[:, 1]; vn = np.hypot(vx, vy) + 1e-9
            cv = (tx * vx + ty * vy) / vn
            head = cv > 0
            col = np.where(head[:, None], np.array([1.0, 0.92, 0.78]) * (0.12 + 1.1 * np.clip(cv, 0, 1) ** 8)[:, None],
                           np.array([1.0, 0.07, 0.03]) * (0.1 + 0.25 * np.clip(-cv, 0, 1) ** 2)[:, None])
            rep = np.repeat(np.arange(len(t)), npt); q = np.arange(npt.sum()) - np.repeat(np.cumsum(npt) - npt, npt)
            back = streak[rep] * (q + 0.5) / npt[rep]                                     # earlier positions
            P_.append(P0[rep] - np.stack([tx[rep] * back, ty[rep] * back, np.zeros(len(rep))], 1))
            C_.append(col[rep] / npt[rep, None])
        return np.concatenate(P_), np.concatenate(C_) * 0.5
    def _street_lights():
        if ROADS is not None: return _osm_lamps()
        return _grid_lights()
    def _grid_lights():
        """street lamps every ~35 m along a street grid aligned with the field pattern, inside the city field"""
        cs = 0.035; rot = (ROT_C, ROT_S)
        I, J, gi = _grid_cells(cs, rot, a.city_km)
        u, v = (I + 0.5) * cs, (J + 0.5) * cs
        x, y = u * ROT_C - v * ROT_S, u * ROT_S + v * ROT_C
        cf_ = city_field(x, y); vf = village_field(x, y) * 0.6
        dens = np.maximum(cf_, vf * (a.villages > 0))
        z = hf(x, y); sl = _slope(x, y)
        keep = (hash2(I, J, 61) < dens * 0.9 + 0.02 * (dens > 0)) & _ground_ok(x, y, z, sl) & (sl < 0.4)
        # snap lamps onto street lines (house grid: streets every 5 x 4 plots of 26 m)
        su = (np.round((u / 0.026 - 0.5) / 5) * 5 + 0.5) * 0.026            # centre lines of the street plots
        sv = (np.round((v / 0.026 - 0.5) / 4) * 4 + 0.5) * 0.026
        alongu = hash2(I, J, 62) < 0.5
        u2 = np.where(alongu, su, u); v2 = np.where(alongu, v, sv)
        x, y = u2 * ROT_C - v2 * ROT_S, u2 * ROT_S + v2 * ROT_C
        x, y, I, J, dens = x[keep], y[keep], I[keep], J[keep], dens[keep]
        # a few highways / avenues: brighter, denser
        z = terrain_z(x, y) + 0.009
        kind = hash2(I, J, 63)
        colr = np.where((kind < 0.55)[:, None], np.array([1.0, 0.55, 0.18]),                # sodium
               np.where((kind < 0.9)[:, None], np.array([0.95, 0.92, 0.85]), np.array([0.75, 0.9, 1.0])))   # LED / cold
        power = (0.6 + 0.8 * hash2(I, J, 64)) * (0.6 + 0.4 * dens)
        return np.stack([x, y, z], 1), colr * power[:, None]
    def _window_lights():
        """one point per lit building face cluster (far buildings are below a pixel anyway)"""
        if not NI: return np.zeros((0, 3)), np.zeros((0, 3))
        b = np.where((INST["tp"] == T_HOUSE) | (INST["tp"] == T_BLOCK))[0]
        n_ = np.where(INST["tp"][b] == T_BLOCK, 6, 2)
        bi = np.repeat(b, n_); k = np.arange(len(bi)) - np.repeat(np.cumsum(n_) - n_, n_)
        h1, h2, h3 = hash2(bi, k, 71), hash2(bi, k, 72), hash2(bi, k, 73)
        lit = h3 < (0.06 + 0.45 * hash2(bi, 0, 50) ** 1.5) * LR_WIN * INST_FRONT[bi]
        yaw = INST["yaw"][bi]; A_, B_ = INST["a"][bi], INST["b"][bi]
        lx = (h1 - 0.5) * 2 * A_; ly = np.where(h2 < 0.5, -B_ - 0.3, B_ + 0.3); lz = 1.5 + (h2 * 2 % 1) * (INST["hw"][bi] - 2)
        c, s = np.cos(yaw), np.sin(yaw)
        x = INST["x"][bi] + (lx * c - ly * s) / 1000; y = INST["y"][bi] + (lx * s + ly * c) / 1000
        z = INST["z"][bi] + lz / 1000
        warm = np.where((hash2(bi, k, 74) < 0.82)[:, None], np.array([1.0, 0.62, 0.3]), np.array([0.8, 0.88, 1.0]))
        far = np.hypot(x - CAM[0], y - CAM[1]) > 1.2                        # near windows are drawn on the facades
        m = lit & far
        return np.stack([x, y, z], 1)[m], warm[m] * 0.09
    def _near_lamps():
        """a sagging string of festoon lamps 3.5-6 m in front of the lens, across the bottom of the frame (bokeh)"""
        n = a.near_lamps
        if n <= 0: return np.zeros((0, 3)), np.zeros((0, 3))
        k = np.arange(n); u = (k + 0.5 + 0.3 * (hash2(k, 0, 80) - 0.5)) / n
        ang = np.radians((u - 0.5) * a.hfov * 1.02)
        dist = (0.0035 + 0.0025 * u) * (0.8 + 0.45 * hash2(k, 0, 81))
        dirh = np.stack([FH[0] * np.cos(ang) - FH[1] * np.sin(ang), FH[0] * np.sin(ang) + FH[1] * np.cos(ang)], 1)
        x = CAM[0] + dirh[:, 0] * dist; y = CAM[1] + dirh[:, 1] * dist
        row = 0.83 + 0.11 * (1 - (2 * u - 1) ** 2)                            # sag: lowest in the middle
        vy = (H * row - H / 2 - 0.02 * H) / fl
        z = CAM[2] + dist * (fwd[2] / math.hypot(fwd[0], fwd[1]) - vy)
        c = np.where((hash2(k, 0, 83) < 0.75)[:, None], np.array([1.0, 0.50, 0.16]), np.array([1.0, 0.70, 0.35]))
        return np.stack([x, y, z], 1), c * 0.0010 * (0.5 + 0.9 * hash2(k, 0, 84))[:, None]
    def write_lights():
        P1, C1 = _LAMPS; P2, C2 = _window_lights(); P3, C3 = _near_lamps()
        P4, C4 = _traffic() if ROADS is not None else (np.zeros((0, 3)), np.zeros((0, 3)))
        C1 = C1 * LR_ST; C3 = C3 * LR_ST; C4 = C4 * LR_ST
        P = np.concatenate([P1, P2, P3, P4]); C = np.concatenate([C1, C2, C3, C4])
        xs, ys, zc = _project(P)
        ok = (zc > 0.001) & (xs > -60) & (xs < W + 60) & (ys > -60) & (ys < H + 60)
        P, C, xs, ys, zc = P[ok], C[ok], xs[ok], ys[ok], zc[ok]
        dvec = P - CAM; d = np.linalg.norm(dvec, axis=1); D = dvec / d[:, None]
        Tv = np.ones((len(P), 3))
        for s0 in range(0, len(P), 50000):
            _, Tv[s0:s0 + 50000] = atmosphere(D[s0:s0 + 50000], d[s0:s0 + 50000])
        # flux on the sensor falls as 1/d^2; LIGHT_K sets lamp brightness in render units
        inv2 = 1.0 / np.maximum(d * 1000, 4.0)[:, None] ** 2                 # point source: flux on the sensor ~ 1/d^2
        flux = C * Tv * inv2 * LIGHT_K * (W / 3000) ** 2
        halo = C * (1 - Tv.mean(1, keepdims=True)) * inv2 * LIGHT_K * 2.5 * (W / 3000) ** 2
        np.savez(os.path.join(BANDS, "lights.npz"), sx=xs, sy=ys, d=d, flux=flux.astype(np.float32), halo=halo.astype(np.float32))
        print(f"city: {len(P)} lights in view ({len(P1)} street, {len(P2)} window, {len(P4)} car-light points)", flush=True)
    LIGHT_K = 6e5
    _LAMPS = _street_lights()
    # ground illumination and sky glow from a light-density map
    _gx = np.arange(CAM[0] - a.city_km - 1, CAM[0] + a.city_km + 1, CITY_GRID)
    _gy = np.arange(CAM[1] - a.city_km - 1, CAM[1] + a.city_km + 1, CITY_GRID)
    _GX, _GY = np.meshgrid(_gx, _gy)
    def _dmap(Mp, x, y):
        return map_coordinates(Mp, [(y - _gy[0]) / CITY_GRID, (x - _gx[0]) / CITY_GRID], order=1, mode="constant", cval=0.0)
    if ROADS is None:
        _dens = city_field(_GX.ravel(), _GY.ravel()).reshape(_GX.shape)
        _dens *= (_water(_GX.ravel(), _GY.ravel()).reshape(_GX.shape) < 0.3)
        _DEN_S = gaussian_filter(_dens, 0.8); _DEN_L = gaussian_filter(_dens, 6)
        def city_ground_light(x, y):
            f = _dmap(_DEN_S, x, y) * (0.5 + 0.5 * fbm2(x * 30, y * 30, 2))
            return (f * 0.022 * LR_ST)[:, None] * np.array([1.0, 0.62, 0.30])
        def _glow(sx, sy, alt):
            shp = sx.shape
            g = (_dmap(_DEN_S, sx.ravel(), sy.ravel()) * 0.6 + _dmap(_DEN_L, sx.ravel(), sy.ravel()) * 0.4).reshape(shp)
            hrel = np.maximum(alt - BASE, 0)
            return (g * np.exp(-hrel / 1.0))[..., None] * np.array([1.0, 0.62, 0.30]) * 0.30 * a.city * LR_ST
    else:
        # ground light: every lamp's pool (sigma 7 m) plus a wider spill, splatted on a 4 m grid of the view wedge
        _LP, _LC = _LAMPS
        _lr = 0.004
        _lx0, _ly0 = _LP[:, 0].min() - 0.1, _LP[:, 1].min() - 0.1
        _ln = (int((_LP[:, 1].max() + 0.1 - _ly0) / _lr) + 1, int((_LP[:, 0].max() + 0.1 - _lx0) / _lr) + 1)
        _LG = np.zeros(_ln + (3,), np.float32)
        _li, _lj = ((_LP[:, 1] - _ly0) / _lr).astype(int), ((_LP[:, 0] - _lx0) / _lr).astype(int)
        for c_ in range(3): np.add.at(_LG[..., c_], (_li, _lj), _LC[:, c_])
        _LG = gaussian_filter(_LG, (1.75, 1.75, 0)) * 0.22 + gaussian_filter(_LG, (6, 6, 0)) * 0.06
        print(f"streets: {len(_LP)} lamps along the real streets", flush=True)
        def city_ground_light(x, y):
            f = [map_coordinates(_LG[..., c_], [(y - _ly0) / _lr - 0.5, (x - _lx0) / _lr - 0.5], order=1, mode="constant", cval=0.0) for c_ in range(3)]
            return np.stack(f, -1) * LR_ST
        # sky glow from the lamp power per 200 m cell, with the lamps' own colour mix
        _gi, _gj = ((_LP[:, 1] - _gy[0]) / CITY_GRID).astype(int), ((_LP[:, 0] - _gx[0]) / CITY_GRID).astype(int)
        _in = (_gi >= 0) & (_gi < len(_gy)) & (_gj >= 0) & (_gj < len(_gx))
        _GM = np.zeros((len(_gy), len(_gx), 3))
        for c_ in range(3): np.add.at(_GM[..., c_], (_gi[_in], _gj[_in]), _LC[_in, c_])
        _GM /= 45.0                                                           # ~45 lamps per 200 m cell = dense centre
        _DEN_S3 = gaussian_filter(_GM, (0.8, 0.8, 0)); _DEN_L3 = gaussian_filter(_GM, (6, 6, 0))
        def _glow(sx, sy, alt):
            shp = sx.shape
            g = np.stack([(_dmap(_DEN_S3[..., c_], sx.ravel(), sy.ravel()) * 0.6 + _dmap(_DEN_L3[..., c_], sx.ravel(), sy.ravel()) * 0.4).reshape(shp)
                          for c_ in range(3)], -1)
            hrel = np.maximum(alt - BASE, 0)
            return g * np.exp(-hrel / 1.0)[..., None] * GLOW_K * a.city * LR_ST
    GLOW_K = 0.30
    GLOW = _glow

# ------------------------------------------------------------------ cloud layer (cumulus / stratus)
if CLOUD_LAYER_ON:
    CLB, CLT = (BASE + 1.5, BASE + 3.0) if a.cloud_layer == "cumulus" else (BASE + 1.9, BASE + 2.4)
    CLB = max(CLB, CAM[2] + 0.4); CLT = max(CLT, CLB + 0.5)
    def layer_density(p):
        hgt = (p[:, 2] - CLB) / (CLT - CLB)
        if a.cloud_layer == "cumulus":
            base_n = fbm3(p[:, 0] * 0.6, p[:, 1] * 0.6, p[:, 2] * 0.3 + 1.0, 3)
            thr = 0.66 - 0.28 * a.cloud_cover + 0.22 * hgt                  # flat bases, tops narrowing into domes
            det = fbm3(p[:, 0] * 2.6, p[:, 1] * 2.6, p[:, 2] * 2.6, 2)
            d = np.clip((base_n - thr) * 6 - det * 0.45, 0, None) * np.clip(hgt * 12, 0, 1) * (hgt < 1) * 6
        else:
            base_n = fbm3(p[:, 0] * 0.22, p[:, 1] * 0.5, p[:, 2] * 0.6, 3)
            prof = np.clip(1 - np.abs(hgt - 0.5) * 2, 0, 1) ** 0.5
            d = np.clip((base_n - (1 - a.cloud_cover) * 0.65) * prof * 3, 0, None) * 2.2
        return d
    def apply_cloud_layer(dirs, tmax, col, O=None, n=64, extra=None):
        Oc = CAM[None] if O is None else O
        dz = np.where(np.abs(dirs[:, 2]) < 1e-4, 1e-4, dirs[:, 2])
        tb, tt = (CLB - Oc[:, 2]) / dz, (CLT - Oc[:, 2]) / dz
        t0 = np.maximum(np.minimum(tb, tt), 0); t1 = np.minimum(np.minimum(np.maximum(tb, tt), tmax), t0 + 22)
        idx = np.where(t1 > t0)[0]
        if not len(idx): return col
        Oi = Oc if O is None else Oc[idx]
        dd = dirs[idx]; a0, a1 = t0[idx], t1[idx]; dtv = (a1 - a0) / n
        Tr = np.ones(len(idx)); Cc = np.zeros((len(idx), 3))
        sun_col = sun_T(np.array([(CLB + CLT) / 2]))[0] * SUN_I * 0.16
        amb = np.array([0.35, 0.45, 0.65]) * 0.16 * AMB_SCALE
        mu = dd @ SUN
        ph0 = (0.7 * hg(mu, 0.8) + 0.3 * hg(mu, -0.25) + 0.35 * hg(mu, 0.97)) * 4 * np.pi * 0.55
        jit = 0.5 + 0.35 * (np.random.default_rng(len(idx)).random(len(idx)) - 0.5)   # mild jitter: less banding, little speckle
        for k in range(n):
            p = Oi + dd * (a0 + (k + jit) * dtv)[:, None]; den = layer_density(p); m = den > 1e-3
            if not m.any(): continue
            odl = sum(layer_density(p[m] + SUN * 0.12 * j * j) * 0.12 * (2 * j - 1) for j in range(1, 5))
            sun_term = np.zeros((m.sum(), 3)); aa, bb, cc_ = 1.0, 1.0, 1.0
            for _o in range(4):
                sun_term += aa * np.exp(-odl * bb)[:, None] * (ph0[m] * cc_ + (1 - cc_))[:, None]
                aa, bb, cc_ = aa * 0.5, bb * 0.4, cc_ * 0.5
            hgt = np.clip((p[m, 2] - CLB) / (CLT - CLB), 0, 1)
            powder = 1 - 0.6 * np.exp(-den[m] * 0.5)
            light = sun_term * (sun_col / 1.6)[None] * powder[:, None] + amb[None] * (0.45 + 0.6 * hgt)[:, None]
            st = np.exp(-den[m] * dtv[m] * 1.4); Cc[m] += (Tr[m] * (1 - st))[:, None] * light; Tr[m] *= st
        # the cloud sits at some distance: blend it through the air in front of it
        tc = (a0 + a1) / 2
        Lc, Tc = atmosphere(dd, np.minimum(tc, tmax[idx]), O=None if O is None else Oi)
        out = col.copy()
        out[idx] = Lc + Tc * Cc + Tr[:, None] * (col[idx] - Lc)
        if extra is not None: extra[idx] *= Tr[:, None]                   # stars behind the layer
        return out
