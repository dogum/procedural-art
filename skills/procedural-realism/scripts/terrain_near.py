# terrain_near.py — near-field detail for pbr_terrain.py. Not a standalone script: pbr_terrain.py runs
# this file inside its own namespace when --trees, --villages, --city or --cloud-layer is used, so it
# reads the renderer's globals (a, CAM, hf, terrain_z, fwd, right, up, fl, W, H, SUN, SKYC, ...).
#
#   trees     conifer (tiered cone), deciduous (lumpy ellipsoid), poplar (flame) on jittered grids,
#             placed by slope / elevation / water / forest-noise masks; poplar rows on field edges
#   houses    convex boxes with gable roofs on a grid aligned with the field pattern; flat-roofed
#             apartment blocks when --city is on
#   render    each instance is projected to its screen bounding box; only those pixels cast rays
#             against it (2x2 sub-samples for anti-aliasing), so cost scales with covered pixels
#   shadows   sun rays walk the instance grids (closed-form shapes) for terrain and instance hits
#   city      street-grid emitters, lit windows and a sky-glow field for post_terrain.py to splat
#   clouds    optional cumulus / stratus layer (raymarched slab, octave multiple scattering)

NEAR = a.near_km
T_CON, T_DEC, T_POP, T_HOUSE, T_BLOCK = 0, 1, 2, 3, 4
ROT_C, ROT_S = 0.82, 0.57                                   # field grid orientation (see terrain_albedo)
FH = fwd[:2] / np.linalg.norm(fwd[:2])

def _in_view(x, y, margin=7.0, rmax=None):
    vx, vy = x - CAM[0], y - CAM[1]; d = np.hypot(vx, vy)
    c = (vx * FH[0] + vy * FH[1]) / np.maximum(d, 1e-9)
    return ((c > math.cos(math.radians(a.hfov / 2 + margin))) | (d < 0.05)) & (d < (rmax or NEAR)), d

def _slope(x, y):
    e = 0.03
    return np.hypot((hf(x + e, y) - hf(x - e, y)) / (2 * e), (hf(x, y + e) - hf(x, y - e)) / (2 * e))

def _water(x, y):
    return water_frac(x, y) if WATER else np.zeros_like(x)

def village_field(x, y):
    """0..1: where settlements are (flat, low, dry ground; patches 1-3 km)"""
    v = fbm2(x * 0.42 + 40.0, y * 0.42 - 17.0, 4)
    thr = 0.66 - 0.22 * max(a.villages, a.city * 0.6)
    return np.clip((v - thr) * 7, 0, 1)

def city_field(x, y):
    """0..1 urban density for the night city: dense core near the camera, thinning with distance"""
    if not CITY_ON: return np.zeros_like(x)
    d = np.hypot(x - CAM[0], y - CAM[1])
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

def near_ground(x, y, z, slope, alb, sn, dist):
    fade = np.clip(1 - dist / (NEAR * 1.3), 0, 1)[:, None]
    td = tree_density(x, y, z, slope)[:, None] * fade * (1 - sn)
    alb = alb * (1 - 0.45 * td) + np.array([0.05, 0.062, 0.032]) * 0.45 * td             # shaded floor under canopy
    if VILLAGES_ON or CITY_ON:
        vf = np.maximum(village_field(x, y) * (a.villages > 0), city_field(x, y))[:, None] * np.clip(1 - dist / (max(NEAR, a.city_km) * 1.5), 0, 1)[:, None]
        yard = np.array([0.24, 0.215, 0.19]) * (0.8 + 0.4 * fbm2(x * 60, y * 60, 2))[:, None]
        alb = alb * (1 - 0.6 * vf) + yard * 0.6 * vf
    return alb

# ------------------------------------------------------------------ instance generation
_near_keys = {k: getattr(a, k) for k in ("peak", "view_from", "cam_height", "cam_offset", "hfov", "layout", "pitch", "horizon", "second",
                                          "trees", "tree_mix", "villages", "near_km", "seed", "biome", "treeline", "season",
                                          "water_level", "water_seed", "city", "city_km")}
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
    return np.tile([0.05, 0.95, 0.0], (len(z), 1))

def build_instances():
    L = {k: [] for k in ("x", "y", "tp", "h", "r", "a", "b", "hw", "hr", "yaw", "seed", "col")}
    grids = []
    def add(x, y, tp, h, r, a_, b_, hw, hr, yaw, seed, col):
        for k, v in zip(L, (x, y, tp, h, r, a_, b_, hw, hr, yaw, seed, col)): L[k].append(np.asarray(v))
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
        occ = np.maximum(vf * 0.75, cf_ * 0.5 * np.clip((fbm2(x * 3.0 + 9, y * 3.0 - 4, 3) - 0.3) * 3, 0.15, 1))   # parks, gaps
        keep = (hash2(I, J, 3) < occ) & _ground_ok(x, y, z, sl) & (sl < 0.35) & ~(street & (cf_ > 0.3))
        d_ = np.hypot(x - CAM[0], y - CAM[1]); keep &= d_ > 0.03
        x, y, z, hu, hv, I, J, cf_ = x[keep], y[keep], z[keep], hu[keep], hv[keep], I[keep], J[keep], cf_[keep]
        big = (cf_ > 0.35) & (hash2(I, J, 4) < 0.1 + 0.28 * cf_)
        h1, h2, h3 = hash2(I, J, 5), hash2(I, J, 6), hash2(I, J, 7)
        A_ = np.where(big, 7 + 5 * h1, 4.0 + 3.2 * h1); B_ = np.where(big, 5.5 + 4 * h2, 3.4 + 1.8 * h2)
        floors = np.where(big, 3 + np.floor(h3 ** 2.2 * 9 * (0.3 + cf_)), 1 + (h3 > 0.55))
        HW = floors * 3.0 + 0.4; HR = np.where(big, 0.0, B_ * np.tan(np.radians(18 + 16 * h2)))
        yaw = math.atan2(ROT_S, ROT_C) + (hash2(I, J, 8) - 0.5) * np.where(cf_ > 0.3, 0.35, 0.12) + (hash2(I, J, 9) > 0.7) * np.pi / 2
        jit = CHc * 1000 / 2 - np.maximum(A_, B_) - 1.0
        x = x + (hu - 0.5) * 2 * np.maximum(jit, 0) / 1000 * 0.8; y = y + (hv - 0.5) * 2 * np.maximum(jit, 0) / 1000 * 0.8
        if a.biome == "temperate":
            walls = np.array([[0.50, 0.48, 0.45], [0.42, 0.39, 0.35], [0.34, 0.34, 0.34], [0.46, 0.43, 0.37]])
            roofs = np.array([[0.10, 0.11, 0.13], [0.16, 0.12, 0.10], [0.10, 0.15, 0.26], [0.22, 0.22, 0.23]])
        else:
            walls = np.array([[0.36, 0.25, 0.21], [0.37, 0.32, 0.26], [0.29, 0.27, 0.25], [0.42, 0.39, 0.34]])
            roofs = np.array([[0.30, 0.12, 0.08], [0.30, 0.31, 0.32], [0.12, 0.20, 0.15], [0.20, 0.14, 0.11]])
        wc = walls[(hash2(I, J, 10) * 4).astype(int)] * (0.85 + 0.3 * hash2(I, J, 11))[:, None]
        # apartment blocks: weathered tuff, concrete and painted plaster, darker and more varied than village walls
        blockw = np.array([[0.30, 0.20, 0.17], [0.27, 0.25, 0.22], [0.33, 0.29, 0.23], [0.22, 0.21, 0.20], [0.36, 0.33, 0.30]])
        bw = blockw[(hash2(I, J, 15) * 5).astype(int)] * (0.65 + 0.5 * hash2(I, J, 16))[:, None]
        wc = np.where(big[:, None], bw, wc)
        rc = roofs[(hash2(I, J, 12) * 4).astype(int)] * (0.8 + 0.4 * hash2(I, J, 13))[:, None]
        rc = np.where(big[:, None], np.array([0.20, 0.19, 0.18]) * (0.7 + 0.6 * hash2(I, J, 17))[:, None], rc)
        col = np.concatenate([wc, rc], 1)
        start = n0()
        add(x, y, np.where(big, T_BLOCK, T_HOUSE), HW + HR, np.hypot(A_, B_), A_, B_, HW, HR, yaw, hash2(I, J, 14), col)
        ids = np.full((gi[2], gi[3]), -1, np.int32); ids[I - gi[0], J - gi[1]] = start + np.arange(len(x))
        grids.append(dict(c=CHc, rot=rotF, i0=gi[0], j0=gi[1], ids=ids))
        house_ids = grids[-1]
        print(f"near field: {len(x)} buildings", flush=True)
    # --- trees on a jittered grid
    if TREES_ON:
        CT = 0.0085
        I, J, gi = _grid_cells(CT, rot0, NEAR)
        x = (I + 0.1 + 0.8 * hash2(I, J, 21)) * CT; y = (J + 0.1 + 0.8 * hash2(I, J, 22)) * CT
        z = hf(x, y); sl = _slope(x, y); d_ = np.hypot(x - CAM[0], y - CAM[1])
        p = tree_density(x, y, z, sl) * np.clip((NEAR - d_) / (0.3 * NEAR), 0, 1)
        keep = (hash2(I, J, 23) < p) & _ground_ok(x, y, z, sl) & (d_ > 0.025)
        if house_ids is not None:                                          # not inside a building plot
            u, v = x * ROT_C + y * ROT_S, -x * ROT_S + y * ROT_C
            hi = np.floor(u / house_ids["c"]).astype(int) - house_ids["i0"]; hj = np.floor(v / house_ids["c"]).astype(int) - house_ids["j0"]
            inb = (hi >= 0) & (hj >= 0) & (hi < house_ids["ids"].shape[0]) & (hj < house_ids["ids"].shape[1])
            occ = np.zeros(len(x), bool); occ[inb] = house_ids["ids"][hi[inb], hj[inb]] >= 0
            keep &= ~occ
        x, y, z, I, J, sl = x[keep], y[keep], z[keep], I[keep], J[keep], sl[keep]
        wmix = _mix_weights(z); r_ = hash2(I, J, 24)
        tp = np.where(r_ < wmix[:, 0], T_CON, np.where(r_ < wmix[:, 0] + wmix[:, 1], T_DEC, T_POP))
        h1, h2 = hash2(I, J, 25), hash2(I, J, 26)
        size = 0.75 + 0.5 * fbm2(x * 5 + 2, y * 5 - 1, 2)
        orchard = (a.biome != "temperate") & (village_field(x, y) > 0.05)
        Ht = np.select([tp == T_CON, tp == T_POP, orchard], [13 + 13 * h1, 16 + 11 * h1, 4.5 + 3 * h1], 8 + 8 * h1) * size
        R = np.select([tp == T_CON, tp == T_POP, orchard], [Ht * (0.17 + 0.07 * h2), 1.3 + 0.8 * h2, 2.0 + 1.4 * h2], (2.6 + 2.2 * h2) * size)
        col = tree_colours(tp, hash2(I, J, 27), hash2(I, J, 28))
        start = n0()
        add(x, y, tp, Ht, R, R, R, Ht, 0 * Ht, 0 * Ht, hash2(I, J, 29), np.concatenate([col, col], 1))
        ids = np.full((gi[2], gi[3]), -1, np.int32); ids[I - gi[0], J - gi[1]] = start + np.arange(len(x))
        grids.append(dict(c=CT, rot=rot0, i0=gi[0], j0=gi[1], ids=ids))
        print(f"near field: {len(x)} trees", flush=True)
        # --- poplar rows along field edges (arid valleys; or when the mix asks for poplars)
        want_rows = a.biome != "temperate" or (a.tree_mix != "auto" and _mix_weights(np.zeros(1))[0, 2] > 0)
        if want_rows:
            CP = 0.005
            I, J, gi = _grid_cells(CP, rotF, NEAR)
            u, v = (I + 0.5) * CP, (J + 0.5) * CP
            x, y = u * ROT_C - v * ROT_S, u * ROT_S + v * ROT_C
            w = fbm2(x * 1.3, y * 1.3, 3) * 0.6
            su, sv = 0.45 + 0.3 * w, 0.3 + 0.15 * w
            gu, gv = u / su + w, v / sv - w                                  # same cell maths as the field colours
            du, dv = np.abs(gu - np.round(gu)) * su, np.abs(gv - np.round(gv)) * sv
            onu, onv = du < CP / 2, dv < CP / 2
            rowid = np.where(onu, np.round(gu) * 7919 + np.floor(gv), np.round(gv) * 104729 + np.floor(gu) + 5e5).astype(np.int64)
            pick = hash2(rowid, 0, 31) < (0.05 + 0.3 * village_field(x, y)) * min(1.0, a.trees * 1.25)
            z = hf(x, y); sl = _slope(x, y); d_ = np.hypot(x - CAM[0], y - CAM[1])
            keep = (onu ^ onv) & pick & _ground_ok(x, y, z, sl) & (z < BASE + 0.45) & (sl < 0.2) & (d_ > 0.03) & (hash2(I, J, 32) < 0.9)
            keep &= (d_ < NEAR * (0.7 + 0.3 * hash2(I, J, 33)))
            # snap onto the edge so the row is straight
            x = np.where(onu, x - (gu - np.round(gu)) * su * ROT_C, x + (gv - np.round(gv)) * sv * ROT_S)
            y = np.where(onu, y - (gu - np.round(gu)) * su * ROT_S, y - (gv - np.round(gv)) * sv * ROT_C)
            x, y, I, J = x[keep], y[keep], I[keep], J[keep]
            h1, h2 = hash2(I, J, 34), hash2(I, J, 35)
            Ht = 14 + 10 * h1; R = 1.4 + 0.8 * h2
            col = tree_colours(np.full(len(x), T_POP), hash2(I, J, 36), hash2(I, J, 37))
            start = n0()
            add(x, y, np.full(len(x), T_POP), Ht, R, R, R, Ht, 0 * Ht, 0 * Ht, hash2(I, J, 38), np.concatenate([col, col], 1))
            ids = np.full((gi[2], gi[3]), -1, np.int32); ids[I - gi[0], J - gi[1]] = start + np.arange(len(x))
            grids.append(dict(c=CP, rot=rotF, i0=gi[0], j0=gi[1], ids=ids))
            print(f"near field: {len(x)} poplars in rows", flush=True)
    if not L["x"]:
        return {k: np.zeros(0) for k in L}, grids
    inst = {k: np.concatenate(v) for k, v in L.items()}
    inst["col"] = inst["col"].reshape(-1, 6)
    inst["tp"] = inst["tp"].astype(np.int8)
    inst["z"] = terrain_z(inst["x"], inst["y"])
    if a.cam_height < 0.03:                                               # keep the lens out of the foliage
        d_ = np.hypot(inst["x"] - CAM[0], inst["y"] - CAM[1]) * 1000
        inst["h"] = np.where(d_ < inst["r"] + 6, 0, inst["h"])
    return inst, grids

def tree_colours(tp, h1, h2):
    base = {T_CON: np.array([0.030, 0.050, 0.030]), T_DEC: FOLIAGE * 1.1, T_POP: FOLIAGE * np.array([1.2, 1.25, 1.0])}
    c = np.stack([base[T_CON]] * len(tp))
    c[tp == T_DEC] = base[T_DEC]; c[tp == T_POP] = base[T_POP]
    c = c * (0.75 + 0.5 * h1)[:, None]
    if a.season == "autumn":
        aut = np.array([[0.20, 0.09, 0.025], [0.24, 0.17, 0.03], [0.15, 0.05, 0.02]])[(h2 * 3).astype(int)]
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

# screen bounding boxes
def _project(P):
    v = P - CAM; zc = v @ fwd
    xs = (v @ right) / np.maximum(zc, 1e-6) * fl + W / 2 - 0.5
    ys = -(v @ up) / np.maximum(zc, 1e-6) * fl + H / 2 + H * 0.02 - 0.5
    return xs, ys, zc
if NI:
    _rb = np.where(INST["tp"] >= T_HOUSE, INST["r"] + 0.9, INST["r"] * 1.3) / 1000
    _corn = [(sx_, sy_, sz_) for sx_ in (-1, 1) for sy_ in (-1, 1) for sz_ in (0, 1)]
    XS, YS, ZC = [], [], []
    for sx_, sy_, sz_ in _corn:
        P = np.stack([INST["x"] + sx_ * _rb, INST["y"] + sy_ * _rb, INST["z"] + (INST["h"] * 1.03 / 1000 if sz_ else -0.002)], 1)
        xs_, ys_, zc_ = _project(P); XS.append(xs_); YS.append(ys_); ZC.append(zc_)
    XS, YS, ZC = np.array(XS), np.array(YS), np.array(ZC)
    BB = np.stack([np.floor(XS.min(0)), np.ceil(XS.max(0)) + 1, np.floor(YS.min(0)), np.ceil(YS.max(0)) + 1], 1).astype(np.int64)
    VIS = (ZC.min(0) > 0.004) & (BB[:, 0] < W) & (BB[:, 1] > 0) & (BB[:, 2] < H) & (BB[:, 3] > 0) & (INST["h"] > 0)
    BB[:, 0] = np.clip(BB[:, 0], 0, W); BB[:, 1] = np.clip(BB[:, 1], 0, W); BB[:, 2] = np.clip(BB[:, 2], 0, H); BB[:, 3] = np.clip(BB[:, 3], 0, H)
    DMIN = np.hypot(INST["x"] - CAM[0], INST["y"] - CAM[1]) - _rb
    SPX = INST["r"] / np.maximum(np.hypot(INST["x"] - CAM[0], INST["y"] - CAM[1]) * 1000, 1) * fl     # radius in px
    print(f"near field: {int(VIS.sum())} instances in view, {int(((BB[:, 1] - BB[:, 0]) * (BB[:, 3] - BB[:, 2]))[VIS].sum())} bbox px", flush=True)

if NI and WATER:                                                      # bounding boxes of the mirror images
    _rb2 = np.where(INST["tp"] >= T_HOUSE, INST["r"] + 0.9, INST["r"] * 1.3) / 1000
    XS, YS = [], []
    for sx_, sy_, sz_ in _corn:
        x_, y_ = INST["x"] + sx_ * _rb2, INST["y"] + sy_ * _rb2
        zw = WL - ((x_ - CAM[0]) ** 2 + (y_ - CAM[1]) ** 2) / (2 * RE)
        zt = INST["z"] + (INST["h"] * 1.03 / 1000 if sz_ else -0.002)
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
    return map_coordinates(NTEX, [x, y, z], order=1, mode="grid-wrap")

def tree_field(q, tp, Ht, R, seed, noisy):
    """signed-ish distance (m): < 0 inside crown or trunk; also returns trunk flag"""
    rr = np.hypot(q[:, 0], q[:, 1]); z = q[:, 2]
    if noisy is not None and noisy.any():
        so = seed * 97.0
        nq = (tnoise(q[:, 0] * 0.33 + so, q[:, 1] * 0.33 - so, z * 0.4 + so * 0.5) - 0.5) * 0.75 \
           + (tnoise(q[:, 0] * 0.95 - so, q[:, 1] * 0.95 + so, z * 0.95) - 0.5) * 0.45 \
           + (tnoise(q[:, 0] * 2.7 + 11, q[:, 1] * 2.7 - 5, z * 2.7 + so) - 0.5) * 0.22
        bump = np.where(noisy, nq, 0.0)
    else:
        bump = np.zeros_like(rr)
    f = np.full(len(rr), 1e3)
    m = tp == T_CON
    if m.any():
        h0 = Ht[m] * 0.12; u = np.clip((z[m] - h0) / (Ht[m] - h0), 0, 1)
        tier = 1 - np.mod(u * 6.5 + seed[m] * 3, 1.0)
        prof = R[m] * (1 - u) ** 0.9 * (0.66 + 0.34 * tier) * (1 + bump[m] * 1.1)
        fc = np.where((z[m] >= h0) & (z[m] <= Ht[m]), rr[m] - prof, np.maximum(h0 - z[m], z[m] - Ht[m]) + 0.2)
        f[m] = fc
    m = tp == T_DEC
    if m.any():
        Rv = np.minimum(R[m] * 0.85, (Ht[m] - 2.0) / 2); zc = Ht[m] - Rv
        zz = (z[m] - zc) / Rv; zz = np.where(zz < 0, zz * 1.25, zz)          # flatter underside
        d = np.sqrt((rr[m] / R[m]) ** 2 + zz ** 2)
        f[m] = (d - 1 - bump[m] * 1.1) * np.minimum(R[m], Rv) * 0.8
    m = tp == T_POP
    if m.any():
        Rv = Ht[m] * 0.45; zc = Ht[m] - Rv; v = (z[m] - zc) / Rv
        prof = R[m] * np.sqrt(np.clip(1 - v * v, 0, 1)) * (1 - 0.3 * v) * (1 + bump[m] * 0.8)
        f[m] = np.where(np.abs(v) < 1, rr[m] - prof, np.abs(v) * Rv - Rv + 0.2)
    rt = 0.14 + 0.02 * R
    ztop = np.where(tp == T_CON, Ht * 0.3, Ht * 0.6)
    ft = np.maximum(rr - rt, np.maximum(-1.5 - z, z - ztop))
    return np.minimum(f, ft), ft < f, bump

def isect_trees(O, D, ids, K=14, noisy=None, want_normal=False):
    """first hit t (m, local = world scale) of rays vs trees ids; inf = miss"""
    q0, d = _local(O, D, ids)
    tp, Ht, R, sd = INST["tp"][ids], INST["h"][ids], INST["r"][ids], INST["seed"][ids]
    rb = R * 1.35 + 0.3
    A = d[:, 0] ** 2 + d[:, 1] ** 2; B = 2 * (q0[:, 0] * d[:, 0] + q0[:, 1] * d[:, 1]); C = q0[:, 0] ** 2 + q0[:, 1] ** 2 - rb ** 2
    disc = B * B - 4 * A * C; sq = np.sqrt(np.maximum(disc, 0)); A_ = np.maximum(A, 1e-12)
    t0 = np.where(A > 1e-12, (-B - sq) / (2 * A_), -1e9); t1 = np.where(A > 1e-12, (-B + sq) / (2 * A_), np.where(C < 0, 1e9, -1e9))
    dz = np.where(np.abs(d[:, 2]) < 1e-9, 1e-9, d[:, 2])
    za, zb = (-1.6 - q0[:, 2]) / dz, (Ht * 1.03 - q0[:, 2]) / dz
    ta = np.maximum.reduce([t0, np.minimum(za, zb), np.zeros_like(t0)]); tb = np.minimum(t1, np.maximum(za, zb))
    ok = (disc > 0) & (tb > ta)
    t_hit = np.full(len(ids), np.inf)
    idx = np.where(ok)[0]
    if not len(idx): return t_hit
    ta, tb = ta[idx], tb[idx]; dtmin = np.maximum((tb - ta) / (K + 6), 0.1)
    nz = noisy[idx] if noisy is not None else None
    found = np.zeros(len(idx), bool); alive = np.ones(len(idx), bool)
    tcur = ta.copy(); lo = ta.copy(); hi = tb.copy(); last = np.zeros(len(idx))
    for k in range(K + 12):                        # distance-guided march: big steps far from the crown
        live = np.where(alive)[0]
        if not len(live): break
        j = idx[live]; tt = tcur[live]
        f, _, _ = tree_field(q0[j] + d[j] * tt[:, None], tp[j], Ht[j], R[j], sd[j], nz[live] if nz is not None else None)
        inside = f < 0
        L_in = live[inside]
        hi[L_in] = tt[inside]; lo[L_in] = np.maximum(tt[inside] - last[L_in], ta[L_in]); found[L_in] = True; alive[L_in] = False
        L_out = live[~inside]
        stp = np.maximum(0.5 * f[~inside], dtmin[L_out]); last[L_out] = stp
        tcur[L_out] += stp; alive[L_out[tcur[L_out] > tb[L_out]]] = False
    f_ = np.where(found)[0]
    lo, hi = lo[f_], hi[f_]
    for _ in range(4):
        mid = (lo + hi) / 2; j = idx[f_]
        f, _, _ = tree_field(q0[j] + d[j] * mid[:, None], tp[j], Ht[j], R[j], sd[j], nz[f_] if nz is not None else None)
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
    if not want: return t
    nl /= np.maximum(np.linalg.norm(nl, axis=1, keepdims=True), 1e-9)
    return t, which, kind, nl, q, d

def isect_any(O, D, ids, noisy=None):
    t = np.full(len(ids), np.inf)
    tr = INST["tp"][ids] <= T_POP
    if tr.any(): t[tr] = isect_trees(O[tr] if len(O) > 1 else O, D[tr], ids[tr], noisy=noisy[tr] if noisy is not None else None)
    if (~tr).any(): t[~tr] = isect_houses(O[~tr] if len(O) > 1 else O, D[~tr], ids[~tr])
    return t

# ------------------------------------------------------------------ sun shadows cast by instances
def _cone_or_ellipsoid_hit(P, ids):
    """closed-form shadow test of rays P + s*SUN against simplified tree shapes; bool"""
    q, d = _local(P, np.tile(SUN, (len(P), 1)), ids)
    tp, Ht, R = INST["tp"][ids], INST["h"][ids], INST["r"][ids]
    hitm = np.zeros(len(ids), bool)
    # ellipsoids for deciduous / poplar crowns
    Rv = np.where(tp == T_POP, Ht * 0.45, np.minimum(R * 0.85, (Ht - 2.0) / 2)); zc = Ht - Rv
    Rh = np.where(tp == T_POP, R * 0.95, R * 0.95)
    qx, qy, qz = q[:, 0] / Rh, q[:, 1] / Rh, (q[:, 2] - zc) / Rv
    dx, dy, dz = d[:, 0] / Rh, d[:, 1] / Rh, d[:, 2] / Rv
    A = dx * dx + dy * dy + dz * dz; B = 2 * (qx * dx + qy * dy + qz * dz); C = qx * qx + qy * qy + qz * qz - 1
    disc = B * B - 4 * A * C; t1 = (-B + np.sqrt(np.maximum(disc, 0))) / (2 * A)
    hitm |= (tp != T_CON) & (disc > 0) & (t1 > 0.3)
    # cone for conifers: rr = k (Ht - z), h0 <= z <= Ht
    h0 = Ht * 0.12; k = R * 0.85 / (Ht - h0)
    A = d[:, 0] ** 2 + d[:, 1] ** 2 - (k * d[:, 2]) ** 2
    B = 2 * (q[:, 0] * d[:, 0] + q[:, 1] * d[:, 1] + k * k * (Ht - q[:, 2]) * d[:, 2])
    C = q[:, 0] ** 2 + q[:, 1] ** 2 - (k * (Ht - q[:, 2])) ** 2
    disc = B * B - 4 * A * C; sq = np.sqrt(np.maximum(disc, 0))
    for sgn in (-1, 1):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (-B + sgn * sq) / (2 * A)
        zz = q[:, 2] + t * d[:, 2]
        hitm |= (tp == T_CON) & (disc > 0) & (t > 0.3) & (zz >= h0) & (zz <= Ht)
    return hitm

def inst_shadow(P, n):
    """1 = lit, 0 = in the shadow of a tree or building (terrain shadow is separate)"""
    out = np.ones(len(P))
    if not NI or SUN_EL <= 0.005: return out
    Pm = P + n * 0.0004
    d = np.hypot(Pm[:, 0] - CAM[0], Pm[:, 1] - CAM[1])
    sel = np.where(d < NEAR * 1.02 + 0.2)[0]
    if not len(sel): return out
    Lh = min(HMAX / 1000 / math.tan(SUN_EL), 0.35)                      # horizontal reach of the tallest shadow
    ch = math.cos(SUN_EL); sh_ = np.zeros(len(sel), bool)
    for g in GRIDS:
        c = g["c"]; cr, sr = g["rot"]; ids_g = g["ids"]
        steps = int(math.ceil(Lh / (c / 2.5)))
        seen = np.full(len(sel), -1, np.int64)
        for k in range(steps + 1):
            live = np.where(~sh_)[0]
            if not len(live): break
            s = (k * c / 2.5) / ch
            Q = Pm[sel[live]] + SUN * s
            if k > 2 and (Q[:, 2] - terrain_z(Q[:, 0], Q[:, 1]) > HMAX / 1000).all(): break
            u, v = Q[:, 0] * cr + Q[:, 1] * sr, -Q[:, 0] * sr + Q[:, 1] * cr
            i = np.floor(u / c).astype(np.int64) - g["i0"]; j = np.floor(v / c).astype(np.int64) - g["j0"]
            inb = (i >= 0) & (j >= 0) & (i < ids_g.shape[0]) & (j < ids_g.shape[1])
            iid = np.full(len(live), -1, np.int64); iid[inb] = ids_g[i[inb], j[inb]]
            m = (iid >= 0) & (iid != seen[live])
            if not m.any(): continue
            seen[live[m]] = iid[m]
            L_ = live[m]; ii = iid[m]; Pq = Pm[sel[L_]]
            tr = INST["tp"][ii] <= T_POP
            hit = np.zeros(len(ii), bool)
            if tr.any(): hit[tr] = _cone_or_ellipsoid_hit(Pq[tr], ii[tr])
            if (~tr).any(): hit[~tr] = np.isfinite(isect_houses(Pq[~tr], np.tile(SUN, ((~tr).sum(), 1)), ii[~tr]))
            sh_[L_[hit]] = True
    out[sel[sh_]] = 0.0
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
    order = np.arange(len(sel)); csum = np.cumsum(cnt); chunk = 300000
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
        keep = DMIN[iid] < tmax[pix]
        pix, iid, col, row = pix[keep], iid[keep], col[keep], row[keep]
        if not len(pix): continue
        noisy = (SPX[iid] > 2.5) & (INST["tp"][iid] <= T_POP)
        for sI, (ox_, oy_) in enumerate(SUB):
            px_ = (col + 0.5 + ox_ - W / 2) / fl; py_ = -(row + 0.5 + oy_ - H / 2 - H * 0.02) / fl
            D = fwd[None] + px_[:, None] * right[None] + py_[:, None] * up[None]; D /= np.linalg.norm(D, axis=1, keepdims=True)
            t = isect_any(CAM[None], D, iid, noisy) / 1000
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
        noisy = (SPX[iid] > 2.5) & (INST["tp"][iid] <= T_POP)
        for sI, (ox_, oy_) in enumerate(SUB2):
            px_ = (col + 0.5 + ox_ - W / 2) / fl; py_ = -(row + 0.5 + oy_ - H / 2 - H * 0.02) / fl
            D = fwd[None] + px_[:, None] * right[None] + py_[:, None] * up[None]; D /= np.linalg.norm(D, axis=1, keepdims=True)
            tp_ = plane_t(np.repeat(CAM[None], len(D), 0), D, WL)
            ok0 = np.isfinite(tp_) & (tp_ > 0)
            P = CAM + D * np.where(ok0, tp_, 0)[:, None]; Dm = D * np.array([1, 1, -1.0])
            s_ = np.full(len(D), np.inf)
            s_[ok0] = isect_any(P[ok0] + np.array([0, 0, 0.0003]), Dm[ok0], iid[ok0], noisy[ok0]) / 1000
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
        c_ = shade_instances(P + Dm * ss[:, None], Dm, ii, ss, O=P)
        Li, Ti = atmosphere(Dm, ss, O=P)
        colsum[m] += c_ * Ti + Li
    nh = np.maximum((best_i[hw] >= 0).sum(1), 1)
    return hp, cov, colsum / nh[:, None]

def _normal_trees(P, ids):
    q = (P - np.stack([INST["x"][ids], INST["y"][ids], INST["z"][ids]], 1)) * 1000
    tp, Ht, R, sd = INST["tp"][ids], INST["h"][ids], INST["r"][ids], INST["seed"][ids]
    noisy = SPX[ids] > 2.5; e = 0.25; g = np.zeros_like(q)
    for k in range(3):
        dq = np.zeros(3); dq[k] = e
        fp, _, _ = tree_field(q + dq, tp, Ht, R, sd, noisy); fm, _, _ = tree_field(q - dq, tp, Ht, R, sd, noisy)
        g[:, k] = fp - fm
    n = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)
    f0, trunk, bump = tree_field(q, tp, Ht, R, sd, noisy)
    return n, q, trunk, bump

_eld = math.degrees(SUN_EL)
LR_WIN = float(np.clip((1.5 - _eld) / 5.0, 0, 1))                         # share of windows lit: 0 in daylight .. 1 at -3.5 deg
LR_ST = float(np.clip((1.0 - _eld) / 2.0, 0, 1))                          # street lamps switch on around sunset
NIGHT_WIN = LR_WIN > 0
def shade_instances(P, D, ids, dist, O=None):
    col = np.zeros((len(ids), 3))
    z = P[:, 2] + ((P[:, 0] - CAM[0]) ** 2 + (P[:, 1] - CAM[1]) ** 2) / (2 * RE)
    sunc = sun_T(np.maximum(z, 0)) * SUN_I * 0.17
    tr = INST["tp"][ids] <= T_POP
    if tr.any():
        k = np.where(tr)[0]; ii = ids[k]
        n, q, trunk, bump = _normal_trees(P[k], ii)
        radial = np.stack([q[:, 0], q[:, 1], np.zeros(len(k)) + 0.35 * INST["r"][ii]], 1)
        radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-9)
        n = n * 0.6 + radial * 0.4; n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
        clump = np.where(SPX[ii] > 2.5, np.clip((bump + 0.25) / 0.5, 0, 1), 0.55)   # hollows between clumps are darker
        sh = soft_shadow(P[k], n, steps=40) * inst_shadow(P[k], n) * (0.35 + 0.65 * clump)
        alb = INST["col"][ii, :3] * (0.75 + 0.5 * tnoise(q[:, 0] * 2.3, q[:, 1] * 2.3, q[:, 2] * 2.3))[:, None]
        alb = np.where(trunk[:, None], np.array([0.06, 0.045, 0.035]), alb)
        hgt = np.clip(q[:, 2] / INST["h"][ii], 0, 1)
        ao = (0.3 + 0.7 * hgt ** 0.7) * (0.45 + 0.55 * clump)
        leaf = tnoise(q[:, 0] * 4.1 + 3, q[:, 1] * 4.1, q[:, 2] * 4.1 - 7) - 0.5   # leaf-scale speckle
        alb = alb * (1 + 0.6 * leaf)[:, None]
        ndl = n @ SUN
        wrap = np.clip((ndl + 0.3) / 1.3, 0, 1) ** 1.5
        back = np.clip(-(D[k] @ SUN), 0, 1)                                 # leaves glow when backlit
        trans = (0.35 * np.clip(-ndl, 0, 1) * back ** 2)[:, None] * np.array([0.9, 1.1, 0.5])
        col[k] = alb * (sunc[k] * (wrap * sh)[:, None] + SKYC[None] * (ao * (0.55 + 0.45 * n[:, 2]))[:, None] + BOUNCE * ao[:, None]) \
               + alb * sunc[k] * trans * sh[:, None]
    if (~tr).any():
        k = np.where(~tr)[0]; ii = ids[k]
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
        sh = soft_shadow(P[k], n, steps=40) * inst_shadow(P[k], n)
        ao = np.clip(0.55 + ql[:, 2] / 6, 0.55, 1.0)
        ndl = np.clip(n @ SUN, 0, 1)
        col[k] = alb * (sunc[k] * (ndl * sh)[:, None] + SKYC[None] * (ao * (0.6 + 0.4 * n[:, 2]))[:, None] + BOUNCE * ao[:, None])
        hv = SUN - D[k]; hv /= np.linalg.norm(hv, axis=1, keepdims=True)
        spec = np.clip((n * hv).sum(1), 0, 1) ** 40 * np.where(win, 0.6, np.where(roof, 0.12, 0.0))
        col[k] += (spec * sh)[:, None] * sunc[k]
        if NIGHT_WIN or CITY_ON:
            fi = np.floor(along / 3.1 + INST["seed"][ii] * 7); fj = np.floor(ql[:, 2] / 3.0)
            occ = 0.06 + 0.45 * hash2(ii, 0, 50) ** 1.5                         # some buildings busy, some asleep
            lit = win & (hash2(fi.astype(np.int64) + ii * 131, fj.astype(np.int64), 51) < occ * LR_WIN)
            hc = hash2(fi.astype(np.int64), ii, 52)
            pal = np.array([[1.0, 0.60, 0.28], [1.0, 0.80, 0.55], [0.80, 0.88, 1.0], [0.45, 0.55, 1.0]])
            wc = pal[np.searchsorted([0.55, 0.8, 0.93], hc)]
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
    def _street_lights():
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
        b = np.where(INST["tp"] >= T_HOUSE)[0]
        n_ = np.where(INST["tp"][b] == T_BLOCK, 6, 2)
        bi = np.repeat(b, n_); k = np.arange(len(bi)) - np.repeat(np.cumsum(n_) - n_, n_)
        h1, h2, h3 = hash2(bi, k, 71), hash2(bi, k, 72), hash2(bi, k, 73)
        lit = h3 < (0.06 + 0.45 * hash2(bi, 0, 50) ** 1.5) * LR_WIN
        yaw = INST["yaw"][bi]; A_, B_ = INST["a"][bi], INST["b"][bi]
        lx = (h1 - 0.5) * 2 * A_; ly = np.where(h2 < 0.5, -B_ - 0.3, B_ + 0.3); lz = 1.5 + (h2 * 2 % 1) * (INST["hw"][bi] - 2)
        c, s = np.cos(yaw), np.sin(yaw)
        x = INST["x"][bi] + (lx * c - ly * s) / 1000; y = INST["y"][bi] + (lx * s + ly * c) / 1000
        z = INST["z"][bi] + lz / 1000
        warm = np.where((hash2(bi, k, 74) < 0.7)[:, None], np.array([1.0, 0.62, 0.3]), np.array([0.8, 0.88, 1.0]))
        far = np.hypot(x - CAM[0], y - CAM[1]) > 1.2                        # near windows are drawn on the facades
        m = lit & far
        return np.stack([x, y, z], 1)[m], warm[m] * 0.12
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
        P1, C1 = _street_lights(); P2, C2 = _window_lights(); P3, C3 = _near_lamps()
        C1 = C1 * LR_ST; C3 = C3 * LR_ST
        P = np.concatenate([P1, P2, P3]); C = np.concatenate([C1, C2, C3])
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
        print(f"city: {len(P)} lights in view ({len(P1)} street, {len(P2)} window)", flush=True)
    LIGHT_K = 6e5
    # ground illumination and sky glow from a light-density map
    _gx = np.arange(CAM[0] - a.city_km - 1, CAM[0] + a.city_km + 1, CITY_GRID)
    _gy = np.arange(CAM[1] - a.city_km - 1, CAM[1] + a.city_km + 1, CITY_GRID)
    _GX, _GY = np.meshgrid(_gx, _gy)
    _dens = city_field(_GX.ravel(), _GY.ravel()).reshape(_GX.shape)
    _dens *= (_water(_GX.ravel(), _GY.ravel()).reshape(_GX.shape) < 0.3)
    _DEN_S = gaussian_filter(_dens, 0.8); _DEN_L = gaussian_filter(_dens, 6)
    def _dmap(Mp, x, y):
        return map_coordinates(Mp, [(y - _gy[0]) / CITY_GRID, (x - _gx[0]) / CITY_GRID], order=1, mode="constant", cval=0.0)
    def city_ground_light(x, y):
        f = _dmap(_DEN_S, x, y) * (0.5 + 0.5 * fbm2(x * 30, y * 30, 2))
        return (f * 0.022 * LR_ST)[:, None] * np.array([1.0, 0.62, 0.30])
    def _glow(sx, sy, alt):
        shp = sx.shape
        g = (_dmap(_DEN_S, sx.ravel(), sy.ravel()) * 0.6 + _dmap(_DEN_L, sx.ravel(), sy.ravel()) * 0.4).reshape(shp)
        hrel = np.maximum(alt - BASE, 0)
        return (g * np.exp(-hrel / 1.0))[..., None] * np.array([1.0, 0.62, 0.30]) * 0.30 * a.city * LR_ST
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
