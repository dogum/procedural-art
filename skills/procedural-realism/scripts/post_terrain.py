#!/usr/bin/env python3
"""post_terrain.py — stitch bands, then (only if the render produced them) water streak blur, depth of
   of field, city lights with bokeh; then bloom, ACES filmic tonemap, vignette, grain.
   python post_terrain.py --bands pbr_work/bands_1500x500 --out render.png [--exposure 0.8|auto]
   Refuses (exit 1) when rows are missing or the last pbr_terrain.py run on the folder was refused,
   so a chained command can't save an incomplete or stale picture."""
import argparse, glob, json, math, os, re, sys, numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import fftconvolve
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument("--bands", default=None); ap.add_argument("--out", required=True)
ap.add_argument("--hdr-in", default=None, help="tonemap this composited linear image (.npy) instead of reading bands")
ap.add_argument("--seed", type=int, default=3, help="grain seed (vary per video frame)")
ap.add_argument("--exposure", default=None, help="linear exposure, or 'auto' (log-average key), or auto*1.3 to bias "
                "(default 0.8; auto*0.25 for --time night renders)")
ap.add_argument("--partial", action="store_true", help="accept bands that cover only part of the frame (a --rows crop)")
ap.add_argument("--grain", type=float, default=0.012)
ap.add_argument("--save-hdr", default=None, help="also save the composited linear image (.npy), e.g. for timelapse exposure")
ap.add_argument("--no-lights", action="store_true"); ap.add_argument("--no-dof", action="store_true")
ap.add_argument("--light-gain", type=float, default=1.0, help="scale city light brightness")
ap.add_argument("--bokeh-rim", type=float, default=0.25, help="brighter bokeh edge (0 = flat discs)")
ap.add_argument("--aperture", type=float, default=None, help="override the render's lens aperture, mm (0 = pinhole); DOF and bokeh are post effects")
ap.add_argument("--focus", type=float, default=None, help="override the focus distance, km")
ap.add_argument("--wb", default="auto", help="white balance in kelvin ('auto': 4800 K at blue hour, the setting blue-hour "
                "photographers use, else the sun as white; 'sun' = never shift)")
ap.add_argument("--sat", type=float, default=None, help="saturation (default 1; 1.15 at blue hour, like a camera's standard picture style)")
a = ap.parse_args()

if a.hdr_in:
    img = np.load(a.hdr_in); files = []; a.bands = os.path.dirname(a.hdr_in) or "."; a.no_lights = a.no_dof = True
else:
    if not a.bands: sys.exit("give --bands <folder> (or --hdr-in FILE.npy)")
    files = sorted(f for f in glob.glob(os.path.join(a.bands, "*.npy")) if re.fullmatch(r"\d{5}\.npy", os.path.basename(f)))
    if not files: sys.exit(f"no bands in {a.bands} (expected files like 00000.npy from pbr_terrain.py)")
    rf = os.path.join(a.bands, "refused.json")
    if os.path.exists(rf):
        d_ = json.load(open(rf)).get("diff", {})
        sys.exit(f"the last pbr_terrain.py run on {a.bands} was refused because its settings differ {d_}; these bands "
                 "are from the earlier settings. Render the new settings with --tag NAME and post that folder, "
                 f"or delete {rf} to post the old bands anyway.")
    starts = [int(os.path.basename(f)[:5]) for f in files]
    rows = [np.load(f, mmap_mode="r").shape[0] for f in files]
    Hc = json.load(open(os.path.join(a.bands, "camera.json")))["H"] if os.path.exists(os.path.join(a.bands, "camera.json")) else None
    gaps = [(starts[i - 1] + rows[i - 1], starts[i]) for i in range(1, len(files)) if starts[i] != starts[i - 1] + rows[i - 1]]
    missing = ([(0, starts[0])] if starts[0] > 0 else []) + gaps + \
              ([(starts[-1] + rows[-1], Hc)] if Hc and starts[-1] + rows[-1] < Hc else [])
    if gaps and any(e < s_ for s_, e in gaps):
        sys.exit(f"{a.bands}: band files overlap ({gaps}); delete the folder and render it again")
    if missing and not (a.partial and not gaps):
        sys.exit(f"{a.bands}: rows " + ", ".join(f"{s_}-{e}" for s_, e in missing) + " are missing. Run pbr_terrain.py "
                 "with the same arguments again (without --rows it fills every gap), then post."
                 + ("" if gaps else " Use --partial to post a --rows crop on purpose."))
    img = np.concatenate([np.load(f) for f in files], 0)
H, W, _ = img.shape
aux = os.path.join(a.bands, "aux")
def load_aux(kind):
    if not files: return None
    fs = [os.path.join(aux, os.path.basename(f)[:-4] + f"_{kind}.npy") for f in files]
    return np.concatenate([np.load(f) for f in fs], 0) if all(os.path.exists(f) for f in fs) else None
cam = json.load(open(os.path.join(a.bands, "camera.json"))) if files and os.path.exists(os.path.join(a.bands, "camera.json")) else None
if cam is not None and a.aperture is not None: cam["aperture_mm"] = a.aperture
if cam is not None and a.focus is not None: cam["focus_km"] = a.focus

# ---- water: rough reflections smear vertically; blur water pixels by the per-pixel spread
wb = load_aux("wblur")
if wb is not None and (wb > 0).any():
    m = (wb > 0).astype(np.float32); out = img.copy()
    levels = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    stack = [img]
    for s in levels[1:]:
        num = gaussian_filter1d(img * m[..., None], s, axis=0); num = gaussian_filter1d(num, s * 0.12, axis=1)
        den = gaussian_filter1d(m, s, axis=0); den = gaussian_filter1d(den, s * 0.12, axis=1)
        stack.append(num / np.maximum(den, 1e-6)[..., None])
    r = np.clip(wb * 0.5, 0, levels[-1])          # smear is +-2 sigma; gaussian sigma ~ half of it
    idx = np.clip(np.searchsorted(levels, r) - 1, 0, len(levels) - 2)
    lo = np.take(levels, idx); hi = np.take(levels, idx + 1); f = ((r - lo) / (hi - lo))[..., None]
    for k in range(len(levels) - 1):
        sel = (idx == k) & (m > 0)
        out[sel] = stack[k][sel] * (1 - f[sel]) + stack[k + 1][sel] * f[sel]
    img = out

# ---- depth of field (thin lens), layered gather by circle of confusion
depth = load_aux("depth")
def coc_px(d_km):
    A = cam["aperture_mm"] / 1000.0; fo = cam["focus_km"] * 1000.0
    return 0.5 * A * cam["fl"] * np.abs(1.0 / np.maximum(d_km * 1000.0, 1.0) - 1.0 / fo) * (W / cam["W"])
def disc_kernel(r, rim=0.0):
    n = int(math.ceil(r)) + 1; yy, xx = np.mgrid[-n:n + 1, -n:n + 1]; rr = np.hypot(xx, yy)
    k = np.clip(r + 0.5 - rr, 0, 1) * (1 + rim * np.clip(rr / max(r, 1e-6), 0, 1) ** 4)
    return k / k.sum()
if cam and cam.get("aperture_mm", 0) > 0 and depth is not None and not a.no_dof:
    c = coc_px(depth); bins = [0, 0.75, 1.5, 3, 5, 8, 12, 18, 26]
    out = img.copy()
    for lo_, hi_ in zip(bins[1:-1], bins[2:]):
        sel = (c >= lo_) & (c < hi_)
        if not sel.any(): continue
        r_ = (lo_ + hi_) / 2; k = disc_kernel(r_)
        bl = np.stack([fftconvolve(img[..., ch], k, mode="same") for ch in range(3)], -1)
        out[sel] = bl[sel]
    sel = c >= bins[-1]
    if sel.any():
        k = disc_kernel(bins[-1] + 4); bl = np.stack([fftconvolve(img[..., ch], k, mode="same") for ch in range(3)], -1); out[sel] = bl[sel]
    img = out

night = bool(cam and cam.get("night"))
# blue hour (physical sky, sun 1 deg above to 3 deg below the horizon and lower, not --time night): the exposure keys on
# the sky and the far mountains instead of the whole frame, so a city or a dark foreground doesn't set it
tw = 0.0
if cam and cam.get("sky") == "physical" and cam.get("sun_el") is not None and not night:
    tw = float(np.clip((1.0 - cam["sun_el"]) / 3.0, 0, 1))
def _log_key(v):
    return math.exp(float(np.mean(np.log(v @ np.array([0.2126, 0.7152, 0.0722]) + 1e-4))))
key_img = _log_key(img)
if tw > 0:
    far = (depth > 30) if depth is not None else (np.arange(H) < H // 3)[:, None] & np.ones((1, W), bool)
    if far.sum() > 0.05 * H * W:
        key_img = key_img ** (1 - tw) * _log_key(img[far]) ** tw
if night:
    # moonlight is dim to the eye: pull the scene (not the city lights added below) toward a cool grey, the scotopic look
    lum = (img @ np.array([0.2126, 0.7152, 0.0722]))[..., None]
    img = lum * np.array([0.78, 0.9, 1.12]) * 0.55 + img * 0.45

# ---- city lights: depth-tested points, aerial glow, bokeh discs
lf = os.path.join(a.bands, "lights.npz")
if os.path.exists(lf) and not a.no_lights:
    L = np.load(lf); sx, sy, d, flux, halo = L["sx"], L["sy"], L["d"], L["flux"] * a.light_gain, L["halo"] * a.light_gain
    if H != cam["H"]:                                   # a partial render (row crop): shift to the first band
        sy = sy - int(os.path.basename(files[0])[:5])
    ix, iy = np.floor(sx).astype(int), np.floor(sy).astype(int)
    ok = (ix >= 0) & (ix < W - 1) & (iy >= 0) & (iy < H - 1)
    if depth is not None:
        db = depth[np.clip(iy, 0, H - 1), np.clip(ix, 0, W - 1)]
        ok &= d <= db * 1.004 + 0.012
    sx, sy, d, flux, halo, ix, iy = sx[ok], sy[ok], d[ok], flux[ok], halo[ok], ix[ok], iy[ok]
    fx, fy = sx - ix, sy - iy
    rads = coc_px(d) if cam.get("aperture_mm", 0) > 0 else np.zeros(len(d))
    bins = [0, 1.0, 1.6, 2.4, 3.5, 5, 7, 10, 14, 20, 28, 40, 60]
    layer_all = np.zeros_like(img)
    for b0, b1 in zip(bins[:-1], bins[1:] + [1e9]):
        sel = (rads >= b0) & (rads < b1)
        if not sel.any(): continue
        layer = np.zeros_like(img)
        for wx, wy, dx, dy in ((1 - fx, 1 - fy, 0, 0), (fx, 1 - fy, 1, 0), (1 - fx, fy, 0, 1), (fx, fy, 1, 1)):
            for ch in range(3):
                np.add.at(layer[..., ch], (iy[sel] + dy, ix[sel] + dx), flux[sel, ch] * (wx * wy)[sel])
        if b0 == 0:
            layer = gaussian_filter(layer, (0.45, 0.45, 0))
        else:
            k = disc_kernel((b0 + min(b1, b0 * 1.5)) / 2, a.bokeh_rim)
            layer = np.stack([fftconvolve(layer[..., ch], k, mode="same") for ch in range(3)], -1)
        layer_all += np.maximum(layer, 0)
    hl = np.zeros_like(img)
    for ch in range(3): np.add.at(hl[..., ch], (iy, ix), halo[:, ch])
    glow = gaussian_filter(hl, (H / 400, H / 400, 0)) * 0.5 + gaussian_filter(hl, (H / 90, H / 90, 0)) * 0.35 + gaussian_filter(hl, (H / 25, H / 25, 0)) * 0.15
    img = img + layer_all + glow

# white balance: a camera set below the sun's colour temperature renders dusk a deeper blue and sodium lamps orange
def _rgb_of(T):
    lam = np.arange(380.0, 731.0, 5.0); g = lambda m, s1, s2: np.exp(-0.5 * ((lam - m) / np.where(lam < m, s1, s2)) ** 2)
    xyz = np.stack([1.056 * g(599.8, 37.9, 31.0) + 0.362 * g(442.0, 16.0, 26.7) - 0.065 * g(501.1, 20.4, 26.2),
                    0.821 * g(568.8, 46.9, 40.5) + 0.286 * g(530.9, 16.3, 31.1), 1.217 * g(437.0, 11.8, 36.0) + 0.681 * g(459.0, 26.0, 13.8)])
    rgb = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]]) @ (xyz @ (1 / (lam ** 5 * (np.exp(1.4388e7 / (lam * T)) - 1))))
    return rgb / (rgb @ np.array([0.2126, 0.7152, 0.0722]))
wbk = None if a.wb == "sun" else (4800.0 if tw > 0 else None) if a.wb == "auto" else float(a.wb)
if wbk is not None:
    m = _rgb_of(5778.0) / _rgb_of(wbk)
    m = 1 + (m / (m @ np.array([0.2126, 0.7152, 0.0722])) - 1) * (tw if a.wb == "auto" else 1.0)
    img = img * m
    print(f"white balance {wbk:.0f} K: x{np.round(m, 3)}")
sat = a.sat if a.sat is not None else 1 + 0.15 * tw
if sat != 1:
    lum = (img @ np.array([0.2126, 0.7152, 0.0722]))[..., None]; img = np.maximum(lum + (img - lum) * sat, 0)
stars = load_aux("stars")                                 # night: point stars, added after bloom
if a.save_hdr: np.save(a.save_hdr, (img if stars is None else img + stars).astype(np.float32))
if a.exposure is None:
    a.exposure = "auto*0.25" if night else "0.8"
if a.exposure.startswith("auto"):
    # blue hour: keyed on the sky and far terrain before the lights were added; otherwise the whole frame
    key = key_img if tw > 0 else _log_key(img)
    ex = 0.30 / key * (float(a.exposure.split("*")[1]) if "*" in a.exposure else 1.0)
    print(f"auto exposure {ex:.3f} (key {key:.4f})")
else:
    ex = float(a.exposure)
img = img * ex
img = img + gaussian_filter(np.clip(img - 1.2, 0, None), (H / 60, H / 60, 0)) * 0.6 + gaussian_filter(np.clip(img - 0.8, 0, None), (H / 15, H / 15, 0)) * 0.15
if stars is not None:
    # display-referred point stars: brightness from the star catalogue value (<= 0.02) and the air/cloud in front,
    # independent of the scene exposure, so they neither vanish nor bloom into discs
    img = img + np.clip(stars / 0.02, 0, 1) * 1.1
img = np.clip((img * (2.51 * img + 0.03)) / (img * (2.43 * img + 0.59) + 0.14), 0, 1)
yy, xx = np.mgrid[0:H, 0:W]
img *= (1 - 0.22 * (((xx - W / 2) / (W * 0.7)) ** 2 + ((yy - H / 2) / (H * 0.9)) ** 2))[..., None]
img = img ** (1 / 2.2); img += np.random.default_rng(a.seed).normal(0, a.grain, img.shape) * (1 - img)
im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
im.save(a.out, quality=93) if a.out.lower().endswith((".jpg", ".jpeg")) else im.save(a.out, optimize=True)
print(a.out, im.size)
