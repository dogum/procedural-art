"""post_pathtrace.py - finish a path-tracer accumulator into an image.

Edge-aware a-trous denoise (albedo-demodulated, guided by first-hit normal / depth / albedo; glass and
metal pixels are left alone), bloom, ACES tone map, vignette, gamma, a little grain.
Defaults come from the scene's "post" block (stored in the accumulator); flags override them.

usage: python post_pathtrace.py --tag draft --out draft.png [--exposure 1.2] [--denoise 5] [--jpg]
       (legacy: python post_pathtrace.py <tag> [exposure] [denoise_iters] [out.png])
"""
import os, sys, json, argparse
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

def parse(argv):
    if argv and not argv[0].startswith("-"):                    # legacy positional form
        new = ["--tag", argv[0]]
        if len(argv) > 1: new += ["--exposure", argv[1]]
        if len(argv) > 2: new += ["--denoise", argv[2]]
        if len(argv) > 3: new += ["--out", argv[3]]
        argv = new
    ap = argparse.ArgumentParser(description="denoise + tone map a pathtracer.py accumulator")
    ap.add_argument("--tag", required=True, help="accumulator name used with pathtracer.py --tag")
    ap.add_argument("--work", default=os.environ.get("PT_WORK", "./pt_work"), help="accumulator folder (env PT_WORK)")
    ap.add_argument("--out", default=None, help="output .png (default <tag>.png)")
    ap.add_argument("--exposure", type=float, default=None, help="linear exposure multiplier")
    ap.add_argument("--denoise", type=int, default=None, help="a-trous iterations (0 = off)")
    ap.add_argument("--vignette", type=float, default=None, help="corner darkening strength")
    ap.add_argument("--bloom", type=float, default=None, help="glow strength around values above 1")
    ap.add_argument("--grain", type=float, default=None, help="film grain standard deviation")
    ap.add_argument("--saturation", type=float, default=None, help="colour saturation after tone mapping")
    ap.add_argument("--sigma-l", type=float, default=None, help="denoise edge-stop in units of the pixel noise (higher = smoother)")
    ap.add_argument("--no-variance", action="store_true", help="ignore the variance map (old fixed luminance edge-stop)")
    ap.add_argument("--jpg", action="store_true", help="also write a .jpg next to the png (quality 92)")
    return ap.parse_args(argv)

def load(path):
    z = np.load(path)
    if "sum" in z.files:
        cnt = z["cnt"]; img = z["sum"] / np.maximum(cnt, 1)[..., None]
        Y = img @ np.array([0.2126, 0.7152, 0.0722]); n = np.maximum(cnt, 1)
        var_mean = np.maximum(z["sy2"] / n - Y * Y, 0) / np.maximum(n - 1, 1)     # variance of the pixel mean
        post = json.loads(str(z["scene"])).get("post", {}) if "scene" in z.files else {}
    else:                                                       # accumulator from the old script
        cnt = None; img = z["acc"] / int(z["n"]); post = {}; var_mean = None
    return img, z["alb"], z["nrm"], z["dep"], z["spc"], cnt, post, var_mean

def atrous(img, alb, nrm, dep, iters, var=None, sigma_l=4.0):
    """edge-aware a-trous wavelet filter on albedo-demodulated radiance.
    With a per-pixel variance map (new accumulators) the luminance edge-stop is scaled by the local
    noise level, as in SVGF: shadow and caustic edges that are above the noise survive, and the filter
    fades out by itself as samples accumulate. Without it, a fixed relative threshold is used."""
    H, W, _ = img.shape
    a = np.maximum(alb, 0.02); c = img / a                      # demodulate texture
    k = np.array([1 / 16, 1 / 4, 3 / 8, 1 / 4, 1 / 16])
    la = np.maximum(a @ np.array([0.2126, 0.7152, 0.0722]), 0.02)
    v = None if var is None else gaussian_filter(var, 0.8) / la ** 2          # variance in demodulated units
    lum_sig = 4.0
    for it in range(iters):
        step = 2 ** it; acc = np.zeros_like(c); wsum = np.zeros((H, W, 1))
        vacc = None if v is None else np.zeros((H, W)); w2sum = None if v is None else np.zeros((H, W))
        lc = c @ np.array([0.2126, 0.7152, 0.0722])
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                sh = lambda x: np.roll(np.roll(x, dy * step, 0), dx * step, 1)
                cn, nn, dn, an = sh(c), sh(nrm), sh(dep), sh(alb)
                wn = np.clip(np.sum(nrm * nn, -1), 0, 1) ** 64
                wd = np.exp(-np.abs(dep - dn) / (0.02 * dep * step + 1e-3))
                wa = np.exp(-np.sum((alb - an) ** 2, -1) / 0.004)
                ln = cn @ np.array([0.2126, 0.7152, 0.0722])
                if v is None: wl = np.exp(-np.abs(lc - ln) / (lum_sig * (0.3 + lc)))
                else: wl = np.exp(-np.abs(lc - ln) / (sigma_l * np.sqrt(v) + 1e-4 + 0.02 * lc))
                w = k[dy + 2] * k[dx + 2] * wn * wd * wa * wl
                acc += cn * w[..., None]; wsum += w[..., None]
                if v is not None: vacc += sh(v) * w * w; w2sum += w
        c = acc / np.maximum(wsum, 1e-8); lum_sig *= 0.6
        if v is not None: v = vacc / np.maximum(w2sum, 1e-8) ** 2
    return c * a

def finish(img, alb, nrm, dep, spc, exposure=1.0, denoise=5, vignette=0.35, bloom=0.5, grain=0.008, saturation=1.0,
           var=None, sigma_l=4.0):
    H, W, _ = img.shape
    if denoise:
        den = atrous(img, alb, nrm, dep, denoise, var, sigma_l)
        s = gaussian_filter(spc.astype(float), 1.0)[..., None]
        raw = gaussian_filter(img, (0.5, 0.5, 0))
        img = den * (1 - s) + raw * s                           # never smear reflections/refractions
    img = img * exposure
    img = img + gaussian_filter(np.clip(img - 1.0, 0, None), (H / 80, H / 80, 0)) * bloom
    aces = lambda x: np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0, 1)
    img = aces(img)
    if saturation != 1.0:
        g = img.mean(-1, keepdims=True); img = np.clip(g + (img - g) * saturation, 0, 1)
    yy, xx = np.mgrid[0:H, 0:W]
    img *= (1 - vignette * (((xx - W * 0.52) / (W * 0.62)) ** 2 + ((yy - H * 0.55) / (H * 0.95)) ** 2))[..., None].clip(0.3, 1)
    img = np.clip(img, 0, 1) ** (1 / 2.2)
    img += np.random.default_rng(1).normal(0, grain, img.shape)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)

def main(argv=None):
    a = parse(sys.argv[1:] if argv is None else argv)
    img, alb, nrm, dep, spc, cnt, post, var = load(os.path.join(a.work, f"acc_{a.tag}.npz"))
    pick = lambda flag, key, dflt: flag if flag is not None else post.get(key, dflt)
    out8 = finish(img, alb, nrm, dep, spc, exposure=pick(a.exposure, "exposure", 1.0), denoise=pick(a.denoise, "denoise", 5),
                  vignette=pick(a.vignette, "vignette", 0.35), bloom=pick(a.bloom, "bloom", 0.5),
                  grain=pick(a.grain, "grain", 0.008), saturation=pick(a.saturation, "saturation", 1.0),
                  var=None if a.no_variance else var, sigma_l=pick(a.sigma_l, "sigma_l", 4.0))
    out = a.out or f"{a.tag}.png"
    im = Image.fromarray(out8); im.save(out); print(out)
    if a.jpg:
        j = os.path.splitext(out)[0] + ".jpg"; im.save(j, quality=92); print(j)
    if cnt is not None:
        print(f"samples per pixel: mean {cnt.mean():.1f}, min {cnt.min()}, max {cnt.max()}")

if __name__ == "__main__":
    main()
