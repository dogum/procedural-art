#!/usr/bin/env python3
"""
render.py — render any mountain in one of six print styles.

Examples
  # Ararat from Yerevan, X/Twitter header, engraving style
  python render.py --peak 39.7019,44.2986 --from 40.1792,44.4991 --style survey \
      --label "ՄԱՍԻՍ|5137 m" --label "ՍԻՍ|3896 m@39.6517,44.4011" --out ararat.png

  # Fuji from Lake Kawaguchi, 16:9 wallpaper, centred, woodcut
  python render.py --peak 35.3606,138.7274 --from 35.5100,138.7550 --style woodcut \
      --size 3840x2160 --align center --title 富士山 --out fuji.png

  # every style + a contact sheet
  python render.py --peak ... --from ... --style all --out outdir/

Labels: "Name|subtitle" for the main peak, "Name|subtitle@lat,lon" for others.
"""
import argparse, os, sys, json
import numpy as np
sys.dont_write_bytecode = True                  # the skill folder may be read-only; keep it clean
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.ndimage import gaussian_filter1d, shift
from terrain_art import *

STYLES = ["survey", "topo", "nocturne", "stipple", "riso", "woodcut"]
DEFAULTS = {
    "survey":   dict(paper="#f1e7d0", ink="#3b4549", accent="#ee8f54", light=(-0.75, -0.25, 0.6)),
    "topo":     dict(paper="#f0e6cf", ink="#47545c", accent="#9ea3a3", light=(-0.75, -0.25, 0.6)),
    "nocturne": dict(paper="#0b1020", ink="#cdd9f2", accent="#f9edc7", light=(0.7, -0.1, 0.55)),
    "stipple":  dict(paper="#f6f3ec", ink="#1f1f21", accent="#c73829", light=(-0.6, -0.35, 0.5)),
    "riso":     dict(paper="#f7f3ec", ink="#0073c2", accent="#ff4785", light=(-0.7, -0.2, 0.55)),
    "woodcut":  dict(paper="#efe7d4", ink="#1f2126", accent="#cc3b24", light=(-0.7, -0.3, 0.5)),
}
CAM = {"topo": dict(camh=24, fy=3000, hmul=2.4, hy=-110)}


def scene_for(style, t, cfg):
    cam = dict(camh=3.4, fx=4500, fy=10200, cx=2150, hy=290, hmul=1.0)
    cam.update(CAM.get(style, {}))
    z = cfg.get("zoom", 1.0)
    cam["fx"] *= z; cam["fy"] *= z * cfg.get("relief", 1.0)
    cam["cx"] += cfg.get("pan_x", 0); cam["hy"] += cfg.get("pan_y") or 0
    if cfg.get("camh") is not None: cam["camh"] = cfg["camh"]
    elif style not in CAM and getattr(t, "shape", {}).get("camh"): cam["camh"] = t.shape["camh"]
    elif style == "topo" and getattr(t, "shape", {}).get("horn", 0) > 0.05:
        cam["camh"] -= 10 * t.shape["horn"]          # a lower aerial camera so the horn stands up out of its contours
    sc = Scene(t, cfg["W"], cfg["H"], align=cfg.get("align", "right"), **cam)
    if cfg.get("pan_y") is None and cfg.get("fit", True):
        dy = fit_offset(sc, t, cfg)
        if dy: cam["hy"] += dy; sc = Scene(t, cfg["W"], cfg["H"], align=cfg.get("align", "right"), **cam)
    sc.fade_pow = fade_pow(t)
    return sc


def fade_pow(t):
    sh = getattr(t, "shape", {})
    return 1.0 + min(1.0, sh.get("horn", 0) + sh.get("canyon", 0))


def fit_offset(sc, t, cfg):
    """Vertical pan (reference px) that keeps the skyline off the top edge and the foreground off the
    bottom edge. Only acts on shapes the analysis flagged (horn, canyon); cones keep the standard framing."""
    sh = getattr(t, "shape", {})
    if not sh or (sh.get("horn", 0) < 0.05 and sh.get("canyon", 0) < 0.05): return 0.0
    s = sc.s; Href = sc.H / s
    cols = np.arange(sc.W)[np.arange(sc.W) / s > sc.fadeX + 300]
    sil = sc.silhouette[cols]; sil = sil[np.isfinite(sil)] / s
    if not sil.size: return 0.0
    ib = int(np.searchsorted(sc.vs, sc.v0 + 0.5 * sc.nf))
    yb = sc.Y[ib, cols]; yb = yb[np.isfinite(yb)] / s
    top = np.percentile(sil, 2); bottom = np.median(yb) if yb.size else top + 400
    lo_top = (0.21 if cfg.get("labels", True) else 0.12) * Href    # room for a label above the skyline
    hi_bot = 0.9 * Href
    need_min, need_max = lo_top - top, hi_bot - bottom          # dy must be >= need_min and <= need_max
    if need_min > 0: dy = need_min                               # the skyline wins over the foreground
    elif need_max < 0: dy = max(need_max, need_min)
    elif bottom < 0.55 * Href: dy = max(0.0, min(0.55 * Href - bottom, 0.28 * Href - top))   # don't float high
    else: dy = 0.0
    return float(dy)


# ---------------------------------------------------------------- styles
def style_survey(t, sc, cfg):
    W, H, s = sc.W, sc.H, sc.s
    ink, acc = hexrgb(cfg["ink"]), hexrgb(cfg["accent"])
    n1 = fbm(t.h.shape, 6, 5, 0.55, cfg["seed"])
    def alpha(c):
        i, px = c["i"], c["px"]
        hh, sh, sn, nn = (sc.sample(f, i, px) for f in (t.h, t.shade, t.snow, n1))
        pres = (0.02 + 0.05 * np.clip(nn + 0.1, 0, 1) + 0.81 * np.clip((hh - 0.15) / 0.9, 0, 1) ** 0.8) * sc.edge_fade(c["u"], c["v"])
        return pres * (0.95 - 0.55 * c["depth"]) * (0.35 + 0.75 * (1 - sh)) * (1 - 0.8 * sn) * sc.xfade(px)
    segs, cols, wids = sc.ridgelines(cfg["spacing"], alpha, lambda c: 1.25 - 0.6 * c["depth"], ink)
    pts = sc.peaks_screen([p["latlon"] for p in cfg["peaks"]])
    bg = parchment(W, H, cfg["seed"], hexrgb(cfg["paper"]))
    if cfg["sun"]:
        a = sun_anchor(pts, s)
        avoid = label_spans(sc, cfg["peaks"], pts) if cfg["labels"] else ()
        bg = draw_sun(bg, sc, sun_spot(sc, (a[0] - 20 * s, a[1] + 30 * s), 135 * s, pts[0], avoid=avoid), 135 * s, acc)
    fig, ax = figure(bg); add_lines(ax, segs, cols, wids)
    ok = np.where(np.isfinite(sc.silhouette))[0]
    sy = gaussian_filter1d(sc.silhouette[ok], 1.0)
    al = sc.xfade(ok) * np.clip((sc.s * (sc.hy + 270) - sy) / (120 * s), 0, 1)
    P = np.stack([ok, sy], 1); S = np.stack([P[:-1], P[1:]], 1)
    C = np.zeros((len(S), 4)); C[:, :3] = np.array(ink) * 0.8; C[:, 3] = al[:-1] * 0.9
    ax.add_collection(LineCollection(S, colors=C, linewidths=1.3 * s))
    if cfg["network"]: draw_network(ax, sc, pts, acc, dashed=True, seed=cfg["seed"])
    if cfg["labels"]: draw_labels(ax, sc, cfg["peaks"], pts, np.array(ink) * 0.9); footnote(ax, sc, cfg["footnote"], ink)
    return fig


def style_topo(t, sc, cfg):
    import contourpy
    W, H, s = sc.W, sc.H, sc.s
    ink, acc = np.array(hexrgb(cfg["ink"])), hexrgb(cfg["accent"])
    gen = contourpy.contour_generator(t.us, t.vs, gaussian_filter(t.h, 1.0))
    fig, ax = figure(parchment(W, H, cfg["seed"] + 4, hexrgb(cfg["paper"])))
    step = 0.04 * cfg["spacing"] / 0.375
    for k, lev in enumerate(np.arange(0.12, REF_H + 0.2, step)):
        idx = k % 5 == 0
        for line in gen.lines(lev):
            u, v = line[:, 0], line[:, 1]
            vis, sx, sy = sc.visible(u, v, np.full_like(u, lev), tol=2.0)
            fade = sc.xfade(sx) * np.clip((46 - np.abs(u)) / 12, 0, 1) * np.clip((v - sc.v0) / sc.nf, 0, 1) * np.clip((V1 - v) / 8, 0, 1) * np.clip(lev / 0.6, 0.15, 1)
            a = (0.85 if idx else 0.5) * fade
            P = np.stack([sx, sy], 1); S = np.stack([P[:-1], P[1:]], 1)
            keep = vis[:-1] & vis[1:] & (a[:-1] > 0.03) & (np.hypot(*(P[1:] - P[:-1]).T) < 25 * s)
            if not keep.any(): continue
            C = np.zeros((keep.sum(), 4)); C[:, :3] = ink; C[:, 3] = a[:-1][keep]
            ax.add_collection(LineCollection(S[keep], colors=C, linewidths=(0.95 if idx else 0.5) * s))
    pts = sc.peaks_screen([p["latlon"] for p in cfg["peaks"]])
    if cfg["network"]: draw_network(ax, sc, pts, np.array(acc) * 0.75, dashed=False, seed=cfg["seed"])
    if cfg["labels"]: draw_labels(ax, sc, cfg["peaks"], pts, ink * 0.9)
    real = step / t.hs * 1000
    footnote(ax, sc, f"contour interval ≈ {int(round(real / 10) * 10)} m", ink, 0.55)
    return fig


def style_nocturne(t, sc, cfg):
    W, H, s = sc.W, sc.H, sc.s
    rng = np.random.default_rng(cfg["seed"])
    yy, xx = np.mgrid[0:H, 0:W]
    sky_c = np.array(hexrgb(cfg["paper"])); pale = hexrgb(cfg["ink"]); moon_c = np.array(hexrgb(cfg["accent"]))
    tt = (yy / H) ** 1.3
    bg = sky_c * 0.55 * (1 - tt[..., None]) + np.clip(sky_c * 1.9 + 0.03, 0, 1) * tt[..., None]
    sky = sc.sky_mask(horizon_pad=90, fade_len=260)
    star = np.zeros((H, W)); n = int(2600 * s * H / 1000 + 200)
    sx, sy, mag = rng.uniform(0, W, n), rng.uniform(0, H, n), rng.pareto(2.2, n) + 0.3
    keep = rng.random(n) < np.clip(1.15 - sy / (650 * s), 0, 1)
    for x, y, m in zip(sx[keep], sy[keep], mag[keep]): star[int(y), int(x)] += min(m, 6) * 0.35
    star = gaussian_filter(star, 0.6 * s) * 2.2 + gaussian_filter(star, 5 * s) * 3
    lw, lh = max(W // 8, 8), max(H // 8, 8)
    band = np.exp(-(((yy - 0.35 * xx + 350 * s) / (230 * s)) ** 2)) * (0.5 + 0.5 * zoom(fbm((lh, lw), 5, 3, 0.6, 9), (H / lh, W / lw), order=1)[:H, :W])
    bg += (star[..., None] * np.array([0.85, 0.88, 1.0]) + band[..., None] * np.array([0.05, 0.06, 0.09])) * sky[..., None]
    if cfg["sun"]:
        mc, mr = np.array([(sc.cx - 670) * s, (sc.hy - 120) * s]), 34 * s
        d1 = np.hypot(xx - mc[0], yy - mc[1]); d2 = np.hypot(xx - mc[0] - 14 * s, yy - mc[1] + 8 * s)
        cres = np.clip((mr - d1) / 1.5, 0, 1) * np.clip((d2 - mr * 0.92) / 1.5, 0, 1)
        halo = np.exp(-(d1 / (180 * s)) ** 2) * 0.10
        bg += (cres[..., None] * moon_c + halo[..., None] * np.array([0.6, 0.65, 0.8])) * sky[..., None]
    g = gaussian_filter(1 - sky, 1.0)[..., None]
    bg = np.clip(bg * (1 - 0.5 * g) + (sky_c * 0.9 + 0.02) * 0.5 * g, 0, 1)
    def alpha(c):
        i, px = c["i"], c["px"]
        hh, sh, sn = (sc.sample(f, i, px) for f in (t.h, t.shade, t.snow))
        pres = (0.02 + 0.98 * np.clip((hh - 0.2) / 1.0, 0, 1) ** 0.9) * sc.edge_fade(c["u"], c["v"])
        return pres * (1 - 0.6 * c["depth"]) * (0.18 + 1.05 * sh ** 1.5 + 0.8 * sn) * sc.xfade(px)
    segs, cols, wids = sc.ridgelines(cfg["spacing"], alpha, lambda c: 1.15 - 0.5 * c["depth"], pale)
    fig, ax = figure(bg); add_lines(ax, segs, cols, wids, scale=0.75)
    if cfg["labels"]:
        pts = sc.peaks_screen([p["latlon"] for p in cfg["peaks"]]); draw_labels(ax, sc, cfg["peaks"], pts, pale)
    return fig


def style_stipple(t, sc, cfg):
    W, H, s = sc.W, sc.H, sc.s
    G = sc.gbuffer({"shade": t.shade, "h": t.h, "snow": t.snow})
    rng = np.random.default_rng(cfg["seed"])
    yy, xx = np.mgrid[0:H, 0:W]
    pres = np.clip((G["h"] - 0.2) / 1.7, 0, 1) ** 1.2 * sc.xfade(xx) * G["edge"]
    sm = G["shade"][G["mask"] & (pres > 0.3)]
    lo_, hi_ = (np.percentile(sm, 3), np.percentile(sm, 97)) if sm.size else (0.2, 0.9)
    dark = np.clip((hi_ - G["shade"]) / (hi_ - lo_ + 1e-6), 0, 1)
    tone = gaussian_filter(G["mask"] * pres * (0.10 + 0.85 * dark ** 1.3) * (1 - 0.85 * G["snow"]), 0.8 * s)
    rowf = gaussian_filter(G["row"].astype(float), 0.7)
    tone = np.clip(tone + 0.55 * np.clip(np.abs(np.gradient(rowf, axis=0)) / (6 * s), 0, 1) * G["mask"] * pres, 0, 1)
    sp = 2.3 * s
    gx, gy = np.meshgrid(np.arange(0, W, sp), np.arange(0, H, sp))
    px = (gx + rng.uniform(0, sp, gx.shape)).ravel(); py = (gy + rng.uniform(0, sp, gy.shape)).ravel()
    tv = tone[py.astype(int).clip(0, H - 1), px.astype(int).clip(0, W - 1)]
    keep = rng.random(tv.shape) < tv ** 1.1 * 1.05
    fig, ax = figure(parchment(W, H, cfg["seed"] + 8, hexrgb(cfg["paper"]), 0.06, 0.008))
    ax.scatter(px[keep], py[keep], s=rng.uniform(0.35, 1.3, keep.sum()) * s * s, c=[hexrgb(cfg["ink"])], linewidths=0)
    pts = sc.peaks_screen([p["latlon"] for p in cfg["peaks"]])
    if pts and pts[0] is not None: ax.scatter([pts[0][0]], [pts[0][1] - 22 * s], s=16 * s * s, c=[hexrgb(cfg["accent"])], linewidths=0)
    if cfg["labels"]: draw_labels(ax, sc, cfg["peaks"], pts, hexrgb(cfg["ink"]))
    footnote(ax, sc, cfg["footnote"], (0.2, 0.2, 0.2), 0.7)
    return fig


def halftone(xx, yy, tone, cell, angle=15):
    a = np.radians(angle); X = xx * np.cos(a) + yy * np.sin(a); Y = -xx * np.sin(a) + yy * np.cos(a)
    fx_, fy_ = np.mod(X, cell) - cell / 2, np.mod(Y, cell) - cell / 2
    return np.clip(cell * 0.62 * np.sqrt(np.clip(tone, 0, 1)) - np.hypot(fx_, fy_) + 0.5, 0, 1)


def style_riso(t, sc, cfg):
    W, H, s = sc.W, sc.H, sc.s
    rng = np.random.default_rng(cfg["seed"] + 11)
    yy, xx = np.mgrid[0:H, 0:W]
    PINK, BLUE = np.array(hexrgb(cfg["accent"])), np.array(hexrgb(cfg["ink"]))
    sky = sc.sky_mask()
    pts = sc.peaks_screen([p["latlon"] for p in cfg["peaks"]]); a = sun_anchor(pts, s)
    sun_r = 210 * s; sun_c = sun_spot(sc, (a[0] + 10 * s, a[1] + 10 * s), sun_r, pts[0])
    d = np.hypot(xx - sun_c[0], yy - sun_c[1])
    pink = halftone(xx, yy, np.clip(1 - d / (1300 * s), 0, 1) ** 1.8 * 0.75 * sky, 11 * s) * sky
    if cfg["sun"]: pink = np.maximum(pink, np.clip((sun_r - d) / 1.5, 0, 1) * sky * 0.92)
    G = sc.gbuffer({"shade": t.shade, "h": t.h})
    pres = np.clip((G["h"] - 0.15) / 1.3, 0, 1) * sc.xfade(xx) * G["edge"]
    pink = np.maximum(pink, halftone(xx, yy, G["mask"] * pres * G["shade"] ** 2 * 0.55, 7 * s))
    def alpha(c):
        i, px = c["i"], c["px"]; hh, sh = sc.sample(t.h, i, px), sc.sample(t.shade, i, px)
        return np.clip((hh - 0.15) / 0.9, 0, 1) ** 0.8 * sc.edge_fade(c["u"], c["v"]) * (0.5 + 0.8 * (1 - sh)) * sc.xfade(px)
    segs, cols, wids = sc.ridgelines(cfg["spacing"] * 4 / 3, alpha, lambda c: 1.6 - 0.7 * c["depth"], (0, 0, 0))
    fig, ax = figure(np.ones((H, W, 3))); add_lines(ax, segs, cols, wids, scale=0.8)
    if cfg["labels"] and cfg["title"]:
        ax.text(120 * s, 190 * s + sc.oy * s, cfg["title"], fontsize=46 * s, color="k", fontproperties=font_for(cfg["title"], serif=False))
        if cfg["footnote"]: ax.text(124 * s, 245 * s + sc.oy * s, cfg["footnote"], fontsize=9 * s, color="k", fontproperties=font_for(cfg["footnote"], serif=False))
    blue = np.clip((1 - fig_to_array(fig).mean(-1)) * 1.15, 0, 1)
    tex = np.clip(1 - 0.25 * np.abs(rng.standard_normal((H, W))) * gaussian_filter(rng.random((H, W)), 1.5), 0, 1)
    pink = shift(pink, (-3 * s, 5 * s), order=1) * tex
    blue = blue * np.clip(tex + 0.1, 0, 1)
    paper = parchment(W, H, cfg["seed"] + 2, hexrgb(cfg["paper"]), 0.03, 0.006)
    return paper * (1 - pink[..., None] * (1 - PINK)) * (1 - blue[..., None] * (1 - BLUE))


def style_woodcut(t, sc, cfg):
    W, H, s = sc.W, sc.H, sc.s
    G = sc.gbuffer({"h": t.h})
    yy, xx = np.mgrid[0:H, 0:W]
    paper_c = np.array(hexrgb(cfg["paper"])); ink = np.array(hexrgb(cfg["ink"])); red = np.array(hexrgb(cfg["accent"]))
    body = np.clip((G["h"] - 0.25) / 0.6, 0, 1) * sc.xfade(xx, 350, 50) * G["edge"]
    body = gaussian_filter(body, 0.7 * s)
    bg = parchment(W, H, cfg["seed"] + 13, tuple(paper_c), 0.1)
    pts = sc.peaks_screen([p["latlon"] for p in cfg["peaks"]])
    p0 = pts[0] if pts and pts[0] is not None else np.array([2150 * s, 300 * s])
    if cfg["sun"]:
        sun_c = sun_spot(sc, (p0[0] + 330 * s, p0[1] + 60 * s), 150 * s, p0); d = np.hypot(xx - sun_c[0], yy - sun_c[1])
        sun = np.clip((150 * s - d) / 1.5, 0, 1) * (1 - body)
        sun *= 1 - 0.85 * ((np.mod(d, 22 * s) < 3.2 * s) & (d > 30 * s))
        bg = bg * (1 - sun[..., None] * 0.92) + red * sun[..., None] * 0.92
    rt = gaussian_filter(np.random.default_rng(1).random((H, W)), 1.2)
    bg = bg * (1 - body[..., None]) + ink * body[..., None] * (0.92 + 0.12 * rt[..., None])
    def alpha(c):
        i, px = c["i"], c["px"]; hh = sc.sample(t.h, i, px)
        return np.clip((hh - 0.3) / 0.5, 0, 1) * sc.xfade(px, 300, 80) * np.clip((46 - np.abs(c["u"])) / 12, 0, 1)
    def width(c):
        i, px = c["i"], c["px"]; sh, sn = sc.sample(t.shade, i, px), sc.sample(t.snow, i, px)
        return (0.15 + 3.2 * np.clip((sh - 0.45) / 0.5, 0, 1) ** 1.6 + 4 * sn) * (1.15 - 0.45 * c["depth"])
    segs, cols, wids = sc.ridgelines(cfg["spacing"] * 4 / 3, alpha, width, paper_c)
    fig, ax = figure(np.clip(bg, 0, 1)); add_lines(ax, segs, cols, wids, scale=0.8, cap="butt")
    if cfg["sun"]:   # a few carved wind strokes, kept clear of the peak and sun
        for k, (dx, dy, L) in enumerate([(-800, -200, 300), (-380, -30, 210), (-60, -230, 250), (560, -200, 220)]):
            x0, y0 = p0[0] / s + dx, p0[1] / s + dy
            if y0 < 40: continue
            xs = np.clip(((x0 + L * np.linspace(0, 1, 21)) * s).astype(int), 0, W - 1)
            if np.any((y0 + 20) * s > np.where(np.isfinite(sc.silhouette[xs]), sc.silhouette[xs], H + 10) - 20 * s): continue   # would cross terrain
            if np.min(np.hypot(xs - sun_c[0], (y0 + 5) * s - sun_c[1])) < 175 * s: continue                                  # or the sun
            tt = np.linspace(0, 1, 80)
            for off, lw in ((0, 1.6), (9, 0.9)):
                ax.plot((x0 + off * 2 + L * tt * (1 - off / 40)) * s, (y0 + off + 5 * np.sin(tt * 5 + k)) * s, color=ink, lw=lw * s, alpha=0.8, solid_capstyle="round")
    if cfg["labels"] and cfg["title"]:
        ax.text(W - 70 * s, H - 60 * s, cfg["title"], fontsize=12 * s, ha="right", color=red, fontproperties=font_for(cfg["title"]))
    return fig


def draw_network(ax, sc, pts, color, dashed=True, seed=4):
    from scipy.spatial import Delaunay
    rng = np.random.default_rng(seed); s = sc.s; t = sc.t
    cand = np.array([(rng.uniform(-30, 40), rng.uniform(max(-25, sc.v0 + 2), 22)) for _ in range(900)])
    hh = np.array([t.h[np.argmin(np.abs(sc.vs - v)), np.argmin(np.abs(sc.us - u))] for u, v in cand])
    vis, sx, sy = sc.visible(cand[:, 0], cand[:, 1], hh, tol=2)
    sel = np.where(vis & (sx > (sc.fadeX + 150) * s) & (sx < sc.W - 50 * s) & (sy < 0.86 * sc.H) & (sy > 0.2 * sc.H))[0]
    nodes = [p for p in pts if p is not None]
    for i in sel[np.argsort(-hh[sel])]:
        if all(np.hypot(sx[i] - a, sy[i] - b) > 150 * s for a, b in nodes): nodes.append(np.array([sx[i], sy[i]]))
        if len(nodes) >= 20: break
    nodes = np.array(nodes)
    if len(nodes) < 3: return
    E = set()
    for tri in Delaunay(nodes).simplices:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[0], tri[2])):
            if np.hypot(*(nodes[a] - nodes[b])) < 420 * s: E.add((min(a, b), max(a, b)))
    for a, b in E:
        ax.plot(*nodes[[a, b]].T, color=color, lw=0.6 * s, alpha=0.45, ls=(0, (5, 5)) if dashed else "-")
    for k, (x, y) in enumerate(nodes):
        r = (9 if k < 2 else 5) * s
        ax.add_patch(plt.Circle((x, y), r + 5 * s, fill=False, ec=color, lw=0.8 * s, alpha=0.7))
        ax.add_patch(plt.Circle((x, y), r * 0.5, color=color, alpha=0.9))


RENDERERS = dict(survey=style_survey, topo=style_topo, nocturne=style_nocturne, stipple=style_stipple, riso=style_riso, woodcut=style_woodcut)


def render(style, dem, cfg, out, preview=None):
    c = dict(DEFAULTS[style]); c.update({k: v for k, v in cfg.items() if v is not None})
    for k in ("paper", "ink", "accent"):
        if cfg.get(f"{style}_{k}"): c[k] = cfg[f"{style}_{k}"]
    q = c.get("quality", 1.0)
    t = terrain(dem, c["summit"], view_from=c.get("view_from"), facing=c.get("facing"), extent_km=c.get("extent"),
                base_m=c.get("base"), yaw=c.get("yaw", 0.0), NX=int(1400 * q), NZ=int(520 * q),
                shape_aware=not c.get("no_shape", False))
    sh = t.shape
    # shape-aware defaults; anything given explicitly (CLI flag or cfg key) wins
    c.setdefault("relief", sh.get("relief", 1.0))
    c.setdefault("spacing", 0.375 * sh.get("spacing", 1.0))
    c.setdefault("zoom", sh.get("zoom", 1.0))
    c["pan_y"] = cfg.get("pan_y")
    c["fit"] = not c.get("no_shape", False)
    lighting(t, c["light"], c.get("snowline"))
    if not c["peaks"]: c["peaks"] = [dict(name="", sub="", latlon=t.summit)]
    c["peaks"][0]["latlon"] = t.summit
    sc = scene_for(style, t, c)
    res = RENDERERS[style](t, sc, c)
    return save_png(res, out, preview), t


def parse_ll(s):
    a, b = s.split(","); return float(a), float(b)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--peak", required=True, help="lat,lon of the main summit")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--from", dest="view_from", help="lat,lon of the viewpoint (e.g. a town)")
    g.add_argument("--facing", type=float, help="compass bearing the camera looks toward (0 = north)")
    ap.add_argument("--style", default="survey", choices=STYLES + ["all"], help="print style, or all six plus a contact sheet")
    ap.add_argument("--out", required=True, help="file (.png/.jpg) or directory when --style all")
    ap.add_argument("--dem", help="cached .npz (default: dem_<lat>_<lon>.npz next to --out, refetched if it covers too small an area)")
    ap.add_argument("--size", default="3000x1000", help="WxH, e.g. 3000x1000 (X header), 1584x396 (LinkedIn), 3840x2160")
    ap.add_argument("--align", default="right", choices=["right", "center"], help="right leaves the left third empty (headers)")
    ap.add_argument("--label", action="append", default=[], help='"Name|sub" or "Name|sub@lat,lon"')
    ap.add_argument("--title", default="", help="big title used by riso/woodcut")
    ap.add_argument("--footnote", default=None)
    ap.add_argument("--extent", type=float, help="scene half-width km (default ~10x the mountain's height)")
    ap.add_argument("--base", type=float, help="plain elevation in m (default: auto)")
    ap.add_argument("--snowline", type=float, help="m (default: 80%% of the way up)")
    ap.add_argument("--yaw", type=float, default=0.0, help="orbit the camera around the summit, degrees")
    ap.add_argument("--zoom", type=float, help="default 1")
    ap.add_argument("--relief", type=float, help="vertical exaggeration multiplier (default: 1, a little less on rough terrain)")
    ap.add_argument("--camh", type=float, help="camera height, normalised km (default 3.4; canyons look down from higher)")
    ap.add_argument("--pan-x", type=float, default=0)
    ap.add_argument("--pan-y", type=float, help="reference px; default 0, or an automatic fit for horns and canyons")
    ap.add_argument("--spacing", type=float, help="ridge-line gap (normalised km, default 0.375, a little wider on rough terrain)")
    ap.add_argument("--no-shape", action="store_true", help="cone framing rules for every landform (ignore the landform analysis)")
    ap.add_argument("--paper"); ap.add_argument("--ink"); ap.add_argument("--accent")
    ap.add_argument("--no-sun", action="store_true"); ap.add_argument("--no-labels", action="store_true"); ap.add_argument("--network", action="store_true")
    ap.add_argument("--seed", type=int, default=1915)
    ap.add_argument("--quality", type=float, default=1.0, help="terrain grid density multiplier (0.5 = fast drafts)")
    a = ap.parse_args()

    summit = parse_ll(a.peak)
    W, H = map(int, a.size.lower().split("x"))
    styles = STYLES if a.style == "all" else [a.style]
    outdir = a.out if a.style == "all" else os.path.dirname(os.path.abspath(a.out))
    os.makedirs(outdir, exist_ok=True)
    dem_path = a.dem or os.path.join(outdir, f"dem_{summit[0]:.3f}_{summit[1]:.3f}.npz")
    ensure_dem(*summit, (a.extent or 45) * 1.45, dem_path, explicit=bool(a.dem), log=lambda m: print(m, flush=True))
    dem = load_dem(dem_path)
    peaks = []
    for L in a.label:
        name, _, rest = L.partition("|"); sub, _, ll = rest.partition("@")
        peaks.append(dict(name=name, sub=sub, latlon=parse_ll(ll) if ll else summit))
    peaks.sort(key=lambda p: p["latlon"] != summit)
    if peaks and peaks[0]["latlon"] != summit: peaks.insert(0, dict(name="", sub="", latlon=summit))
    cfg = dict(summit=summit, view_from=parse_ll(a.view_from) if a.view_from else None, facing=a.facing,
               W=W, H=H, align=a.align, peaks=peaks, title=a.title, extent=a.extent, base=a.base, snowline=a.snowline,
               yaw=a.yaw, zoom=a.zoom, relief=a.relief, camh=a.camh, pan_x=a.pan_x, pan_y=a.pan_y, spacing=a.spacing,
               sun=not a.no_sun, labels=not a.no_labels, network=a.network, seed=a.seed, quality=a.quality, no_shape=a.no_shape,
               footnote=a.footnote if a.footnote is not None else f"{abs(summit[0]):.2f}°{'N' if summit[0] >= 0 else 'S'}  {abs(summit[1]):.2f}°{'E' if summit[1] >= 0 else 'W'}")
    for k in ("paper", "ink", "accent"):
        if getattr(a, k):
            for st in styles: cfg[f"{st}_{k}"] = getattr(a, k)
    outs = []
    for st in styles:
        out = os.path.join(outdir, f"{st}.png") if a.style == "all" else a.out
        path, t = render(st, dem, dict(cfg, peaks=[dict(p) for p in peaks]), out)
        outs.append(path); print(f"{st:9s} -> {path}", flush=True)
    sh = t.shape
    print(json.dumps(dict(summit_found=t.summit, peak_m=round(t.peak_m), base_m=round(t.base_m), extent_km=round(t.extent_km, 1),
                          kind=sh.get("kind", "legacy"), shape={k: sh[k] for k in ("steep", "front", "rough", "above", "horn", "rough_score", "canyon") if k in sh})))
    if len(outs) > 1:
        from PIL import Image
        tw = min(1200, W); th = int(tw * H / W); sheet = Image.new("RGB", (2 * tw + 60, 3 * th + 80), (24, 24, 26))
        for k, p in enumerate(outs):
            r_, c_ = divmod(k, 2); sheet.paste(Image.open(p).convert("RGB").resize((tw, th), Image.LANCZOS), (20 + c_ * (tw + 20), 20 + r_ * (th + 20)))
        sheet.save(os.path.join(outdir, "contact_sheet.jpg"), quality=90); print("contact sheet ->", os.path.join(outdir, "contact_sheet.jpg"))


if __name__ == "__main__":
    main()
