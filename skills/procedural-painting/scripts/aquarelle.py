"""aquarelle: watercolour on paper, simulated.

    from aquarelle import *
    p = Paper(1200, 1600, seed=3)
    p.wet(sky)                                   # clean water first: what follows will spread in it
    p.wash(sky, 'cerulean', strength=p.grade(90, 0.5, 0.1))
    p.drop([(0.4, 0.2)], 'paynes grey', size=0.05)   # charged in: it blooms
    p.dry()
    p.wash(hill, 'sap green', strength=0.6)      # on dry paper: a hard edge that dries darker at the rim
    p.stroke(grass, 'burnt umber', size=0.004, dry=0.7)
    p.save('out.png')

What is simulated (nothing here is a filter over a finished picture):

  paper     a cold-press sheet: a height field of bumps and fibres. White is the paper; there is
            no white paint. Every layer is transparent: absorption (K/S in 12 bands, the same
            pigment tables as atelier) adds up over the paper's own.
  water     a film on the surface and dampness in the sheet. Water creeps only through damp
            paper, so a wash on dry paper keeps the edge it was given, and a wash on wet paper
            has no edge at all.
  pigment   each wash is paint in suspension. It diffuses where the film is deep (wet-in-wet),
            is carried by the water as it moves from wetter to drier paper (a backrun, when a
            wetter touch lands in damp paint), settles as the film thins, and is carried to the
            drying rim, where it piles up (the dark edge of a wash). Carrying is done with a
            scheme that conserves pigment, so it really collects where the flow stops.
  grain     granulating pigments (ultramarine, cerulean, the earths) settle into the valleys
            of the paper; a dry brush touches only the crowns.

numpy, scipy and pillow only (ffmpeg for the process film). See ../references/aquarelle.md.
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atelier import (F32, NB, PIGMENTS, blur, fill, load_image, q_to_rgb, rgb, rgb_to_q, sd_box, sd_circle,  # noqa: E402,F401
                     sd_curve, sd_ellipse, sd_poly, sd_segment, smin, smoothstep, value_noise, curve, _resize)

# how strongly a pigment settles into the paper's valleys, and how hard it is to move once down
GRANULATION = {'ultramarine': 0.9, 'cerulean': 0.85, 'cobalt blue': 0.6, 'cobalt violet': 0.7, 'cobalt turquoise': 0.6,
               'burnt sienna': 0.55, 'raw sienna': 0.45, 'raw umber': 0.55, 'burnt umber': 0.5, 'yellow ochre': 0.35,
               'viridian': 0.45, 'terre verte': 0.5, 'ivory black': 0.45, 'venetian red': 0.4, 'paynes grey': 0.3}
STAINING = {'phthalo blue': 0.9, 'prussian blue': 0.85, 'alizarin crimson': 0.8, 'quinacridone rose': 0.85,
            'dioxazine violet': 0.85, 'sap green': 0.6, 'indian yellow': 0.6, 'cadmium red': 0.5}


def _ks(color):
    """A colour (name, hex, rgb triple or H x W x 3 field) as absorption K/S in NB bands."""
    q = rgb_to_q(rgb(color) if isinstance(color, str) or np.ndim(color) == 1 else np.asarray(color, F32))
    return (q * q).astype(F32)


def mixture(recipe):
    """Tube colours mixed in the palette, by parts: mixture({'ultramarine': 2, 'burnt sienna': 1}).
    Returns something any `color` argument takes; the mix keeps its pigments' granulation."""
    tot = float(sum(recipe.values()))
    ks = sum(_ks(k) * (v / tot) for k, v in recipe.items())
    g = sum(GRANULATION.get(k, 0.15) * v for k, v in recipe.items()) / tot
    s = sum(STAINING.get(k, 0.3) * v for k, v in recipe.items()) / tot
    return _Mix(ks, g, s)


class _Mix:
    def __init__(self, ks, g, s):
        self.ks, self.g, self.s = ks.astype(F32), g, s


class _Paint:
    """One wash while it is wet: pigment in the water (a), pigment already down (s)."""
    __slots__ = ('ks', 'a', 's', 'g', 'stain', 'crisp', 'name')


class Paper:
    """A sheet of cold-press paper, `width` x `height` pixels. Fields p.x (0..aspect) and p.y
    (0 at the top..1) are in picture heights, as in atelier.

    grain   depth of the paper's tooth (0 = hot-press smooth, 1 = cold-press, 1.6 = rough)
    paper   colour of the sheet
    tilt    how steeply the board is tipped towards the painter: washes run down and bead
    sim     height of the grid the water moves on (default: half the height, at most 900)
    film    True keeps a frame after every step of the water, for p.film(path)
    """

    def __init__(self, width, height, seed=1, grain=1.0, paper='#f8f4ea', tilt=0.15, sim=None, film=False, log=True):
        self.W, self.H, self.seed = int(width), int(height), seed
        self.rng = np.random.default_rng(seed)
        self._hrng = np.random.default_rng([seed, 91])        # the wobble of the hand: its own stream, so turning it off changes nothing else
        self._trng = np.random.default_rng([seed, 77])        # textures the size of a pixel; kept apart so the picture is the same at any size
        H, W = self.H, self.W
        yy, xx = np.mgrid[0:H, 0:W].astype(F32)
        self.x, self.y = (xx + 0.5) / H, (yy + 0.5) / H
        self.hs = int(sim or min(900, max(240, H // 2)))
        self.ws = max(2, int(round(self.hs * W / H)))
        self.px = self.hs / 1000.0                          # one thousandth of the height, in water-grid cells
        self.grain, self.tilt = float(grain), float(tilt)
        # the sheet: bumps a couple of millimetres across, with finer fibre between them
        s = H / 1000.0
        t = 0.62 * value_noise(H, W, 2.6 * s, self._trng, 2) + 0.30 * value_noise(H, W, 1.2 * s, self._trng) \
            + 0.16 * value_noise(H, W, 0.6 * s, self._trng) + 0.25 * value_noise(H, W, 9.0 * s, self._trng)
        lo, hi = np.percentile(t, [0.5, 99.5])
        self.tooth = np.clip((t - lo) / (hi - lo), 0, 1).astype(F32)              # 1 = crown, 0 = valley
        self._tooth_soft = ndi.gaussian_filter(self.tooth, 0.45 * s)
        self._ksp = _ks(paper)
        self.dep = np.zeros((H, W, NB), np.float32)         # dried pigment, as absorption
        self.dep_s = np.zeros((self.hs, self.ws, NB), F32)  # the same on the water grid, for looking
        self.w = np.zeros((self.hs, self.ws), F32)          # water on the surface
        self.d = np.zeros((self.hs, self.ws), F32)          # dampness in the sheet
        self.perm = np.exp(0.75 * self._noise(self.hs, self.ws, 0.016, 4, 0.65)).astype(F32)   # where water runs easily
        self.wet_paints = []
        self._hand_f = None
        self._log, self._frames, self.n_ops = log, ([] if film else None), 0

    # ---- small helpers ---------------------------------------------------------------
    def noise(self, scale=0.05, octaves=3):
        """Smooth noise over the sheet (zero mean, unit spread); `scale` in picture heights."""
        return self._noise(self.H, self.W, scale, octaves)

    def _noise(self, h, w, scale, octaves=3, gain=0.5, rng=None, rh=384):
        # made on a fixed grid and enlarged, so the same seed gives the same picture at any size
        rw = max(2, int(round(rh * self.W / self.H)))
        n = value_noise(rh, rw, scale * rh, rng or self.rng, octaves, gain)
        return _resize(n, w, h, Image.BICUBIC)

    def _hand(self, m, hand):
        """The edge of a shape as a loaded brush leaves it: it drifts off the drawn line, and bulges
        in small scallops where the brush was set down again. Returns the mask and, along its
        edge, how unevenly the brush was loaded (a field around 0)."""
        H, W = self.H, self.W
        if hand <= 0 or float(m.min()) > 0.5 or float(m.max()) < 0.5:
            return m, 0.0
        r = self._hrng
        if self._hand_f is None:                # one hand for the whole sheet: washes that share an edge still meet along it
            self._hand_f = ([self._noise(H, W, 0.16, 2, rng=r) for _ in range(2)], [self._noise(H, W, 0.035, 2, rng=r, rh=512) for _ in range(2)])
        drift, lob = self._hand_f
        amp = hand * H
        dx = 0.0045 * amp * drift[0] + 0.0022 * amp * (np.abs(lob[0]) - 0.8)       # |noise|: rounded bulges meeting in cusps
        dy = 0.0045 * amp * drift[1] + 0.0022 * amp * (np.abs(lob[1]) - 0.8)
        # never move an edge further than a third of the shape's own thickness there
        thick = ndi.distance_transform_edt(m > 0.5) + ndi.distance_transform_edt(m <= 0.5)
        lim = np.clip(blur(np.minimum(ndi.maximum_filter(thick, int(0.02 * H) | 1), 0.05 * H).astype(F32), 0.006 * H) / (3 * 0.0067 * amp + 1e-6), 0, 1)
        yy, xx = np.mgrid[0:H, 0:W].astype(F32)
        out = ndi.map_coordinates(m, [yy + dy * lim, xx + dx * lim], order=1, mode='nearest').astype(F32)
        band = np.clip(out * (1 - blur(out, 0.012 * H)) * 3.0, 0, 1)
        return out, band * self._noise(H, W, 0.05, 2, rng=r) * 0.6 * hand

    def grade(self, angle=90.0, start=1.0, end=0.2, mask=None):
        """A strength that runs from `start` to `end` across the sheet (or across `mask`) in the
        direction `angle` (degrees; 90 = downwards): pass it as `strength` for a graded wash."""
        a = np.deg2rad(angle)
        t = self.x * np.cos(a) + self.y * np.sin(a)
        sel = t[mask > 0.5] if mask is not None and (mask > 0.5).any() else t
        lo, hi = float(sel.min()), float(sel.max())
        return (start + (end - start) * np.clip((t - lo) / max(hi - lo, 1e-6), 0, 1)).astype(F32)

    def _small(self, f):
        f = np.asarray(f, F32)
        if f.shape[:2] == (self.hs, self.ws):
            return f
        return np.clip(_resize(f, self.ws, self.hs, Image.BOX), 0, None)

    def _big(self, f):
        return np.clip(_resize(f, self.W, self.H, Image.BICUBIC), 0, None)

    def _field(self, v):
        return np.broadcast_to(np.asarray(v, F32), (self.H, self.W))

    def _pigment(self, color, granulate, staining):
        if isinstance(color, _Mix):
            ks, g, s = color.ks, color.g, color.s
        elif isinstance(color, str):
            ks, g, s = _ks(color), GRANULATION.get(color, 0.15), STAINING.get(color, 0.3)
        elif np.ndim(color) == 3:                           # a study: kept on the water grid
            c = np.stack([_resize(np.asarray(color, F32)[..., i], self.ws, self.hs, Image.BOX) for i in range(3)], -1)
            ks, g, s = _ks(np.clip(c, 0, 1)), 0.25, 0.3
        else:
            ks, g, s = _ks(color), 0.2, 0.3
        return ks, (g if granulate is None else granulate), (s if staining is None else staining)

    # ---- putting water and paint down --------------------------------------------------
    def wet(self, mask, amount=0.8, feather=0.012, hand=1.0):
        """Clean water on the paper: what is painted here next will have soft edges. `feather`
        (picture heights) is how far the damp spreads past the brushed edge."""
        m, _ = self._hand(np.clip(self._field(mask), 0, 1).astype(F32), hand)
        m = self._small(blur(m, feather * self.H))
        self._rewet(m * amount)
        self.w = np.maximum(self.w, m * amount)
        self.d = np.maximum(self.d, np.clip(m * 1.5, 0, 1))
        self._step(3)

    def _rewet(self, new_water):
        """Water landing on paint that is down but not dry lifts some of it back into the water."""
        for p in self.wet_paints:
            k = np.clip((new_water - self.w) * 1.6, 0, 1) * 0.9 * (1 - p.stain)
            p.a += p.s * k
            p.s *= 1 - k

    def wash(self, mask, color, strength=1.0, wetness=0.75, vary=0.2, granulate=None, staining=None, settle=4, name=None,
             hand=1.0):
        """Lay a wash: paint in water over `mask` (H x W, 0..1). On dry paper it keeps the edge of
        the mask and dries darker at the rim; where the paper is wet (p.wet, or an earlier wash
        that has not dried) it spreads and mixes.

        color     a tube name ('ultramarine'), a hex colour, mixture({...}), or an H x W x 3 study
        strength  how much pigment is in the water (a number or a field; see p.grade). A tube
                  colour at 1 is near its full depth; 0.1-0.4 is a tint.
        wetness   how much water is on the brush: more spreads further on wet paper and takes
                  longer to dry
        vary      unevenness of the loading (pigment pools here and there)
        settle    how long the water is left to move before the next touch (steps)
        hand      how much the edge wanders off the mask, as a hand-held brush does: 0 keeps the
                  mask exactly, 1 is a steady hand, 2 a loose one. Thin shapes are moved less.
        """
        H, W = self.H, self.W
        m, uneven = self._hand(np.clip(self._field(mask), 0, 1).astype(F32), hand)
        ks, g, stain = self._pigment(color, granulate, staining)
        # where the paper is dry the brush leaves a crisp edge that follows the tooth a little
        hard = smoothstep(-0.06, 0.06, m - 0.5 - 0.22 * self.grain * (self.tooth - 0.5) * (m < 0.98))
        damp = np.clip(self._big(np.clip(np.maximum(self.d, self.w * 2), 0, 1)), 0, 1)
        foot = hard * (1 - damp) + m * damp
        load = np.asarray(strength, F32) * (1 + vary * self.noise(0.07, 3) + uneven)
        ms = self._small(foot)
        water = ms * wetness
        self._rewet(water)
        p = _Paint()
        p.ks, p.g, p.stain, p.name = ks, g, stain, name
        tot = self._small(foot * np.clip(load, 0, None))
        held = float(np.clip(1 - 0.85 * wetness, 0.1, 0.9))                 # a drier brush leaves its pigment where it touches
        p.a, p.s = tot * (1 - held), tot * held
        p.crisp = (np.maximum(hard, damp) * 255).astype(np.uint8)         # what may show on the sheet, at full size
        self.wet_paints.append(p)
        self.w = np.maximum(self.w, water)
        self.d = np.maximum(self.d, np.clip(ms * 1.5, 0, 1))
        self.n_ops += 1
        if self._log:
            print(f"  {name or 'wash':<18s} {100 * float(m.mean()):5.1f}% of the sheet, granulation {g:.2f}", flush=True)
        self._step(settle)

    def drop(self, points, color, size=0.03, strength=1.0, water=1.0, ragged=0.35, within=None, **kw):
        """Touch a loaded brush to the paper at each (x, y) (picture heights): on wet paper the
        colour blooms outwards; with color=None it is clean water, which pushes the paint that
        is there into a backrun. `size` is the radius (one value or one per point); `within` is a
        mask the touch stays inside (the wet shape being charged)."""
        pts = np.asarray(points, F32).reshape(-1, 2)
        sz = np.broadcast_to(np.asarray(size, F32), (len(pts),))
        m = np.zeros((self.H, self.W), F32)
        n = self.noise(0.03, 3)
        for (cx, cy), r in zip(pts, sz):
            y0, y1 = int(max(0, (cy - 2 * r) * self.H)), int(min(self.H, (cy + 2 * r) * self.H + 1))
            x0, x1 = int(max(0, (cx - 2 * r) * self.H)), int(min(self.W, (cx + 2 * r) * self.H + 1))
            if y1 <= y0 or x1 <= x0:
                continue
            sl = (slice(y0, y1), slice(x0, x1))
            dd = np.hypot(self.x[sl] - cx, self.y[sl] - cy) / r * (1 + ragged * n[sl])
            m[sl] = np.maximum(m[sl], smoothstep(1.0, 0.75, dd))
        if within is not None:
            m = m * np.clip(self._field(within), 0, 1)
        self._backrun(m, water, kw.pop('bloom', 1.0))
        if color is None:
            ms = self._small(m) * water
            self._rewet(ms)
            self.w = np.maximum(self.w, ms)
            self.d = np.maximum(self.d, np.clip(ms * 1.5, 0, 1))
            self._step(kw.get('settle', 4))
        else:
            kw.setdefault('hand', 0)
            self.wash(m, color, strength=strength, wetness=water, name=kw.pop('name', 'drop'), **kw)

    def _backrun(self, m, water, bloom=1.0):
        """A wetter touch in paint that has started to dry: the new water creeps out through the
        damp wash and sweeps the pigment ahead of it. The middle is left pale; the pigment stops
        where the water stops, in a dark fringe with a lobed, feathered outline. On paper that is
        still running wet nothing of the kind happens (the touch just blends), and dry paper
        does not carry the water at all."""
        if bloom <= 0 or not self.wet_paints:
            return
        px = self.px
        ms = self._small(m)
        ready = smoothstep(0.03, 0.15, self.d) * (1 - smoothstep(0.32, 0.6, self.w))         # damp, no longer flooded
        extra = np.clip(ms * water - self.w, 0, 1)
        if float((extra * ready).max()) < 0.15:
            return
        # how far the water gets: further where the sheet lets it through, in lobes on lobes
        r = self._hrng
        lobes = 0.55 * np.abs(self._noise(self.hs, self.ws, 0.030, 2, rng=r, rh=512)) + 0.30 * np.abs(self._noise(self.hs, self.ws, 0.011, 2, rng=r, rh=768)) \
            + 0.15 * np.abs(value_noise(self.hs, self.ws, 3.0 * px, self._trng))
        reach = (9.0 + 26.0 * float(np.clip(water, 0, 1.5))) * px * bloom
        src_ = (extra * ready > 0.15)
        dist = ndi.distance_transform_edt(~src_).astype(F32)
        R = (dist < reach * (0.35 + 1.1 * lobes) * self.perm.clip(0.6, 1.6)) & (ready > 0.2)
        R = ndi.binary_fill_holes(ndi.binary_opening(R, iterations=1) | src_)
        lab, _ = ndi.label(R)                                               # only water that is joined to the touch
        R = np.isin(lab, np.unique(lab[src_])) & (lab > 0)
        if R.sum() < 4:
            return
        ys, xs = np.where(R)
        pad = 3
        y0, y1, x0, x1 = max(0, ys.min() - pad), min(self.hs, ys.max() + pad + 1), max(0, xs.min() - pad), min(self.ws, xs.max() + pad + 1)
        sl = (slice(y0, y1), slice(x0, x1))
        Rb = R[sl]
        inside = ndi.distance_transform_edt(Rb).astype(F32)                 # distance to the edge of the bloom
        gy, gx = np.gradient(ndi.gaussian_filter(inside, 1.0))
        nrm = np.maximum(np.hypot(gx, gy), 1e-4)
        slow = 0.42 * (0.12 + 0.88 * smoothstep(0.0, 3.5, inside)) * Rb     # outwards, slowing into the edge, where it piles up
        ux, uy = -slow * gx / nrm, -slow * gy / nrm
        steps = int(min(200, float(inside.max()) / 0.42 * 1.1 + 10))
        swept = smoothstep(0.0, 3.0 * px, inside) * ready[sl]
        for p in self.wet_paints:
            lifted = (p.a[sl] + p.s[sl]) * swept * float(np.clip(0.78 * (1 - 0.8 * p.stain) * bloom, 0, 0.92))
            if float(lifted.max()) <= 1e-6:
                continue
            from_s = np.minimum(p.s[sl], lifted)
            p.s[sl] -= from_s
            p.a[sl] = np.maximum(p.a[sl] - (lifted - from_s), 0)
            t = lifted.copy()
            for _ in range(steps):
                self._advect(t, ux, uy)
            cap = 5.0 * float(np.percentile(lifted[lifted > 0], 90))         # a fringe, not beads: what piles too high spreads along it
            over = np.maximum(t - cap, 0)
            t = np.minimum(t, cap) + ndi.gaussian_filter(over, 2.5) * Rb
            p.s[sl] += ndi.gaussian_filter(t, 0.6)                           # it dries where it was left: the fringe
        self.w[sl] = np.maximum(self.w[sl], 0.30 * Rb)
        self._frame()

    def stroke(self, lines, color, size=0.008, strength=1.0, dry=0.0, wetness=0.3, taper=0.6, granulate=None,
               staining=None, name=None):
        """Brush strokes: each line is a list of (x, y) points (picture heights). dry=0 is a
        loaded brush (a small wash: soft where the paper is wet, hard and rim-dark where it is
        dry). dry>0 is a brush with little water dragged over the sheet: it catches the crowns
        of the paper, skips the valleys, and gives out along the stroke. `size` is the width
        (one value or one per line); `taper` thins the stroke to a point at its end."""
        H, W = self.H, self.W
        sizes = np.broadcast_to(np.asarray(size, F32), (len(lines),))
        m = np.zeros((H, W), F32)          # coverage
        e = np.zeros((H, W), F32)          # 1 on the stroke's centre line, 0 at its edge
        u = np.zeros((H, W), F32)          # how far along its stroke each pixel is (0..1)
        for ln, wd in zip(lines, sizes):
            P = np.asarray(ln, F32)
            if len(P) < 2:
                P = np.concatenate([P, P + [[wd * 0.5, 0]]])
            seg = np.hypot(*np.diff(P, axis=0).T)
            cum = np.concatenate([[0], np.cumsum(seg)])
            L = max(float(cum[-1]), 1e-6)
            pad = wd + 2.0 / H
            y0, y1 = int(max(0, (P[:, 1].min() - pad) * H)), int(min(H, (P[:, 1].max() + pad) * H + 1))
            x0, x1 = int(max(0, (P[:, 0].min() - pad) * H)), int(min(W, (P[:, 0].max() + pad) * H + 1))
            if y1 <= y0 or x1 <= x0:
                continue
            sl = (slice(y0, y1), slice(x0, x1))
            X, Y = self.x[sl], self.y[sl]
            best = np.full(X.shape, 1e9, F32)
            ub = np.zeros(X.shape, F32)
            for i in range(len(P) - 1):
                ax, ay = P[i]
                bx, by = P[i + 1]
                dx, dy = bx - ax, by - ay
                t = np.clip(((X - ax) * dx + (Y - ay) * dy) / max(dx * dx + dy * dy, 1e-12), 0, 1)
                along = (cum[i] + t * seg[i]) / L
                half = 0.5 * wd * (1 - taper * smoothstep(0.45, 1.0, along)) * np.minimum(1, 0.35 + along * L / (0.6 * wd + 1e-9))
                sd = np.hypot(X - (ax + t * dx), Y - (ay + t * dy)) - half
                better = sd < best
                best = np.where(better, sd, best)
                ub = np.where(better, along, ub)
            cov = smoothstep(0.8 / H, -0.8 / H, best)
            better = cov > m[sl]
            u[sl] = np.where(better, ub, u[sl])
            e[sl] = np.where(better, np.clip(-best / (0.5 * wd), 0, 1), e[sl])
            m[sl] = np.maximum(m[sl], cov)
        if dry <= 0:
            self.wash(m, color, strength=strength, wetness=wetness, vary=0.12, granulate=granulate, staining=staining,
                      settle=2, name=name or 'strokes', hand=0)
            return
        # dry brush: straight onto the paper, no water to move it
        ks, g, _ = self._pigment(color, granulate, staining)
        streak = value_noise(H, W, 1.6 * H / 1000.0, self._trng)                     # hairs of the brush
        need = dry * (0.25 + 0.75 * u) + 0.10 * streak + 0.55 * dry * (1 - smoothstep(0.0, 0.6, e))      # the edges of a dry stroke break up
        a = m * smoothstep(need - 0.07, need + 0.07, self.tooth * (0.6 + 0.4 * self.grain) + 0.25 * (1 - dry)) \
            * np.asarray(strength, F32)
        self._deposit(a, ks if ks.ndim == 1 else np.stack([self._big(ks[..., b]) for b in range(NB)], -1), 0.0)
        self.n_ops += 1
        if self._log:
            print(f"  {name or 'dry brush':<18s} {len(lines):5d} strokes", flush=True)
        self._frame()

    def lift(self, mask, amount=0.6):
        """Lift dried paint off with a damp brush or a tissue (clouds, highlights). Not all of it
        comes away."""
        k = 1 - np.clip(self._field(mask), 0, 1) * amount * (0.75 + 0.25 * self.tooth)
        self.dep *= k[..., None]
        self.dep_s *= self._small(k)[..., None]
        self._frame()

    # ---- the water moving ----------------------------------------------------------------
    def _advect(self, a, ux, uy):
        """Carry `a` with the flow, conserving it: what leaves one cell arrives in the next."""
        fx = 0.5 * (ux[:, :-1] + ux[:, 1:])
        F = np.where(fx > 0, fx * a[:, :-1], fx * a[:, 1:])
        a[:, :-1] -= F
        a[:, 1:] += F
        fy = 0.5 * (uy[:-1] + uy[1:])
        F = np.where(fy > 0, fy * a[:-1], fy * a[1:])
        a[:-1] -= F
        a[1:] += F
        np.maximum(a, 0, out=a)

    def _step(self, n=1):
        px = self.px
        for _ in range(int(n)):
            w, d = self.w, self.d
            if w.max() <= 0.004:
                break
            here = smoothstep(0.02, 0.15, d)                         # paper the water can move through
            m = smoothstep(0.015, 0.22, w) * here                    # where pigment is afloat
            # water levels out through damp paper, never across dry
            w += 0.3 * (ndi.gaussian_filter(w, 2.2 * px, mode='nearest') - w)
            w += 0.10 * (blur(w, 9.0 * px) - w)        # and soaks outwards through the damp
            w *= here
            # it runs from deeper to shallower (towards a drying rim, out of a wetter touch), and down the board
            ws_ = ndi.gaussian_filter(w, 1.8 * px, mode='nearest') / np.maximum(ndi.gaussian_filter(here, 1.8 * px, mode='nearest'), 0.3)   # (dry paper beside a wash is a wall, not a slope)
            gy, gx = np.gradient(ws_)
            k = 130.0 * px * self.perm * m
            ux, uy = -k * gx, -k * gy + self.tilt * 0.10 * px * m * w
            sp = np.maximum(np.hypot(ux, uy), 1e-6)
            lim = np.minimum(1.0, 0.42 / sp)
            ux *= lim
            uy *= lim
            edge = 1 - ndi.gaussian_filter(m, 4.0 * px, mode='nearest')       # near a rim (or on its own): dries first
            mb = np.maximum(blur(m, 7.0 * px), 0.08)
            flood = np.clip((w - 0.5) * 1.1, 0, 0.4) if float(w.max()) > 0.55 else None      # very wet paper: colour travels far
            mb2 = np.maximum(blur(m, 20.0 * px), 0.08) if flood is not None else None
            sink = np.clip(1 - w / 0.5, 0, 1)
            for p in self.wet_paints:
                if p.a.max() <= 1e-5:
                    continue
                self._advect(p.a, ux, uy)
                # wet into wet: pigment wanders through deep water, and only through water
                wide = m * blur(p.a, 7.0 * px) / mb
                p.a += np.clip(w * w * 1.6, 0, 0.55) * self.perm.clip(0.6, 1.5) / 1.5 * (wide - p.a)
                if flood is not None:
                    p.a += flood * (m * blur(p.a, 20.0 * px) / mb2 - p.a)
                # it settles as the film thins; heavy pigments first
                s = np.clip((0.03 + 0.25 * sink * sink) * (1 + 0.8 * p.g), 0, 1) * p.a
                s = np.where(w < 0.02, p.a, s)
                p.a -= s
                p.s += s
            w -= 0.016 * (1 + 2.6 * edge)
            np.maximum(w, 0, out=w)
            d -= 0.006 * (w <= 0)
            np.maximum(d, 0, out=d)
            self._frame()

    def settle(self, steps=10):
        """Leave the water to move for a while without drying the sheet."""
        self._step(steps)

    def _deposit(self, a, ks, g):
        """Pigment `a` (H x W) of absorption `ks` comes to rest on the paper."""
        if g > 0:                                                    # heavy grains gather in the valleys and in flocs
            tex = 1 + g * self.grain * (1.25 * (0.5 - self._tooth_soft) + 0.14 * value_noise(self.H, self.W, 3.0 * self.H / 1000, self._trng, 2))
            a = a * np.clip(tex, 0.4, 2.2)
        else:
            a = a * (1 + 0.12 * self.grain * (0.5 - self.tooth))
        a = a.astype(F32)
        if ks.ndim == 1:
            for b in range(NB):
                self.dep[..., b] += a * ks[b]
        else:
            self.dep += a[..., None] * ks
        asm = self._small(a)
        self.dep_s += asm[..., None] * (ks if ks.ndim == 1 else np.stack([self._small(ks[..., b]) for b in range(NB)], -1))

    def dry(self):
        """Let the sheet dry (a hairdryer): the water finishes moving, every wash settles where
        it is, and what is painted next goes on dry paper with hard edges."""
        n = 0
        while self.w.max() > 0.004 and n < 120:
            self._step(1)
            n += 1
        for p in self.wet_paints:
            tot = p.a + p.s
            if tot.max() <= 1e-6:
                continue
            a = self._big(tot) * (p.crisp.astype(F32) / 255)
            ks = p.ks if p.ks.ndim == 1 else np.stack([self._big(p.ks[..., b]) for b in range(NB)], -1)
            self._deposit(a, ks, p.g)
        self.wet_paints = []
        self.w[:] = 0
        self.d[:] = 0
        self._frame()

    # ---- looking ---------------------------------------------------------------------------
    def _wet_ks(self):
        ks = self.dep_s.copy()
        for p in self.wet_paints:
            ks += ((p.a + p.s) * self._small(p.crisp.astype(F32) / 255))[..., None] * p.ks
        return ks

    def _frame(self):
        if self._frames is None:
            return
        rgb_ = q_to_rgb(np.sqrt(self._wet_ks() + self._ksp))
        rgb_ = rgb_ * (1 - 0.10 * np.clip(self.w, 0, 1)[..., None])            # wet paper is a little darker
        self._frames.append((np.clip(rgb_, 0, 1) * 255 + 0.5).astype(np.uint8))

    def render(self):
        """The sheet as sRGB (H x W x 3, 0..1). Paint that is still wet is shown where it is."""
        ks = self.dep
        if self.wet_paints:
            ks = ks.copy()
            for p in self.wet_paints:
                a = self._big(p.a + p.s) * (p.crisp.astype(F32) / 255)
                ks += a[..., None] * (p.ks if p.ks.ndim == 1 else np.stack([self._big(p.ks[..., b]) for b in range(NB)], -1))
        out = np.empty((self.H, self.W, 3), F32)
        for y0 in range(0, self.H, 256):                                 # in bands, to keep memory down
            out[y0:y0 + 256] = q_to_rgb(np.sqrt(ks[y0:y0 + 256] + self._ksp))
        # the sheet itself: a raking light on the tooth, barely there
        gy, gx = np.gradient(ndi.gaussian_filter(self.tooth, 0.8))
        out *= (1 + 0.3 * self.grain * (-0.6 * gx - 0.8 * gy))[..., None].clip(0.9, 1.08)
        return np.clip(out, 0, 1)

    def save(self, path):
        img = Image.fromarray((self.render() * 255 + 0.5).astype(np.uint8))
        if str(path).lower().endswith(('.jpg', '.jpeg')):
            img.save(path, quality=93, subsampling=0)
        else:
            img.save(path)
        if self._log:
            print(f"wrote {path}  {self.W}x{self.H}", flush=True)
        return path

    def film(self, path, seconds=12, fps=30, hold=2.0):
        """The sheet being painted, as an mp4 (construct the Paper with film=True)."""
        if not self._frames:
            raise RuntimeError("construct the Paper with film=True to record a film")
        fr = self._frames
        n = max(2, int(seconds * fps))
        idx = np.unique(np.linspace(0, len(fr) - 1, min(n, len(fr))).astype(int))
        h, w = fr[0].shape[:2]
        h2, w2 = h - h % 2, w - w % 2
        cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w2}x{h2}',
               '-r', str(fps * len(idx) / n), '-i', '-', '-vf', f'tpad=stop_mode=clone:stop_duration={hold}',
               '-r', str(fps), '-pix_fmt', 'yuv420p', '-crf', '18', '-movflags', '+faststart', path]
        pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for i in idx:
            pr.stdin.write(fr[i][:h2, :w2].tobytes())
        pr.stdin.close()
        pr.wait()
        if self._log:
            print(f"wrote {path}  ({len(idx)} frames)", flush=True)
        return path
