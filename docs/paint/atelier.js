/* atelier.js: the oil-paint simulator of skills/procedural-painting/scripts/atelier.py, in the
   browser. It replays a stroke tape (scripts/tape.py) on a canvas: 12-band Kubelka-Munk colour,
   bristle ribbons that run dry and catch on the weave, wet-into-wet pickup, paint relief under
   a raking light. No libraries. The numbers below are the Python engine's; keep them in step. */
(function (global) {
  'use strict';
  const NB = 12, RMIN = 0.02, T = 32;
  const BASIS = [
    [0.66480, 0.04914, 0.28606], [0.00000, 0.06231, 0.93769], [0.00000, 0.00000, 1.00000],
    [0.00000, 0.24246, 0.75754], [0.00000, 0.94994, 0.05006], [0.00000, 0.99791, 0.00209],
    [0.00000, 1.00000, 0.00000], [0.44173, 0.48132, 0.07695], [1.00000, 0.00000, 0.00000],
    [1.00000, 0.00000, 0.00000], [1.00000, 0.00000, 0.00000], [1.00000, 0.00000, 0.00000]];
  const SPD = [
    [0.007770, 0.048250, 0.009250, -0.101660, -0.205220, -0.157430, 0.152580, 0.469430, 0.489990, 0.230670, 0.051220, 0.005150],
    [-0.005760, -0.046640, -0.031610, 0.079620, 0.253700, 0.385080, 0.295600, 0.109210, -0.015230, -0.020980, -0.003400, 0.000420],
    [0.039770, 0.356880, 0.491830, 0.201430, 0.019300, -0.036250, -0.038860, -0.023070, -0.008130, -0.002120, -0.000630, -0.000160]];

  // spectrum -> linear RGB, with the rounding of the tables removed (FIX = inv(SPD . BASIS))
  const MR = new Float32Array(NB), MG = new Float32Array(NB), MB = new Float32Array(NB);
  (function () {
    const A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let b = 0; b < NB; b++) A[i][j] += SPD[i][b] * BASIS[b][j];
    const [a, b, c] = A[0], [d, e, f] = A[1], [g, h, k] = A[2];
    const det = a * (e * k - f * h) - b * (d * k - f * g) + c * (d * h - e * g);
    const inv = [[(e * k - f * h) / det, (c * h - b * k) / det, (b * f - c * e) / det],
                 [(f * g - d * k) / det, (a * k - c * g) / det, (c * d - a * f) / det],
                 [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det]];
    for (let bnd = 0; bnd < NB; bnd++) {
      MR[bnd] = (inv[0][0] * SPD[0][bnd] + inv[0][1] * SPD[1][bnd] + inv[0][2] * SPD[2][bnd]) / (1 - RMIN);
      MG[bnd] = (inv[1][0] * SPD[0][bnd] + inv[1][1] * SPD[1][bnd] + inv[1][2] * SPD[2][bnd]) / (1 - RMIN);
      MB[bnd] = (inv[2][0] * SPD[0][bnd] + inv[2][1] * SPD[1][bnd] + inv[2][2] * SPD[2][bnd]) / (1 - RMIN);
    }
  })();

  const S2L = new Float32Array(256);
  for (let i = 0; i < 256; i++) { const c = i / 255; S2L[i] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); }
  const L2S = new Uint8Array(4097);
  for (let i = 0; i <= 4096; i++) { const c = i / 4096; L2S[i] = Math.round(255 * (c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055)); }

  /** sRGB bytes -> paint state sqrt(K/S) in 12 bands, written to out[o..o+12). */
  function rgbToQ(r8, g8, b8, out, o) {
    const r = S2L[r8], g = S2L[g8], b = S2L[b8];
    for (let k = 0; k < NB; k++) {
      let R = r * BASIS[k][0] + g * BASIS[k][1] + b * BASIS[k][2];
      R = RMIN + (1 - RMIN) * (R < 0 ? 0 : R > 1 ? 1 : R);
      out[o + k] = (1 - R) / Math.sqrt(2 * R);
    }
  }

  function mulberry(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(rnd) { return Math.sqrt(-2 * Math.log(1 - rnd())) * Math.cos(6.283185307 * rnd()); }
  function normalise(a) {
    let m = 0, s = 0;
    for (let i = 0; i < a.length; i++) m += a[i];
    m /= a.length;
    for (let i = 0; i < a.length; i++) { a[i] -= m; s += a[i] * a[i]; }
    s = Math.sqrt(s / a.length) + 1e-8;
    for (let i = 0; i < a.length; i++) a[i] /= s;
    return a;
  }
  /** Smooth noise, zero mean and unit deviation, features about `scale` pixels across. */
  function valueNoise(W, H, scale, rnd, octaves) {
    const out = new Float32Array(W * H);
    let amp = 1, tot = 0;
    for (let o = 0; o < (octaves || 1); o++) {
      const sc = Math.max(scale / Math.pow(2, o), 1.0);
      const gw = Math.ceil(W / sc) + 3, gh = Math.ceil(H / sc) + 3;
      const g = new Float32Array(gw * gh);
      for (let i = 0; i < g.length; i++) g[i] = gauss(rnd);
      const n = new Float32Array(W * H);
      for (let y = 0; y < H; y++) {
        const fy = y / sc, y0 = fy | 0;
        let ty = fy - y0; ty = ty * ty * (3 - 2 * ty);
        for (let x = 0; x < W; x++) {
          const fx = x / sc, x0 = fx | 0;
          let tx = fx - x0; tx = tx * tx * (3 - 2 * tx);
          const i = y0 * gw + x0;
          const a = g[i] + (g[i + 1] - g[i]) * tx, b = g[i + gw] + (g[i + gw + 1] - g[i + gw]) * tx;
          n[y * W + x] = a + (b - a) * ty;
        }
      }
      normalise(n);
      for (let i = 0; i < n.length; i++) out[i] += amp * n[i];
      tot += amp * amp; amp *= 0.5;
    }
    const k = 1 / Math.sqrt(tot);
    for (let i = 0; i < out.length; i++) out[i] *= k;
    return out;
  }
  /** Gaussian blur of an n x n tile along one axis, wrapping. */
  function blurWrap(src, n, sigma, alongRows) {
    const r = Math.max(1, Math.ceil(sigma * 3)), ker = new Float32Array(2 * r + 1);
    let s = 0;
    for (let i = -r; i <= r; i++) { ker[i + r] = Math.exp(-0.5 * i * i / (sigma * sigma)); s += ker[i + r]; }
    const out = new Float32Array(n * n);
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      let v = 0;
      for (let i = -r; i <= r; i++) {
        const yy = alongRows ? (((y + i) % n) + n) % n : y, xx = alongRows ? x : (((x + i) % n) + n) % n;
        v += ker[i + r] * src[yy * n + xx];
      }
      out[y * n + x] = v / s;
    }
    return out;
  }
  function bristleTile(rnd) {
    const n = 256, a = new Float32Array(n * n), b = new Float32Array(n * n);
    for (let i = 0; i < n * n; i++) { a[i] = gauss(rnd); b[i] = gauss(rnd); }
    const t = blurWrap(blurWrap(a, n, 9.0, true), n, 0.6, false);        // long along the stroke, fine across
    const u = blurWrap(blurWrap(b, n, 25.0, true), n, 2.5, false);
    for (let i = 0; i < n * n; i++) t[i] += 3 * u[i];
    return normalise(t);
  }
  function canvasWeave(W, H, period, rnd) {
    const wx = valueNoise(W, H, 5 * period, rnd, 1), wy = valueNoise(W, H, 5 * period, rnd, 1);
    const slub = valueNoise(W, H, 2.4 * period, rnd, 1), h = new Float32Array(W * H);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const i = y * W + x;
      const fx = (x + 0.35 * period * wx[i]) / period, fy = (y + 0.35 * period * wy[i]) / period;
      const ix = Math.floor(fx), iy = Math.floor(fy), rx = fx - ix, ry = fy - iy;
      const sx = Math.sin(Math.PI * rx), sy = Math.sin(Math.PI * ry);
      const v = (((ix + iy) % 2) + 2) % 2 === 0 ? Math.pow(sx, 0.8) * (0.55 + 0.45 * sy) : Math.pow(sy, 0.8) * (0.55 + 0.45 * sx);
      h[i] = v * (1 + 0.2 * slub[i]) + 0.08 * gauss(rnd) * 0.5;
    }
    const samp = [];
    for (let i = 0; i < h.length; i += 7) samp.push(h[i]);
    samp.sort((p, q) => p - q);
    const lo = samp[Math.floor(samp.length * 0.01)], hi = samp[Math.floor(samp.length * 0.99)];
    for (let i = 0; i < h.length; i++) { const v = (h[i] - lo) / (hi - lo); h[i] = v < 0 ? 0 : v > 1 ? 1 : v; }
    return h;
  }
  function ss(a, b, x) { let t = (x - a) / (b - a); t = t < 0 ? 0 : t > 1 ? 1 : t; return t * t * (3 - 2 * t); }

  /** Fetch and unpack a tape written by scripts/tape.py. */
  async function loadTape(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error('could not load ' + url + ' (' + res.status + ')');
    if (typeof DecompressionStream === 'undefined') throw new Error('this browser cannot unpack the tape (no DecompressionStream)');
    const buf = await new Response(res.body.pipeThrough(new DecompressionStream('deflate'))).arrayBuffer();
    return parseTape(buf);
  }
  function parseTape(buf) {
    const dv = new DataView(buf);
    if (dv.getUint32(0, true) !== 0x314c5441) throw new Error('not a stroke tape');
    const n = dv.getUint32(4, true);
    const head = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, n)));
    const base = 8 + n;
    const cut = (off, bytes) => buf.slice(base + off, base + off + bytes);
    let first = 0;
    const weights = [];
    for (const op of head.ops) {
      if (op.op === 'glaze') {
        op.mask = new Uint8Array(cut(op.off, op.w * op.h));
        op.color = new Uint8Array(cut(op.off + op.w * op.h, op.w * op.h * 3));
      }
      if (op.op !== 'strokes') continue;
      let o = op.off;
      const m = op.n;
      op.x0 = new Int16Array(cut(o, 2 * m)); o += 2 * m;
      op.y0 = new Int16Array(cut(o, 2 * m)); o += 2 * m;
      op.width = new Uint16Array(cut(o, 2 * m)); o += 2 * m;
      op.length = new Uint16Array(cut(o, 2 * m)); o += 2 * m;
      op.count = new Uint16Array(cut(o, 2 * m)); o += 2 * m;
      op.c1 = new Uint8Array(cut(o, 3 * m)); o += 3 * m;
      op.c2 = new Uint8Array(cut(o, 3 * m)); o += 3 * m;
      op.runout = new Uint8Array(cut(o, m)); o += m;
      op.offs = new Uint8Array(cut(o, 4 * m)); o += 4 * m;
      op.delta = new Int8Array(cut(o, 2 * op.points));
      if (op.mask) op.mask.data = new Uint8Array(cut(op.mask.off, op.mask.w * op.mask.h));
      if (op.seed !== undefined) op.seedC = new Uint8Array(cut(op.seed, 3 * m));       // what the study said where the stroke read it
      if (op.at !== undefined) op.atXY = new Int16Array(cut(op.at, 4 * m));
      op.start = new Uint32Array(m);                       // where each stroke's steps begin
      let acc = 0;
      for (let i = 0; i < m; i++) { op.start[i] = acc; acc += op.count[i] - 1; }
      op.first = first; first += m;
      for (let i = 0; i < m; i++) weights.push(Math.sqrt((op.width[i] / 262144) * (op.length[i] / 16384)) + 0.004);
    }
    // the clock of the film: a big stroke takes longer to pull than a small one
    head.clock = new Float64Array(weights.length + 1);
    for (let i = 0; i < weights.length; i++) head.clock[i + 1] = head.clock[i] + weights[i];
    return head;
  }

  /** The light sheet of a moving painting: n small studies, first to last moment of the film. */
  async function loadLight(url, info) {
    const im = new Image();
    im.src = url;
    await im.decode();
    const c = document.createElement('canvas');
    c.width = im.naturalWidth; c.height = im.naturalHeight;
    const g = c.getContext('2d', { willReadFrequently: true });
    g.drawImage(im, 0, 0);
    const frames = [];
    for (let j = 0; j < info.n; j++) frames.push(g.getImageData((j % info.cols) * info.w, ((j / info.cols) | 0) * info.h, info.w, info.h).data);
    return { n: info.n, w: info.w, h: info.h, frames };
  }

  class Painting {
    /** A canvas `height` pixels high that will lay `tape`. */
    constructor(tape, height) {
      this.tape = tape;
      const H = this.H = Math.round(height), W = this.W = Math.round(height * tape.aspect);
      this.px = H / 1000;
      const n = W * H;
      this.q = new Float32Array(n * NB); this.sq = new Float32Array(n * NB);
      this.h = new Float32Array(n); this.sh = new Float32Array(n);
      this.wet = new Float32Array(n); this.swet = new Float32Array(n);
      this.lin = new Float32Array(n * 3);
      this.hgt = new Float32Array(n); this.occ = new Float32Array(n);
      this.owner = new Int32Array(n).fill(-1); this.zu = new Uint16Array(n); this.zv = new Uint16Array(n);
      this.list = new Int32Array(n);
      this.tw = Math.ceil(W / T); this.th = Math.ceil(H / T);
      this.dirty = new Uint8Array(this.tw * this.th);      // tiles to relight
      this.stale = new Uint8Array(this.tw * this.th);      // tiles whose snapshot is out of date
      const rnd = mulberry(20261116);
      this.tooth = canvasWeave(W, H, 3.6 * this.px, rnd);
      this.btile = bristleTile(rnd);
      this.scrub = valueNoise(W, H, 60 * this.px, rnd, 3);   // a toned ground is scrubbed on, never flat
      this.img = new ImageData(W, H);
      this.light = [-0.55, -0.62]; this.relief = 1; this.gloss = 0.35;
      this.study = null;                                    // set by setMoment for a moving painting
      this.pathX = new Float32Array(1024); this.pathY = new Float32Array(1024);
      this.reset();
    }

    /** Paint by the light of moment t (0..1) of a moving painting; null = as it was recorded.
        Takes effect for strokes laid from now on: reset() to lay the whole canvas in that light. */
    setMoment(light, t) {
      if (!light || t === null) { this.study = null; return; }
      const f = Math.min(Math.max(t, 0), 1) * (light.n - 1), i0 = Math.min(light.n - 2, f | 0), k = f - i0;
      const a = light.frames[i0], b = light.frames[i0 + 1], n = light.w * light.h;
      if (!this.study || this.study.length !== n * 3) this.study = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) this.study[i * 3 + c] = a[i * 4 + c] * (1 - k) + b[i * 4 + c] * k;
      this.sw = light.w; this.sh2 = light.h;
    }

    reset() {
      const { W, H, q, tape } = this, n = W * H, g = new Float32Array(NB);
      rgbToQ(Math.round(tape.ground[0] * 255), Math.round(tape.ground[1] * 255), Math.round(tape.ground[2] * 255), g, 0);
      for (let i = 0; i < n; i++) { const k = 1 + 0.05 * this.scrub[i]; for (let b = 0; b < NB; b++) q[i * NB + b] = g[b] * k; }
      this.h.fill(0); this.wet.fill(0);
      for (let i = 0; i < n; i++) this._lin(i);
      this.opI = 0; this.sitI = 0; this.sitLeft = -1; this.strokeI = 0; this.done = 0; this.serial = 0;
      this.owner.fill(-1); this.stale.fill(1); this.dirty.fill(1);
      this.finished = false;
      if (tape.strokes === 0) this._drain();
    }

    _lin(i) {
      const q = this.q, o = i * NB;
      let r = 0, g = 0, b = 0;
      for (let k = 0; k < NB; k++) {
        const v = q[o + k], ks = v * v, R = 1 + ks - Math.sqrt(ks * ks + 2 * ks) - RMIN;
        r += MR[k] * R; g += MG[k] * R; b += MB[k] * R;
      }
      const l = this.lin, j = i * 3;
      l[j] = r; l[j + 1] = g; l[j + 2] = b;
    }

    /** Lay strokes until the film clock reaches `until` (0..1) or `ms` have passed. Returns true
        when the target was reached. */
    advance(until, ms) {
      const t0 = performance.now(), clock = this.tape.clock, total = clock[clock.length - 1];
      while (!this.finished && clock[this.done] < until * total) {
        this._next();
        if (performance.now() - t0 > ms) return false;
      }
      return true;
    }
    get progress() { const c = this.tape.clock; return c[this.done] / c[c.length - 1]; }

    _next() {
      const ops = this.tape.ops;
      for (;;) {
        if (this.opI >= ops.length) { this.finished = true; return; }
        const op = ops[this.opI];
        if (op.op === 'dry') { const k = 1 - op.amount, w = this.wet; for (let i = 0; i < w.length; i++) w[i] *= k; this.stale.fill(1); this.opI++; continue; }
        if (op.op === 'glaze') { this._glaze(op); this.opI++; continue; }
        if (this.sitLeft < 0) { this.sitI = 0; this.strokeI = 0; this.sitLeft = 0; }
        if (this.sitLeft === 0) {
          if (this.sitI >= op.sittings.length) { this.opI++; this.sitLeft = -1; continue; }
          this.sitLeft = op.sittings[this.sitI++];
          this._snapshot();
          if (this.sitLeft === 0) continue;
        }
        this._stroke(op, this.strokeI++);
        this.sitLeft--; this.done++;
        if (this.done >= this.tape.strokes) this._drain();
        return;
      }
    }

    /** After the last stroke: apply whatever follows it on the tape (a final glaze, a dry). */
    _drain() {
      const ops = this.tape.ops;
      for (; this.opI < ops.length; this.opI++) {
        const op = ops[this.opI];
        if (op.op === 'dry') { const k = 1 - op.amount, w = this.wet; for (let i = 0; i < w.length; i++) w[i] *= k; }
        else if (op.op === 'glaze') this._glaze(op);
      }
      this.stale.fill(1);
      this.finished = true;
    }

    /** Every stroke of a sitting sees the canvas as it was when the sitting began. */
    _snapshot() {
      const { W, H, tw, th, stale } = this;
      for (let ty = 0; ty < th; ty++) for (let tx = 0; tx < tw; tx++) {
        if (!stale[ty * tw + tx]) continue;
        stale[ty * tw + tx] = 0;
        const x0 = tx * T, x1 = Math.min(W, x0 + T), y1 = Math.min(H, ty * T + T);
        for (let y = ty * T; y < y1; y++) {
          const a = y * W + x0, b = y * W + x1;
          this.sq.set(this.q.subarray(a * NB, b * NB), a * NB);
          this.sh.set(this.h.subarray(a, b), a);
          this.swet.set(this.wet.subarray(a, b), a);
        }
      }
    }

    _touch(x0, y0, x1, y1, pad) {
      const { tw, th } = this;
      const a = Math.max(0, ((x0 - pad) / T) | 0), b = Math.min(tw - 1, ((x1 + pad) / T) | 0);
      const c = Math.max(0, ((y0 - pad) / T) | 0), d = Math.min(th - 1, ((y1 + pad) / T) | 0);
      for (let ty = c; ty <= d; ty++) for (let tx = a; tx <= b; tx++) { this.dirty[ty * tw + tx] = 1; this.stale[ty * tw + tx] = 1; }
    }

    _glaze(op) {
      const { W, H, q } = this, g = new Float32Array(NB);
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const j = Math.min(op.h - 1, (y / H * op.h) | 0) * op.w + Math.min(op.w - 1, (x / W * op.w) | 0);
        const m = op.mask[j] / 255;
        if (m <= 0) continue;
        rgbToQ(op.color[j * 3], op.color[j * 3 + 1], op.color[j * 3 + 2], g, 0);
        const i = y * W + x;
        for (let b = 0; b < NB; b++) { const v = q[i * NB + b]; q[i * NB + b] = Math.sqrt(v * v + m * g[b] * g[b]); }
        this._lin(i);
      }
      this.dirty.fill(1); this.stale.fill(1);
    }

    _stroke(op, i) {
      const { W, H, px, q, sq, h, sh, wet, swet, owner, zu, zv, list, btile, tooth } = this;
      const P = op.params, shape = P.shape === 'round' ? 1 : (P.shape === 'knife' || P.knife) ? 2 : 0;
      const w = Math.max(op.width[i] / 262144 * H, 1.3), L = Math.max(op.length[i] / 16384 * H, 1.0), step = op.step * H;
      const K = op.count[i];
      if (K > this.pathX.length) { this.pathX = new Float32Array(K * 2); this.pathY = new Float32Array(K * 2); }
      const PX = this.pathX, PY = this.pathY, dl = op.delta, ds = step / 100;
      PX[0] = op.x0[i] / 8192 * H; PY[0] = op.y0[i] / 8192 * H;
      for (let k = 1, d = op.start[i] * 2; k < K; k++, d += 2) { PX[k] = PX[k - 1] + dl[d] * ds; PY[k] = PY[k - 1] + dl[d + 1] * ds; }
      const fmax = K - 1.001, taper = P.taper;
      const o0 = op.offs[i * 4] / 256, o1 = op.offs[i * 4 + 1] / 256, o2 = op.offs[i * 4 + 2] / 256, o3 = op.offs[i * 4 + 3] / 256;
      let ax = 0, ay = 0, nx = 0, ny = 0;
      const path = (u) => {
        let f = u / step; f = f < 0 ? 0 : f > fmax ? fmax : f;
        const i0 = f | 0, t = f - i0, dx = PX[i0 + 1] - PX[i0], dy = PY[i0 + 1] - PY[i0], inv = 1 / (Math.sqrt(dx * dx + dy * dy) + 1e-9);
        ax = PX[i0] + dx * t; ay = PY[i0] + dy * t; nx = -dy * inv; ny = dx * inv;
      };
      const profile = (u) => {
        if (shape === 1) { const v = 1 - (2 * u / L - 1) * (2 * u / L - 1); return v > 0 ? 0.5 * Math.sqrt(v) : 0; }
        if (shape === 2) {
          let a = u / (0.05 * w), b = (L - u) / (0.05 * w);
          a = a < 0 ? 0 : a > 1 ? 1 : a; b = b < 0 ? 0 : b > 1 ? 1 : b;
          return 0.5 * Math.sqrt(a * b);
        }
        let cap = u / (0.16 * w); cap = cap < 0 ? 0 : cap > 1 ? 1 : cap;
        let end = (L - u) / (0.22 * w); end = end < 0 ? 0 : end > 1 ? 1 : end;
        return 0.5 * Math.pow(cap, 0.35) * (1 - taper * Math.pow(ss(0.55, 1.0, u / L), 1.5)) * Math.sqrt(end);
      };
      const tile = (a, b) => btile[((((a * 256) | 0) & 255) << 8) | (((b * 256) | 0) & 255)];

      // the ribbon: which pixels the stroke owns, and where on the stroke each one lies
      const du = 0.75, nu = Math.ceil(L / du) + 1, nv = Math.ceil(w / 0.6) + 1, serial = this.serial++;
      let n = 0, bx0 = W, by0 = H, bx1 = -1, by1 = -1;
      for (let iu = 0; iu < nu; iu++) {
        const u = iu * du;
        if (u > L) break;
        const half = profile(u) * (1 - 0.10 * Math.abs(tile(u / (w * 1.2) + o0, o1)));
        if (half <= 0) continue;
        path(u);
        const lo = Math.max(0, Math.ceil((0.5 - half) * (nv - 1))), hi = Math.min(nv - 1, Math.floor((0.5 + half) * (nv - 1)));
        for (let iv = lo; iv <= hi; iv++) {
          const vw = (iv / (nv - 1) - 0.5) * w, x = Math.round(ax + vw * nx), y = Math.round(ay + vw * ny);
          if (x < 0 || x >= W || y < 0 || y >= H) continue;
          const id = y * W + x;
          if (owner[id] !== serial) {
            owner[id] = serial; list[n++] = id;
            if (x < bx0) bx0 = x; if (x > bx1) bx1 = x; if (y < by0) by0 = y; if (y > by1) by1 = y;
          }
          zu[id] = iu; zv[id] = iv;
        }
      }
      if (n === 0) return;
      this._touch(bx0, by0, bx1, by1, this._pad);

      // the paint: what each of those pixels gets
      const q1 = Painting._q1, q2 = Painting._q2, qc = Painting._qc;
      let lit = 1;
      if (this.study && P.field && op.seedC) {
        // the same stroke in another light: the study's colour here now, plus this stroke's own departure from it
        const S = this.study, sw = this.sw, sh_ = this.sh2;
        let fx = (op.atXY ? op.atXY[i * 2] : op.x0[i]) / 8192 * sh_ - 0.5, fy = (op.atXY ? op.atXY[i * 2 + 1] : op.y0[i]) / 8192 * sh_ - 0.5;
        fx = fx < 0 ? 0 : fx > sw - 1.001 ? sw - 1.001 : fx; fy = fy < 0 ? 0 : fy > sh_ - 1.001 ? sh_ - 1.001 : fy;
        const ix = fx | 0, iy = fy | 0, tx = fx - ix, ty = fy - iy, o = (iy * sw + ix) * 3;
        const now = Painting._now;
        for (let c = 0; c < 3; c++) now[c] = (S[o + c] * (1 - tx) + S[o + 3 + c] * tx) * (1 - ty) + (S[o + sw * 3 + c] * (1 - tx) + S[o + sw * 3 + 3 + c] * tx) * ty;
        const w0 = op.seedC[i * 3], w1 = op.seedC[i * 3 + 1], w2 = op.seedC[i * 3 + 2];
        const ln = 0.299 * now[0] + 0.587 * now[1] + 0.114 * now[2], lw = 0.299 * w0 + 0.587 * w1 + 0.114 * w2;
        let g = (ln + 20.4) / (lw + 20.4); g = g < 0.3 ? 0.3 : g > 1.6 ? 1.6 : g;
        const cl = (v) => (v < 0 ? 0 : v > 255 ? 255 : Math.round(v));
        rgbToQ(cl(now[0] + (op.c1[i * 3] - w0) * g), cl(now[1] + (op.c1[i * 3 + 1] - w1) * g), cl(now[2] + (op.c1[i * 3 + 2] - w2) * g), q1, 0);
        rgbToQ(cl(now[0] + (op.c2[i * 3] - w0) * g), cl(now[1] + (op.c2[i * 3 + 1] - w1) * g), cl(now[2] + (op.c2[i * 3 + 2] - w2) * g), q2, 0);
        if (P.fade) lit = ss(0.35, 0.8, (ln + 7.65) / (lw + 7.65));        // a lamp is painted only while it is lit
      } else {
        rgbToQ(op.c1[i * 3], op.c1[i * 3 + 1], op.c1[i * 3 + 2], q1, 0);
        rgbToQ(op.c2[i * 3], op.c2[i * 3 + 1], op.c2[i * 3 + 2], q2, 0);
      }
      if (lit <= 0) {                                       // nothing to lay: give the pixels back untouched
        for (let j = 0; j < n; j++) { const id = list[j], o = id * NB; for (let k = 0; k < NB; k++) q[o + k] = sq[o + k]; h[id] = sh[id]; wet[id] = swet[id]; this._lin(id); }
        return;
      }
      const run = op.runout[i] / 255, load = P.load, opacity = P.opacity * lit, pickup = P.pickup, dry = P.dry, marble = P.marble;
      const clip = P.clip, soft = P.soft, mask = op.mask, bpx = 1.5 * Math.max(px, 0.6);
      const body = P.thickness * lit * px * (0.55 + 0.45 * Math.sqrt(w / (30 * px)));
      const st = Math.max(1, Math.floor(this.tape.height / 256)) * H / this.tape.height;   // one mask cell, in our pixels
      for (let j = 0; j < n; j++) {
        const id = list[j], uu = zu[id] * du, vv = zv[id] / (nv - 1) - 0.5, un = uu / L;
        const ac = (vv + 0.5) * w / (bpx * 256), al = uu / (bpx * 768);
        let b1 = tile(al + o0, ac + o1);
        const b2 = tile(al * 0.6 + o2, ac + o3), b3 = tile(al * 0.25 + o1, ac * 0.45 + o2);
        const edge = profile(uu) - Math.abs(vv), th = tooth[id];
        let amt, a;
        if (shape === 2) {
          amt = load * (1 - 0.45 * run * un * un) * (1 + 0.12 * b3);
          const skip = ss(0.35, 1.1, b2 * 0.8 + (1 - th) - 0.35) * (0.25 + 2.2 * dry + 0.6 * un);
          a = opacity * ss(-0.06, 0.06, amt - 0.3 - skip) * ss(0, Math.max(0.25 * soft, 1.0 / w), edge);
          b1 = 0.3 * b3;
        } else if (shape === 1) {
          amt = load * (1 + 0.25 * b1);
          a = opacity * ss(-0.18, 0.18, amt - 0.3 - (1 - th) * dry * 1.5) * ss(0, Math.max(soft, 1.2 / w), edge);
        } else {
          amt = load * (1 - run * Math.pow(un, 1.5)) * (1 + 0.42 * b1);
          const need = (1 - th) * (dry + 0.35 * un * un) * 1.5;
          a = opacity * ss(need - 0.18, need + 0.18, amt - 0.3) * ss(0, Math.max(soft, 1.2 / w), edge);
        }
        const x = id % W, y = (id / W) | 0;
        if (clip > 0 && mask) {
          let fx = x / st, fy = y / st;
          fx = fx > mask.w - 1.001 ? mask.w - 1.001 : fx; fy = fy > mask.h - 1.001 ? mask.h - 1.001 : fy;
          const ix = fx | 0, iy = fy | 0, tx = fx - ix, ty = fy - iy, m = mask.data, o = iy * mask.w + ix;
          const mv = ((m[o] * (1 - tx) + m[o + 1] * tx) * (1 - ty) + (m[o + mask.w] * (1 - tx) + m[o + mask.w + 1] * tx) * ty) / 255;
          a *= 1 - clip + clip * mv;
        }
        const o = id * NB;
        if (a <= 0) {
          for (let k = 0; k < NB; k++) q[o + k] = sq[o + k];
          h[id] = sh[id]; wet[id] = swet[id];
          this._lin(id);
          continue;
        }
        const mixc = marble * (shape === 2 ? ss(0.0, 0.25, b3) : ss(-0.3, 1.3, b3));
        let m = 0;
        if (pickup > 0) {
          // wet-into-wet: the bristle has been dragging through the paint behind it
          let wc = 0;
          for (let k = 0; k < NB; k++) qc[k] = 0;
          for (let lb = 0; lb < 3; lb++) {
            const back = lb === 0 ? 0.10 : lb === 1 ? 0.28 : 0.6, wt = lb === 0 ? 0.45 : lb === 1 ? 0.35 : 0.20;
            const ub = uu - back * L - 1.0;
            path(ub > 0 ? ub : 0);
            let bx = Math.round(ax + vv * w * nx), by = Math.round(ay + vv * w * ny);
            bx = bx < 0 ? 0 : bx > W - 1 ? W - 1 : bx; by = by < 0 ? 0 : by > H - 1 ? H - 1 : by;
            const bi = by * W + bx, bo = bi * NB;
            for (let k = 0; k < NB; k++) qc[k] += wt * sq[bo + k];
            wc += wt * swet[bi];
          }
          let bb = 0.75 + 0.6 * b2; bb = bb < 0.1 ? 0.1 : bb > 1.6 ? 1.6 : bb;
          m = pickup * wc * (1 - Math.exp(-uu / (0.7 * w))) * bb;
          m = m < 0 ? 0 : m > 0.92 ? 0.92 : m;
        }
        for (let k = 0; k < NB; k++) {
          let qs = q1[k] + (q2[k] - q1[k]) * mixc;
          if (m > 0) qs = qs * (1 - m) + qc[k] * m;
          q[o + k] = sq[o + k] * (1 - a) + qs * a;
        }
        const am = amt < 0 ? 0 : amt > 2 ? 2 : amt;
        let t, flatten;
        if (shape === 2) {
          const e = (edge - 0.02) / 0.03;
          t = body * a * (am + 0.1 * b3 + 1.3 * Math.exp(-e * e) + 0.9 * ss(0.86, 0.98, un)); flatten = 0.92;
        } else if (shape === 1) {
          let e = edge / 0.5; e = e < 0 ? 0 : e > 1 ? 1 : e;
          t = body * a * (0.4 + 1.1 * Math.pow(e, 0.6) + 0.2 * b1 + 0.5 * ss(0.5, 0.9, un)); flatten = 0.8;
        } else {
          const e = (edge - 0.05) / 0.06;
          t = body * a * (0.5 * am + 0.22 * b1 + (un > 0.03 ? 0.45 * Math.exp(-e * e) : 0) + (un < 0.97 ? 0.35 * un : 0)); flatten = 0.65;
        }
        h[id] = sh[id] * (1 - flatten * a) + (t > 0 ? t : 0);
        wet[id] = swet[id] * (1 - a) + a;
        this._lin(id);
      }
    }

    get _pad() { return Math.ceil(this._occR) + 3; }
    get _occR() { return Math.max(2, Math.round(3.0 * this.px * 1.6)); }    // box radius matching a 3 px blur

    /** Light the relief where the paint changed (or everywhere) and draw it. `all` = relight
        everything, for a new light direction. Returns the rectangle drawn, or null. */
    paint(ctx, all) {
      const { W, H, tw, th, dirty, h, hgt, occ, tooth, lin, px } = this;
      if (all === 'light') { this._shade(0, 0, W, H); ctx.putImageData(this.img, 0, 0); return [0, 0, W, H]; }
      let X0 = W, Y0 = H, X1 = 0, Y1 = 0, any = false;
      const c = this.tape.weave * 0.55 * px, ih = 1 / (1.2 * px), R = this._occR, n = 2 * R + 1;
      // 1. height of the surface: paint softened a little, plus the weave where paint is thin
      for (let ty = 0; ty < th; ty++) for (let tx = 0; tx < tw; tx++) {
        if (!all && !dirty[ty * tw + tx]) continue;
        const x0 = tx * T, x1 = Math.min(W, x0 + T), y0 = ty * T, y1 = Math.min(H, y0 + T);
        for (let y = y0; y < y1; y++) {
          const ym = y > 0 ? y - 1 : 0, yp = y < H - 1 ? y + 1 : y;
          for (let x = x0; x < x1; x++) {
            const xm = x > 0 ? x - 1 : 0, xp = x < W - 1 ? x + 1 : x;
            const b = (h[ym * W + xm] + h[ym * W + xp] + h[yp * W + xm] + h[yp * W + xp]) * 0.0625
              + (h[ym * W + x] + h[yp * W + x] + h[y * W + xm] + h[y * W + xp]) * 0.125 + h[y * W + x] * 0.25;
            hgt[y * W + x] = b + c * tooth[y * W + x] * Math.exp(-h[y * W + x] * ih);
          }
        }
      }
      // 2. cavities: the surface against its own blur
      const row = this._row || (this._row = new Float32Array((T + 2 * 16) * (T + 2 * 16)));
      for (let ty = 0; ty < th; ty++) for (let tx = 0; tx < tw; tx++) {
        if (!all && !dirty[ty * tw + tx]) continue;
        any = true;
        const x0 = tx * T, x1 = Math.min(W, x0 + T), y0 = ty * T, y1 = Math.min(H, y0 + T), bw = x1 - x0;
        if (x0 < X0) X0 = x0; if (x1 > X1) X1 = x1; if (y0 < Y0) Y0 = y0; if (y1 > Y1) Y1 = y1;
        for (let y = y0 - R; y < y1 + R; y++) {                 // horizontal box sums, rows clamped
          const yy = y < 0 ? 0 : y > H - 1 ? H - 1 : y, ro = (y - y0 + R) * bw;
          let s = 0;
          for (let k = -R; k <= R; k++) { const xx = x0 + k; s += hgt[yy * W + (xx < 0 ? 0 : xx > W - 1 ? W - 1 : xx)]; }
          for (let x = x0; x < x1; x++) {
            row[ro + x - x0] = s;
            const xa = x - R, xb = x + R + 1;
            s += hgt[yy * W + (xb > W - 1 ? W - 1 : xb)] - hgt[yy * W + (xa < 0 ? 0 : xa)];
          }
        }
        for (let x = x0; x < x1; x++) {
          let s = 0;
          for (let k = 0; k < n; k++) s += row[k * bw + x - x0];
          for (let y = y0; y < y1; y++) {
            let v = (hgt[y * W + x] - s / (n * n)) / (1.5 * px);
            occ[y * W + x] = v < -1 ? -1 : v > 1 ? 1 : v;
            s += row[(y - y0 + n) * bw + x - x0] - row[(y - y0) * bw + x - x0];
          }
        }
      }
      if (!any) return null;
      // 3. the light
      for (let ty = 0; ty < th; ty++) for (let tx = 0; tx < tw; tx++) {
        if (!all && !dirty[ty * tw + tx]) continue;
        dirty[ty * tw + tx] = 0;
        this._shade(tx * T, ty * T, Math.min(W, tx * T + T), Math.min(H, ty * T + T));
      }
      ctx.putImageData(this.img, 0, 0, X0, Y0, X1 - X0, Y1 - Y0);
      return [X0, Y0, X1, Y1];
    }

    _shade(x0, y0, x1, y1) {
      const { W, H, hgt, occ, h, lin, px, relief } = this, d = this.img.data;
      let lx = this.light[0], ly = this.light[1], lz = 0.62;
      const ln = Math.sqrt(lx * lx + ly * ly + lz * lz); lx /= ln; ly /= ln; lz /= ln;
      let hx = lx, hy = ly, hz = lz + 1;
      const hn = Math.sqrt(hx * hx + hy * hy + hz * hz); hx /= hn; hy /= hn; hz /= hn;
      const k = 1.15 * relief, base = Math.pow(hz, 60), gl = this.gloss, it = 1 / (0.6 * px);
      for (let y = y0; y < y1; y++) {
        const ym = y > 0 ? y - 1 : y, yp = y < H - 1 ? y + 1 : y, ky = k / (yp - ym || 1);
        for (let x = x0; x < x1; x++) {
          const i = y * W + x, xm = x > 0 ? x - 1 : x, xp = x < W - 1 ? x + 1 : x;
          let r = lin[i * 3], g = lin[i * 3 + 1], b = lin[i * 3 + 2];
          if (relief > 0) {
            const gx = (hgt[y * W + xp] - hgt[y * W + xm]) * k / (xp - xm || 1), gy = (hgt[yp * W + x] - hgt[ym * W + x]) * ky;
            const nz = 1 / Math.sqrt(1 + gx * gx + gy * gy), nx = -gx * nz, ny = -gy * nz;
            let sh = 1 + 0.42 * ((nx * lx + ny * ly + nz * lz) / lz - 1) + 0.06 * occ[i] * relief;
            sh = sh < 0.35 ? 0.35 : sh > 1.9 ? 1.9 : sh;
            let nh = nx * hx + ny * hy + nz * hz; nh = nh < 0 ? 0 : nh > 1 ? 1 : nh;
            let tk = h[i] * it; tk = tk > 1 ? 1 : tk;
            let sp = nh > 0.9 ? gl * (Math.pow(nh, 60) - base) * tk : -gl * base * tk;
            sp = 0.5 * (sp < -0.02 ? -0.02 : sp);
            r = r * sh + sp; g = g * sh + sp; b = b * sh + sp;
          }
          const o = i * 4;
          d[o] = L2S[r <= 0 ? 0 : r >= 1 ? 4096 : (r * 4096 + 0.5) | 0];
          d[o + 1] = L2S[g <= 0 ? 0 : g >= 1 ? 4096 : (g * 4096 + 0.5) | 0];
          d[o + 2] = L2S[b <= 0 ? 0 : b >= 1 ? 4096 : (b * 4096 + 0.5) | 0];
          d[o + 3] = 255;
        }
      }
    }
  }
  Painting._q1 = new Float32Array(NB); Painting._q2 = new Float32Array(NB); Painting._qc = new Float32Array(NB); Painting._now = new Float32Array(3);

  global.Atelier = { Painting, loadTape, parseTape, loadLight };
})(typeof self !== 'undefined' ? self : this);
