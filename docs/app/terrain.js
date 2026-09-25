/* terrain.js: real elevation for any point on Earth, measured and framed in the browser.

   This is the JavaScript twin of the Python side of skills/procedural-terrain-art:
     dem.py            tile maths, terrarium decode (R*256 + G + B/256 - 32768 m), void filling
     terrain_art.py    summit refinement, analyse_shape(), terrain() auto values, water_mask()
     build_studio.py   the 512 x 512 east/north grid and the CONFIG object the engine reads
   Keep the two in step. The engine itself (six styles, camera) is not here: it is extracted from
   assets/studio_template.html into engine.js by scripts/build-app.py.

   Difference from Python: instead of one ~200-tile mosaic at z10-z12, the page fetches two small ones.
   Level A (about 9-16 tiles, z8-z10) covers the 65 km the landform analysis needs; level B (at most 25 tiles)
   covers the drawn grid at about 1.5 source pixels per grid sample. Sampling uses the finest level that
   covers a point. */
"use strict";
const Terrain = (() => {
  const TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png";
  const KY = 110.57, REF_H = 4.3, REF_L = 46, V0 = -35, V1 = 30, ZCAM = -95, D2R = Math.PI / 180;
  const RADIUS_A = 45 * 1.45;              // what render.py / build_studio.py fetch by default (km)
  const kxAt = lat => 111.32 * Math.cos(lat * D2R);
  const clamp = (v, a, b) => v < a ? a : v > b ? b : v;
  const smooth01 = (x, lo, hi) => { const t = clamp((x - lo) / (hi - lo), 0, 1); return t * t * (3 - 2 * t); };
  const round = (v, d) => { const m = 10 ** d; return Math.round(v * m) / m; };

  /* ---------------- tiles (dem.py) ---------------- */
  function tileXY(lat, lon, z){ const n = 2 ** z; return [(lon + 180) / 360 * n, (1 - Math.asinh(Math.tan(lat * D2R)) / Math.PI) / 2 * n]; }
  function tileRange(lat, lon, rkm, z){
    const dlat = rkm / KY, dlon = rkm / kxAt(lat), n = 2 ** z;
    const [x0, y0] = tileXY(Math.min(85, lat + dlat), lon - dlon, z), [x1, y1] = tileXY(Math.max(-85, lat - dlat), lon + dlon, z);
    const r = { z, tx0: Math.floor(x0), tx1: Math.floor(x1), ty0: clamp(Math.floor(y0), 0, n - 1), ty1: clamp(Math.floor(y1), 0, n - 1) };
    r.count = (r.tx1 - r.tx0 + 1) * (r.ty1 - r.ty0 + 1); return r;
  }
  const pxKm = (lat, z) => 40075.016 * Math.cos(lat * D2R) / (256 * 2 ** z);   // ground size of one tile pixel

  const tileCache = new Map();             // url -> Promise<Float32Array | null>
  let decodeCanvas = null;
  function decodeCtx(){
    if (!decodeCanvas) decodeCanvas = typeof OffscreenCanvas !== "undefined" ? new OffscreenCanvas(256, 256) : Object.assign(document.createElement("canvas"), { width: 256, height: 256 });
    return decodeCanvas.getContext("2d", { willReadFrequently: true });
  }
  async function decodePNG(blob){
    let src;
    try { src = await createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none" }); }
    catch { src = await new Promise((ok, bad) => { const im = new Image(); im.onload = () => ok(im); im.onerror = bad; im.src = URL.createObjectURL(blob); }); }
    const ctx = decodeCtx(); ctx.clearRect(0, 0, 256, 256); ctx.drawImage(src, 0, 0);
    const d = ctx.getImageData(0, 0, 256, 256).data, e = new Float32Array(65536);
    for (let i = 0, k = 0; i < 65536; i++, k += 4) e[i] = d[k] * 256 + d[k + 1] + d[k + 2] / 256 - 32768;
    if (src.close) src.close();
    return e;
  }
  function getTile(z, x, y, signal){
    const n = 2 ** z, url = TILE_URL.replace("{z}", z).replace("{x}", ((x % n) + n) % n).replace("{y}", y);   // x wraps across the antimeridian
    if (tileCache.has(url)) return tileCache.get(url);
    const p = (async () => {
      let last;
      for (let attempt = 0; attempt < 3; attempt++){
        try {
          const r = await fetch(url, { mode: "cors", signal });
          if (r.status === 404 || r.status === 403) return null;            // no tile: treated as void and filled
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          return await decodePNG(await r.blob());
        } catch (e){
          if (signal && signal.aborted) throw e;
          last = e; await new Promise(ok => setTimeout(ok, 400 * (attempt + 1) ** 2));
        }
      }
      const err = new Error(`elevation tile ${z}/${x}/${y} failed: ${last && last.message}`); err.code = "network"; throw err;
    })();
    tileCache.set(url, p);
    p.catch(() => tileCache.delete(url));
    return p;
  }

  /* A mosaic of whole tiles. Pixel i sits at tile coordinate ty0 + i/256, as in dem.py. */
  async function mosaic(range, signal, onTile){
    const { z, tx0, tx1, ty0, ty1 } = range, nx = tx1 - tx0 + 1, ny = ty1 - ty0 + 1, W = nx * 256, H = ny * 256;
    const e = new Float32Array(W * H).fill(NaN);
    const jobs = [];
    for (let x = tx0; x <= tx1; x++) for (let y = ty0; y <= ty1; y++) jobs.push([x, y]);
    let got = 0, missing = 0;
    await Promise.all(jobs.map(async ([x, y]) => {
      const t = await getTile(z, x, y, signal);
      if (t){ const ox = (x - tx0) * 256, oy = (y - ty0) * 256;
        for (let r = 0; r < 256; r++) e.set(t.subarray(r * 256, r * 256 + 256), (oy + r) * W + ox); }
      else missing++;
      got++; if (onTile) onTile(got, jobs.length);
    }));
    if (missing === jobs.length){ const err = new Error("no elevation tiles here"); err.code = "notiles"; throw err; }
    fillVoids(e, W, H); despike(e, W, H);
    return { z, n: 2 ** z, tx0, ty0, W, H, e };
  }
  /* load_dem(): pixels outside -12000..9000 m (and missing tiles) take the value of their nearest valid neighbour */
  function fillVoids(e, W, H){
    const N = W * H, bad = new Uint8Array(N); let nb = 0;
    for (let i = 0; i < N; i++){ const v = e[i]; if (!(v >= -12000 && v <= 9000)){ bad[i] = 1; nb++; } }
    if (!nb || nb === N) return;
    const q = new Int32Array(N); let qh = 0, qt = 0;
    for (let i = 0; i < N; i++) if (!bad[i]){ const x = i % W;
      if ((x > 0 && bad[i - 1]) || (x < W - 1 && bad[i + 1]) || (i >= W && bad[i - W]) || (i < N - W && bad[i + W])) q[qt++] = i; }
    while (qh < qt){ const i = q[qh++], x = i % W, v = e[i];
      const nb4 = [x > 0 ? i - 1 : -1, x < W - 1 ? i + 1 : -1, i >= W ? i - W : -1, i < N - W ? i + W : -1];
      for (const j of nb4) if (j >= 0 && bad[j]){ bad[j] = 0; e[j] = v; q[qt++] = j; } }
  }
  /* Browser-only clean-up: some tiles (Iceland, a few coasts) carry towers of bogus elevation standing in the sea,
     several hundred metres tall and a few pixels wide. A pixel above 150 m whose 11x11 neighbourhood is at least half
     sea takes the neighbourhood median. Real islands that high are far wider than the window, so they are untouched. */
  function despike(e, W, H){
    const out = [], r = 5, ring = [[-r, 0], [r, 0], [0, -r], [0, r], [-r, -r], [r, r], [-r, r], [r, -r]];
    for (let y = r; y < H - r; y++) for (let x = r; x < W - r; x++){
      const i = y * W + x; if (e[i] <= 150) continue;
      let quick = 0; for (const [dx, dy] of ring) if (e[i + dy * W + dx] <= 0.5) quick++;
      if (quick < 4) continue;
      let sea = 0; const win = [];
      for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++){ const w = e[i + dy * W + dx]; win.push(w); if (w <= 0.5) sea++; }
      if (sea * 2 >= win.length){ win.sort((a, b) => a - b); out.push([i, win[win.length >> 1]]); }
    }
    for (const [i, v] of out) e[i] = v;
    return out.length;
  }
  function toPix(M, lat, lon){ return [((lon + 180) / 360 * M.n - M.tx0) * 256, ((1 - Math.asinh(Math.tan(lat * D2R)) / Math.PI) / 2 * M.n - M.ty0) * 256]; }
  function bilinear(F, W, H, x, y){        // scipy map_coordinates(order=1, mode="nearest")
    x = x < 0 ? 0 : x > W - 1 ? W - 1 : x; y = y < 0 ? 0 : y > H - 1 ? H - 1 : y;
    const x0 = Math.min(W - 2, x | 0), y0 = Math.min(H - 2, y | 0), fx = x - x0, fy = y - y0, i = y0 * W + x0;
    return (F[i] * (1 - fx) + F[i + 1] * fx) * (1 - fy) + (F[i + W] * (1 - fx) + F[i + W + 1] * fx) * fy;
  }
  /* The finest level that covers a point wins; the coarsest one clamps at its edge. */
  function sampler(levels){
    const f = (lat, lon) => {
      for (let k = 0; k < levels.length; k++){ const M = levels[k], [x, y] = toPix(M, lat, lon);
        if (k === levels.length - 1 || (x >= 0 && y >= 0 && x <= M.W - 1 && y <= M.H - 1)) return bilinear(M.e, M.W, M.H, x, y); }
    };
    f.levels = levels; return f;
  }
  /* distance (km) from a point to the nearest edge of a mosaic */
  function coverKm(M, lat, lon){
    const [x, y] = toPix(M, lat, lon), k = pxKm(lat, M.z);
    return Math.max(0, Math.min(x, M.W - 1 - x, y, M.H - 1 - y)) * k;
  }

  /* ---------------- statistics as numpy computes them ---------------- */
  function pct(arr, q){                     // np.percentile, linear interpolation
    const a = Float64Array.from(arr).sort(), n = a.length; if (!n) return NaN;
    const pos = (n - 1) * q / 100, lo = Math.floor(pos), hi = Math.min(n - 1, lo + 1);
    return a[lo] + (a[hi] - a[lo]) * (pos - lo);
  }
  function gauss1d(sigma){ const r = Math.floor(4 * sigma + 0.5), w = []; let s = 0;
    for (let i = -r; i <= r; i++){ const v = Math.exp(-0.5 * i * i / (sigma * sigma)); w.push(v); s += v; }
    return w.map(v => v / s); }

  /* ---------------- analyse_shape() ---------------- */
  function analyseShape(S, la, lo, peak, viewDir, fetchedR, kx){
    kx = kx || kxAt(la);
    const rr = Math.min(fetchedR * 0.9, 40), NR = 121, NA = 180;
    const rs = Float64Array.from({ length: NR }, (_, i) => rr * i / (NR - 1)), ang = Float64Array.from({ length: NA }, (_, a) => 2 * Math.PI * a / NA);
    const E = new Float64Array(NR * NA);
    for (let i = 0; i < NR; i++) for (let a = 0; a < NA; a++)
      E[i * NA + a] = Math.max(0, S(la + rs[i] * Math.cos(ang[a]) / KY, lo + rs[i] * Math.sin(ang[a]) / kx));
    const ring = (r0, r1, sector) => { const out = [];
      for (let i = 0; i < NR; i++){ if (rs[i] < r0 || rs[i] > r1) continue;
        for (let a = 0; a < NA; a++) if (!sector || sector(a)) out.push(E[i * NA + a]); }
      return out.length ? out : Array.from(E.subarray((NR - 1) * NA)); };
    // legacy base: 12 radii x 180 angles (endpoint included) on 0.35..1 x rr, unclipped
    const bl = [];
    for (let i = 0; i < 12; i++){ const r = rr * 0.35 + (rr - rr * 0.35) * i / 11;
      for (let a = 0; a < 180; a++){ const t = 2 * Math.PI * a / 179; bl.push(S(la + r * Math.cos(t) / KY, lo + r * Math.sin(t) / kx)); } }
    const base0 = Math.max(pct(bl, 20), 0);
    const H = Math.max(peak - base0, 300);
    const med = rs.map((_, i) => pct(E.subarray(i * NA, i * NA + NA), 50));
    const half = base0 + 0.5 * H;
    let k = med.findIndex(m => m <= half); if (k < 0) k = NR - 1;
    let rHalf;
    if (k > 0){ const x0 = med[k], x1 = med[k - 1];                    // np.interp(half, [med[k], med[k-1]], [rs[k], rs[k-1]])
      rHalf = half <= x0 ? rs[k] : half >= x1 ? rs[k - 1] : rs[k] + (half - x0) * (rs[k - 1] - rs[k]) / (x1 - x0); }
    else rHalf = rs[1];
    rHalf = Math.max(rHalf, 0.3);
    const steep = H / 1000 / rHalf;
    const around = (pct(ring(2 * rHalf, 6 * rHalf), 90) - base0) / H;
    // roughness: std of E minus gaussian_filter(E, (0.8/dr, 3), mode=("nearest", "wrap")), on 1..4 x r_half
    const dr = rs[1] - rs[0], wr = gauss1d(0.8 / dr), wa = gauss1d(3), hr = (wr.length - 1) / 2, ha = (wa.length - 1) / 2;
    const T = new Float64Array(NR * NA), G = new Float64Array(NR * NA);
    for (let i = 0; i < NR; i++) for (let a = 0; a < NA; a++){ let s = 0;
      for (let q = -hr; q <= hr; q++) s += wr[q + hr] * E[clamp(i + q, 0, NR - 1) * NA + a]; T[i * NA + a] = s; }
    for (let i = 0; i < NR; i++) for (let a = 0; a < NA; a++){ let s = 0;
      for (let q = -ha; q <= ha; q++) s += wa[q + ha] * T[i * NA + (((a + q) % NA) + NA) % NA]; G[i * NA + a] = s; }
    const hp = []; for (let i = 0; i < NR; i++) if (rs[i] > rHalf && rs[i] < 4 * rHalf) for (let a = 0; a < NA; a++) hp.push(E[i * NA + a] - G[i * NA + a]);
    let rough = 0;
    if (hp.length){ const m = hp.reduce((x, y) => x + y, 0) / hp.length; rough = Math.sqrt(hp.reduce((x, y) => x + (y - m) ** 2, 0) / hp.length) / H; }
    const above = (pct(ring(3, Math.min(20, rr)), 95) - peak) / H;
    const roughS = smooth01(rough, 0.02, 0.04), canyon = smooth01(above, 0.1, 0.25), shield = smooth01(0.5 - steep, 0, 0.12);
    const vb = Math.atan2(-viewDir[0], -viewDir[1]);                   // compass bearing summit -> viewer
    const dang = a => { let d = ang[a] - vb; d = Math.atan2(Math.sin(d), Math.cos(d)); return Math.abs(d); };
    const baseView = pct(ring(rHalf, 3 * rHalf, a => dang(a) < 70 * D2R), 20);
    const front = (pct(ring(2 * rHalf, 6 * rHalf, a => dang(a) < 45 * D2R), 90) - base0) / H;
    const horn = smooth01(steep, 1.0, 1.4) * smooth01(front, 0.35, 0.55);
    const out = { base0: Math.round(base0), r_half_km: round(rHalf, 2), steep: round(steep, 2), around: round(around, 2), rough: round(rough, 3),
      above: round(above, 2), front: round(front, 2), horn: round(horn, 2), rough_score: round(roughS, 2), canyon: round(canyon, 2), shield: round(shield, 2) };
    if (canyon > 0){
      const floor = pct(ring(0, Math.min(12, rr)), 3), rim = pct(ring(3, Math.min(20, rr)), 90);
      return Object.assign(out, { kind: "canyon", base_m: floor, top_m: rim, extent_factor: 1.0, snowline_m: Math.round(rim + 1500),
        relief: 1.0, spacing: 1.0, smooth: 0.0, camh: 6.2, zoom: 1.0 });
    }
    const base = base0 + horn * Math.max(baseView - base0, 0);
    const kind = horn >= 0.5 ? "horn" : steep >= 1.2 ? "range" : roughS >= 0.5 ? "massif" : steep < 0.5 ? "shield" : "cone";
    return Object.assign(out, { kind, base_m: base, top_m: peak, extent_factor: round((1 - 0.4 * horn) * (1 - 0.1 * roughS), 3),
      relief: round((1 - 0.08 * roughS * (1 - horn)) * (1 + 0.1 * horn), 3), spacing: round(1 + 0.12 * roughS, 3), smooth: round(0.06 * roughS, 3),
      camh: horn > 0.05 ? round(3.4 - 1.4 * horn, 2) : null, snowline_m: null, zoom: 1.0 });
  }

  /* ---------------- terrain(): summit, base, extent, normalisation ---------------- */
  function refineSummit(S, lat, lon){
    const kx = kxAt(lat); let best = -Infinity, bi = 0, bj = 0;
    for (let i = 0; i < 61; i++) for (let j = 0; j < 61; j++){
      const e = S(lat + (-1.5 + i * 0.05) / KY, lon + (-1.5 + j * 0.05) / kx);
      if (e > best){ best = e; bi = i; bj = j; } }
    return { lat: lat + (-1.5 + bi * 0.05) / KY, lon: lon + (-1.5 + bj * 0.05) / kx, peak: best, kx };
  }
  /* half the span of a mosaic, the way terrain() measures `fetched_r` */
  function spanKm(M, kx){
    const lat = y => Math.atan(Math.sinh(Math.PI * (1 - 2 * (y / 256 + M.ty0) / M.n))) / D2R;
    const lon = x => (x / 256 + M.tx0) / M.n * 360 - 180;
    return Math.min((lat(0) - lat(M.H - 1)) * KY, (lon(M.W - 1) - lon(0)) * kx) / 2;
  }
  /* summit (lat, lon) as given, view (lat, lon). Returns what terrain() and build_studio.py compute. */
  function frame(S, fetchedR, summit, view){
    const rf = refineSummit(S, summit[0], summit[1]), { kx } = rf, la = rf.lat, lo = rf.lon;
    const E = (view[1] - lo) * kx, N = (view[0] - la) * KY, r = Math.hypot(E, N) || 1, dir = [-E / r, -N / r];
    const sh = analyseShape(S, la, lo, rf.peak, dir, fetchedR, kx);
    const base = Math.max(sh.base_m, 0), top = sh.top_m, hp = Math.max((top - base) / 1000, 0.3);
    let extent = clamp(10.5 * hp * sh.extent_factor, 8, 60);
    extent = Math.min(extent, fetchedR / 1.3);
    const k = extent / REF_L, hs = REF_H / hp;
    const D = Math.hypot(view[0] - la, (view[1] - lo) * kx / KY) * KY;
    const v0 = Math.max(V0, -0.8 * D / k);
    return { summit: [la, lo], peak: rf.peak, base, top, extent, k, hs, v0, shape: sh, dir, distKm: D, R: extent * 1.36 };
  }

  /* ---------------- water_mask() ---------------- */
  function waterMask(M){
    const { W, H, e } = M, N = W * H, sea = new Uint8Array(N), flat = new Uint8Array(N);
    let any = false;
    for (let i = 0; i < N; i++) if (e[i] <= 0){ sea[i] = 1; any = true; }
    // lakes: 3x3 elevation range under 1 cm (maximum_filter - minimum_filter, mode reflect ~ clamp), then dilated by one pixel
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++){
      const i = y * W + x; if (sea[i]) continue; let mn = Infinity, mx = -Infinity;
      for (let dy = -1; dy <= 1; dy++){ const yy = clamp(y + dy, 0, H - 1);
        for (let dx = -1; dx <= 1; dx++){ const v = e[yy * W + clamp(x + dx, 0, W - 1)]; if (v < mn) mn = v; if (v > mx) mx = v; } }
      if (mx - mn < 0.01) flat[i] = 1;
    }
    const m = new Uint8Array(N);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++){ const i = y * W + x;
      if (sea[i] || flat[i] || (x > 0 && flat[i - 1]) || (x < W - 1 && flat[i + 1]) || (y > 0 && flat[i - W]) || (y < H - 1 && flat[i + W])){ m[i] = 1; any = true; } }
    if (!any) return null;
    // connected pieces (4-connectivity, as ndimage.label) of 0.5 km2 or more
    const lab = new Int32Array(N).fill(-1), q = new Int32Array(N), sizes = [];
    const latC = Math.atan(Math.sinh(Math.PI * (1 - 2 * (H / 512 + M.ty0) / M.n))) / D2R, cell = pxKm(latC, M.z) ** 2;
    for (let s = 0; s < N; s++){ if (!m[s] || lab[s] >= 0) continue;
      const id = sizes.length; let qh = 0, qt = 0; q[qt++] = s; lab[s] = id;
      while (qh < qt){ const i = q[qh++], x = i % W;
        for (const j of [x > 0 ? i - 1 : -1, x < W - 1 ? i + 1 : -1, i >= W ? i - W : -1, i < N - W ? i + W : -1])
          if (j >= 0 && m[j] && lab[j] < 0){ lab[j] = id; q[qt++] = j; } }
      sizes.push(qt); }
    const out = new Float32Array(N); let kept = false;
    for (let i = 0; i < N; i++) if (lab[i] >= 0 && sizes[lab[i]] * cell >= 0.5){ out[i] = 1; kept = true; }
    return kept ? out : null;
  }

  /* ---------------- fonts (build_studio.py pick_fonts) ---------------- */
  const SCRIPT_FONTS = [
    [/[԰-֏]/, "Noto Serif Armenian", "Noto Sans Armenian"], [/[Ⴀ-ჿ]/, "Noto Serif Georgian", "Noto Sans Georgian"],
    [/[぀-ヿ]/, "Noto Serif JP", "Noto Sans JP"], [/[가-힯]/, "Noto Serif KR", "Noto Sans KR"], [/[一-鿿]/, "HAN", "HAN"],
    [/[ऀ-ॿ]/, "Noto Serif Devanagari", "Noto Sans Devanagari"], [/[؀-ۿ]/, "Noto Naskh Arabic", "Noto Sans Arabic"],
    [/[֐-׿]/, "Noto Serif Hebrew", "Noto Sans Hebrew"], [/[฀-๿]/, "Noto Serif Thai", "Noto Sans Thai"],
    [/[ༀ-࿿]/, "Noto Serif Tibetan", "Noto Sans Tibetan"],
  ];
  const HAN_FONTS = { ja: ["Noto Serif JP", "Noto Sans JP"], ko: ["Noto Serif KR", "Noto Sans KR"], "zh-Hans": ["Noto Serif SC", "Noto Sans SC"], "zh-Hant": ["Noto Serif TC", "Noto Sans TC"] };
  function hanLang(lat, lon){
    if (lat >= 24 && lat <= 46 && lon >= 122.5 && lon <= 154) return "ja";
    if (lat >= 33 && lat <= 39 && lon >= 124 && lon <= 131) return "ko";
    if (lat >= 21.8 && lat <= 25.4 && lon >= 119.3 && lon <= 122.1) return "zh-Hant";
    return "zh-Hans";
  }
  function pickFonts(text, lat, lon){
    for (const [re, serif, sans] of SCRIPT_FONTS) if (re.test(text)) return serif === "HAN" ? HAN_FONTS[hanLang(lat, lon)] : [serif, sans];
    return ["Noto Serif", "Noto Sans"];
  }

  /* ---------------- the grid and CONFIG (build_studio.py) ---------------- */
  function buildGrid(S, water, fr, N = 512){
    const [la, lo] = fr.summit, kx = kxAt(la), R = fr.R;
    const dem = new Float32Array(N * N); let wat = null;
    const levels = S.levels;
    // rows have one latitude and columns one longitude, so the pixel coordinates separate per level
    const px = levels.map(M => { const xs = new Float64Array(N), ys = new Float64Array(N);
      for (let j = 0; j < N; j++) xs[j] = toPix(M, la, lo + (-R + 2 * R * j / (N - 1)) / kx)[0];
      for (let i = 0; i < N; i++) ys[i] = toPix(M, la + (R - 2 * R * i / (N - 1)) / KY, lo)[1];
      return { xs, ys }; });
    let nw = 0;
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++){
      let k = 0;
      for (; k < levels.length - 1; k++){ const M = levels[k], x = px[k].xs[j], y = px[k].ys[i]; if (x >= 0 && y >= 0 && x <= M.W - 1 && y <= M.H - 1) break; }
      const M = levels[k], x = px[k].xs[j], y = px[k].ys[i];
      dem[i * N + j] = clamp(bilinear(M.e, M.W, M.H, x, y), 0, 9000) / 1000;          // km, as the engine wants it
      if (water){
        if (M.water === undefined) M.water = waterMask(M);                          // computed once per mosaic, only where needed
        if (M.water && bilinear(M.water, M.W, M.H, x, y) > 0.5){ if (!wat) wat = new Uint8Array(N * N); wat[i * N + j] = 1; nw++; } }
    }
    return { dem, wat: nw ? wat : null, N };
  }
  function config(fr, meta){
    const [la, lo] = fr.summit, kx = kxAt(la), sh = fr.shape;
    const horn = sh.horn || 0, canyon = sh.canyon || 0, shield = sh.shield || 0;
    const auto = { kind: sh.kind || "legacy", relief: sh.relief ?? 1, spacing: sh.spacing ?? 1, camh: sh.camh ?? null,
      topoCamh: horn > 0.05 ? round(24 - 10 * horn, 2) : null, snowline: sh.snowline_m ? Math.round(sh.snowline_m) : null,
      fit: horn >= 0.05 || canyon >= 0.05, fadePow: round(1 + Math.min(1, horn + canyon + shield), 2),
      crowd: round(Math.max(horn, sh.rough_score || 0), 2), canyon: round(canyon, 2), shield: round(shield, 2) };
    const peaks = (meta.labels || []).map(p => ({ name: p.name, sub: p.sub, E: p.ll ? (p.ll[1] - lo) * kx : 0, N: p.ll ? (p.ll[0] - la) * KY : 0 }));
    if (!peaks.length || Math.abs(peaks[0].E) + Math.abs(peaks[0].N) > 0.01){
      const main = peaks.filter(p => Math.abs(p.E) + Math.abs(p.N) <= 0.01), others = peaks.filter(p => !main.includes(p));
      peaks.splice(0, peaks.length, ...(main.length ? main : [{ name: "", sub: "", E: 0, N: 0 }]), ...others);
    }
    const native = meta.native || meta.name;
    const alltext = native + (meta.nativeH1 || "") + peaks.map(p => p.name + p.sub).join("");
    const [serif, sans] = pickFonts(alltext, la, lo);
    const view = [(meta.view[1] - lo) * kx, (meta.view[0] - la) * KY];
    return { DN: 512, R: fr.R, k: fr.k, hs: fr.hs, base_km: fr.base / 1000, peak_km: fr.peak / 1000, v0: fr.v0, auto, view, peaks,
      name: meta.name, native, slug: meta.name.toLowerCase().normalize("NFKD").replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "") || "mountain",
      labelFont: `"${serif}", Georgia, serif`, titleFont: `"${sans}", sans-serif`, fonts: [serif, sans],
      footnote: `${Math.abs(la).toFixed(2)}°${la >= 0 ? "N" : "S"}  ${Math.abs(lo).toFixed(2)}°${lo >= 0 ? "E" : "W"}  ·  elevation data, viewed from ${meta.fromName || "the viewpoint"}`,
      risoSub: `${meta.name.toUpperCase()}  ·  ` + peaks.filter(p => p.sub).map(p => p.sub).join("  ·  ") };
  }

  /* ---------------- automatic viewpoint ----------------
     For 16 bearings and three distances, look at the summit the way the engine's camera does (from v = -95, height
     camh, rows from the viewpoint's trim to the far edge) and score: how much of the mountain rises clear of the
     foreground in the summit's columns, how far the summit stands above the rest of the skyline, and a penalty when
     higher ground sits behind it. */
  function autoView(S, fr){
    const [la, lo] = fr.summit, kx = kxAt(la), k = fr.k, hs = fr.hs, base = fr.base;
    const camh = 3.4, hS = (fr.peak - base) / 1000 * hs, tS = (hS - camh) / 95, t0 = -camh / 95, span = Math.max(1e-6, tS - t0);
    const H = (e) => Math.max(0, (e - base) / 1000) * hs;
    let best = null;
    for (let b = 0; b < 16; b++){
      const brg = b * 22.5, sb = Math.sin(brg * D2R), cb = Math.cos(brg * D2R), dE = -sb, dN = -cb, rE = dN, rN = -dE;
      const at = (u, v) => H(S(la + (u * rN + v * dN) * k / KY, lo + (u * rE + v * dE) * k / kx));
      // skyline over the whole frame (all rows) and behind the summit
      let side = -Infinity, behind = -Infinity;
      for (let u = -40; u <= 40; u += 2){ let m = -Infinity;
        for (let v = -2; v <= V1; v += 1.5) m = Math.max(m, (at(u, v) - camh) / (v - ZCAM));
        if (Math.abs(u) >= 8) side = Math.max(side, m); else if (Math.abs(u) <= 2) behind = Math.max(behind, m); }
      // foreground in the summit's columns, per trim distance
      const fg = [];
      for (let v = V0; v <= -1.5; v += 0.75){ let m = -Infinity; for (let u = -2; u <= 2; u += 0.5) m = Math.max(m, (at(u, v) - camh) / (v - ZCAM)); fg.push([v, m]); }
      for (const f of [0.8, 1.3, 2.0]){
        const D = clamp(f * fr.extent, 6, 60), v0 = Math.max(V0, -0.8 * D / k);
        let tf = t0; for (const [v, m] of fg) if (v >= v0 + 2) tf = Math.max(tf, m);
        const vis = clamp((tS - tf) / span, 0, 1), prom = clamp((tS - side) / span, -1, 1), back = Math.max(0, (behind - tS) / span);
        const score = 2 * vis + prom - 1.5 * back + 0.04 * f;
        if (!best || score > best.score) best = { score, bearing: brg, distKm: D, vis, prom, back,
          lat: la + cb * D / KY, lon: lo + sb * D / kx };
      }
    }
    return best;
  }

  /* ---------------- the whole pipeline ---------------- */
  function zoomA(lat, lon){ for (let z = 10; z > 6; z--) if (tileRange(lat, lon, RADIUS_A, z).count <= 20) return z; return 6; }
  function zoomB(lat, lon, R, zA){
    const want = 2 * R / 511 / 1.5;                    // about 1.5 source pixels per grid sample
    let z = zA; while (z < 12 && pxKm(lat, z) > want) z++;
    while (z > zA && tileRange(lat, lon, R * 1.05, z).count > 25) z--;
    return z;
  }

  /* prepare({summit:[lat,lon], view:[lat,lon]|null, ...meta}, {onProgress, signal})
     -> { cfg, dem, wat, frame, S, view, auto } ready for makeEngine(cfg, dem, wat) */
  const levelCache = new Map();
  async function level(range, signal, onTile){
    const key = `${range.z}/${range.tx0}/${range.ty0}/${range.tx1}/${range.ty1}`;
    if (levelCache.has(key)) return levelCache.get(key);
    const M = await mosaic(range, signal, onTile);
    if (levelCache.size > 6) levelCache.delete(levelCache.keys().next().value);
    levelCache.set(key, M); return M;
  }
  async function prepare(req, { onProgress = () => {}, signal } = {}){
    const [lat, lon] = req.summit, t0 = performance.now();
    const zA = zoomA(lat, lon), rA = tileRange(lat, lon, RADIUS_A, zA);
    onProgress({ stage: "tiles", done: 0, total: rA.count });
    const A = await level(rA, signal, (d, n) => onProgress({ stage: "tiles", done: d, total: n }));
    const fetchedR = spanKm(A, kxAt(lat));
    let S = sampler([A]);
    let view = req.view, autoInfo = null;
    // first pass on level A: the frame for the automatic viewpoint and for choosing level B
    let fr = frame(S, fetchedR, [lat, lon], view || [lat - 0.3, lon]);
    if (fr.peak <= 2){ const e = new Error("open sea"); e.code = "sea"; throw e; }
    const zB = zoomB(fr.summit[0], fr.summit[1], fr.R, zA);
    let B = null;
    if (zB > zA){
      const rB = tileRange(fr.summit[0], fr.summit[1], fr.R * 1.05, zB);
      onProgress({ stage: "detail", done: 0, total: rB.count });
      B = await level(rB, signal, (d, n) => onProgress({ stage: "detail", done: d, total: n }));
      S = sampler([B, A]);
    }
    const Sgrid = S;
    // level C: the summit itself at z11 (what render.py fetches at mid latitudes), for the refinement and the
    // analysis near the peak. The grid is not sampled from it, so no seam can show in the drawing.
    if (Math.max(zA, zB) < 11){
      const rC = tileRange(fr.summit[0], fr.summit[1], 2.5, 11);
      if (rC.count <= 4){ const C = await level(rC, signal); S = sampler([C, ...S.levels]); }
    }
    if (!view){
      fr = frame(S, fetchedR, [lat, lon], [lat - 0.3, lon]);
      autoInfo = autoView(S, fr); view = [autoInfo.lat, autoInfo.lon];
    }
    fr = frame(S, fetchedR, [lat, lon], view);
    onProgress({ stage: "shape" });
    const g = buildGrid(Sgrid, req.water !== false, fr);
    const cfg = config(fr, { ...req, view });
    return { cfg, dem: g.dem, wat: g.wat, frame: fr, S, view, auto: autoInfo, zoom: [zA, B ? B.z : null],
      tiles: rA.count + (B ? tileRange(fr.summit[0], fr.summit[1], fr.R * 1.05, B.z).count : 0), ms: performance.now() - t0 };
  }

  /* hillshade for the minimap: size x size pixels over +-rKm around the summit */
  function hillshade(S, la, lo, rKm, size, dark){
    const kx = kxAt(la), e = new Float32Array(size * size), img = new ImageData(size, size), d = img.data;
    for (let i = 0; i < size; i++) for (let j = 0; j < size; j++)
      e[i * size + j] = S(la + (rKm - 2 * rKm * (i + 0.5) / size) / KY, lo + (-rKm + 2 * rKm * (j + 0.5) / size) / kx);
    let mx = 1; for (const v of e) if (v > mx) mx = v;
    const cell = 2 * rKm * 1000 / size, L = [-0.55, 0.55, 0.63];
    for (let i = 0; i < size; i++) for (let j = 0; j < size; j++){
      const k = i * size + j, gx = (e[i * size + Math.min(size - 1, j + 1)] - e[i * size + Math.max(0, j - 1)]) / (2 * cell),
        gy = (e[Math.max(0, i - 1) * size + j] - e[Math.min(size - 1, i + 1) * size + j]) / (2 * cell);
      const n = 1 / Math.hypot(gx * 2.2, gy * 2.2, 1), sh = clamp((-gx * 2.2 * L[0] - gy * 2.2 * L[1] + L[2]) * n, 0, 1), h = clamp(e[k] / mx, 0, 1);
      let c;
      if (e[k] <= 0) c = dark ? [34, 52, 70] : [178, 198, 214];
      else if (dark) c = [30 + 150 * sh + 30 * h, 32 + 145 * sh + 25 * h, 34 + 135 * sh + 20 * h];
      else c = [120 + 120 * sh + 12 * h, 118 + 118 * sh + 8 * h, 108 + 112 * sh];
      d[k * 4] = c[0]; d[k * 4 + 1] = c[1]; d[k * 4 + 2] = c[2]; d[k * 4 + 3] = 255;
    }
    return img;
  }

  return { prepare, frame, autoView, analyseShape, refineSummit, sampler, hillshade, pickFonts, kxAt, KY, tileRange, zoomA, zoomB, pxKm, spanKm, RADIUS_A, _pct: pct };
})();
