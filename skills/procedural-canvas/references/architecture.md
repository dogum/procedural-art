# How a piece is organised

## Skeleton

```html
<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Name of the picture</title>
<style>html,body{margin:0;height:100%;background:#0c0d0f}
canvas{display:block;margin:auto;max-width:100vw;max-height:100vh;aspect-ratio:16/9}
#status{position:fixed;left:12px;bottom:10px;font:12px ui-monospace,monospace;color:#9aa}</style></head>
<body><canvas id="painting" width="1920" height="1080"></canvas><div id="status"></div>
<script>
'use strict';
const W = 1920, H = 1080, ctx = document.getElementById('painting').getContext('2d');
const note = document.getElementById('status');      // not `status`: that name is a window property
const SEED = new URLSearchParams(location.search).get('seed') || 'default-17';
// ... helpers, then the stages ...
const STAGES = [['sky', sky], ['mountains', mountains], ['lake', lake], ['foreground', foreground], ['light', light], ['grade', grade]];
(async () => {
  for (let i = 0; i < STAGES.length; i++) {
    note.textContent = `${STAGES[i][0]} (${i + 1}/${STAGES.length})`;
    await new Promise(requestAnimationFrame);
    STAGES[i][1]();
  }
  note.textContent = `seed ${SEED}`;
  document.documentElement.dataset.ready = 'true';
})();
</script></body></html>
```

## Randomness

One hash of the seed string, then a separate generator per subsystem.

```js
function hashString(s) { let h = 2166136261; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return h >>> 0; }
function stream(name) {                       // mulberry32 seeded by the seed and the stream's name
  let a = hashString(SEED + '/' + name);
  return () => { a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
```

Noise tables (the permutation for value or gradient noise) come from a stream too. Never call `Math.random`, `Date.now` or `performance.now` for anything that reaches a pixel.

Let the seed change detail (which leaf, which ripple, where a stone sits within its group) and keep the layout authored: peaks, shoreline, main trunks and flower drifts placed by hand. A layout that the seed can rearrange has to be good for every seed, which is much harder than one good composition.

## Two kinds of drawing

- **Per pixel**: fill an `ImageData` in a loop. Use it for anything with shading that varies continuously: sky, clouds, terrain, water, rock, ground, bark. Work in floats, write bytes once.
- **Vector calls**: `ctx.beginPath`, `fill`, `stroke`. Use it for countable things: leaves, needles, blades, flowers, birds. Sort back to front.

Offscreen canvases (`document.createElement('canvas')`) hold layers that are needed twice: a landform drawn once is used for the picture and again, flipped, for its reflection; a mask of occluders feeds the light shafts.

## World space

Pick a pinhole camera and keep it for the whole file:

```js
const FOCAL = 1260, HX = W / 2, HY = 611;          // focal length in pixels, principal point, horizon row
// a point x metres right, h metres above the eye, z metres ahead:
//   sx = HX + FOCAL * x / z,  sy = HY - FOCAL * h / z
// and back: a pixel on the ground plane `eye` metres below the camera is z = eye * FOCAL / (sy - HY) metres away
```

Author terrain, tree sizes, wave lengths and fog depths in metres. A tree at distance z is `FOCAL / z` pixels per metre tall, its colour is hazed by z, and its base sits where the camera ray meets the ground. When everything shares one camera, scale, haze and reflections agree without tuning.

It is fine to place a peak by where it should appear in the frame and back-project that to world coordinates (`x = (sx - HX) * z / FOCAL`, `h = (HY - sy) * z / FOCAL`): the composition stays under control and the geometry is still real.

## Linear light

Keep two lookup tables and mix light only in linear values:

```js
const toLinear = new Float32Array(256), toSRGB = new Uint8ClampedArray(8193);
for (let i = 0; i < 256; i++) { const c = i / 255; toLinear[i] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); }
for (let i = 0; i <= 8192; i++) { const c = i / 8192; toSRGB[i] = 255 * (c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055); }
```

Water (reflection plus transmission), fog over a scene, glow added to a picture and haze toward the sky colour all go wrong in gamma space: mixes come out too dark in the middle and added light shifts hue.

## The look loop

```bash
node scripts/render_headless.js piece.html out.png --repeat --crop 360,140,1200,520 crop.png
```

Read the full frame first (does it read as the subject at a glance, where does the eye go, is anything blown out or muddy), then 1:1 crops of the three places a viewer will look: the far subject, the waterline, the foreground. Write the faults down in plain words before touching code. Fix the worst one. A render takes seconds, so do this many times. Check a second seed near the end; a piece tuned on one seed often has a tree or cloud in a bad place on another.

The harness exits 3 on a page error and 2 if the repeat render differs. A difference means something unseeded reached a pixel: look for `Math.random`, time, or iteration over an object whose order depends on insertion timing.

## Size and speed

A finished landscape is typically 70 to 100 KB and 1,300 to 1,600 lines, and paints in 3 to 10 seconds in headless Chromium. Per-pixel stages dominate: compute expensive layers (clouds, light shafts) at half or quarter resolution and scale up; bail out of a pixel early when a later layer will cover it. Baked elevation adds its size as base64 (about 1.33 bytes per byte).
