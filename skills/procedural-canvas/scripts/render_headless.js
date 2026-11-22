#!/usr/bin/env node
/* Render a single-file canvas piece in headless Chromium and save its canvas as a PNG.

   node render_headless.js PIECE.html OUT.png [--query "?seed=42"] [--repeat] [--crop x,y,w,h OUT2.png] [--timeout 600]

   Waits for the piece to say it is done: document.documentElement.dataset.ready === 'true'
   (also accepted: document.body.dataset.ready, window.__renderDone). Prints the render time and
   any page error. --repeat renders a second time in a fresh page and exits 2 if the two PNGs
   differ (the determinism check). --crop also saves a 1:1 crop, for looking closely.
   A page's own Content-Security-Policy is bypassed so the wait can run; the page is not changed.
   Needs the `playwright` npm package and a Chromium (set CHROME to a browser binary if needed). */
const path = require('path'), fs = require('fs'), zlib = require('zlib');
const { chromium } = require('playwright');

const args = process.argv.slice(2);
const opt = (name, n = 1) => { const i = args.indexOf(name); return i < 0 ? null : args.splice(i, n + 1).slice(1); };
const query = (opt('--query') || [''])[0], repeat = !!opt('--repeat', 0);
const crop = opt('--crop', 2), timeout = 1000 * parseInt((opt('--timeout') || ['600'])[0], 10);
if (args.length < 2) { console.error('usage: render_headless.js PIECE.html OUT.png [--query Q] [--repeat] [--crop x,y,w,h OUT2.png]'); process.exit(1); }
const [piece, out] = args;
const READY = "document.documentElement.dataset.ready==='true'||(document.body&&document.body.dataset.ready==='true')||window.__renderDone===true";

async function render(browser, clip) {
  const ctx = await browser.newContext({ viewport: { width: 1920, height: 1080 }, bypassCSP: true });
  const page = await ctx.newPage(), errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  const t = Date.now();
  await page.goto('file://' + path.resolve(piece) + query);
  await page.waitForFunction(READY, null, { timeout });
  const seconds = (Date.now() - t) / 1000;
  const grab = (c) => page.evaluate((c) => {
    const cv = document.getElementById('painting') || document.getElementById('c') || document.querySelector('canvas');
    if (!c) return cv.toDataURL('image/png');
    const k = document.createElement('canvas'); k.width = c[2]; k.height = c[3];
    k.getContext('2d').drawImage(cv, c[0], c[1], c[2], c[3], 0, 0, c[2], c[3]);
    return k.toDataURL('image/png');
  }, c).then((u) => Buffer.from(u.split(',')[1], 'base64'));
  const png = await grab(null), part = clip ? await grab(clip) : null;
  await ctx.close();
  return { png, part, seconds, errors };
}

(async () => {
  const launch = { args: ['--no-sandbox'] };
  if (process.env.CHROME) launch.executablePath = process.env.CHROME;
  const browser = await chromium.launch(launch);
  const clip = crop ? crop[0].split(',').map(Number) : null;
  const a = await render(browser, clip);
  fs.writeFileSync(out, a.png);
  if (crop) fs.writeFileSync(crop[1], a.part);
  console.log(`${path.basename(piece)}${query}: ${a.seconds.toFixed(1)} s, page errors: ${a.errors.length ? a.errors.join(' | ') : 'none'} -> ${out}`);
  let code = a.errors.length ? 3 : 0;
  if (repeat) {
    const b = await render(browser, null);
    const same = a.png.equals(b.png);
    console.log(same ? 'repeat render identical' : 'repeat render DIFFERS');
    if (!same) code = 2;
  }
  await browser.close();
  process.exit(code);
})().catch((e) => { console.error(String(e)); process.exit(1); });
