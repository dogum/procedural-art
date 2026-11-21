---
name: procedural-canvas
description: Paint a whole landscape or scene as ONE self-contained HTML file that draws itself in the browser with Canvas 2D and plain JavaScript, with no images, no libraries, no network and no generative image model. Use this when the user asks for a single HTML file, an HTML5 Canvas or vanilla JavaScript picture, a "procedural" or "every pixel computed" scene that must run offline in a browser, or a web page or artifact that renders a photorealistic or painterly landscape (mountains, lakes, forests, sea, ruins, skies) by code. You build a small renderer for that one picture (world-space terrain, linear-light water with reflection and refraction, trees, clouds, light shafts, a restrained grade), give it a seed, and check it by rendering headless and looking. It can bake real elevation data into the file so the mountain is the real one. Not for Python renders or prints of real terrain (procedural-terrain-art, procedural-realism) or for oil and watercolour simulation (procedural-painting).
license: Apache-2.0
metadata:
  version: "1.2.0"
  homepage: https://dogum.github.io/procedural-art/canvas/
---

# Procedural canvas

One HTML file, one picture. The file holds no image and loads nothing: it is a small renderer written for this one scene, and the browser computes every pixel when the page opens. You write that renderer.

`SK` below is this skill's folder.

| File | What it is |
|---|---|
| `assets/fuji-at-first-light.html` | A finished piece to read before writing your own: a real mountain from baked elevation data, a lake, a foreground. Open it, read its stage list, borrow its scaffolding |
| `scripts/render_headless.js` | Renders a piece in headless Chromium, times it, reports page errors, saves the canvas and a 1:1 crop, and checks that a second render is identical |
| `scripts/bake_dem.py` | Turns real elevation into the block a piece embeds, so the mountain in the picture is the real one |
| `references/architecture.md` | How the file is organised: stages, seed, the done flag, world space, the conventions tools rely on |
| `references/techniques.md` | How each part is made: terrain, water, reflections, trees, clouds, light shafts, the grade; and the traps |

More finished pieces, with the prompts that produced them, are at https://dogum.github.io/procedural-art/canvas/ (source in `docs/canvas/` of the repository).

## The contract of the file

Keep these every time; tools and people rely on them.

- **One file.** HTML, CSS and JavaScript inline. No `<img>`, no fonts, no fetch, no library. It must work from `file://` with the network off.
- **A fixed canvas**, 1920 × 1080 unless asked otherwise, scaled to the window with CSS.
- **A seed.** `?seed=` in the address (any string), with a default written in the file. Never `Math.random`, never the clock. Give each subsystem its own named random stream (`stream('oak')`, `stream('flora')`), so changing one does not reshuffle the others.
- **Stages with a progress line.** Paint back to front in named stages and yield to the browser between them (`await new Promise(requestAnimationFrame)`), so the picture builds up and the page never freezes.
- **A done flag.** When the last stage has drawn, set `document.documentElement.dataset.ready = 'true'`.
- **The same picture every time** for the same seed in the same browser.

## How a piece is made

1. **Decide what the picture is** before any code: the subject, where the light comes from, the three or four big shapes, what is near and what is far. A long list of features is not a composition. Place the anchors by hand (peaks, the shoreline, the main trees); let the seed vary the detail, not the layout.
2. **Build the far world as geometry in world units.** A camera (position, focal length in pixels, horizon row) and a heightfield or mesh in metres, rendered front to back. Do not stamp mountain silhouettes: terrain with real slopes gets shading, cast shadows, snow lines and haze that agree with each other. This is the largest single difference between a piece that looks rendered and one that looks drawn. For a real mountain, bake its elevation with `bake_dem.py` and say in the file what is real.
3. **Work in linear light** for anything that mixes light (water, fog, glow, haze) and encode to sRGB once at the end.
4. **Add the middle distance and the water**, then the reflection: mirror each landform about its own waterline, column by column, then let the water shader bend and tint that image.
5. **Add the foreground last**, and give it the most care per pixel: bark that follows the limb, leaves lit by which side faces the sun, stones with an up-facing side.
6. **Light and grade with restraint.** One sun, small. Light shafts only where something breaks the light. No stacked glows.
7. **Look.** Render with the harness, read the whole frame and several 1:1 crops, write down what is wrong, fix the worst thing, render again. Expect five or more rounds. Try a second seed before you call it done.

```bash
node $SK/scripts/render_headless.js piece.html out.png --repeat
node $SK/scripts/render_headless.js piece.html out.png --crop 360,140,1200,520 mountains.png
node $SK/scripts/render_headless.js piece.html seed2.png --query "?seed=harbour-7"
```

The harness needs Node with the `playwright` package and a Chromium (set `CHROME` to a browser binary if Playwright has none). If neither is available, say so and ask the user to open the file; do not claim the piece was checked.

Read `references/architecture.md` before writing the first line and `references/techniques.md` when you reach each part.

## A real mountain in the file

```bash
python $SK/scripts/bake_dem.py --npz work/fuji/dem.npz --camera 35.5222,138.7650 --look 35.3606,138.7274 \
    --lake 829 --date 2026-11-10 --time 07:05 --tz 9 --name "Mount Fuji from the north shore of Lake Kawaguchi" \
    --out block.js --into piece.html
```

It takes an elevation cache (the `dem.npz` that `procedural-realism` and `procedural-terrain-art` write, or any file with `elev`, `lats`, `lons`), a camera position and a point to look at. It bakes the ground as that camera sees it, a fan of depth profiles with Earth curvature already removed, computes where the Sun is at the given time, and writes the block between the `DEM-BEGIN` and `DEM-END` markers of the piece (and to `--out`, `block.js` by default). The command above reproduces the block in `assets/fuji-at-first-light.html` exactly. The format is documented at the top of the script and of `assets/fuji-at-first-light.html`, which shows how to read it and scan it into a picture. Getting the elevation needs one of the terrain skills (`dem.py`); this skill does not fetch it. Write in a comment, and in a small caption on the page, what is real (terrain, viewpoint, sun) and what is invented (foreground, trees, weather).

## What to tell the user

Give the file, the seed, the render time you measured, and whether the repeat render was identical. Say what is real and what is invented. Name the weak spots you saw in the crops; every piece has some.

## Limits

Say so when a request lands here.

- People, animals up close, vehicles and architecture with fine ornament take far longer than landscape and usually look modelled, not photographed.
- It is a still picture by default. Gentle motion (mist, water, a few lights) is possible; a film is not.
- Different browsers and GPUs differ by a level here and there in anti-aliasing and gradients. The scene is the same; the bytes of a screenshot are not.
- `ctx.filter` is ignored by some Safari versions: blur by hand.
- A piece is 70 to 100 KB of bespoke code and several rounds of looking. It cannot be rushed into one pass.
