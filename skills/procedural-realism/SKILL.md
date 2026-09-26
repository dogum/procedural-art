---
name: procedural-realism
description: Make photographic-style images with pure code and physics, no generative image model. Ray-marched terrain of any real mountain with physical sky, haze, shadows, snow, mist and clouds; mirror lakes; forests and villages; blue hour with a lit city or a moonlit starry night; sunset timelapse videos along the real sun path. A from-scratch path tracer renders JSON still lifes (glass, prisms with rainbow caustics, copper, fruit). A painterly finisher turns any image, including an uploaded photo, into oil, impasto, gouache, watercolour or ink. Use whenever someone wants a realistic or "CGI" render made from code, a photoreal landscape of a real place, a 3D still life without Blender, a painting of a render or photo, or an image path traced or ray traced from scratch with "no AI image". Not for help with other renderers (Blender, Cycles, Unreal), explaining rendering in words, photo filters or real-paint advice. For line-art, contour, riso or woodcut styles, use procedural-terrain-art.
license: Apache-2.0
metadata:
  version: "1.1.0"
  homepage: https://dogum.github.io/procedural-art/
---

# Procedural realism

Four tools, numpy/scipy/pillow only, that push how close pure math gets to a photograph or a painting:

| Tool | Best at | Time on 1 CPU |
|---|---|---|
| `scripts/pbr_terrain.py` (+ `post_terrain.py`) | Real mountains from real elevation data: atmosphere, golden light, shadows, snow, mist, clouds; optional lake reflections, trees and villages, blue hour with city lights along real streets (OpenStreetMap), moonlit night | 7 s at 600×200 · ~2.5 min at 3000×1000 plain · 5–13 min with water, trees or a city |
| `scripts/timelapse.py` | Sun-angle video (golden hour to night) along the real solar path for a date and place, every frame rendered so shadows slide | ~15 s per 600×200 frame, ~50 s at 1200×400; a 20 s clip (480 frames) ≈ 6 h at 1200×400 |
| `scripts/pathtracer.py` (+ `post_pathtrace.py`) | Tabletop scenes from a JSON file: glass, liquids, metal, glossy fruit, soft or hard light, caustics (photon map), rainbow dispersion, depth of field | ~3 s per pass at 900×300; 30–70 s per pass at 2400×800; 60–80 adaptive spp (about an hour) for a clean final |
| `scripts/paint.py` | Any image (a render from above, or a photo the user uploads) as oil, impasto, gouache, watercolour or ink | 10–60 s at 3000×1000 |

Requirements: Python 3 with numpy, scipy and pillow; ffmpeg for timelapses. The terrain renderer needs network access to AWS Terrain Tiles for elevation data, and with `--city` to the OpenStreetMap Overpass API for streets (`scripts/osm_roads.py`; cached in `--work/osm`, with a synthetic street grid when it can't be reached).

Work in a writable folder (for example the session's outputs folder) and call the scripts by path: the installed skill folder may be read-only, and every script writes only to `--work`, `--out` and the current folder. In the commands below `SK` is this skill's folder (the one containing this file) and `S=$SK/scripts`.

If the user wants stylized or graphic output instead (line engravings, contour maps, risograph prints, woodcuts), that is the `procedural-terrain-art` skill.

## Terrain: workflow

```bash
SK=/path/to/procedural-realism; S=$SK/scripts            # run from a writable folder
ARARAT="--peak 39.7019,44.2986 --from 40.1792,44.4991 --second 39.6517,44.4011 --work work/ararat"
python $S/pbr_terrain.py $ARARAT --size 600x200 && \
python $S/post_terrain.py --bands work/ararat/bands_600x200 --out draft.png       # draft: fetch + ~7 s
# final, in row chunks that each fit a time-limited shell (rows already on disk are skipped)
python $S/pbr_terrain.py $ARARAT --size 3000x1000 --rows 0:500
python $S/pbr_terrain.py $ARARAT --size 3000x1000 --rows 500:1000 && \
python $S/post_terrain.py --bands work/ararat/bands_3000x1000 --out final.png
```

- **Where:** `--peak` is the summit and `--from` is where the famous view is taken. Look both up; don't guess. The camera auto-backs along that line of sight until the mountain spans about 5° of view, which gives the telephoto "mountain from the city" look. Override with `--cam-offset` (km toward the peak; 0 = stand at `--from`).
- **Two peaks:** `--second lat,lon` frames two peaks together, for example Ararat and Lesser Ararat.
- **Light:** this matters most. By default the sun lights the faces the camera sees, at 10° elevation (golden hour). Backlit mountains read as flat grey shapes. Tune with `--sun-az` and `--sun-el`, or `--time day|golden|blue|night`.
- **Air:** `--haze` sets dust (0.1 crisp, 1+ dusty), `--air` sets how blue distant things get, and `--mist` sets the valley mist density (0 turns it off; above about 6 it becomes a flat featureless sheet).
- **Snow:** `--snowline` in metres. The default is 68% of the way up.
- **Framing:** `--layout right|center|left`, `--hfov`, `--pitch`, `--horizon F` (true horizon at fraction F from the top), `--cam-height` (km above ground). If foreground ridges hide the base, that is real geography. Raise `--cam-height`, change `--cam-offset`, or keep it: framing between dark ridges often looks better.
- **Variants:** `--tag NAME` gives a variant its own band folder (`bands_WxH_NAME`). Resuming a folder with different pixel-affecting flags stops with a message instead of mixing two pictures, and `post_terrain.py` then refuses that folder too, so a chained `&&` block can't save the old picture under a new name. Chunks can have any row range: missing rows are found and filled on the next run, and post refuses a folder with missing rows.
- **Check:** each run prints the peak elevation it found. Compare it with the real value to catch wrong coordinates (tiles clip sharp summits a little: Fuji comes out at 3742 m, real 3776 m).

### Lakes, forests, villages, night, timelapse

All off by default. Full flag list, recipes for every showcase, tuning notes and the pitfalls of each feature: `references/terrain_features.md` (read it before using any of these).

- **Water:** `--water-level auto` turns a lake near the camera into water that reflects terrain, sky and cloud; `--waves 0.1` is a calm morning mirror. Stand on the water: `--cam-offset 0 --cam-height 0.004 --horizon 0.5`. `auto` only accepts a basin that is level in the elevation data; on a dry plain it prints why and renders without water. For a lake it can't find, or the sea, give `--water-level <metres> --water-seed lat,lon`.
- **Near field:** `--trees 0.8 --villages 0.6 --near-km 5` adds trees, orchard and poplar rows, houses with pitched roofs, and field walls and fences, reflected in water. They only read with a low camera (`--cam-height 0.03` or less). Standing in a field (`--cam-height 0.003`), trees and houses from about 40 m hold up at 3000 px; add `--sky physical` and post with `--exposure 0.6`. `--biome temperate --treeline 2400` for green mountains.
- **Twilight and night:** `--time blue` or `--sun-el -4` switches to the physical sky (Earth shadow, Belt of Venus, alpenglow). `--city 0.8` lights a city along its real streets from OpenStreetMap: lamps by road class (white LED on avenues, orange sodium on side streets), car light streaks (`--traffic`, `--shutter`), lit windows in the buildings that front a street, dark parks and fields. Road data © OpenStreetMap contributors (ODbL). The first run fetches the roads; public Overpass mirrors can take several minutes for a city, and `--streets grid` skips them. Post with `--exposure "auto*0.3"`: at blue hour post keys the exposure on the sky and sets the white balance to 4800 K. `--near-lamps 16 --aperture 40` adds out-of-focus lamps. `--time night` gives a moonlit night with point stars, and post picks a night exposure by itself; `--moon 0` is starlight only. A city reads best at blue hour; under `--time night` it dominates the frame.
- **Clouds:** `--cloud-model ms` gives the lenticular multiple scattering and a silver lining.

Fuji mirrored in Lake Kawaguchi, the recipe of the showcase final (water makes a view about 4× slower than a dry one; the 3000×1000 final with trees took 12 min):

```bash
FUJI="--peak 35.3606,138.7274 --from 35.5215,138.7460 --cam-offset 0 --cam-height 0.004 --hfov 68 --layout center \
  --horizon 0.5 --water-level auto --waves 0.12 --base 900 --mist 1.0 --haze 0.15 --biome temperate \
  --treeline 2400 --snowline 2900 --trees 0.8 --villages 0.6 --near-km 5 --cloud-model ms --work work/fuji"
python $S/pbr_terrain.py $FUJI --size 900x300 && \
python $S/post_terrain.py --bands work/fuji/bands_900x300 --out fuji_lake_draft.png
```

Sunset timelapse over Yerevan (real sun path for 20 April, 18:40–20:40 local, 20 s at 24 fps; every frame is rendered at its own sun position, so shadows slide; everything after `--` goes to `pbr_terrain.py`; rerun to resume). Draft a few seconds at 600×200 first (`--start 19:26 --end 19:56 --frames 121` covers sunset):

```bash
python $S/timelapse.py --work work/tl --out ararat_sunset.mp4 --frames 480 --size 1200x400 \
  --date 2026-04-20 --utc-offset 4 --start 18:40 --end 20:40 -- \
  --peak 39.7019,44.2986 --from 40.1963,44.5238 --second 39.6517,44.4011 --cam-offset 0 --cam-height 0.05 \
  --hfov 34 --air 0.8 --city 0.8 --trees 0.5 --near-km 8 --near-lamps 16 --aperture 40
```

Add `--only-plan` before `--` to print the sun positions without rendering.

## Still life: workflow

Scenes are JSON files in `$SK/scenes/` (`pomegranate.json`, `wine_and_marbles.json`, `prism_and_crystal.json`). Copy one and edit it; the renderer never needs editing. Read `references/scenes.md` for every field (camera, lights, materials, objects, render and post settings) before writing a new scene.

```bash
python $S/pathtracer.py --scene $SK/scenes/pomegranate.json --size 600x200 --spp 8 --tag draft --work work/still
python $S/post_pathtrace.py --tag draft --work work/still --out draft.png
# final: every run ADDS samples to the same accumulator; --time stops cleanly inside a time-limited shell
python $S/pathtracer.py --scene $SK/scenes/wine_and_marbles.json --size 2400x800 --spp 80 --tag final \
  --work work/still --time 3300
python $S/post_pathtrace.py --tag final --work work/still --out wine_and_marbles.png --jpg
```

- **Sampling is adaptive** after 8 spp: later passes put samples where the noise is still visible (glass, caustics, metal) and fewer on diffuse areas the denoiser will clean. `--spp` is the frame average added by this run.
- **Primitives:** `sphere`, `ellipsoid`, `cylinder`, `box`, `disk`, `tube` (drinking glass), `bowl` (open spherical shell: balloon glass, metal bowl), `prism`, `gem` (round brilliant), a `scatter` generator, plus `planes` for table and wall. All but the sphere take `rot`.
- **Liquid in a glass:** give the `tube` or `bowl` a `fill` (`level`, `material`). Do not model the liquid as a separate solid touching the glass.
- **Materials:** `diffuse`, `coat` (pigment under clear coat: fruit, lacquer, ceramic), `metal` (`f0` colour, `rough` GGX), `glass` (`ior`, `sigma` absorption per cm for colour, optional `abbe` for rainbow dispersion).
- **Caustics:** add `"caustics": {"photons": 250000, "k": 40, "rmax": 0.8}` to `render` when glass or mirrors sit in a small, strong light. Without it caustics come only from lucky paths and stay speckled.
- **Lights:** rectangles; `spot` narrows the beam into a pool of light. A small, far, bright light gives crisp shadows and caustics; a large near one gives soft window light.
- **Look at a top view** (`"camera": {"position": [0,110,-5], "target": [0,0,0.5], "up": [0,0,1]}`) when caustics do not show: it tells you where they land and what hides them.
- **Useful flags:** `--crop x0:x1,y0:y1` renders one window (fixing one region), `--no-caustics`, `--clamp`, `--basic` (no MIS and no specular light connections, for comparisons). Post: `--exposure`, `--denoise`, `--sigma-l`, `--bloom`, `--vignette`, `--grain`, `--saturation`; defaults come from the scene's `post` block.

## Painterly finish

`scripts/paint.py` turns any image into a painting with procedurally placed brush strokes or washes: a terrain render, a path-traced still life, or a photo the user uploads. Ink draws objects line-first and landscapes as sumi-e washes (`--ink-mode`, chosen automatically). Read `references/painting.md` to choose a style and tune it.

```bash
python $S/paint.py --in final.png --out painting.png --style watercolor --size 1500x500   # draft, 5–20 s
python $S/paint.py --in final.png --out painting.png --style oil --size 3000x1000 --seed 1 --jpg
python $S/paint_compare.py --out sheet.jpg --width 1800 --item final.png "render" --item painting.png "oil"
```

| Style | Looks like | Use it for |
|---|---|---|
| `oil` | Smooth blended oil on canvas | The safe default; subjects that must stay readable |
| `impasto` | Thick loaded strokes, strong relief, broken colour | Landscapes and skies when energy beats fidelity |
| `gouache` | Opaque, matte, flat shapes from an 18-colour palette | Poster and illustration looks |
| `watercolor` | Transparent glazes, wet edges, granulation, deckled paper | Landscapes with sky; the strongest preset |
| `ink` | Monochrome ink wash on rice paper | Mountains and misty landscapes only; poor on busy, dark scenes |

`--size` resizes and centre-crops to fill (brushes scale with it, so a 1500×500 draft looks like the final downscaled). `--detail 1.5` / `0.7` = tighter / looser, `--strokes-scale` multiplies brush sizes, `--relief 0` turns off the raking-light impasto. Peak memory is about 0.9 GB at 3000×1000.

## Rules that came from real failures

- **Look before scaling up.** Render tiny (600×200), view it, fix it, then render big. Every quality jump in this work came from looking at a draft.
- **Light direction beats detail.** Moving the sun did more than any texture work.
- **Caches follow the place.** `pbr_terrain.py` keeps `dem.npz` and the resampled heightfield `hf.npz` in `--work` and reuses them only for the same peak and radius; another mountain rebuilds them. A separate `--work` per place still saves refetching when you switch back and forth.
- **Low cameras, water, trees and cities** each have their own pitfalls (mist around a low camera, streak shadows, rippled reflections); `references/terrain_features.md` lists them.
- **Denoiser vs lighting edges:** a filter guided only by normal, depth and albedo cannot see shadows and caustics and blurs them away. `post_pathtrace.py` scales its edge-stop by each pixel's measured noise, so keep the variance (don't pass `--no-variance`) and raise `--sigma-l` only for very low sample counts.
- **Coincident surfaces:** a glass base exactly on the table (both at y = 0) loses the light under it. Lift glass objects 0.02 cm.
- **Where caustics land:** a low light throws caustics far from the object (about height / tan(elevation)); they are easily hidden behind the next object. Check with a top view before the final.
- **Memory:** rays are processed in chunks (path tracer 200k by default, `--chunk`); the terrain mirror pass is chunked too. Keep chunking if you add geometry.
- **Time limits:** everything is resumable (row bands, sample accumulators, timelapse frames). Background jobs may die when a shell call returns; run chunks in the foreground, or use `pathtracer.py --time` so a run stops cleanly before the shell limit.
- **Showing results:** after writing output files, always present them. A file that is never presented can't be opened on mobile.

## Honest limits (tell the user)

Light, atmosphere, water reflections, glass, metal and real terrain at a distance are convincing. Trees and houses hold up from about 60 m out even with the camera standing in a field, but the nearest big crowns still look rendered on close inspection; cities follow real streets but their buildings are boxes, so they only work at blue hour or night, and water doesn't move. A timelapse renders every frame, so a 20 s clip takes hours. The path tracer lacks tori and rounded boxes, and paintings keep detail down to about 2 px only where the source draws it cleanly (a noisy render's speckled highlight becomes one soft shape). On one CPU core a 3000×1000 terrain frame with water and trees takes about 13 minutes and a clean glass still life about an hour. `references/lessons.md` has the full list of limits, what each technique buys and the mistakes worth not repeating.

`scripts/dem.py` (elevation download and loading with void filling) is shared with procedural-terrain-art; both skills ship an identical copy. Change both or neither.
