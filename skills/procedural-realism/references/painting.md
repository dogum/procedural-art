# Painterly finish: paint.py

`scripts/paint.py` (run as `python $S/paint.py` from a writable folder, with `S` the skill's `scripts/` folder) turns any image into a painting: a `pbr_terrain.py` render, a path-traced still life, or a photo or picture the user uploads. Strokes and washes are placed procedurally from the image itself (numpy, scipy and pillow only). No model is involved, so the painting follows the image's colours and edges, not an understanding of the objects.

```bash
python $S/paint.py --in IN.png --out OUT.png [--style oil|impasto|gouache|watercolor|ink] \
       [--size WxH] [--seed N] [--detail 1.0] [--strokes-scale 1.0] [--relief R] [--ink-mode auto|lines|wash] [--jpg]
python $S/paint_compare.py --out sheet.jpg --width 1800 --item in.png "render" --item oil.png "oil" [--gap 6] [--bg "#16140f"]
```

| Flag | Meaning |
|---|---|
| `--style` | the medium (default `oil`) |
| `--size WxH` | output size; the input is resized and centre-cropped to fill it ("cover"). Default: input size |
| `--seed N` | random seed; output is deterministic per seed |
| `--detail D` | above 1 adds more small strokes and sharper washes; below 1 gives a looser picture |
| `--strokes-scale S` | multiplies all brush sizes; for watercolour and ink it also scales the wash blur sizes |
| `--relief R` | overrides the paint-relief lighting of the stroke styles; 0 renders flat |
| `--ink-mode M` | ink only. `lines`: contour strokes first, then a few flat washes, paper left white (objects, still lifes, portraits). `wash`: sumi-e landscape (graded washes, empty sky, thin lines on long ridges and cloud edges). `auto` (default) picks `wash` when the top fifth of the picture is a light, smooth sky, else `lines`; the log line says which it chose |
| `--jpg` | also writes a `.jpg` next to a `.png` output |

`PAINT_LAYERS=n` (environment variable, debug only) stops the stroke painter after n layers.

## Choosing a style

| Style | Looks like | Use it for | 3000×1000, 1 core |
|---|---|---|---|
| `oil` | Smooth blended oil on canvas: brushwork visible up close, the picture intact from afar | The safe default; still lifes and objects, anything where the subject must stay readable | ~40–55 s |
| `impasto` | Thick, loaded strokes, strong relief, lively broken colour (Van Gogh direction) | Landscapes and skies when energy matters more than fidelity; social-media hero images | ~30–45 s |
| `gouache` | Opaque, matte, flat colour shapes from an 18-colour k-means palette, low relief on paper grain | Poster or illustration looks; strong simple compositions | ~45–60 s |
| `watercolor` | Transparent glazes with hard wet edges, granulation, reserved highlights and a deckled border | Landscapes with sky and atmosphere; also still lifes (it lifts dark images first) | ~15–25 s |
| `ink` | Monochrome ink on rice paper. Objects: brush contours of varying width plus a few flat washes. Landscapes: sumi-e washes, empty sky, thin ridge lines | Mountains and misty landscapes; still lifes and single objects with clear silhouettes | ~10–15 s |

Times are CPU seconds on the 2-core test machine; wall time is longer when the machine is busy.

How the presets did on the showcase inputs (Ararat, the Fuji lake and the pomegranate still life):
- **watercolor** is the strongest overall and reads as a real watercolour at 1:1. Dark saturated colours keep their hue (a dark red stays red), specular points are reserved as paper, and the teal fringe around saturated objects is gone. Its light glazes are soft by nature, so a subtle light structure (the thin gaps between the lenticular cloud's layers) is only partly kept.
- **oil** is very good on the still life and good on landscapes; the detail stage keeps conifer tips, snow patches and the lenticular cloud's layers. A large, near-uniform sky still reads a little like smoothed plaster up close.
- **impasto** is the most painterly and the least faithful, as intended; the detail stage gives the pomegranate seeds and the cloud layers back their shape.
- **gouache** is solid; highlights are crisp, but small round shapes stay lumpy and the dark still-life wall shows some blocky stroke ends.
- **ink** `wash` is good on mountains; `lines` turns the still life into a brush drawing (silhouettes, seeds as small circles, highlights left white). Lines follow strong colour edges, so a low-contrast silhouette (a dark sphere on a dark wall) is drawn only in part; on a landscape, `lines` turns a tree line into busy squiggles, so `wash` stays the better choice there.

## Tips

- **Draft at 1500×500** (5–20 s). Brush sizes scale with the output size (`unit = sqrt(W·H)/1732`), so a draft looks like the final downscaled.
- **Photos from the user.** Any PNG or JPG works. Pick `--size` with the aspect ratio you want; the crop is centred, so pre-crop if the subject is off-centre. Very busy, dark photos suit oil or watercolour better than ink.
- **Too busy / too loose.** `--detail 0.7` or `--strokes-scale 1.5` for a looser, more abstract result; `--detail 1.5` or `--strokes-scale 0.7` for a tighter one.
- **Flat look wanted.** `--relief 0` removes the raking-light impasto from oil, impasto and gouache.
- **Show a comparison.** `paint_compare.py` stacks the input and one or more paintings with captions into one sheet; it is the quickest way to let the user choose a style.
- **Ink mode.** `auto` guesses from the sky; pass `--ink-mode lines` for an object photographed against a light wall, `--ink-mode wash` for a landscape without sky.
- **Memory:** peak about 0.9 GB at 3000×1000 (watercolour 1.2 GB).

## How it works (for tuning the presets in `paint.py`)

- **Stroke styles (oil, impasto, gouache):** 4–5 layers with brush radii from 30 down to 2.2 px at 3000×1000. Each layer paints against a reference blurred to its brush size. The first layer covers the canvas; later layers seed only in grid cells where the flat stroke colours painted so far differ from the reference, so fine strokes land on edges and leave flat sky alone. Strokes follow the smoothed structure-tensor direction, bend with some inertia, and stop when the colour under them drifts. All strokes of a layer are traced and rasterised at once (a packed int64 z-buffer written with `np.maximum.at`). Shading adds bristle streaks, per-stroke colour jitter, an edge blend into the neighbouring strokes (wet-into-wet), pickup of the paint underneath and dry-brush break-up. Relief is the sum of stroke bodies plus bristle grooves and canvas weave, lit by a raking light. Oil at 3000×1000 is about 60k strokes.
- **Detail stage (stroke styles, last):** one or two passes of small brushes (oil 2.2 and 1.1 px, impasto 1.7, gouache 3.0 and 1.5) seeded where the band-passed painting differs from the band-passed source, weighted by how much small-scale structure the source has there and by structure-tensor coherence. So thin rims, tree tips and cloud layers get strokes, while flat passages and render noise (incoherent speckle) keep the big brushwork. Then highlights: a hard-edged mask of spots brighter than their surroundings at two scales, closed so a speckled render highlight becomes one shape, painted with dabs of the brightest nearby colour that grow toward the inside of big highlights and carry extra relief. Colours come from the source; no pixels are copied.
- **Watercolour:** the image goes to optical density against the paper colour (dark images are first lifted toward a mean luminance of 0.4; the soft clip of the darkest passages scales all three channels together, so hue survives). Six nested glazes go from light to dark; each glaze's edge sharpness comes from the local density gradient, so edges are crisp on real image edges and soft on gentle gradients. Glazes get a wobble, a darker wet-edge rim, flow blotches and flocculation. Where the light washes spilled over a small light shape, pigment is lifted back out with a crisp edge. Accents and dry-brush marks (down to 1 px) add the missing darks in the local pigment's hue (a per-channel residual made cyan marks beside saturated reds: the old teal fringe); fine marks and accents skip incoherent render noise. Specular points and thin bright rims are reserved. Granulation settles in paper valleys (weaker in heavy darks); a deckled border of bare paper finishes it.
- **Ink, `wash` mode:** a single-ink notan from luminance, with extra ink where a region is darker than its surroundings, saturated colour read as light, and bare paper in featureless light areas (multi-scale saliency). Washes bleed into rice-paper fibres; calligraphic contour strokes swell, taper and break into dry brush; a last layer of thin, pale lines follows long clean edges (ridges, cloud layers).
- **Ink, `lines` mode:** Canny-style edges on a colour-opponent gradient of the gamma-lifted image (so a red fruit on a brown table has an edge even at equal luminance), with hysteresis and removal of short chains. Brush strokes trace the chains, main contours first (4 px edge scale, 2.4 px brush) and then finer ones only where no line is yet; width follows edge contrast along the stroke (brush pressure), each stroke swells and tapers, and the tail runs dry. Highlights get no outline and stay paper. Tone is three flat washes: the lightest 40% of the picture stays paper and the rest splits into a pale and a dark wash, plus a little extra where a region is darker than its surroundings.

Files: `paint.py` (CLI, presets, stroke painter, shading, relief), `paint_strokes.py` (orientation field, seeding, tracing, rasteriser), `paint_media.py` (canvas, papers, palette, lighting), `paint_wash.py` (glazes, watercolour, ink), `paint_compare.py` (comparison sheets).

## Limits

- No understanding of objects. The detail stage keeps structures down to about 2 px when the source draws them cleanly (tree tips, rims, cloud layers), but a noisy render's speckled highlight becomes one soft shape, and a square specular reflection comes out round in oil.
- Stroke direction comes from the image, not an artist's intent, so large flat areas (plains) can look monotonous.
