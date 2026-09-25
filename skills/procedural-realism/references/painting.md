# Painterly finish: paint.py

`scripts/paint.py` (run as `python $S/paint.py` from a writable folder, with `S` the skill's `scripts/` folder) turns any image into a painting: a `pbr_terrain.py` render, a path-traced still life, or a photo or picture the user uploads. Strokes and washes are placed procedurally from the image itself (numpy, scipy and pillow only). No model is involved, so the painting follows the image's colours and edges, not an understanding of the objects.

```bash
python $S/paint.py --in IN.png --out OUT.png [--style oil|impasto|gouache|watercolor|ink] \
       [--size WxH] [--seed N] [--detail 1.0] [--strokes-scale 1.0] [--relief R] [--jpg]
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
| `--jpg` | also writes a `.jpg` next to a `.png` output |

`PAINT_LAYERS=n` (environment variable, debug only) stops the stroke painter after n layers.

## Choosing a style

| Style | Looks like | Use it for | 3000×1000, 1 core |
|---|---|---|---|
| `oil` | Smooth blended oil on canvas: brushwork visible up close, the picture intact from afar | The safe default; still lifes and objects, anything where the subject must stay readable | ~45 s |
| `impasto` | Thick, loaded strokes, strong relief, lively broken colour (Van Gogh direction) | Landscapes and skies when energy matters more than fidelity; social-media hero images | ~40 s |
| `gouache` | Opaque, matte, flat colour shapes from an 18-colour k-means palette, low relief on paper grain | Poster or illustration looks; strong simple compositions | ~45 s |
| `watercolor` | Transparent glazes with hard wet edges, granulation, bare-paper highlights and a deckled border | Landscapes with sky and atmosphere; also still lifes (it lifts dark images first) | ~20 s |
| `ink` | Monochrome ink wash on rice paper: contour strokes, graded washes, empty sky | Mountains and misty landscapes (sumi-e) only | ~15 s |

How the presets did on the showcase inputs (an Ararat render and the pomegranate still life):
- **watercolor** was the strongest on both. It reads as a real watercolour at 1:1. Weak spot: a slight teal fringe around some saturated objects.
- **oil** is very good on the still life. On Ararat it is good at thumbnail size, but the large, near-uniform sky reads a little like smoothed plaster up close, because the source sky has almost no colour variation.
- **impasto** is the most painterly and the least faithful, as intended.
- **gouache** is solid; on the still life the dark wall shows some blocky horizontal stroke ends.
- **ink** is good on mountains (empty sky, contour strokes on ridges) and weak on the still life: hairy strokes around the copper ball and heavy outlines around highlights look like a charcoal sketch. Recommend it for landscapes only.

## Tips

- **Draft at 1500×500** (5–20 s). Brush sizes scale with the output size (`unit = sqrt(W·H)/1732`), so a draft looks like the final downscaled.
- **Photos from the user.** Any PNG or JPG works. Pick `--size` with the aspect ratio you want; the crop is centred, so pre-crop if the subject is off-centre. Very busy, dark photos suit oil or watercolour better than ink.
- **Too busy / too loose.** `--detail 0.7` or `--strokes-scale 1.5` for a looser, more abstract result; `--detail 1.5` or `--strokes-scale 0.7` for a tighter one.
- **Flat look wanted.** `--relief 0` removes the raking-light impasto from oil, impasto and gouache.
- **Show a comparison.** `paint_compare.py` stacks the input and one or more paintings with captions into one sheet; it is the quickest way to let the user choose a style.
- **Memory:** peak about 0.9 GB at 3000×1000.

## How it works (for tuning the presets in `paint.py`)

- **Stroke styles (oil, impasto, gouache):** 4–5 layers with brush radii from 30 down to 2.2 px at 3000×1000. Each layer paints against a reference blurred to its brush size. The first layer covers the canvas; later layers seed only in grid cells where the flat stroke colours painted so far differ from the reference, so fine strokes land on edges and leave flat sky alone. Strokes follow the smoothed structure-tensor direction, bend with some inertia, and stop when the colour under them drifts. All strokes of a layer are traced and rasterised at once (a packed int64 z-buffer written with `np.maximum.at`). Shading adds bristle streaks, per-stroke colour jitter, an edge blend into the neighbouring strokes (wet-into-wet), pickup of the paint underneath and dry-brush break-up. Relief is the sum of stroke bodies plus bristle grooves and canvas weave, lit by a raking light. Oil at 3000×1000 is about 60k strokes.
- **Watercolour:** the image goes to optical density against the paper colour (dark images are first lifted toward a mean luminance of 0.4). Six nested glazes go from light to dark; each glaze's edge sharpness comes from the local density gradient, so edges are crisp on real image edges and soft on gentle gradients. Glazes get a wobble, a darker wet-edge rim, flow blotches and flocculation; dry-brush marks add accents; granulation settles in paper valleys (weaker in heavy darks); a deckled border of bare paper finishes it.
- **Ink:** a single-ink notan from luminance, with extra ink where a region is darker than its surroundings, saturated colour read as light, and bare paper in featureless light areas (multi-scale saliency). Washes bleed into rice-paper fibres; calligraphic contour strokes swell, taper and break into dry brush.

Files: `paint.py` (CLI, presets, stroke painter, shading, relief), `paint_strokes.py` (orientation field, seeding, tracing, rasteriser), `paint_media.py` (canvas, papers, palette, lighting), `paint_wash.py` (glazes, watercolour, ink), `paint_compare.py` (comparison sheets).

## Limits

- No understanding of objects: structures under about 3 px (pomegranate seeds, thin glass rims) become blobs.
- Stroke direction comes from the image, not an artist's intent, so large flat areas (plains) can look monotonous.
