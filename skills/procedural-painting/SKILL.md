---
name: procedural-painting
description: Paint an original picture from imagination with code, no generative image model. You compose the subject yourself as a short Python script (shapes, light, colour, brush direction per mass) and a paint simulator lays it as real brush strokes with pigment mixing (Kubelka-Munk, so blue into wet yellow goes green), bristles that run dry, wet-into-wet pickup, and impasto relief under a raking light. It can also record a film of the canvas being painted. Use whenever someone asks for a painting, an oil painting or a painterly picture of any subject (a street in the rain, a still life, a landscape, a seascape, figures) made from code or "without an image model", wants to watch a picture being painted stroke by stroke, or wants an existing image repainted with physical paint. Not for prompts for image models, photo filters, advice on real paints, or vector illustration. For real mountains from elevation data use procedural-terrain-art or procedural-realism.
license: Apache-2.0
metadata:
  version: "1.2.0"
  homepage: https://dogum.github.io/procedural-art/
---

# Procedural painting

You are the painter. `scripts/atelier.py` is the paint, the brushes and the canvas; the picture itself (what is in it, where the light comes from, what colour each thing is, which way the brush moves) is a Python script you write for this one painting. Nothing is sampled from an image and no model draws anything.

| Tool | Does | Time on 1 CPU |
|---|---|---|
| `scripts/sitting.py` + `scripts/atelier.py` | An original painting of any subject, painted in sittings with a look at the canvas between them | 10 to 20 s per look at 1200×800 · 1 to 3 min for a 2400×1600 final |
| `scripts/stage.py` | Sets the subject up as a small 3D scene (solids, posable figures, camera, light) and hands the painter a colour study, masks, depth and form-following brush directions | 10 to 16 s per render |
| `scripts/repaint.py` | Repaints an existing image with the same paint; with `--aov` it paints a real mountain by its ground (fall lines, snow, water) | 15 s at 1800×600 |
| `scripts/aquarelle.py` | Watercolour, a second medium with its own simulation: transparent washes over white paper, wet-in-wet, hard edges that dry darker at the rim, granulation in the paper grain, dry brush (see `references/aquarelle.md`) | about 1 min at 2133×1600 |
| `scripts/motion.py` | A painting that moves: a fixed-camera film (a sunset timelapse) painted with one set of strokes that hold still while the light changes | about 2.5 s per frame at 1200×400, frames in parallel |
| `scripts/tape.py` | Exports the stroke list so the painting can paint itself live in a browser (`docs/paint`) | under 1 s |

Requirements: Python 3 with numpy, scipy and pillow; ffmpeg for the process film. Work in a writable folder and call scripts by path. In the commands below `SK` is this skill's folder and `S=$SK/scripts`.

Seven finished paintings live in `studies/`. Read one before writing your own: `the_letter.py` (3D stage: an interior, a seated figure, window light), `the_white_cup.py` (3D stage: a half-length figure at a table, face and hands), `pomegranates_from_the_scene.py` (the path tracer's scene file, painted), `harbour_at_first_light.py` (painted in sittings; limited palette, boats, water, drawn rigging), `rain_at_blue_hour.py` (night street, lamps, reflections, figures), `lemons_and_blue_jug.py` (still life, form and cast shadows), `wind_over_the_barley.py` (daylight landscape, clouds). Each runs as `python $SK/studies/NAME.py --out out.png [--height 1600] [--film out.mp4] [--study study.png] [--tape out.tape]`.

## How a painting is made

Paint in sittings and look at the canvas after each one. The looking is the method: most of the quality comes from what you change after seeing the picture, so never write all the passes blind and render once.

```bash
python $S/sitting.py work/harbour --init 1200x800 --ground "#8a6f5a" --seed 5    # canvas.json + an empty study.py
python $S/sitting.py work/harbour        # runs study.py and every sitting not yet run; writes look.jpg
```

1. **Decide the picture in words first.** Subject, format, where the horizon or table edge sits, where the light comes from, the two or three colours that carry it (one dominant, one opposing accent), and the single place the eye should land. Write it as the docstring of `study.py`.
2. **Build the colour study in `study.py`.** With numpy, on the coordinate fields `x` (0 to aspect) and `y` (0 at the top to 1), compute `study`, one H×W×3 image of flat intended colour: gradients for sky and ground, shapes for objects (`sd_circle`, `sd_ellipse`, `sd_box`, `sd_segment`, `sd_poly`, `sd_curve`, `lathe`, joined with `np.minimum` or `smin`, turned into masks with `fill`), light and shadow from `shade` and `cast_shadow`, noise for variation. **For anything that needs drawing (figures, interiors, streets, perspective, cast shadows) build the study on the 3D stage instead of by hand:** `sc = Scene(Camera(...), key=...)`, `sc.add(name, shapes, colour)`, `figure(sc, 'reader', at=..., pose='read')`, `B = sc.render(p)`; then `B.study` is the study, `B.masks` and `B.flow` are the masks and brush directions, and `paint(p, B, focus='reader')` lays it back to front (see `references/stage.md` and `the_letter.py`). Define here every mask and direction field the sittings will share. Run `sitting.py` and read `look.jpg`: fix drawing, values and colour now, where a change costs seconds. If the subject and the light don't read in the small grey panel, no brushwork will rescue it.
3. **Add one sitting file at a time**, named so they sort (`10_lay_in.py`, `20_sky.py`, `30_water.py`, ...), each a few `p.strokes` / `p.mass` / `p.marks` calls with a comment saying what the sitting is for. Back to front, big brush first: a thin lay-in over everything, then each mass with its own brush direction (sweeps across a sky, upright pulls down a wall, round a form, level across water). That direction per mass is what an image filter cannot do and is most of why the result reads as painted.
4. **Run `sitting.py` and read `look.jpg` after every sitting.** It shows the canvas beside the study, both again small and grey (the squint test), and a map of where they differ; `looks/NAME.jpg` keeps the canvas after each sitting. Criticise it as a painter would (muddy sky? edges all the same? strokes poking over an outline? focal point lost?), then either edit that sitting's file and run again (only the changed sittings replay), or move on.
5. **Lights and drawn lines in the last sitting.** Thick clean highlights (`thickness` 2.5 and up, `pickup` near 0), `p.dabs` for single touches, `p.marks` for masts, posts, stems and branches.
6. **Final and bundle.** `python $S/sitting.py work/harbour --final 1600 --out harbour.png --film harbour.mp4` replays every sitting at full size (sizes are fractions of the picture height, so the draft is the final seen smaller). Look at a 1:1 crop before showing it. `--bundle harbour.py` writes the whole painting as one standalone script.

Names shared between sittings must be defined in `study.py`: it runs before every sitting, while earlier sittings are restored from a checkpoint and not run again. Give `p.noise` a `seed` in `study.py` so the study is the same on every run.

Tell the user what you decided in step 1 when you show the picture, and offer the film.

## The canvas API

```python
import sys; sys.path.insert(0, f"{SK}/scripts")
from atelier import *

p = Painting(1500, 1000, seed=3, ground='burnt sienna', film=False)   # a toned canvas
x, y = p.x, p.y
study = ...                                    # H x W x 3 sRGB in 0..1: the picture in flat colour
p.strokes(np.ones_like(x), blur(study, 20), flow=0.2, size=0.085, pickup=0, dry=0.35, thickness=0.35)  # lay-in
p.mass(sky_mask, study, flow=sky_flow, sizes=(0.055, 0.028), length=7, pickup=0.55)
p.mass(jug_mask, study, flow=around(x, y, 0.7, 0.5), sizes=(0.03, 0.014, 0.006), clip=0.7, tol=0.05)
p.dry(0.6)                                     # let it set before the clean top notes
p.strokes(highlight_mask, study, size=0.012, length=2.5, pickup=0.1, load=1.3, thickness=2.6)
p.marks([[(0.33, 0.18), (0.33, 0.80)]], '#14142a', size=0.008)        # one pulled line
p.dabs([(0.4, 0.3)], 'titanium white', size=0.02, thickness=3)        # one loaded touch
p.cut_in(objects_mask, study)                  # background painted back up to the outlines
p.save('out.png'); p.film('out.mp4')           # film needs Painting(..., film=True)
```

- `p.strokes(mask, color, flow, size, ...)` lays one pass of one brush size. `color` is the study (each stroke takes the colour under its start), a triple, a hex string or a pigment name. `flow` is the brush direction in radians, a number or an H×W field (`around`, `along(field)` to follow contour lines, `across(field)` to run downhill, or your own formula).
- `p.mass(...)` is the block-in plus refining passes: the first size covers the mask, later sizes go only where the canvas is still further than `tol` from the study.
- `p.marks(lines, color, size)` draws each polyline once, in order. `curve(points)` gives a smooth line through points. `p.dabs(points, color, size, stretch)` lays single round touches (petals and leaves with `stretch` above 1).
- `size` may be an H×W field, so strokes shrink into the distance in one pass: `size=0.004 + 0.03 * near` with `gu, gv, near = ground_plane(x, y, horizon)`. `p.ground_noise(horizon, ...)` is noise lying on that ground.
- Forms: `ellipsoid_normal` and `lathe` (a jug, bottle or vase from its profile) give normals; `shade(n, light, lit, half, core, refl)` gives the light families; `specular` the highlight; `cast_shadow` a shadow lying on the table. `flow_ellipse` and `flow_lathe` are the brush directions for those forms.
- `knife=True` on any pass lays it with a palette knife (flat slabs, hard edges, ridges, skipped patches); `Painting(knife=True)` or `--knife` does the whole picture that way.
- `Painting(palette=['yellow ochre', 'vermilion', 'ivory black', 'titanium white'])` limits every colour on the brush to what those tubes can mix (that one is the Zorn palette). Three to six tubes give a picture an automatic harmony; `p.gamut.project(study)` previews the study through the palette.
- `p.glaze(mask, color, strength)` adds a transparent film (it darkens and tints, as a glaze does). `p.dry(amount)` stops later strokes dragging what is underneath.
- Colour helpers: `mix(a, b, t)` mixes two colours as pigments, `lerp` blends as light (both take an H×W weight as it is; don't add a trailing axis), `palette({'ultramarine': 2, 'burnt sienna': 1, 'titanium white': 3})` mixes tube colours by parts, `rgb('cobalt blue')` looks one up. `p.noise(scale, octaves, seed)` is smooth noise.

Every parameter, the pigment list and how the simulator works: `references/atelier.md`.

## What makes it read as a painting

- **Values before colour.** Three or four value groups, and the lightest light and darkest dark meeting at the focal point. Keep shadows connected into big shapes.
- **Temperature opposition.** Warm light gives cool shadows and the other way round. Grey down most of the picture so the one saturated accent (a red umbrella, a lemon) carries.
- **Edges.** `overshoot` lets strokes run past a mass (lost edges, background, distance); `clip` cuts them at the outline (found edges, the focal object). Use both in one picture; a painting with every edge crisp looks like a cut-out, every edge soft looks out of focus.
- **Brush size is detail.** Finish the focal area with small sizes and a tight `tol`; leave the corners at the block-in stage. `density` below 1 lets the layer beneath show through.
- **Wet into wet where things melt** (`pickup` 0.5 to 0.7, `soft` 0.2: skies, water, reflections dragged straight down), **dry over set paint where they sparkle** (`dry` 0.3, low `load`: light on water, foliage against sky).
- **Reflections in water or wet ground:** mirror the scene above the waterline into the study, smear it vertically, paint it with vertical strokes, then cut across with a sparse pass of short horizontal strokes.
- **Figures and small objects:** build them from a few `sd_segment` and `sd_poly` shapes, paint them late with small brushes and `clip` near 0.8, and keep them simple. A silhouette with one lit edge reads better than a detailed figure.
- **Cut in for a clean silhouette.** After an object is painted, repaint the background in a narrow band around it with `clip=1` on the background mask. `clip` below 1 on the object alone leaves a hairy outline.
- **Round forms:** `along()` of a shading field makes whorls and `around()` makes target rings. Build the direction by hand instead: tip to tip along a lemon, down the profile of a jug (see `lemons_and_blue_jug.py`).
- **Sun through a window (3D stage):** make the wall a `box` with `cut=box(...)`, set `soft` to 40 or more, and put an `emit` plane just outside the opening.
- **Thick paint only in the lights.** Darks thin (`thickness` 0.3 to 0.8), lights loaded (2 to 3.5). The relief lighting does the rest.

## Repainting an image

```bash
python $S/repaint.py --in render.png --out painted.png --height 1600 [--looseness 1.3] [--film process.mp4]
```

The image supplies the colours; brush direction follows its edges. This is the quick way to put a photo or a render on canvas with pigment mixing and relief. It has no idea what the objects are, so a composed study script will always look more intended.

A real mountain painted by its ground: render it with `procedural-realism`'s `pbr_terrain.py ... --aov`, post it, then pass the `aov` folder inside the band folder. The sky is swept in as one mass, slopes are painted with strokes running down the fall lines, plains level, snow late and thick, water dragged straight down and cut across, and the skyline is cut in. `--push` (default 1 here) is the painter's eye: more chroma, warmer lights, cooler shadows. `--geom` (depth only) remains for renders made with `--geom-cache` and no `--aov`.

```bash
R=/path/to/procedural-realism/scripts
python $R/pbr_terrain.py --peak 39.7019,44.2986 --from 40.1792,44.4991 --work work/ararat --size 2400x1000 --sun-el 6 --aov
python $R/post_terrain.py --bands work/ararat/bands_2400x1000 --out ararat.png
python $S/repaint.py --in ararat.png --aov work/ararat/bands_2400x1000/aov --out ararat_oil.png --height 1000
```

## A painting that moves

```bash
python $S/motion.py --frames work/tl/frames_HASH --aov work/tl/bands_1200x400_aov/aov --out sunset_oil.mp4 --keys 9
# for the browser easel: a looser cut as a stroke tape plus a sheet of the film's light (keep it near 40k strokes)
python $S/motion.py --frames work/tl/frames_HASH --aov work/tl/bands_1200x400_aov/aov --out web.mp4 --every 10 --keys 5 --looseness 1.9 --tape sunset.tape
```

`--frames` is a folder of frames from a fixed camera: a `procedural-realism` timelapse (`timelapse.py` leaves them in `<work>/frames_<hash>`), or any numbered image sequence. Painting each frame separately would make the brushwork boil, so the film is painted as one canvas: the first frame is painted in full and every stroke is remembered; at `--keys` frames spread over the film the same strokes are laid again in that frame's colours and the painter adds only what they cannot show there (a shadow edge that moved, lamps that came on, stars); then every frame is the whole set of strokes laid again, in the same order on the same weave, in that frame's colours. Lights are laid only while their spot is lit. `--aov` (buffers of the same view from `pbr_terrain.py --aov`, at the frame size) makes the first frame a terrain-aware painting. More `--keys` follow fast changes better and cost a few seconds each.

For the buffers of a timelapse view, render one more frame with the same view flags as after the `--` of the timelapse command, plus `--sky physical --geom-cache --size WxH --tag aov --aov`, into the same `--work`.

In your own script the pieces are `Painting(..., tape='full')`, `p.replay(tape, study)` (the same strokes with colours taken from another study) and `fade=True` on passes that paint lights.

## Watercolour

When the user asks for watercolour, use `scripts/aquarelle.py` instead of the oil engine: `from aquarelle import *`, `p = Paper(W, H, seed=1)`. The order is the medium's own: light to dark, whites are paper left bare, a wash on wet paper has no edge and a wash on dry paper keeps one. `p.wet(mask)`, `p.wash(mask, color, strength)`, `p.drop(points, color)` (charge colour or clean water into a wet wash), `p.stroke(lines, color, size, dry=...)`, `p.dry()` between layers, `p.save(out)`. Read `references/aquarelle.md` and one of `studies/estuary_in_watercolour.py` (wet sky, granulating hills, dry-brush water) or `studies/poppies_in_watercolour.py` (transparent petal layers) first. To paint an existing picture or a terrain render as a watercolour: `python $S/aquarelle_repaint.py --in ararat.png --out ararat_wc.png [--aov work/ararat/bands_2400x1000/aov] [--height 1000] [--border 0]`. Its limits: the mottling inside a wash is soft at 1:1, the dry-brush accents suggest a slope more than draw it, and without `--aov` a mountain is split into planes by tone alone.

## Watching it paint in a browser

Every study takes `--tape FILE` (and `sitting.py --final ... --tape FILE`): the stroke list in a 0.7 to 2 MB file. The page in `docs/paint` holds a JavaScript port of the simulator and lays the strokes again live, at any size, with a movable light. It is not a video. To add a painting there, export to `docs/paint/tapes/NAME.tape` and add `{ id: 'NAME', title: '...' }` to `PAINTINGS` in `docs/paint/app.js`. In your own script: `Painting(..., tape=True)`, then `from tape import save_tape; save_tape(p, 'x.tape', title='...')`.

## Limits

Say so when a request lands here.

- Likeness and anatomy. Figures from the 3D stage hold up to half length (pose, proportion, a plain face, mitten-like hands, a few folds); separate fingers, a gaze, real drapery and portraits of specific people are still out of reach.
- The picture is only as good as the study. The engine will paint a badly drawn study beautifully and it will still be badly drawn. Spend the iterations there.
- Fine lettering, architecture with exact perspective and anything that needs ruler precision fight the medium.
- Two media: oil (`atelier.py`) and watercolour (`aquarelle.py`). For gouache or ink finishes of an existing image use `paint.py` in `procedural-realism`.
