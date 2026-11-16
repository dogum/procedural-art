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
| a study script importing `scripts/atelier.py` | An original painting of any subject | 10 s at 1200×800 · about 70 s at 2400×1600 |
| `scripts/repaint.py` | Repaints an existing image (a render from the other skills, a photo) with the same paint | 15 s at 1800×600 |

Requirements: Python 3 with numpy, scipy and pillow; ffmpeg for the process film. Work in a writable folder and call scripts by path. In the commands below `SK` is this skill's folder and `S=$SK/scripts`.

Three finished paintings live in `studies/`. Read one before writing your own: `rain_at_blue_hour.py` (night street, lamps, reflections, figures), `lemons_and_blue_jug.py` (still life, form and cast shadows), `wind_over_the_barley.py` (daylight landscape, clouds). Each runs as `python $SK/studies/NAME.py --out out.png [--height 1600] [--film out.mp4] [--study study.png]`.

## How a painting is made

Work the way a painter does, and look at the picture at every step. The looking is the method: most of the quality comes from the corrections you make after seeing a draft.

1. **Decide the picture in words first.** Subject, format, where the horizon or table edge sits, where the light comes from, the two or three colours that carry it (one dominant, one opposing accent), and the single place the eye should land. Write it in the script's docstring.
2. **Build the colour study.** With numpy, on the coordinate fields `p.x` (0 to aspect) and `p.y` (0 at the top to 1), compute one H×W×3 image of flat intended colour: gradients for sky and ground, signed-distance shapes for objects (`sd_circle`, `sd_ellipse`, `sd_box`, `sd_segment`, `sd_poly`, `sd_curve`, joined with `np.minimum` or `smin`, turned into masks with `fill`), light and shadow computed from the shapes, noise for variation. Save it with `--study` and look at it. Fix drawing, values and colour here, where a change costs one second. Squint test: shrink it to 300 px; if the subject and the light don't read, no brushwork will rescue it.
3. **Paint it in masses, back to front, big brush first.** One thin lay-in over the whole canvas, then each mass with `p.mass(mask, study, flow=..., sizes=(big, medium, small))`. Give every mass its own brush direction: sweeps across a sky, upright pulls down a wall, around a round form, along a wave. This is what an image filter cannot do, and it is most of why the result reads as painted.
4. **Lights and drawn marks last.** Thick clean highlights (`thickness=2.5`, `pickup` near 0) and single deliberate strokes with `p.marks(...)` for posts, masts, branches, stems.
5. **Draft at `--height 800`, look, criticise, fix; then render the final at 1600 or more.** Sizes are fractions of the picture height, so a draft is the final picture seen smaller. Look at a 1:1 crop of the final before showing it.

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
p.save('out.png'); p.film('out.mp4')           # film needs Painting(..., film=True)
```

- `p.strokes(mask, color, flow, size, ...)` lays one pass of one brush size. `color` is the study (each stroke takes the colour under its start), a triple, a hex string or a pigment name. `flow` is the brush direction in radians, a number or an H×W field (`around`, `along(field)` to follow contour lines, `across(field)` to run downhill, or your own formula).
- `p.mass(...)` is the block-in plus refining passes: the first size covers the mask, later sizes go only where the canvas is still further than `tol` from the study.
- `p.marks(lines, color, size)` draws each polyline once, in order. `curve(points)` gives a smooth line through points.
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
- **Thick paint only in the lights.** Darks thin (`thickness` 0.3 to 0.8), lights loaded (2 to 3.5). The relief lighting does the rest.

## Repainting an image

```bash
python $S/repaint.py --in render.png --out painted.png --height 1600 [--looseness 1.3] [--film process.mp4]
```

The image supplies the colours; brush direction follows its edges. This is the quick way to put a photo or a render on canvas with pigment mixing and relief. It has no idea what the objects are, so a composed study script will always look more intended.

A real mountain painted by depth: render it with `procedural-realism`'s `pbr_terrain.py ... --geom-cache`, post it, then pass the `geom_<hash>` folder from its `--work` folder. The sky goes in first as one swept mass, far ground stays simple, near ground gets broken colour, and the skyline is a found edge. `--push` (default 1 with `--geom`) is the painter's eye: more chroma, warmer lights, cooler shadows.

```bash
python $S/repaint.py --in ararat.png --geom work/ararat/geom_d4d184947a --out ararat_oil.png --height 1000
```

## Limits

Say so when a request lands here.

- Likeness and anatomy. Faces, hands and portraits of specific people come out as simple painted shapes. Figures work at a distance, as silhouettes or as a few planes of light.
- The picture is only as good as the study. The engine will paint a badly drawn study beautifully and it will still be badly drawn. Spend the iterations there.
- Fine lettering, architecture with exact perspective and anything that needs ruler precision fight the medium.
- It is oil paint. For watercolour, gouache or ink finishes of an existing image use `paint.py` in `procedural-realism`.
