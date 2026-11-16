# atelier.py: the canvas, the brushes and the paint

`from atelier import *` gives you everything below. Lengths are fractions of the picture height (so a script renders the same picture at any `--height`); `p.x` runs 0 to `p.aspect`, `p.y` runs 0 at the top to 1 at the bottom; angles are radians with 0 along +x and π/2 pointing down; colours are sRGB in 0..1.

## Painting

`Painting(width, height, seed=1, ground=(0.62, 0.50, 0.36), weave=1.0, film=False, film_px=900, log=True)`

| Argument | Meaning |
|---|---|
| `ground` | Colour of the toned canvas. It shows through dry-brush and thin passes, so it tints the whole picture: a warm ground under a cool picture (or the reverse) makes the paint vibrate |
| `weave` | Strength of the canvas texture in the final lighting (0 = a smooth panel) |
| `film` | Record a frame after every sitting for `p.film()` |

Fields and state: `p.x`, `p.y`, `p.W`, `p.H`, `p.aspect`, `p.view()` (the flat colour on the canvas now, without relief lighting), `p.noise(scale=0.2, octaves=4, seed=None)` (smooth noise, unit standard deviation, features `scale` of the height across; give a `seed` to get the same field at every size), `p.blur(field, s)` (s in picture heights).

### p.strokes(mask, color, flow=0, ...)

One pass of one brush size. Returns the number of strokes laid.

| Parameter | Default | Meaning |
|---|---|---|
| `mask` | | Where the mass is, H×W in 0..1. Strokes start inside it |
| `color` | | H×W×3 study, a triple, `'#hex'` or a pigment name. Each stroke loads the colour under its starting point |
| `flow` | 0 | Brush direction, a number or H×W |
| `size` | 0.03 | Brush width |
| `length` | 5 | Stroke length in brush widths |
| `density` | 1 | 1 covers the mass about twice; 0.3 to 0.6 leaves the layer below showing between strokes |
| `load` | 1 | Paint on the brush. Below 1 the stroke breaks up sooner; 1.3 to 1.5 for loaded lights |
| `opacity` | 1 | Coverage of a full-bodied stroke. 0.5 to 0.8 for scumbles and halos |
| `pickup` | 0.35 | Wet-into-wet: the fraction of wet paint under the stroke that is dragged into it. 0 for clean notes, 0.5 to 0.7 for skies, water and reflections |
| `dry` | 0.1 | Dry-brush: paint catches only on the crowns of the weave. 0.3 gives a sparkling broken stroke |
| `jitter`, `hue` | 0.05, 0.03 | Per-stroke value and hue variation (broken colour). Raise both for foliage and fields, lower for skin and smooth objects |
| `marble` | 0.5 | A second, imperfectly mixed colour on the brush, visible as streaks along the stroke |
| `wobble` | 0.2 | How much a stroke wanders off the flow direction |
| `overshoot` | 0.3 | How far strokes may run past the mask edge, in brush widths. Loose edges |
| `clip` | 0 | 0..1: cut strokes at the mask outline. Crisp edges |
| `ctol` | 0.14 | A stroke stops when the study colour under it differs this much from the colour on the brush. Lower keeps strokes inside small colour shapes; raise it (0.2 to 0.3) for long drags over busy passages |
| `refine` | None | A tolerance: lay strokes only where the canvas differs from `color` by more than this. `p.mass` sets it for you |
| `thickness` | 1 | Impasto relief of the pass |
| `taper` | 0.45 | How much the stroke narrows as the brush lifts |
| `soft` | 0.08 | How far the sides of a stroke melt into what is under them, in brush widths |
| `size_var` | 0.25 | Spread of brush widths within the pass |
| `bidir` | True | Strokes run either way along the flow. False makes every stroke go the same way (rain, grass leaning in wind) |
| `batches` | 3 | The pass is laid in this many successive sittings; each one picks up paint from the ones before |

### p.mass(mask, color, flow=0, sizes=(0.06, 0.03, 0.014), tol=0.06, **stroke parameters)

Covers the mask with `sizes[0]`, then comes back with each smaller size only where the canvas is still further than `tol` from the study. Smaller `tol` and more sizes give a tighter finish. Typical: sky and backgrounds two sizes and `tol=0.06`; the focal object three sizes down to 0.006 and `tol=0.045`.

### p.marks(lines, color, size=0.006, mask=None, ...)

Each line is a list of `(x, y)` points and is drawn once, in order: later lines go over earlier ones. `color` is one colour, a list with one per line, or a study field sampled at each line's start; `size` may be a list. `curve(points, n=16)` returns a smooth line through points. `wobble` (in picture heights) shakes the points a little so ruled lines look hand-pulled. A dab is a line about as long as the brush is wide.

### p.glaze(mask, color, strength=0.3), p.dry(amount=1.0)

A glaze adds the colour's absorption on top of what is there: it can only darken and tint (a thin `'alizarin crimson'` glaze over a shadow, `'indian yellow'` over a sunlit wall). `dry` lowers the wetness of the whole surface so later strokes stay clean.

### p.save(path, relief=1.0, gloss=0.35, light=(-0.55, -0.62)), p.film(path, seconds=8, hold=2, fps=30)

`save` lights the relief (raking light from the upper left by default) and writes PNG or JPG. `relief=0` gives the flat colour. `film` writes an mp4 (or a GIF if the path ends in `.gif`) of the sittings in order.

## Shapes, fields and colour

| Function | Returns |
|---|---|
| `sd_circle(x, y, cx, cy, r)`, `sd_ellipse(x, y, cx, cy, rx, ry, angle)`, `sd_box(x, y, cx, cy, hx, hy, angle, r)` | Signed distance fields: negative inside |
| `sd_segment(x, y, ax, ay, bx, by, r0, r1)` | A capsule tapering from radius r0 to r1: limbs, trunks, bottles' necks |
| `sd_poly(x, y, points)`, `sd_curve(x, y, points, r0, r1)` | A polygon; a smooth tube through points |
| `smin(a, b, k)`, `np.minimum(a, b)`, `np.maximum(a, -b)` | Smooth union; union; subtraction |
| `fill(sd, soft=0.004)` | Mask from a distance field |
| `smoothstep(a, b, x)`, `blur(a, pixels)` | The usual |
| `along(field, smooth_px)`, `across(field, smooth_px)`, `around(x, y, cx, cy)`, `image_flow(img, sigma_px)` | Brush direction fields |
| `mix(a, b, t)` | Pigment mix of two colours (fields broadcast) |
| `lerp(a, b, t)` | Light blend. Use it for gradients of light; use `mix` where paints would be mixed |
| `rgb(name_or_hex)`, `palette({...})`, `PIGMENTS` | Tube colours: titanium white, naples yellow, cadmium lemon, cadmium yellow, indian yellow, yellow ochre, raw sienna, cadmium orange, cadmium red, vermilion, alizarin crimson, quinacridone rose, burnt sienna, venetian red, burnt umber, raw umber, ultramarine, cobalt blue, cerulean, prussian blue, phthalo blue, cobalt turquoise, viridian, sap green, chromium green, terre verte, dioxazine violet, cobalt violet, ivory black, paynes grey |

Shading a round form: with `d = sd_circle(...)` and a light direction `(lx, ly)`, a usable sphere normal is `nx, ny = (x - cx) / r, (y - cy) / r`, `nz = sqrt(max(1 - nx² - ny², 0))`, and `lambert = nx*lx + ny*ly + nz*lz`. Map `lambert` to colour with three stops (light, halftone, shadow) and add a reflected light near the shadow edge; a cast shadow is the same shape shifted away from the light and squashed onto the ground plane, darkest where it touches the object.

## How the simulator works

- **Colour.** Every sRGB colour becomes a 12-band reflectance spectrum (405 to 695 nm) from three smooth primaries that sum to a flat white, so an unmixed colour round-trips exactly. The canvas stores `sqrt(K/S)` per band (Kubelka-Munk absorption over scattering); averaging two states is a pigment mix. That is why ultramarine dragged into wet cadmium yellow turns green and complements grey each other, where an RGB blend would give grey and mud. The darkest paint reflects 2%, which sets how strongly darks tint lights.
- **Strokes.** A pass seeds a jittered grid inside the mask, traces each stroke along the flow field with some inertia and curl, and stops it at the mask edge (plus `overshoot`), when the study colour drifts (`ctol`) or at its length. Each stroke is rasterised as a ribbon in its own (along, across) coordinates; later strokes in a sitting cover earlier ones.
- **Bristles.** Across the ribbon a noise texture, long in the stroke direction, gives every bristle track its own paint load. Load falls toward the end of the stroke; where it falls below what the local canvas tooth needs, the stroke breaks up (dry brush). A second texture marbles two slightly different colours along the stroke.
- **Wet into wet.** Each pixel of a stroke looks back along its own bristle track at the canvas as it was before the sitting and mixes in a share of that paint, more the further the brush has travelled and the wetter the surface. Paint is therefore dragged forward in streaks, in pigment space.
- **Relief.** Stroke body, ridges squeezed to the sides, bristle grooves and a build-up toward the lift-off add to a height field; new paint partly flattens what is under it. `save` lights the height field plus the canvas weave (which thick paint buries) with a diffuse raking light, a little cavity shading and a gloss highlight.

## Performance and memory

About 30k strokes and 70 s for a 2400×1600 picture on one core; drafts at 1200×800 take about 10 s. Memory is about 0.2 GB per megapixel plus up to 0.5 GB while a sitting of large strokes is laid. Recording a film adds about 10%.
