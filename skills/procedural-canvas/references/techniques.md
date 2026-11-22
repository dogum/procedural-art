# How each part is made

These are the methods that held up across the finished pieces in `docs/canvas/`. Each note says what to do and what goes wrong if you don't.

## Terrain: a heightfield, scanned front to back

1. **Fill a grid of heights in metres** (say 1200 × 600 cells at 12 m). For an invented range: a maximum over a list of peaks, each a cone with its own slope and rotation measured in a warped distance, minus folded ("ridged") noise, `relief = a - b * |noise|`, at three scales. For a real mountain: the baked elevation.
2. **Carve runoff.** Visit cells from highest to lowest; each passes its accumulated flow to its steepest lower neighbour of eight. Lower the ground by a little of `log(flow)`. Gullies appear where water would run, and the same field later says where snow collects.
3. **Scan it.** For each depth slice `z` from near to far, and each screen column `x`: world `wx = (x - HX) * z / FOCAL`, projected row `py = HY - FOCAL * height(wx, z) / z`. Keep a per-column ceiling (the highest row drawn so far). If `py` is above it, the rows between belong to this slice: store their depth, interpolated between this slice and the last. One pass gives a depth for every terrain pixel, with correct occlusion.
4. **Shade each pixel from its world position.** Normal from height differences at two scales (a broad one for form, a fine one for texture). Sun term `n · L`. **Cast shadows**: step from the point toward the sun at a few growing distances and compare the terrain there with the ray's height; soften the test. Add a second, broad light for the bright sky opposite the shadows, or shadowed faces go dead.
5. **Materials from the same fields.** Rock colour from the shade plus strata (`|sin|` of a mix of position and height, bent by noise). Snow where `height`, flat slope, the flow field and noise together pass a threshold: it then lies in gullies and on ledges, not as a stripe at one altitude.
6. **Air.** Mix toward the horizon colour by distance and by low altitude (valleys are hazier). Haze toward a *dimmed* version of the sky glow near the sun, so ranges stay silhouettes against it.
7. **Anti-alias the skyline** only: at the top pixel of each column, blend by how far the true silhouette is from the pixel edge.

What goes wrong without geometry: a 1D ridge line with painted shading has no cast shadows, its snow and rock do not agree with its form, and it reads as a cut-out.

A middle-distance valley is a second, finer heightfield nearer the camera. Root trees on it by casting camera rays to the ground; size them `FOCAL / z` and take their colour from the ground beneath.

## Reflections: each landform about its own waterline

A lake is not one mirror line across the frame: a near bank meets the water lower on the screen than a far shore. Draw each landform into its own offscreen canvas, record its waterline row per column, and build the reflection image by flipping every column about that column's waterline. Sky and distant mountains flip about the far shore. Stones standing in the water flip about their own base. Then the water shader samples this image.

## Water: one physically motivated shader

For each water pixel `y` rows below the horizon, with the eye `e` metres above the water: distance `wz = e * FOCAL / (y - HY)`, and a pixel footprint on the water `foot = wz / (y - HY)` metres.

- **Waves that fade by footprint.** Sum two or three directional sinusoids with wavelengths in metres, each multiplied by `exp(-(k * foot)^2)` for its own `k`, so a wave vanishes smoothly once it is smaller than a pixel. Without this the far water shimmers and bands.
- **Calm and breeze.** Low-frequency noise in world coordinates scales the waves: patches of mirror, patches of ruffle.
- **Reflection.** Displace the lookup into the reflection image by the wave slope, more toward the viewer (`amp` growing with `y`). Sample bilinearly.
- **Fresnel.** Cosine of the view angle to the water `c = (y - HY) / hypot(FOCAL, x - HX, y - HY)`; reflectance `F = 0.02 + 0.98 * (1 - c)^5`. Far water is all reflection; water at your feet is mostly see-through.
- **Refraction and depth.** Snell: horizontal component of the transmitted ray `s = sqrt(1 - c²) / 1.333`. The bed is seen at `wz + depth * s / sqrt(1 - s²)`. Depth grows with distance from the bank.
- **Absorption.** Per channel `T = exp(-k * depth / sqrt(1 - s²))` with `k` about 0.78, 0.36, 0.27 per metre for red, green, blue: shallows show the bed, deeper water goes green-blue.
- **Bed.** Two scales of Voronoi "shingle", brighter caustic lines where a noise field crosses zero, only where enough light gets through to matter.
- **Sun.** A small disc in the reflection plus a narrow glitter streak under it, gated by the ripples.
- Mix `F * reflection + (1 - F) * transmitted` in linear light, then encode.

An artistic Fresnel over screen `y` in gamma space looks acceptable alone and wrong next to this.

## Trees

- **Large foreground trees: a swept tube.** Author the trunk and main limbs as node lists `(x, y, radius)`, smooth them with Catmull-Rom, and rasterise the tube per pixel so each pixel knows its angle around the limb and its distance along it. Bark fissures, moss on the upper side and lichen are functions of those two coordinates, so they wrap the wood. Grow the twigs and leaves from the scaffold by seeded recursion. A fully generic branching grammar gives a tree; authored scaffolds give *that* tree, in the right place for the composition.
- **Width.** Children of a branch of width `w` get `w * f^(1/2.4)` for their share `f` (the pipe model).
- **Leaves.** Thousands of small shapes, each shaded by which side of its cluster faces the sun; rim-lit and slightly translucent toward the light, dark and cool away from it. Uniformly bright leaves read as cartoon.
- **Conifers.** Whorls of boughs down the trunk, sorted front to back, drooping with length; needle strokes only on near trees. A solid tiered fan reads better than a curtain of strokes.
- **Distant forest.** Small silhouettes coloured by the haze at their distance; many, varied in height.

## Sky and clouds

- Sky: an analytic gradient from zenith to horizon, warmer toward the sun, plus a tight halo around a **small** sun disc.
- Clouds as a deck in perspective: plane coordinates `u = (x - HX) / q, v = k / q` with `q` the height above the horizon, so they flatten toward it. Domain-warped fBm for coverage, lit edges from the density gradient toward the sun.
- For storm skies, march a 3D density toward the sun for self-shadowing.

## Light shafts

Rays are visible only where something near the light breaks it. Build a mask of what blocks the sun (boughs, tracery, cloud edges, peaks), then either blur it radially from the sun's position at quarter resolution, or march from each pixel toward the sun through the mask with exponential extinction. Add the result in linear light. No occluder, no shaft: a fan of rays from an open sky is a graphic device, not light.

## Grade

Contrast with a gentle curve, a little chroma, a light vignette, fine grain. Stop there unless the brief asks for a painting: bloom, split toning and canvas weave each cost realism.

## Traps

- **Stacked glows.** Halo plus bloom plus rays plus a bright sky turns the sun into a 300-pixel white hole and washes out the mountains beside it. Keep one, keep it tight.
- **Screen blend above 1.** `1 - (1 - a)(1 - b)` with `a > 1` inverts the hue (a cyan sun). Clamp first.
- **Shearing noise by a noisy slope** makes marble swirls on rock. Use a heavily smoothed fall direction.
- **Submerged parts of stones** drawn opaque look like bowls. Fade them out below the waterline and let the water shader show them.
- **Glitter as connected lines** looks like scribbles. Use isolated hash sparkles gated by the ripple field.
- **`ctx.filter`** is not reliable in every Safari: blur with your own box passes.
- **A Content-Security-Policy meta tag** in the piece blocks the evaluation automation uses to wait on it. The harness bypasses it; other tools may not.
- **`status`** as a variable name collides with `window.status`. Name the progress element something else.
