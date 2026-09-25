# Scene files for pathtracer.py

A scene is one JSON object. Units are centimetres, y is up, the table is usually the plane y = 0 and the camera looks roughly along +z. Unknown keys are ignored, so `"name"` and `"description"` can be used anywhere as notes.

Start from the closest of the three shipped scenes and change it one part at a time:

| File | What it shows | Final used |
|---|---|---|
| `scenes/pomegranate.json` | Pomegranates, apricots, copper sphere, a glass of juice and a crystal ball on walnut, window light from the upper left | the first showcase still life |
| `scenes/wine_and_marbles.json` | Balloon glass of red wine, water tumbler, crystal ball, five glass marbles on wood; a low warm spot from the left throws long shadows filled with coloured caustics; a dim gradient backdrop, and a card above the frame that backlights the wine | 2400×800, 56 adaptive spp, then extra passes cropped to the glasses (`--crop`), ~59 min |
| `scenes/prism_and_crystal.json` | Low sun through a window on white linen; a flint prism fans a spectrum, a brilliant-cut paperweight, a small gem and a crystal ball throw sparkles and rainbows | 2400×800, 60 adaptive spp, ~54 min |

## camera
| key | meaning | default |
|---|---|---|
| `position`, `target` | eye and look-at point | required |
| `up` | up vector | `[0,1,0]` |
| `hfov` | horizontal field of view, degrees | 50 |
| `aperture` | lens radius in cm (0 = pinhole); depth of field | 0 |
| `focus` | `"target"`, a distance, or a point `[x,y,z]` to focus on | `"target"` |

## lights (list, at least one)
Rectangles that emit on one side. Either `corner`, `u`, `v` (edges; emits along normalize(u × v)) or `center`, `aim` (a point it faces) and `size` `[w,h]`.
`color` (rgb), `intensity` (radiance multiplier), `spot` (emission × cos^spot of the angle from the light's normal; 0 = even, 8 = broad pool, 30+ = tight beam). Lights are visible to the camera and in reflections. The light used for each shadow ray is picked by power.

Two light tricks from `wine_and_marbles.json`:
- **Backdrop:** a large, dim light standing behind the table is a background. A `spot` of 20 to 30 makes it fade away from the point that faces the camera, a soft gradient. Glass is cleaner in front of a backdrop light than in front of a lit wall: a path through glass that ends on a light is counted exactly, while one that ends on a diffuse wall still needs a random shadow ray, and a large share of those go to the brightest light, not the one lighting the wall.
- **Backlighting a liquid:** a curved glass bowl shows an upside-down view of what is above and behind it. A bright card just above the top of the frame lights the wine without ever being seen directly. Keep its `spot` low (0 to 3): the rays leaving the bowl reach it 10 to 40° off its axis.

## environment
rgb radiance returned by rays that leave the scene.

## materials (name -> dict), referenced by name or given inline
| type | keys |
|---|---|
| `diffuse` | `color` or `tex` (+ texture parameters), `tint` |
| `coat` | pigment `color`/`tex` under a clear coat: `ior` (sets coat reflectance, 1.5 -> 4%), `coat_rough` (GGX alpha, 0.06) |
| `metal` | `f0` rgb reflectance at normal incidence (copper `[0.95,0.64,0.54]`, gold `[1.0,0.78,0.34]`, steel `[0.56,0.57,0.58]`), `rough` GGX alpha (0 = mirror, 0.05 polished, 0.3 brushed) |
| `glass` | `ior`, `sigma` absorption per cm per channel (colour of thick glass and liquids: red wine `[0.18,2.2,1.7]` (reads almost black), backlit ruby `[0.09,0.85,0.75]`, cobalt `[1.1,0.6,0.06]`), `abbe` (dispersion: 59 crown glass, 30 flint, lower = stronger rainbows), `outside_ior`/`outside_sigma` for a solid embedded in another medium (coloured core in a marble) |

Textures (`tex`): `walnut`, `oak` (planks), `marble`, `linen`, `plaster` (params `base`, `dark`/`vein`, `scale`, `plank`), and the fruit textures `pom`, `apricot`, `crown`. A new texture is a function `f(q_local, n, obj, mat) -> (albedo, bumped normal)` added to `TEXTURES` in `pathtracer.py`.

## planes
`{"point": [...], "normal": [...], "material": ...}`: infinite, used for table and wall.

## objects
| type | keys |
|---|---|
| `sphere` | `center`, `radius` |
| `ellipsoid` | `center`, `radii`, `rot` |
| `cylinder` | `base` (centre of the bottom disk), `radius`, `height`, `rot` |
| `box` | `center`, `size` (full extents), `rot` |
| `disk` | `center`, `radius`, `rot` (normal is local +y) |
| `tube` | drinking glass: `base`, `r_outer`, `r_inner`, `height`, `bottom` (base thickness), `fill`, `rot` |
| `bowl` | open spherical shell: `center`, `r_outer`, `r_inner`, `cut` (height of the opening above the centre; positive = narrowing balloon glass, negative = open bowl), `fill`, `rot` |
| `prism` | triangular prism lying on a face: `center` (middle of the bottom face), `side`, `length` (along local z), `rot` |
| `gem` | round brilliant: `center` (girdle centre), `radius`, optional `table`, `crown_angle`, `pavilion_angle`, `girdle`, `rot` |
| `scatter` | generator: `count` spheres on the table, `center` [x,z], `ring` [r0,r1], `scale` [sx,sz], `radius` [a,b], `seed`, `material` |

`rot` is `[rx, ry, rz]` degrees (x, then y, then z) or a list of steps about world axes, e.g. `[["z", -40.75], ["y", 30]]` lays a gem on a pavilion facet and then turns it.
`fill` is `{"level": h, "material": m}`; the level is measured from the tube base or from the bowl centre.
Every object needs `material`. Leave a 0.02 cm gap between glass and the table.

## render
`max_depth` (8), `clamp` (limit on indirect contributions per sample, 0 = off; 4 to 8 keeps fireflies down), `caustics`: `{"photons": per pass, "k": neighbours, "rmax": max gather radius in cm, "glossy_alpha": metals up to this roughness also bounce photons}`.

## post (defaults for post_pathtrace.py)
`exposure`, `denoise` (a-trous iterations), `sigma_l`, `vignette`, `bloom`, `grain`, `saturation`. Command-line flags of the same names override them.

## Accumulators

Each run of `pathtracer.py` adds `--spp` samples to `<work>/acc_<tag>.npz` (per-pixel sum and count, luminance second moment, aux buffers, and the scene JSON text and hash). Resuming with an edited scene prints a warning; a size mismatch is refused. Use a new `--tag` whenever you change the scene or the size. `--work` defaults to `$PT_WORK` or `./pt_work`. The legacy positional form `pathtracer.py W H SPP [tag]` still works and renders the pomegranate scene; `post_pathtrace.py <tag> [exposure] [iters] [out]` still works too.

## What the sampler does (so you know which knob to reach for)

- Multiple importance sampling between light sampling and BSDF sampling on diffuse, clear-coat and rough-metal surfaces.
- At every glass or mirror hit, the reflected and refracted directions are tested against the lights directly, so window reflections in glass are smooth instead of a spray of white dots.
- Adaptive sampling after `--adaptive-after` spp (default 8): each pass spends the same number of samples as a uniform pass, sent where the noise will be visible after tone mapping. `--diffuse-weight` (default 0.4) lowers the priority of diffuse pixels the denoiser cleans; no pixel gets more than 32× the mean. It judges noise by luminance only, so pure colour noise from dispersion is under-weighted.
- The caustic photon map covers light -> glass/mirror/glossy metal -> diffuse paths; camera paths skip exactly those, so nothing is counted twice. Photon shooting and gathering cost about 2 to 5 s per pass regardless of image size. The caustics are slightly soft (gather radius) and each pass adds low-frequency blotches that average out over passes.
- Spectral dispersion: a path picks a wavelength the first time it meets glass with an `abbe` number. The showcases exaggerate it (Abbe 12 for the prism, 22 for the "crystal") so the colours read.

Measured on the pomegranate scene (equal CPU time, raw un-denoised mean): reaching the same glass noise takes 6.7× less time than the basic sampler (`--basic`), copper 12×, the table caustic 2.4×.

## Known limits

- Nested media are one level deep: a solid inside another names its outside medium; tube/bowl fills are handled specially. A diffuse object submerged in liquid does not get the liquid's absorption.
- No handles, tori or rounded boxes (an espresso cup is out of reach without adding a primitive).
- Convex polyhedra find their bounding sphere by enumerating plane triples, about 1 s per 57-plane gem at load.
- `clamp` trades a little energy in secondary bounces for fewer fireflies.
