# Changelog

All notable changes to this project are documented here. Versions follow [semantic versioning](https://semver.org/).

## [Unreleased]

### Web app
- The picture stays pinned at the top while you scroll through the controls, so every slider change is visible as it happens (phones, and 3:1 or 4:1 sizes on larger screens).

### procedural-realism
- Close foregrounds: trees get leaf clumps on branches, sky gaps, a visible trunk and limbs, self-shadowed crowns, backlit leaves and dappled shadows; detail appears only once it is large enough in pixels, so far trees look as before. Houses get tuff, plaster or board walls, tile, tin or slate roofs, framed windows, doors, chimneys, gutters and weathering. Dry-stone walls, hedges and fences run along field and yard edges with gates; fields have crop rows, grass, flowers and rutted lanes. A low camera 60 m from a house now reads as a village rather than a model. Fixed two old bugs that smeared everything within 50 m of a low camera.
- Blue hour: below about 8° of sun the sky is computed per wavelength (ozone's Chappuis band) with a Hillaire multiple-scattering table, so the zenith is blue instead of lavender and twilight darkens at the measured rate. Daylight skies are unchanged.
- Cities at dusk are lit along their real streets: new `scripts/osm_roads.py` fetches OpenStreetMap roads through the Overpass API (cached; the synthetic grid is the fallback, or `--streets grid`). Lamps by road class, buildings facing the street, dark parks and fields, `--traffic` for car light streaks. Post keys the blue-hour exposure on the sky and sets a 4800 K white balance (`--wb`, `--sat`).
- Timelapse renders every frame at its own sun position, so shadows slide instead of cross-fading (`--frames`; `--blend K` renders every K-th frame). Exposure is smoothed over sun time, refraction tapers smoothly below the horizon, and the frames land on the blue-hour look. Faster frames with a warm `--geom-cache`.

## [1.1.0] - 2026-09-25

### Web app
- **[Any mountain](https://dogum.github.io/procedural-art/app/)**: type any mountain on Earth, drag the viewpoint on a hillshade map, pick one of the six print styles, and save an X, LinkedIn, YouTube or wallpaper PNG. Everything runs in the browser: elevation tiles are fetched and measured on the page with a JavaScript port of the landform analysis (it matches the Python on the seven regression landforms). 94 curated peaks with classic viewpoints, OpenStreetMap search for everything else and an automatic viewpoint, and links that reopen the same view.
- The six styles and the camera are one code path: `scripts/build-app.py` copies the engine from the studio template into `docs/app/engine.js`, and CI fails if it is out of date. Studio pages render pixel for pixel as before.

### procedural-terrain-art
- Topo on rough terrain uses illuminated contours: lines on slopes facing away from the light are heavier, so horns and ranges read as relief instead of stacked steps. Crowded intermediate contours drop out whole, and index lines that peek over a ridge are drawn light (no more dark ledges). Cones are unchanged.
- Grand Canyon woodcut carves by depth below the rim: the gorge and shaded side canyons print black, the rim, buttes and lit walls stay paper.
- Water layer: sea and flat lakes in view are drawn per style (coastline, sparse level lines), with `--no-water` to turn it off. A new shield landform score thins the crowded rows of broad plateaus (the Mauna Kea saddle no longer starts as a hard slab).
- All of the above in both the Python renderer and the studio page.

### procedural-realism
- `paint.py`: a detail stage (small brushes where structure was lost, hard-edged highlights last) for oil, impasto and gouache; no more teal fringe in watercolour; `--ink-mode lines|wash|auto`, with a line-first brush drawing for objects.
- Wine and marbles scene: reframed, a backdrop and a glow card behind the wine so it reads ruby, less grain in the glass.
- `post_pathtrace.py`: the denoiser no longer blacks out pixels where a light is directly visible.

## [1.0.0] - 2026-09-24

First public release: two Claude skills, packaged as one Claude Code plugin and as two Claude.ai skill files.

### procedural-terrain-art
- Any real mountain, range, canyon or island drawn from SRTM-derived elevation in six print styles: survey (engraved ridgelines), topo (true contours), nocturne, stipple, riso and woodcut.
- Landform-aware framing: the engine measures the terrain around the summit and adapts base, extent, relief, line spacing, camera height and sun placement for horns, ranges, canyons and island volcanoes. Cone volcanoes keep the original framing pixel for pixel.
- Looping orbit videos (`animate.py`) and a self-contained interactive studio page (`build_studio.py`, about 0.76 MB, no server; only web fonts load over the network).
- Output sizes for X, LinkedIn and YouTube banners, wallpapers and prints; labels in the local script.

### procedural-realism
- Ray-marched photographic terrain for any mountain: Rayleigh and Mie scattering, soft shadows, snow, valley mist, lenticular clouds with multiple scattering.
- Lakes with Fresnel reflection and ripples, trees and villages near the camera, blue hour and night with city lights and lens bokeh, and `timelapse.py` along the real sun path for a date and place.
- A Monte Carlo path tracer in numpy driven by JSON scenes: spheres, boxes, prisms, cut gems, hollow glasses with liquid, spectral dispersion. MIS and adaptive sampling make glass converge about 6.7× faster and copper 12× faster than a plain sampler.
- `paint.py`: oil, impasto, gouache, watercolour and ink finishes for any image, including photos.

### Both
- Shared `scripts/dem.py` (download plus void filling); the build refuses to package if the two copies differ.
- Every long render is resumable; caches are keyed to the place; scripts write to the working directory, so read-only installs work.
- Evals for `claude plugin eval` and skill-creator.
