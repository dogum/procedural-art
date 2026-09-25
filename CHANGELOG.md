# Changelog

All notable changes to this project are documented here. Versions follow [semantic versioning](https://semver.org/).

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
