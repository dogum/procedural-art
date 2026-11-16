# Changelog

All notable changes to this project are documented here. Versions follow [semantic versioning](https://semver.org/).

## [Unreleased]

### procedural-painting (new skill)
- Original oil paintings of any subject, composed in code. A study script describes the picture in flat colour (numpy gradients and signed-distance shapes) and gives each mass a brush direction; `scripts/atelier.py` lays it as brush strokes on a toned canvas, back to front, big brushes first.
- Pigment mixing: colours are 12-band reflectance spectra mixed on Kubelka-Munk absorption and scattering, so blue into wet yellow goes green and complements grey each other.
- Bristle strokes that run dry, catch on the canvas weave (dry brush) and drag the wet paint they cross (wet-into-wet); paint relief with ridges and bristle grooves under a raking light; single drawn marks (`marks`), transparent glazes, and a process film of the canvas being painted (`film`).
- Three finished studies: `rain_at_blue_hour.py`, `wind_over_the_barley.py`, `lemons_and_blue_jug.py`. About 30k strokes and 70–90 s at 2400×1600 on one core.
- The 3D stage: `scripts/stage.py` sets the subject up as a small scene of solids (sphere, ellipsoid, box, capsule, cylinder, cone, torus, plane, lathe), posable figures (`figure`, eight poses and joint angles, skirt, coat, hat), a camera and a light, and renders what a painter needs: a colour study in light and shadow families with cast shadows, a mask and depth per object, highlights, and brush directions that follow each form. `paint(p, B, focus=...)` lays it back to front. `load_pathtracer_scene` reads `procedural-realism` scene files. Two studies: `the_letter.py` (an interior with a seated figure) and `pomegranates_from_the_scene.py`.
- Watercolour, a second medium: `scripts/aquarelle.py` simulates transparent pigment layers over white paper on a wetness field (wet-in-wet spread, hard edges that dry darker at the rim, granulation into the paper grain per pigment, dry brush, clean-water drops, lifting, a process film). `Paper`, `wet`, `wash`, `drop`, `stroke`, `dry`, `lift`; two studies, `estuary_in_watercolour.py` and `poppies_in_watercolour.py`; `references/aquarelle.md`.
- Watercolour of any picture: `scripts/aquarelle_repaint.py --in IMAGE --out wc.png [--aov BANDS/aov]` reserves the lights as bare paper, drops the sky into wet paper and lays the land as washes from far to near; with the terrain renderer's painter buffers it separates snow, cloud, water and depth planes. In the engine: wash edges wander like a loaded brush (`hand=`), clean water in a drying wash leaves a backrun (`bloom=`), and colour travels further on very wet paper.
- A closer figure: the stage's `figure()` now has a head with a jaw, brow, eye sockets, nose, mouth and ears, hair styles (`hair_style`), hands with a palm, thumb and jointed mitt that can hold things (`grip_*` joints), and cloth with folds (`drape`, `cloth=`). It holds up to half length; a third stage study, `the_white_cup.py`, shows it.
- Moving paintings: `scripts/motion.py` paints a fixed-camera film (a sunset timelapse) as one canvas. The first frame is painted in full, a few key frames add what the fixed strokes cannot show, and every frame is the same strokes laid again in that frame's colours, so the brushwork holds still while the light moves. In the engine: `Painting.replay(tape, study)`, `tape='full'`, `fade=True` for lights that are laid only while lit.
- Painting in sittings: `scripts/sitting.py` keeps a painting as a folder (`study.py` plus one file per sitting), checkpoints the wet canvas after each sitting and writes a look sheet (canvas, study, both in grey, difference map), so the next sitting is decided after looking; editing a sitting replays only from there. `--final` renders at full size with film and tape, `--bundle` writes one standalone script. A fourth study, `harbour_at_first_light.py`, was painted this way.
- Form helpers: `lathe` (turned forms from a profile), `ellipsoid_normal`, `shade` (light, halftone, core shadow, reflected light), `specular`, `cast_shadow`, `flow_ellipse`, `flow_lathe`, `ground_plane`, `p.ground_noise`; `along`/`across` take a `fallback` direction for flat areas.
- `p.dabs` (single round touches, petals), `p.cut_in` (the background painted back up to an outline), brush size as a field (strokes that shrink into the distance in one pass), `p.checkpoint` / `Painting.resume`, `p.look`.
- Palette knife: `knife=True` on a pass, or on the whole painting. Limited palettes: `Painting(palette=[...])` keeps every colour on the brush inside what a few tubes can mix.
- Strokes no longer poke over a mask edge, and a soft mask thins strokes instead of turning them into stray dabs.
- The easel (`docs/paint`): `scripts/tape.py` and `--tape` export a painting's strokes (0.7–2 MB); a JavaScript port of the simulator replays them live in the browser at any size, with a movable light.
- Moving paintings on the easel: `motion.py --tape FILE` writes the film's whole set of strokes as a tape with a sheet of small studies beside it (`FILE.light.jpg`, `--lights` moments). The browser player lays the same strokes in the light of any moment (`Painting.setMoment`); the easel page has an hour slider and plays the evening through. Tapes now carry, for study-coloured passes, the colour each stroke read and where.
- Your turn on the easel: drag on any canvas to lay a stroke of your own with the same simulator (13 tube colours, brush width, palette knife, one-step undo), and a blank toned canvas to start from. In the player: `beginStroke`, `userStroke`, `undoStroke`, `Atelier.blankTape`.
- `marks()` and `dabs()` coloured from a study follow a new study on `replay`, like stroke passes.
- `scripts/repaint.py` repaints an existing image with the same paint. With `--aov` (the painter buffers of a `pbr_terrain.py --aov` render) it paints a real mountain by its ground: slopes down the fall lines, snow late and thick, water dragged down and cut across, the skyline cut in. With `--geom` (the geometry cache of a `pbr_terrain.py --geom-cache` render) it paints a real mountain by depth: the sky as one swept mass, far ground kept simple, near ground in broken colour, the skyline as a found edge.

### Web app
- A mountain picked from the OpenStreetMap suggestions gets its surveyed elevation (OSM `ele`) and native name: Photon search carries neither, so one Nominatim lookup by OSM id fills them in while the elevation tiles load. The label, caption and share link show the surveyed height instead of the elevation-data estimate. `ele` values in feet are converted.

### procedural-terrain-art
- Woodcut and riso print in solid ink at their edges. Where the woodcut block thins out (near foreground, the left fade, low slopes, water) it breaks into gouge-like strokes along the terrain rows instead of fading to grey; fading riso lines break into dashes of full ink instead of a pale tint. Same in the web app and studios.
- Fixed a crash in riso and woodcut when a title's script has no installed font (matplotlib 3.11 rejected the `sans-serif` fallback).

### procedural-realism
- `pbr_terrain.py --aov` saves painter buffers per band (depth, fall-line direction, slope, elevation, snow, water, cloud, tree coverage) for `procedural-painting`. The picture is unchanged.
- Timelapse frames render four at a time (`--jobs`, one core each): the first frame fills the caches alone, the rest only read them, and the frames come out bit-identical. A 9-frame draft went from 202 s to 76 s on 4 cores.
- When OpenStreetMap can't be reached, `--city` remembers it in `<work>/osm` and uses the synthetic streets straight away. Before, every timelapse frame waited minutes on the Overpass mirrors again, and a frame rendered after they recovered switched the city to real streets mid-clip.

## [1.2.0] - 2026-09-26

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
