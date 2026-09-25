---
name: procedural-terrain-art
description: Turn any real mountain, volcano, range, canyon or island on Earth into high-resolution procedural art from actual elevation data, in six print styles (engraved ridgelines, topographic contours, moonlit nocturne, stipple, risograph, woodcut), plus looping orbit videos and a self-contained interactive web studio. Framing adapts to the landform, so cones, horns like the Matterhorn, massifs, ranges, canyons and islands all work. Use this whenever someone wants a mountain or landscape drawn with code, a profile header or banner (X/Twitter, LinkedIn, YouTube), poster, wallpaper or print of a real place, Unknown Pleasures-style ridgeline art of a real mountain, topographic or contour-line art, or asks to recreate an AI-generated landscape with code, even if they never say "terrain" or "elevation". Not for copying the album cover itself, GIS work (hillshade, viewshed, QGIS maps), 3D-print models or logos. For photographic renders, lakes, night cities or painted looks, use procedural-realism.
license: Apache-2.0
metadata:
  version: "1.0.0"
  homepage: https://dogum.github.io/procedural-art/
---

# Procedural terrain art

This skill draws a real place, not an invented one. It downloads real elevation data, looks at the mountain from a real viewpoint, hides the lines that would be behind other terrain, and renders the result in a chosen print style. Everything is code, so any output can be re-rendered at any size, angle, light or palette.

Requirements: Python 3 with numpy, scipy, matplotlib, pillow and contourpy; network access to AWS Terrain Tiles; ffmpeg for video (optional).

Work in a writable folder (for example the session's outputs folder) and call the scripts by path. The skill folder itself may be read-only once installed, so nothing should be written there. In the commands below, `SK` is this skill's folder (the one containing this file). Images, DEM caches and video frames go next to `--out`, relative to the current folder.

If the user wants a photographic look (atmosphere, lake reflections, forests, a city at blue hour) or a painted one (oil, watercolour, ink), that is the `procedural-realism` skill.

## What you can make

| Output | Tool | Typical time |
|---|---|---|
| Still image, one style (PNG/JPG, any size) | `$SK/scripts/render.py --style <name>` | 5–15 s at 3000×1000 |
| All six styles + contact sheet | `$SK/scripts/render.py --style all` | ~40–60 s |
| Looping orbit video (MP4) | `$SK/scripts/animate.py` | 1–4 s per frame at 1500×500 |
| Interactive web studio (sliders, six styles, orbit, PNG export) | `$SK/scripts/build_studio.py` | ~5 s to build |

Styles: `survey` (engraved ridgelines on parchment, sun in the saddle), `topo` (true contour lines from a high camera, triangulation network), `nocturne` (moonlit lines, stars, crescent), `stipple` (pointillist dots), `riso` (two spot inks, halftone, misregistration), `woodcut` (solid ink mass with carved lines that widen in the light).

## Workflow

1. **Pin down the place.** You need the summit's lat/lon and a viewpoint: the town or spot the famous view is from, passed as `--from lat,lon` (or `--facing <bearing>` if there's no obvious viewpoint). The viewpoint decides which side of the mountain faces the camera and which peak sits on the left or right. Look coordinates up with web search if you're not certain; don't guess. If the user shows an image, identify the mountain from its silhouette and work out the viewpoint from which peak is on which side.
2. **Draft fast.** Run with `--quality 0.5` (and a smaller `--size` if you like) and look at the output (view a downscaled copy; the full file is too big to inspect well). The first run fetches elevation tiles and caches them next to the output as `dem_<lat>_<lon>.npz`. Later runs into the same folder (render, animate and the studio) reuse it if it covers the area they need, and fetch a larger one if it doesn't; `--dem` reuses a file from anywhere.
3. **Frame it.** Framing is automatic for the landform (see below). Adjust with the knobs in the table, one or two at a time, looking at each result.
4. **Render final** at `--quality 1` and the target size. Deliver the PNG plus a JPG if the PNG is over the platform's upload limit.
5. **Offer the next step:** the other styles as a contact sheet, an orbit video, or the web studio.

```bash
SK=/path/to/procedural-terrain-art          # this skill's folder; run from a writable folder
# Ararat from Yerevan (two labelled peaks, Armenian names)
python $SK/scripts/render.py --peak 39.7019,44.2986 --from 40.1792,44.4991 --style survey \
  --label "ՄԱՍԻՍ|5137 m" --label "ՍԻՍ|3896 m@39.6517,44.4011" --out out/ararat.png
# Fuji from Lake Kawaguchi, every style + contact_sheet.jpg
python $SK/scripts/render.py --peak 35.3606,138.7274 --from 35.5100,138.7550 --style all \
  --label "富士山|3776 m" --title "富士山" --out out/fuji/
# A horn: the Matterhorn from Riffelsee (all framing automatic)
python $SK/scripts/render.py --peak 45.97639,7.65861 --from 45.98333,7.76222 --style nocturne \
  --label "Matterhorn|4478 m" --out out/matterhorn.png
# A canyon: Brahma Temple from Mather Point
python $SK/scripts/render.py --peak 36.1304634,-112.0387030 --from 36.0616497,-112.1079463 --style riso \
  --label "Brahma Temple|2302 m" --label "Zoroaster Temple|2171 m@36.1188,-112.0452" \
  --title "Grand Canyon" --out out/canyon.png
# An island volcano seen from the coast is a quiet dome; this manual framing gives it shape
python $SK/scripts/render.py --peak 19.82056,-155.46806 --from 19.70556,-155.08583 --style topo \
  --yaw 40 --relief 1.3 --spacing 0.5 --pan-y -90 --label "Mauna Kea|4207 m" --out out/mauna_kea.png
```

Labels: `"Name|subtitle"` labels the main summit; add `@lat,lon` for other peaks. `--title` is the big word in riso and woodcut, usually the name in the local script. `--footnote` replaces the coordinates line. The script ends with a JSON line: the summit it found, peak and base elevations, extent, the detected `kind` and the shape scores. Check the peak elevation against the real one: a large mismatch means the coordinates are off. (Tiles smooth sharp summits a little, so 1–3% low is normal: the Matterhorn comes out at 4350 m instead of 4478 m.)

## Landform-aware defaults

Before framing, the renderer measures the terrain around the summit: how steep the peak is relative to its footprint, how much high ground sits between the peak and the viewer, how rough the surface is, and whether the target sits below the surrounding plateau.

| `kind` | Example | What the defaults do |
|---|---|---|
| cone, shield | Fuji, Ararat, Mauna Kea | The standard framing, which the other rules adjust. Shields also thin ridgelines that pile up on screen, so their broad far flanks don't print as a solid slab |
| horn | Matterhorn | Higher base, tighter scene, lower camera, so the peak owns the frame instead of the whole massif |
| massif, range (rough) | Denali, Tetons | Slightly wider line spacing and a little smoothing. Topo is drawn like a survey map (below) |
| canyon | Grand Canyon | Drawn from its floor to its rim, seen from just above the rim, no snow. Woodcut is carved by depth below the rim |

Horns and canyons also get an automatic vertical pan (skyline below about 21% of the frame, foreground above 90%) and a softer left fade; shields get the softer fade too. Any explicit `--extent`, `--base`, `--relief`, `--spacing`, `--camh`, `--zoom` or `--pan-y` overrides the automatic value. `--no-shape` applies the cone rules to every landform, for comparison.

- **Close viewpoints.** Pick the viewpoint the famous photo is taken from; for horns it changes the silhouette a lot. The Matterhorn from Riffelsee (45.98333, 7.76222) shows the familiar profile with the steep north face on the right; from Zermatt (46.0170, 7.7500) the same peak reads as a broader pyramid.
- **Canyons.** Target a butte or temple inside the canyon and use a rim viewpoint. The summit refinement snaps to the highest point within 1.5 km, so label the peak it actually finds. The result is a flat far rim with the side canyons cut into it. In woodcut the inner gorge and shaded side canyons print black and the rim, buttes and lit walls stay paper with engraved lines. Topo is still the busiest style here, because every wall is a stack of contours.
- **Topo on rough ground** (horns, ranges, canyons) is drawn like a survey map: every fifth contour is heavier, intermediate contours are dropped a whole visible piece at a time where they crowd closer than about 3 px (next to the index lines first), and short index pieces that peek over a ridge are drawn at intermediate weight instead of as dark ledges. Line weight follows the light (illuminated contours), so faces turned from the light print heavier and the peak reads as a solid. Cones and shields keep plain contours.
- **Water.** Sea (elevation at or below 0 m, since the tiles clip ocean to 0) and lakes (patches the DEM holds exactly flat, 0.5 km² or more) are their own layer. Ridgelines fade out over water, a coastline is drawn, and each style marks the water its own way: sparse wave strokes (survey, nocturne), a thin shoreline plus a few sparse level lines (topo), dotted shore and waves (stipple), a flat blue tint (riso), a flat ink block with carved waves (woodcut). Views with no water render exactly as before. The default framing often keeps the sea out of view (Mauna Kea from Hilo shows none); a larger `--extent`, `--align center` or a coastal viewpoint (Vesuvius from Naples) brings it in. Dry land below sea level (Dead Sea shore, Death Valley) reads as water; `--no-water` turns the layer off.
- **Detection is tuned on seven places** (Fuji, Ararat, Matterhorn, Denali, Tetons, Grand Canyon, Mauna Kea). A new landform can land between the rules; if the draft looks wrong, set `--base`/`--extent` by hand.

## Framing knobs

| Symptom | Fix |
|---|---|
| Mountain too small or too big | `--zoom 1.3` / `--zoom 0.8` |
| Looks flat, or too spiky | `--relief 1.3` / `--relief 0.8` (vertical exaggeration) |
| Busy foreground ridges crowd the peak | `--extent` smaller (tighter scene), or `--base` higher (drops the foothills), or `--yaw ±15` |
| A neighbouring peak should show / hide | `--yaw` (orbits around the summit, degrees) |
| Want it centred (posters, wallpapers) | `--align center`; default `right` keeps the left third empty for header avatars and text |
| Lines too dense / sparse | `--spacing 0.5` / `--spacing 0.28` (auto default 0.375, wider on rough terrain) |
| Camera too high / low | `--camh` (normalised km, default 3.4; horns lower, canyons higher) |
| Snow cap too big / small | `--snowline <metres>` |
| Composition sits too high / low | `--pan-y 80` / `--pan-y -80` (reference px, 1000 = full height); an explicit value turns off the automatic fit |
| Sun or labels in the way | `--no-sun`, `--no-labels`; `--network` adds the triangulation network to other styles |
| Water drawn where there is none | `--no-water` |
| Palette | `--paper #hex --ink #hex --accent #hex` |

Everything is normalised to a reference mountain (4.3 km tall, 46 km half-width), so a 1 km hill and an 8 km giant both fill the frame the same way. `--relief` is how you restore or exaggerate real proportions.

## Sizes that work

| Target | `--size` | Notes |
|---|---|---|
| X / Twitter header | `3000x1000` (2× of 1500×500) | Profile photo covers the bottom-left, hence `--align right`. Upload limit is about 5 MB, so ship a JPG if the PNG is bigger |
| LinkedIn banner | `1584x396` or `3168x792` | Very wide; try `--zoom 1.2` |
| YouTube banner | `2560x1440` | Safe area is the centre strip; use `--align center` |
| Desktop wallpaper | `3840x2160` | `--align center` |
| Print | e.g. `6000x2000` | Render time grows with pixel count |
| Phone wallpaper (portrait) | e.g. `1170x2532` | Not a tuned layout: the scene is built for wide frames, so the mountain sits in a band across the middle. `--align center --zoom 2.5` fills the width; check the draft, or crop a wide render instead |

## Web studio

```bash
python $SK/scripts/build_studio.py --peak 35.3606,138.7274 --from 35.5100,138.7550 --from-name "Lake Kawaguchi" \
  --name "Mount Fuji" --native "富士山" --label "富士山|3776 m" --out fuji-studio.html
```

This produces one self-contained HTML file of about 0.8 MB. It embeds a 512×512 elevation grid, renders with plain Canvas 2D, and makes no network calls at runtime (Google Fonts only). It has all six styles, sliders for camera, light and colour, an orbit animation, and a 3000×1000 PNG export. The same landform-aware framing, water layer, sun placement and label lifting run in the page; `--no-shape` uses the cone rules and `--no-water` leaves water out. It takes `--from` or `--facing` like render.py and reuses the `dem_<lat>_<lon>.npz` next to `--out`. To publish it as a claude.ai artifact, declare the `downloads` capability so the Save button works; the page falls back to showing the image to save manually if the capability is unavailable. Verify it headless with Playwright if available: load it, click each style tile, and check for console errors.

The studio's drawing code is shared with the public web app at https://dogum.github.io/procedural-art/app/, which fetches elevation for any mountain in the browser. If someone only wants to try a mountain quickly and has no Python, point them there.

## Video

```bash
python $SK/scripts/animate.py --peak 39.7019,44.2986 --from 40.1792,44.4991 --style nocturne --out orbit.mp4   # 144 frames, 6 s loop
python $SK/scripts/animate.py --peak 39.7019,44.2986 --from 40.1792,44.4991 --style nocturne --out orbit.mp4 --start 0 --end 50
python $SK/scripts/animate.py --peak 39.7019,44.2986 --from 40.1792,44.4991 --style nocturne --out orbit.mp4 --start 50   # encodes when all exist
```

Frames go to `<out>_frames/`; frames that already exist are skipped, so a killed run resumes where it stopped. The folder remembers the settings: rerunning the same `--out` with another style, size, label or amplitude clears the old frames first, and only frames 0 to `--frames`−1 are encoded. Other options: `--frames`, `--fps`, `--amplitude` (degrees of swing, default 28), `--size` (default 1500×500), `--quality` (default 0.6), `--label`, `--title`. Horn and canyon orbits keep one vertical framing, so they don't bob. Background processes may be killed when a shell call returns, so run chunks in the foreground.

## Things that bite

- **Network.** Elevation comes from AWS Terrain Tiles (`s3.amazonaws.com/elevation-tiles-prod`), which needs no key and covers the whole globe, including ocean bathymetry (clipped to 0). If that domain is blocked, say so and ask the user to allow it. Void pixels in the tiles (±32768 m, seen around Denali) are filled from their neighbours when the DEM is loaded.
- **DEM size matters for the analysis.** The landform measurements use rings out to 40 km, so a default run fetches a 65 km radius. A run with a small `--extent` fetches less; a later default run into the same folder sees that the cache is too small and fetches the full area again. An explicit `--dem` that is too small only prints a warning.
- **Fonts for non-Latin labels.** The engine looks for Noto CJK (then Hiragino, Yu Gothic, MS Gothic and similar on macOS and Windows) for Chinese, Japanese and Korean, and uses DejaVu (bundled with matplotlib) for Latin, Cyrillic, Greek, Armenian and Georgian. If no CJK font is found it prints a warning and the labels render as boxes; other scripts need a font installed. Check the preview. The web studio loads the matching Noto family from Google Fonts; for Han-only labels such as 富士山 it picks Japanese, Korean, Traditional or Simplified glyph forms from the summit's location, or from `--lang ja|ko|zh-Hant|zh-Hans`.
- **Contour lines need altitude.** Seen from eye level, contours collapse into horizontal stripes. The `topo` style therefore uses a high camera with 2.4× vertical exaggeration. Keep that if you tweak it. Horns still show some of the stepped-pyramid look in topo; the light-weighted lines soften it.
- **Labels can fall outside the frame.** With `--align right`, a second peak far to the left of the summit (Mount Moran from Snake River Overlook, Mauna Loa from Hilo) may land in the faded zone, and its label is skipped. Use `--align center` or `--yaw` if it matters.
- **Near-edge artefacts.** If vertical streaks appear under the terrain, something is sampling below the grid's front edge. The G-buffer masks those pixels; keep that mask if you edit the engine.
- **Seeing the output.** Always view a downscaled preview before delivering. Two render passes with a look in between beat five blind tweaks.

## Extending

Read `references/engine.md` before adding a style or changing the camera or the auto framing. It covers the data loader, the landform analysis, the coordinate frames (reference 3000×1000 space versus output pixels, which was the source of most bugs while building this), the floating horizon, the G-buffer, a template for a new style function, and how the web template mirrors the Python.

`scripts/dem.py` (tile download, coverage check and loading with void filling) is shared with procedural-realism. Both skills ship an identical copy; if you change it, change both.
