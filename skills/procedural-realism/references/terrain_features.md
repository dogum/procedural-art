# Terrain features: water, trees and villages, clouds, night city, timelapse

All of these are opt-in flags on `pbr_terrain.py`; with none of them you get the plain render. Run the commands from a writable folder with `SK` set to the skill's folder and `S=$SK/scripts`.

## Recipes used for the showcase finals

All at 3000×1000, then `post_terrain.py` with the defaults unless noted. Draft at 900×300 first (change `--size`; band folders are per size and tag). Each recipe has its own `--tag`, so recipes that share a place and size never meet in one band folder, and the steps are chained with `&&`, so post only runs when the render succeeded.

**Fuji mirrored in Lake Kawaguchi** (12 min; 52k trees and houses in view, water rows are 70% of the time):

```bash
python $S/pbr_terrain.py --peak 35.3606,138.7274 --from 35.5215,138.7460 --cam-offset 0 --cam-height 0.004 \
  --hfov 68 --layout center --horizon 0.5 --water-level auto --waves 0.12 --base 900 --mist 1.0 --haze 0.15 \
  --biome temperate --treeline 2400 --snowline 2900 --trees 0.8 --villages 0.6 --near-km 5 --cloud-model ms \
  --size 3000x1000 --work work/fuji --tag lake && \
python $S/post_terrain.py --bands work/fuji/bands_3000x1000_lake --out fuji_kawaguchi_reflection.png
```

**Ararat over orchards and villages** (6 min; low camera, sun raised to 11° so tree shadows don't streak, mist lowered because the camera sits in it):

```bash
python $S/pbr_terrain.py --peak 39.7019,44.2986 --from 40.1792,44.4991 --second 39.6517,44.4011 \
  --cam-height 0.03 --trees 0.8 --villages 0.7 --near-km 6 --cloud-model ms --sun-el 11 --mist 0.6 --haze 0.18 \
  --size 3000x1000 --work work/ararat --tag orchards && \
python $S/post_terrain.py --bands work/ararat/bands_3000x1000_orchards --out ararat_orchards_villages.png
```

**Village and orchards close up, Ararat behind** (camera 3 m up at the edge of a village on the Ararat plain: a dirt lane leads past a house 60 m away, yard walls, orchard and poplar rows; the physical sky keeps shadows from going black; about 8 min):

```bash
python $S/pbr_terrain.py --peak 39.7019,44.2986 --from 39.92877,44.42231 --second 39.6517,44.4011 \
  --cam-offset 0 --cam-height 0.003 --hfov 50 --layout left --trees 0.8 --villages 0.7 --near-km 4 \
  --cloud-model ms --sun-el 16 --mist 0.5 --haze 0.18 --sky physical \
  --size 3000x1000 --work work/ararat --tag village && \
python $S/post_terrain.py --bands work/ararat/bands_3000x1000_village --out ararat_village_close.png --exposure 0.6
```

**Ararat over Yerevan at blue hour** (5 min; 19k street lamps on the real streets, 20k lit window points and car light streaks in view; the sun has set behind Ararat's western shoulder, as in late autumn; the first run also fetches Yerevan's roads from OpenStreetMap, which took 11 minutes through busy mirrors):

```bash
python $S/pbr_terrain.py --peak 39.7019,44.2986 --from 40.1963,44.5238 --second 39.6517,44.4011 \
  --cam-offset 0 --cam-height 0.15 --hfov 34 --sun-az 245 --sun-el -5 --air 1.0 --city 0.8 --trees 0.5 \
  --near-km 8 --cloud 0 --size 3000x1000 --work work/yerevan --tag bluehour && \
python $S/post_terrain.py --bands work/yerevan/bands_3000x1000_bluehour --out ararat_yerevan_blue_hour.png --exposure "auto*0.3"
```

Road data © OpenStreetMap contributors (ODbL).

**Ararat on a moonlit night** (about 5 min, twice a plain render because of the moon shadows and a second sky pass; full moon from the side, point stars; post picks the night exposure):

```bash
python $S/pbr_terrain.py --peak 39.7019,44.2986 --from 40.1792,44.4991 --second 39.6517,44.4011 --time night \
  --size 3000x1000 --work work/ararat --tag night && \
python $S/post_terrain.py --bands work/ararat/bands_3000x1000_night --out ararat_night.png
```

**Sunset timelapse** (1200×400, 480 frames = 20 s at 24 fps, every frame rendered). With the geometry cache warm a frame takes about 50 s with the sun up and less after sunset, so the clip takes about 6 hours; the first frame builds the caches and takes about 80 s. At 600×200 a frame takes 13–16 s, so a 5 s draft around sunset (`--frames 121 --start 19:26 --end 19:56`) takes about 30 minutes:

```bash
python $S/timelapse.py --work work/tl --out ararat_sunset_timelapse.mp4 --frames 480 --size 1200x400 \
  --date 2026-04-20 --utc-offset 4 --start 18:40 --end 20:40 -- \
  --peak 39.7019,44.2986 --from 40.1963,44.5238 --second 39.6517,44.4011 --cam-offset 0 --cam-height 0.05 \
  --hfov 34 --air 0.8 --city 0.8 --trees 0.5 --near-km 8 --near-lamps 16 --aperture 40
```

**Vertical 4:5 for feeds** (1080×1350). Drop `--second`, centre the summit and use a long lens so Great Ararat stands tall over the city, with the lenticular above it and rooftops across the bottom third: `--layout center --hfov 10.5 --horizon 0.66` in place of `--second … --hfov 34`. A frame takes about 2 minutes (20–27 s at 432×540), so 480 frames take about 15 hours; `--blend 2` halves that. At this focal length a pixel covers about 10 m on the mountain against a 58 m DEM, so the slopes are softer than in the landscape view.

Viewpoints: Lake Kawaguchi just off the Oishi Park shore (lake surface 830 m in the DEM, 833 m published); Victory Park in Yerevan (40.1963, 44.5238). A separate `--work` folder per place avoids refetching elevation when you alternate between them.

## pbr_terrain.py flags

| flag | meaning |
|---|---|
| `--tag NAME` | band folder suffix, so variants of one view don't collide |
| `--horizon F` | put the true horizon at fraction F of the height from the top (overrides the auto pitch; `--pitch` still adds) |
| `--time day\|golden\|blue\|night` | sun elevation preset (35 / 6 / −4.5 / −14°); blue and night switch on the physical sky; night adds moonlight and stars |
| `--moon M`, `--moon-az DEG`, `--moon-el DEG` | with `--time night`: moonlight strength (1 full moon, 0.3 crescent, 0 starlight only), direction (default: lights the faces the camera sees) and elevation (28°) |
| `--sky classic\|physical` | the plain daylight sky, or Earth shadow + ozone + twilight + sky-derived ambient. Default: physical for `--time blue/night`, `--sun-el` below 1.5°, or `--city`; classic otherwise |
| `--cloud 1\|0` | lenticular cap over the summit (default on) |
| `--cloud-model classic\|ms` | lenticular lighting: single scattering, or multiple scattering + dual-lobe/silver-lining phase |
| `--cloud-layer none\|cumulus\|stratus`, `--cloud-cover C` | extra cloud slab above the valley (coverage 0..1, default 0.45) |
| `--biome arid\|temperate`, `--treeline M`, `--season summer\|autumn\|spring` | ground palette; temperate = forest / meadow / fields / scoria for green mountains. Treeline default 55% of the way up |
| `--water-level M\|auto`, `--water-seed LAT,LON` | lake/sea surface in metres and which basin; `auto` = the largest flat basin within 12 km of the camera, accepted only if it is level (within 3 m across the whole flat patch) and flooding it stays inside it; otherwise the render continues without water and says why. With `auto`, a seed must be on (or within 500 m of) a flat water surface |
| `--waves S` | 0 mirror, 0.1 calm morning, 0.5 calm lake, 1 breeze (default), 3 choppy |
| `--wind DEG`, `--water-depth M`, `--water-spp N`, `--water-color lake\|clear\|sea` | wave direction, synthetic bed depth (15 m), reflection samples per pixel (3), absorption preset |
| `--trees D`, `--tree-mix c,d,p` | tree density 0–1; conifer/deciduous/poplar weights (default from elevation and biome) |
| `--villages D` | village density 0–1 (houses with pitched roofs, yards, orchards, poplar rows) |
| `--near-km K` | trees and houses only within K km of the camera (bounds cost; default 6) |
| `--seed N` | layout seed for trees, villages, city |
| `--city D`, `--city-km K` | night city density and extent (street lamps, lit windows, sky glow; default 12 km) |
| `--streets osm\|grid` | with `--city`: real streets from OpenStreetMap (default; needs the Overpass API once, then cached in `<work>/osm`) or the synthetic grid |
| `--traffic T`, `--shutter S` | with OSM streets: car lights on trunk to tertiary roads (0 = off, default 0.5); exposure in seconds, which sets the streak length (default 0.5) |
| `--near-lamps N` | with `--city`: a string of N lamps 3.5–6 m in front of the lens (bokeh) |
| `--aperture MM`, `--focus KM` | thin-lens depth of field and bokeh (applied in post; focus defaults to the peak). Changing them on a finished folder doesn't re-render it |
| `--geom-cache` | keep what doesn't depend on the sun in `--work` (ray hits and near-field visibility per band, the blurred heightfield, the sun transmittance table); timelapses |
| `--radius KM` | terrain radius around the summit (default 45) |

`post_terrain.py`: `--exposure 0.8|auto|auto*k` (log-average key, biased by k; the default is 0.8, or `auto*0.25` for a `--time night` render), `--partial` (post a `--rows` crop on purpose; otherwise missing rows are an error), `--aperture` / `--focus` (refocus without re-rendering), `--light-gain`, `--bokeh-rim` (0 = flat discs), `--no-lights`, `--no-dof`, `--grain`, `--save-hdr FILE.npy`, `--hdr-in FILE.npy` (tone-map a linear image instead of bands), `--seed` (grain per video frame), `--wb K|auto|sun` (white balance; auto = 4800 K at blue hour, else the sun as white), `--sat S` (saturation; auto 1.15 at blue hour). At blue hour (physical sky, sun below +1°, not `--time night`) `auto` keys on the sky and terrain beyond 30 km, measured before the city lights are added.

`timelapse.py`: `--work`, `--out`, `--size` (1200×400), `--frames` (480 = 20 s), `--blend K` (render every K-th frame and cross-fade the rest; 1 = render all), `--fps` (24), `--date YYYY-MM-DD`, `--utc-offset` (hours), `--start`/`--end` (local HH:MM), `--sun-path az0,el0:az1,el1` (straight sweep instead of a date), `--night-bias` (0.4), `--key` (0.30), `--max-exposure` (14), `--smooth` (exposure smoothing in minutes of sun time, 4.5), `--only-plan`. The old `--keyframes N --interp I` still work and mean `--frames (N−1)·I+1 --blend I`. Everything after `--` goes to `pbr_terrain.py`. Frames are rendered with `--sky physical --geom-cache` into `<work>/render_<hash>` as half-float linear images named by sun position (the hash covers size, pbr arguments and the renderer's source files), so a rerun resumes and a longer clip or another `--blend` reuses every sun position it shares. Tone-mapped frames go to `<work>/frames_<hash>`, keyed also by the plan, `--blend` and exposure settings. Odd frame sizes are scaled to even ones for the encoder.

## How to use each feature

**Water.** DEM lakes are flat from shore to shore, so `--water-level auto` finds them. Irrigated plains are locally flat too (the Ararat plain from Yerevan has 600 km² of them), but they rise tens of metres across their extent, so `auto` rejects them and renders dry. A lake whose flat surface runs into a flat valley floor (Jackson Lake under the Tetons) is rejected the same way; give its level in metres plus `--water-seed lat,lon`, as for the sea. The lake reflects terrain, sky and cloud (a second ray is traced), shows a synthetic bed in the shallows, and glints in the sun. Stand on the water: `--cam-offset 0 --cam-height 0.004` (4 m up) and `--horizon 0.5` so the mountain and its reflection both fit. `--waves 0.1` is a morning mirror; 0.5 and up break the reflection into streaks. Post smears water pixels vertically by the per-pixel reflection spread, which removes the sparkle noise and gives the streaky look of real lake reflections. A dry 900×300 view takes 7–10 s; with water about 40 s.

**Trees and villages.** Instances sit on jittered grids: trees (8.5 m cells), poplar rows on the field-edge lines, houses on 26 m plots with lanes along every fifth and fourth plot. In the arid biome, fields at the edge of villages within 1.5 km are orchards planted in rows (5.5 m grid, irrigated grass under them). Within 600 m, dry-stone walls, hedges and post-and-rail fences run the full length of some field edges where the edge is fairly straight (with a gate every 60 m; walls under 3 px tall are dropped) and along the sides of house plots (one gate per side). Winding dirt lanes cross the fields within about a kilometre; nothing is placed on them. Everything is masked by slope, elevation and treeline, water and shore, forest noise and village density, and cached per layout (`near_<hash>.npz`, rebuilt when `terrain_near.py` changes). Trees are whorled conifers, deciduous crowns or flame-shaped poplars; they cast shadows on the ground and on each other and are reflected in water. Detail follows the pixels a tree spans: far crowns are solid lumpy shapes; once clumps span a few pixels, a deciduous crown becomes a dense core with fourteen leaf clumps, each on a branch from the top of the trunk, so sky shows between clumps but nothing floats. Leaf texture on close crowns is shading only. Light inside a crown falls off with the foliage between the point and the sun, so crowns shade themselves, shadows under trees are dappled, and backlit edges glow. Close up, houses show tuff, basalt, plaster or boards, tile, tin or slate roofs, a basalt plinth, framed windows (also in the gable), a front door and often a gable door, a chimney, a gutter with its shadow line, mottled plaster with rain streaks and patches; field and yard walls are dry stone, 0.9–1.2 m high, in rough courses of 20–35 cm stones under a level capstone course, with grass at the foot. The ground near the lens gets green-to-gold crop and grass fields, standing grass tufts and blades, wild flowers, crop rows and clods lit by the low sun, rutted farm tracks and lanes, all faded out by pixel size. With `--cam-height 0.03` the near field starts a few hundred metres out; with 0.002–0.005 (2–5 m, standing in a field) trees and houses from 40 m on hold up at 3000 px. `--near-km` bounds the cost. With `--biome temperate --treeline 2400` the forest follows the same noise that colours the ground.

**Twilight and night.** `--time night` (sun at −14°) lights terrain and sky with a full moon (`--moon`, `--moon-az`, `--moon-el`) and adds a star field. The stars are kept in a separate layer that post adds after bloom, so they stay single-pixel points; post uses `auto*0.25` exposure and a cool, desaturated grade unless you pass `--exposure`. `--moon 0` leaves starlight only: silhouettes under a dark starry sky. `--time blue` or `--sun-el -4` switches to the physical sky: Earth shadow, pink Belt of Venus, alpenglow on summits that stay lit after sunset, blue hour. Below about 8° of sun the sunlight reaching the sky is computed per wavelength (ozone's Chappuis band, Rayleigh, haze) and projected on sRGB, and multiple scattering comes from a precomputed table after Hillaire (2020), so the zenith turns blue from ozone rather than lavender and the sky darkens after sunset at about the measured rate (−6° is roughly 250× darker than sunset). Above 10° the sky is unchanged. At −4° the sky 90° from the sunset is still pastel; a deep blue sky over the city reads best at −5° to −6°, or facing the afterglow. `--city 0.8` lights a city on its real streets: `scripts/osm_roads.py` fetches the OpenStreetMap highways in the view wedge through the Overpass API (one query per 0.05° tile, several public mirrors tried in turn, cached in `<work>/osm`) and rasterises them to an 8 m grid of distance, class, width and direction. Road length per area sets the city density, so parks, gorges and fields stay dark. Poles stand on the kerbs every 32–42 m by road class, on both sides of avenues; avenues get white LED, side streets mostly orange sodium, one type per street. Buildings front the nearest street (facade parallel to it, set back 3–14 m), apartment blocks line the bigger roads, and plots deep inside a block are rarer and mostly dark. `--traffic 0.5` puts cars in the lanes of trunk to tertiary roads (right-hand traffic): white head lights toward the lens, red tail lights away, each a streak as long as the car moves during `--shutter` seconds. Every lamp lights a pool on the ground and washes the walls next to it, and the sky glow follows the lamp power per 200 m. Road data © OpenStreetMap contributors (ODbL). With no network, or `--streets grid`, the city falls back to the synthetic street grid. Post with `--exposure "auto*0.3"`: at blue hour post keys the auto exposure on the sky and far terrain (the lights and a dark foreground no longer set it) and sets the white balance to 4800 K with saturation 1.15, the way a blue-hour photographer sets the camera (`--wb sun --sat 1` turns this off). Lights fade in with sun elevation (windows from 1.5° to −3.5°, lamps from 1° to −1°), so a timelapse turns the city on gradually. Realistic apertures only blur lights within about 100 m, which is why `--near-lamps 16 --aperture 40` adds a string of lamps a few metres from the lens for bokeh. DOF and bokeh are post effects, so `post_terrain.py --aperture 0` shows the pinhole version without re-rendering.

**Clouds.** `--cloud-model ms` lights the lenticular with octave multiple scattering and a silver-lining lobe: brighter interior, bright edges against the sun. `--cloud-layer stratus` looks plausible; `cumulus` has the right flat bases and domes but speckled edges and casts no shadows on the ground, so treat it as experimental.

**Timelapse.** The sun follows the real solar path for the date, the `--from` viewpoint and the local time window, and every frame is rendered with the sun where it is at that instant: building shadows lengthen and the Earth-shadow line climbs the mountain frame by frame. A 20 s clip over two hours puts 15 s of sun time between frames. Exposure follows the log-average key smoothed over 4.5 minutes of sun time, darker after sunset (`--night-bias`). `--blend K` is the old method: render every K-th frame and cross-fade the rest in linear light. It is K times cheaper, but shadows fade instead of moving once rendered frames are more than about a minute of sun apart; `--blend 2` is hard to tell from the full render. Refraction lifts the sun by up to 0.65° near the horizon and tapers to nothing between −1° and −3°, so the light never jumps between frames. The Yerevan run went from sun +11.5° to −10°: golden light, sunset, pink twilight, blue hour, night, city lights fading in.

## Things that bite

- **Low cameras sit in the mist.** At 30 m the camera is inside the valley-mist layer and every ray crosses tens of km of it; the mountain washes out. Lower `--mist` (about 0.6).
- **Standing in the scene.** A camera 2–5 m up sees the ground from 20 m on and trees and houses from 30–40 m (the lens is kept clear of anything within about twice a tree's height). Use `--sky physical`: the classic sky's ambient is so weak that shadows under trees go black. The physical sky is brighter, so post with `--exposure 0.6`. Pick a spot with a subject 50–150 m out (a house, a lane leading in, orchard rows) rather than an empty field; a nearby tree in front of the summit is easiest fixed by moving the camera 10–20 m sideways (`--from` with `--cam-offset 0`).
- **Low sun + low camera = streak shadows.** Tree shadows at 10° sun are 6× the tree height and, seen from 30 m, become thin horizontal lines. Correct geometry, but it reads as an artefact; raise the sun to 12–15° for foregrounds.
- **Mirror lakes need calm water.** Resolved ripples at mid distance smear the reflection; `--waves 0.1` keeps the mountain readable.
- **City density.** Only tall blocks looks like wallpaper at 3000 px. Mostly low buildings, parks and trees, with a few towers, reads as a place.
- **Physical sky is brighter.** Its ambient at +10° is 5–9× the classic constant (the classic one is darker than a real sky), which is why it is opt-in for daylight.
- **Resuming.** Each band folder stores `params.json`; resuming with different pixel-affecting settings stops with a message and leaves a `refused.json`, which makes `post_terrain.py` refuse the folder until a matching run. Use `--tag` for variants. Rows missing from a folder (a killed run, or a `--rows` chunk that ended mid-band) are filled by the next run, and post names them if you post too early.
- **Night with a city.** At full night the city dominates and its box buildings look synthetic; blue hour (`--sun-el -5`, `--exposure "auto*0.3"`) reads much better. At −4° the sky 90° from the sunset is still pastel; face the afterglow or go to −5°/−6°.
- **Memory.** The mirror pass is chunked like the primary pass; unchunked it took 3.2 GB at 3000 px.

## Limits

The remaining weak points (synthetic lake bed, static waves, close trees and houses, box cities, slow timelapses, layered depth of field) are listed in `references/lessons.md`.
