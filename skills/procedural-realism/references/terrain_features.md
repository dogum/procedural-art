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

**Ararat over Yerevan at blue hour** (6 min; about 40k lights in view; the festoon lamps near the lens make the bokeh):

```bash
python $S/pbr_terrain.py --peak 39.7019,44.2986 --from 40.1963,44.5238 --second 39.6517,44.4011 \
  --cam-offset 0 --cam-height 0.05 --hfov 34 --sun-az 290 --sun-el -4 --air 1.0 --city 0.8 --trees 0.5 \
  --near-km 8 --near-lamps 16 --aperture 40 --size 3000x1000 --work work/yerevan --tag bluehour && \
python $S/post_terrain.py --bands work/yerevan/bands_3000x1000_bluehour --out ararat_yerevan_blue_hour.png --exposure "auto*0.45"
```

**Ararat on a moonlit night** (about 5 min, twice a plain render because of the moon shadows and a second sky pass; full moon from the side, point stars; post picks the night exposure):

```bash
python $S/pbr_terrain.py --peak 39.7019,44.2986 --from 40.1792,44.4991 --second 39.6517,44.4011 --time night \
  --size 3000x1000 --work work/ararat --tag night && \
python $S/post_terrain.py --bands work/ararat/bands_3000x1000_night --out ararat_night.png
```

**Sunset timelapse** (1200×400, 40 keyframes blended 4× = 157 frames at 24 fps; 40 keyframes took 36 min with the geometry cache, frames and encoding about 3 min more):

```bash
python $S/timelapse.py --work work/tl --out ararat_sunset_timelapse.mp4 --keyframes 40 --interp 4 --size 1200x400 \
  --date 2026-04-20 --utc-offset 4 --start 18:40 --end 20:40 -- \
  --peak 39.7019,44.2986 --from 40.1963,44.5238 --second 39.6517,44.4011 --cam-offset 0 --cam-height 0.05 \
  --hfov 34 --air 0.8 --city 0.8 --trees 0.5 --near-km 8 --near-lamps 16 --aperture 40
```

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
| `--near-lamps N` | with `--city`: a string of N lamps 3.5–6 m in front of the lens (bokeh) |
| `--aperture MM`, `--focus KM` | thin-lens depth of field and bokeh (applied in post; focus defaults to the peak). Changing them on a finished folder doesn't re-render it |
| `--geom-cache` | cache sun-independent geometry per band (timelapses) |
| `--radius KM` | terrain radius around the summit (default 45) |

`post_terrain.py`: `--exposure 0.8|auto|auto*k` (log-average key, biased by k; the default is 0.8, or `auto*0.25` for a `--time night` render), `--partial` (post a `--rows` crop on purpose; otherwise missing rows are an error), `--aperture` / `--focus` (refocus without re-rendering), `--light-gain`, `--bokeh-rim` (0 = flat discs), `--no-lights`, `--no-dof`, `--grain`, `--save-hdr FILE.npy`, `--hdr-in FILE.npy` (tone-map a linear image instead of bands), `--seed` (grain per video frame).

`timelapse.py`: `--work`, `--out`, `--size` (1200×400), `--keyframes` (40), `--interp` (output frames per keyframe step, 4), `--fps` (24), `--date YYYY-MM-DD`, `--utc-offset` (hours), `--start`/`--end` (local HH:MM), `--sun-path az0,el0:az1,el1` (straight sweep instead of a date), `--night-bias` (0.4), `--key` (0.30), `--max-exposure` (14), `--only-plan`. Everything after `--` goes to `pbr_terrain.py`. Keyframes are rendered with `--sky physical --geom-cache` into `<work>/keys_<hash>` and frames go to `<work>/frames_<hash>`; the hashes cover the sun plan, size, pbr arguments, `--interp` and exposure settings. A keyframe is done when its linear `.npy` exists, so rerunning resumes, and changing any setting starts fresh folders instead of reusing stale frames. Odd frame sizes are scaled to even ones for the encoder.

## How to use each feature

**Water.** DEM lakes are flat from shore to shore, so `--water-level auto` finds them. Irrigated plains are locally flat too (the Ararat plain from Yerevan has 600 km² of them), but they rise tens of metres across their extent, so `auto` rejects them and renders dry. A lake whose flat surface runs into a flat valley floor (Jackson Lake under the Tetons) is rejected the same way; give its level in metres plus `--water-seed lat,lon`, as for the sea. The lake reflects terrain, sky and cloud (a second ray is traced), shows a synthetic bed in the shallows, and glints in the sun. Stand on the water: `--cam-offset 0 --cam-height 0.004` (4 m up) and `--horizon 0.5` so the mountain and its reflection both fit. `--waves 0.1` is a morning mirror; 0.5 and up break the reflection into streaks. Post smears water pixels vertically by the per-pixel reflection spread, which removes the sparkle noise and gives the streaky look of real lake reflections. A dry 900×300 view takes 7–10 s; with water about 40 s.

**Trees and villages.** Instances sit on jittered grids: trees (8.5 m cells), poplar rows on the field-edge lines, houses on 26 m plots. They are masked by slope, elevation and treeline, water and shore, forest noise and village density, and cached per layout (`near_<hash>.npz`). Trees are conifer cones, lumpy deciduous crowns or flame-shaped poplars, all with trunks; houses have gables and overhanging roofs; they cast shadows on the ground and on each other and are reflected in water. They only read with a low camera (`--cam-height 0.03` or less) so the foreground is a few hundred metres away. `--near-km` bounds the cost. With `--biome temperate --treeline 2400` the forest follows the same noise that colours the ground.

**Twilight and night.** `--time night` (sun at −14°) lights terrain and sky with a full moon (`--moon`, `--moon-az`, `--moon-el`) and adds a star field. The stars are kept in a separate layer that post adds after bloom, so they stay single-pixel points; post uses `auto*0.25` exposure and a cool, desaturated grade unless you pass `--exposure`. `--moon 0` leaves starlight only: silhouettes under a dark starry sky. `--time blue` or `--sun-el -4` switches to the physical sky: Earth shadow, pink Belt of Venus, alpenglow on summits that stay lit after sunset, blue hour. `--city 0.8` builds street lamps on street centre lines, lit windows (per-building occupancy, four colour temperatures) and window points for far buildings, plus ground light and sky glow. Post with `--exposure "auto*0.45"` so it still looks like dusk. Lights fade in with sun elevation (windows from 1.5° to −3.5°, lamps from 1° to −1°), so a timelapse turns the city on gradually. Realistic apertures only blur lights within about 100 m, which is why `--near-lamps 16 --aperture 40` adds a string of lamps a few metres from the lens for bokeh. DOF and bokeh are post effects, so `post_terrain.py --aperture 0` shows the pinhole version without re-rendering.

**Clouds.** `--cloud-model ms` lights the lenticular with octave multiple scattering and a silver-lining lobe: brighter interior, bright edges against the sun. `--cloud-layer stratus` looks plausible; `cumulus` has the right flat bases and domes but speckled edges and casts no shadows on the ground, so treat it as experimental.

**Timelapse.** The sun follows the real solar path for the date, the `--from` viewpoint and the local time window. Exposure follows a smoothed log-average, darker after sunset (`--night-bias`), and in-between frames are blended in linear light. The Yerevan run went from sun +11.5° to −10°: golden light, sunset, pink twilight, blue hour, night, city lights fading in.

## Things that bite

- **Low cameras sit in the mist.** At 30 m the camera is inside the valley-mist layer and every ray crosses tens of km of it; the mountain washes out. Lower `--mist` (about 0.6).
- **Low sun + low camera = streak shadows.** Tree shadows at 10° sun are 6× the tree height and, seen from 30 m, become thin horizontal lines. Correct geometry, but it reads as an artefact; raise the sun to 12–15° for foregrounds.
- **Mirror lakes need calm water.** Resolved ripples at mid distance smear the reflection; `--waves 0.1` keeps the mountain readable.
- **City density.** Only tall blocks looks like wallpaper at 3000 px. Mostly low buildings, parks and trees, with a few towers, reads as a place.
- **Physical sky is brighter.** Its ambient at +10° is 5–9× the classic constant (the classic one is darker than a real sky), which is why it is opt-in for daylight.
- **Resuming.** Each band folder stores `params.json`; resuming with different pixel-affecting settings stops with a message and leaves a `refused.json`, which makes `post_terrain.py` refuse the folder until a matching run. Use `--tag` for variants. Rows missing from a folder (a killed run, or a `--rows` chunk that ended mid-band) are filled by the next run, and post names them if you post too early.
- **Night with a city.** At full night the city carpet dominates and looks synthetic; blue hour (`--sun-el -4`, `--exposure "auto*0.45"`) reads much better.
- **Memory.** The mirror pass is chunked like the primary pass; unchunked it took 3.2 GB at 3000 px.

## Limits

The remaining weak points (synthetic lake bed, static waves, close trees and houses, box cities, cross-faded timelapse shadows, layered depth of field) are listed in `references/lessons.md`.
