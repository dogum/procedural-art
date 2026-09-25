# Engine internals

Files: `scripts/dem.py` (elevation download and loading), `scripts/terrain_art.py` (landform analysis, terrain, camera, helpers), `scripts/render.py` (styles, framing fit, CLI), `scripts/animate.py` (orbit video), `scripts/build_studio.py` + `assets/studio_template.html` (the same engine ported to Canvas 2D JavaScript, config-driven).

The scripts never write into the skill folder, which may be read-only: images, DEM caches and frames go next to `--out` (relative to the working folder), and `sys.dont_write_bytecode` keeps `__pycache__` out of it. Only `assets/` and the scripts themselves are read from the skill folder.

## 1. Data

`scripts/dem.py` holds `fetch_dem()`, `ensure_dem()` and `load_dem()`. It is shared with procedural-realism: both skills ship a byte-identical copy, so each installs on its own. Change both or neither. `terrain_art.py` re-exports these functions.

- `fetch_dem(lat, lon, radius_km, out)` downloads Terrarium PNG tiles from AWS Terrain Tiles and decodes each pixel as `R*256 + G + B/256 - 32768` metres. The zoom level is picked from the radius (z13 up to 15 km, z12 up to 70 km, z11 up to 160 km, z10 beyond) and lowered until the square fits in 260 tiles. Each tile is retried up to three times, and tile x wraps across the antimeridian. The mosaic is saved to exactly `out` (through a temporary file) as `.npz` with per-row latitudes (Web Mercator, non-linear), per-column longitudes (linear), the zoom and the requested radius.
- `ensure_dem(lat, lon, radius_km, path, explicit)` reuses `path` when it covers `radius_km` around the point (measured from its lat/lon arrays, so older caches work too), and refetches it otherwise. For a user-given `--dem` (`explicit=True`) it only warns.
- `load_dem(path)` returns `(elev, lats, lons)`. Void pixels (outside −12000..9000 m, such as the ±32768 values in the Denali tiles) are filled from their nearest valid neighbour, so they can't become spikes or pits. DEMs without voids come back unchanged.
- `render.py`, `animate.py` and `build_studio.py` need a radius of `(extent or 45) × 1.45` km, 65 km by default, which the landform analysis needs. All three cache it as `dem_<lat>_<lon>.npz` next to `--out`.

## 2. Terrain grid (the view frame)

`terrain()` builds a grid of `NZ` rows × `NX` columns aligned with the camera:
- `u` runs left to right across the view, from -46 to 46.
- `v` is depth away from the viewer, from -35 (near) to 30 (far). The summit sits at `(0, 0)`.
- The view direction is the unit vector from `view_from` to the summit (or from `facing`), rotated by `yaw`. The right vector is that direction turned 90° clockwise.

**Normalisation.** A real distance in km equals `u × k`, where `k = extent_km / 46`. The drawn height is `(elev − base) / 1000 × hs`, where `hs = 4.3 / (peak − base in km)`. Every tuned constant (fade widths, presence thresholds such as `(h − 0.15) / 0.9`, contour step 0.04, row spacing 0.375) lives in these normalised units, which is why any mountain works without retuning. Use `t.m_to_h(metres)` to convert real elevations such as the snowline, and `t.to_uv((lat, lon))` to place peaks.

**Auto values.**
- The summit is refined to the DEM maximum within 1.5 km of the given point.
- `analyse_shape()` measures the landform on rings around the summit (see its docstring for the metrics) and returns three scores: `horn` (steep peak with high ground in front of it), `rough` (alpine texture) and `canyon` (the target sits below the surrounding plateau). Cones score 0 on all three. It also returns a `kind` label (canyon / horn / range / massif / shield / cone), which is only a label: the rules below use the scores.
- Base: the 20th percentile on the 0.35..1 × 40 km ring (the cone rule), raised towards the local base in the viewer-facing sector by the horn score. For canyons, the 3rd percentile within 12 km (the floor).
- Height scale: the peak, or for canyons the rim (90th percentile of the 3..20 km ring), maps to 4.3.
- Extent: `clip(10.5 × height_km × (1 − 0.4·horn) × (1 − 0.1·rough), 8, 60)`, limited by how much DEM was fetched.
- Relief, line spacing, extra smoothing and camera height come from the same scores. The analysis ignores `yaw`, so orbits keep one framing.
- `terrain(..., shape_aware=False)` (CLI `--no-shape`) applies the cone rules to every landform. The analysis result is stored as `t.shape` (metrics, scores, suggested values) and `t.top_m`.

**Checking a change.** Cones must stay pixel-identical. Before editing, render Fuji, Ararat and Mauna Kea in all six styles at draft quality with a fixed `--dem`, then render them again with the edited engine and compare the PNGs (maximum difference 0/255). Also look at one horn, one range and one canyon, since those use the shape rules.

## 3. Camera

The camera is defined in a **reference frame 3000 wide × 1000 tall**. For a point `(u, v, h)`:

```
d  = v − ZCAM                          (ZCAM = −95)
sx = s · (cx + fx·u/d)
sy = s · (hy − fy·(h·hmul − camh)/d)
s  = W / 3000
```

Defaults: `fx 4500`, `fy 10200` (about 2.3× vertical exaggeration on screen), `camh 3.4`, `cx 2150`, `hy 290`. The `topo` style uses `camh 24`, `fy 3000`, `hmul 2.4` and `hy −110`.

For frames that aren't 3:1, `oy` shifts `hy` so the composition stays vertically centred. `fadeX = cx − 1150` is where the left-side fade begins, in reference px. It is disabled for `align=center`.

`fit_offset()` in `render.py` pans vertically for horns and canyons only, so the skyline stays below 21% of the frame (12% without labels, leaving room for a label) and the foreground above 90%. `animate.py` measures it once at yaw 0 and passes it to every frame. `Scene.fade_pow` squares the left fade for those shapes, because their flattened plains are dense stacks of rows that otherwise start as a hard edge.

**The one rule that prevents most bugs:** `cx`, `hy`, `fadeX` and every hard-coded offset are *reference* units, while `Y`, `CM`, silhouette values and pixel indices are *output* units. Convert with `× s` or `/ s`, always explicitly. Expressions like `sc.hy + 270 * s` mix the two frames and were the main source of mispositioned moons and fades while building this.

## 4. Floating horizon and G-buffer

- `Y[i, px]` is the screen y of row `i` at column `px`, or `inf` where the row doesn't cover that column.
- `CM = minimum.accumulate(Y)` over rows from near to far. `CM[i]` is the horizon after drawing rows `0..i`.
- A ridgeline point on row `i` is visible if `Y[i] < CM[i-1] − 0.3·s`.
- An arbitrary surface point (contours, network nodes) is visible if `sy ≤ CM[row_below] + tol`.
- The **G-buffer** records, for each pixel, the owning row: the first row `i` with `CM[i] ≤ y`, found by `searchsorted` down each column. From the owning row and `u` you can sample any grid field (height, shade, snow) per pixel. Pixels below the nearest row's projection (`y > Y[0]`) are *not* terrain and must be masked; forgetting this causes vertical streaks when the foreground is hilly. `G["edge"]` holds the combined side, near and far fades. Multiply it into anything drawn from the G-buffer.

## 5. Lighting

`light = (right, away, up)` is in view space, so it rotates with the camera. For example, `(-0.75, -0.25, 0.6)` is light from the left and slightly behind the viewer. `shade = max(0, n·L)`. The snow mask combines height above the snowline with a slope limit.

## 6. Writing a new style

```python
def style_blueprint(t, sc, cfg):
    W, H, s = sc.W, sc.H, sc.s
    ink = hexrgb(cfg["ink"])
    def alpha(c):                      # c: i, px (array), u (array), v, depth 0..1
        hh = sc.sample(t.h, c["i"], c["px"]); sh = sc.sample(t.shade, c["i"], c["px"])
        pres = np.clip((hh - 0.15) / 0.9, 0, 1) * sc.edge_fade(c["u"], c["v"]) * sc.xfade(c["px"])
        return pres * (0.4 + 0.6 * (1 - sh))
    segs, cols, wids = sc.ridgelines(cfg["spacing"], alpha, lambda c: 1.2 - 0.5 * c["depth"], ink)
    bg = np.ones((H, W, 3)) * hexrgb(cfg["paper"])
    fig, ax = figure(bg); add_lines(ax, segs, cols, wids)
    return fig                         # or return an (H, W, 3) float array for pixel styles
```

Then register it in `RENDERERS`, `STYLES` and `DEFAULTS` (palette and light), and in `CAM` if it needs its own camera.

Tools already available:
- `sc.gbuffer({...})` for per-pixel styles.
- `sc.sky_mask()` to clip suns, moons and stars behind the silhouette.
- `sc.peaks_screen(latlons)` for label and sun anchors, and `sun_anchor()` to place a sun in the saddle or beside a lone peak.
- `sun_spot(sc, centre, r, peak, avoid)` keeps a sun or moon where the style put it unless the disc is mostly buried or sits on a jagged skyline, then searches along the skyline for a spot about 70% visible, away from the summit. Pass `label_spans(sc, peaks, pts)` as `avoid` if the style draws labels.
- `draw_labels()` lifts a label above the skyline (with a longer leader) when the skyline would cross its text, and skips peaks in the faded left zone.
- `parchment()`, `halftone()`, `draw_network()`, `draw_labels()`, `footnote()`.

Line widths passed to matplotlib are in points at 200 dpi, where 1 pt ≈ 2.8 px. `ridgelines()` already multiplies by `s`.

## 7. The web template

`assets/studio_template.html` is the same pipeline in JavaScript. `build_studio.py` fills its placeholders:
- `__CONFIG__`: grid radius `R`, normalisation `k` and `hs`, base and peak in km, viewer E/N offset, peaks with E/N and labels, fonts, name and slug. It also carries `v0` (the near-edge trim Python uses) and `auto` = {kind, relief, spacing, camh, topoCamh, snowline, fit, fadePow}, all computed in Python with the shape analysis.
- `__DEM__`: a base64 little-endian uint16 512×512 grid in metres, spanning ±R km east/north of the summit.
- Text and font placeholders.

Two differences from the Python version:
- Ridgelines are grouped into `Path2D` buckets by (alpha, width) so each render needs only a few dozen `stroke()` calls.
- Draft renders (60% size, fewer rows) run while a slider moves, and a full render follows after 260 ms idle.

JS `autoPanY()` and `sunSpot()`/`labelSpans()` mirror `fit_offset()` and `sun_spot()`/`label_spans()`. Keep the two sides in step when you change either.

To add a style to the web studio, add a preset to `PRESETS`, a render function to `RENDER`, and its name to `ORDER`.
