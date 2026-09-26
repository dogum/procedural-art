# Lessons from building this, and the limits that remain

## What each technique buys

| Technique | Visual payoff | Cost |
|---|---|---|
| Real elevation data | Silhouettes, gorges and proportions read as real, because they are | One download, cached |
| Earth curvature + refraction (R × 7/6) | Distant bases sink realistically, about 200 m hidden at 56 km | Free |
| Single-scattering Rayleigh + Mie | Blue sky, warm low sun, blue distant haze; the biggest single realism jump | 28 samples per ray |
| Soft terrain shadows (min-ratio march toward the sun) | Gorges and ridges gain depth | 70 steps per hit pixel |
| Sub-DEM erosion noise in normals | Surface texture finer than the data | Cheap |
| Patchy exponential mist | Atmosphere in the valley at density about 3; a blank sheet above about 6 | Noise per atmosphere sample |
| Volumetric lenticular cloud | Stacked-plate cap; convincing at a distance | Bounded to its ellipsoid |
| Water plane + traced reflection + Fresnel | The mountain doubled in the lake; the strongest single "photo" cue after the atmosphere | A second terrain march per water pixel (×3 samples) |
| Footprint-filtered wave normals + vertical smear in post | Crisp near ripples, streaky far reflections, no sparkle noise | 36 cosines per sample; one blur pass |
| Screen-box instancing (trees, houses) | Believable forest edges, villages, orchards and poplar rows | Only covered pixels cast rays; 2×2 AA |
| Mirrored instances in water | The tree line reflected under the real tree line | Second pass on water pixels |
| Instance shadows by grid walk | Tree and house shadows on the ground and each other | ~60 grid steps per shaded pixel near the camera |
| Crowns carved into clumps, sprays and leaves, by level of detail | Close trees with gaps, limbs and single leaves; far trees stay solid and cheap | One baked 3-octave noise lookup per march step, more only for trees that span many pixels |
| Light through the crown by chord length | Self-shadowed crowns, dappled shade under trees, backlit edges that glow | A closed-form ellipsoid chord per shaded point |
| Close-up house and wall materials | Tuff, plaster and boards, tiles and tin, windows with frames and sills, doors, dry-stone walls, fences | Shading only; patterns fade with pixel size |
| Ground texture filtered by the pixel footprint | Crop rows, stubble, grass and tracks near the lens without aliasing | A few 2-D noise lookups per ground pixel within 1 km |
| Physical sky (sun LUT, Earth shadow, ozone) | Belt of Venus, alpenglow, blue hour | One 61×581 LUT per wavelength band, plus a 31×121 multiple-scattering LUT (about 3 s) |
| Octave multiple scattering for clouds | Brighter interiors, silver linings when backlit | 4 exponentials per sample |
| City lights as splatted points with CoC | Thousands of lights, bokeh and glow for free in post | Negligible |
| Path tracing + direct light sampling | Glass, copper, juice and caustics are physically right from the first draft | Noise; needs samples |
| MIS + specular light connections | Smooth window reflections in glass, clean rough metal | Extra shadow rays at glass hits |
| Adaptive sampling (noise weighted by the display curve) | Samples go to glass, metal and caustics; with MIS, 2.4–12× less time than the basic sampler (`--basic`) for the same noise | Per-pixel second moment |
| Caustic photon map | Continuous caustics from small lights, where path tracing leaves dots | 2–5 s per pass |
| Spectral dispersion (Cauchy index, per-path wavelength) | Rainbow caustics from prisms and cut crystal | Colour noise; needs samples |
| Clear-coat material | Fruit skin gets a sheen instead of plastic or chalk | Cheap |
| Variance-guided à-trous denoise | Diffuse areas clean at low spp, while shadows and caustics keep their edges | Variance filtered with the image |
| Coarse-to-fine strokes seeded on colour error (Hertzmann) | Big brushes block in, small brushes only where detail is missing, so flat sky stays calm | One pass per brush size |
| Structure-tensor stroke direction | Strokes follow ridges, contours and horizons | Four blurs per layer |
| Z-buffer stroke rasteriser (one packed int64 per pixel, `np.maximum.at`) | Tens of thousands of curved strokes without a Python loop per stroke | ~1–3 s per layer at 3000×1000 |
| Nested optical-density glazes with gradient-scaled edges | Watercolour: crisp wet edges on real edges, soft wet-in-wet on gradients | ~6 blur passes |

## Mistakes worth not repeating

1. **Afternoon sun behind the mountain.** Every visible face was in shadow and the result looked flat and grey. Always put light on the faces the camera sees.
2. **Very thick fog.** It saturated into a featureless band. Believable mist needs multiple scattering or self-shadowing; keep it thin.
3. **Ray-march iteration cap.** Grazing rays over flat plains need about 900 steps with distance-growing minimum steps. With too few, "holes" of sky appear below the horizon.
4. **Juice as a solid cylinder.** It read as a red block. Use a `tube` or `bowl` with a `fill`, so the glass/liquid interface refracts correctly and the liquid absorbs along its own path.
5. **Denoising glass.** A filter guided by the first surface's colour and shape turned the crystal ball to frosted mush. Mask specular first hits out of the filter.
6. **Two million rays at once.** Out of memory. Process in chunks.
7. **Close viewpoints.** The camera sat too close to Fuji and the peak overflowed the frame. Auto-back along the line of sight to a fixed angular size (about 5.4°).
8. **Per-pixel value noise in the tree march.** Hash-based 3-D noise made tree intersection the bottleneck. A precomputed tileable 48³ noise texture sampled with `map_coordinates(mode="grid-wrap")` plus distance-guided steps made it 3.4× faster.
9. **Stochastic reflection noise.** Three random reflection samples per pixel sparkle badly on far water. Blurring water pixels vertically by the per-pixel reflection spread fixes the noise and looks like the real thing.
10. **Multiple-scattering hack that lets light in below the horizon.** A +3° fudge in the twilight scattering source lit the whole horizon yellow at blue hour. Keep the twilight term separate, bluish, and fading by sun angle.
11. **Pair-array memory.** Mirror-pass pairs at 3000 px took 3 GB until they were chunked like the primary pass.
12. **A denoiser that cannot see lighting edges.** With a fixed luminance threshold, the à-trous filter blurred shadows and caustics, which the normal/depth/albedo guides cannot see. Scaling the edge-stop by each pixel's measured noise keeps them.
13. **Glass exactly on the table.** A glass base at y = 0 on a table at y = 0 loses the light under it. Lift glass 0.02 cm.
14. **Painting strokes one at a time in Python.** The first stroke renderer took two minutes at 900×300. Stamping all strokes of a layer at once in a z-buffer took it to 10 s, and 3000×1000 now takes under a minute.
15. **Stroke outlines.** Every stroke looked like a separate pebble, with a light rim and a dark gap. Three causes: a height profile with an infinite slope at the stroke edge (use one with zero slope there), a dark toned ground showing between strokes, and strokes whose colours all differ slightly. Blending each stroke's edge into the average colour of the strokes around it removed the cut-out look.
16. **Measuring error on the textured canvas.** If bristle texture counts as error, every layer re-paints everything. Keep a flat-colour copy of the canvas for seeding.
17. **Watercolour on a dark image.** Transparent glazes over a dark render give mud. Lift the image toward a mid-light key first. Granulation must also weaken in heavy darks, or they turn gritty.
18. **Ink on a dark, colourful scene.** Luminance alone turned red fruit into black blobs. Read saturated colour as light, and leave featureless light areas as bare paper.
19. **Terrain march for a camera a few metres up.** The march started 50 m out and flagged a hit up to 0.4 m above the ground, so near ground came out smeared and quantised into blocks. It now starts at twice the camera height and walks near hits below the surface before bisecting.
20. **Foliage as a noise isosurface.** Carving a crown with 3-D noise gives crumpled green foil up close, and a carve deeper than its features leaves thin films and clumps floating in the sky. Build a close crown from clumps that each hang on a branch, displace surfaces only by noise sampled by direction (so the shape stays star-shaped), keep every displacement shallower than it is wide, and leave single leaves to shading.

## Measured

The path tracer with MIS and adaptive sampling reaches the same glass noise 6.7× sooner than the basic sampler (`pathtracer.py --basic`) on the pomegranate scene (copper 12×, table caustic 2.4×). A 3000×1000 terrain frame takes 5–6 min (Yerevan blue hour, Ararat orchards) to 10 min (Ararat village close up, camera 3 m up) and 13 min (Fuji lake with 52k trees and houses in view). Paintings take 10–60 s at 3000×1000.

## Limits that remain

These are the honest weak points after all of the above. Tell the user when a request runs into one.

- **Close foregrounds.** Standing in a field 3 m up, houses, orchards, yard walls and poplars from about 40 m out hold up at 3000 px. Crowns are clumps on branches with a dense core rather than real branching, so close crowns look sculpted rather than leafy; houses are boxes with a chimney, gutter and material detail but no balconies, porches or roof relief that reads beyond 50 m; dry-stone walls are courses on a flat box, so their silhouette is straight; the grass and flowers are texture, not geometry, so nothing breaks the ground's silhouette.
- **Cities.** Lamps, traffic and building frontage follow real OpenStreetMap streets, but the buildings are boxes with window grids and made-up heights (OSM footprints and levels are not used). Fine at blue hour, synthetic in daylight. The first run needs the public Overpass mirrors, which can be slow or refuse (11 minutes for Yerevan's 16 tiles here); without them the city falls back to a synthetic grid.
- **Water.** The bed is synthetic (no bathymetry in the DEM), reflections of trees and houses ignore ripples, and waves are static, so water doesn't move in video. `--water-level auto` only finds lakes that are level in the DEM and don't merge with a flat valley floor; others need an explicit level and seed.
- **Clouds.** Hand-shaped volumes. The cumulus layer has speckled edges and casts no shadows on the terrain.
- **Depth of field on terrain** is a layered post blur, so it bleeds a little at depth edges. True thin-lens ray sampling would fix it at a large cost.
- **Timelapse cost.** Every frame is a full render so that shadows slide, which makes a 20 s clip at 1200×400 about 6 hours and a 1080×1350 vertical about 15 hours on one core. `--blend 2` halves it and is hard to tell apart; larger blends cross-fade shadows again.
- **Twilight colour.** Twilight single scattering matches a 71-wavelength reference within about 5° of hue, and multiple scattering is Hillaire's isotropic approximation, not a path-traced reference. The atmosphere ends at 60 km, so direct sunlight leaves the zenith below −8° and the sky there is multiple scattering only. Daylight (sun above 10°) keeps the older three-wavelength model and the by-eye multiple-scattering fit; the two blend between 6° and 10°.
- **Night.** The moon is a directional light 400× brighter than a real one relative to the sun (tuned so moonlit terrain beats the airglow floor), trees and houses get only its sky light, and the lenticular cloud is not moonlit, so it reads as a dark lens. A city at full night overwhelms the frame; blue hour suits it.
- **Path tracer.** Nested media one level deep (a diffuse object in liquid doesn't get the liquid's colour), no torus or rounded box primitives, adaptive sampling that judges noise by luminance only (dispersion colour noise lingers), photon-map caustics slightly soft at their edges, and about an hour for a clean 2400×800 final.
- **Paintings.** They follow the image, not the objects. A detail pass brings back small shapes and hard highlights, but on noisy sources (a low-sample render) fine detail still softens, and stroke direction on large flat areas can look monotonous. Ink in wash mode is poor on busy, dark scenes; lines mode (auto-picked for objects) handles still lifes, though low-contrast silhouettes are only partly drawn.
