# aquarelle.py: watercolour

A second medium beside the oils of `atelier.py`. It is a simulation of water and pigment on paper, not a filter: you say where water and paint go and in what order, and the edges, blooms and granulation come from what the water does.

```python
from aquarelle import *
p = Paper(1200, 1600, seed=3)
x, y = p.x, p.y                     # picture heights: x 0..aspect, y 0 (top)..1
```

Two finished studies: `studies/estuary_in_watercolour.py` (wet-in-wet sky, graded hills, dry-brush water) and `studies/poppies_in_watercolour.py` (wet shapes charged and left to run, transparent second layer). Each runs as `python NAME.py --out out.png [--height 1600] [--film out.mp4]`. About a minute at 1600 px high on one core.

## How to think

- **White is the paper.** There is no white paint. Decide first what stays bare (clouds, sparkle, highlights) and keep every mask off it.
- **Light to dark, big to small.** Layers are transparent and only ever darken. Start with pale washes; put the darks in last.
- **Wet or dry decides the edge.** Paint on dry paper keeps the edge of its mask and dries darker at the rim. Paint on wet paper (`p.wet`, or a wash that has not dried) spreads and has no edge. Call `p.dry()` when you want the next layer to have hard edges and to leave what is underneath alone.
- **Timing makes blooms.** Paint or clean water touched into a wash that has started to dry (`p.settle(12)` first) pushes pigment outwards into a backrun. Touched in straight away it just blends.
- **Strength is dilution.** A tube colour at `strength=1` is close to its full depth, which is very dark. Skies are 0.05 to 0.3, mid-tones 0.3 to 0.5, darks 0.6 to 0.9. Too much strength is the usual mistake: the picture stops glowing.
- A soft mask on dry paper is cut at its half-way line. For a wash that fades out on dry paper, pass a hard mask and a `strength` field (`p.grade`, or any H×W array).

## Paper

`Paper(width, height, seed=1, grain=1.0, paper='#f8f4ea', tilt=0.15, sim=None, film=False, log=True)`

| Argument | Meaning |
|---|---|
| `grain` | depth of the tooth: 0 hot-press smooth, 1 cold-press, 1.6 rough |
| `tilt` | how far the board is tipped: washes drift down the sheet |
| `sim` | height of the grid the water moves on (default half the height, at most 900) |
| `film` | keep a frame after every step of the water, for `p.film(path)` |

The same seed gives the same picture at any size.

## Putting water and paint down

| Call | What it does |
|---|---|
| `p.wet(mask, amount=0.8, feather=0.012, hand=1.0)` | clean water: what follows here has soft edges |
| `p.wash(mask, color, strength=1.0, wetness=0.75, vary=0.2, granulate=None, staining=None, settle=4, hand=1.0)` | a wash. `strength` and `mask` may be fields. More `wetness` spreads further on wet paper, takes longer to dry and leaves a stronger rim. `hand` is how far the edge wanders off the mask, as a hand-held brush does: 0 keeps the mask exactly, 1 is a steady hand, 2 a loose one |
| `p.drop(points, color, size=0.03, strength=1.0, water=1.0, ragged=0.35, within=None, bloom=1.0)` | touch a loaded brush in at each point (charging in). `color=None` is clean water. In a wash that has started to dry (`p.settle(10)` to `(25)` first) the touch makes a backrun: a pale middle and a dark, lobed fringe where the water stopped. `bloom=0` turns that off; `within` keeps the touch inside a mask |
| `p.stroke(lines, color, size=0.008, strength=1.0, dry=0.0, wetness=0.3, taper=0.6)` | brush strokes along lists of points. `dry=0` is a loaded brush (a small wash). `dry` 0.3 to 0.9 is dry brush: it catches the crowns of the paper, skips the valleys and gives out along the stroke |
| `p.settle(steps=10)` | leave the water to move without drying the sheet |
| `p.dry()` | dry the sheet: every wash settles where it is |
| `p.lift(mask, amount=0.6)` | lift dried paint off (soft lights, clouds) |
| `p.grade(angle=90, start=1.0, end=0.2, mask=None)` | a strength that runs across the sheet or a mask (90 = downwards): a graded wash |
| `p.noise(scale=0.05, octaves=3)` | smooth noise over the sheet, for breaking up masks and strengths |
| `p.save(path)`, `p.render()`, `p.film(path, seconds=12)` | the result |

`color` is a tube name from atelier's `PIGMENTS` (`'ultramarine'`), a hex colour, `mixture({'ultramarine': 2, 'burnt sienna': 1})` (mixed in the palette, by parts), or an H×W×3 study. Shapes come from atelier: `sd_circle`, `sd_ellipse`, `sd_box`, `sd_segment`, `sd_poly`, `sd_curve`, `curve`, `smin`, `fill`, `smoothstep`, `blur`.

## The hand

Masks are drawn with exact shapes; a brush does not follow them exactly. Every wash and every wetting is moved off its mask by one slow drift and a row of small scallops (where the brush was set down again), and the edge is loaded a little unevenly. The same displacement is used for the whole sheet, so two washes that share an edge still meet along it. Thin shapes are moved less than broad ones; strokes and drops are not moved at all. `hand=0` on a call keeps the mask exactly (use it for anything that must register with a drawn line).

## A watercolour of an existing picture

```bash
python $S/aquarelle_repaint.py --in photo.png --out watercolour.png [--height 1200] [--seed 1] \
       [--aov work/NAME/bands_WxH/aov] [--planes 4] [--key 1.0] [--border 0.035] [--accents 1.0] [--film f.mp4]
```

It works in a watercolourist's order: the lights are reserved as bare paper (snow, cloud tops, sparkle), the sky goes on wet paper in one wash, the land is a few flat washes laid from each skyline down so that nearer planes are the sum of the layers behind them, then one or two smaller washes for shadow shapes and darks, and a few dry-brush accents. Each wash is given exactly the absorption the picture still lacks there, so the layers add up to the picture's colour with no opaque paint.

With `--aov` (the painter buffers of a `pbr_terrain.py --aov` render of the same view) the planes are the real ranges, split where the depth jumps, far ranges granulate, snow and cloud are reserved from the buffers and the accents run down the real fall lines. Without it the sky is the calm region hanging from the top edge and the planes are steps of tone. `--border 0` paints to the edge of the sheet; `--key 0` keeps the picture's own values. About 40 s at 2400×1000.

## Pigments

Granulating pigments settle into the paper's valleys and give a speckled wash: ultramarine, cerulean, cobalt blue, cobalt violet, the siennas and umbers, viridian, terre verte. Staining pigments barely lift or backrun once down: phthalo and prussian blue, alizarin, quinacridone rose, dioxazine violet. Override either with `granulate=` and `staining=` (0 to 1). A mixture keeps the weighted properties of its tubes; a hex colour or a study gets middling defaults.

## What is simulated

- **Layers.** Absorption (K/S in 12 bands, atelier's pigment tables) adds up over the paper's own, so a yellow wash under a blue one is green and two layers of one colour are deeper than one.
- **Water.** A surface film and dampness in the sheet. Water levels out and soaks only through damp paper; dry paper beside a wash is a wall. Rims and isolated touches dry first.
- **Pigment in the water.** It wanders through deep water (wet-in-wet), is carried from deeper to shallower water by a scheme that conserves it, so it collects where the flow stops (the dark rim, the edge of a backrun), and settles as the film thins, heavy pigments first. A drier brush leaves most of its pigment where it touches.
- **Rewetting.** Water landing on paint that is down but not dry lifts part of it again, less for staining pigments.
- **Paper.** A height field. Granulation and dry brush read it; a wash's edge on dry paper follows it slightly.

## Limits

- The water moves on a grid about half the picture's size, so a rim is two or three pixels wide and soft at 1:1.
- A backrun's outline is drawn from noise where the water would stop, not found by letting the water run; the fringe can pool into blobs in its hollows.
- `aquarelle_repaint.py` simplifies hard: a few flat washes. Fine texture (trees, rock, ripples) is gone, shadow shapes are soft at 1:1, and without `--aov` a plane is only a step of tone, so it can cut across forms.
- A brush stroke is one clean shape: no split hairs or side-loaded brush except in dry brush.
- No salt, wax resist, masking-fluid edges or spatter physics (spatter is small drops).
- Drawing is as hard as in `atelier`: the engine paints what the masks say.
