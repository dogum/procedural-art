# stage.py: set the subject up in 3D, then paint it

A flat study has to be drawn by hand: perspective, how a form turns into shadow, where a cast shadow falls. On the stage the subject is a small scene of solids and posable figures with a camera and a light, and a sphere tracer turns it into what a painter needs. Use it for anything that depends on drawing: figures, interiors, streets, furniture, a table top seen in perspective.

```python
from atelier import *
from stage import *

sc = Scene(Camera((0, 1.5, -4), (0, 1.0, 2), fov=38), key=(-0.6, 0.6, -0.3), shadow=0.4, soft=45)
sc.add('floor', plane((0, 0, 0), (0, 1, 0)), 'raw sienna', grain=(1, 0, 0), pattern=planks())
sc.add('wall', box((0, 1.5, 4), (4, 1.5, 0.1)), '#979a8e', cut=box((-1.5, 1.4, 4), (0.5, 0.9, 0.3)))   # a window hole
sc.add('jug', lathe((0.4, 0.8, 2), [(0, 0.06), (0.10, 0.10), (0.24, 0.05), (0.30, 0.06)]), 'cobalt blue', gloss=0.6)
j = figure(sc, 'reader', at=(0, 0, 2), facing=-90, pose='read', seat=0.45, skirt=True)

p = Painting(1200, 800, seed=3)
B = sc.render(p)                    # 10 to 16 s
study = B.study                     # edit it like any study, or leave it
paint(p, B, focus='reader', study=study)
p.save('out.png')
```

Units are yours (metres for a room, centimetres for a table top). y is up; with the camera looking along +z, +x is to the right.

## Scene

`Scene(camera, key=(-0.5, 0.7, -0.4), key_color, sky, bounce, shadow=0.36, soft=10, backdrop, fills=(), air=None)`

| Argument | Meaning |
|---|---|
| `camera` | `Camera(position, look_at, fov=38)` |
| `key` | Direction to the light (a sun or a window) |
| `key_color`, `sky`, `bounce` | Colour of the light; colours that fill the shadows from above and from below |
| `shadow` | How dark the shadow family is: 0.2 dramatic, 0.5 airy |
| `soft` | Crispness of cast shadows: small is soft, 30 a crisp sun. For sun through a window use 40 or more, or the light passing the opening is darkened |
| `backdrop` | Colour where a ray leaves the scene (through a window, past a table edge) |
| `fills` | `[(position, colour, strength, radius), ...]`: unshadowed local lights (a lamp, a window's glow on the wall beside it) |
| `air` | `(colour, density per unit distance)`: haze |

`sc.add(name, shapes, color, **options)` adds an object: one shape or a list joined by smooth union. Options: `k` (blend radius of that union), `cut` (a shape removed from it: a window, a doorway), `kind` (`'matte'`, `'metal'` with a one-bounce mirror, `'glass'` with refraction, `'emit'` for a lamp or a sky seen through a window), `gloss` (0..1, strength of the highlight), `grain` (a direction in the scene the brush follows on this object, e.g. along floorboards), `flow='around'` (brush circles the object's vertical axis), `pattern` (`planks(axis, width)` or `mottle(scale)`), `importance` (above 1 gets smaller brushes and a tighter finish).

Shapes: `sphere(c, r)`, `ellipsoid(c, radii, rot)`, `box(c, half, rot, r)`, `capsule(a, b, r0, r1)`, `cylinder(base, height, r, rot)`, `cone(base, height, r0, r1, rot)`, `torus(c, R, r, rot)`, `plane(point, normal)`, `lathe(base, profile, rot)` with `profile = [(height, radius), ...]` from the foot up. `rot` is `(rx, ry, rz)` in degrees.

## Figures

`figure(scene, name, at=(x, 0, z), facing=0, height=1.7, pose='stand', skin, hair, top, bottom, shoes, skirt=False, coat=False, hat=None, build=1.0, seat=None, **joints)`

`at` is the point between the feet (for a sitting pose, the floor under the pelvis). `facing` is degrees about the vertical: 0 looks along +z (away from a camera looking along +z), 180 toward it, 90 toward +x. Poses: `stand`, `contrapposto`, `walk`, `walk2` (the other leg forward), `sit`, `read`, `reach`, `carry`; or a dict of joint angles. Keyword joints override the pose: `hip_l`, `hip_r`, `knee_l`, `knee_r`, `shoulder_l`, `shoulder_r`, `elbow_l`, `elbow_r`, `abduct_l`, `abduct_r`, `lean`, `twist`, `head_yaw`, `head_pitch` (degrees; hips and shoulders swing forward with +, knees and elbows bend with +). `skirt` and `coat` are clothing volumes, `hat` a colour, `seat` the seat height for sitting poses.

It adds the objects `name.skin`, `name.hair`, `name.top`, `name.bottom`, `name.shoes` and returns joint positions (`j['hand_l']`, `j['head']`, `j['neck']`, `j['lap']`, ...) for putting a book, a cup or an umbrella where it belongs.

A figure reads as a person at mid distance from any camera: proportion, pose and cast shadow are right. It is a mannequin up close: no face beyond a nose, mitten hands, no cloth folds. Give the focal figure a light garment against a dark one so the pose reads.

## What render() returns

`B = sc.render(p)` (or `sc.render((W, H))`). Buffers are traced at 560 px high at most and enlarged.

| Field | Is |
|---|---|
| `B.study` | H×W×3 colour study in light, halftone, core-shadow and reflected-light families, with cast shadows, contact shadow and bounce |
| `B.masks[name]`, `B.group('reader')` | A mask per object (and `'backdrop'`); one mask for every object named `reader` or `reader.*` |
| `B.depth`, `B.normal`, `B.light`, `B.spec` | Distance from the camera; normals (3, H, W); how lit each pixel is, 0..1; highlight strength |
| `B.flow` | Brush direction following each form: down limbs, along `grain`, upright on walls, level on floors |
| `B.project(point)` | A point in the scene as `(x, y)` on the picture, for `p.marks` and `p.dabs` |
| `B.far_to_near()` | Object names, back to front |

## Painting it

`paint(p, B, focus=None, study=None, plan=None, loose=1.0, lay_in=True, cut_in=True, highlights=True, touches=True)` lays a thin lay-in, then every object back to front with its own flow and brush sizes scaled by distance and `importance`, cuts the background in around the objects, and puts the highlights on last. `focus` names the object (or figure) that gets the tightest finish. Pass your own `study` if you changed colours, added things the stage cannot make (a pattern on a wall, a face), or keyed the picture warmer.

`plan = masses(p, B, focus=...)` returns the list `paint` would use (per object: mask, sizes, tolerance, flow, stroke parameters); edit it and pass it back as `plan=`. For full control call `p.mass` yourself with `B.masks` and `B.flow`. The stage composes with `sitting.py`: build the scene and `B` in `study.py`, then paint in sittings.

## The path tracer's scenes

`sc = load_pathtracer_scene('pomegranate.json', aspect=2.0)` turns a `procedural-realism` scene file into a stage scene, so one file gives a photograph (`pathtracer.py`) and a painting. Camera, the first light, planes, spheres, ellipsoids, cylinders, boxes, disks, tubes and bowls (as thin glass, with their liquid) and scatters (as spheres, in the stage's own random places) map. A prism becomes a box and a gem an ellipsoid (`sc.unmapped` lists the stand-ins). Textures, depth of field, caustics, dispersion and extra lights are not carried over.

## Limits

- Glass casts no shadow and there are no caustics; a glass of liquid reads as a block of colour.
- Round forms get a flat light plateau with a fairly hard edge into shadow; soften it in the study if it reads like a billiard ball.
- Not yet used in a finished painting: `cone`, `flow='around'`, `air`, `hat`, `coat`.
