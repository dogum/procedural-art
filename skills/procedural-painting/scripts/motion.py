"""A painting that moves: one canvas, one set of brush strokes, the light changing across them.

    python motion.py --frames FOLDER --out film.mp4 [--aov BANDS/aov] [--keys 7] [--fps 24]
                     [--jobs 4] [--looseness 1.0] [--push 1.0] [--ground C] [--seed 1] [--work DIR]
                     [--gif film.gif] [--stills]

FOLDER holds the frames of a film with a fixed camera (a procedural-realism timelapse: its
<work>/frames_<hash> folder; or any numbered image sequence). Painting every frame separately
would make the brushwork boil. Instead:

  1. The first frame is painted in full (as repaint.py would, with --aov by the mountain's
     ground). Every stroke is remembered: its path, width, order, and how its own colour differs
     from the picture's colour at the spot where it started.
  2. At a few key frames (--keys, spread over the film) the same strokes are laid again with
     that frame's colours, and the painter adds only what the fixed strokes cannot show there:
     a shadow edge that has moved, street lamps that have come on, stars. Those strokes join the
     set. Lights are remembered as lights: they are laid only while their spot is lit.
  3. Every frame of the film is then that whole set of strokes laid again, in the same order
     on the same canvas weave, with the colours of that frame. Strokes hold still; colour moves.

The result is written as an mp4 (and a GIF with --gif). --stills also keeps every painted frame.
Without ffmpeg on the machine no film is written and the painted frames are kept instead.
"""
import argparse
import glob
import hashlib
import os
import shutil
import subprocess
import time
from multiprocessing import Pool

import numpy as np
from PIL import Image

from atelier import Painting, blur, image_flow, load_image, smoothstep, F32
from repaint import paint_picture, painters_eye

G = {}                                                    # the film's settings; handed to each worker by _init


def study_at(i):
    return painters_eye(load_image(G['files'][i], G['W'], G['H']), G['push'])


def additions(p, src, k):
    """What the strokes on the canvas cannot say about this frame: drawn smaller and smaller where
    the canvas is still wrong, then the lights, thick and clean."""
    H = p.H
    L = src @ np.array([0.299, 0.587, 0.114], F32)
    ef = image_flow(src, 0.012 * H)
    flow = np.where(np.cos(ef) < 0, ef + np.pi, ef)
    one = np.ones((p.H, p.W), F32)
    p.strokes(one, src, flow=flow, size=0.016 * k, length=6, pickup=0.4, refine=0.075 * k, ctol=0.1, jitter=0.03,
              name='moved masses')
    p.strokes(one, src, flow=flow, size=0.007 * k, length=5, pickup=0.3, refine=0.07 * k, ctol=0.08, jitter=0.025,
              name='moved edges')
    p.dry(0.6)
    lum = blur(L, 0.0015 * H)
    hi = smoothstep(0.05, 0.14, lum - blur(lum, 0.02 * H)) * smoothstep(0.3, 0.6, lum)
    p.strokes(hi, src, flow=flow, size=0.005 * k, length=2.2, pickup=0.05, load=1.3, thickness=2.4, jitter=0.02,
              ctol=0.12, overshoot=0.0, refine=0.05 * k, fade=True, name='lights')


def _init(state):
    G.update(state)


def render_frame(i):
    out = os.path.join(G['work'], f'{i:05d}.png')
    if not (G['reuse'] and os.path.exists(out)):
        p = Painting(G['W'], G['H'], seed=G['seed'], ground=G['ground'], log=False)
        p.replay(G['tape'], study_at(i))
        p.save(out)
    return i


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--frames', required=True, help='folder (or glob) of the frames of a fixed-camera film')
    ap.add_argument('--out', required=True)
    ap.add_argument('--aov', default=None, help='painter buffers of the same view (pbr_terrain.py --aov)')
    ap.add_argument('--height', type=int, default=None)
    ap.add_argument('--keys', type=int, default=7, help='frames at which the painter may add strokes')
    ap.add_argument('--every', type=int, default=1, help='use every N-th frame of the folder')
    ap.add_argument('--fps', type=int, default=24)
    ap.add_argument('--jobs', type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument('--looseness', type=float, default=1.0)
    ap.add_argument('--push', type=float, default=None, help="painter's-eye colour (default 1 with --aov, else 0)")
    ap.add_argument('--ground', default='#7a5a40')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--work', default=None, help='folder for the painted frames (default: next to --out)')
    ap.add_argument('--gif', default=None)
    ap.add_argument('--stills', action='store_true', help='keep the painted frames')
    a = ap.parse_args()

    pat = os.path.join(a.frames, '*') if os.path.isdir(a.frames) else a.frames
    files = sorted(f for f in glob.glob(pat) if f.lower().endswith(('.png', '.jpg', '.jpeg')))[::a.every]
    if len(files) < 2:
        raise SystemExit(f"need at least two frames in {a.frames}")
    im = Image.open(files[0])
    H = a.height or im.height
    H -= H % 2
    W = int(round(H * im.width / im.height))
    W -= W % 2
    work = a.work or (os.path.splitext(a.out)[0] + '_frames')
    made_dir = not os.path.isdir(work)
    os.makedirs(work, exist_ok=True)
    # painted frames left by an earlier run are reused only if that run had the same frames and settings
    sig = [(os.path.abspath(f), os.path.getsize(f), os.path.getmtime(f)) for f in files]
    if a.aov:
        sig += [(os.path.abspath(f), os.path.getsize(f), os.path.getmtime(f)) for f in sorted(glob.glob(os.path.join(a.aov, '*')))]
    key = hashlib.sha1(repr((sig, W, H, a.keys, a.looseness, a.push, a.ground, a.seed, a.aov)).encode()).hexdigest()
    stamp = os.path.join(work, 'motion.key')
    reuse = os.path.exists(stamp) and open(stamp).read().strip() == key
    with open(stamp, 'w') as f:
        f.write(key)
    G.update(files=files, W=W, H=H, push=a.push if a.push is not None else (1.0 if a.aov else 0.0),
             seed=a.seed, ground=a.ground, work=work, reuse=reuse)
    k = a.looseness
    t0 = time.time()

    # 1. the first frame, painted in full, every stroke remembered
    p = Painting(W, H, seed=a.seed, ground=a.ground, tape='full')
    paint_picture(p, study_at(0), k, a.aov, None, 0.0)                 # (the study is already keyed)
    tape = list(p.tape)
    print(f"first frame: {p.n_strokes} strokes, {time.time() - t0:.0f} s", flush=True)

    # 2. key frames: the same strokes under that frame's light, plus what they cannot show
    keys = sorted(set(int(round(v)) for v in np.linspace(0, len(files) - 1, max(a.keys, 2))))[1:]
    for i in keys:
        src = study_at(i)
        p = Painting(W, H, seed=a.seed, ground=a.ground, tape='full', log=False)
        p.replay(tape, src)
        n0 = p.n_strokes
        additions(p, src, k)
        tape += p.tape
        print(f"key frame {i}: +{p.n_strokes - n0} strokes ({sum(len(r['width']) for r in tape if r['op'] == 'strokes')} in all), "
              f"{time.time() - t0:.0f} s", flush=True)
    G['tape'] = tape

    # 3. every frame: all the strokes again with that frame's colours
    todo = list(range(len(files)))
    with Pool(max(a.jobs, 1), initializer=_init, initargs=(G,)) as pool:
        for n, i in enumerate(pool.imap_unordered(render_frame, todo), 1):
            if n % 20 == 0 or n == len(todo):
                print(f"painted {n}/{len(todo)} frames, {time.time() - t0:.0f} s", flush=True)

    if not shutil.which('ffmpeg'):                           # no encoder here: the painted frames are the result
        print(f"ffmpeg not found: no film written; the {len(files)} painted frames ({W}x{H}) are in {work}")
        return
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(a.fps), '-i', os.path.join(work, '%05d.png'),
           '-pix_fmt', 'yuv420p', '-crf', '17', '-movflags', '+faststart', a.out]
    subprocess.run(cmd, check=True)
    print(f"wrote {a.out}  {W}x{H}  {len(files)} frames")
    if a.gif:
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(a.fps), '-i', os.path.join(work, '%05d.png'),
                        '-vf', f'fps={min(a.fps, 15)},scale={min(W, 720)}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=160[p];'
                               '[b][p]paletteuse=dither=bayer:bayer_scale=4', a.gif], check=True)
        print(f"wrote {a.gif}")
    if not a.stills:
        for f in [os.path.join(work, f'{i:05d}.png') for i in todo] + [stamp]:      # only what this film wrote
            os.remove(f)
        if made_dir and not os.listdir(work):
            os.rmdir(work)


if __name__ == '__main__':
    main()
