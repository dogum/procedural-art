"""Paint in sittings, looking at the canvas between them.

A painting lives in a folder:

    canvas.json        {"size": [1200, 800], "ground": "burnt sienna", "seed": 3, "palette": null}
    study.py           builds `study` (H x W x 3) and any masks; lays no paint
    10_lay_in.py       one sitting: a few p.strokes / p.mass / p.marks calls
    20_sky.py          the next one, written after looking at the canvas
    ...

    python sitting.py DIR                 run every sitting that has not been run yet; write DIR/look.jpg
    python sitting.py DIR --init 1200x800 [--ground C] [--seed N] [--palette "a, b, c"]
    python sitting.py DIR --final 1600 --out painting.png [--film process.mp4] [--tape t.bin]
    python sitting.py DIR --bundle painting.py        the whole painting as one script (it finds the engine
                                                      in ../scripts, in $ATELIER, or where it was when bundled)

Each file runs with the canvas `p`, the fields `x`, `y`, the sizes `W`, `H` and everything from
atelier already in scope; names defined in study.py and in earlier sittings stay available.
Sittings run in name order. The canvas is checkpointed after each one, so adding a sitting only
runs the new one, and editing an earlier file replays from that file on. look.jpg shows the
canvas beside the study, both again in grey, and a map of where they differ.
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atelier  # noqa: E402
from atelier import Painting  # noqa: E402


def sources(d):
    files = sorted(f for f in glob.glob(os.path.join(d, '*.py')) if os.path.basename(f) != 'study.py')
    return os.path.join(d, 'study.py'), files


def namespace(p):
    ns = {k: getattr(atelier, k) for k in dir(atelier) if not k.startswith('_')}
    ns.update(p=p, x=p.x, y=p.y, W=p.W, H=p.H)
    return ns


def run_file(path, ns):
    with open(path) as f:
        exec(compile(f.read(), path, 'exec'), ns)


def new_canvas(cfg, height=None, **kw):
    w, h = cfg['size']
    if height:
        w, h = int(round(w * height / h)), int(height)
    kw.setdefault('palette', cfg.get('palette'))
    kw.setdefault('knife', cfg.get('knife', False))
    return Painting(w, h, seed=cfg.get('seed', 1), ground=cfg.get('ground', '#7a5a40'), **kw)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dir')
    ap.add_argument('--init', metavar='WxH', help='create the folder with a canvas.json and an empty study.py')
    ap.add_argument('--ground', default='#7a5a40')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--palette', default=None, help='comma-separated tube colours for a limited palette '
                    '(with --init: stored; with --final: overrides the stored one, "full" = no limit)')
    ap.add_argument('--knife', action='store_true', help='with --init or --final: lay every pass with a palette knife')
    ap.add_argument('--final', type=int, metavar='HEIGHT', help='replay every sitting at this height')
    ap.add_argument('--out', default=None)
    ap.add_argument('--film', default=None)
    ap.add_argument('--tape', default=None)
    ap.add_argument('--bundle', metavar='FILE', help='write the whole painting as one standalone script')
    ap.add_argument('--look', default=None, help='where to write the look sheet (default DIR/look.jpg)')
    a = ap.parse_args()
    d = a.dir

    if a.init:
        os.makedirs(d, exist_ok=True)
        w, h = (int(v) for v in a.init.lower().split('x'))
        cfg = dict(size=[w, h], ground=a.ground, seed=a.seed,
                   palette=[t.strip() for t in a.palette.split(',')] if a.palette else None, knife=a.knife)
        with open(os.path.join(d, 'canvas.json'), 'w') as f:
            json.dump(cfg, f, indent=1)
        sp = os.path.join(d, 'study.py')
        if not os.path.exists(sp):
            with open(sp, 'w') as f:
                f.write('"""What the picture is, where the light comes from, what carries it."""\n'
                        "study = np.broadcast_to(rgb('#808080'), (H, W, 3)).copy()   # H x W x 3, sRGB 0..1\n")
        print(f"created {d}/canvas.json and study.py")
        return

    with open(os.path.join(d, 'canvas.json')) as f:
        cfg = json.load(f)
    study_py, files = sources(d)

    if a.bundle:
        w, h = cfg['size']
        parts = ['"""' + (os.path.basename(os.path.abspath(d)).replace('_', ' ')) + ': painted in sittings with sitting.py."""',
                 'import argparse', 'import os', 'import sys', '', 'import numpy as np', '',
                 '# the engine: beside the skill (this file in studies/), $ATELIER, or where it was when bundled',
                 "for _d in (os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'),",
                 "           os.environ.get('ATELIER', ''), " + repr(os.path.dirname(os.path.abspath(__file__))) + '):',
                 "    if _d and os.path.exists(os.path.join(_d, 'atelier.py')):",
                 '        sys.path.insert(0, _d)', '        break',
                 'from atelier import *  # noqa: E402,F401,F403', '',
                 'ap = argparse.ArgumentParser()', "ap.add_argument('--out', default='painting.png')",
                 f"ap.add_argument('--height', type=int, default={h})", "ap.add_argument('--film', default=None)",
                 "ap.add_argument('--study', default=None)", "ap.add_argument('--tape', default=None)",
                 f"ap.add_argument('--palette', default={','.join(cfg['palette']) if cfg.get('palette') else 'full'!r}, help='tube colours, comma-separated, or full')",
                 "ap.add_argument('--knife', action='store_true', help='lay every pass with a palette knife')", f"ap.add_argument('--seed', type=int, default={cfg.get('seed', 1)})",
                 'a = ap.parse_args()', '',
                 f"H = a.height\nW = int(round(H * {w} / {h}))",
                 "PALETTE = None if a.palette == 'full' else [t.strip() for t in a.palette.split(',')]",
                 f"p = Painting(W, H, seed=a.seed, ground={cfg.get('ground')!r}, palette=PALETTE, film=bool(a.film),\n             tape=bool(a.tape), knife=a.knife or {bool(cfg.get('knife', False))})",
                 'x, y = p.x, p.y', '']
        for fpath in [study_py] + files:
            parts += [f"# ---- {os.path.basename(fpath)[:-3]} " + '-' * 60, open(fpath).read().rstrip(), '']
            if fpath == study_py:
                parts += ['if a.study:', '    from PIL import Image',
                          "    Image.fromarray((np.clip(study, 0, 1) * 255).astype(np.uint8)).save(a.study)", '']
        parts += ['p.save(a.out)', 'if a.film:', '    p.film(a.film)', 'if a.tape:', '    import tape', '    tape.save_tape(p, a.tape)', '']
        with open(a.bundle, 'w') as f:
            f.write('\n'.join(parts))
        print(f"wrote {a.bundle}  ({len(files)} sittings)")
        return

    if a.final:
        over = {}
        if a.palette:
            over['palette'] = None if a.palette == 'full' else [t.strip() for t in a.palette.split(',')]
        if a.knife:
            over['knife'] = True
        p = new_canvas(cfg, a.final, film=bool(a.film), tape=bool(a.tape), **over)
        ns = namespace(p)
        run_file(study_py, ns)
        for fpath in files:
            print(f"-- {os.path.basename(fpath)}", flush=True)
            run_file(fpath, ns)
        out = a.out or os.path.join(d, 'final.png')
        p.save(out)
        if a.film:
            p.film(a.film)
        if a.tape:
            import tape
            tape.save_tape(p, a.tape)
        return

    # incremental: find the last checkpoint whose history is unchanged, carry on from there
    state = os.path.join(d, '.state')
    os.makedirs(state, exist_ok=True)
    hsh = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode() + open(study_py, 'rb').read())
    keys = []
    for fpath in files:
        hsh.update(os.path.basename(fpath).encode() + open(fpath, 'rb').read())
        keys.append(hsh.hexdigest()[:16])
    start, p = 0, None
    for i in range(len(files) - 1, -1, -1):
        ck = os.path.join(state, keys[i] + '.npz')
        if os.path.exists(ck):
            p, start = Painting.resume(ck), i + 1
            break
    fresh = p is None
    if fresh:
        p = new_canvas(cfg)
    rng_state = p.rng.bit_generator.state
    ns = namespace(p)
    t0 = time.time()
    run_file(study_py, ns)
    if not fresh:
        p.rng.bit_generator.state = rng_state                  # the study must not disturb a resumed canvas
    if start:
        print(f"resuming after {os.path.basename(files[start - 1])}")
    # names defined by earlier sittings: re-run them on a scratch canvas only if a later one needs them
    for i in range(start, len(files)):
        print(f"-- {os.path.basename(files[i])}", flush=True)
        try:
            run_file(files[i], ns)
        except NameError as e:
            raise SystemExit(f"{os.path.basename(files[i])}: {e}. Names shared between sittings belong in study.py "
                             "(it runs before every sitting).")
        p.checkpoint(os.path.join(state, keys[i] + '.npz'))
        os.makedirs(os.path.join(d, 'looks'), exist_ok=True)           # the canvas after every sitting
        p.look(os.path.join(d, 'looks', os.path.basename(files[i])[:-3] + '.jpg'), None, width=1200)
    keep = set(keys[-2:])
    for f in glob.glob(os.path.join(state, '*.npz')):
        if os.path.basename(f)[:-4] not in keep:
            os.remove(f)
    look = a.look or os.path.join(d, 'look.jpg')
    p.look(look, ns.get('study'))
    print(f"{len(files) - start} sitting(s) run in {time.time() - t0:.0f} s, {p.n_strokes} strokes on the canvas; look at {look}")


if __name__ == '__main__':
    main()
