#!/usr/bin/env python3
"""
animate.py — looping orbit video: the camera swings ±amplitude degrees around the summit.

  python animate.py --peak 39.7019,44.2986 --from 40.1792,44.4991 --style nocturne --out orbit.mp4
  # long renders: run in chunks (each call skips frames that already exist), encode on the last one
  python animate.py ... --start 0 --end 50 ; python animate.py ... --start 50 --end 144

Frames go to <out>_frames/ with a params.json; a rerun with different settings clears the old frames first.
Needs ffmpeg for encoding. ~1-4 s per frame at 1500x500 depending on style.
"""
import argparse, glob, hashlib, json, os, subprocess, sys
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from terrain_art import ensure_dem, load_dem
import render as R

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--peak", required=True); ap.add_argument("--from", dest="view_from"); ap.add_argument("--facing", type=float)
ap.add_argument("--style", default="nocturne", choices=R.STYLES); ap.add_argument("--out", required=True)
ap.add_argument("--dem"); ap.add_argument("--size", default="1500x500"); ap.add_argument("--frames", type=int, default=144)
ap.add_argument("--fps", type=int, default=24); ap.add_argument("--amplitude", type=float, default=28)
ap.add_argument("--start", type=int, default=0); ap.add_argument("--end", type=int, default=None)
ap.add_argument("--label", action="append", default=[]); ap.add_argument("--title", default="")
ap.add_argument("--quality", type=float, default=0.6); ap.add_argument("--extent", type=float); ap.add_argument("--base", type=float)
a = ap.parse_args()

summit = R.parse_ll(a.peak); W, H = map(int, a.size.lower().split("x"))
fdir = os.path.splitext(a.out)[0] + "_frames"; os.makedirs(fdir, exist_ok=True)
# frames belong to one set of settings: a rerun with other settings starts the frame folder afresh
_pk = {k: v for k, v in vars(a).items() if k not in ("out", "dem", "start", "end", "fps")}
_ph = hashlib.md5(json.dumps(_pk, sort_keys=True).encode()).hexdigest()[:10]
_pf = os.path.join(fdir, "params.json")
_old = json.load(open(_pf)).get("hash") if os.path.exists(_pf) else None
_stale = glob.glob(os.path.join(fdir, "*.png"))
if _stale and _old != _ph:
    print(f"{fdir} holds frames made with other settings; removing {len(_stale)} old frames", flush=True)
    for x in _stale: os.remove(x)
json.dump({"hash": _ph, "args": _pk}, open(_pf, "w"), indent=1)
dem_path = a.dem or os.path.join(os.path.dirname(os.path.abspath(a.out)), f"dem_{summit[0]:.3f}_{summit[1]:.3f}.npz")
ensure_dem(*summit, (a.extent or 45) * 1.45, dem_path, explicit=bool(a.dem), log=lambda m: print(m, flush=True))
dem = load_dem(dem_path)
peaks = []
for L in a.label:
    name, _, rest = L.partition("|"); sub, _, ll = rest.partition("@")
    peaks.append(dict(name=name, sub=sub, latlon=R.parse_ll(ll) if ll else summit))
end = min(a.end or a.frames, a.frames)
# shape-aware vertical fit (horns, canyons): measure it once at yaw 0 so the frame doesn't bob during the orbit
from terrain_art import terrain as _terrain
_t = _terrain(dem, summit, view_from=R.parse_ll(a.view_from) if a.view_from else None, facing=a.facing, extent_km=a.extent,
              base_m=a.base, NX=int(1400 * a.quality), NZ=int(520 * a.quality))
_cfg = dict(W=W, H=H, align="right", labels=bool(peaks), relief=_t.shape.get("relief", 1.0), zoom=1.0)
pan_y = R.fit_offset(R.scene_for(a.style, _t, dict(_cfg, pan_y=0.0)), _t, _cfg)
for f in range(a.start, end):
    path = os.path.join(fdir, f"{f:04d}.png")
    if os.path.exists(path): continue
    yaw = a.amplitude * np.sin(2 * np.pi * f / a.frames)
    cfg = dict(summit=summit, view_from=R.parse_ll(a.view_from) if a.view_from else None, facing=a.facing, W=W, H=H,
               align="right", peaks=[dict(p) for p in peaks], title=a.title, footnote="", extent=a.extent, base=a.base,
               yaw=yaw, sun=True, labels=bool(peaks), network=False, seed=7, quality=a.quality, pan_y=pan_y)
    R.render(a.style, dem, cfg, path); print(f"frame {f}", flush=True)
missing = [f for f in range(a.frames) if not os.path.exists(os.path.join(fdir, f"{f:04d}.png"))]
if not missing:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", os.path.join(fdir, "%04d.png"),
                    "-frames:v", str(a.frames), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "18", a.out], check=True)
    print("video ->", a.out)
else:
    print(f"{a.frames - len(missing)}/{a.frames} frames rendered (first missing: {missing[0]}); "
          "run again with --start 0 (finished frames are skipped) to continue")
