#!/usr/bin/env python3
"""
timelapse.py — sun-angle timelapse from pbr_terrain.py, resumable per frame, encoded with ffmpeg.

Renders keyframes with the sun at real solar positions for a date and time window (or a straight
az/el path), reusing the cached elevation data and, with --geom-cache, the sun-independent ray hits.
Keyframes are blended in linear light to make the in-between frames, exposure follows a smoothed
log-average so the brightness drifts the way a camera on auto would, and night frames are biased
darker. Everything after `--` goes to pbr_terrain.py unchanged.

  python timelapse.py --work work/tl --keyframes 40 --interp 4 --size 1200x400 \\
      --date 2026-04-20 --utc-offset 4 --start 18:10 --end 20:40 --out ararat_timelapse.mp4 -- \\
      --peak 39.7019,44.2986 --from 40.1925,44.5150 --cam-offset 0 --city 0.8
  (run it again to resume; finished keyframes are skipped)

Keyframes live in <work>/keys_<hash>, frames in <work>/frames_<hash>. The hashes cover everything that changes
the pictures (sun plan, size, pbr_terrain arguments; for frames also --interp and the exposure settings), so a
rerun with other settings starts fresh folders instead of reusing stale ones, and switching back resumes.
"""
import argparse, datetime as dt, hashlib, json, math, os, shutil, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--work", required=True, help="folder for keyframes, frames and pbr_terrain caches")
ap.add_argument("--out", required=True, help="output .mp4")
ap.add_argument("--size", default="1200x400")
ap.add_argument("--keyframes", type=int, default=40, help="rendered frames")
ap.add_argument("--interp", type=int, default=4, help="output frames per keyframe step (linear-light blends)")
ap.add_argument("--fps", type=int, default=24)
ap.add_argument("--date", default="2026-04-20", help="YYYY-MM-DD for the solar path")
ap.add_argument("--utc-offset", type=float, default=0.0, help="hours; local time = UTC + offset")
ap.add_argument("--start", default="18:00", help="local start time HH:MM")
ap.add_argument("--end", default="20:30", help="local end time HH:MM")
ap.add_argument("--sun-path", default=None, help="instead of a date: az0,el0:az1,el1 straight sweep")
ap.add_argument("--night-bias", type=float, default=0.4, help="exposure multiplier once the sun is 5 deg down (darker nights)")
ap.add_argument("--key", type=float, default=0.30, help="target log-average brightness")
ap.add_argument("--max-exposure", type=float, default=14.0, help="cap, so night falls instead of being pushed to daylight")
ap.add_argument("--only-plan", action="store_true", help="print the sun positions and stop")
ap.add_argument("pbr", nargs=argparse.REMAINDER, help="-- then pbr_terrain.py arguments")
a = ap.parse_args()
pbr = [x for x in a.pbr if x != "--"]

def opt(name, default=None):
    return pbr[pbr.index(name) + 1] if name in pbr else default

def sun_position(lat, lon, when_utc):
    """solar azimuth (deg from north, clockwise) and elevation (deg); ~0.1 deg, plus refraction near the horizon"""
    d = (when_utc - dt.datetime(2000, 1, 1, 12)).total_seconds() / 86400.0
    g = math.radians((357.529 + 0.98560028 * d) % 360); q = (280.459 + 0.98564736 * d) % 360
    L = math.radians((q + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g)) % 360)
    e = math.radians(23.439 - 0.00000036 * d)
    ra = math.atan2(math.cos(e) * math.sin(L), math.cos(L)); dec = math.asin(math.sin(e) * math.sin(L))
    gmst = (18.697374558 + 24.06570982441908 * d) % 24
    ha = math.radians(gmst * 15 + lon) - ra
    la = math.radians(lat)
    el = math.asin(math.sin(la) * math.sin(dec) + math.cos(la) * math.cos(dec) * math.cos(ha))
    az = math.atan2(-math.sin(ha), math.tan(dec) * math.cos(la) - math.sin(la) * math.cos(ha))
    el_d = math.degrees(el)
    if el_d > -1.0: el_d += 1.02 / math.tan(math.radians(el_d + 10.3 / (el_d + 5.11))) / 60     # Saemundsson refraction
    return math.degrees(az) % 360, el_d

n = a.keyframes
if a.sun_path:
    (az0, el0), (az1, el1) = [tuple(map(float, p.split(","))) for p in a.sun_path.split(":")]
    plan = [(az0 + (az1 - az0) * k / (n - 1), el0 + (el1 - el0) * k / (n - 1)) for k in range(n)]
else:
    lat, lon = map(float, opt("--from").split(","))
    day = dt.datetime.strptime(a.date, "%Y-%m-%d")
    t0 = day + dt.timedelta(hours=int(a.start[:2]), minutes=int(a.start[3:])) - dt.timedelta(hours=a.utc_offset)
    t1 = day + dt.timedelta(hours=int(a.end[:2]), minutes=int(a.end[3:])) - dt.timedelta(hours=a.utc_offset)
    plan = [sun_position(lat, lon, t0 + (t1 - t0) * k / (n - 1)) for k in range(n)]
for k, (az, el) in enumerate(plan[:: max(1, n // 8)]):
    print(f"  key {k * max(1, n // 8):3d}: sun az {az:6.1f}  el {el:6.2f}")
if a.only_plan: sys.exit()

os.makedirs(a.work, exist_ok=True)
W, H = map(int, a.size.lower().split("x"))
_md5 = lambda o: hashlib.md5(json.dumps(o, sort_keys=True).encode()).hexdigest()[:10]
KH = _md5(dict(plan=[[round(az, 4), round(el, 4)] for az, el in plan], size=[W, H], pbr=pbr))
FH = _md5(dict(keys=KH, interp=a.interp, night_bias=a.night_bias, key=a.key, max_exposure=a.max_exposure))
KEYS = os.path.join(a.work, f"keys_{KH}"); FR = os.path.join(a.work, f"frames_{FH}")
os.makedirs(KEYS, exist_ok=True); os.makedirs(FR, exist_ok=True)
json.dump(dict(keys_hash=KH, frames_hash=FH, plan=plan, args=vars(a)), open(os.path.join(KEYS, "plan.json"), "w"), indent=1)
print(f"keyframes -> {KEYS}\nframes    -> {FR}", flush=True)

# ---- 1. keyframes (resumable: a keyframe is done when its linear image exists)
for k, (az, el) in enumerate(plan):
    hdr = os.path.join(KEYS, f"k{k:03d}.npy")
    if os.path.exists(hdr): continue
    tag = f"tl{KH}_{k:03d}"
    cmd = [sys.executable, os.path.join(HERE, "pbr_terrain.py"), *pbr, "--work", a.work, "--size", a.size, "--tag", tag,
           "--sun-az", f"{az:.3f}", "--sun-el", f"{el:.3f}", "--sky", "physical", "--geom-cache"]
    print(f"keyframe {k + 1}/{n}: az {az:.1f} el {el:.2f}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode: print(r.stdout[-2000:], r.stderr[-3000:]); sys.exit(f"pbr_terrain failed on keyframe {k}")
    bands = os.path.join(a.work, f"bands_{W}x{H}_{tag}")
    subprocess.run([sys.executable, os.path.join(HERE, "post_terrain.py"), "--bands", bands, "--out", os.path.join(KEYS, f"k{k:03d}.jpg"),
                    "--exposure", "auto", "--save-hdr", hdr + ".tmp.npy"], check=True, capture_output=True)
    os.replace(hdr + ".tmp.npy", hdr)
    shutil.rmtree(bands, ignore_errors=True)                 # the linear keyframe is all we need from here on

# ---- 2. exposure: smoothed log-average key, darker once the sun is down
lum = np.array([0.2126, 0.7152, 0.0722])
keys = np.array([math.exp(float(np.mean(np.log(np.load(os.path.join(KEYS, f"k{k:03d}.npy"), mmap_mode="r") @ lum + 1e-4)))) for k in range(n)])
logk = np.log(keys); sm = logk.copy()
for i in range(n):                                          # gentle smoothing so exposure never pumps
    w = np.exp(-0.5 * ((np.arange(n) - i) / 1.5) ** 2); sm[i] = (w * logk).sum() / w.sum()
els = np.array([p[1] for p in plan])
bias = a.night_bias + (1 - a.night_bias) * np.clip((els + 5) / 7, 0, 1)
expo = a.key / np.exp(sm) * bias
expo = a.max_exposure * np.tanh(expo / a.max_exposure)                  # soft cap
efile = os.path.join(FR, "exposure.json")
redo = not os.path.exists(efile) or json.load(open(efile)).get("exposure") != expo.tolist()   # re-tone frames if exposure changed
json.dump(dict(keys=keys.tolist(), exposure=expo.tolist()), open(efile, "w"), indent=1)

# ---- 3. in-between frames by blending keyframes in linear light, then the normal finish
idx = 0; tmp = os.path.join(a.work, "blend.npy")
total = (n - 1) * a.interp + 1
for k in range(n - 1 + 1):
    steps = [0] if k == n - 1 else range(a.interp)
    for s in steps:
        out = os.path.join(FR, f"f{idx:05d}.png"); idx += 1
        if os.path.exists(out) and not redo: continue
        f = s / a.interp
        A = np.load(os.path.join(KEYS, f"k{k:03d}.npy"))
        img = A if f == 0 else (1 - f) * A + f * np.load(os.path.join(KEYS, f"k{k + 1:03d}.npy"))
        e = math.exp((1 - f) * math.log(expo[k]) + f * math.log(expo[min(k + 1, n - 1)]))
        np.save(tmp, img.astype(np.float32))
        subprocess.run([sys.executable, os.path.join(HERE, "post_terrain.py"), "--hdr-in", tmp, "--out", out,
                        "--exposure", f"{e:.5f}", "--seed", str(idx)], check=True, capture_output=True)
    print(f"frames {idx}/{total}", flush=True)

# ---- 4. encode exactly this run's frames; libx264 / yuv420p needs even dimensions, so odd sizes are padded down
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", os.path.join(FR, "f%05d.png"),
                "-frames:v", str(total), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "slow", "-movflags", "+faststart", a.out], check=True)
print(a.out, f"{total} frames, {total / a.fps:.1f} s")
