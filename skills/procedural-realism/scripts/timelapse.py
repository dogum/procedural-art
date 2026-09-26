#!/usr/bin/env python3
"""
timelapse.py — sun-angle timelapse from pbr_terrain.py, resumable per frame, encoded with ffmpeg.

Every output frame is rendered with the sun where it is at that instant (real solar position for a date, place
and local time window, or a straight az/el sweep), so shadows and the Earth-shadow line slide instead of
cross-fading. Renders reuse the cached elevation data and, through --geom-cache, everything that does not depend
on the sun (ray hits, near-field visibility, the blurred heightfield, the sun transmittance table). Exposure follows
a log-average key smoothed over a few minutes of sun time, so brightness drifts the way a camera on auto would, and
night frames are biased darker. Everything after `--` goes to pbr_terrain.py unchanged.

  python timelapse.py --work work/tl --out ararat_timelapse.mp4 --frames 480 --size 1200x400 \\
      --date 2026-04-20 --utc-offset 4 --start 18:40 --end 20:40 -- \\
      --peak 39.7019,44.2986 --from 40.1963,44.5238 --cam-offset 0 --city 0.8
  (run it again to resume; finished frames are skipped)

--blend K renders only every K-th frame and cross-fades the rest in linear light (the old, cheaper method: about
K times faster, but shadows fade instead of moving). --keyframes/--interp are the old names: n keyframes blended
i times give (n - 1) * i + 1 frames.

Renders live in <work>/render_<hash> as half-float linear images named by sun position, so any plan that asks
for the same sun (a longer clip, another --blend) reuses them. The hash covers the size, the pbr_terrain.py
arguments and the renderer's source files. Tone-mapped frames go to <work>/frames_<hash>, keyed also by the plan,
--blend and the exposure settings.
"""
import argparse, datetime as dt, hashlib, json, math, os, re, shutil, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--work", required=True, help="folder for renders, frames and pbr_terrain caches")
ap.add_argument("--out", required=True, help="output .mp4")
ap.add_argument("--size", default="1200x400")
ap.add_argument("--frames", type=int, default=480, help="output frames (480 = 20 s at 24 fps)")
ap.add_argument("--blend", type=int, default=1, help="render every K-th frame and cross-fade between them (1 = render all)")
ap.add_argument("--keyframes", type=int, default=None, help="old name: rendered frames, with --interp")
ap.add_argument("--interp", type=int, default=None, help="old name for --blend")
ap.add_argument("--fps", type=int, default=24)
ap.add_argument("--date", default="2026-04-20", help="YYYY-MM-DD for the solar path")
ap.add_argument("--utc-offset", type=float, default=0.0, help="hours; local time = UTC + offset")
ap.add_argument("--start", default="18:00", help="local start time HH:MM")
ap.add_argument("--end", default="20:30", help="local end time HH:MM")
ap.add_argument("--sun-path", default=None, help="instead of a date: az0,el0:az1,el1 straight sweep")
ap.add_argument("--night-bias", type=float, default=0.3, help="exposure multiplier once the sun is 5 deg down; 0.3 matches a "
                "blue-hour still posted with auto*0.3 (ramps from 1 at +2 deg)")
ap.add_argument("--key", type=float, default=0.30, help="target log-average brightness")
ap.add_argument("--max-exposure", type=float, default=60.0, help="soft cap, so night falls instead of being pushed to daylight "
                "(60 leaves blue hour at -5 deg within 5 %% of the still's exposure)")
ap.add_argument("--smooth", type=float, default=4.5, help="exposure smoothing, minutes of sun time (a --sun-path sweep counts as 2 h)")
ap.add_argument("--only-plan", action="store_true", help="print the sun positions and stop")
ap.add_argument("pbr", nargs=argparse.REMAINDER, help="-- then pbr_terrain.py arguments")
a = ap.parse_args()
pbr = [x for x in a.pbr if x != "--"]
if a.keyframes is not None or a.interp is not None:                      # old flags
    a.blend = a.interp or 4; a.frames = ((a.keyframes or 40) - 1) * a.blend + 1
K = max(1, a.blend)
n = a.frames
if (n - 1) % K:
    n = ((n - 1) // K + 1) * K + 1; print(f"--blend {K}: {n} frames, so the last frame is a rendered one")

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
    # Saemundsson refraction; below -1 deg it tapers to zero at -3 deg instead of switching off, which made the
    # sun jump 0.65 deg between two frames
    h = max(el_d, -1.0)
    el_d += 1.02 / math.tan(math.radians(h + 10.3 / (h + 5.11))) / 60 * min(1.0, max(0.0, (el_d + 3.0) / 2.0))
    return math.degrees(az) % 360, el_d

# ---- plan: one sun position per output frame; minutes of sun time drive the exposure smoothing
if a.sun_path:
    (az0, el0), (az1, el1) = [tuple(map(float, p.split(","))) for p in a.sun_path.split(":")]
    plan = [(az0 + (az1 - az0) * k / (n - 1), el0 + (el1 - el0) * k / (n - 1)) for k in range(n)]
    minutes = np.linspace(0, 120, n)
else:
    lat, lon = map(float, opt("--from").split(","))
    day = dt.datetime.strptime(a.date, "%Y-%m-%d")
    t0 = day + dt.timedelta(hours=int(a.start[:2]), minutes=int(a.start[3:])) - dt.timedelta(hours=a.utc_offset)
    t1 = day + dt.timedelta(hours=int(a.end[:2]), minutes=int(a.end[3:])) - dt.timedelta(hours=a.utc_offset)
    plan = [sun_position(lat, lon, t0 + (t1 - t0) * k / (n - 1)) for k in range(n)]
    minutes = np.linspace(0, (t1 - t0).total_seconds() / 60, n)
plan = [(round(az, 4), round(el, 4)) for az, el in plan]
rendered = list(range(0, n, K))
for k in range(0, n, max(1, n // 8)):
    print(f"  frame {k:4d}: sun az {plan[k][0]:6.1f}  el {plan[k][1]:6.2f}")
step = (minutes[1] - minutes[0]) * 60 if n > 1 else 0
print(f"{n} frames ({n / a.fps:.1f} s at {a.fps} fps), {len(rendered)} rendered, {step:.0f} s of sun time per frame")
if a.only_plan: sys.exit()

os.makedirs(a.work, exist_ok=True)
W, H = map(int, a.size.lower().split("x"))
_md5 = lambda o: hashlib.md5(json.dumps(o, sort_keys=True).encode()).hexdigest()[:10]
src = {f: hashlib.md5(open(os.path.join(HERE, f), "rb").read()).hexdigest()
       for f in ("pbr_terrain.py", "terrain_near.py", "post_terrain.py") if os.path.exists(os.path.join(HERE, f))}
RH = _md5(dict(size=[W, H], pbr=pbr, src=src, fmt=2))
FH = _md5(dict(render=RH, plan=plan, blend=K, night_bias=a.night_bias, key=a.key, max_exposure=a.max_exposure, smooth=a.smooth))
RD = os.path.join(a.work, f"render_{RH}"); FR = os.path.join(a.work, f"frames_{FH}")
os.makedirs(RD, exist_ok=True); os.makedirs(FR, exist_ok=True)
json.dump(dict(size=[W, H], pbr=pbr, src=src), open(os.path.join(RD, "args.json"), "w"), indent=1)
json.dump(dict(render_hash=RH, frames_hash=FH, plan=plan, args=vars(a)), open(os.path.join(FR, "plan.json"), "w"), indent=1)
print(f"renders -> {RD}\nframes  -> {FR}", flush=True)
name = lambda k: os.path.join(RD, f"sun_{plan[k][0]:.4f}_{plan[k][1]:+.4f}")
lum = np.array([0.2126, 0.7152, 0.0722], np.float32)

# ---- 1. renders (resumable: a frame is done when its linear image exists; an interrupted one resumes its bands)
todo = [k for k in rendered if not os.path.exists(name(k) + ".npy")]
t_start = time.time()
for j, k in enumerate(todo):
    az, el = plan[k]; hdr = name(k)
    tag = f"tl{RH}_{az:.4f}_{el:+.4f}"
    cmd = [sys.executable, os.path.join(HERE, "pbr_terrain.py"), *pbr, "--work", a.work, "--size", a.size, "--tag", tag,
           "--sun-az", f"{az:.4f}", "--sun-el", f"{el:.4f}", "--sky", "physical", "--geom-cache"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode: print(r.stdout[-2000:], r.stderr[-3000:]); sys.exit(f"pbr_terrain failed on frame {k}")
    bands = os.path.join(a.work, f"bands_{W}x{H}_{tag}")
    tmp = os.path.join(RD, "composite.tmp.npy")                      # city lights, bokeh and DOF composited in linear light
    # post applies the blue-hour white balance and saturation (they ramp in with the sun from +1 to -2 deg) before
    # saving the linear image, and reports its exposure key: whole frame by day, sky and far terrain at blue hour
    p = subprocess.run([sys.executable, os.path.join(HERE, "post_terrain.py"), "--bands", bands, "--out", os.path.join(RD, "preview.tmp.jpg"),
                        "--exposure", "auto", "--save-hdr", tmp], check=True, capture_output=True, text=True)
    img = np.load(tmp)
    m_ = re.search(r"auto exposure ([0-9.eE+-]+)", p.stdout)                 # exposure = 0.30 / key
    key = 0.30 / float(m_.group(1)) if m_ else math.exp(float(np.mean(np.log(img @ lum + 1e-4))))
    # half floats divided by the key, so dusk and night (keys down to ~1e-4) keep ~0.1 % steps like daylight
    np.save(hdr + ".tmp.npy", np.minimum(img / key, 65000).astype(np.float16))
    open(hdr + ".key", "w").write(repr(key))
    os.replace(hdr + ".tmp.npy", hdr + ".npy")
    shutil.rmtree(bands, ignore_errors=True)                         # the linear frame is all we need from here on
    el_t = time.time() - t_start
    print(f"render {j + 1}/{len(todo)} (frame {k}): az {az:.2f} el {el:+.2f}  {el_t / (j + 1):.1f} s/frame, "
          f"~{el_t / (j + 1) * (len(todo) - j - 1) / 60:.0f} min left", flush=True)
for f in ("composite.tmp.npy", "preview.tmp.jpg"):
    if os.path.exists(os.path.join(RD, f)): os.remove(os.path.join(RD, f))

# ---- 2. exposure: log-average key smoothed over sun time, darker once the sun is down
def key_of(k):
    return float(open(name(k) + ".key").read())
def load(k):
    return np.load(name(k) + ".npy").astype(np.float32) * np.float32(key_of(k))
logk = np.log([key_of(k) for k in rendered]); tr = minutes[rendered]
sig = max(a.smooth, 1e-6)
wts = np.exp(-0.5 * ((minutes[:, None] - tr[None]) / sig) ** 2)
sm = (wts * logk[None]).sum(1) / wts.sum(1)                             # per output frame, so exposure never steps
els = np.array([p[1] for p in plan])
bias = a.night_bias + (1 - a.night_bias) * np.clip((els + 5) / 7, 0, 1)
expo = a.key / np.exp(sm) * bias
expo = a.max_exposure * np.tanh(expo / a.max_exposure)                  # soft cap
efile = os.path.join(FR, "exposure.json")
redo = not os.path.exists(efile) or json.load(open(efile)).get("exposure") != expo.tolist()   # re-tone frames if exposure changed
json.dump(dict(keys=np.exp(logk).tolist(), exposure=expo.tolist()), open(efile, "w"), indent=1)

# ---- 3. tone-map every frame (cross-fading neighbours in linear light only with --blend > 1)
tmp = os.path.join(a.work, f"tone_{FH}.npy")
for i in range(n):
    out = os.path.join(FR, f"f{i:05d}.png")
    if os.path.exists(out) and not redo: continue
    k0 = i // K * K; f = (i - k0) / K
    img = load(k0)
    if f > 0: img = (1 - f) * img + f * load(k0 + K)
    np.save(tmp, img)
    subprocess.run([sys.executable, os.path.join(HERE, "post_terrain.py"), "--hdr-in", tmp, "--out", out,
                    "--exposure", f"{expo[i]:.5f}", "--seed", str(i + 1)], check=True, capture_output=True)
    if i % 24 == 0 or i == n - 1: print(f"frames {i + 1}/{n}", flush=True)
if os.path.exists(tmp): os.remove(tmp)

# ---- 4. encode exactly this run's frames; libx264 / yuv420p needs even dimensions, so odd sizes are padded down
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", os.path.join(FR, "f%05d.png"),
                "-frames:v", str(n), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "slow", "-movflags", "+faststart", a.out], check=True)
print(a.out, f"{n} frames, {n / a.fps:.1f} s")
