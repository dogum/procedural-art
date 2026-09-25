#!/usr/bin/env python3
"""
regression_terrain.py — render a fixed set of terrain shapes in all six styles and build one
labelled contact sheet (rows = mountains, columns = styles).

The set covers the shapes the auto-framing has to handle: cone (Fuji), twin volcano (Ararat),
horn (Matterhorn), massif (Denali), range (Teton Range), canyon (Grand Canyon) and island
volcano (Mauna Kea, ocean clipped to 0).

  nice -n 10 python scripts/regression_terrain.py                  # draft sheet, cached renders reused
  nice -n 10 python scripts/regression_terrain.py --rerender       # re-render everything
  nice -n 10 python scripts/regression_terrain.py --only matterhorn,canyon --styles survey,woodcut --rerender
  # compare against another copy of the engine (e.g. a backup made before a change)
  nice -n 10 python scripts/regression_terrain.py --scripts work/terrain/orig_scripts --renders work/terrain/regression_before \
      --sheet work/terrain/regression_before.jpg

DEMs are cached under work/terrain/dem/<key>.npz and never re-downloaded unless deleted.
Renders go to work/terrain/regression/<key>/<style>.png and are skipped when present.
Each render runs render.py in a subprocess, so a crash in one style does not stop the sheet.
"""
import argparse, json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "skills", "procedural-terrain-art", "scripts")
STYLES = ["survey", "topo", "nocturne", "stipple", "riso", "woodcut"]

# Summits and viewpoints looked up on Wikipedia / photographylife.com (Wikipedia; Snake River Overlook from photographylife.com).
# Zermatt (46.0170,7.7500) is the other classic Matterhorn viewpoint; Riffelsee gives the better-known profile.
# `args` are extra render.py flags; keep them empty unless the shape really needs a manual choice,
# because the point of this set is to exercise the automatic framing.
MOUNTAINS = [
    dict(key="fuji", name="Fuji", shape="cone", peak="35.3606,138.7274", view="35.5100,138.7550",
         where="from Lake Kawaguchi", labels=["富士山|3776 m"], title="富士山", radius=65),
    dict(key="ararat", name="Ararat", shape="twin volcano", peak="39.7019,44.2986", view="40.1792,44.4991",
         where="from Yerevan", labels=["ՄԱՍԻՍ|5137 m", "ՍԻՍ|3896 m@39.6517,44.4011"], title="ԱՐԱՐԱՏ", radius=65),
    dict(key="matterhorn", name="Matterhorn", shape="horn", peak="45.97639,7.65861", view="45.98333,7.76222",
         where="from Riffelsee", labels=["Matterhorn|4478 m"], title="Cervino", radius=65),
    dict(key="denali", name="Denali", shape="massif", peak="63.06917,-151.00639", view="63.47611,-150.87694",
         where="from Wonder Lake", labels=["Denali|6190 m"], title="Denali", radius=80),
    dict(key="teton", name="Teton Range", shape="range", peak="43.741208,-110.802414", view="43.752979,-110.624420",
         where="from Snake River Overlook", labels=["Grand Teton|4199 m", "Mount Moran|3842 m@43.83528,-110.77639"],
         title="Tetons", radius=65),
    dict(key="canyon", name="Grand Canyon", shape="canyon", peak="36.1304634,-112.0387030", view="36.0616497,-112.1079463",
         where="from Mather Point", labels=["Brahma Temple|2302 m", "Zoroaster Temple|2171 m@36.1188,-112.0452"],
         title="Grand Canyon", radius=65),
    dict(key="mauna_kea", name="Mauna Kea", shape="island volcano", peak="19.82056,-155.46806", view="19.70556,-155.08583",
         where="from Hilo", labels=["Mauna Kea|4207 m", "Mauna Loa|4169 m@19.47944,-155.60278"], title="Maunakea", radius=65),
]


def ensure_dem(m, dem_dir, scripts):
    path = os.path.join(dem_dir, f"{m['key']}.npz")
    if os.path.exists(path): return path
    sys.path.insert(0, scripts)
    from terrain_art import fetch_dem
    lat, lon = map(float, m["peak"].split(","))
    print(f"[dem] fetching {m['key']} (radius {m['radius']} km)", flush=True)
    fetch_dem(lat, lon, m["radius"], path)
    return path


def render_one(m, style, dem, out, a):
    cmd = [sys.executable, os.path.join(a.scripts, "render.py"), "--peak", m["peak"], "--from", m["view"],
           "--style", style, "--dem", dem, "--size", a.size, "--quality", str(a.quality), "--out", out,
           "--title", m["title"], "--footnote", f"{m['name']} · {m['where']}"]
    for L in m["labels"]: cmd += ["--label", L]
    cmd += m.get("args", [])
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    info = {}
    for line in r.stdout.splitlines():
        if line.startswith("{"):
            try: info = json.loads(line)
            except ValueError: pass
    if r.returncode != 0:
        print(r.stderr[-2000:], file=sys.stderr)
    return r.returncode == 0, dt, info


def sheet(rows, styles, renders, path, tw, size):
    from PIL import Image, ImageDraw, ImageFont
    W, H = map(int, size.lower().split("x")); th = int(tw * H / W)
    lw, pad, head = 250, 14, 44
    font = lambda n: ImageFont.truetype(os.path.join("/usr/share/fonts/truetype/dejavu", n), 20) \
        if os.path.exists("/usr/share/fonts/truetype/dejavu/" + n) else ImageFont.load_default()
    f1, f2 = font("DejaVuSans-Bold.ttf"), font("DejaVuSans.ttf")
    try: f3 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    except OSError: f3 = f2
    im = Image.new("RGB", (lw + len(styles) * (tw + pad) + pad, head + len(rows) * (th + pad) + pad), (24, 24, 26))
    d = ImageDraw.Draw(im)
    for c, st in enumerate(styles):
        d.text((lw + pad + c * (tw + pad), 12), st, fill=(225, 225, 225), font=f1)
    for r, (m, info) in enumerate(rows):
        y = head + r * (th + pad)
        d.text((16, y + 6), m["name"], fill=(240, 240, 240), font=f1)
        d.text((16, y + 34), m["shape"], fill=(170, 170, 170), font=f2)
        d.text((16, y + 60), m["where"], fill=(140, 140, 140), font=f3)
        yy = y + 86
        for k in ("kind", "peak_m", "base_m", "extent_km", "relief"):
            if k in info:
                d.text((16, yy), f"{k}: {info[k]}", fill=(140, 140, 140), font=f3); yy += 20
        for c, st in enumerate(styles):
            p = os.path.join(renders, m["key"], f"{st}.png")
            x = lw + pad + c * (tw + pad)
            if os.path.exists(p):
                im.paste(Image.open(p).convert("RGB").resize((tw, th), Image.LANCZOS), (x, y))
            else:
                d.rectangle([x, y, x + tw, y + th], outline=(90, 40, 40)); d.text((x + 10, y + 10), "failed", fill=(200, 80, 80), font=f2)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    im.save(path, quality=88)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="comma-separated mountain keys (" + ",".join(m["key"] for m in MOUNTAINS) + ")")
    ap.add_argument("--styles", default=",".join(STYLES), help="comma-separated styles")
    ap.add_argument("--rerender", action="store_true", help="render again even if a PNG exists")
    ap.add_argument("--size", default="1200x400"); ap.add_argument("--quality", type=float, default=0.5)
    ap.add_argument("--scripts", default=SCRIPTS, help="engine scripts dir (render.py, terrain_art.py)")
    ap.add_argument("--dem-dir", default=os.path.join(ROOT, "work", "terrain", "dem"))
    ap.add_argument("--renders", default=os.path.join(ROOT, "work", "terrain", "regression"))
    ap.add_argument("--sheet", default=os.path.join(ROOT, "work", "terrain", "regression_sheet.jpg"))
    ap.add_argument("--thumb", type=int, default=600, help="thumbnail width in the sheet")
    ap.add_argument("--no-sheet", action="store_true")
    a = ap.parse_args()
    a.scripts = os.path.abspath(a.scripts)
    keys = a.only.split(",") if a.only else [m["key"] for m in MOUNTAINS]
    styles = a.styles.split(",")
    os.makedirs(a.dem_dir, exist_ok=True)
    timings = {}
    log_path = os.path.join(a.renders, "info.json")
    infos = json.load(open(log_path)) if os.path.exists(log_path) else {}
    for m in MOUNTAINS:
        if m["key"] not in keys: continue
        dem = ensure_dem(m, a.dem_dir, a.scripts)
        os.makedirs(os.path.join(a.renders, m["key"]), exist_ok=True)
        for st in styles:
            out = os.path.join(a.renders, m["key"], f"{st}.png")
            if os.path.exists(out) and not a.rerender: continue
            ok, dt, info = render_one(m, st, dem, out, a)
            timings[f"{m['key']}/{st}"] = round(dt, 1)
            if info: infos[m["key"]] = info
            print(f"{m['key']:11s} {st:9s} {'ok' if ok else 'FAILED'}  {dt:5.1f} s  {json.dumps(info) if st == styles[0] else ''}", flush=True)
    os.makedirs(a.renders, exist_ok=True)
    json.dump(infos, open(log_path, "w"), indent=1)
    if timings:
        tot = sum(timings.values())
        print(f"rendered {len(timings)} images in {tot:.0f} s (mean {tot / len(timings):.1f} s)")
        json.dump(timings, open(os.path.join(a.renders, "timings.json"), "w"), indent=1)
    if not a.no_sheet:
        rows = [(m, infos.get(m["key"], {})) for m in MOUNTAINS if m["key"] in keys]
        print("sheet ->", sheet(rows, STYLES, a.renders, a.sheet, a.thumb, a.size))


if __name__ == "__main__":
    main()
