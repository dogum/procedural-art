#!/usr/bin/env python3
"""osm_roads.py — real streets from OpenStreetMap for the night city (pbr_terrain.py --city).

Fetches highways (and optionally building footprints and residential landuse) for a lat/lon box through the
Overpass API, caches the raw JSON per 0.05° tile in the work folder, and rasterises the roads onto the renderer's
local grid (km east/north of the summit, the same equirectangular frame pbr_terrain.py uses):

  segments   every road piece as a straight segment in local metres, with class, width and way index
  grid       per cell: distance to the nearest road centre line (m), that road's class, width and direction

Classes, largest first: motorway, trunk, primary, secondary, tertiary, residential (with unclassified and
living_street), service. *_link roads take their parent's class. Footways, paths and tracks are left out.

Road data © OpenStreetMap contributors, available under the Open Database License (ODbL 1.0,
https://www.openstreetmap.org/copyright). Anything published from it must carry that credit.

When every Overpass mirror fails, load_roads() returns None and the caller keeps the synthetic street grid.

  python osm_roads.py --bbox 40.08,44.40,40.21,44.60 --origin 39.7019,44.2986 --work work/yerevan --debug roads.png
"""
import argparse, gzip, json, math, os, socket, ssl, sys, time, http.client, urllib.parse
import numpy as np
from scipy.ndimage import distance_transform_edt

MIRRORS = ["https://overpass-api.de/api/interpreter",
           "https://overpass.kumi.systems/api/interpreter",
           "https://overpass.private.coffee/api/interpreter",
           "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]
USER_AGENT = "procedural-art (github.com/dogum/procedural-art)"
TILE = 0.05                                                   # degrees; one cached Overpass query per tile
_GOOD = []                                                    # mirrors that answered, most recent first
CLASSES = ["motorway", "trunk", "primary", "secondary", "tertiary", "residential", "service"]
_CLS = {"motorway": 0, "motorway_link": 0, "trunk": 1, "trunk_link": 1, "primary": 2, "primary_link": 2,
        "secondary": 3, "secondary_link": 3, "tertiary": 4, "tertiary_link": 4, "residential": 5,
        "unclassified": 5, "living_street": 5, "service": 6}
WIDTH = np.array([24.0, 20.0, 15.0, 12.0, 9.5, 7.0, 4.5])   # typical paved width per class, metres
LANE_W = 3.3


# ------------------------------------------------------------------ Overpass
def _post(url, query, timeout):
    """POST one Overpass query; honours HTTPS_PROXY. Keeps the connection header default (keep-alive):
    some proxies drop tunnelled requests that ask for Connection: close."""
    u = urllib.parse.urlsplit(url)
    body = urllib.parse.urlencode({"data": query}).encode()
    headers = {"User-Agent": USER_AGENT, "Content-Type": "application/x-www-form-urlencoded", "Accept": "*/*",
               "Accept-Encoding": "gzip"}
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    ctx = ssl.create_default_context()
    if proxy and not _no_proxy(u.hostname):
        p = urllib.parse.urlsplit(proxy if "://" in proxy else "http://" + proxy)
        conn = http.client.HTTPSConnection(p.hostname, p.port or 80, timeout=timeout, context=ctx)
        conn.set_tunnel(u.hostname, u.port or 443)
    else:
        conn = http.client.HTTPSConnection(u.hostname, u.port or 443, timeout=timeout, context=ctx)
    try:
        conn.request("POST", u.path, body=body, headers=headers)
        r = conn.getresponse(); data = r.read()
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status} {data[:200]!r}")
        if r.getheader("Content-Encoding", "") == "gzip":
            data = gzip.decompress(data)
        return json.loads(data)
    finally:
        conn.close()

def _no_proxy(host):
    for n in (os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or "").split(","):
        n = n.strip().lstrip("*")
        if n and (host == n.lstrip(".") or host.endswith(n if n.startswith(".") else "." + n)):
            return True
    return False

def _query(s, w, n, e, what):
    hw = "|".join(_CLS)
    parts = [f'way["highway"~"^({hw})$"];']
    if "buildings" in what: parts.append('way["building"];')
    if "landuse" in what: parts.append('way["landuse"="residential"];')
    return f'[out:json][timeout:120][bbox:{s:.4f},{w:.4f},{n:.4f},{e:.4f}];({"".join(parts)});out tags geom qt;'

def fetch_tile(s, w, what, cache_dir, mirrors=None, timeout=120, log=print):
    """one TILE×TILE box (south-west corner s, w); returns the Overpass JSON dict or None"""
    n, e = s + TILE, w + TILE
    key = f"{'+'.join(sorted(what))}_{s:.2f}_{w:.2f}_{n:.2f}_{e:.2f}"
    f = os.path.join(cache_dir, f"osm_{key}.json.gz")
    if os.path.exists(f):
        with gzip.open(f, "rt") as fh: return json.load(fh)
    q = _query(s, w, n, e, what)
    urls = mirrors or [m for m in os.environ.get("OVERPASS_URL", "").split(",") if m] or MIRRORS
    urls = sorted(urls, key=lambda u: _GOOD.index(u) if u in _GOOD else len(_GOOD))   # last good mirror first
    for rnd in range(2):
        for url in urls:
            try:
                t0 = time.time(); d = _post(url, q, min(timeout, 45) if rnd == 0 else timeout)   # quick pass first
                if d.get("remark", "").lower().startswith(("runtime error", "error")):
                    raise RuntimeError(d["remark"][:200])
                os.makedirs(cache_dir, exist_ok=True)
                with gzip.open(f + ".part", "wt") as fh: json.dump(d, fh)
                os.replace(f + ".part", f)
                if url in _GOOD: _GOOD.remove(url)
                _GOOD.insert(0, url)
                log(f"  osm tile {key}: {len(d.get('elements', []))} elements from {urllib.parse.urlsplit(url).hostname} "
                    f"in {time.time() - t0:.0f}s")
                return d
            except (OSError, socket.timeout, http.client.HTTPException, RuntimeError, ValueError) as ex:
                log(f"  osm tile {key}: {urllib.parse.urlsplit(url).hostname} failed ({str(ex)[:80]})")
        if rnd == 0: time.sleep(10)                                   # busy servers: one more pass
    return None

def load_roads(bbox, cache_dir, what=("roads",), mirrors=None, log=print):
    """bbox = (south, west, north, east). Returns dict(roads=[...], buildings=[...], landuse=[...]) with lat/lon
    polylines, or None if any tile could not be fetched (then use the synthetic street grid)."""
    what = tuple(sorted(set(what) | {"roads"}))
    s, w, n, e = bbox
    i0, i1 = math.floor(s / TILE + 1e-9), math.ceil(n / TILE - 1e-9)
    j0, j1 = math.floor(w / TILE + 1e-9), math.ceil(e / TILE - 1e-9)
    seen, out = set(), dict(roads=[], buildings=[], landuse=[])
    for i in range(i0, i1):
        for j in range(j0, j1):
            d = fetch_tile(round(i * TILE, 4), round(j * TILE, 4), what, cache_dir, mirrors, log=log)
            if d is None:
                log("OpenStreetMap unavailable (every Overpass mirror failed); falling back to the synthetic streets")
                return None
            for el in d.get("elements", []):
                if el.get("type") != "way" or el["id"] in seen or "geometry" not in el: continue
                seen.add(el["id"]); t = el.get("tags", {})
                pts = np.array([[g["lat"], g["lon"]] for g in el["geometry"] if g], float)
                if len(pts) < 2: continue
                if t.get("highway") in _CLS:
                    c = _CLS[t["highway"]]; wd = _width(t, c)
                    if t["highway"].endswith("_link"): wd = max(wd * 0.55, 5.0)
                    if t.get("area") == "yes": continue
                    out["roads"].append(dict(id=el["id"], cls=c, width=wd, pts=pts, lit=t.get("lit"),
                                             oneway=t.get("oneway") == "yes", service=t.get("service"),
                                             tunnel=t.get("tunnel", "no") != "no", bridge=t.get("bridge", "no") != "no",
                                             name=t.get("name:en") or t.get("name")))
                elif "building" in t:
                    lv = t.get("building:levels", "")
                    out["buildings"].append(dict(id=el["id"], pts=pts, levels=float(lv) if _num(lv) else None,
                                                 kind=t.get("building")))
                elif t.get("landuse") == "residential":
                    out["landuse"].append(dict(id=el["id"], pts=pts))
    return out

def _num(v):
    try: float(v); return True
    except (TypeError, ValueError): return False

def _width(t, c):
    w = t.get("width", "").replace("m", "").strip()
    if _num(w) and 2 < float(w) < 80: return float(w)
    if _num(t.get("lanes")) and 0 < float(t["lanes"]) < 12:
        return float(t["lanes"]) * LANE_W + (2.0 if c <= 2 else 1.0)
    return float(WIDTH[c])


# ------------------------------------------------------------------ local frame + raster
def local_xy(pts, origin):
    """lat/lon (n, 2) -> metres east/north of origin (the summit), same equirectangular frame as pbr_terrain.py"""
    la0, lo0 = origin
    kx, ky = 111.32 * math.cos(math.radians(la0)), 110.57
    return np.stack([(pts[:, 1] - lo0) * kx * 1000.0, (pts[:, 0] - la0) * ky * 1000.0], 1)

def segments(roads, origin, skip_tunnels=True):
    """all road pieces as straight segments in local metres: dict of arrays x0 y0 x1 y1 cls width way"""
    cols = {k: [] for k in ("x0", "y0", "x1", "y1", "cls", "width", "way")}
    for k, r in enumerate(roads):
        if skip_tunnels and r["tunnel"]: continue
        p = local_xy(r["pts"], origin)
        cols["x0"].append(p[:-1, 0]); cols["y0"].append(p[:-1, 1]); cols["x1"].append(p[1:, 0]); cols["y1"].append(p[1:, 1])
        n = len(p) - 1
        cols["cls"].append(np.full(n, r["cls"], np.int8)); cols["width"].append(np.full(n, r["width"], np.float32))
        cols["way"].append(np.full(n, k, np.int32))
    return {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in cols.items()}

def rasterise(seg, extent, res=8.0):
    """extent = (x0, x1, y0, y1) metres in the local frame. Returns a dict with the grid axes and per-cell
    dist (m to the nearest centre line), cls (-1 = none), width (m) and ang (road direction, radians from east).
    A cell that two roads cross keeps the larger class."""
    x0, x1, y0, y1 = extent
    nx, ny = int(math.ceil((x1 - x0) / res)), int(math.ceil((y1 - y0) / res))
    cls = np.full((ny, nx), 127, np.int8); wid = np.zeros((ny, nx), np.float32); ang = np.zeros((ny, nx), np.float32)
    if len(seg["x0"]):
        L = np.hypot(seg["x1"] - seg["x0"], seg["y1"] - seg["y0"])
        k = np.maximum(np.ceil(L / (res * 0.5)).astype(int), 1)
        idx = np.repeat(np.arange(len(L)), k + 1)
        t = np.concatenate([np.linspace(0, 1, kk + 1) for kk in k])
        px = seg["x0"][idx] + (seg["x1"] - seg["x0"])[idx] * t; py = seg["y0"][idx] + (seg["y1"] - seg["y0"])[idx] * t
        ix, iy = np.floor((px - x0) / res).astype(int), np.floor((py - y0) / res).astype(int)
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        ix, iy, idx = ix[ok], iy[ok], idx[ok]
        order = np.argsort(-seg["cls"][idx].astype(int), kind="stable")       # small classes first, big ones overwrite
        ix, iy, idx = ix[order], iy[order], idx[order]
        cls[iy, ix] = seg["cls"][idx]; wid[iy, ix] = seg["width"][idx]
        ang[iy, ix] = np.arctan2(seg["y1"] - seg["y0"], seg["x1"] - seg["x0"])[idx]
    road = cls != 127
    if road.any():
        dist, (ii, jj) = distance_transform_edt(~road, sampling=res, return_indices=True)
        c_, w_, a_ = cls[ii, jj], wid[ii, jj], ang[ii, jj]
        del ii, jj
    else:
        dist = np.full((ny, nx), 1e9); c_ = np.full((ny, nx), -1, np.int8); w_ = np.zeros((ny, nx), np.float32); a_ = w_.copy()
    c_ = np.where(c_ == 127, -1, c_).astype(np.int8)
    return dict(x0=x0, y0=y0, res=res, dist=dist.astype(np.float32), cls=c_, width=w_, ang=a_)

def sample(grid, x, y, key="dist"):
    """nearest-cell lookup of one grid array at local metres x, y (outside the grid: dist 1e9, cls -1)"""
    ix = np.floor((np.asarray(x) - grid["x0"]) / grid["res"]).astype(int)
    iy = np.floor((np.asarray(y) - grid["y0"]) / grid["res"]).astype(int)
    A = grid[key]; ok = (ix >= 0) & (ix < A.shape[1]) & (iy >= 0) & (iy < A.shape[0])
    fill = {"dist": 1e9, "cls": -1}.get(key, 0)
    out = np.full(np.shape(ix), fill, A.dtype)
    out[ok] = A[iy[ok], ix[ok]]
    return out

def road_field(bbox, origin, cache_dir, res=8.0, what=("roads",), log=print):
    """everything a renderer needs in one call: (data, segments, grid), cached as an .npz next to the tiles;
    None when OpenStreetMap is unreachable"""
    tag = f"{bbox[0]:.4f}_{bbox[1]:.4f}_{bbox[2]:.4f}_{bbox[3]:.4f}_{origin[0]:.5f}_{origin[1]:.5f}_{res:g}"
    data = load_roads(bbox, cache_dir, what, log=log)
    if data is None: return None
    seg = segments(data["roads"], origin)
    c = local_xy(np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]]), origin)
    gf = os.path.join(cache_dir, f"roadgrid_{tag}.npz")
    if os.path.exists(gf):
        z = np.load(gf); grid = {k: (z[k] if z[k].ndim else float(z[k])) for k in z.files}
    else:
        grid = rasterise(seg, (c[0, 0], c[1, 0], c[0, 1], c[1, 1]), res)
        np.savez_compressed(gf + ".part.npz", **grid); os.replace(gf + ".part.npz", gf)
    return data, seg, grid


# ------------------------------------------------------------------ debug view
def debug_image(data, seg, grid, out, origin, cam=None, view=None, max_px=1800):
    """view = (bearing°, hfov°, km): draw the camera's field of view from cam"""
    from PIL import Image, ImageDraw
    d, c = grid["dist"], grid["cls"]
    H_, W_ = d.shape
    pal = np.array([[255, 120, 60], [255, 170, 60], [255, 215, 90], [230, 230, 150], [190, 210, 230],
                    [150, 165, 190], [95, 105, 125]], float)
    glow = np.exp(-d / 60.0)[..., None]                                   # distance field as a soft falloff
    base = np.where(c[..., None] >= 0, pal[np.clip(c, 0, 6)], 0) * glow * 0.45 + np.array([12, 14, 22]) * (1 - glow)
    on = d <= np.maximum(grid["width"], grid["res"]) * 0.5                 # paved cells
    base[on] = pal[np.clip(c[on], 0, 6)]
    im = Image.fromarray(np.clip(base[::-1], 0, 255).astype(np.uint8))
    s = min(1.0, max_px / max(W_, H_)); im = im.resize((int(W_ * s), int(H_ * s)), Image.LANCZOS)
    dr = ImageDraw.Draw(im)
    for b in data.get("buildings", [])[:200000]:
        p = local_xy(b["pts"], origin)
        q = [((x - grid["x0"]) / grid["res"] * s, (H_ - (y - grid["y0"]) / grid["res"]) * s) for x, y in p]
        dr.polygon(q, outline=(70, 90, 70))
    if cam is not None:
        cx, cy = (cam[0] - grid["x0"]) / grid["res"] * s, (H_ - (cam[1] - grid["y0"]) / grid["res"]) * s
        dr.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), outline=(255, 255, 255), width=2)
        if view is not None:
            b, fov, km = view; r = km * 1000 / grid["res"] * s
            for t in (b - fov / 2, b + fov / 2):
                dr.line((cx, cy, cx + r * math.sin(math.radians(t)), cy - r * math.cos(math.radians(t))), fill=(255, 255, 255), width=1)
            dr.arc((cx - r, cy - r, cx + r, cy + r), b - fov / 2 - 90, b + fov / 2 - 90, fill=(255, 255, 255))
    y = 8
    for k, nm in enumerate(CLASSES):
        dr.rectangle((8, y, 22, y + 10), fill=tuple(int(v) for v in pal[k])); dr.text((28, y - 1), nm, fill=(230, 230, 230)); y += 15
    dr.text((8, im.size[1] - 16), "Road data (c) OpenStreetMap contributors, ODbL", fill=(200, 200, 200))
    im.save(out); return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bbox", required=True, help="south,west,north,east (degrees)")
    ap.add_argument("--origin", required=True, help="lat,lon of the renderer's origin (the --peak summit)")
    ap.add_argument("--work", default="./pbr_work", help="cache folder (tiles go to <work>/osm)")
    ap.add_argument("--res", type=float, default=8.0, help="grid cell, metres")
    ap.add_argument("--buildings", action="store_true", help="also fetch building footprints and residential landuse")
    ap.add_argument("--cam", default=None, help="lat,lon to mark on the debug image")
    ap.add_argument("--debug", default=None, help="write a debug image of the road field")
    ap.add_argument("--view", default=None, help="bearing,hfov,km: draw the camera's field of view on the debug image")
    a = ap.parse_args()
    bbox = tuple(map(float, a.bbox.split(","))); origin = tuple(map(float, a.origin.split(",")))
    t0 = time.time()
    r = road_field(bbox, origin, os.path.join(a.work, "osm"), a.res, ("roads", "buildings", "landuse") if a.buildings else ("roads",))
    if r is None: sys.exit(1)
    data, seg, grid = r
    n_by = np.bincount(np.clip(seg["cls"].astype(int), 0, 6), weights=np.hypot(seg["x1"] - seg["x0"], seg["y1"] - seg["y0"]), minlength=7)
    print(f"{len(data['roads'])} roads, {len(seg['x0'])} segments, {len(data['buildings'])} buildings, "
          f"{len(data['landuse'])} residential areas; grid {grid['dist'].shape[1]}x{grid['dist'].shape[0]} at {grid['res']:g} m; "
          f"{time.time() - t0:.1f}s")
    print("km of road by class: " + ", ".join(f"{n} {v / 1000:.0f}" for n, v in zip(CLASSES, n_by)))
    if a.debug:
        cam = local_xy(np.array([list(map(float, a.cam.split(",")))]), origin)[0] if a.cam else None
        view = tuple(map(float, a.view.split(","))) if a.view else None
        print(debug_image(data, seg, grid, a.debug, origin, cam, view))
