"""dem.py: real elevation data for the procedural-art skills.

procedural-terrain-art and procedural-realism each ship an identical copy of this file, so either skill
installs on its own. Keep the two copies the same.

Source: AWS Terrain Tiles, terrarium encoding (SRTM and other public sources, global, no key).
Each pixel decodes as R*256 + G + B/256 - 32768 metres.
"""
import io, math, os, sys, time, urllib.request, concurrent.futures as cf
import numpy as np
from PIL import Image

TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"


def _tile_xy(lat, lon, z):
    n = 2 ** z
    return (lon + 180) / 360 * n, (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n


def fetch_dem(lat, lon, radius_km, out_path, zoom_level=None, max_tiles=260, retries=3):
    """Download a square of elevation around (lat, lon) and save it as .npz (elev m, lats, lons, zoom, radius_km).
    The zoom level follows the radius (z13 up to 15 km, z12 up to 70 km, z11 up to 160 km) and drops
    until the square fits in max_tiles tiles. Rows are Web Mercator (non-linear in latitude).
    The file is written to exactly `out_path` (no .npz is appended), via a temporary file."""
    if zoom_level is None:
        zoom_level = 13 if radius_km <= 15 else 12 if radius_km <= 70 else 11 if radius_km <= 160 else 10
    dlat = radius_km / 110.57; dlon = radius_km / (111.32 * math.cos(math.radians(lat)))
    while True:
        x0, y0 = _tile_xy(lat + dlat, lon - dlon, zoom_level); x1, y1 = _tile_xy(lat - dlat, lon + dlon, zoom_level)
        tx, ty = range(math.floor(x0), math.floor(x1) + 1), range(int(y0), int(y1) + 1)
        if len(tx) * len(ty) <= max_tiles or zoom_level <= 8: break
        zoom_level -= 1
    n = 2 ** zoom_level

    def get(a):
        x, y = a
        url = TILE_URL.format(z=zoom_level, x=x % n, y=y)          # x % n wraps across the antimeridian
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(url, timeout=30) as r:
                    im = np.asarray(Image.open(io.BytesIO(r.read())).convert("RGB")).astype(float)
                return a, im[..., 0] * 256 + im[..., 1] + im[..., 2] / 256 - 32768
            except Exception as e:                                   # network hiccup: wait and try again
                if attempt == retries - 1:
                    raise RuntimeError(f"elevation tile {url} failed after {retries} tries: {e}") from e
                time.sleep(1.5 * (attempt + 1))

    M = np.zeros((len(ty) * 256, len(tx) * 256), np.float32)
    with cf.ThreadPoolExecutor(16) as ex:
        for (x, y), e in ex.map(get, [(x, y) for x in tx for y in ty]):
            M[(y - ty[0]) * 256:(y - ty[0] + 1) * 256, (x - tx[0]) * 256:(x - tx[0] + 1) * 256] = e
    lons = (np.arange(M.shape[1]) / 256 + tx[0]) / n * 360 - 180
    lats = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (np.arange(M.shape[0]) / 256 + ty[0]) / n))))
    d = os.path.dirname(os.path.abspath(out_path)); os.makedirs(d, exist_ok=True)
    tmp = out_path + ".part"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, elev=M, lats=lats, lons=lons, zoom=zoom_level, radius_km=float(radius_km),
                            center=np.array([lat, lon]))
    os.replace(tmp, out_path)
    return out_path


def dem_coverage_km(path, lat, lon):
    """Distance (km) from (lat, lon) to the nearest edge of a cached DEM; 0 if the point is outside it."""
    d = np.load(path)
    la, lo = d["lats"], d["lons"]
    kx, ky = 111.32 * math.cos(math.radians(lat)), 110.57
    return max(0.0, min((la[0] - lat) * ky, (lat - la[-1]) * ky, (lon - lo[0]) * kx, (lo[-1] - lon) * kx))


def ensure_dem(lat, lon, radius_km, path, explicit=False, log=print):
    """Fetch `path` unless it already covers `radius_km` around (lat, lon).
    A cache that is too small is refetched (and overwritten); an explicit --dem file is only warned about."""
    if os.path.exists(path):
        cov = dem_coverage_km(path, lat, lon)
        if cov >= radius_km * 0.98:
            return path
        if explicit:
            print(f"warning: {path} covers only {cov:.0f} km around the peak, {radius_km:.0f} km are needed; "
                  "framing and landform detection may differ from a full fetch", file=sys.stderr, flush=True)
            return path
        log(f"cached {os.path.basename(path)} covers {cov:.0f} km, {radius_km:.0f} km needed: fetching a larger one")
    log(f"fetching elevation, radius {radius_km:.0f} km …")
    return fetch_dem(lat, lon, radius_km, path)


def load_dem(path):
    """Load a cached DEM -> (elev float32, lats, lons). Tiles occasionally carry void values (±32768);
    those pixels are filled from their nearest valid neighbour so they can't turn into spikes or pits."""
    d = np.load(path)
    elev = d["elev"].astype(np.float32)
    bad = ~np.isfinite(elev) | (elev < -12000) | (elev > 9000)
    if bad.any() and not bad.all():
        from scipy.ndimage import distance_transform_edt
        elev = elev[tuple(distance_transform_edt(bad, return_distances=False, return_indices=True))]
    return elev, d["lats"], d["lons"]
