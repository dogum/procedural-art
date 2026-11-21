#!/usr/bin/env python3
"""Bake real elevation into the block a single-file canvas landscape embeds.

    python bake_dem.py --npz work/fuji/dem.npz --camera 35.5226,138.7672 --look 35.3606,138.7274 \
        --lake 830 --date 2026-11-10 --time 07:05 --tz 9 --name "Mount Fuji from Lake Kawaguchi" --out block.js

The .npz is the elevation cache the terrain skills write (`dem.py`: arrays `elev` [metres, rows
north to south], `lats`, `lons`). Any DEM with those three arrays works. numpy only.

What is baked is not a square map but the ground as this one camera sees it, a fan of depth
profiles, so every sample is spent where the picture needs it:

    column i   a direction: u = x / z = U0 + i * (U1 - U0) / (NU - 1)      (x right, z forward, metres)
    row j      a distance z(j) along that direction, in three runs:
                 near   z = Z0 + A * j                       while z < Z1      (A metres a row)
                 middle z = Z1 * exp(K * (j - J1))           while z < Z2      (rows K apart in log z)
                 far    z = Z2 + B * (j - J2)                                   (B metres a row)
    sample     ground height in metres above the lake (or above --datum), with the drop of the
               curved Earth (and standard refraction) already taken off, so the page can treat the
               world as flat: h -= 0.87 * d^2 / (2 R)

Stored as two base64 strings so 16-bit accuracy costs 8 bits a sample:
    c16   every STEP-th sample in both directions (rows and columns 0, STEP, 2*STEP, ...), row by row,
          unsigned 16-bit little-endian, metres = HMIN + value * CQ; NU - 1 and NV - 1 are multiples of STEP
    r8    every sample, row by row (row = one distance), one byte: what is left after the coarse grid is interpolated bilinearly,
          metres = (value - 128) * RQ

The block also carries the camera (latitude, longitude, heading, eye height), the lake level and
the Sun's azimuth and altitude at the stated date and time (NOAA formulas, about 0.1 degree).
The output is one JavaScript statement, `const DEM = {...};`, to paste between the markers
`/*DEM-BEGIN*/` and `/*DEM-END*/` of the page (see --into).
"""
import argparse
import base64
import datetime as dt
import json
import math
import re

import numpy as np

R_EARTH = 6371000.0


def sun_position(lat, lon, when_utc):
    """Azimuth (degrees clockwise from north) and altitude (degrees) of the Sun. NOAA's formulas."""
    jd = when_utc.timestamp() / 86400.0 + 2440587.5
    t = (jd - 2451545.0) / 36525.0
    l0 = (280.46646 + t * (36000.76983 + 0.0003032 * t)) % 360
    m = math.radians(357.52911 + t * (35999.05029 - 0.0001537 * t))
    e = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)
    c = (math.sin(m) * (1.914602 - t * (0.004817 + 0.000014 * t)) + math.sin(2 * m) * (0.019993 - 0.000101 * t)
         + math.sin(3 * m) * 0.000289)
    lam = math.radians(l0 + c - 0.00569 - 0.00478 * math.sin(math.radians(125.04 - 1934.136 * t)))
    eps0 = 23 + (26 + (21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))) / 60) / 60
    eps = math.radians(eps0 + 0.00256 * math.cos(math.radians(125.04 - 1934.136 * t)))
    dec = math.asin(math.sin(eps) * math.sin(lam))
    y = math.tan(eps / 2) ** 2
    l0r = math.radians(l0)
    eot = 4 * math.degrees(y * math.sin(2 * l0r) - 2 * e * math.sin(m) + 4 * e * y * math.sin(m) * math.cos(2 * l0r)
                           - 0.5 * y * y * math.sin(4 * l0r) - 1.25 * e * e * math.sin(2 * m))
    minutes = when_utc.hour * 60 + when_utc.minute + when_utc.second / 60 + eot + 4 * lon
    ha = math.radians(minutes / 4 - 180)
    la = math.radians(lat)
    alt = math.asin(math.sin(la) * math.sin(dec) + math.cos(la) * math.cos(dec) * math.cos(ha))
    az = math.atan2(-math.sin(ha), math.tan(dec) * math.cos(la) - math.sin(la) * math.cos(ha))
    alt_d = math.degrees(alt)
    alt_d += 1.02 / math.tan(math.radians(alt_d + 10.3 / (alt_d + 5.11))) / 60 if alt_d > -1 else 0   # refraction
    return math.degrees(az) % 360, alt_d


def rows(z0, a, z1, k, z2, b, zfar):
    """The distances of the rows, and the row numbers where the runs change."""
    z = [z0]
    while z[-1] + a < z1:
        z.append(z[-1] + a)
    j1 = len(z)
    z.append(z1)
    while z[-1] * math.exp(k) < z2:
        z.append(z[-1] * math.exp(k))
    j2 = len(z)
    z.append(z2)
    while z[-1] + b <= zfar:
        z.append(z[-1] + b)
    return np.array(z), j1, j2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--npz', required=True, help='elevation cache with elev, lats, lons')
    ap.add_argument('--camera', required=True, help='lat,lon of the viewpoint')
    ap.add_argument('--look', required=True, help='lat,lon the camera points at (the peak)')
    ap.add_argument('--out', default='block.js')
    ap.add_argument('--into', default=None, help='an HTML page: replace what is between /*DEM-BEGIN*/ and /*DEM-END*/')
    ap.add_argument('--name', default='')
    ap.add_argument('--eye', type=float, default=2.2, help='eye height above the datum, metres')
    ap.add_argument('--lake', type=float, default=None, help='lake surface, metres above sea level (the datum)')
    ap.add_argument('--datum', type=float, default=None, help='datum when there is no lake (default: ground at the camera)')
    ap.add_argument('--halfwidth', type=float, default=0.53, help='largest |x/z| baked (0.53 covers a 1920 px frame at focal 2000 with margin)')
    ap.add_argument('--cols', type=int, default=257)
    ap.add_argument('--near', type=float, default=400.0, help='Z0, metres')
    ap.add_argument('--far', type=float, default=23000.0)
    ap.add_argument('--steps', default='20,1667,0.012,8333,100', help='A,Z1,K,Z2,B of the three runs of rows')
    ap.add_argument('--coarse', type=int, default=4, help='STEP of the 16-bit grid')
    ap.add_argument('--despike', type=float, default=30.0, help='a sample that stands this many metres above (or below) all eight samples of a ring 2, 3 or 5 steps around it is a spike: it gets the ring\'s median')
    ap.add_argument('--date', default='2026-11-10')
    ap.add_argument('--time', default='07:05', help='local clock time')
    ap.add_argument('--tz', type=float, default=9.0, help='hours ahead of UTC')
    a = ap.parse_args()

    d = np.load(a.npz)
    elev, lats, lons = d['elev'].astype(np.float64), d['lats'], d['lons']
    clat, clon = (float(v) for v in a.camera.split(','))
    plat, plon = (float(v) for v in a.look.split(','))
    # work on the part of the map the fan can reach, and take out spikes and voids there (a tower in a
    # surface model, a missing value): a patch a few samples wide that stands clear of everything round it.
    # A ridge or a summit is not touched, because some of the ring lies along it at its own height.
    reach = a.far * 1.2 / (R_EARTH * math.pi / 180)
    ii = np.where(np.abs(lats - clat) < reach)[0]
    jj = np.where(np.abs(lons - clon) < reach / math.cos(math.radians(clat)))[0]
    elev, lats, lons = elev[ii[0]:ii[-1] + 1, jj[0]:jj[-1] + 1].copy(), lats[ii[0]:ii[-1] + 1], lons[jj[0]:jj[-1] + 1]
    spikes = 0
    for _ in range(4):
        for rad in (2, 3, 5):
            P = np.pad(elev, rad, mode='edge')
            ring = np.stack([P[rad + di:rad + di + elev.shape[0], rad + dj:rad + dj + elev.shape[1]]
                             for di in (-rad, 0, rad) for dj in (-rad, 0, rad) if di or dj])
            thr = a.despike * (1 if rad < 5 else 1.6)
            bad = (elev > ring.max(0) + thr) | (elev < ring.min(0) - thr)
            spikes += int(bad.sum())
            elev[bad] = np.median(ring, axis=0)[bad]
    mlat = R_EARTH * math.pi / 180
    mlon = mlat * math.cos(math.radians(clat))
    east, north = (plon - clon) * mlon, (plat - clat) * mlat
    heading = math.degrees(math.atan2(east, north)) % 360
    fx, fz = math.sin(math.radians(heading)), math.cos(math.radians(heading))      # forward, as (east, north)
    rx, rz = fz, -fx                                                               # right

    def ground(e, n):
        """Bilinear elevation at metres east and north of the camera."""
        fi = np.interp(clat + n / mlat, lats[::-1], np.arange(len(lats))[::-1])
        fj = np.interp(clon + e / mlon, lons, np.arange(len(lons)))
        i0 = np.clip(np.floor(fi).astype(int), 0, len(lats) - 2)
        j0 = np.clip(np.floor(fj).astype(int), 0, len(lons) - 2)
        ti, tj = fi - i0, fj - j0
        return ((elev[i0, j0] * (1 - tj) + elev[i0, j0 + 1] * tj) * (1 - ti)
                + (elev[i0 + 1, j0] * (1 - tj) + elev[i0 + 1, j0 + 1] * tj) * ti)

    A, Z1, K, Z2, B = (float(v) for v in a.steps.split(','))
    z, j1, j2 = rows(a.near, A, Z1, K, Z2, B, a.far)
    u = np.linspace(-a.halfwidth, a.halfwidth, a.cols)
    Z, U = np.meshgrid(z, u, indexing='ij')
    X = U * Z
    cam_ground = float(ground(np.array(0.0), np.array(0.0)))
    datum = a.lake if a.lake is not None else (a.datum if a.datum is not None else cam_ground)
    dist2 = X * X + Z * Z
    H = ground(X * rx + Z * fx, X * rz + Z * fz) - datum - 0.87 * dist2 / (2 * R_EARTH)

    if a.lake is not None:                                    # nothing in view lies below the lake: such samples are voids
        H = np.maximum(H, -0.87 * dist2 / (2 * R_EARTH))
    st = a.coarse
    if (a.cols - 1) % st:
        raise SystemExit(f"--cols must be a multiple of {st} plus one (e.g. {a.cols // st * st + 1})")
    nv = (H.shape[0] - 1) // st * st + 1                      # so the coarse grid ends on the last row
    H, z, dist2 = H[:nv], z[:nv], dist2[:nv]
    nu = H.shape[1]
    hmin = float(np.floor(H.min()))
    cq = max((float(H.max()) - hmin) / 65535.0, 1e-6)
    c16 = np.rint((H[::st, ::st] - hmin) / cq).astype('<u2')
    C = hmin + c16.astype(np.float64) * cq
    # bilinear interpolation of the coarse grid, exactly as the page does it
    i0 = np.minimum(np.arange(nv) // st, C.shape[0] - 2)
    j0 = np.minimum(np.arange(nu) // st, C.shape[1] - 2)
    ti, tj = (np.arange(nv) / st - i0)[:, None], (np.arange(nu) / st - j0)[None, :]
    up = ((C[np.ix_(i0, j0)] * (1 - tj) + C[np.ix_(i0, j0 + 1)] * tj) * (1 - ti)
          + (C[np.ix_(i0 + 1, j0)] * (1 - tj) + C[np.ix_(i0 + 1, j0 + 1)] * tj) * ti)
    res = H - up
    rq = max(float(np.abs(res).max()) / 127.0, 1e-6)
    r8 = np.clip(np.rint(res / rq) + 128, 0, 255).astype(np.uint8)
    err = np.abs(up + (r8.astype(np.float64) - 128) * rq - H)

    hh, mm = (int(v) for v in a.time.split(':'))
    local = dt.datetime.fromisoformat(a.date).replace(hour=hh, minute=mm, tzinfo=dt.timezone(dt.timedelta(hours=a.tz)))
    saz, salt = sun_position(clat, clon, local.astimezone(dt.timezone.utc))
    block = dict(
        name=a.name, lat=round(clat, 5), lon=round(clon, 5), lookLat=round(plat, 5), lookLon=round(plon, 5),
        heading=round(heading, 2), eye=a.eye, datum=round(datum, 1), lake=a.lake is not None,
        date=a.date, time=a.time, tz=a.tz, sunAz=round(saz, 2), sunAlt=round(salt, 2),
        nu=nu, nv=nv, u0=-a.halfwidth, u1=a.halfwidth, z0=a.near, a=A, z1=Z1, k=K, z2=Z2, b=B, j1=j1, j2=j2,
        step=st, cn=[int(C.shape[0]), int(C.shape[1])],
        hmin=hmin, cq=cq, rq=rq,
        c16=base64.b64encode(c16.tobytes()).decode(), r8=base64.b64encode(r8.tobytes()).decode())
    js = 'const DEM = ' + json.dumps(block, separators=(',', ':')) + ';'
    with open(a.out, 'w') as f:
        f.write(js + '\n')
    if a.into:
        page = open(a.into).read()
        new, n = re.subn(r'/\*DEM-BEGIN\*/.*?/\*DEM-END\*/', lambda m_: '/*DEM-BEGIN*/\n' + js + '\n/*DEM-END*/', page, flags=re.S)
        if n != 1:
            raise SystemExit(f"{a.into}: expected one /*DEM-BEGIN*/ ... /*DEM-END*/ pair, found {n}")
        open(a.into, 'w').write(new)
    peak = np.unravel_index(H.argmax(), H.shape)
    print(f"{nu} directions x {nv} distances ({a.near:.0f} m to {z[-1]:.0f} m), {len(js) / 1000:.0f} kB; "
          f"heading {heading:.1f} deg; ground at the camera {cam_ground:.0f} m, datum {datum:.0f} m; "
          f"highest sample {H.max() + datum + 0.87 * dist2[peak] / (2 * R_EARTH):.0f} m at {math.sqrt(dist2[peak]) / 1000:.1f} km; "
          f"{spikes} spike samples replaced; largest error {err.max():.2f} m (coarse step {cq:.3f} m, residual step {rq:.3f} m); "
          f"Sun {a.date} {a.time} (UTC{a.tz:+g}): azimuth {saz:.1f}, altitude {salt:.1f}; wrote {a.out}"
          + (f" and {a.into}" if a.into else ''))


if __name__ == '__main__':
    main()
