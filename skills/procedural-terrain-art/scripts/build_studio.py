#!/usr/bin/env python3
"""
build_studio.py — build the interactive, self-contained web studio for any mountain.

  python build_studio.py --peak 39.7019,44.2986 --from 40.1792,44.4991 --from-name Yerevan \
      --name "Masis & Sis" --native "ԱՐԱՐԱՏ" --native-h1 "Մասիս · Սիս" \
      --label "ՄԱՍԻՍ|5137 m" --label "ՍԻՍ|3896 m@39.6517,44.4011" --out ararat-studio.html

The page (~0.8 MB) embeds a 512x512 elevation grid, renders live on Canvas 2D (no libraries), has
six styles, orbit animation, light/camera/colour controls, and a 3000x1000 PNG export.
Publish it as an artifact (declare the `downloads` capability so Save PNG works) or open it locally.
"""
import argparse, base64, html as htmlmod, json, math, os, re, sys
import numpy as np
from scipy.ndimage import map_coordinates
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from terrain_art import ensure_dem, load_dem, terrain, water_mask

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "..", "assets", "studio_template.html")

# Google Fonts families per script (label font, big-title font)
SCRIPT_FONTS = [
    (r"[\u0530-\u058f]", "Noto Serif Armenian", "Noto Sans Armenian"),
    (r"[\u10a0-\u10ff]", "Noto Serif Georgian", "Noto Sans Georgian"),
    (r"[\u3040-\u30ff]", "Noto Serif JP", "Noto Sans JP"),
    (r"[\uac00-\ud7af]", "Noto Serif KR", "Noto Sans KR"),
    (r"[\u4e00-\u9fff]", "HAN", "HAN"),                         # Han only: resolved by --lang or location
    (r"[\u0900-\u097f]", "Noto Serif Devanagari", "Noto Sans Devanagari"),
    (r"[\u0600-\u06ff]", "Noto Naskh Arabic", "Noto Sans Arabic"),
    (r"[\u0590-\u05ff]", "Noto Serif Hebrew", "Noto Sans Hebrew"),
    (r"[\u0e00-\u0e7f]", "Noto Serif Thai", "Noto Sans Thai"),
    (r"[\u0f00-\u0fff]", "Noto Serif Tibetan", "Noto Sans Tibetan"),
]


HAN_FONTS = {"ja": ("Noto Serif JP", "Noto Sans JP"), "ko": ("Noto Serif KR", "Noto Sans KR"),
             "zh-Hans": ("Noto Serif SC", "Noto Sans SC"), "zh-Hant": ("Noto Serif TC", "Noto Sans TC")}


def han_lang(lat, lon):
    """Kanji / hanja / hanzi share code points; pick the regional glyph forms from where the summit is."""
    if 24 <= lat <= 46 and 122.5 <= lon <= 154: return "ja"
    if 33 <= lat <= 39 and 124 <= lon <= 131: return "ko"
    if 21.8 <= lat <= 25.4 and 119.3 <= lon <= 122.1: return "zh-Hant"
    return "zh-Hans"


def pick_fonts(text, lang=None, latlon=(0.0, 0.0)):
    for pat, serif, sans in SCRIPT_FONTS:
        if re.search(pat, text):
            if serif == "HAN": return HAN_FONTS[lang or han_lang(*latlon)]
            return serif, sans
    return "Noto Serif", "Noto Sans"


def parse_ll(s):
    a, b = s.split(","); return float(a), float(b)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--peak", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--from", dest="view_from", help="lat,lon of the viewpoint (e.g. a town)")
    g.add_argument("--facing", type=float, help="compass bearing the camera looks toward (0 = north), if there is no obvious viewpoint")
    ap.add_argument("--from-name", default="the viewpoint")
    ap.add_argument("--name", required=True, help="page title / heading, Latin script")
    ap.add_argument("--native", default="", help="name in the local script (used as big title in riso/woodcut)")
    ap.add_argument("--native-h1", default=None, help="small subtitle next to the heading (default: --native)")
    ap.add_argument("--label", action="append", default=[], help='"Name|sub" or "Name|sub@lat,lon"')
    ap.add_argument("--extent", type=float); ap.add_argument("--base", type=float)
    ap.add_argument("--dem", help="cached .npz (default: dem_<lat>_<lon>.npz next to --out, the same file render.py uses)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", choices=sorted(HAN_FONTS), help="glyph forms for Han-only labels (default: from the summit's location)")
    ap.add_argument("--no-shape", action="store_true", help="cone framing rules for every landform (ignore the landform analysis)")
    ap.add_argument("--no-water", action="store_true", help="don't treat sea and lakes as a separate water layer")
    a = ap.parse_args()

    summit = parse_ll(a.peak); vf = parse_ll(a.view_from) if a.view_from else None
    dem_path = a.dem or os.path.join(os.path.dirname(os.path.abspath(a.out)), f"dem_{summit[0]:.3f}_{summit[1]:.3f}.npz")
    ensure_dem(*summit, (a.extent or 45) * 1.45, dem_path, explicit=bool(a.dem), log=lambda m: print(m, file=sys.stderr, flush=True))
    dem = load_dem(dem_path)
    t = terrain(dem, summit, view_from=vf, facing=a.facing, extent_km=a.extent, base_m=a.base, NX=64, NZ=32, shape_aware=not a.no_shape)
    la, lo = t.summit
    kx, ky = 111.32 * math.cos(math.radians(la)), 110.57
    R = t.extent_km * 1.36
    elev, lats, lons = dem
    N = 512
    e = np.linspace(-R, R, N); n = np.linspace(R, -R, N)
    EE, NN = np.meshgrid(e, n)
    ii = np.interp(la + NN / ky, lats[::-1], np.arange(len(lats))[::-1].astype(float))
    jj = (lo + EE / kx - lons[0]) / (lons[1] - lons[0])
    G = map_coordinates(elev, [ii, jj], order=1, mode="nearest").clip(0, 9000).astype("<u2")
    wm = None if a.no_water else water_mask(dem)
    if wm is not None:                          # bit 15 flags open water (elevations need only 14 bits)
        G[map_coordinates(wm.astype(np.float32), [ii, jj], order=1, mode="nearest") > 0.5] |= 0x8000

    peaks = []
    for L in a.label:
        name, _, rest = L.partition("|"); sub, _, ll = rest.partition("@")
        p = parse_ll(ll) if ll else (la, lo)
        peaks.append(dict(name=name, sub=sub, E=(p[1] - lo) * kx, N=(p[0] - la) * ky))
    if not peaks or abs(peaks[0]["E"]) + abs(peaks[0]["N"]) > 0.01:
        main_ = [p for p in peaks if abs(p["E"]) + abs(p["N"]) <= 0.01]
        others = [p for p in peaks if p not in main_]
        peaks = (main_ or [dict(name="", sub="", E=0.0, N=0.0)]) + others
    native = a.native or a.name
    alltext = native + (a.native_h1 or "") + "".join(p["name"] + p["sub"] for p in peaks)
    serif, sans = pick_fonts(alltext, a.lang, (la, lo))
    if vf is not None:
        view = [(vf[1] - lo) * kx, (vf[0] - la) * ky]
    else:                                        # --facing: a virtual viewpoint far behind the camera
        b = math.radians(a.facing); view = [-math.sin(b) * 100.0, -math.cos(b) * 100.0]
    fam = lambda f, w: "&family=" + f.replace(" ", "+") + f":wght@{w}"
    fonts = fam(serif, "400;500") + fam(sans, "500")
    sh = t.shape
    horn, canyon, shield = sh.get("horn", 0), sh.get("canyon", 0), sh.get("shield", 0)
    auto = dict(kind=sh.get("kind", "legacy"), relief=sh.get("relief", 1.0), spacing=sh.get("spacing", 1.0),
                camh=sh.get("camh"), topoCamh=round(24 - 10 * horn, 2) if horn > 0.05 else None,
                snowline=round(sh["snowline_m"]) if sh.get("snowline_m") else None, fit=bool(horn >= 0.05 or canyon >= 0.05),
                fadePow=round(1 + min(1.0, horn + canyon + shield), 2),
                crowd=round(max(horn, sh.get("rough_score", 0)), 2), canyon=round(canyon, 2), shield=round(shield, 2))
    cfg = dict(DN=N, R=R, k=t.k, hs=t.hs, base_km=t.base_m / 1000, peak_km=t.peak_m / 1000, v0=float(t.vs[0]), auto=auto,
               view=view, peaks=peaks, name=a.name, native=native,
               slug=re.sub(r"[^a-z0-9]+", "-", a.name.lower()).strip("-") or "mountain",
               labelFont=f'"{serif}", Georgia, serif', titleFont=f'"{sans}", sans-serif',
               footnote=f"{abs(la):.2f}°{'N' if la >= 0 else 'S'}  {abs(lo):.2f}°{'E' if lo >= 0 else 'W'}  ·  elevation data, viewed from {a.from_name}",
               risoSub=f"{a.name.upper()}  ·  " + "  ·  ".join(p["sub"] for p in peaks if p["sub"]))
    html = open(TEMPLATE, encoding="utf-8").read()
    html = re.sub(r"/\* @@ENGINE-(BEGIN|END)\b.*?\*/\n", "", html, flags=re.S)   # engine markers, used by the web app build
    esc = htmlmod.escape                          # user text goes into HTML text and attributes
    subs = {"__CONFIG__": json.dumps(cfg, ensure_ascii=False).replace("</", "<\\/"),   # no </script> inside the JSON
            "__DEM__": base64.b64encode(G.tobytes()).decode(),
            "__PAGETITLE__": esc(f"{a.name} — a procedural terrain studio"), "__FONTS__": fonts,
            "__NATIVEFONT_CSS__": f'"{serif}"', "__H1__": esc(a.name),
            "__NATIVE_H1__": esc(a.native_h1 if a.native_h1 is not None else a.native),
            "__LEDE__": esc(f"{a.name} as seen from {a.from_name},"),
            "__RES__": str(int(round(2 * R / (N - 1) * 1000 / 10) * 10)), "__FROM__": esc(a.from_name), "__NAME__": esc(a.name)}
    for k, v in subs.items(): html = html.replace(k, v)
    left = [k for k in subs if k in html]
    if left: sys.exit(f"template placeholders not filled: {left}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    open(a.out, "w", encoding="utf-8").write(html)
    print(json.dumps(dict(out=a.out, bytes=len(html), peak_m=round(t.peak_m), base_m=round(t.base_m), extent_km=round(t.extent_km, 1), auto=auto)))


if __name__ == "__main__":
    main()
