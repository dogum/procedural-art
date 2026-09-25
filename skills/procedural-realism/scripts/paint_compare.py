#!/usr/bin/env python3
"""Stack images vertically with small captions (for before/after comparisons).

    python paint_compare.py --out cmp.jpg --width 1800 \
        --item photo.png "path-traced render" --item oil.png "oil" --item wc.png "watercolor"
"""
import argparse

from PIL import Image, ImageDraw, ImageFont


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--item', nargs=2, action='append', metavar=('IMAGE', 'CAPTION'), required=True)
    ap.add_argument('--width', type=int, default=1800)
    ap.add_argument('--gap', type=int, default=6)
    ap.add_argument('--bg', default='#16140f')
    a = ap.parse_args()
    ims = []
    for path, cap in a.item:
        im = Image.open(path).convert('RGB')
        h = round(im.height * a.width / im.width)
        ims.append((im.resize((a.width, h), Image.LANCZOS), cap))
    H = sum(im.height for im, _ in ims) + a.gap * (len(ims) - 1)
    sheet = Image.new('RGB', (a.width, H), a.bg)
    d = ImageDraw.Draw(sheet)
    fs = max(14, a.width // 70)
    try:
        font = ImageFont.load_default(size=fs)
    except TypeError:
        font = ImageFont.load_default()
    y = 0
    for im, cap in ims:
        sheet.paste(im, (0, y))
        if cap:
            pad = fs // 2
            tw = d.textlength(cap, font=font)
            d.rectangle([0, y + im.height - fs - 2 * pad, tw + 2 * pad, y + im.height], fill=(20, 18, 14))
            d.text((pad, y + im.height - fs - pad - 1), cap, fill=(236, 230, 214), font=font)
        y += im.height + a.gap
    sheet.save(a.out, quality=90)


if __name__ == '__main__':
    main()
