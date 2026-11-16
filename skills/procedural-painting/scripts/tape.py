"""Stroke tapes: everything a painting did, small enough to ship, so another program can lay
the same strokes again. The browser player in docs/paint replays these live.

    p = Painting(W, H, ..., tape=True)        # records every pass as it is laid
    ...paint...
    save_tape(p, 'rain.tape', title='Rain at blue hour')

The studies take --tape FILE. A tape is independent of resolution: lengths are stored as
fractions of the picture height, so a player may lay it on a canvas of any size.

File format (version 1). The whole file is one zlib stream (browser: DecompressionStream
('deflate')). Inside, little-endian:

    bytes 0..3   b'ATL1'
    bytes 4..7   uint32 n, the length of the header
    n bytes      header, UTF-8 JSON
    the rest     the data blob; header offsets count from its start

Header: {version, title, aspect (W/H), height (pixels it was recorded at), ground [r,g,b in
0..1], weave, strokes (total), ops: [...]}. Ops, in the order they happened:

    {"op": "strokes", "n", "step", "params", "sittings", "off", "points", "mask"}
        n         strokes in this pass, already in the order they are laid
        step      distance between path points, in picture heights
        params    load, opacity, pickup, dry, marble, clip, thickness, taper, soft, knife, shape
        sittings  [count, ...]: the pass is laid in these successive groups; inside a group a
                  later stroke covers an earlier one, and all of them pick up the paint that was
                  on the canvas before the group started
        mask      null, or {"w", "h", "off"}: h*w bytes, the outline strokes are cut at (0..255)
        off       start of this pass's arrays in the blob (always even), in this order:
                    int16  x0[n], y0[n]      start of each path, 1/8192 of the height
                    uint16 width[n]          1/262144 of the height
                    uint16 length[n]         1/16384 of the height
                    uint16 count[n]          points on the path, including the start
                    uint8  c1[3n], c2[3n]    the two colours on the brush, sRGB
                    uint8  runout[n]         1/255
                    uint8  offs[4n]          bristle texture offsets, 1/256
                    int8   delta[2*points]   path steps (dx, dy), 1/100 of `step`; points is the
                                             sum of (count - 1)
    {"op": "glaze", "w", "h", "off"}   h*w bytes of strength (0..255), then h*w*3 bytes of sRGB
    {"op": "dry", "amount"}
"""
import json
import struct
import zlib

import numpy as np


def _pad(blob):
    if len(blob) % 2:
        blob.append(0)


def save_tape(p, path, title=None):
    """Write the tape of a Painting made with tape=True. Returns the file size in bytes."""
    if p.tape is None:
        raise RuntimeError("construct the Painting with tape=True to record a tape")
    blob = bytearray()
    ops, total = [], 0
    for rec in p.tape:
        if rec['op'] == 'dry':
            ops.append(dict(op='dry', amount=rec['amount']))
            continue
        if rec['op'] == 'glaze':
            _pad(blob)
            m, c = rec['mask'], rec['color']
            ops.append(dict(op='glaze', w=int(m.shape[1]), h=int(m.shape[0]), off=len(blob)))
            blob += m.tobytes() + c.tobytes()
            continue
        H = float(rec['H'])
        order = np.concatenate(rec['sittings'])
        pos, step = rec['pos'][order].astype(np.float64), rec['step']
        wid, length = rec['width'][order], rec['length'][order]
        n, K = pos.shape[:2]
        count = np.minimum(np.ceil(length / step).astype(int) + 2, K)
        x0 = np.clip(np.rint(pos[:, 0, 0] / H * 8192), -32768, 32767).astype('<i2')
        y0 = np.clip(np.rint(pos[:, 0, 1] / H * 8192), -32768, 32767).astype('<i2')
        # steps are quantised against the path as the reader will rebuild it, so errors don't add up
        cur = np.stack([x0, y0], -1) * (H / 8192.0)
        deltas = np.zeros((n, K - 1, 2), np.int8)
        for k in range(1, K):
            d = np.clip(np.rint((pos[:, k] - cur) / step * 100), -127, 127)
            deltas[:, k - 1] = d
            cur = cur + d * (step / 100.0)
        keep = np.arange(K - 1)[None, :] < (count - 1)[:, None]
        _pad(blob)
        mask = None
        off = len(blob)
        blob += x0.tobytes() + y0.tobytes()
        blob += np.clip(np.rint(wid / H * 262144), 1, 65535).astype('<u2').tobytes()
        blob += np.clip(np.rint(length / H * 16384), 1, 65535).astype('<u2').tobytes()
        blob += count.astype('<u2').tobytes()
        blob += (np.clip(rec['c1'][order], 0, 1) * 255 + 0.5).astype(np.uint8).tobytes()
        blob += (np.clip(rec['c2'][order], 0, 1) * 255 + 0.5).astype(np.uint8).tobytes()
        blob += (np.clip(rec['runout'][order], 0, 1) * 255 + 0.5).astype(np.uint8).tobytes()
        blob += (rec['offs'][order] * 256).astype(np.uint8).tobytes()
        blob += deltas[keep].tobytes()
        if rec['clip_mask'] is not None:
            cm = rec['clip_mask']
            mask = dict(w=int(cm.shape[1]), h=int(cm.shape[0]), off=len(blob))
            blob += cm.tobytes()
        prm = {k: (v if isinstance(v, (str, bool)) else float(v)) for k, v in rec['params'].items()}
        ops.append(dict(op='strokes', n=int(n), step=step / H, params=prm, sittings=[int(len(g)) for g in rec['sittings']],
                        off=off, points=int(keep.sum()), mask=mask))
        total += int(n)
    head = json.dumps(dict(version=1, title=title or '', aspect=p.W / p.H, height=p.H,
                           ground=[float(v) for v in p._ground], weave=float(p.weave), strokes=total, ops=ops),
                      separators=(',', ':')).encode()
    data = zlib.compress(b'ATL1' + struct.pack('<I', len(head)) + head + bytes(blob), 9)
    with open(path, 'wb') as f:
        f.write(data)
    return len(data)
