"""Pattern metrics. These are only used by the experimenter to SCORE outcomes;
the simulated blocks never see them."""
import zlib

import numpy as np


def similarity(p, ref):
    """Fraction of positions (over the union of both bodies) with the same tissue label."""
    u = (p > 0) | (ref > 0)
    if not u.any():
        return 1.0
    return float((p[u] == ref[u]).mean())


def class_iou(p, ref, classes=range(1, 7)):
    out = {}
    for c in classes:
        a, b = p == c, ref == c
        un = (a | b).sum()
        out[c] = float((a & b).sum() / un) if un else 1.0
    return out


def dilate(mask, r=2):
    out = mask.copy()
    H, W = mask.shape
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            sh = np.zeros_like(mask)
            ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
            xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
            sh[yd, xd] = mask[ys, xs]
            out |= sh
    return out


def overgrowth(p, ref, r=2):
    """Cells alive outside the (dilated) reference body outline."""
    return int(((p > 0) & ~dilate(ref > 0, r)).sum())


def blueprint_bytes(pattern):
    """Sizes of explicit blueprints of a grown body: raw per-cell raster (1 byte/cell over
    the bounding box), zlib-9 compressed raster, and the source doc's 1 KB-per-block spec."""
    ys, xs = np.nonzero(pattern)
    box = pattern[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8)
    return dict(raw=int(box.size), zlib=len(zlib.compress(box.tobytes(), 9)),
                per_block_1KB=int((pattern > 0).sum()) * 1024)


def resample(pattern, size):
    """Nearest-neighbour resample of a body's bounding box to size x size (for cross-scale comparison)."""
    ys, xs = np.nonzero(pattern)
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    h = max(ys.max() - ys.min(), xs.max() - xs.min()) / 2 + 1
    g = np.linspace(-h, h, size)
    yi = np.clip(np.round(cy + g).astype(int), 0, pattern.shape[0] - 1)
    xi = np.clip(np.round(cx + g).astype(int), 0, pattern.shape[1] - 1)
    return pattern[np.ix_(yi, xi)]


def rotate_quarter(p, c):
    """Rotate a pattern +90 deg about grid centre c (image coords, y down): new(c+dy, c+dx) = old(c-dx, c+dy)."""
    N = p.shape[0]
    yy, xx = np.mgrid[0:N, 0:N]
    dy, dx = yy - c, xx - c
    sy, sx = c - dx, c + dy
    ok = (sy >= 0) & (sy < N) & (sx >= 0) & (sx < N)
    out = np.zeros_like(p)
    out[ok] = p[sy[ok], sx[ok]]
    return out


def aligned_similarity(p, ref, max_shift=2):
    """Best similarity over small rigid translations (|dy|,|dx| <= max_shift): pattern fidelity
    up to a small offset of the whole body."""
    best = 0.0
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            best = max(best, similarity(np.roll(np.roll(p, dy, 0), dx, 1), ref))
    return best
