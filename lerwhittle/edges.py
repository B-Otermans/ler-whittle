"""Sub-pixel edge detection by matching a simulated edge profile (model library)."""

from dataclasses import replace

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .synth import PatternParams, generate

WINDOW = 3  # rows searched on each side of an edge
MIN_LINE = 3


def model_library(pattern, sem, step=0.02):
    """Noise-free edge profile M(t), t = row - true edge, per polarity (+1 top, -1 bottom).

    One straight line imaged at many sub-pixel offsets, interleaved: M finely sampled.
    """
    clean = replace(sem, dose=np.inf, read_noise=0.0, row_noise=0.0, scan_noise=0.0, gradient=0.0, bit_depth=None)
    samples = {+1: ([], []), -1: ([], [])}
    for phi in np.arange(0.0, 1.0, step):
        y0 = 30.0 + phi
        line = PatternParams(size=(60, 8), pitch=pattern.pitch, cd=pattern.cd, y_offset=y0, ler_sigma=0.0,
                             cd_line_sigma=0.0, placement_sigma=0.0)
        col = generate(line, clean, seed=0)[0][0][:, 4].astype(float)
        rows = np.arange(col.size)
        for pol, edge in ((+1, y0 - pattern.cd / 2), (-1, y0 + pattern.cd / 2)):
            sel = np.abs(rows - edge) <= WINDOW + 2
            samples[pol][0].extend(rows[sel] - edge)
            samples[pol][1].extend(col[sel])
    out = {}
    for pol, (t, v) in samples.items():
        o = np.argsort(t)
        out[pol] = (np.asarray(t)[o], np.asarray(v)[o])
    return out


def _match(win, r0, guess, polarity, library, search=1.5, step=0.01):
    """Per column: edge position of the best least-squares fit of the profile (free gain and offset)."""
    t, m = library[polarity]
    rows = r0 + np.arange(win.shape[0])
    cand = guess + np.arange(-search, search + step / 2, step)
    mod = np.interp(rows[None, :] - cand[:, None], t, m)
    mc = mod - mod.mean(1, keepdims=True)
    pc = win - win.mean(0, keepdims=True)
    dot = mc @ pc
    score = np.where(dot > 0, dot**2 / (mc**2).sum(1, keepdims=True), 0.0)
    k = np.clip(np.argmax(score, axis=0), 1, cand.size - 2)
    cols = np.arange(win.shape[1])
    sl, sc, sr = score[k - 1, cols], score[k, cols], score[k + 1, cols]
    den = sl - 2 * sc + sr
    off = np.where(den < 0, 0.5 * (sl - sr) / np.where(den == 0, -1, den), 0.0)
    return np.where(sc > 0, cand[k] + np.clip(off, -0.5, 0.5) * step, np.nan)


def edge_traces(img, library):
    """Sub-pixel top and bottom edge per column of every line not touching the image border."""
    img = img.astype(float)
    level = 0.5 * (np.percentile(img, 5) + np.percentile(img, 95))
    prof = gaussian_filter1d(img.mean(axis=1), 0.7)
    d = np.diff(np.r_[0, (prof > level).astype(int), 0])
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1
    h = img.shape[0]
    lines = []
    for s, e in zip(starts, ends):
        if s <= WINDOW or e >= h - 1 - WINDOW or e - s + 1 < MIN_LINE:
            continue
        guesses = (s - 1 + (level - prof[s - 1]) / (prof[s] - prof[s - 1]),
                   e + (prof[e] - level) / (prof[e] - prof[e + 1]))
        edges = []
        for guess, pol in zip(guesses, (+1, -1)):
            r0 = int(np.floor(guess)) - WINDOW
            r1 = int(np.floor(guess)) + WINDOW + 1
            edges.append(np.full(img.shape[1], np.nan) if r0 < 0 or r1 >= h
                         else _match(img[r0 : r1 + 1], r0, guess, pol, library))
        lines.append(dict(top=edges[0], bottom=edges[1], center_row=0.5 * sum(guesses)))
    return lines


def detrend(y):
    x = np.arange(y.size)
    ok = np.isfinite(y)
    if ok.sum() < 10:
        return np.full_like(y, np.nan)
    return y - np.polyval(np.polyfit(x[ok], y[ok], 1), x)
