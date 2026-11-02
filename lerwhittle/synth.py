"""Synthetic SEM images of rough line/space patterns with exact ground truth.

Defaults are calibrated to a real SEM line/space image (gray levels, edge bloom, blur,
noise spectrum, roughness spectrum, line-to-line CD and placement spread).
"""

from dataclasses import dataclass

import cv2
import numpy as np

SUPERSAMPLE = (16, 4)  # sub-samples per pixel in (y, x)


@dataclass
class PatternParams:
    size: tuple = (236, 217)  # (height, width) px
    pitch: float = 17.8
    cd: float = 7.9
    y_offset: float | None = None  # centre of the first line; random if None
    tilt: float = 0.0
    ler_sigma: float = 0.09  # roughness of an infinitely long edge, px
    ler_corr_length: float = 2.2
    ler_hurst: float = 0.33
    ler_psd: str = "palasantzas"  # one of PSD_SHAPES
    cd_line_sigma: float = 0.145
    placement_sigma: float = 0.095


@dataclass
class SEMParams:
    space_level: float = 32.0
    line_level: float = 90.0
    bloom_top: float = 190.0  # extra brightness at upward-facing edges
    bloom_bottom: float = 86.0
    bloom_side: float = 120.0  # sideways-facing facets of a rough edge
    bloom_length: float = 4.3
    psf_sigma: float = 0.49  # beam blur, before the noise
    gradient: float = 0.0
    dose: float = 25.0  # counts per pixel at gray level 100; np.inf = no shot noise
    read_noise: float = 1.1
    row_noise: float = 0.0
    scan_noise: float = 0.06  # relative gain noise, correlated along each scan line
    scan_length: float = 17.0
    noise_blur: float = 0.70  # blur after the noise: correlates the noise
    bit_depth: int | None = 8


PSD_SHAPES = ("palasantzas", "two_scale", "peak", "gaussian")


def psd_shape(f, corr_length, hurst, shape="palasantzas"):
    def pal(xi, h):
        return (1.0 + (2 * np.pi * f * xi) ** 2) ** -(h + 0.5)

    def mix(parts):
        return sum(w * c / c.sum() for w, c in parts)

    if shape == "palasantzas":
        return pal(corr_length, hurst)
    if shape == "two_scale":
        return mix([(0.65, pal(corr_length, hurst)), (0.35, pal(5 * corr_length, 0.5))])
    if shape == "peak":
        return mix([(0.75, pal(corr_length, hurst)), (0.25, np.exp(-0.5 * ((f - 0.1) / 0.01) ** 2))])
    if shape == "gaussian":
        return np.exp(-((np.pi * f * corr_length) ** 2))
    raise ValueError(f"unknown roughness shape {shape!r}")


def rough_edge(n, dx, sigma, corr_length, hurst, rng, shape="palasantzas"):
    """Gaussian edge roughness with the given spectrum; sigma is that of an infinitely long edge."""
    if sigma == 0:
        return np.zeros(n)
    m = 2 * n + int(10 * corr_length / dx)  # generate longer and crop: not periodic
    f = np.fft.rfftfreq(m, d=dx)
    psd = psd_shape(f, corr_length, hurst, shape)
    e = np.fft.irfft(np.fft.rfft(rng.standard_normal(m)) * np.sqrt(psd), m)
    weight = np.full(psd.size, 2.0)
    weight[0] = 0.0
    if m % 2 == 0:
        weight[-1] = 1.0
    e = e[:n] * (sigma / np.sqrt(np.sum(weight * psd) / m))
    return e - e.mean()


def _make_lines(p, xs, rng):
    h, w = p.size
    y0 = rng.uniform(0, p.pitch) if p.y_offset is None else p.y_offset
    centers = y0 + p.pitch * np.arange(-1, int(np.ceil(h / p.pitch)) + 2)
    dx = xs[1] - xs[0]
    tops, bots = [], []
    for c in centers:
        if c + p.cd / 2 + 2 < -0.5 or c - p.cd / 2 - 2 > h - 0.5:
            continue
        cd = p.cd
        if p.placement_sigma > 0:
            c = c + rng.normal(0, p.placement_sigma)
        if p.cd_line_sigma > 0:
            cd = cd + rng.normal(0, p.cd_line_sigma)
        r_top = rough_edge(xs.size, dx, p.ler_sigma, p.ler_corr_length, p.ler_hurst, rng, p.ler_psd)
        r_bot = rough_edge(xs.size, dx, p.ler_sigma, p.ler_corr_length, p.ler_hurst, rng, p.ler_psd)
        base = c + p.tilt * (xs - (w - 1) / 2)
        tops.append(base - cd / 2 + r_top)
        bots.append(base + cd / 2 + r_bot)
    return tops, bots


def _coverage(tops, bots, h):
    """Exact line coverage of each pixel row, per sub-column."""
    cov = np.zeros((h, tops[0].size))
    for t, b in zip(tops, bots):
        r0 = max(0, int(np.floor(t.min() + 0.5)))
        r1 = min(h, int(np.floor(b.max() + 0.5)) + 1)
        if r1 <= r0:
            continue
        r = np.arange(r0, r1)[:, None]
        cov[r0:r1] += np.clip(np.minimum(r + 0.5, b[None, :]) - np.maximum(r - 0.5, t[None, :]), 0.0, 1.0)
    return np.clip(cov, 0.0, 1.0)


def _run_distance(mask, axis, reverse):
    """Distance (in samples) into each run of True from its start; runs from the border count as far."""
    m = np.moveaxis(mask, axis, 0)
    m = m[::-1] if reverse else m
    k = np.arange(m.shape[0], dtype=np.float32)[:, None]
    last_bg = np.maximum.accumulate(np.where(m, np.float32(-1e6), k), axis=0)
    out = np.where(m, k - last_bg, 0).astype(np.float32)
    return np.moveaxis(out[::-1] if reverse else out, 0, axis)


def _bloom(tops, bots, xs, ys, sem):
    """Edge bloom on the supersampled grid, box-averaged to pixels."""
    sy, sx = SUPERSAMPLE
    canvas = np.zeros((ys.size, xs.size), bool)
    for t, b in zip(tops, bots):
        r0 = max(0, np.searchsorted(ys, t.min()) - 1)
        r1 = min(ys.size, np.searchsorted(ys, b.max()) + 1)
        if r1 > r0:
            y = ys[r0:r1, None]
            canvas[r0:r1] |= (y >= t[None, :]) & (y <= b[None, :])
    d_top = (_run_distance(canvas, 0, False)[canvas] - 0.5) / sy
    d_bot = (_run_distance(canvas, 0, True)[canvas] - 0.5) / sy
    d_side = (np.minimum(_run_distance(canvas, 1, False), _run_distance(canvas, 1, True))[canvas] - 0.5) / sx
    inv = np.float32(-1.0 / sem.bloom_length)
    bloom = np.maximum(sem.bloom_top * np.exp(d_top * inv), sem.bloom_bottom * np.exp(d_bot * inv))
    out = np.zeros(canvas.shape, np.float32)
    out[canvas] = np.maximum(bloom, sem.bloom_side * np.exp(d_side * inv))
    h, w = out.shape[0] // sy, out.shape[1] // sx
    return out.reshape(h, sy, w, sx).mean(axis=(1, 3))


def gaussian_blur(img, sigma):
    """Exact Gaussian blur in Fourier space, reflected borders."""
    h, w = img.shape
    pad = np.pad(img, ((h, h), (w, w)), mode="symmetric")
    fy = np.fft.fftfreq(pad.shape[0])[:, None]
    fx = np.fft.rfftfreq(pad.shape[1])[None, :]
    out = np.fft.irfft2(np.fft.rfft2(pad) * np.exp(-2 * np.pi**2 * sigma**2 * (fx**2 + fy**2)), s=pad.shape)
    return out[h : 2 * h, w : 2 * w].astype(img.dtype)


def image_from_signal(signal, sem, rng):
    img = signal.astype(np.float32)
    if sem.psf_sigma > 0:
        img = gaussian_blur(img, sem.psf_sigma)
    h, w = img.shape
    img = img * (1 + sem.gradient * (np.linspace(0, 1, w, dtype=np.float32) - 0.5))[None, :]
    if sem.row_noise > 0:
        img = img + rng.normal(0, sem.row_noise, (h, 1)).astype(np.float32)
    if sem.scan_noise > 0:
        g = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), sigmaX=sem.scan_length,
                             sigmaY=1e-3, borderType=cv2.BORDER_REFLECT)
        img = img * (1 + sem.scan_noise * g / max(float(g.std()), 1e-9))
    if np.isfinite(sem.dose):
        scale = sem.dose / 100.0
        img = rng.poisson(np.clip(img, 0, None) * scale).astype(np.float32) / scale
    if sem.read_noise > 0:
        img = img + rng.normal(0, sem.read_noise, img.shape).astype(np.float32)
    if sem.noise_blur > 0:
        img = gaussian_blur(img, sem.noise_blur)
    if sem.bit_depth is None:
        return img
    levels = 2**sem.bit_depth - 1
    return np.clip(np.round(img * levels / 255), 0, levels).astype(np.uint8)


def generate(pattern=None, sem=None, seed=None, n_frames=1):
    """Image the same rough geometry n_frames times with independent noise.

    Returns (frames, lines); lines[i] has the exact per-column edges 'top' and 'bottom'
    and 'complete' (the line lies fully inside the image).
    """
    pattern = pattern or PatternParams()
    sem = sem or SEMParams()
    rng = np.random.default_rng(seed)
    sy, sx = SUPERSAMPLE
    h, w = pattern.size
    xs = (np.arange(w)[:, None] + ((np.arange(sx) + 0.5) / sx - 0.5)[None, :]).ravel()
    ys = (np.arange(h)[:, None] + ((np.arange(sy) + 0.5) / sy - 0.5)[None, :]).ravel()
    tops, bots = _make_lines(pattern, xs, rng)
    coverage = _coverage(tops, bots, h).reshape(h, w, sx).mean(axis=2)
    signal = sem.space_level + (sem.line_level - sem.space_level) * coverage + _bloom(tops, bots, xs, ys, sem)
    frames = [image_from_signal(signal, sem, rng) for _ in range(n_frames)]
    lines = [dict(top=t.reshape(w, sx).mean(1), bottom=b.reshape(w, sx).mean(1),
                  complete=bool(t.min() >= -0.5 and b.max() <= h - 0.5)) for t, b in zip(tops, bots)]
    return frames, lines
