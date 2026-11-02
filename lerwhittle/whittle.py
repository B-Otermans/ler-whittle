"""Joint two-frame Whittle likelihood for line-edge roughness (LER).

Two frames a, b of the same edges. Per edge and Fourier bin f:

    difference (a - b) / 2:   I_d(f) ~ Exp( N_side(f) )
    average    (a + b) / 2:   I_s(f) ~ Exp( LER^2 u(f; xi, H) + N_side(f) )

u: Palasantzas roughness spectrum at unit LER (aliased, column-averaged) times the edge
transfer H^2(f) of imaging + detector. N: noise spectrum per edge side, its log level free at
frequency nodes. LER is the pixel-sampled LER; the in-band LER counts only frequencies where
H^2 >= 0.1. 90% interval: profile likelihood with every other parameter re-fitted.
"""

from dataclasses import replace

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.optimize import minimize

from .edges import detrend, edge_traces
from .synth import generate

K_MIN = 2  # lowest Fourier bins are removed by detrending
CHI2_90 = 2.706
F_MAX = 2.0  # cycles/px: Nyquist of the generator's supersampled edge
ALIAS = (-2, -1, 0, 1, 2)
H2_BAND = 0.1
NODES_K = np.array([2, 3, 4, 6, 9, 13, 18, 25, 34, 46, 60, 76, 92, 108])
LOG_PIX = (np.log(1e-3), np.log(2.0))
ROUGH_BOUNDS = [LOG_PIX, (np.log(0.3), np.log(50.0)), (0.05, 1.0)]
ETA_BOUNDS = (-30.0, 5.0)
STARTS = ((np.log(0.08), np.log(2.0), 0.5), (np.log(0.04), np.log(5.0), 0.3), (np.log(0.15), np.log(1.0), 0.8))
PROFILE_STEP = 0.025
EPS = 1e-4
_FG = np.linspace(0, F_MAX, 4001)


# ------------------------------------------------------------------ spectra

def periodogram(a, b=None):
    """One-sided (cross-)periodogram; sum(p) / n = covariance."""
    b = a if b is None else b
    n = a.size
    p = np.real(np.fft.rfft(a) * np.conj(np.fft.rfft(b))) / n
    p[1 : (n + 1) // 2] *= 2
    return p


def palasantzas(f, sigma, xi, hurst):
    def shape(x):
        return (1 + (2 * np.pi * xi * x) ** 2) ** -(hurst + 0.5)

    return np.where(f <= F_MAX, sigma**2 * shape(f) / np.trapezoid(shape(_FG), _FG), 0.0)


def rough_spectrum(f, sigma, xi, hurst):
    """Column-averaged, aliased roughness spectrum at pixel frequencies."""
    return sum(palasantzas(np.abs(f + m), sigma, xi, hurst) * np.sinc(f + m) ** 2 for m in ALIAS)


def unit_rough(f, xi, hurst, n, h2):
    """Roughness spectrum of pixel LER 1, seen through the edge transfer."""
    unit = np.sqrt(np.sum(rough_spectrum(np.fft.rfftfreq(n)[1:], 1.0, xi, hurst)) / n)
    return h2 * rough_spectrum(f, 1.0 / unit, xi, hurst)


def in_band_fraction(n, kc, xi, hurst):
    rs = rough_spectrum(np.fft.rfftfreq(n)[1:], 1.0, xi, hurst)
    return float(np.sqrt(rs[K_MIN - 1 : kc].sum() / rs.sum()))


# ------------------------------------------------------------------ data

def _pair(lines, center, tol=3.0):
    d = [abs(ln["center_row"] - center) for ln in lines]
    return lines[int(np.argmin(d))] if d and min(d) < tol else None


def paired_edges(frames, truth, library):
    """Per complete line and side: detrended edge traces of every frame and the exact edge."""
    found = [edge_traces(f, library) for f in frames]
    out = []
    for t in truth:
        if not t["complete"]:
            continue
        m = [_pair(f, float(np.mean(0.5 * (t["top"] + t["bottom"])))) for f in found]
        if any(x is None for x in m):
            continue
        for k, side in enumerate(("top", "bottom")):
            traces = [x[side] for x in m]
            if all(np.isfinite(tr).all() for tr in traces):
                out.append((k, [detrend(tr) for tr in traces], detrend(t[side])))
    return out


def edge_transfer(pattern, sem, library, seeds=range(500, 504)):
    """H^2(f) of imaging + edge detector, from noise-free images: cross(measured, true) / PSD(true)."""
    clean = replace(sem, dose=np.inf, read_noise=0.0, scan_noise=0.0, bit_depth=None)
    cross, power = 0.0, 0.0
    for seed in seeds:
        frames, truth = generate(pattern, clean, seed)
        for _, (measured,), true in paired_edges(frames, truth, library):
            cross = cross + periodogram(measured, true)
            power = power + periodogram(true)
    h = cross / power
    h[1:] = uniform_filter1d(h[1:], 5, mode="nearest")
    return np.clip(h, 1e-4, None) ** 2


def band_limit(h2):
    """Highest Fourier bin up to which the edge transfer stays >= H2_BAND."""
    ok = h2 >= H2_BAND
    ok[0] = True
    return h2.size - 1 if ok.all() else int(np.argmin(ok)) - 1


def panel_spectra(frames, truth, library, h2):
    """Average and difference periodograms of a two-frame panel, plus its exact LER."""
    edges = paired_edges(frames, truth, library)
    n = frames[0].shape[1]
    kc = band_limit(h2)
    sel = slice(K_MIN, None)
    return dict(
        n=n, kc=kc, f=np.fft.rfftfreq(n)[sel], h2=h2[sel], side=np.array([k for k, _, _ in edges]),
        Is=np.array([periodogram(0.5 * (a + b))[sel] for _, (a, b), _ in edges]),
        Id=np.array([periodogram(0.5 * (a - b))[sel] for _, (a, b), _ in edges]),
        true=float(np.sqrt(np.mean([np.mean(t**2) for _, _, t in edges]))),
        true_band=float(np.sqrt(np.mean([periodogram(t)[K_MIN : kc + 1].sum() / n for _, _, t in edges]))))


# ------------------------------------------------------------------ likelihood

def node_weights(n):
    ks = np.arange(K_MIN, n // 2 + 1)
    return np.column_stack([np.interp(ks, NODES_K, np.eye(NODES_K.size)[b]) for b in range(NODES_K.size)])


def _nll(lp, u, eta, d, w):
    """Negative log-likelihood and its gradients w.r.t. log LER, u (per bin) and the noise nodes."""
    nz = np.exp((eta.reshape(2, -1) @ w.T)[d["side"]])
    lam = np.exp(2 * lp)
    s = lam * u[None, :] + nz
    val = float(np.sum(np.log(s) + d["Is"] / s) + np.sum(np.log(nz) + d["Id"] / nz))
    gs = (s - d["Is"]) / s**2
    gnz = (gs + (nz - d["Id"]) / nz**2) * nz
    g_eta = np.stack([gnz[d["side"] == k].sum(0) @ w for k in (0, 1)]).ravel()
    return val, float(np.sum(gs * 2 * lam * u[None, :])), lam * gs.sum(0), g_eta


def _objective(theta, d, w):
    lp, lx, hu = theta[:3]
    u = unit_rough(d["f"], np.exp(lx), hu, d["n"], d["h2"])
    du_x = (unit_rough(d["f"], np.exp(lx + EPS), hu, d["n"], d["h2"]) - u) / EPS
    du_h = (unit_rough(d["f"], np.exp(lx), hu + EPS, d["n"], d["h2"]) - u) / EPS
    val, g_lp, g_u, g_eta = _nll(lp, u, theta[3:], d, w)
    return val, np.r_[g_lp, g_u @ du_x, g_u @ du_h, g_eta]


def _noise_start(d):
    ks = np.arange(K_MIN, d["n"] // 2 + 1)
    return np.ravel([np.interp(NODES_K, ks, np.log(uniform_filter1d(d["Id"][d["side"] == k].mean(0), 5,
                                                                    mode="nearest"))) for k in (0, 1)])


def _profile(obj, best_lp, best_rest, best_val, bounds):
    """90% profile-likelihood interval for log LER."""
    ends = []
    for step in (-PROFILE_STEP, PROFILE_STEP):
        lp, rest, prev = best_lp, best_rest, 0.0
        while True:
            lp_new = lp + step
            if not LOG_PIX[0] <= lp_new <= LOG_PIX[1] or abs(lp_new - best_lp) > 2.5:
                ends.append(0.0 if step < 0 else np.inf)
                break
            r = minimize(lambda x: obj(lp_new, x), rest, method="L-BFGS-B", bounds=bounds, jac=True)
            val = 2 * (r.fun - best_val)
            if val > CHI2_90:
                t = (CHI2_90 - prev) / (val - prev)
                ends.append(float(np.exp(lp + t * step)))
                break
            lp, rest, prev = lp_new, r.x, max(val, 0.0)
    return ends


def fit(d):
    """LER and in-band LER with 90% intervals, plus the fitted spectra."""
    w = node_weights(d["n"])
    bounds = ROUGH_BOUNDS + [ETA_BOUNDS] * (2 * NODES_K.size)
    e0 = _noise_start(d)
    best = min((minimize(_objective, [*s, *e0], args=(d, w), jac=True, method="L-BFGS-B", bounds=bounds)
                for s in STARTS), key=lambda r: r.fun)
    x = best.x

    def profiled(lp, rest):
        val, g = _objective(np.r_[lp, rest], d, w)
        return val, g[1:]

    lo, hi = _profile(profiled, x[0], x[1:], best.fun, bounds[1:])
    xi, hurst = float(np.exp(x[1])), float(x[2])
    frac = in_band_fraction(d["n"], d["kc"], xi, hurst)
    ler = float(np.exp(x[0]))
    return dict(ler=ler, lo=lo, hi=hi, band=ler * frac, band_lo=lo * frac, band_hi=hi * frac, xi=xi, hurst=hurst,
                rough=np.exp(2 * x[0]) * unit_rough(d["f"], xi, hurst, d["n"], d["h2"]),
                noise=np.exp((x[3:].reshape(2, -1) @ w.T)[d["side"]]).mean(0))
