"""LER of synthetic two-frame SEM panels vs exact truth, per imaging condition.

    python benchmark.py            noise and roughness levels (results/benchmark.json)
    python benchmark.py --shapes   non-Palasantzas roughness spectra (results/shapes.json)
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np

from lerwhittle import PatternParams, SEMParams, edge_transfer, fit, generate, model_library, panel_spectra
from lerwhittle.synth import PSD_SHAPES

P0, S0 = PatternParams(), SEMParams()
CONDITIONS = {  # name: (pattern, sem, panels, first seed)
    "calibrated": (P0, S0, 40, 3000),
    "low LER (0.05 px)": (replace(P0, ler_sigma=0.05), S0, 20, 3100),
    "high LER (0.2 px)": (replace(P0, ler_sigma=0.2), S0, 20, 3200),
    "low dose (8)": (P0, replace(S0, dose=8.0), 20, 3300),
    "high dose (100)": (P0, replace(S0, dose=100.0), 20, 3400),
}
SHAPES = {shape: (replace(P0, ler_psd=shape), S0, 20, 6000 + 100 * i) for i, shape in enumerate(PSD_SHAPES)}


def run_panel(seed, pattern, sem, library, h2):
    frames, truth = generate(pattern, sem, seed, n_frames=2)
    d = panel_spectra(frames, truth, library, h2)
    r = fit(d)
    return dict(seed=seed, true=d["true"], true_band=d["true_band"],
                **{k: r[k] for k in ("ler", "lo", "hi", "band", "band_lo", "band_hi")})


def summary(panels):
    out = {}
    for est, true, lo, hi in (("ler", "true", "lo", "hi"), ("band", "true_band", "band_lo", "band_hi")):
        err = np.array([p[est] / p[true] - 1 for p in panels])
        out[est] = dict(bias=float(err.mean()), spread=float(err.std(ddof=1)),
                        coverage=float(np.mean([p[lo] <= p[true] <= p[hi] for p in panels])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", action="store_true", help="roughness spectrum shapes instead of noise levels")
    args = ap.parse_args()
    conditions = SHAPES if args.shapes else CONDITIONS
    library = model_library(P0, S0)  # the calibrated imaging model, as a real recipe would use
    results = {}
    print(f"{'condition':>18} | panels | LER error (mean ± spread), 90% coverage | in-band LER error, 90% coverage")
    for name, (pattern, sem, n, seed0) in conditions.items():
        h2 = edge_transfer(P0 if args.shapes else pattern, sem, library)
        with ProcessPoolExecutor() as ex:
            panels = list(ex.map(partial(run_panel, pattern=pattern, sem=sem, library=library, h2=h2),
                                 range(seed0, seed0 + n)))
        s = summary(panels)
        results[name] = dict(summary=s, panels=panels)
        print(f"{name:>18} | {n:6d} | {s['ler']['bias']:+6.1%} ± {s['ler']['spread']:5.1%}, {s['ler']['coverage']:4.0%}"
              f"               | {s['band']['bias']:+6.1%} ± {s['band']['spread']:5.1%}, {s['band']['coverage']:4.0%}")
    out = Path(__file__).parent / "results" / ("shapes.json" if args.shapes else "benchmark.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
