"""Figures: one panel step by step (figures/pipeline.png) and the benchmark (figures/benchmark.png).

Run benchmark.py first for the benchmark figure.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lerwhittle import PatternParams, SEMParams, edge_traces, edge_transfer, fit, generate, model_library, panel_spectra

ROOT = Path(__file__).parent
OUT = ROOT / "figures"
SEED = 3000


def pipeline():
    pattern, sem = PatternParams(), SEMParams()
    library = model_library(pattern, sem)
    frames, truth = generate(pattern, sem, SEED, n_frames=2)
    d = panel_spectra(frames, truth, library, edge_transfer(pattern, sem, library))
    r = fit(d)
    fa, fb = (edge_traces(f, library) for f in frames)
    k = len(fa) // 2
    la, lb = fa[k], fb[k]
    true = min((t for t in truth if t["complete"]),
               key=lambda t: abs(np.mean(0.5 * (t["top"] + t["bottom"])) - la["center_row"]))

    fig = plt.figure(figsize=(14, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.25, 1.25], wspace=0.3)
    cols = np.arange(frames[0].shape[1])

    ax = fig.add_subplot(gs[0])
    r0 = int(la["center_row"]) - 24
    r1, c1 = r0 + 48, 64
    ax.imshow(frames[0][r0:r1, :c1], cmap="gray", interpolation="nearest", extent=(-0.5, c1 - 0.5, r1 - 0.5, r0 - 0.5))
    for ln in fa:
        for side in ("top", "bottom"):
            ax.plot(cols[:c1], ln[side][:c1], color="#f2a623", lw=1.0)
    ax.plot(cols[:c1], la["top"][:c1], color="#2a78d6", lw=2.0)
    ax.set_xlim(-0.5, c1 - 0.5)
    ax.set_ylim(r1 - 0.5, r0 - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"1. Synthetic SEM image, zoomed ({c1}×{r1 - r0} px)\ndetected sub-pixel edges overlaid",
                 fontsize=9.5, loc="left")

    ax = fig.add_subplot(gs[1])
    ref = np.mean(true["top"])
    ax.plot(cols, la["top"] - ref, color="#7f77dd", lw=1.0, label="frame 1")
    ax.plot(cols, lb["top"] - ref, color="#d4537e", lw=1.0, label="frame 2")
    ax.plot(cols, true["top"] - ref, color="k", lw=1.8, label="true edge")
    ax.set_xlabel("position along the line [px]")
    ax.set_ylabel("edge position [px]")
    ax.set_ylim(-0.55, 0.7)
    ax.legend(fontsize=7.5, ncol=3, loc="upper right", frameon=False, columnspacing=1.0, handlelength=1.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("2. The blue edge, traced in two frames\nsame roughness, different noise; all well below 1 px",
                 fontsize=9.5, loc="left")

    ax = fig.add_subplot(gs[2])
    f, fc = d["f"], d["kc"] / d["n"]
    ax.axvspan(fc, 0.5, color="0.5", alpha=0.12, lw=0)
    inband = f <= fc
    ax.fill_between(f[inband], 1e-5, r["rough"][inband], color="#1D9E75", alpha=0.22, lw=0)
    ax.semilogy(f, d["Is"].mean(0), "o", ms=2.6, color="#378ADD", alpha=0.6)
    ax.semilogy(f, d["Id"].mean(0), "o", ms=2.6, color="#888780", alpha=0.6)
    ax.semilogy(f, r["noise"], "--", color="#5F5E5A", lw=1.4)
    ax.semilogy(f, r["rough"] + r["noise"], color="#2a78d6", lw=1.8)
    ax.semilogy(f, r["rough"], color="#1D9E75", lw=2)
    for label, fx, spec, xy, color in (("roughness + noise\n(average of frames)", 0.06, d["Is"], 0.025, "#185FA5"),
                                       ("noise only\n(difference of frames)", 0.26, d["Id"], 0.245, "#444441")):
        i = np.argmin(abs(f - fx))
        ax.annotate(label, xy=(f[i], spec.mean(0)[i]), xytext=(xy, 0.9), va="top", fontsize=8, color=color,
                    arrowprops=dict(arrowstyle="->", color=color))
    ax.text(0.03, 1.4e-4, "the roughness\nwe measure", fontsize=9, color="#085041")
    ax.text(fc + 0.01, 1.3e-5, "blurred out\nby the SEM", fontsize=8, color="0.35")
    ax.set_ylim(1e-5, 1.0)
    ax.set_xlim(0, 0.5)
    ax.set_xticks([1 / t for t in (10, 5, 3, 2)], [f"{t} px" for t in (10, 5, 3, 2)])
    ax.set_xlabel("wiggle length along the edge  (long waves → fine jitter)")
    ax.set_ylabel("wiggle strength (log scale)")
    ax.set_yticks([])
    ax.minorticks_off()
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"3. What the fit separates, all {len(d['Is'])} edges (lines: fitted model)\n"
                 f"LER {r['band']:.3f} px (90% interval {r['band_lo']:.3f}–{r['band_hi']:.3f}), "
                 f"truth {d['true_band']:.3f} px", fontsize=9.5, loc="left")
    fig.savefig(OUT / "pipeline.png", dpi=150, bbox_inches="tight")


def benchmark():
    res = json.loads((ROOT / "results" / "benchmark.json").read_text())
    names = list(res)
    x = np.arange(len(names))
    s = [res[n]["summary"]["ler"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.axhspan(-3, 3, color="#2a78d6", alpha=0.08, lw=0)
    ax.errorbar(x, [100 * v["bias"] for v in s], [100 * v["spread"] for v in s], fmt="o", color="#2a78d6", ms=8,
                capsize=5, lw=1.8, label="mean error ± spread per SEM panel")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, [f"{n}\n\n90% interval held\nthe truth: {v['coverage']:.0%}" for n, v in zip(names, s)],
                  fontsize=9)
    ax.set_ylabel("LER error vs exact truth [%]")
    ax.set_ylim(-24, 22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, loc="upper right", frameon=False)
    ax.set_title("LER from two SEM frames, noise and roughness fitted jointly", fontsize=10.5, loc="left")
    fig.text(0.01, 0.01, "Synthetic panels with known truth, ~26 edges each. Shaded: ±3%.", fontsize=7.5, color="0.4")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "benchmark.png", dpi=150)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    pipeline()
    if (ROOT / "results" / "benchmark.json").exists():
        benchmark()
