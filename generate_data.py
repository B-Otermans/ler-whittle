"""Write synthetic two-frame SEM panels with their exact edges.

    python generate_data.py 10 data            10 calibrated panels into data/
    python generate_data.py 10 data --dose 8   lower dose (noisier)

Per panel: <id>_frame1.png, <id>_frame2.png, <id>_truth.npz (exact top/bottom edge per line and
column, and which lines lie fully inside the image) and <id>.json (all generator parameters).
"""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import numpy as np

from lerwhittle import PatternParams, SEMParams, generate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("out", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dose", type=float, default=SEMParams.dose)
    ap.add_argument("--ler", type=float, default=PatternParams.ler_sigma, help="roughness sigma, px")
    ap.add_argument("--shape", default=PatternParams.ler_psd, help="roughness spectrum shape")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pattern = replace(PatternParams(), ler_sigma=args.ler, ler_psd=args.shape)
    sem = replace(SEMParams(), dose=args.dose)
    for i in range(args.n):
        seed = args.seed + i
        frames, truth = generate(pattern, sem, seed, n_frames=2)
        name = f"{seed:05d}"
        for k, img in enumerate(frames, 1):
            cv2.imwrite(str(args.out / f"{name}_frame{k}.png"), img)
        np.savez_compressed(args.out / f"{name}_truth.npz", top=np.array([t["top"] for t in truth]),
                            bottom=np.array([t["bottom"] for t in truth]),
                            complete=np.array([t["complete"] for t in truth]))
        (args.out / f"{name}.json").write_text(json.dumps(dict(seed=seed, pattern=asdict(pattern),
                                                               sem=asdict(sem)), indent=1, default=str))
    print(f"wrote {args.n} panels to {args.out}")


if __name__ == "__main__":
    main()
