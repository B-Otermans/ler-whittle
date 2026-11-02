import numpy as np
import pytest

from lerwhittle import PatternParams, SEMParams, edge_traces, edge_transfer, fit, generate, model_library, panel_spectra
from lerwhittle.synth import PSD_SHAPES, rough_edge

NOISE_FREE = dict(dose=np.inf, read_noise=0.0, scan_noise=0.0, bit_depth=None)


@pytest.mark.parametrize("shape", PSD_SHAPES)
def test_roughness_has_the_requested_sigma(shape):
    e = rough_edge(2**17, 0.25, 0.1, 2.2, 0.33, np.random.default_rng(1), shape)
    assert e.std() == pytest.approx(0.1, rel=0.05)


def test_model_detector_is_unbiased_and_sub_hundredth_pixel_on_noise_free_edges():
    from dataclasses import replace

    pattern = PatternParams(ler_sigma=0.0, cd_line_sigma=0.0, placement_sigma=0.0, tilt=1 / 217)
    sem = replace(SEMParams(), **NOISE_FREE)
    frames, truth = generate(pattern, sem, seed=0)
    found = edge_traces(frames[0], model_library(pattern, sem))
    errors = []
    for t in (t for t in truth if t["complete"]):
        ln = min(found, key=lambda f: abs(f["center_row"] - np.mean(0.5 * (t["top"] + t["bottom"]))))
        errors += [ln["top"] - t["top"], ln["bottom"] - t["bottom"]]
    errors = np.concatenate(errors)
    assert abs(errors.mean()) < 0.005
    assert errors.std() < 0.01


def test_fit_recovers_the_ler_of_one_panel():
    pattern, sem = PatternParams(), SEMParams()
    library = model_library(pattern, sem)
    frames, truth = generate(pattern, sem, seed=3000, n_frames=2)
    d = panel_spectra(frames, truth, library, edge_transfer(pattern, sem, library))
    r = fit(d)
    assert r["lo"] <= d["true"] <= r["hi"]
    assert r["ler"] == pytest.approx(d["true"], rel=0.1)
