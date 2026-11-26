"""Phase 8: validation tests (plan table + regression values quoted in the plan).

The analysis tests run the full analysis (no figures) once, ~10 s. The export tests read
outputs/ and are skipped if `python -m src.main` has not been run yet.
"""
import numpy as np
import pandas as pd
import pytest

from src.main import OUT, analyse_panel
from src.parse import load_all, load_config
from src.pvcore import build_curve, clean_points, fit_sweep, metrics


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def data(cfg):
    return load_all(cfg)


@pytest.fixture(scope="module")
def recs(cfg, data):
    out = {}
    for tag, pairs in cfg["panels"].items():
        for p, pair in enumerate(pairs, start=1):
            for r in analyse_panel(tag, p, pair, {n: data[tag][n] for n in pair}, cfg):
                out[(tag, r["sample"])] = r
    return out


def fitted(recs):
    return {k: r for k, r in recs.items() if r.get("c") is not None}


# ---------------------------------------------------------------- Phase 1 checkpoint
def test_parser_counts(cfg, data):
    assert len(data["18 Oct"]) == 50 and len(data["1 Nov"]) == 41
    assert all(len(s["V"]) == 150 for S in data.values() for s in S.values())
    for tag, pairs in cfg["panels"].items():
        assert all(n in data[tag] for pair in pairs for n in pair)


# ---------------------------------------------------------------- plan table
def test_monotone(recs):
    for k, r in fitted(recs).items():
        assert np.all(np.diff(r["c"]["Is"]) <= 1e-9), k


def test_no_self_crossing(recs):
    for k, r in fitted(recs).items():
        assert np.all(np.diff(r["c"]["Vs"]) > 0), k


def test_curve_ends_at_I0_and_starts_at_Isc(recs):
    for k, r in fitted(recs).items():
        assert r["c"]["Is"][-1] == 0.0 and r["c"]["Vs"][0] == 0.0, k


def test_pmax_fidelity(recs, cfg):
    """|eps| <= 1 % (normal) / 2.5 % (stepped), or within 2x the reference's own standard error,
    for every qc == ok sweep whose data bracket the MPP. Degraded low-FF panels (FF < 0.3,
    plan section 5: 'report as degraded') are reported, not held to the tolerance."""
    bad = []
    for k, r in fitted(recs).items():
        if r["qc"] != "ok" or not r["m"]["ref_bracketed"] or r["m"]["FF"] < 0.3:
            continue
        if r["accept"] != "pass":
            bad.append((k, round(r["m"]["err_Pmax_pct"], 2), r["shape"]))
    assert not bad, bad


def test_median_pmax_error(recs):
    eps = [abs(r["m"]["err_Pmax_pct"]) for r in fitted(recs).values() if r["m"]["ref_bracketed"]]
    assert np.median(eps) <= 0.5          # plan prototype: 0.39 %


@pytest.mark.parametrize("key", [("18 Oct", 19), ("18 Oct", 20), ("1 Nov", 27), ("1 Nov", 28)])
def test_step_preserved(recs, key):
    assert recs[key]["shape"] == "stepped"


@pytest.mark.parametrize("key", [("1 Nov", 11), ("1 Nov", 12), ("18 Oct", 14)])
def test_no_false_steps(recs, key):
    assert recs[key]["shape"] == "normal"


@pytest.mark.parametrize("key", [("18 Oct", 13), ("1 Nov", 36), ("1 Nov", 37)])
def test_qc_invalid(recs, key):
    assert recs[key]["qc"] == "invalid" and recs[key].get("c") is None


def test_only_known_sweeps_invalid(recs):
    assert {k for k, r in recs.items() if r["qc"] == "invalid"} == {("18 Oct", 13), ("1 Nov", 36), ("1 Nov", 37)}


@pytest.mark.parametrize("seed", range(5))
def test_robust_to_injected_spikes(cfg, data, seed):
    """Inject 10 random +/-5 V spikes into a clean sweep: Pmax changes by < 0.5 %."""
    s = data["1 Nov"][12]

    def pmax(V):
        st = clean_points(V, cfg)
        f = fit_sweep(V, s["I"], st, s["meta"]["voc"], s["meta"]["isc"], cfg)
        c = build_curve(f, "normal", cfg)
        return metrics(c, V, s["I"], V * s["I"], cfg)["Pmax"]

    rng = np.random.default_rng(seed)
    V = s["V"].copy()
    idx = rng.choice(np.arange(1, len(V)), 10, replace=False)
    V[idx] += rng.choice([-5.0, 5.0], 10)
    V = np.clip(V, 0, None)
    p0, p1 = pmax(s["V"]), pmax(V)
    assert abs(p1 - p0) / p0 * 100 < 0.5, (p0, p1)


# ---------------------------------------------------------------- Rs/Rsh regression (plan 5.1, 5.3)
def test_sdm_passes_on_clean_curve(recs):
    sdm = recs[("1 Nov", 11)]["sdm"]
    assert sdm["valid"]
    assert sdm["Rs"] == pytest.approx(0.24, abs=0.03) and sdm["n"] == pytest.approx(1.46, abs=0.05)
    assert sdm["Rsh"] == pytest.approx(113, rel=0.1)


def test_sdm_rejected_on_step_curve(recs):
    assert not recs[("18 Oct", 19)]["sdm"]["valid"]


def test_geometric_rs_is_upper_bound(recs):
    r = recs[("1 Nov", 12)]
    assert r["g"]["Rs_geo"] == pytest.approx(1.12, abs=0.03)
    assert r["g"]["Rs_geo"] > r["sl"]["Rs_slope"] > r["sdm"]["Rs"]


# ---------------------------------------------------------------- exports (Phase 7)
@pytest.fixture(scope="module")
def summary():
    p = OUT / "summary.xlsx"
    if not p.exists():
        pytest.skip("run `python -m src.main` first")
    # keep_default_na=False: pandas would otherwise read the literal text "n/a" back as NaN
    return pd.read_excel(p, sheet_name=None, keep_default_na=False)


def test_no_nan_in_exports(summary):
    """Every exported number is finite, or explicitly 'n/a'."""
    for name, df in summary.items():
        for col in df.columns:
            for v in df[col]:
                if isinstance(v, str):
                    assert v == "" or v.startswith("n/a") or not v.replace(".", "").lstrip("-").isdigit(), (name, col, v)
                    continue
                assert pd.notna(v) and np.isfinite(float(v)), (name, col, v)


def test_summary_sheets(summary):
    assert set(summary) == {"Per Sweep", "Per Panel", "Point Status"}
    assert len(summary["Per Sweep"]) == 56 and len(summary["Per Panel"]) == 28
    assert len(summary["Point Status"]) == 56 * 150


def test_figure_files():
    figs = OUT / "figures"
    if not figs.exists():
        pytest.skip("run `python -m src.main` first")
    pdfs = list(figs.rglob("*.pdf"))
    pngs = list(figs.rglob("*.png"))
    # 53 fitted sweeps x 4 figures + 3 invalid x 2 + 28 overlays (+ per file: E, F, appendix; + contact_sheet_all)
    assert len(pngs) == 53 * 4 + 3 * 2 + 28 + 2 * 2 + 1
    assert len(pdfs) == len(pngs) + 2
    assert (OUT / "tables" / "panel_table.tex").exists() and (OUT / "methods.md").exists()
