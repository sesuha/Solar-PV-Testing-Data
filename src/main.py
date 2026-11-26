"""Runs everything: parse -> QC -> clean -> smooth -> metrics -> Rs/Rsh -> figures -> export.

    python -m src.main                              # all 28 panels / 56 sweeps
    python -m src.main --file "1 Nov" --panel 4     # one panel (summary not rewritten)
    python -m src.main --show-sdm --no-raw          # SDM curve on Fig. A, hide raw points
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from src import pvplots as PL
from src.export import methods_text, per_panel_table, per_sweep_table, write_latex, write_summary
from src.parse import ROOT, load_all, load_config
from src.pvcore import (acceptance, build_curve, clean_points, decide_shape, fit_sdm, fit_sweep, metrics,
                        rs_rsh_geometric, rs_rsh_slope)
from src.qc import qc_sweep

OUT = ROOT / "outputs"


def slug(tag):
    return tag.replace(" ", "_")


def panel_dir(tag, p):
    d = OUT / "figures" / slug(tag) / f"Panel_{p:02d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def analyse_panel(tag, p, pair, S, cfg):
    """QC -> clean -> fit both sweeps -> pair rule -> curve, metrics, Rs/Rsh. No figures."""
    recs, fits = [], {}
    for pos, (n, other) in enumerate(((pair[0], pair[1]), (pair[1], pair[0])), start=1):
        s = S[n]
        status = clean_points(s["V"], cfg)
        q, why = qc_sweep(s["V"], s["I"], s["meta"], float(S[other]["I"].max()), status == "used", cfg)
        fits[n] = None if q == "invalid" else fit_sweep(s["V"], s["I"], status, s["meta"]["voc"], s["meta"]["isc"], cfg)
        recs.append(dict(file=tag, panel=p, sample=n, pos=pos, dt=s["meta"].get("dt", ""), qc=q, qc_reasons=why,
                         V=s["V"], I=s["I"], P=s["P"], status=status))
    shape = decide_shape(fits[pair[0]], fits[pair[1]], cfg)
    for r in recs:
        f = fits[r["sample"]]
        if f is not None:
            c = build_curve(f, shape, cfg)
            m = metrics(c, r["V"], r["I"], r["P"], cfg)
            g, sl = rs_rsh_geometric(m), rs_rsh_slope(c, cfg["metrics"]["rs_window_frac"], cfg["metrics"]["rsh_window_frac"])
            u = c["status"] == "used"
            sdm = fit_sdm(r["V"][u], r["I"][u], m["Voc"], m["Isc"], cfg)
            r.update(status=c["status"], shape=shape, c=c, m=m, g=g, sl=sl, sdm=sdm, accept=acceptance(m, shape, cfg))
        r["n_used"] = int((r["status"] == "used").sum())
        r["n_spike"] = int((r["status"] == "spike").sum())
        r["n_excluded"] = int((r["status"] == "excluded").sum())
    return recs


def process_panel(job):
    """Worker: one panel (both sweeps - the pair rule needs both), analysis + figures."""
    tag, p, pair, S, cfg, opts = job
    recs = analyse_panel(tag, p, pair, S, cfg)
    d = panel_dir(tag, p)
    for r in recs:
        sty = PL.S_STYLE[r["pos"]]
        base = str(d / f"S{r['sample']}")
        if r.get("c") is None:
            PL.fig_invalid(r["V"], r["I"], r["status"], "; ".join(r["qc_reasons"]), sty, base + "_IVPV")
            PL.fig_invalid(r["V"], r["I"], r["status"], "; ".join(r["qc_reasons"]), sty, base + "_IVPV_combined",
                           combined=True)
            continue
        c, m = r["c"], r["m"]
        PL.fig_iv_pv(r["V"], r["I"], c, m, sty, base + "_IVPV", sdm=r["sdm"] if opts["show_sdm"] else None,
                     show_raw=opts["raw"])
        PL.fig_iv_pv_combined(r["V"], r["I"], c, m, sty, base + "_IVPV_combined", show_raw=opts["raw"])
        PL.fig_rs(c, m, r["g"], r["sl"], r["sdm"], sty, base + "_Rs")
        PL.fig_rsh(c, m, r["g"], r["sl"], r["sdm"], sty, base + "_Rsh")
    PL.fig_panel_overlay([(r["V"], r["I"], r.get("c"), r.get("m"), r["status"]) for r in recs],
                         str(d / "overlay"), show_raw=opts["raw"])
    return recs


def run(file=None, panel=None, show_sdm=False, raw=True, workers=None):
    t0 = time.time()
    cfg = load_config()
    data = load_all(cfg)
    opts = dict(show_sdm=show_sdm, raw=raw)
    jobs = []
    for tag, pairs in cfg["panels"].items():
        if file and tag != file:
            continue
        for p, pair in enumerate(pairs, start=1):
            if panel and p != panel:
                continue
            jobs.append((tag, p, pair, {n: data[tag][n] for n in pair}, cfg, opts))
    workers = workers or max(1, min(len(jobs), (os.cpu_count() or 2) - 1))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        recs = [r for out in ex.map(process_panel, jobs) for r in out]

    for r in recs:
        m = r.get("m")
        line = (f"{r['file']:6} P{r['panel']:<2} S{r['pos']} #{r['sample']:<3} {r['qc']:8} {r.get('shape', '-'):8}")
        if m:
            line += (f" Pmax {m['Pmax']:6.2f} W  eps {m['err_Pmax_pct']:+6.2f} % ({r['accept']})  FF {m['FF']:.3f}"
                     f"  Rs {r['sl']['Rs_slope']:.2f}  SDM {'ok' if r['sdm']['valid'] else '--'}")
        print(line)
    if file or panel:
        print(f"subset run: {len(recs)} sweeps, figures in {OUT / 'figures'} (summary not rewritten)")
        return recs

    # per-file figures: E (Pmax summary), F (contact sheet), multi-page appendix PDF of Fig. A
    sheet_all = []
    for tag in cfg["panels"]:
        fr = [r for r in recs if r["file"] == tag]
        fd = OUT / "figures" / slug(tag)
        PL.fig_summary_pmax([dict(panel=r["panel"], pos=r["pos"], qc=r["qc"], Pmax_raw=float(r["P"].max()),
                                  Pmax=r["m"]["Pmax"] if r.get("m") else np.nan,
                                  u_Pmax=r["m"]["u_Pmax"] if r.get("m") else np.nan) for r in fr],
                            str(fd / "summary_pmax"), tag)
        cells = []
        for p in sorted({r["panel"] for r in fr}):
            pr = sorted([r for r in fr if r["panel"] == p], key=lambda r: r["pos"])
            shape = next((r["shape"] for r in pr if r.get("shape")), "invalid")
            cells.append((f"{tag} P{p}", [(r["V"], r["I"], r.get("c")) for r in pr], shape))
        PL.fig_contact_sheet(cells, str(fd / "contact_sheet"), ncols=4)
        sheet_all += cells
        with PL.open_multipage(fd / "appendix_IVPV.pdf") as pdf:
            for r in fr:
                if r.get("m"):
                    PL.fig_iv_pv(r["V"], r["I"], r["c"], r["m"], PL.S_STYLE[r["pos"]], None, show_raw=raw, pdf=pdf,
                                 title=f"{tag} · Panel {r['panel']} · S{r['pos']} (#{r['sample']}) · {r['shape']}")
    PL.fig_contact_sheet(sheet_all, str(OUT / "figures" / "contact_sheet_all"), ncols=5)

    # Phase 7 exports
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    ps = per_sweep_table(recs)
    pp = per_panel_table(ps, cfg)
    points = pd.concat([pd.DataFrame({"file": r["file"], "panel": r["panel"], "sample": r["sample"],
                                      "point": np.arange(1, len(r["V"]) + 1), "V (V)": r["V"], "I (A)": r["I"],
                                      "P (W)": r["P"], "status": r["status"]}) for r in recs], ignore_index=True)
    write_summary(OUT / "summary.xlsx", ps, pp, points)
    write_latex(OUT / "tables" / "panel_table.tex", ps)
    (OUT / "methods.md").write_text(methods_text(ps, cfg), encoding="utf-8")
    curves = pd.concat([pd.DataFrame({"file": r["file"], "sample": r["sample"], "Vs": r["c"]["Vs"], "Is": r["c"]["Is"]})
                        for r in recs if r.get("c")], ignore_index=True)
    curves.to_csv(OUT / "curves.csv", index=False)
    print(f"done: {len(recs)} sweeps in {time.time() - t0:.0f} s -> {OUT}")
    return recs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help='"18 Oct" or "1 Nov"')
    ap.add_argument("--panel", type=int)
    ap.add_argument("--show-sdm", action="store_true", help="draw the single-diode curve on Fig. A when valid")
    ap.add_argument("--no-raw", action="store_true", help="hide raw points")
    ap.add_argument("--workers", type=int)
    a = ap.parse_args()
    run(a.file, a.panel, show_sdm=a.show_sdm, raw=not a.no_raw, workers=a.workers)
