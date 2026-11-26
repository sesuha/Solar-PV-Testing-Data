"""Phase 7: summary.xlsx, LaTeX panel table, Methods paragraph."""
import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

NA = "n/a"


def _num(x, nd=None):
    """Finite number, or the string 'n/a' (exports never contain NaN/inf)."""
    if x is None or isinstance(x, str):
        return NA if x is None else x
    if not np.isfinite(x):
        return NA if np.isnan(x) else NA + " (flat: > 1 MΩ)"
    return round(float(x), nd) if nd is not None else float(x)


def per_sweep_table(records):
    rows = []
    for r in records:
        m, g, sl, sdm = r.get("m"), r.get("g"), r.get("sl"), r.get("sdm")
        row = {"file": r["file"], "panel": r["panel"], "sample": r["sample"], "sweep": f"S{r['pos']}",
               "timestamp": r["dt"], "qc_flag": r["qc"], "qc_reason": "; ".join(r["qc_reasons"]),
               "shape": r.get("shape", NA), "n_used": r["n_used"], "n_spike": r["n_spike"],
               "n_excluded": r["n_excluded"],
               "Max V raw (V)": _num(r["V"].max()), "Max I raw (A)": _num(r["I"].max()),
               "Max P raw (W)": _num(r["P"].max())}
        if m is None:
            for k in ("Voc (V)", "Isc (A)", "Vmp (V)", "Imp (A)", "Pmax (W)", "Pmax_ref (W)", "ref bracketed",
                      "Pmax_ref SE (%)", "eps_Pmax (%)", "Pmax acceptance", "eps_raw (%)", "u(Pmax) (W)", "FF",
                      "dV (V)", "dI (A)", "Rs geo (Ω)", "Rs slope (Ω)", "Rs SDM (Ω)", "Rsh geo (Ω)",
                      "Rsh slope (Ω)", "Rsh SDM (Ω)", "SDM n", "SDM nRMSE (%)", "SDM valid", "SDM reject reason",
                      "curve nRMSE", "Isc end line"):
                row[k] = NA
        else:
            row.update({"Voc (V)": _num(m["Voc"]), "Isc (A)": _num(m["Isc"]), "Vmp (V)": _num(m["Vmp"]),
                        "Imp (A)": _num(m["Imp"]), "Pmax (W)": _num(m["Pmax"]), "Pmax_ref (W)": _num(m["Pmax_ref"]),
                        "ref bracketed": bool(m["ref_bracketed"]),
                        "Pmax_ref SE (%)": _num(100 * m["Pmax_ref_se"] / m["Pmax_ref"]),
                        "eps_Pmax (%)": _num(m["err_Pmax_pct"]), "Pmax acceptance": r["accept"],
                        "eps_raw (%)": _num(m["err_Pmax_raw_pct"]), "u(Pmax) (W)": _num(m["u_Pmax"]),
                        "FF": _num(m["FF"]), "dV (V)": _num(g["dV"]), "dI (A)": _num(g["dI"]),
                        "Rs geo (Ω)": _num(g["Rs_geo"]), "Rs slope (Ω)": _num(sl["Rs_slope"]),
                        "Rs SDM (Ω)": _num(sdm["Rs"]) if sdm["valid"] else NA,
                        "Rsh geo (Ω)": _num(g["Rsh_geo"]), "Rsh slope (Ω)": _num(sl["Rsh_slope"]),
                        "Rsh SDM (Ω)": _num(sdm["Rsh"]) if sdm["valid"] else NA,
                        "SDM n": _num(sdm["n"]), "SDM nRMSE (%)": _num(100 * sdm["nrmse"]),
                        "SDM valid": bool(sdm["valid"]), "SDM reject reason": sdm["reject_reason"] or "",
                        "curve nRMSE": _num(r["c"]["nrmse"]), "Isc end line": r["c"]["end_line"]})
        row["eps_rated (%)"] = NA + " (no G/T recorded)"
        rows.append(row)
    return pd.DataFrame(rows)


def per_panel_table(ps, cfg):
    """The notebook table: Panel data | Measured S1 | Measured S2 | dPmax % | Rs | Rsh."""
    out = []
    for (file, panel), g in ps.groupby(["file", "panel"], sort=False):
        s1, s2 = g[g.sweep == "S1"].iloc[0], g[g.sweep == "S2"].iloc[0]
        ds = cfg["module"]["datasheet"].get(f"{file}/{panel}", {})
        rec = {"file": file, "panel": panel, "samples": f"{s1['sample']}, {s2['sample']}",
               "Panel ID": ds.get("id", ""), "Panel Voc (V)": ds.get("Voc", NA), "Panel Isc (A)": ds.get("Isc", NA),
               "Panel Pmp (W)": ds.get("Pmax", NA)}
        for tag, s in (("S1", s1), ("S2", s2)):
            for k in ("Voc (V)", "Isc (A)", "Pmax (W)", "FF"):
                rec[f"{tag} {k.replace('Pmax', 'Pmp')}"] = s[k]
            rec[f"{tag} QC"] = s["qc_flag"]
        p1, p2 = s1["Pmax (W)"], s2["Pmax (W)"]
        valid = not (isinstance(p1, str) or isinstance(p2, str))
        rec["dPmax S2-S1 (%)"] = round(100 * (p2 - p1) / ((p1 + p2) / 2), 3) if valid else NA
        rec["within repeatability"] = (abs(rec["dPmax S2-S1 (%)"]) <= cfg["acceptance"]["repeat_pct"]) if valid else NA
        for k in ("Rs slope (Ω)", "Rsh slope (Ω)", "Rs SDM (Ω)", "Rsh SDM (Ω)"):
            vals = [v for v in (s1[k], s2[k]) if not isinstance(v, str)]
            rec[f"mean {k}"] = round(float(np.mean(vals)), 3) if vals else NA
        rec["shape"] = s1["shape"] if not isinstance(s1["shape"], float) else NA
        rec["eps_rated (%)"] = NA + " (no G/T recorded)"
        out.append(rec)
    return pd.DataFrame(out)


def _style(ws, df, wide=()):
    head = PatternFill("solid", fgColor="DCE6F2")
    for c in ws[1]:
        c.font = Font(bold=True)
        c.fill = head
        c.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
    ws.row_dimensions[1].height = 42
    ws.freeze_panes = "E2"
    for j, col in enumerate(df.columns, start=1):
        ws.column_dimensions[get_column_letter(j)].width = 60 if col in wide else max(9, min(16, len(str(col)) + 2))


def write_summary(path, ps, pp, points):
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        ps.to_excel(xw, sheet_name="Per Sweep", index=False)
        pp.to_excel(xw, sheet_name="Per Panel", index=False)
        points.to_excel(xw, sheet_name="Point Status", index=False)
        _style(xw.sheets["Per Sweep"], ps, wide=("qc_reason", "SDM reject reason"))
        _style(xw.sheets["Per Panel"], pp)
        _style(xw.sheets["Point Status"], points)
        ws, warn = xw.sheets["Per Sweep"], PatternFill("solid", fgColor="F8CBAD")
        for j, col in enumerate(ps.columns, start=1):
            if col in ("qc_flag", "Pmax acceptance"):
                for i, v in enumerate(ps[col], start=2):
                    if v in ("invalid", "low confidence"):
                        ws.cell(row=i, column=j).fill = warn


def write_latex(path, ps):
    """Panel table for the paper: mean +/- spread (half the S1-S2 difference) of the valid sweeps."""
    rows = []
    for (file, panel), g in ps.groupby(["file", "panel"], sort=False):
        ok = g[g.qc_flag != "invalid"]
        rec = {"File": file, "Panel": panel}
        for k, lab, nd in (("Voc (V)", "$V_{oc}$ (V)", 2), ("Isc (A)", "$I_{sc}$ (A)", 3), ("Pmax (W)", "$P_{max}$ (W)", 2),
                           ("FF", "FF", 3), ("Rs slope (Ω)", "$R_s$ ($\\Omega$)", 2), ("Rsh slope (Ω)", "$R_{sh}$ ($\\Omega$)", 0)):
            vals = np.array([v for v in ok[k] if not isinstance(v, str) and np.isfinite(v) and v < 1e6], float)
            if len(vals) == 0:
                rec[lab] = "n/a"
            elif len(vals) == 1:
                rec[lab] = f"{vals[0]:.{nd}f}$^\\dagger$"
            else:
                rec[lab] = f"{vals.mean():.{nd}f} $\\pm$ {abs(vals[1] - vals[0]) / 2:.{nd}f}"
        rows.append(rec)
    df = pd.DataFrame(rows)
    tex = df.to_latex(index=False, escape=False, column_format="ll" + "r" * (df.shape[1] - 2),
                      caption="Measured characteristics per panel: mean $\\pm$ half the difference of the two "
                              "repeat sweeps (S1, S2). $^\\dagger$ only one usable value (other sweep invalid, or a "
                              "flat shunt segment with $R_{sh} > 1$ M$\\Omega$). $R_s$, $R_{sh}$: slope method.",
                      label="tab:panels")
    path.write_text(tex, encoding="utf-8")
    return df


def methods_text(ps, cfg):
    fit = ps[ps.qc_flag != "invalid"]
    eps = np.array([v for v in fit["eps_Pmax (%)"] if not isinstance(v, str)], float)
    br = fit["ref bracketed"] == True  # noqa: E712
    eps_br = np.array([v for v in fit.loc[br, "eps_Pmax (%)"]], float)
    n_inv = int((ps.qc_flag == "invalid").sum())
    n_step = int((fit["shape"] == "stepped").sum())
    n_sdm = int((fit["SDM valid"] == True).sum())  # noqa: E712
    acc = fit["Pmax acceptance"].value_counts().to_dict()
    c, s = cfg["cleaning"], cfg["smoothing"]
    return (
        f"I–V curves were recorded with a PROVA 200A analyser (150 points per sweep, current-swept). "
        f"Voltage outliers were removed with a Hampel filter (window {c['hampel_window']}, {c['hampel_k']:.0f}σ), "
        f"and residual off-trend points with an iterative robust criterion ({s['resid_k']:.0f}σ, orthogonal residuals). "
        f"Each curve was represented by a cubic B-spline ({s['n_knots']} knots) constrained to be monotonically "
        f"decreasing (and concave unless a bypass-diode step was present in both repeat sweeps; {n_step} of "
        f"{len(fit)} valid sweeps). Near short circuit the spline was replaced by a robust straight line (the shunt "
        f"segment) blended in over {s['blend_width']:.2f} Voc. Pmax was taken from the fitted curve and validated "
        f"against a fourth-order polynomial fit of P(V) around the maximum power point (median deviation "
        f"{np.median(np.abs(eps_br)):.2f} % over the {len(eps_br)} sweeps whose data bracket the peak; "
        f"{np.median(np.abs(eps)):.2f} % over all {len(eps)}). Rs and Rsh were obtained from the slopes of the "
        f"fitted curve at Voc and Isc and from a single-diode model fit, reported only when the fit passed its "
        f"quality gate (nRMSE ≤ {100 * cfg['sdm_gate']['max_nrmse']:.1f} %, {cfg['sdm_gate']['n_min']} ≤ n ≤ "
        f"{cfg['sdm_gate']['n_max']}; {n_sdm} of {len(fit)} sweeps). {n_inv} sweeps failed the validity check "
        f"(incomplete current range or dark panel) and were excluded from the statistics. "
        f"Pmax acceptance: {acc}.\n")
