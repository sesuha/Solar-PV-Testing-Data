"""Phase 6: publication-quality figures (IEEE/Elsevier single column 3.5 in, 8 pt serif)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

S_STYLE = {1: dict(color="#257de0", ls="-", mk="o"), 2: dict(color="#f47230", ls="--", mk="s")}
INK, MUTED, BAD = "#222222", "#777777", "#c0392b"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif", "font.size": 8, "axes.labelsize": 8,
    "axes.linewidth": 0.6, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "legend.fontsize": 7, "legend.frameon": False,
    "lines.linewidth": 1.4, "savefig.dpi": 600, "savefig.bbox": "tight", "pdf.fonttype": 42})

FORMATS = ("pdf", "png")


def _save(fig, path, formats=FORMATS):
    for ext in formats:
        fig.savefig(f"{path}.{ext}")
    plt.close(fig)


def eps_text(m):
    if not m["ref_bracketed"]:
        return r"$\varepsilon_{P_{max}}$ = n/a (ref. not bracketed)"
    return f"$\\varepsilon_{{P_{{max}}}}$ = {m['err_Pmax_pct']:+.2f} %"


def _points(ax, V, I, st, sty, y="I", show=True):
    if not show:
        return
    Y = I if y == "I" else V * I
    u, e, s = st == "used", st == "excluded", st == "spike"
    ax.plot(V[u], Y[u], sty["mk"], ms=2.2, mfc="none", mec=sty["color"], mew=0.5, alpha=0.6, ls="none")
    if e.any():
        ax.plot(V[e], Y[e], "o", ms=3, mfc="none", mec=MUTED, mew=0.6, ls="none")
    if s.any():
        ax.plot(V[s], Y[s], "x", ms=3, color=MUTED, mew=0.6, ls="none")


def _point_handles(st, sty):
    h = [Line2D([], [], ls="none", marker=sty["mk"], ms=3, mfc="none", mec=sty["color"], mew=0.6, label="Measured (used)")]
    if (st == "excluded").any():
        h.append(Line2D([], [], ls="none", marker="o", ms=3.5, mfc="none", mec=MUTED, mew=0.6, label="Excluded (off-trend)"))
    if (st == "spike").any():
        h.append(Line2D([], [], ls="none", marker="x", ms=3.5, color=MUTED, mew=0.6, label="Spike (Hampel)"))
    return h


# ---------------------------------------------------------------- Fig. A
def fig_iv_pv(V, I, c, m, sty, path, sdm=None, show_raw=True, pdf=None, title=None):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.5, 4.4), sharex=True, gridspec_kw=dict(hspace=0.08))
    _points(a1, V, I, c["status"], sty, show=show_raw)
    a1.plot(c["Vs"], c["Is"], color=sty["color"], ls="-")
    _points(a2, V, I, c["status"], sty, y="P", show=show_raw)
    a2.plot(c["Vs"], c["Ps"], color=sty["color"], ls="-")
    if sdm is not None and sdm["valid"]:
        from src.pvcore import i_from_v
        vv = np.linspace(0, m["Voc"], 400)
        ii = i_from_v(vv, sdm["Iph"], sdm["I0"], sdm["Rs"], sdm["Rsh"], sdm["a"])
        k = ii >= 0
        a1.plot(vv[k], ii[k], ls=(0, (3, 2)), lw=0.8, color=INK, label="Single-diode model")
        a1.legend(loc="lower left")
    for ax, y in ((a1, m["Imp"]), (a2, m["Pmax"])):
        ax.plot(m["Vmp"], y, "o", ms=5, color=INK, zorder=5)
        ax.axvline(m["Vmp"], color=MUTED, lw=0.6, ls=":")
    a1.axhline(m["Imp"], color=MUTED, lw=0.6, ls=":")
    a1.annotate("MPP", (m["Vmp"], m["Imp"]), xytext=(-22, -14), textcoords="offset points", fontsize=7)
    txt = (f"$V_{{oc}}$ = {m['Voc']:.2f} V\n$I_{{sc}}$ = {m['Isc']:.3f} A\n$V_{{mp}}$ = {m['Vmp']:.2f} V\n"
           f"$I_{{mp}}$ = {m['Imp']:.3f} A\n$P_{{max}}$ = {m['Pmax']:.2f} W\nFF = {m['FF']:.3f}\n" + eps_text(m))
    a2.text(0.03, 0.95, txt, transform=a2.transAxes, va="top", fontsize=7, linespacing=1.3)
    a1.set_ylabel("Current, $I$ (A)")
    a2.set_ylabel("Power, $P$ (W)")
    a2.set_xlabel("Voltage, $V$ (V)")
    for ax, lab in ((a1, "(a)"), (a2, "(b)")):
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
        ax.text(0.97, 0.93, lab, transform=ax.transAxes, ha="right", fontweight="bold")
    if pdf is not None:                       # multi-page appendix PDF (Phase 9)
        if title:
            fig.suptitle(title, fontsize=8, y=0.995)
        pdf.savefig(fig)
    if path:
        _save(fig, path)
    else:
        plt.close(fig)


# ---------------------------------------------------------------- Fig. A2
def fig_iv_pv_combined(V, I, c, m, sty, path, show_raw=True):
    """I-V and P-V on one graph (I left axis, P right axis)."""
    PCOL = "#444444"
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax2 = ax.twinx()
    st = c["status"]
    u, e, s = st == "used", st == "excluded", st == "spike"
    if show_raw:
        ax.plot(V[u], I[u], sty["mk"], ms=2.2, mfc="none", mec=sty["color"], mew=0.5, alpha=0.6, ls="none")
        ax2.plot(V[u], (V * I)[u], "^", ms=2.2, mfc="none", mec=PCOL, mew=0.5, alpha=0.45, ls="none")
        for a, Y in ((ax, I), (ax2, V * I)):
            if e.any():
                a.plot(V[e], Y[e], "o", ms=3, mfc="none", mec=MUTED, mew=0.6, ls="none")
            if s.any():
                a.plot(V[s], Y[s], "x", ms=3, color=MUTED, mew=0.6, ls="none")
    hI, = ax.plot(c["Vs"], c["Is"], color=sty["color"], ls="-", label="Current, $I$ (left axis)")
    hP, = ax2.plot(c["Vs"], c["Ps"], color=PCOL, ls="--", label="Power, $P$ (right axis)")
    ax.plot(m["Vmp"], m["Imp"], "o", ms=5, color=INK, zorder=6)
    ax2.plot(m["Vmp"], m["Pmax"], "o", ms=5, color=INK, zorder=6)
    ax.axvline(m["Vmp"], color=MUTED, lw=0.6, ls=":")
    ax.plot([0, m["Vmp"]], [m["Imp"]] * 2, ":", color=MUTED, lw=0.6)
    ax2.plot([m["Vmp"], 1.05 * m["Voc"]], [m["Pmax"]] * 2, ":", color=MUTED, lw=0.6)
    ax2.annotate("MPP", (m["Vmp"], m["Pmax"]), xytext=(-24, 6), textcoords="offset points", fontsize=7)
    ax.set_xlim(0, 1.05 * m["Voc"])
    ax.set_ylim(0, 1.18 * max(m["Isc"], np.nanmax(I[u])))
    ax2.set_ylim(0, 1.18 * max(m["Pmax"], np.nanmax((V * I)[u])))
    ax.set_xlabel("Voltage, $V$ (V)")
    ax.set_ylabel("Current, $I$ (A)")
    ax2.set_ylabel("Power, $P$ (W)")
    ax.tick_params(right=False)
    ax2.tick_params(direction="in")
    ax.legend(handles=[hI, hP], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              fontsize=7, handlelength=2.2, borderaxespad=0.3)
    txt = (f"$V_{{oc}}$ = {m['Voc']:.2f} V    $I_{{sc}}$ = {m['Isc']:.3f} A    "
           f"$V_{{mp}}$ = {m['Vmp']:.2f} V    $I_{{mp}}$ = {m['Imp']:.3f} A\n"
           f"$P_{{max}}$ = {m['Pmax']:.2f} W    FF = {m['FF']:.3f}    " + eps_text(m))
    ax.text(0.5, -0.19, txt, transform=ax.transAxes, ha="center", va="top", fontsize=7, linespacing=1.5)
    _save(fig, path)


# ---------------------------------------------------------------- Fig. B (Rs)
def _fmt_r(x, nd=2):
    return "∞" if (not np.isfinite(x) or x > 1e6) else f"{x:.{nd}f}"


def fig_rs(c, m, g, sl, sdm, sty, path):
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.plot(c["Vs"], c["Is"], color=sty["color"])
    ax.plot(m["Vmp"], m["Imp"], "o", ms=5, color=INK)
    ax.plot([m["Vmp"]] * 2, [0, m["Imp"]], ":", color=MUTED, lw=0.7)
    ax.plot([0, m["Vmp"]], [m["Imp"]] * 2, ":", color=MUTED, lw=0.7)
    ax.annotate(r"$I_{mp}$", (0.55 * m["Voc"], m["Imp"]), xytext=(3, -9), textcoords="offset points", fontsize=7)
    ax.annotate(r"$V_{mp}$", (m["Vmp"], 0), xytext=(3, 3), textcoords="offset points", fontsize=7)
    y0 = 0.12 * m["Isc"]
    ax.annotate("", (m["Voc"], y0), (m["Vmp"], y0), arrowprops=dict(arrowstyle="<->", lw=0.7, color=INK))
    ax.text((m["Vmp"] + m["Voc"]) / 2, y0 * 1.25, r"$\Delta V$", ha="center")
    ii = np.linspace(0, 0.45 * m["Isc"], 50)
    ax.plot(np.polyval(sl["tan_rs"], ii), ii, "--", color=INK, lw=0.8)
    sd = f"{sdm['Rs']:.2f} Ω" if sdm["valid"] else "n/a (fit rejected)"
    ax.text(0.03, 0.05, f"$\\Delta V = V_{{oc}}-V_{{mp}}$ = {g['dV']:.2f} V\n"
            f"$R_s \\approx \\Delta V/I_{{mp}}$ = {g['Rs_geo']:.2f} Ω\n"
            f"$R_s = -dV/dI|_{{V_{{oc}}}}$ = {_fmt_r(sl['Rs_slope'])} Ω\n$R_s$ (single-diode) = {sd}",
            transform=ax.transAxes, fontsize=7, linespacing=1.35,
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    ax.set_xlim(0.55 * m["Voc"], 1.04 * m["Voc"])
    ax.set_ylim(0, 1.08 * m["Isc"])
    ax.set_xlabel("Voltage, $V$ (V)")
    ax.set_ylabel("Current, $I$ (A)")
    _save(fig, path)


# ---------------------------------------------------------------- Fig. C (Rsh)
def fig_rsh(c, m, g, sl, sdm, sty, path):
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.plot(c["Vs"], c["Is"], color=sty["color"])
    ax.plot(m["Vmp"], m["Imp"], "o", ms=5, color=INK)
    ax.axhline(m["Isc"], color=MUTED, lw=0.7, ls=":")
    ax.axhline(m["Imp"], color=MUTED, lw=0.7, ls=":")
    ax.plot([m["Vmp"]] * 2, [0, m["Isc"]], ":", color=MUTED, lw=0.7)
    ax.annotate(r"$I_{sc}$", (1.0 * m["Voc"], m["Isc"]), xytext=(-12, 2), textcoords="offset points", fontsize=7)
    ax.annotate(r"$I_{mp}$", (1.0 * m["Voc"], m["Imp"]), xytext=(-12, 2), textcoords="offset points", fontsize=7)
    x0 = 0.5 * m["Vmp"]
    ax.annotate("", (x0, m["Isc"]), (x0, m["Imp"]), arrowprops=dict(arrowstyle="<->", lw=0.7, color=INK))
    ax.text(x0 * 1.04, m["Imp"] + 0.25 * (m["Isc"] - m["Imp"]), r"$\Delta I$", va="center")   # below the curve
    vv = np.linspace(0, 0.8 * m["Vmp"], 50)
    ax.plot(vv, np.polyval(sl["tan_rsh"], vv), "--", color=INK, lw=0.8)
    sd = f"{sdm['Rsh']:.1f} Ω" if sdm["valid"] else "n/a (fit rejected)"
    ax.text(0.03, 0.05, f"$\\Delta I = I_{{sc}}-I_{{mp}}$ = {g['dI']:.3f} A\n"
            f"$R_{{sh}} \\approx V_{{mp}}/\\Delta I$ = {_fmt_r(g['Rsh_geo'], 1)} Ω\n"
            f"$R_{{sh}} = -dV/dI|_{{I_{{sc}}}}$ = {_fmt_r(sl['Rsh_slope'], 1)} Ω\n$R_{{sh}}$ (single-diode) = {sd}",
            transform=ax.transAxes, fontsize=7, linespacing=1.35,
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    ax.set_xlim(0, 1.05 * m["Voc"])
    ax.set_ylim(0.6 * m["Imp"], 1.06 * m["Isc"])
    ax.set_xlabel("Voltage, $V$ (V)")
    ax.set_ylabel("Current, $I$ (A)")
    _save(fig, path)


# ---------------------------------------------------------------- invalid sweep
def fig_invalid(V, I, status, reason, sty, path, combined=False):
    """Invalid sweeps: raw points only, no fitted curve, red note."""
    if combined:
        fig, ax = plt.subplots(figsize=(3.5, 3.0))
        axes = [(ax, "I")]
    else:
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.5, 4.4), sharex=True, gridspec_kw=dict(hspace=0.08))
        axes = [(a1, "I"), (a2, "P")]
    for ax, y in axes:
        _points(ax, V, I, status, sty, y=y)
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
        ax.set_ylabel("Current, $I$ (A)" if y == "I" else "Power, $P$ (W)")
    axes[-1][0].set_xlabel("Voltage, $V$ (V)")
    import textwrap
    note = "\n".join(textwrap.fill(r, 42) for r in reason.split("; "))
    axes[0][0].text(0.03, 0.06, "INVALID – re-measure\n" + note, transform=axes[0][0].transAxes,
                    color=BAD, fontsize=7, va="bottom", linespacing=1.3)
    _save(fig, path)


# ---------------------------------------------------------------- Fig. D (panel overlay)
def fig_panel_overlay(pair, path, show_raw=True):
    """pair = [(V, I, curve_or_None, metrics_or_None, status) for S1, S2]."""
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for k, (V, I, c, m, st) in enumerate(pair, start=1):
        sty = S_STYLE[k]
        _points(ax, V, I, st, sty, show=show_raw)
        if c is None:
            ax.plot([], [], color=sty["color"], ls=sty["ls"], label=f"Sweep S{k}: INVALID (raw points only)")
            continue
        ax.plot(c["Vs"], c["Is"], color=sty["color"], ls=sty["ls"], label=f"Sweep S{k}: $P_{{max}}$ = {m['Pmax']:.2f} W")
        ax.plot(m["Vmp"], m["Imp"], sty["mk"], ms=5, color=sty["color"], mec=INK, mew=0.6)
    ms = [p[3] for p in pair]
    if all(x is not None for x in ms):
        p1, p2 = ms[0]["Pmax"], ms[1]["Pmax"]
        title = f"$\\Delta P_{{max}}$(S2–S1) = {100 * (p2 - p1) / ((p1 + p2) / 2):+.2f} %"
    else:
        title = r"$\Delta P_{max}$(S2–S1) = n/a (invalid sweep)"
    ax.legend(loc="lower left", title=title, title_fontsize=7)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.set_xlabel("Voltage, $V$ (V)")
    ax.set_ylabel("Current, $I$ (A)")
    _save(fig, path)


# ---------------------------------------------------------------- Fig. E (Pmax summary)
def fig_summary_pmax(rows, path, tag):
    """Dot plot of Pmax per panel, S1/S2 side by side, +/- u(Pmax) (k = 1). Invalid sweeps hollow
    at their raw max P."""
    panels = sorted({r["panel"] for r in rows})
    fig, ax = plt.subplots(figsize=(7.16, 2.8))
    for r in rows:
        k = r["pos"]
        sty = S_STYLE[k]
        x = r["panel"] + (-0.15 if k == 1 else 0.15)
        if r["qc"] == "invalid":
            ax.plot(x, r["Pmax_raw"], sty["mk"], ms=4.5, mfc="white", mec=sty["color"], mew=0.9)
        else:
            ax.errorbar(x, r["Pmax"], yerr=r["u_Pmax"], fmt=sty["mk"], ms=4, color=sty["color"],
                        mec=INK, mew=0.4, elinewidth=0.8, capsize=2)
    hs = [Line2D([], [], ls="none", marker=S_STYLE[k]["mk"], ms=4, color=S_STYLE[k]["color"], mec=INK, mew=0.4,
                 label=f"S{k}") for k in (1, 2)]
    hs.append(Line2D([], [], ls="none", marker="o", ms=4.5, mfc="white", mec=MUTED, label="Invalid (raw max P)"))
    ax.legend(handles=hs, loc="upper left", ncol=3)
    ax.set_xticks(panels)
    ax.set_xticklabels([f"P{p}" for p in panels])
    ax.set_xlim(min(panels) - 0.6, max(panels) + 0.6)
    ax.set_ylim(0, None)
    ax.set_ylabel("$P_{max}$ (W)")
    ax.set_xlabel(f"Panel ({tag}); error bars: ±u($P_{{max}}$), k = 1")
    _save(fig, path)


# ---------------------------------------------------------------- Fig. F (contact sheet)
def fig_contact_sheet(panels, path, ncols=5):
    """QA only. panels = [(label, [(V, I, c_or_None), (V, I, c_or_None)], shape)]."""
    n = len(panels)
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 1.8))
    axs = np.atleast_1d(axs).ravel()
    for ax in axs[n:]:
        ax.axis("off")
    for ax, (label, sweeps, shape) in zip(axs, panels):
        for k, (V, I, c) in enumerate(sweeps, start=1):
            sty = S_STYLE[k]
            ax.plot(V, I, sty["mk"], ms=1.2, mfc="none", mec=sty["color"], mew=0.3, alpha=0.5, ls="none")
            if c is not None:
                ax.plot(c["Vs"], c["Is"], color=sty["color"], ls=sty["ls"], lw=1.0)
            else:
                ax.text(0.04, 0.08 + 0.12 * (k - 1), f"S{k} INVALID", transform=ax.transAxes, color=BAD, fontsize=6)
        ax.set_title(f"{label} ({shape})", fontsize=7)
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
        ax.tick_params(labelsize=6)
    fig.tight_layout()
    _save(fig, path)


def open_multipage(path):
    return PdfPages(path)
