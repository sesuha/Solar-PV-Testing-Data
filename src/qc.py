"""Phase 1.2 - sweep-validity gate. Runs before any smoothing.

qc_flag: 'ok' | 'warning' | 'invalid', with reason strings. 'invalid' sweeps are plotted as raw
points with an 'INVALID - re-measure' note, get no fitted curve and stay out of the statistics.
"""
import numpy as np


def qc_sweep(V, I, meta, partner_isc, clean, cfg):
    q = cfg["qc"]
    isc = float(I.max())
    v0 = float(V[np.argmin(I)])                    # V at I = 0 (first point of every sweep)
    voc = float(V.max())
    invalid, warning = [], []

    # current coverage: vs the header Ishort and vs the partner sweep
    cov = q["min_current_coverage"]
    if isc < cov * meta["isc"] or (partner_isc is not None and isc < cov * partner_isc):
        part = f", partner {partner_isc:.3f} A" if partner_isc is not None else ""
        invalid.append(f"current stops at {isc:.3f} A (header Ishort {meta['isc']:.3f} A{part})")
    # dark / covered panel
    if isc < q["min_isc_A"]:
        invalid.append(f"Isc {isc:.3f} A < {q['min_isc_A']} A: panel dark or covered")
    # voltage coverage: at least one clean point below 0.25 Voc
    if not np.any(clean & (V < q["min_v_coverage"] * voc)):
        lo = V[clean].min() if clean.any() else np.nan
        warning.append(f"Isc extrapolated: no clean point below {q['min_v_coverage']:.2f} Voc (lowest {lo:.2f} V)")
    # header consistency
    if abs(v0 - meta["voc"]) / meta["voc"] >= q["max_header_mismatch"]:
        warning.append(f"V(I=0) {v0:.2f} V differs from header Vopen {meta['voc']:.2f} V by >= "
                       f"{100 * q['max_header_mismatch']:.0f} %")
    # pair agreement
    if partner_isc is not None:
        d = abs(isc - partner_isc) / ((isc + partner_isc) / 2)
        if d >= q["max_pair_isc_diff"]:
            warning.append(f"Isc differs from the partner sweep by {100 * d:.0f} % (light changed?)")

    flag = "invalid" if invalid else ("warning" if warning else "ok")
    return flag, invalid + warning
