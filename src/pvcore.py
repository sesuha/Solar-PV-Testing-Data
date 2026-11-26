"""Phases 2-5: point cleaning, shape-constrained smoothing, metrics, Rs/Rsh."""
import numpy as np
from scipy.interpolate import BSpline
from scipy.ndimage import median_filter
from scipy.optimize import least_squares, lsq_linear


# ---------------------------------------------------------------- Phase 2: cleaning
def hampel_V(V, win=7, k=3.0, floor_frac=0.01):
    """True = keep. A point is a spike if it differs from the rolling median of its neighbours
    (recording order) by more than max(k * local MAD, floor_frac * Vmax). A local median recovers
    within a few points, unlike the old running-minimum filter."""
    med = median_filter(V, size=win, mode="nearest")
    mad = 1.4826 * median_filter(np.abs(V - med), size=win, mode="nearest")
    return np.abs(V - med) <= np.maximum(k * mad, floor_frac * V.max())


def clean_points(V, cfg):
    c = cfg["cleaning"]
    keep = hampel_V(V, c["hampel_window"], c["hampel_k"], c["hampel_floor_frac"]) & (V > 0)
    return np.where(keep, "used", "spike").astype(object)


# ---------------------------------------------------------------- Phase 3: smoothing engine
def shape_spline(v, i, concave=True, n_knots=18, lam=1e-3):
    """Cubic B-spline in normalised v. The coefficients are reparametrised so that 'decreasing'
    (and optionally 'concave') are plain non-negativity bounds; a second-difference penalty adds
    smoothness. Bounded linear least squares -> deterministic, cannot wiggle or overshoot."""
    deg = 3
    t = np.r_[[0] * deg, np.linspace(0, 1, n_knots), [1] * deg]
    nb = len(t) - deg - 1
    B = BSpline.design_matrix(np.clip(v, 0, 1), t, deg).toarray()
    L = np.tril(np.ones((nb, nb)))[:, 1:]               # c_k = c0 - sum d_j, d >= 0 -> decreasing
    if concave:
        L = L @ np.tril(np.ones((nb - 1, nb - 1)))       # d_j = sum e_m, e >= 0 -> concave
    T = np.hstack([np.ones((nb, 1)), -L])
    D = np.diff(np.eye(nb), 2, axis=0) @ T
    A = np.vstack([B @ T, np.sqrt(lam * len(v)) * D])
    b = np.r_[i, np.zeros(D.shape[0])]
    # bvls: exact active-set solver. The default trf stops early on this badly scaled
    # (cumulative-sum) basis - at 30 knots it ignored five clean points on 1 Nov #6.
    p = lsq_linear(A, b, bounds=(np.r_[-np.inf, np.zeros(nb - 1)], np.inf), method="bvls").x
    return BSpline(t, T @ p, deg, extrapolate=False)


def _eval(f, v):
    return np.nan_to_num(f(np.clip(v, 0, 1)))


def ortho_resid(f, v, i):
    """Orthogonal residual in normalised (v, i) coordinates."""
    g = np.linspace(0, 1, 600)
    y = _eval(f, g)
    s = np.interp(np.clip(v, 0, 1), g, np.gradient(y, g))
    return (i - _eval(f, v)) / np.sqrt(1 + s ** 2)


def robust_spline(v, i, concave, sc):
    """3.2: iterative robust exclusion of off-trend points (|r| > max(floor, k*sigma)), <= 4 passes."""
    ex = np.zeros(len(v), bool)
    for _ in range(4):
        f = shape_spline(v[~ex], i[~ex], concave, sc["n_knots"], sc["lambda"])
        r = ortho_resid(f, v, i)
        sig = 1.4826 * np.median(np.abs(r[~ex] - np.median(r[~ex])))
        new = np.abs(r) > max(sc["resid_floor"], sc["resid_k"] * sig)
        if (new == ex).all():
            break
        ex = new
    return f, ex


def rms(f, v, i):
    return float(np.sqrt(np.mean(ortho_resid(f, v, i) ** 2)))


def fit_sweep(V, I, status, voc_hdr, isc_hdr, cfg):
    """3.2-3.3 per sweep: robust MONOTONE fit first (flexible enough to keep a real step, so step
    points are not wrongly excluded), then monotone+concave on the remaining clean points.
    The final shape is decided per PANEL by decide_shape (pair rule)."""
    sc = cfg["smoothing"]
    voc, isc = max(V.max(), voc_hdr), max(I.max(), isc_hdr)
    k = status == "used"
    v, i = V[k] / voc, I[k] / isc
    fm, ex = robust_spline(v, i, False, sc)
    fc = shape_spline(v[~ex], i[~ex], True, sc["n_knots"], sc["lambda"])
    r_m, r_c = rms(fm, v[~ex], i[~ex]), rms(fc, v[~ex], i[~ex])
    # step location: where the flexible fit bends the 'wrong' way (most convex). A bypass-diode
    # step sits mid-curve, so the Isc tail (< 0.2 Voc, replaced by the end line) and the Voc end
    # are not searched.
    g = np.linspace(0.2, 0.9, 700)
    d2 = fm.derivative(2)(g)
    st = status.copy()
    st[np.flatnonzero(k)[ex]] = "excluded"
    return dict(voc=voc, isc=isc, v=v, i=i, ex=ex, fm=fm, fc=fc, r_m=r_m, r_c=r_c,
                flagged=bool(r_c > sc["step_ratio"] * r_m), step_V=float(g[np.argmax(d2)] * voc), status=st)


def decide_shape(f1, f2, cfg):
    """3.3 pair rule: the PANEL is 'stepped' only if both sweeps are flagged, or one is flagged and
    its step voltage matches the other sweep's within step_match_V. A feature in both repeat
    sweeps is physical; a feature in only one sweep is noise. f1/f2: fit_sweep() output, or None
    for an invalid sweep (then nothing can be confirmed -> 'normal')."""
    tol = cfg["smoothing"]["step_match_V"]
    if f1 is None or f2 is None:
        return "normal"
    if f1["flagged"] and f2["flagged"]:
        return "stepped"
    if f1["flagged"] or f2["flagged"]:
        return "stepped" if abs(f1["step_V"] - f2["step_V"]) <= tol else "normal"
    return "normal"


def _end_line(f, v, i, a, w):
    """3.4: straight line i = alpha - beta*v (0 <= beta <= spline slope) for the Isc end - the
    physical Rsh segment. The line is pinned to the spline at the middle of the blend zone, so it
    joins without a step, and only its slope is fitted (robust, soft_l1) to the clean points
    below 'a'. Capping beta at the spline's own slope there keeps a concave curve concave.
    With < 2 points below 'a' the line is the spline's tangent there."""
    vb = a + w / 2
    yb = float(_eval(f, np.array([vb]))[0])
    g = np.linspace(a, a + w, 50)
    s_cap = max(0.0, -np.polyfit(g, _eval(f, g), 1)[0])
    m = v < a
    if m.sum() >= 2 and np.ptp(v[m]) >= 0.02 and s_cap > 0:
        res = least_squares(lambda x: yb + x[0] * (vb - v[m]) - i[m], [0.5 * s_cap],
                            bounds=([0.0], [s_cap]), loss="soft_l1", f_scale=0.01)
        beta, how = float(res.x[0]), "fit"
    else:
        beta, how = s_cap, "tangent"
    return yb + beta * vb, beta, how


def build_curve(fit, shape, cfg):
    """Final smooth curve: the chosen spline, the Isc-end line blended in with a cosine weight over
    [anchor_v, anchor_v + blend_width], clipped at the Voc root so it never goes below I = 0."""
    sc = cfg["smoothing"]
    f = fit["fm"] if shape == "stepped" else fit["fc"]
    v, i = fit["v"][~fit["ex"]], fit["i"][~fit["ex"]]
    w = sc["blend_width"]
    # If the sweep has no clean points near Isc (QC 'Isc extrapolated'), the spline below the
    # lowest point is unconstrained; start the end line at the lowest clean point instead, so its
    # level and slope come from where data exist (1 Nov #6 has nothing below 7 V).
    a = max(sc["anchor_v"], float(v.min()) + w)
    alpha, beta, how = _end_line(f, v, i, a, w)
    g = np.linspace(0, 1, sc["n_curve_points"])
    y = _eval(f, g)
    wgt = np.where(g <= a, 1.0, np.where(g >= a + w, 0.0, 0.5 * (1 + np.cos(np.pi * (g - a) / w))))
    y = wgt * (alpha - beta * g) + (1 - wgt) * y
    y = np.minimum.accumulate(y)                          # guard: never rises
    j = np.flatnonzero(y <= 0)
    if j.size and j[0] > 0:                               # clip at the Voc root
        k = j[0]
        g0 = g[k - 1] + (g[k] - g[k - 1]) * y[k - 1] / (y[k - 1] - y[k])
        g, y = np.r_[g[:k], g0], np.r_[y[:k], 0.0]
    elif 0 < y[-1] <= 0.01:                               # ends within 1 % of Isc at the measured
        y = np.r_[y[:-1], 0.0]                            # I = 0 point: that point is Voc
    Vs, Is = g * fit["voc"], y * fit["isc"]
    return dict(Vs=Vs, Is=Is, Ps=Vs * Is, shape=shape, end_line=how, reaches_zero=bool(y[-1] <= 0),
                nrmse=fit["r_c"] if shape == "normal" else fit["r_m"], status=fit["status"])


# ---------------------------------------------------------------- Phase 4: metrics and Pmax error
def pmax_local_poly(V, I, frac=0.10, order=4, min_side=3):
    """IEC 60904-1 practice: Pmax from a polynomial fitted to P(V) around the MPP, instead of
    trusting one sample (a +V spike gives a +P spike).

    Returns (Pmax_ref, bracketed). The fit is only a valid reference when the data BRACKET the
    peak: the polynomial maximum must be inside the window (not at an edge) with >= min_side
    points on each side. Current-swept sweeps often have no points just left of a sharp knee;
    a one-sided 4th-order fit then shoots up at the window edge (18 Oct #11: 13.90 W, above
    every measured point)."""
    P = V * I
    j = np.argmax(P)
    m = np.abs(V - V[j]) <= frac * V.max()
    deg = min(order, m.sum() - 1)
    c = np.polyfit(V[m], P[m], deg)
    g = np.linspace(V[m].min(), V[m].max(), 400)
    y = np.polyval(c, g)
    k = int(np.argmax(y))
    vpk = g[k]
    interior = 0.03 * len(g) < k < 0.97 * len(g)
    sides = (V[m] < vpk).sum() >= min_side and (V[m] > vpk).sum() >= min_side
    # standard error of the fitted value at the peak: sigma * sqrt(leverage)
    x = (V[m] - vpk) / max(np.ptp(V[m]), 1e-9)
    X = np.vander(x, deg + 1)
    dof = max(m.sum() - (deg + 1), 1)
    sigma = np.sqrt(np.sum((P[m] - np.polyval(c, V[m])) ** 2) / dof)
    h = np.linalg.pinv(X.T @ X)[-1, -1]                    # x0 = [0, ..., 0, 1] at the peak
    return float(y[k]), bool(interior and sides), float(sigma * np.sqrt(max(h, 0.0)))


def metrics(c, V, I, P, cfg):
    mc = cfg["metrics"]
    Vs, Is, Ps = c["Vs"], c["Is"], c["Ps"]
    j = int(np.argmax(Ps))
    used = c["status"] == "used"
    pref, bracketed, se = pmax_local_poly(V[used], I[used], mc["pref_window_frac"], mc["pref_poly_order"])
    out = dict(Voc=float(Vs[-1]), Isc=float(Is[0]), Vmp=float(Vs[j]), Imp=float(Is[j]), Pmax=float(Ps[j]),
               Pmax_raw=float(P.max()), Pmax_ref=pref, ref_bracketed=bracketed, Pmax_ref_se=se,
               Vmax_raw=float(V.max()), Imax_raw=float(I.max()))
    out["FF"] = out["Pmax"] / (out["Voc"] * out["Isc"])
    out["err_Pmax_pct"] = 100 * (out["Pmax"] - pref) / pref
    out["err_Pmax_raw_pct"] = 100 * (out["Pmax"] - P.max()) / P.max()
    out["u_Pmax"] = u_pmax(out, cfg)
    return out


def acceptance(m, shape, cfg):
    """Pmax fidelity verdict. 'pass' if |eps| <= tolerance (1 % normal, 2.5 % stepped) OR within
    twice the reference's own standard error (a polynomial fitted to scattered points is itself
    uncertain); 'low confidence' otherwise; 'n/a' if the reference does not bracket the peak."""
    if not m["ref_bracketed"]:
        return "n/a (reference not bracketed)"
    acc = cfg["acceptance"]
    tol = acc["err_pmax_stepped_pct"] if shape == "stepped" else acc["err_pmax_normal_pct"]
    lim = max(tol, 200 * m["Pmax_ref_se"] / m["Pmax_ref"])
    return "pass" if abs(m["err_Pmax_pct"]) <= lim else "low confidence"


def _limit(x, full, spec):
    off = spec["offset_low"] if x < spec["split"] else spec["offset_high"]
    return spec["rdg_pct"] / 100 * x + spec["fs_pct"] / 100 * full + off


def u_pmax(m, cfg):
    """4.3: standard uncertainty of Pmax (k = 1) from the PROVA 200A accuracy limits at the MPP,
    limits treated as rectangular (/sqrt 3): u(Pmax) = Pmax * sqrt(u_V^2 + u_I^2), relative."""
    ins = cfg["instrument"]
    uV = _limit(m["Vmp"], m["Voc"], ins["V"]) / np.sqrt(3) / m["Vmp"]
    uI = _limit(m["Imp"], m["Isc"], ins["I"]) / np.sqrt(3) / m["Imp"]
    return float(m["Pmax"] * np.hypot(uV, uI))


# ---------------------------------------------------------------- Phase 5: Rs and Rsh
def rs_rsh_geometric(m):
    """5.1 (notebook sketch): Rs ~ dV/Imp, Rsh ~ Vmp/dI. Quick upper-bound approximations."""
    dV, dI = m["Voc"] - m["Vmp"], m["Isc"] - m["Imp"]
    return dict(dV=dV, dI=dI, Rs_geo=dV / m["Imp"], Rsh_geo=m["Vmp"] / dI if dI > 0 else np.inf)


def rs_rsh_slope(c, frac_rs=0.05, frac_rsh=0.20):
    """5.2: -dV/dI of the SMOOTH curve near Voc (I <= frac_rs*Isc) and near Isc (V <= frac_rsh*Voc)."""
    Vs, Is = c["Vs"], c["Is"]
    voc, isc = Vs[-1], Is[0]
    a = Is <= frac_rs * isc
    if a.sum() < 3:
        a = np.arange(len(Is)) >= len(Is) - 3
    b = Vs <= frac_rsh * voc
    ps = np.polyfit(Is[a], Vs[a], 1)
    ph = np.polyfit(Vs[b], Is[b], 1)
    return dict(Rs_slope=float(-ps[0]), Rsh_slope=float(-1 / ph[0]) if ph[0] < 0 else np.inf,
                tan_rs=ps, tan_rsh=ph)


from scipy.special import wrightomega  # noqa: E402


def i_from_v(V, Iph, I0, Rs, Rsh, a):
    """Single-diode model, explicit Lambert-W form; wrightomega avoids overflow."""
    z = np.log(Rs * I0 * Rsh / (a * (Rs + Rsh))) + Rsh * (Rs * (Iph + I0) + V) / (a * (Rs + Rsh))
    return (Rsh * (Iph + I0) - V) / (Rs + Rsh) - (a / Rs) * np.real(wrightomega(z))


def fit_sdm(V, I, voc, isc, cfg, T=298.15):
    """5.3: robust single-diode fit on the used points, with a quality gate. Module temperature was
    not recorded, so T = 25 C is assumed for the thermal voltage (affects n only)."""
    ns, gate = cfg["module"]["cells_in_series"], cfg["sdm_gate"]
    vth = 1.380649e-23 * T / 1.602176634e-19
    a0 = ns * 1.3 * vth
    x0 = [isc, np.log(isc / np.expm1(voc / a0)), 0.5, 200.0, a0]
    lo = [0.8 * isc, -40, 1e-3, 1.0, 0.5 * ns * vth]
    hi = [1.2 * isc, -2, 20.0, 1e5, 3.0 * ns * vth]
    x0 = np.clip(x0, np.array(lo) + 1e-9, np.array(hi) - 1e-9)

    def res(x):
        r = (i_from_v(V, x[0], np.exp(x[1]), x[2], x[3], x[4]) - I) / isc
        return np.nan_to_num(r, nan=10.0, posinf=10.0, neginf=-10.0)

    s = least_squares(res, x0, bounds=(lo, hi), loss="soft_l1", f_scale=0.01, x_scale="jac")
    Iph, lnI0, Rs, Rsh, a = s.x
    p = dict(Iph=Iph, I0=np.exp(lnI0), Rs=Rs, Rsh=Rsh, a=a, n=a / (ns * vth),
             nrmse=float(np.sqrt(np.mean(s.fun ** 2))))
    reasons = []
    if not s.success:
        reasons.append("did not converge")
    if p["nrmse"] > gate["max_nrmse"]:
        reasons.append(f"nRMSE {100 * p['nrmse']:.1f} % > {100 * gate['max_nrmse']:.1f} %")
    if not gate["n_min"] <= p["n"] <= gate["n_max"]:
        reasons.append(f"n {p['n']:.2f} outside {gate['n_min']}-{gate['n_max']}")
    if Rs <= 0.01:
        reasons.append("Rs at lower bound")
    if Rsh >= 0.99e5:
        reasons.append("Rsh at upper bound")
    p["valid"] = not reasons
    p["reject_reason"] = "; ".join(reasons)
    return p
