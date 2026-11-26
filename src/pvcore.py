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
    p = lsq_linear(A, b, bounds=(np.r_[-np.inf, np.zeros(nb - 1)], np.inf)).x
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
    a, w = sc["anchor_v"], sc["blend_width"]
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
    Vs, Is = g * fit["voc"], y * fit["isc"]
    return dict(Vs=Vs, Is=Is, Ps=Vs * Is, shape=shape, end_line=how,
                nrmse=fit["r_c"] if shape == "normal" else fit["r_m"], status=fit["status"])
