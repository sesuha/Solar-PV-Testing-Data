"""Phases 2-5: point cleaning, shape-constrained smoothing, metrics, Rs/Rsh."""
import numpy as np
from scipy.ndimage import median_filter


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
    status = np.where(keep, "used", "spike").astype(object)
    return status
