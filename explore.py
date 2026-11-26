# %% [markdown]
# # Tune one sweep
# Run cell by cell in VS Code (Python + Jupyter extensions: "Run Cell" above each `# %%`).
# Change FILE / PANEL and the thresholds, re-run, compare. Copy final values into config.yaml.

# %%
import copy

import matplotlib.pyplot as plt
import numpy as np

from src.main import analyse_panel
from src.parse import load_all, load_config

FILE, PANEL = "1 Nov", 9
base = load_config()
data = load_all(base)
pair = base["panels"][FILE][PANEL - 1]

# %% thresholds to try (config.yaml keys)
cfg = copy.deepcopy(base)
cfg["smoothing"].update(n_knots=30, **{"lambda": 1e-3}, resid_k=4.0, step_ratio=1.6)
cfg["cleaning"].update(hampel_window=7, hampel_k=3.0)

recs = analyse_panel(FILE, PANEL, pair, {n: data[FILE][n] for n in pair}, cfg)
for r in recs:
    m = r.get("m")
    print(f"S{r['pos']} #{r['sample']}: qc {r['qc']}, shape {r.get('shape')}, used/spike/excl "
          f"{r['n_used']}/{r['n_spike']}/{r['n_excluded']}"
          + (f", Pmax {m['Pmax']:.2f} W, eps {m['err_Pmax_pct']:+.2f} % ({r['accept']})" if m else ""))

# %% plot both sweeps: points by status + smooth curve
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
for r, col in zip(recs, ("#257de0", "#f47230")):
    st = r["status"]
    for ax, Y in ((axs[0], r["I"]), (axs[1], r["V"] * r["I"])):
        for s, mk in (("used", "o"), ("excluded", "s"), ("spike", "x")):
            k = st == s
            ax.plot(r["V"][k], Y[k], mk, ms=3, mfc="none", color=col if s == "used" else "grey", ls="none")
    if r.get("c"):
        axs[0].plot(r["c"]["Vs"], r["c"]["Is"], color=col)
        axs[1].plot(r["c"]["Vs"], r["c"]["Ps"], color=col)
for ax in axs:
    ax.set_xlim(0)
    ax.set_ylim(0)
axs[0].set_title(f"{FILE} panel {PANEL}: I-V")
axs[1].set_title("P-V")
plt.show()

# %% optional: live slider for the step ratio (needs `pip install ipywidgets` in a notebook kernel)
try:
    import ipywidgets as w

    def _run(step_ratio=1.6, n_knots=30):
        c2 = copy.deepcopy(cfg)
        c2["smoothing"].update(step_ratio=step_ratio, n_knots=n_knots)
        rr = analyse_panel(FILE, PANEL, pair, {n: data[FILE][n] for n in pair}, c2)
        print([(r["sample"], r.get("shape"), round(r["m"]["Pmax"], 2) if r.get("m") else None) for r in rr])

    w.interact(_run, step_ratio=(1.2, 2.5, 0.05), n_knots=(12, 45, 3))
except ImportError:
    print("ipywidgets not installed - edit the values in the thresholds cell instead")
