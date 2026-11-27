# PV I–V smoothing, Rs/Rsh extraction and publication plots

Implements `PV_IV_Smooth_Plot_Plan_v2.md`: 28 panels × 2 sweeps = 56 PROVA 200A I–V curves.

## Run (VS Code: F5 → "Run pipeline" / "One panel" / "Tests")

```
uv venv .venv --python 3.12 && uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.venv/Scripts/python.exe -m src.main                              # everything, ~3 min
.venv/Scripts/python.exe -m src.main --file "1 Nov" --panel 4     # one panel
.venv/Scripts/python.exe -m src.main --show-sdm --no-raw          # SDM curve on Fig. A / hide raw points
.venv/Scripts/python.exe -m pytest -q                             # 28 tests
```
Tune thresholds only in `config.yaml`; try them on one sweep with `explore.py` (`# %%` cells).

## Outputs (`outputs/`)

| | |
|---|---|
| `figures/<file>/Panel_<nn>/S<n>_IVPV` | Fig. A – I–V (a) and P–V (b), shared V axis, ε_Pmax |
| `…/S<n>_IVPV_combined` | Fig. A2 – I and P on one graph (twin axes) |
| `…/S<n>_Rs`, `…/S<n>_Rsh` | Fig. B / C – ΔV, ΔI construction; geometric, slope and single-diode Rs/Rsh |
| `…/overlay` | Fig. D – S1 vs S2, ΔPmax(S2–S1) % |
| `figures/<file>/summary_pmax` | Fig. E – Pmax per panel ± u(Pmax) |
| `figures/<file>/contact_sheet`, `figures/contact_sheet_all` | Fig. F – QA only |
| `figures/<file>/appendix_IVPV.pdf` | all Fig. A in one multi-page PDF |
| `summary.xlsx` | Per Sweep · Per Panel (notebook table) · Point Status |
| `tables/panel_table.tex`, `methods.md` | paper table (booktabs) and Methods paragraph with the actual numbers |

Every figure is saved as vector PDF and 600-dpi PNG. Invalid sweeps are plotted as raw points with an
"INVALID – re-measure" note and no curve.

## Where the implementation differs from the plan text (each found while validating)

| Plan | Implemented | Why |
|---|---|---|
| `n_knots: 18`, `step_ratio: 1.4` | **30**, **1.6** (config) | 18 knots rounded sharp knees off (Pmax ~1.5 % low on 1 Nov #11; median ε 0.42 % → 0.26 %). More flexibility raises the RMSE ratio of normal sweeps to 1.4–1.6, so the plan's own tuning knob `step_ratio → 1.6` restores 1 Nov #11/#12 = normal; real steps are 1.7–10. |
| `lsq_linear` default (`trf`) | `method="bvls"` | trf stopped early on the cumulative-sum basis at 30 knots and ignored 5 clean points (1 Nov #6). |
| step voltage = largest fit difference | most convex point of the monotone fit in 0.2–0.9 Voc | the largest difference sat at Voc for every sweep, so any single flag "matched" its partner. |
| 3.4 free robust end line | line pinned to the spline at the blend, only its slope fitted (capped at the spline's slope); starts at the lowest clean point if the sweep has no data near Isc | free lines left a visible step at 0.15 Voc (18 Oct #22, #29). |
| Pmax reference always used | flagged "not bracketed" when the polynomial peak is at the window edge / < 3 points each side; acceptance also allows 2× the reference's own standard error | current-swept data often have no points left of a sharp knee; a one-sided 4th-order fit overshoots (18 Oct #11: 13.90 W, above every measured point). |
| Voltage-coverage check | counts clean points (after Hampel) | as the plan's wording ("at least 1 clean point"). |

## Validation summary (from the last run)
- QC: invalid = 18 Oct #13, 1 Nov #36, #37 (exactly the plan's list).
- Pmax: median |ε| 0.26 % (46 bracketed sweeps); 43 pass, 7 reference not bracketed, 3 low confidence – all
  degraded low-FF panels (18 Oct P8 #33, P12 #46; 1 Nov P12 #33), as the plan's §5 anticipates.
- Steps kept: 18 Oct P4, P5, P7, P8, P9; 1 Nov P9, P10. 1 Nov #11/#12 and 18 Oct #14 normal.
- Rs/Rsh regression: 1 Nov #11 SDM Rs 0.24 Ω, Rsh 113 Ω, n 1.46 (plan's values); 1 Nov #12 Rs 1.12 / 0.38 / 0.25 Ω
  (geometric / slope / SDM); 18 Oct #19 SDM rejected (n 0.65).

## Inputs still needed (plan §7)
1. **Irradiance G and module temperature T** per session – without them there is no STC translation and no
   ε_rated (shown as "n/a (no G/T recorded)").
2. **One PROVA 200A PDF export** in `data/reference/` to fine-tune Fig. A (Phase 6.3).
3. **Cell count** per module (36 assumed) and **module area** (efficiency).
4. **Instrument accuracy**: values from PROVA 200A listings are in `config.yaml`; check the 0.6–1 A current offset
   against your manual (listings disagree).
5. Nameplate ratings exist for 18 Oct panels 1–13 (P05 none); 1 Nov panels have no nameplate mapping.
