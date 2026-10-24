# TEP / AEO evaluation audit — discrepancy report

**Subject:** why the Tennessee Eastman results contradicted the DAA paper
**Scope:** `Tennesse dataset/` (all scripts), `results/tables/all20/`, `p_aeo (1).py`
**Reference papers:** Augustine et al., *Autoencoder with Ordered Variance* (arXiv 2402.14031);
Pandere et al., *Fault Prediction using Autoencoder with Ordered Variance*

---

## 0. Summary

Two TEP results contradicted the DAA paper:

1. AEO's false-alarm rate came out **lower** than PCA's, but its missed-detection rate **higher**.
2. PCA's detection **delay** was shorter than AEO's on most faults — the reverse of the paper's
   headline claim.

Both observations are real in the data. **Neither is a property of AEO.** Both are artifacts of the
evaluation protocol, and they have a common root: the control limits were never calibrated, so the
two methods were compared at wildly different operating points, and the tuner's ranking metric
actively rewarded whichever method drifted to the trigger-happy end.

A third, genuine, methodological gap sits underneath: **AEO has no residual statistic**, so faults
that live in the low-variance subspace are invisible to it — while PCA at `Vpc=99` silently
acquired one.

| # | Finding | Severity |
|---|---|---|
| A1 | Detection delay compared at unmatched false-alarm rates | **Invalidates the delay claim** |
| A2 | F1 ranking is degenerate on a 160-normal / 800-faulty split | **Caused A1** |
| A3 | PCA's win on IDV(4)/(11)/(14)/(17) is a residual statistic in disguise | **Invalidates those comparisons** |
| A4 | The 90%-variance rule for `p` is degenerate under AEO's own regularizer | Major |
| A5 | Control limits miscalibrated in both directions; nominal α ≠ measured FAR | **Explains finding 1** |
| A6 | The seed-stability study never varied the model seed | Major |
| A7 | Eleven secondary code issues | Minor–moderate |
| A8 | IDV(3)/(9)/(15) failure is expected, not a bug | Documentation |

---

## A1. The delay comparison is confounded by false-alarm rate

`evaluate_detection` (`tep_evaluation.py:245`) defines detection delay as *the first alarm at or
after onset*. A detector already alarming continuously before onset scores delay ≈ 0 for free.

PCA on IDV(1), varying **only** the control limit (held-out runs 2–10):

| Vpc / α | FAR | MDR | delay |
|---|---|---|---|
| 85 / 0.01 | 0.001 | 0.0093 | 7.33 |
| 90 / 0.05 | 0.006 | 0.0078 | 6.11 |
| 95 / 0.10 | 0.053 | 0.0065 | 4.44 |
| **99 / 0.10 — what the tuner picked** | **0.199** | 0.0044 | **2.67** |

Delay falls monotonically as FAR rises. It is a knob, not a result.

The reported AEO-higher-p on IDV(1) has FAR 0.0063, delay 5.89. At *matched* FAR — PCA at
Vpc 90/α 0.05 (FAR 0.006, delay 6.11) or Vpc 85/α 0.05 (FAR 0.003, delay 6.78) — **AEO is faster
than PCA.** The reported conclusion reverses once FAR is held constant.

> Delay comparisons at unmatched FAR are meaningless and must not appear in the paper.

## A2. F1 is a degenerate ranking criterion on this split — and it caused A1

`score_metrics` (`tep_tuning.py:62`) ranks configs by F1. The monitored region holds 160 normal and
800 faulty samples, so an **"always alarm" detector already scores F1 = 0.909**, against a best
achievable ≈ 0.995. The entire discriminative range is 0.086 wide.

Consequences:

- The tuner selected `Vpc=99` for PCA on **13 of 20 faults** and `α=0.1` on **14 of 20** — it is
  actively choosing the trigger-happy corner, because F1 barely penalizes at most 160 false
  positives against 800 true ones.
- Selection is noisy: IDV(7)'s best AEO trial scores F1 = 0.731 on the tuning run but 0.997 on
  held-out validation. The ranking is not tracking generalization.
- **Search-budget asymmetry:** AEO got 15 random-search trials over a *7-dimensional* space
  (`latent_dim, epochs, loss_alpha, beta, gamma, lr, alpha`); PCA got an exhaustive 12-point grid
  over *2*. AEO was far more likely to be left under-tuned. (Mean trial cost was only 20.3 s —
  the budget was not compute-bound.)

## A3. PCA's win on IDV(4)/(11)/(14)/(17) is a residual statistic in disguise

IDV(4)'s signature lives in principal components **36–39** — the directions carrying the *last 1%*
of the variance. Mean MDR vs number of retained PCs (α = 0.01, held-out runs):

| PCs kept | 25 | 30 | 32 (=Vpc 95) | 35 | 38 | 39 (=Vpc 99) | 52 |
|---|---|---|---|---|---|---|---|
| IDV(4) | 0.863 | 0.698 | 0.690 | 0.313 | 0.012 | **0.001** | 0.001 |
| IDV(1) | 0.007 | 0.007 | 0.007 | 0.007 | 0.007 | 0.006 | 0.001 |

At `Vpc=99` PCA retains 39 of 52 directions, so its "T²" is effectively a whitened full-space
residual test — which is exactly why the tuner chose that setting for IDV(4).

AEO with `latent_dim ≤ 16` **cannot represent those directions at any value of `p`**. Sweeping `p`
(as `tep_p_sweep.py` does) only selects among latent dims that exist; the discarded 36 directions
live solely in the reconstruction error `x − x̂`. This is a representational gap, not a tuning gap.

Confirmation — a plain SPE = ‖x − x̂‖² on just 10 PCs:

| | IDV(1) | IDV(4) | IDV(6) |
|---|---|---|---|
| SPE MDR | 0.004 | **0.001** | 0.000 |

The faults where PCA beat AEO in the all-20 table — 4, 11, 14, 17 — are all small localized
step/sticking faults in the reactor cooling loops, i.e. precisely the residual-subspace class.

## A4. The 90%-variance rule for `p` is degenerate under AEO's own regularizer

`J2 = β·Σⱼ qⱼ Vⱼ` penalizes variance in every latent dim, so AEO deliberately crushes all but the
first. Measured top-1 variance concentration (`results/tables/tep_aeo_diagnostics.csv`): **88–99%**.

Therefore `p = max(2, (cum_var < 0.90).sum() + 1)` returned **2 in 15 of 16 diagnostic runs**, and
for `latent_dim=4` faults even a **99%** threshold still returns 2
(`results/tables/tep_p_sweep_summary.csv`, faults 3/4/6/9/15). AEO's T² was watching 2 directions
while PCA's watched 39.

> The ordering property and the variance-threshold selection rule are in direct conflict. This is a
> real finding about the method, worth stating in the paper — not merely a bug.

## A5. Control limits are miscalibrated in both directions — this explains finding 1

The F-based limit assumes i.i.d. multivariate-normal scores. TEP data is strongly autocorrelated,
and AEO's `covY` is the **in-sample** covariance of a network trained 4000–8000 epochs on the very
same 200 points. Nominal α vs measured FAR (`tep_all20_higher_p_vs_pca_summary.csv`):

| Method / fault | nominal α | measured FAR | error |
|---|---|---|---|
| AEO, IDV(1) | 0.087 | 0.006 | 7× conservative |
| AEO, IDV(3) | 0.194 | 0.242 | liberal |
| PCA, Vpc 99 | 0.100 | 0.199 | 2× liberal |

**α does not mean the same thing for the two methods.** That is the whole of finding 1: AEO's limit
happened to land conservative (low FAR, high MDR) and PCA's liberal (high FAR, low MDR). It is a
calibration artifact, not a difference in detection power.

## A6. The seed-stability study never varied the model seed

`retry_unstable` (`modal_tep_all20_pipeline.py:271`) re-ran faults 6 and 15 across seeds 0–4. That
`seed` feeds only the hyperparameter-search RNG (`tep_tuning.py:76`). Both `aeo_monitoring`
(`p_aeo (1).py:131`) and `train_aeo_once` (`tep_p_sweep.py:46`) hardcode `set_seed(42)`, so **the
network initialization was identical in all five runs.** AEO's run-to-run variance was never
measured, and the "unstable config" conclusion rests on nothing.

## A7. Secondary code issues

| # | Location | Issue |
|---|---|---|
| a | `p_aeo (1).py:898` `train_pca` | `np.linalg.eig` on a *symmetric* covariance. Returns complex values (silently `np.real`'d) and non-orthogonal eigenvectors for the near-degenerate tail — precisely PCs 32–39, the ones the winning Vpc=99 configs depend on. Should be `eigh`. |
| b | `tep_p_sweep.py:96` | `p_dims_for_strategy(("fixed", 8))` returns `min(8, latent_dim)`, so for `latent_dim=4` faults `fixed_p8`, `fixed_p4` and `fixed_p_full_latent` are the same run. The dedup at `tep_p_sweep.py:161` then drops `fixed_p_full_latent`, yet downstream tables label the survivor "higher-p = full latent". |
| c | `tep_evaluation.py:238` | FAR is measured on the 160 pre-onset samples of the faulty run. **Verified: those samples are bit-identical to the fault-free run of the same `simulationRun` (max abs diff 0.0).** So all 20 faults shared one FAR measurement replicated 20×, on only 160 samples — which is why `far_mean` is exactly `0.19930555…` for 13 different faults in the all-20 summary. |
| d | `p_aeo (1).py:165` | `j1`/`j2` scale with `N·n` (sum reduction) but `j3 = γ·Σθ²` does not, so γ is effectively ~10⁴× weaker than α/β. The paper's α:β:γ ratios do not transfer to this data size. |
| e | `p_aeo (1).py:142` | `Q = diag(linspace(0, m−1))` is hardcoded linear. The AEO paper treats the shape of `Q` as a *primary* tuning parameter (`diag(1, q²)`, `q` swept 1–10) and shows results depend on it. Never tuned here. |
| f | `p_aeo (1).py:44` | Encoder is `52 → 16 → latent_dim`. For `latent_dim=16` the 16-unit hidden layer is a harder bottleneck than the latent layer itself. Inherited unchanged from the 38-variable DAA setup. |
| g | `p_aeo (1).py:243` vs paper Eq. (15) | Code uses the F-based Hotelling limit; the paper specifies `Cl = χ²₁₋α(d)`. Paper and code disagree. |
| h | `tep_evaluation.py:229` / `p_aeo (1).py:949` | AEO flags on `T2 > Cl`, PCA on `t2 >= Cl`. Cosmetic, but should match. |
| i | `p_aeo (1).py` | `aeo_monitoring`/`pca_monitoring` are each defined **three times** in one file; the last definition silently wins. `tep_evaluation.py` imports via `exec_module`, so reordering the file would change results invisibly. |
| j | `p_aeo (1).py:204` | If Algorithm 1 never converges within `max_iterations`, monitoring proceeds on **unordered** variances, so `Y0[:, :p]` is not the top-p dims and the whole selection rule is meaningless. No warning is emitted. **Confirmed to actually happen**: `latent_dim=16, epochs=2000` fails to order within 10 iterations on this data. |
| k | `window_size=200`, n=52 | 200 samples to estimate a 52×52 covariance and a 39-PC projection is marginal. Fault-free test runs have 960 samples available. |

## A8. IDV(3)/(9)/(15) — expected, not a bug

Both methods score ≈ 0.35 accuracy on these. That matches the published TEP literature: they carry
no detectable mean or variance shift and are near-undetectable by any T²/SPE method. Measured here
at Np=40, α=0.01: MDR = 0.93 / 0.93 / 0.93; even SPE only reaches 0.85–0.87. Report them as a known
negative control, and use them as a sanity check — any protocol that suddenly "detects" them has a
calibration leak.

---

## B. What was changed

New modules, all under `Tennesse dataset/`. `p_aeo (1).py` is left untouched as the reference
implementation.

| File | Purpose |
|---|---|
| `tep_protocol.py` | Corrected protocol: empirical control limits, leak-free data splits, `evaluate_detection_v2`, `score_metrics_v2`. |
| `tep_models.py` | `AEOMonitor` and `PCAMonitor`, each exposing **both** T² and SPE. Fixes A6, A7a, A7e, A7f, A7j. |
| `tep_data.py` | Indexed TEP loader; no more fault-free/faulty stacking (fixes A7c). |
| `tep_corrected_pipeline.py` | Driver producing the two head-to-head comparisons. |

### Protocol changes

**Leak-free splits** (`tep_protocol.py`):

| Split | Data | Used for |
|---|---|---|
| train | fault-free run 1, samples 0–479 | fitting AEO / PCA |
| calibration | fault-free runs 2–4 | setting control limits |
| FAR measurement | fault-free runs 5–10 (all 960 samples each) | measuring false-alarm rate |
| tuning | faulty runs 1–3 | hyperparameter selection |
| validation | faulty runs 4–10 | **all reported numbers** |

**Empirical control limits.** The limit is the `(1 − target_FAR)` quantile of the statistic on
held-out fault-free data, replacing the F-distribution limit. This is what makes FAR comparable
across methods and removes `alpha` from the search space entirely (one fewer hyperparameter).

Two limits are reported. `calibrated` uses only the calibration runs — deployable, no peeking, but
its measured FAR drifts because TEP fault-free runs genuinely differ from one another (measured
per-run FAR spans 0.004–0.025 for one fixed limit; this is benchmark heterogeneity, not estimation
noise). `matched` pins the measured FAR on the held-out fault-free runs to exactly the target, which
is the standard "compare at equal FAR" convention and is what the head-to-head tables use.

**Sustained-alarm delay.** Detection now requires `k = 3` consecutive violations, so a lone false
alarm can no longer score delay 0 (fixes A1).

**Ranking by delay at fixed FAR** instead of F1 (fixes A2). Since the limit is pinned to a fixed
measured FAR, delay can no longer be bought with false alarms.

**AEO gains an SPE statistic**, `SPE = ‖x − x̂‖²`, computed from the `x_recon` the model already
returns. The combined index alarms if either statistic exceeds its limit, with the two limits
bisected jointly so the combined measured FAR hits the target exactly. PCA gains the same, making it
the textbook T²+SPE monitor rather than a T²-with-39-PCs hybrid.

**Search efficiency.** Both models are fitted on the fault-free window only, so they do not depend
on the fault number. Each candidate config is now trained **once** and scored against every fault,
turning `n_faults × n_trials` trainings into `n_trials` — which is what makes a 60-trial search
affordable on CPU where the original managed 15.

---

## C. Corrected results

Local re-run: 7 faults (IDV 1, 3, 4, 6, 8, 9, 15) from `data/tep_test_subset.csv`, 60-trial AEO
random search, all four monitors at **measured FAR = 0.0101** on held-out fault-free runs 5–10.
Reported numbers are the mean over faulty validation runs 4–10, with configs selected on faulty
runs 1–3. Total AEO training cost: 737 s for 60 configs (vs 6085 s for 300 in the original
per-fault loop — because the model does not depend on the fault number).

### C1. Before / after

`old_*` columns come from `results/tables/all20/tep_all20_higher_p_vs_pca_summary.csv`;
`new_*` from `results/tables/corrected/comparison.csv`.

| fault | old FAR AEO / PCA | old delay AEO / PCA | **new delay AEO-T² / PCA-T²** | new MDR AEO-T² / PCA-T² | new MDR AEO+SPE / PCA+SPE |
|---|---|---|---|---|---|
| IDV(1) | 0.006 / 0.054 | 5.9 / **4.4** | **6.4 / 6.4** | 0.008 / 0.008 | 0.005 / 0.002 |
| IDV(4) | 0.204 / 0.113 | 0.4 / **0.0** | 43.3 / **11.1** | 0.663 / 0.494 | **0.028** / 0.002 |
| IDV(6) | 0.067 / 0.113 | **0.0** / 2.9 | **0.0** / 7.4 | **0.000** / 0.009 | 0.000 / 0.000 |
| IDV(8) | 0.001 / 0.054 | 30.6 / **14.3** | **31.0 / 30.6** | 0.050 / 0.040 | 0.044 / 0.034 |
| IDV(3) | 0.242 / 0.199 | 5.1 / 6.0 | 256 / 249 | 0.989 / 0.989 | 0.989 / 0.988 |
| IDV(9) | 0.185 / 0.199 | 4.4 / 6.6 | 256 / never | 0.988 / 0.990 | 0.988 / 0.988 |
| IDV(15) | 0.256 / 0.199 | 2.1 / 6.1 | 256 / 337 | 0.988 / 0.987 | 0.988 / 0.986 |

**The two puzzles are resolved:**

1. **"AEO has lower FAR but higher MDR."** A pure calibration artifact. AEO's limit happened to land
   conservative (FAR 0.001–0.006 on IDV 1/8) and PCA's liberal (0.054–0.113). Once both are pinned
   to FAR = 0.0101 the asymmetry vanishes: MDR is 0.008 vs 0.008 on IDV(1), 0.050 vs 0.040 on
   IDV(8).

2. **"PCA detects faster."** Mostly artifact. On IDV(1) the 5.9-vs-4.4 gap was bought with a 9×
   higher FAR; matched, both sit at **6.4**. On IDV(8) the 30.6-vs-14.3 gap was bought with a 38×
   higher FAR; matched, both sit at **~31**. On IDV(6) AEO was already better and stays better
   (0.0 vs 7.4). Only IDV(4) remains a genuine PCA win, and that is the residual-subspace issue
   (A3), not speed.

3. **The old table's most impressive numbers were false alarms.** IDV(3)/(9)/(15) showed delays of
   2–6 samples — apparently the fastest detections in the whole study. They were running at FAR
   0.18–0.26: the "detection" was noise crossing the limit, not the fault. Correctly calibrated,
   these faults are detected in only 14–43% of runs, with delays of 250+ samples. **Any table row
   for IDV(3), (9) or (15) in the current draft must be removed or re-labelled as undetectable.**

### C2. The residual-subspace gap is real, and SPE closes it

| IDV(4) monitor | MDR |
|---|---|
| AEO-T² (full latent space) | 0.663 |
| PCA-T² capped at Vpc ≤ 95 | 0.494 |
| **AEO-T² + SPE** | **0.028** |
| PCA-T² + SPE | 0.002 |

Capping PCA at Vpc ≤ 95 removes its accidental residual statistic, and its IDV(4) MDR jumps from the
reported 0.0003 to 0.494 — confirming A3. Giving AEO an SPE statistic drops its MDR from 0.663 to
0.028, a **24× reduction**. Neither method can do this with T² alone.

### C3. The Q matrix drives the `p`-selection degeneracy (new finding)

The hardcoded linear `Q` is itself part of why the 90%-variance rule collapsed to `p = 2`. Across
the 60 trials:

| Q shape | mean top-1 variance concentration | median `p` from the 90% rule |
|---|---|---|
| linear (the hardcoded choice) | 85.5% (68.7–97.5) | **2** |
| quadratic | 75.1% (44.7–99.2) | 2.5 |
| geometric | **48.9%** (19.3–93.0) | **5** |

A geometric `Q` spreads variance across the latent space and makes the variance-threshold rule
behave sensibly. This is a cheap, concrete fix for A4 — and it is exactly the tuning knob the AEO
paper treats as primary and that the code never exposed (A7e).

### C4. Seed stability (verification 5)

Three independent model seeds (42, 7, 123), identical search:

| AEO-T²+SPE MDR | IDV(1) | IDV(4) | IDV(6) | IDV(8) |
|---|---|---|---|---|
| seed 42 | 0.005 | 0.028 | 0.000 | 0.044 |
| seed 7 | 0.006 | 0.032 | 0.000 | 0.043 |
| seed 123 | 0.006 | 0.030 | 0.000 | 0.044 |
| **std** | 0.001 | 0.002 | 0.000 | 0.001 |

Max MDR std across seeds on the detectable faults is **0.0024**, far below the 0.05 flag threshold —
the corrected AEO+SPE pipeline is reproducible, in direct contrast to the DAA setup in C5. PCA is
bit-identical across seeds, confirming the harness itself is deterministic. AEO-T²-only is **not**
stable on IDV(4) (delay 43.3 / 18.3 / 26.6), which is expected: that fault is out of T²'s reach, so
the number is noise. Delays on IDV(3)/(9)/(15) are also unstable, for the same reason.

### C6. Verification criteria (from the plan)

| # | Criterion | Result |
|---|---|---|
| 1 | Measured FAR within ±0.5 pp of the 1% target for every method | **PASS** — exactly 0.01007 for all methods, seeds and faults |
| 2 | AEO delay ≤ PCA delay at equal FAR on IDV(1) | **PASS (as a tie)** — 6.43 vs 6.43; the old 5.9-vs-4.4 PCA lead is gone |
| 3 | AEO-T²+SPE on IDV(4) reaches MDR ≲ 0.05 | **PASS** — 0.030 (mean over 3 seeds), down from 0.663 for T²-only |
| 3b | PCA-T² capped at Vpc ≤ 95 loses IDV(4) | **PASS** — MDR 0.494, vs the 0.0003 reported at Vpc = 99 |
| 4 | IDV(3)/(9)/(15) stay undetectable (MDR > 0.85) for both methods | **PASS** — min MDR 0.984 across all methods and seeds; no calibration leak |
| 5 | Seed-to-seed accuracy std < 0.05 | **PASS** — max 0.0024 |
| 6 | DAA result survives the corrected protocol | **PARTIAL — see C5.** The lead survives matched α (+2) but not a change of model seed. |

### C5. Regression guard on DAA — the published result needs a caveat

`daa_regression_guard.py`. First, fidelity: the new `AEOMonitor` reproduces `p_aeo (1).py`
**bit-identically** on DAA Scenario 1 (correlation of the two T² trajectories = 1.0000, identical
maxima), so what follows is not an artifact of the rewrite. The published Table 1 also reproduces
exactly: PCA first detection **505**, AEO **502**.

| Test | Result |
|---|---|
| Published configuration (α_AEO = 0.1, α_PCA = 0.01) | AEO 502, PCA 505 → **lead +3** ✔ reproduced |
| Same nominal α for both (α = 0.01) | AEO 503, PCA 505 → **lead +2** — survives |
| Matched *measured* FAR (empirical limit) | both detect at 500 → **lead 0** |
| Across model seeds {0, 1, 2, 7, 42} at α = 0.01 | AEO detects in only **2 of 5**; leads +2 (seed 42) and −3 (seed 2); seeds 0, 1, 7 **never detect** |

Two things follow. The lead is **not** purely a threshold artifact — it survives matching α, which
is genuinely in AEO's favour. But it is **seed-fragile**: the published number holds for the
hardcoded `set_seed(42)` and fails for three of five alternatives. This is audit A6 landing on the
already-published result, and it is the most urgent item in this report.

The DAA limits are also badly miscalibrated, in the same way as TEP: on Scenario 1 the AEO T² reaches
158 inside the training window but only **0.19** on held-out fault-free samples 300–499 — an 800×
gap between the in-sample distribution the F-limit is derived from and the out-of-sample
distribution it is applied to. Measured FAR is 0.000 at every nominal α from 0.01 to 0.10.

> Scenario 2 could not be checked: the local `scenario2.xlsx` has 372 variables, not the 38 of the
> filtered file the paper used. `Scenario2_filtered.xlsx` is needed to complete this guard.

---

## D. What this means for the paper

**Must change:**

1. Drop every detection-delay comparison made at unmatched FAR; re-state all delay results at a
   fixed measured false-alarm rate (fig1, fig3).
2. Remove or re-label the IDV(3)/(9)/(15) rows — the reported 2–6 sample delays were false alarms.
3. Re-run the DAA result across model seeds before resubmission. As it stands the headline claim
   rests on one hardcoded seed and does not survive four of five alternatives.
4. Stop reporting nominal α as if it were the false-alarm rate; report measured FAR.

**Honest bottom line for TEP:** after correcting the protocol, **AEO is competitive with, but not
superior to, PCA on TEP.** It wins clearly on IDV(6), ties on IDV(1) and IDV(8), and loses on
IDV(4). That is a weaker claim than the DAA paper's and should be stated as such — the value of the
TEP study is that it shows *where* the ordered-variance construction helps (nonlinear,
high-variance-subspace faults) and where it structurally cannot (residual-subspace faults, unless
SPE is added).

**Strongest available contribution:** AEO-T²+SPE with a tuned `Q` shape. It fixes IDV(4)
(MDR 0.663 → 0.028), is reproducible across seeds, and keeps the ordered-variance story intact —
the AEO paper's own Eq. (24) already treats the low-variance dims as the model residual, so this is
a natural extension rather than a departure.

**Reproduce:**

```
venv/Scripts/python.exe tep_corrected_pipeline.py --trials 60
venv/Scripts/python.exe tep_corrected_pipeline.py --trials 60 --model-seed 7 --tag seed7
venv/Scripts/python.exe tep_corrected_plots.py
venv/Scripts/python.exe daa_regression_guard.py
```

Outputs land in `results/tables/corrected/`, `results/plots/corrected/`,
`results/logs/corrected_run_*.log`.

**Not yet done:** the full 20-fault Modal re-run (plan step B7). The corrected pipeline is
fault-agnostic, so it needs only the all-20 CSV plus `--faults 1 2 … 20`; budget ~750 s of AEO
training plus statistics, well under the original run's cost.

### Figures

| file | shows |
|---|---|
| `results/plots/corrected/fig1_delay_before_after.png` | delay before/after matching FAR |
| `results/plots/corrected/fig2_mdr_by_index.png` | MDR by method and index; the SPE fix on IDV(4) |
| `results/plots/corrected/fig3_delay_vs_far.png` | the delay-vs-FAR trade-off that caused the confound |

---

## E. Corrected all-20 re-run (Modal)

`modal_tep_corrected_all20.py`, workspace `liochessmag`, 8 containers max, 60 AEO configs,
all 20 faults, every method at measured FAR = **0.01007**. Results in
`results/tables/corrected_all20/`.

Parallelisation note: the old `modal_tep_all20_pipeline.py` gave each *fault* its own container
because the old code re-trained the AEO per fault. It never needed to — both models are fitted on
the fault-free window only, so they are fault-independent. The corrected pipeline shards the
*trial list* instead and scores every config against all 20 faults, which is why 60 configs × 20
faults fits comfortably in 8 containers.

### E1. Cross-platform validation

| check | result |
|---|---|
| Config lists identical (Windows local vs Modal Linux) | **identical** for all 8 hyperparameters |
| PCA results | **bit-identical** — deterministic, platform-independent |
| AEO training outcome (top-1 variance concentration) | mean \|diff\| 1.77 pp, max 19.8 pp — float/BLAS differences |
| Downstream MDR on the 7 shared faults | agrees to within **0.016**, and only on IDV(4) |

So the local 7-fault conclusions reproduce on Modal. The residual AEO drift is worth stating in the
paper as a reproducibility caveat: identical seeds do **not** give identical AEO models across
platforms, only statistically equivalent ones.

### E2. Head-to-head, 17 detectable faults (IDV 3/9/15 excluded)

| | mean MDR | mean balanced acc | mean delay |
|---|---|---|---|
| AEO-T² | **0.315** | **0.838** | 43.5 |
| PCA-T² (Vpc ≤ 95) | 0.329 | 0.830 | 54.1 |
| AEO-T²+SPE | 0.241 | 0.875 | 41.3 |
| PCA-T²+SPE | **0.191** | **0.900** | **20.8** |

**Like-for-like (T² only): AEO edges PCA** — 0.315 vs 0.329 mean MDR, and its wins are the larger
ones (IDV 5 by 0.281, IDV 16 by 0.176, IDV 10 by 0.124, IDV 20 by 0.116). PCA's wins are IDV 4
(0.181), IDV 11 (0.105), IDV 19 (0.052) — all residual-subspace faults, exactly the class A3
predicts AEO cannot see.

**With SPE added, PCA pulls ahead** — 0.191 vs 0.241, winning 7 faults decisively to AEO's 1.

### E3. The new finding: AEO's residual is weaker than PCA's

Adding SPE improves both methods, but it improves **PCA much more** (0.329 → 0.191, a 42% cut)
than AEO (0.315 → 0.241, 23%). The faults where PCA+SPE pulls away — IDV 16 (0.428 vs 0.761),
IDV 20 (0.423 vs 0.627), IDV 10 (0.346 vs 0.529), IDV 19 (0.768 vs 0.939) — are ones where AEO's
T² was *better* than PCA's T².

The explanation is the classic autoencoder failure mode: a nonlinear decoder reconstructs faulty
data too well. It generalises to inputs it was never trained on, so `‖x − x̂‖²` does not spike.
PCA's residual subspace is a fixed linear projection that cannot absorb the fault, so its SPE
reacts sharply. AEO's flexibility helps it inside the model (T²) and hurts it outside (SPE).

> This is the cleanest scientific result in the whole study, and it is a genuine contribution:
> **the ordered-variance latent space is a better monitoring subspace than PCA's scores, but a
> nonlinear autoencoder is a worse residual generator than PCA.** It suggests the right design is
> AEO's T² combined with a *constrained* residual (e.g. a linear residual on the AEO latent
> reconstruction, or a contractive/denoising penalty to stop the decoder generalising) — not the
> plain reconstruction-error SPE used here.

### E4. Monitoring-chart figures (all 20 faults)

`modal_tep_paper_figures.py` -> `results/plots/paper_figures/Fault_NN/`, two standalone
manuscript-ready figures per fault (PNG at 300 dpi plus vector PDF):

| file | statistic |
|---|---|
| `IDV01_T2.png` / `.pdf` | Hotelling's $T^2$ |
| `IDV01_T2_SPE.png` / `.pdf` | Combined $T^2$ and SPE |

Each curve is plotted as **statistic / that method's own control limit**, so the threshold is a
single dashed line at 1.0 and the two methods share one axis legitimately. This matters: the raw
statistics differ by orders of magnitude between methods (on DAA the AEO $T^2$ runs ~10^4x larger
than PCA's), so a shared raw axis would misrepresent the comparison -- which is part of how the
original DAA lead arose.

Each figure carries the held-out performance beneath the axes: detection delay (samples and
minutes), MDR, and the number of runs in which the fault was detected, followed by a one-line
statement of which method leads and by how much.

The normalisation makes the metrics readable directly off the chart:

- **MDR** = the fraction of time *after* the onset line that a curve sits **below** the dashed
  limit. IDV(3)/(9)/(15) never lift off it (MDR ~ 0.99); IDV(2)/(6) sit far above it throughout
  (MDR ~ 0).
- **FAR** = the same thing in the shaded *normal operation* band, pinned to 1% for every monitor.
- **delay** = distance from the onset line to the first sustained (k = 3) crossing, marked with a
  triangle.

Curves are cached to `curves.npz` on the volume, so restyling the figures requires no retraining
(`modal run modal_tep_paper_figures.py --replot`).

### E5. Honest bottom line

On TEP, with a correct protocol:

- **AEO-T² beats PCA-T²** on mean MDR and mean delay. The paper's core claim — that ordering the
  latent space by variance yields a better monitoring statistic than PCA scores — **holds**.
- **AEO-T²+SPE loses to PCA-T²+SPE.** Against the full standard TEP protocol, AEO is not
  state-of-the-art, and claiming otherwise would not survive review.
- IDV(3)/(9)/(15) remain undetectable for every method (MDR ≥ 0.985), as expected.

The defensible framing is the like-for-like T² comparison plus an explicit, well-evidenced account
of the residual-statistic gap (E3) as future work — not a claim of overall superiority.

**Reproduce:**

```
modal run modal_tep_corrected_all20.py --n-trials 60 --n-shards 8 --model-seed 42
modal volume get tep-data all20/corrected <local_dir>
```

---

## F. Ablation: does the variance-regularization term earn its place?

`modal_tep_ae_ablation.py` / `tep_ablation.py`. All 20 faults, 40 configs per arm,
every arm at measured FAR = 0.01007. Results in `results/tables/ablation/`.

### F0. Which coefficient to zero — the two papers disagree

| | reconstruction | variance | weights |
|---|---|---|---|
| AEO paper, Eq. (19) | **α** | **β** | **γ** |
| Fault-detection paper, Eq. (3) | (fixed at 1) | **α** | **β** |
| `p_aeo (1).py` / `tep_models.py` | `loss_alpha` | `beta` | `gamma` |

The code follows the **AEO paper's** convention exactly, so nothing in the earlier tuning was
mis-parameterized. "Remove the regularizers" therefore means **`beta = 0, gamma = 0`** — setting
`loss_alpha = 0` would delete the reconstruction term itself.

Two notes for the manuscript:

1. **Symbol collision.** Eq. (3) uses α for the variance-regularization weight, while §3.2–3.3 use α
   for the control-limit significance level. Two quantities, one symbol, same paper. Rename one.
2. **Eq. (19) is over-parameterized; Eq. (3) is the better form.** Only the ratios matter, since
   scaling J leaves the minimizer unchanged. Under Adam this is *approximately* true rather than
   exactly so — the ε in the update breaks it once gradients get small. Measured here: varying
   `loss_alpha` over 0.1 → 17 with everything else fixed moves the latent variances by only ~5–6%.
   So `loss_alpha` is a near-redundant search dimension, and the ablation fixes it at 1.0 for every
   arm, matching Eq. (3).

### F1. Arms

| Arm | Loss | Latent ordering |
|---|---|---|
| **AEO** | reconstruction + variance + weights | Algorithm 1, during training |
| **AE** | reconstruction only (`beta = gamma = 0`) | none |
| **AE-sorted** | reconstruction only | sorted by variance *after* training |
| **AE+wd** | reconstruction + weights (`beta = 0`) | sorted after training |

`AE-sorted` is the arm that matters: anyone can sort a plain AE's latent dimensions post-hoc, so
comparing AEO only against unsorted dimensions would be a strawman. Architecture and optimiser
search spaces are identical across arms, so the *only* difference is the loss.

### F2. Result — the regularizer helps, at every p

Mean MDR over the 17 detectable faults:

| Arm | \(T^2\) p=2 | p=4 | p=8 | p=all | \(T^2\)+SPE p=2 | p=4 | p=8 | p=all |
|---|---|---|---|---|---|---|---|---|
| AE | 0.454 | 0.414 | 0.390 | 0.363 | 0.283 | 0.270 | 0.268 | 0.271 |
| AE-sorted | 0.447 | 0.392 | 0.375 | 0.358 | 0.277 | 0.286 | 0.284 | 0.275 |
| AE+wd | 0.454 | 0.405 | 0.384 | 0.357 | 0.281 | 0.276 | 0.271 | 0.270 |
| **AEO** | **0.432** | **0.395** | **0.320** | **0.277** | **0.264** | **0.261** | **0.228** | **0.211** |

**AEO is best at every p under both indices**, and at full latent dimension it cuts MDR by **22%
relative** to the best AE variant — 0.277 vs 0.357 (\(T^2\)) and 0.211 vs 0.270 (\(T^2\)+SPE).

Two findings that were *not* the expected ones:

- **The gap widens with p, rather than AE collapsing at small p.** The original hypothesis was that
  ordering would let AEO truncate gracefully while AE's arbitrary leading dimensions failed. Instead
  all four arms are close at p = 2, and the three AE arms **plateau after p ≈ 8** while AEO keeps
  improving. So the variance term is improving *what the network learns*, not merely the order in
  which it is read out. That is consistent with the permutation argument: at p = latent_dim, \(T^2\)
  is invariant to any reordering, so a pure ordering effect could not show up there at all.
- **Post-hoc sorting neither helps nor hurts** (AE vs AE-sorted differ by < 0.01 throughout), and
  weight decay contributes almost nothing. The benefit comes specifically from the variance term.

### F3. Caveat — the mean is driven by a few faults

Per-fault, \(T^2\)+SPE at full p, AEO vs the best AE arm: AEO wins decisively on **3 of 17**, loses
on 5, ties on 9. But the asymmetry is stark — AEO's wins are large and its losses are marginal:

| Fault | best AE | AEO | Δ |
|---|---|---|---|
| IDV(5) | 0.723 | **0.076** | **−0.647** |
| IDV(16) | 0.808 | **0.614** | −0.194 |
| IDV(10) | 0.644 | **0.550** | −0.094 |
| … | | | |
| IDV(13) | 0.053 | 0.066 | +0.013 |
| IDV(17) | 0.139 | 0.155 | +0.016 |
| IDV(19) | 0.955 | 0.978 | +0.023 |

The largest loss is +0.023; the largest win is −0.647. IDV(5) — a step change in condenser cooling
water inlet temperature, with strongly nonlinear recirculating dynamics — is caught almost perfectly
by AEO (MDR 0.076) and almost entirely missed by every plain AE (MDR 0.72–0.76), a **~10×**
difference on a single fault.

Report it this way rather than as a flat "22% better": the honest claim is that the regularizer
**never meaningfully hurts and occasionally helps enormously**, concentrated on faults with
nonlinear dynamics.

### F4. Where this leaves the method

Ranking by mean MDR over the 17 detectable faults, \(T^2\)+SPE, full latent space:

| Method | mean MDR |
|---|---|
| PCA (T²+SPE) | **0.191** |
| AEO | 0.211 |
| AE / AE+wd / AE-sorted | 0.270 – 0.275 |

So AEO clearly beats its own ablation — the variance term is doing real work and the paper's core
mechanism is supported — while PCA remains narrowly ahead overall on this benchmark, for the
residual-generator reason established in E3.

> Cross-experiment caveat: the AEO figure here (0.211) differs slightly from §E (0.241) because the
> two runs used different search budgets and config spaces (40 vs 60 trials, `loss_alpha` fixed
> vs sampled). Comparisons are valid *within* an experiment; do not mix numbers across the two
> tables.

### F5. Figures

| file | shows |
|---|---|
| `results/plots/ablation/fig_ablation_p_sweep.png` | mean MDR vs p, four arms, both indices |
| `results/plots/ablation/fig_ablation_per_fault.png` | per-fault MDR, AEO vs best AE, ordered by AEO's advantage |

**Reproduce:**

```
modal run modal_tep_ae_ablation.py --n-trials 40 --n-shards 8 --model-seed 42
modal volume get tep-data all20/ablation <local_dir>
venv/Scripts/python.exe tep_ablation_plots.py
```
