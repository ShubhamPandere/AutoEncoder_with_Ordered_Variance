# -*- coding: utf-8 -*-
"""
Corrected AEO-vs-PCA pipeline for TEP.
=====================================

Runs the four monitors the audit calls for, all calibrated to the SAME measured
false-alarm rate so that detection delay is finally comparable:

    AEO-T2        vs  PCA-T2   (Vpc capped at 95)  -- honest like-for-like
    AEO-T2+SPE    vs  PCA-T2+SPE                    -- standard TEP protocol

Key efficiency point: the AEO (and PCA) model is fitted on the FAULT-FREE
training window only, so it does not depend on the fault number at all. Each
candidate config is therefore trained ONCE and scored against every fault --
turning (n_faults x n_trials) trainings into n_trials, which is why this can run
to a much larger search budget on CPU than the original per-fault loop.

Two control limits are reported for every method:
  * `calibrated` -- limit from fault-free calibration runs only (no peeking).
    This is the deployable number; its measured FAR drifts because TEP
    fault-free runs genuinely differ from each other (measured per-run FAR
    ranges 0.004-0.025 for one fixed limit).
  * `matched`    -- limit set so the measured FAR on the held-out fault-free
    runs is exactly the target. Removes calibration drift from the comparison,
    which is the standard "compare at equal FAR" convention.

Usage:
    python tep_corrected_pipeline.py                 # full run
    python tep_corrected_pipeline.py --trials 8      # quick smoke run
"""

import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd

import tep_data as td
import tep_models as tm
import tep_protocol as tp

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_TABLES = os.path.join(_THIS_DIR, "results", "tables", "corrected")
OUT_PLOTS = os.path.join(_THIS_DIR, "results", "plots", "corrected")
for _d in (OUT_TABLES, OUT_PLOTS):
    os.makedirs(_d, exist_ok=True)

FAULTS = [1, 3, 4, 6, 8, 9, 15]

# ----------------------------------------------------------------------
# Search spaces
# ----------------------------------------------------------------------
AEO_SPACE = {
    "latent_dim": [4, 8, 16],
    "hidden": [16, 32, 64],          # audit A7f: 16 was inherited from 38-var DAA
    "epochs": [1000, 2000, 4000],
    "q_shape": ["linear", "quadratic", "geometric"],   # audit A7e
    "loss_alpha": (0.5, 4.0),
    "beta": (0.05, 1.0),
    "gamma": (1e-4, 0.3),            # log-uniform; audit A7d: unscaled by N
    "lr": (5e-4, 5e-3),              # log-uniform
}
# NOTE: `alpha` (the nominal significance level) is GONE from the search space.
# Control limits are now empirical, so there is nothing to tune there -- one
# fewer hyperparameter than the original AEO search, and the one that was doing
# the most damage (audit A5).

PCA_T2_VPC_GRID = [80, 85, 90, 95]        # capped at 95: no free residual statistic
PCA_FULL_VPC_GRID = [85, 90, 95, 99]      # T2+SPE variant may use the full range


def sample_aeo_config(rng):
    cfg = {k: (rng.choice(v).item() if isinstance(v[0], (int, np.integer)) else str(rng.choice(v)))
           for k, v in AEO_SPACE.items() if isinstance(v, list)}
    for k in ("loss_alpha", "beta"):
        lo, hi = AEO_SPACE[k]
        cfg[k] = float(rng.uniform(lo, hi))
    for k in ("gamma", "lr"):
        lo, hi = AEO_SPACE[k]
        cfg[k] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    return cfg


# ----------------------------------------------------------------------
# Statistic bundling
# ----------------------------------------------------------------------
def _limits_and_metrics(stat_calib, stat_far, stat_faulty, target_far):
    """
    Evaluate one scalar statistic under both limit conventions.
    `stat_far` / `stat_faulty` are {run_id: array}.
    """
    cl_cal = tp.empirical_control_limit(stat_calib, target_far)
    m_cal = tp.evaluate_detection_v2(stat_faulty, stat_far, cl_cal)

    pooled_far = np.concatenate(list(stat_far.values()))
    cl_match = tp.empirical_control_limit(pooled_far, target_far)
    m_match = tp.evaluate_detection_v2(stat_faulty, stat_far, cl_match)
    return m_cal, m_match


def _combine(t2, spe, cl_t2, cl_spe):
    """Combined index: >1 iff either statistic exceeds its own limit."""
    return np.maximum(t2 / cl_t2, spe / cl_spe)


def evaluate_monitor(monitor, faults, target_far=tp.TARGET_FAR, csv=td.DEFAULT_CSV):
    """
    Score a fitted monitor on every fault, for both the T2-only and the T2+SPE
    index, under both limit conventions. Returns a list of metric dicts.
    """
    calib_X = np.vstack([td.get_run(0, r, csv) for r in tp.CALIB_RUNS])
    t2_cal, spe_cal = monitor.statistics(calib_X)

    far_stats = {r: monitor.statistics(td.get_run(0, r, csv)) for r in tp.FAR_RUNS}

    # combined-index limits come from calibration data
    cl_t2_c, cl_spe_c = tp.combined_control_limits(t2_cal, spe_cal, target_far)

    comb_far = {r: _combine(v[0], v[1], cl_t2_c, cl_spe_c) for r, v in far_stats.items()}

    rows = []
    for fault in faults:
        # `tune` runs drive config selection, `val` runs are what gets reported.
        # Keeping them in one pass is free (the model is already trained) and
        # makes the leak impossible to reintroduce downstream.
        for split, run_ids in (("tune", tp.TUNE_RUNS), ("val", tp.VAL_RUNS)):
            fa_stats = {r: monitor.statistics(td.get_run(fault, r, csv)) for r in run_ids}
            base = {"fault_number": fault, "split": split}

            # ---- T2 only ----
            m_cal, m_match = _limits_and_metrics(
                t2_cal, {r: v[0] for r, v in far_stats.items()},
                {r: v[0] for r, v in fa_stats.items()}, target_far)
            rows.append({**base, "index": "T2", "limit": "calibrated", **m_cal})
            rows.append({**base, "index": "T2", "limit": "matched", **m_match})

            # ---- SPE only ----
            m_cal, m_match = _limits_and_metrics(
                spe_cal, {r: v[1] for r, v in far_stats.items()},
                {r: v[1] for r, v in fa_stats.items()}, target_far)
            rows.append({**base, "index": "SPE", "limit": "calibrated", **m_cal})
            rows.append({**base, "index": "SPE", "limit": "matched", **m_match})

            # ---- combined T2+SPE ----
            comb_fa = {r: _combine(v[0], v[1], cl_t2_c, cl_spe_c) for r, v in fa_stats.items()}
            m_cal = tp.evaluate_detection_v2(comb_fa, comb_far, 1.0)
            cl_match = tp.empirical_control_limit(
                np.concatenate(list(comb_far.values())), target_far)
            m_match = tp.evaluate_detection_v2(comb_fa, comb_far, cl_match)
            rows.append({**base, "index": "T2+SPE", "limit": "calibrated", **m_cal})
            rows.append({**base, "index": "T2+SPE", "limit": "matched", **m_match})

    for r in rows:
        r["score"] = tp.score_metrics_v2(r)
        r["legacy_f1"] = tp.legacy_f1(r)
    return rows


# ----------------------------------------------------------------------
# Search drivers
# ----------------------------------------------------------------------
def sample_config_list(n_trials, seed=0):
    """
    The full ordered list of configs for a search.

    Split out so a sharded (e.g. Modal) run can generate the identical list in
    every container and take a slice of it -- the trial indices then mean the
    same thing everywhere, which is what lets the shards be re-aggregated.
    """
    rng = np.random.default_rng(seed)
    return [sample_aeo_config(rng) for _ in range(n_trials)]


def search_aeo(n_trials=40, seed=0, faults=FAULTS, model_seed=42,
               target_far=tp.TARGET_FAR, csv=td.DEFAULT_CSV, log_every=1,
               configs=None, trial_offset=0):
    if configs is None:
        configs = sample_config_list(n_trials, seed)
    Xtr = td.get_run(0, tp.TRAIN_RUN, csv)[:tp.WINDOW_SIZE]

    all_rows = []
    n_trials = len(configs)
    for t_local, cfg in enumerate(configs):
        t = trial_offset + t_local
        t0 = time.time()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            mon = tm.AEOMonitor(model_seed=model_seed, **cfg).fit(Xtr)
        elapsed = time.time() - t0

        # T2 is evaluated at the full latent space; the 90%-variance rule is
        # degenerate under AEO's own regulariser (audit A4) so it is reported
        # separately rather than used as the default.
        rows = evaluate_monitor(mon, faults, target_far, csv)
        v = mon.latent_variances
        top1 = float(v[0] / (v.sum() + 1e-12) * 100)
        p90 = int(max(2, (np.cumsum(v) / (v.sum() + 1e-12) < 0.90).sum() + 1))
        for r in rows:
            r.update({"trial": t, "method": "AEO", "elapsed_s": elapsed,
                      "converged": mon.converged, "p_dims": mon.p_dims,
                      "p_dims_90pct_rule": p90, "top1_var_pct": top1,
                      "cond_number": mon.cond_number, **cfg})
        all_rows.extend(rows)

        if log_every and t % log_every == 0:
            best = max(r["score"] for r in rows if r["index"] == "T2+SPE" and r["limit"] == "matched")
            print(f"  [AEO trial {t+1}/{n_trials}] {elapsed:5.1f}s "
                  f"latent={cfg['latent_dim']:2d} hid={cfg['hidden']:2d} ep={cfg['epochs']:4d} "
                  f"Q={cfg['q_shape'][:4]} conv={mon.converged} top1={top1:5.1f}% "
                  f"best_T2SPE_score={best:.3f}", flush=True)
    return pd.DataFrame(all_rows)


def search_pca(faults=FAULTS, target_far=tp.TARGET_FAR, csv=td.DEFAULT_CSV):
    Xtr = td.get_run(0, tp.TRAIN_RUN, csv)[:tp.WINDOW_SIZE]
    all_rows = []
    for Vpc in sorted(set(PCA_T2_VPC_GRID) | set(PCA_FULL_VPC_GRID)):
        t0 = time.time()
        mon = tm.PCAMonitor(Vpc=Vpc).fit(Xtr)
        rows = evaluate_monitor(mon, faults, target_far, csv)
        for r in rows:
            r.update({"method": "PCA", "Vpc": Vpc, "Np": mon.Np,
                      "cond_number": mon.cond_number, "elapsed_s": time.time() - t0})
        all_rows.extend(rows)
        print(f"  [PCA Vpc={Vpc}] Np={mon.Np} cond={mon.cond_number:.1f}", flush=True)
    return pd.DataFrame(all_rows)


# ----------------------------------------------------------------------
# Selection: best config per (fault, method-variant), chosen on TUNE runs
# ----------------------------------------------------------------------
def select_best(df, index, report_limit="matched", extra_filter=None,
                key_cols=("trial", "Vpc")):
    """
    Pick, per fault, the config maximising `score` on the TUNING runs under the
    no-peeking `calibrated` limit, then return that config's VALIDATION row.

    Two separations matter here:
      * selection on `split == "tune"` (faulty runs 1-3), reporting on
        `split == "val"` (faulty runs 4-10) -- no faulty-data leakage.
      * selection under `limit == "calibrated"` (limit from fault-free
        calibration runs only) -- so nothing about the held-out fault-free runs
        influences which config wins. The reported row then uses `report_limit`
        ("matched") purely so the head-to-head comparison sits at an identical
        measured FAR.
    """
    sub = df[df["index"] == index].copy()
    if extra_filter is not None:
        sub = sub[extra_filter(sub)]
    if sub.empty:
        return sub

    keys = [c for c in key_cols if c in sub.columns]
    tune = sub[(sub["split"] == "tune") & (sub["limit"] == "calibrated")]
    val = sub[(sub["split"] == "val") & (sub["limit"] == report_limit)]
    if tune.empty or val.empty:
        return val.iloc[0:0]
    winners = tune.loc[tune.groupby("fault_number")["score"].idxmax(),
                       ["fault_number"] + keys]
    return val.merge(winners, on=["fault_number"] + keys, how="inner").reset_index(drop=True)


def build_comparison(aeo_df, pca_df):
    """Assemble the two head-to-head tables the plan asks for."""
    comparisons = []

    # (1) honest like-for-like: T2 only, PCA capped at Vpc <= 95
    a = select_best(aeo_df, "T2")
    p = select_best(pca_df, "T2", extra_filter=lambda s: s["Vpc"] <= 95)
    a["variant"], p["variant"] = "AEO-T2", "PCA-T2 (Vpc<=95)"
    comparisons += [a, p]

    # (2) standard TEP protocol: T2+SPE for both
    a2 = select_best(aeo_df, "T2+SPE")
    p2 = select_best(pca_df, "T2+SPE")
    a2["variant"], p2["variant"] = "AEO-T2+SPE", "PCA-T2+SPE"
    comparisons += [a2, p2]

    return pd.concat(comparisons, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model-seed", type=int, default=42)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--faults", type=int, nargs="*", default=None)
    args = ap.parse_args()

    faults = args.faults if args.faults else FAULTS
    tag = f"_{args.tag}" if args.tag else ""

    print(f"Corrected TEP pipeline | faults={faults} | trials={args.trials} | "
          f"model_seed={args.model_seed}")
    print(f"Splits: {json.dumps(tp.SPLITS, default=str)}\n")

    print("PCA grid ...")
    pca_df = search_pca(faults)
    pca_df.to_csv(os.path.join(OUT_TABLES, f"pca_trials{tag}.csv"), index=False)

    print(f"\nAEO random search ({args.trials} trials, trained once each and "
          f"scored on all {len(faults)} faults) ...")
    aeo_df = search_aeo(args.trials, seed=args.seed, faults=faults,
                        model_seed=args.model_seed)
    aeo_df.to_csv(os.path.join(OUT_TABLES, f"aeo_trials{tag}.csv"), index=False)

    comp = build_comparison(aeo_df, pca_df)
    comp.to_csv(os.path.join(OUT_TABLES, f"comparison{tag}.csv"), index=False)

    cols = ["fault_number", "variant", "far", "mdr", "delay_samples",
            "detection_rate", "balanced_accuracy", "legacy_f1"]
    pd.set_option("display.width", 220)
    print("\n=== CORRECTED COMPARISON (matched FAR, held-out runs) ===")
    print(comp[cols].sort_values(["fault_number", "variant"]).to_string(index=False))
    print(f"\n[SAVED] {OUT_TABLES}")


if __name__ == "__main__":
    main()
