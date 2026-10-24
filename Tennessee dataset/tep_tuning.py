# -*- coding: utf-8 -*-
"""
Per-fault hyperparameter tuning for AEO (and a small grid for PCA) on the TEP
subset, as discussed: random search on ONE tuning run per fault, validated on
held-out runs the search never saw, ranked by F1 (not raw accuracy).

Does not touch p_aeo (1).py or tep_evaluation.py's core functions -- only
adds a search loop around them.
"""

import os
import time

import numpy as np
import pandas as pd

from tep_evaluation import (
    load_tep_run, build_monitoring_sequence, fault_onset_index,
    evaluate_detection, aeo_monitoring, pca_monitoring,
    DEFAULT_FAULTS, RESULTS_DIR, TABLES_DIR, PLOTS_DIR, DATA_DIR,
)

# ----------------------------------------------------------------------
# Search spaces
# ----------------------------------------------------------------------
AEO_RANGES = {
    "loss_alpha": (0.5, 4.0),
    "beta": (0.05, 1.0),
    "gamma": (0.01, 0.3),
    "lr": (5e-4, 5e-3),   # sampled log-uniform
    "alpha": (0.01, 0.2),  # control-limit significance level
}
AEO_LATENT_DIM_CHOICES = [4, 8, 16]
# Diagnostics showed 2000 epochs was sometimes not enough for Algorithm 1's
# variance-ordering to actually converge to a concentrated latent space
# (e.g. fault 4: only 63% top-dim concentration at 2000 epochs vs 87% at
# 5000), which then inflates `p` and destabilizes the control limit. So
# epochs is now part of the search instead of a fixed constant.
AEO_EPOCHS_CHOICES = [2000, 4000, 6000, 8000]
AEO_FIXED = {"verbose": False}

PCA_GRID = {
    "Vpc": [85, 90, 95, 99],
    "alpha": [0.01, 0.05, 0.1],
}


def sample_aeo_config(rng):
    cfg = {
        "latent_dim": int(rng.choice(AEO_LATENT_DIM_CHOICES)),
        "epochs": int(rng.choice(AEO_EPOCHS_CHOICES)),
    }
    for key, (lo, hi) in AEO_RANGES.items():
        if key == "lr":
            cfg[key] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        else:
            cfg[key] = float(rng.uniform(lo, hi))
    cfg.update(AEO_FIXED)
    return cfg


def score_metrics(metrics):
    """Rank by F1; treat NaN (e.g. no positives) as worst-possible."""
    f1 = metrics.get("f1", np.nan)
    return f1 if (f1 is not None and not np.isnan(f1)) else -1.0


# ----------------------------------------------------------------------
# AEO tuning: random search on tuning_run, validate on validation_runs
# ----------------------------------------------------------------------
def tune_aeo_for_fault(
    fault_number, fault_free_run, window_size, onset_idx, split,
    tuning_run=1, validation_runs=(2, 3), n_trials=15, seed=0,
    data_dir=DATA_DIR,
):
    rng = np.random.default_rng(seed + fault_number)
    faulty_tune = load_tep_run(fault_number, tuning_run, split=split, data_dir=data_dir)
    data_tune, w = build_monitoring_sequence(fault_free_run, faulty_tune, window_size)

    trial_rows = []
    best_cfg, best_score = None, -np.inf
    for t in range(n_trials):
        cfg = sample_aeo_config(rng)
        t0 = time.time()
        T2, _, Cl = aeo_monitoring(
            data=data_tune, window_size=w,
            scenario_name=f"tune fault{fault_number} trial{t}", **cfg,
        )
        metrics = evaluate_detection(T2, Cl, w, onset_idx)
        score = score_metrics(metrics)
        trial_rows.append({
            "fault_number": fault_number, "trial": t, "score": score,
            "elapsed_s": time.time() - t0, **cfg, **metrics,
        })
        if score > best_score:
            best_score, best_cfg = score, cfg

    trials_df = pd.DataFrame(trial_rows)

    # Validate best config on held-out runs
    val_rows = []
    for run_id in validation_runs:
        faulty_val = load_tep_run(fault_number, run_id, split=split, data_dir=data_dir)
        data_val, w_val = build_monitoring_sequence(fault_free_run, faulty_val, window_size)
        T2, _, Cl = aeo_monitoring(
            data=data_val, window_size=w_val,
            scenario_name=f"validate fault{fault_number} run{run_id}", **best_cfg,
        )
        metrics = evaluate_detection(T2, Cl, w_val, onset_idx)
        val_rows.append({
            "fault_number": fault_number, "method": "AEO-tuned", "run": run_id,
            **best_cfg, **metrics,
        })

    return best_cfg, pd.DataFrame(val_rows), trials_df


# ----------------------------------------------------------------------
# PCA tuning: small exhaustive grid, same train/validate split
# ----------------------------------------------------------------------
def tune_pca_for_fault(
    fault_number, fault_free_run, window_size, onset_idx, split,
    tuning_run=1, validation_runs=(2, 3), data_dir=DATA_DIR,
):
    faulty_tune = load_tep_run(fault_number, tuning_run, split=split, data_dir=data_dir)
    data_tune, w = build_monitoring_sequence(fault_free_run, faulty_tune, window_size)

    best_cfg, best_score = None, -np.inf
    for Vpc in PCA_GRID["Vpc"]:
        for alpha in PCA_GRID["alpha"]:
            cfg = {"Vpc": Vpc, "alpha": alpha}
            T2, _, Cl = pca_monitoring(data=data_tune, window_size=w, verbose=False, **cfg)
            metrics = evaluate_detection(T2, Cl, w, onset_idx)
            score = score_metrics(metrics)
            if score > best_score:
                best_score, best_cfg = score, cfg

    val_rows = []
    for run_id in validation_runs:
        faulty_val = load_tep_run(fault_number, run_id, split=split, data_dir=data_dir)
        data_val, w_val = build_monitoring_sequence(fault_free_run, faulty_val, window_size)
        T2, _, Cl = pca_monitoring(data=data_val, window_size=w_val, verbose=False, **best_cfg)
        metrics = evaluate_detection(T2, Cl, w_val, onset_idx)
        val_rows.append({
            "fault_number": fault_number, "method": "PCA-tuned", "run": run_id,
            **best_cfg, **metrics,
        })

    return best_cfg, pd.DataFrame(val_rows)


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------
def run_tuned_sweep(
    fault_numbers=DEFAULT_FAULTS, split="test", window_size=200,
    fault_free_run_id=1, tuning_run=1, validation_runs=(2, 3),
    n_aeo_trials=15, seed=0, data_dir=DATA_DIR,
):
    onset_idx = fault_onset_index(window_size, split=split)
    fault_free_run = load_tep_run(0, fault_free_run_id, split=split, data_dir=data_dir)

    all_val_rows = []
    all_trials = []
    best_configs = []

    for fault_number in fault_numbers:
        print(f"\n=== Tuning fault {fault_number} ===")

        t0 = time.time()
        aeo_cfg, aeo_val_df, aeo_trials_df = tune_aeo_for_fault(
            fault_number, fault_free_run, window_size, onset_idx, split,
            tuning_run=tuning_run, validation_runs=validation_runs,
            n_trials=n_aeo_trials, seed=seed, data_dir=data_dir,
        )
        print(f"  AEO best config: {aeo_cfg}  (tuning took {time.time()-t0:.1f}s)")
        all_val_rows.append(aeo_val_df)
        all_trials.append(aeo_trials_df)
        best_configs.append({"fault_number": fault_number, "method": "AEO", **aeo_cfg})

        pca_cfg, pca_val_df = tune_pca_for_fault(
            fault_number, fault_free_run, window_size, onset_idx, split,
            tuning_run=tuning_run, validation_runs=validation_runs, data_dir=data_dir,
        )
        print(f"  PCA best config: {pca_cfg}")
        all_val_rows.append(pca_val_df)
        best_configs.append({"fault_number": fault_number, "method": "PCA", **pca_cfg})

    validation_results = pd.concat(all_val_rows, ignore_index=True)
    trials_results = pd.concat(all_trials, ignore_index=True)
    configs_df = pd.DataFrame(best_configs)

    validation_results.to_csv(os.path.join(TABLES_DIR, "tep_tuned_validation.csv"), index=False)
    trials_results.to_csv(os.path.join(TABLES_DIR, "tep_tuning_trials.csv"), index=False)
    configs_df.to_csv(os.path.join(TABLES_DIR, "tep_best_configs.csv"), index=False)

    return validation_results, configs_df, trials_results


def summarize_tuned(validation_results):
    summary = validation_results.groupby(["fault_number", "method"]).agg(
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        far_mean=("false_alarm_rate", "mean"), far_std=("false_alarm_rate", "std"),
        mdr_mean=("missed_detection_rate", "mean"), mdr_std=("missed_detection_rate", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        delay_mean=("detection_delay_samples", "mean"), delay_std=("detection_delay_samples", "std"),
    ).reset_index()
    summary.to_csv(os.path.join(TABLES_DIR, "tep_tuned_summary.csv"), index=False)
    return summary


if __name__ == "__main__":
    validation_results, configs_df, trials_results = run_tuned_sweep()
    summary = summarize_tuned(validation_results)
    print("\n=== Best configs per fault ===")
    print(configs_df.to_string())
    print("\n=== Held-out validation summary ===")
    print(summary.to_string())
