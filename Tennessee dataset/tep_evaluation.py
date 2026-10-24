# -*- coding: utf-8 -*-
"""
TEP (Tennessee Eastman Process) evaluation for the AEO fault-detection method.
================================================================================

This script does NOT modify the AEO / P-AEO / PCA algorithm at all. It imports
the reusable, dataset-agnostic pieces (`AutoencoderAEO`, `aeo_monitoring`,
`pca_monitoring`) directly from the sibling script `p_aeo (1).py` (one
directory up) and adds:

  1. A TEP CSV loader (`load_tep_run`) that pulls a single (fault_number,
     simulation_run) trajectory out of the Kaggle
     `averkij/tennessee-eastman-process-simulation-dataset` files.
  2. `build_monitoring_sequence`, which stitches a clean fault-free run in
     front of a faulty run so the existing `aeo_monitoring`/`pca_monitoring`
     functions (which already do `X0 = data[:window_size]` then monitor the
     rest) work unchanged.
  3. `evaluate_detection`, computing accuracy / false-alarm rate / missed-
     detection rate / F1 / detection delay -- metrics the current DAA-only
     paper does not report because it only ever had two fault scenarios.
  4. `run_tep_sweep`, a multi-fault x multi-run driver that runs both AEO and
     PCA over a representative subset of TEP faults and aggregates results
     into a pandas DataFrame + summary plots, mirroring the comparison style
     already used for the DAA scenarios in `p_aeo (1).py`.

Usage:
    python "tep_evaluation.py"

Before running, place the Kaggle TEP CSV file(s) somewhere under
`Tennesse dataset/data/` (see DATA_DIR below) -- see the README note at the
bottom of this file for the expected column layout.
"""

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Import the untouched AEO / PCA implementation from "p_aeo (1).py"
# ----------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PAEO_PATH = os.path.join(os.path.dirname(_THIS_DIR), "p_aeo (1).py")


def _load_paeo_module(path):
    spec = importlib.util.spec_from_file_location("p_aeo_core", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["p_aeo_core"] = module
    spec.loader.exec_module(module)
    return module


_paeo = _load_paeo_module(_PAEO_PATH)

# Reused, unmodified functions/classes:
AutoencoderAEO = _paeo.AutoencoderAEO
aeo_monitoring = _paeo.aeo_monitoring
pca_monitoring = _paeo.pca_monitoring
normalize = _paeo.normalize
set_seed = _paeo.set_seed
plot_paeo_vs_pca = _paeo.plot_paeo_vs_pca

# ----------------------------------------------------------------------
# Paths / constants
# ----------------------------------------------------------------------
DATA_DIR = os.path.join(_THIS_DIR, "data")
RESULTS_DIR = os.path.join(_THIS_DIR, "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
for _d in (RESULTS_DIR, TABLES_DIR, PLOTS_DIR, LOGS_DIR):
    os.makedirs(_d, exist_ok=True)

# Standard Rieth et al. (2017) TEP construction:
#   - "training" split: 500 samples/run, fault introduced at sample 20
#   - "testing"  split: 960 samples/run, fault introduced at sample 160
FAULT_ONSET_SAMPLE = {"train": 20, "test": 160}

# 3 minutes per sample in the real process clock
MINUTES_PER_SAMPLE = 3


# ----------------------------------------------------------------------
# 1. TEP data loading
# ----------------------------------------------------------------------
def _find_tep_csv(split, data_dir=DATA_DIR):
    """
    Locate the Kaggle TEP CSV file for a given split ("train"/"test") inside
    data_dir. The averkij Kaggle package's exact file names can vary by
    version, so we search case-insensitively for the first CSV whose name
    contains the split keyword (falling back to any single CSV present).
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"TEP data directory not found: {data_dir}\n"
            "Download 'averkij/tennessee-eastman-process-simulation-dataset' "
            "from Kaggle and place its CSV file(s) here."
        )

    csvs = [f for f in os.listdir(data_dir) if f.lower().endswith(".csv")]
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    keyword_hits = [f for f in csvs if split in f.lower()]
    if keyword_hits:
        return os.path.join(data_dir, keyword_hits[0])

    if len(csvs) == 1:
        return os.path.join(data_dir, csvs[0])

    raise FileNotFoundError(
        f"Could not unambiguously find a '{split}' CSV among {csvs} in {data_dir}. "
        "Pass an explicit csv_path to load_tep_run() instead."
    )


def _process_variable_columns(df):
    """Return the process-variable columns (xmeas_*, xmv_*), sorted naturally."""
    meas = sorted(
        [c for c in df.columns if str(c).lower().startswith("xmeas")],
        key=lambda c: int(str(c).split("_")[-1]),
    )
    mv = sorted(
        [c for c in df.columns if str(c).lower().startswith("xmv")],
        key=lambda c: int(str(c).split("_")[-1]),
    )
    cols = meas + mv
    if not cols:
        raise ValueError(
            "Could not find TEP process-variable columns (expected names like "
            "'xmeas_1', 'xmv_1'). Found columns: " + ", ".join(map(str, df.columns))
        )
    return cols


def load_tep_run(fault_number, simulation_run, split="test", csv_path=None,
                  data_dir=DATA_DIR):
    """
    Load one (fault_number, simulation_run) trajectory from the TEP dataset.

    Returns
    -------
    data : np.ndarray, shape (n_samples, n_features)
        Process-variable trajectory (rows = time samples, columns = the 52
        TEP process variables). No transpose needed -- unlike the DAA Excel
        files, the TEP CSVs are already samples x variables.
    """
    if csv_path is None:
        csv_path = _find_tep_csv(split, data_dir=data_dir)

    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]

    required = {"faultnumber", "simulationrun", "sample"}
    lower_map = {c.lower(): c for c in df.columns}
    missing = required - set(lower_map.keys())
    if missing:
        raise ValueError(f"TEP CSV at {csv_path} is missing expected columns: {missing}")

    fault_col = lower_map["faultnumber"]
    run_col = lower_map["simulationrun"]
    sample_col = lower_map["sample"]

    subset = df[(df[fault_col] == fault_number) & (df[run_col] == simulation_run)]
    if subset.empty:
        raise ValueError(
            f"No rows found for faultNumber={fault_number}, simulationRun={simulation_run} "
            f"in {csv_path}"
        )

    subset = subset.sort_values(sample_col)
    var_cols = _process_variable_columns(subset)
    return subset[var_cols].to_numpy(dtype=float)


# ----------------------------------------------------------------------
# 2. Building an AEO/PCA-ready monitoring sequence
# ----------------------------------------------------------------------
def build_monitoring_sequence(fault_free_run, faulty_run, window_size):
    """
    Stack a clean training window in front of the full faulty trajectory so
    that `aeo_monitoring`/`pca_monitoring` (which assume `data[:window_size]`
    is fault-free and monitor everything after) work completely unchanged.

    Returns
    -------
    data : np.ndarray, shape (window_size + len(faulty_run), n_features)
    fault_onset_idx : int
        Index into `data` at which the fault actually becomes active
        (ground truth for metrics).
    """
    if fault_free_run.shape[1] != faulty_run.shape[1]:
        raise ValueError("fault_free_run and faulty_run must have the same number of features")
    if window_size > fault_free_run.shape[0]:
        raise ValueError(
            f"window_size={window_size} exceeds the fault-free run length "
            f"({fault_free_run.shape[0]} samples)"
        )

    train_window = fault_free_run[:window_size]
    data = np.vstack([train_window, faulty_run])
    return data, window_size


def fault_onset_index(window_size, split="test"):
    """Ground-truth index (into the stacked `data` array) where the fault starts."""
    return window_size + FAULT_ONSET_SAMPLE[split]


# ----------------------------------------------------------------------
# 3. Metrics
# ----------------------------------------------------------------------
def evaluate_detection(T2, Cl, window_size, onset_idx):
    """
    Turn a T2 curve + control limit into standard fault-detection metrics,
    treating every monitored sample >= window_size as either "normal"
    (index < onset_idx) or "faulty" (index >= onset_idx) ground truth.

    Returns a dict with TP, FP, TN, FN, accuracy, false_alarm_rate,
    missed_detection_rate, precision, f1, and detection_delay (in samples;
    None if the fault was never flagged).
    """
    N = len(T2)
    idx = np.arange(window_size, N)
    pred_fault = T2[idx] > Cl
    true_fault = idx >= onset_idx

    tp = int(np.sum(pred_fault & true_fault))
    fp = int(np.sum(pred_fault & ~true_fault))
    tn = int(np.sum(~pred_fault & ~true_fault))
    fn = int(np.sum(~pred_fault & true_fault))

    accuracy = (tp + tn) / max(len(idx), 1)
    far = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    mdr = fn / (fn + tp) if (fn + tp) > 0 else np.nan
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    f1 = (2 * precision * (1 - mdr) / (precision + (1 - mdr))
          if (precision not in (np.nan,) and not np.isnan(precision) and (precision + (1 - mdr)) > 0)
          else np.nan)

    post_onset = np.where((idx >= onset_idx) & pred_fault)[0]
    if len(post_onset) > 0:
        first_flag_idx = idx[post_onset[0]]
        detection_delay = int(first_flag_idx - onset_idx)
    else:
        detection_delay = None

    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "accuracy": accuracy,
        "false_alarm_rate": far,
        "missed_detection_rate": mdr,
        "precision": precision,
        "f1": f1,
        "detection_delay_samples": detection_delay,
        "detection_delay_minutes": (
            detection_delay * MINUTES_PER_SAMPLE if detection_delay is not None else None
        ),
    }


# ----------------------------------------------------------------------
# 4. Multi-fault / multi-run sweep
# ----------------------------------------------------------------------
# A representative subset spanning easy/medium/hard TEP faults, including the
# classically hard-to-detect IDV(3), IDV(9), IDV(15).
DEFAULT_FAULTS = [1, 4, 6, 8, 3, 9, 15]


def run_tep_sweep(
    fault_numbers=DEFAULT_FAULTS,
    simulation_runs=(1, 2, 3),
    fault_free_run_id=1,
    split="test",
    window_size=200,
    aeo_kwargs=None,
    pca_kwargs=None,
    data_dir=DATA_DIR,
):
    """
    Run AEO and PCA monitoring over a grid of (fault_number x simulation_run),
    compute detection metrics for each, and return a tidy results DataFrame.
    """
    aeo_kwargs = dict(
        alpha=0.1, loss_alpha=1.0, beta=0.5, gamma=0.1,
        epochs=2000, lr=2e-3, latent_dim=8, verbose=False,
    ) | (aeo_kwargs or {})
    pca_kwargs = dict(Vpc=90, alpha=0.01) | (pca_kwargs or {})

    onset_idx = fault_onset_index(window_size, split=split)
    fault_free_run = load_tep_run(0, fault_free_run_id, split=split, data_dir=data_dir)

    rows = []
    for fault_number in fault_numbers:
        for run_id in simulation_runs:
            print(f"[TEP] fault={fault_number} run={run_id} ...")
            faulty_run = load_tep_run(fault_number, run_id, split=split, data_dir=data_dir)
            data, window_size_used = build_monitoring_sequence(
                fault_free_run, faulty_run, window_size
            )

            T2_aeo, detected_aeo, Cl_aeo = aeo_monitoring(
                data=data, window_size=window_size_used,
                scenario_name=f"TEP fault {fault_number} run {run_id}",
                **aeo_kwargs,
            )
            metrics_aeo = evaluate_detection(T2_aeo, Cl_aeo, window_size_used, onset_idx)
            rows.append({"fault_number": fault_number, "run": run_id, "method": "AEO", **metrics_aeo})

            T2_pca, detected_pca, Cl_pca = pca_monitoring(
                data=data, window_size=window_size_used, **pca_kwargs
            )
            metrics_pca = evaluate_detection(T2_pca, Cl_pca, window_size_used, onset_idx)
            rows.append({"fault_number": fault_number, "run": run_id, "method": "PCA", **metrics_pca})

    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(TABLES_DIR, "tep_sweep_results.csv"), index=False)
    return results


def summarize_results(results):
    """Aggregate per-fault mean +/- std across runs, for both methods."""
    summary = results.groupby(["fault_number", "method"]).agg(
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        far_mean=("false_alarm_rate", "mean"), far_std=("false_alarm_rate", "std"),
        mdr_mean=("missed_detection_rate", "mean"), mdr_std=("missed_detection_rate", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        delay_mean=("detection_delay_samples", "mean"), delay_std=("detection_delay_samples", "std"),
    ).reset_index()
    summary.to_csv(os.path.join(TABLES_DIR, "tep_sweep_summary.csv"), index=False)
    return summary


def plot_summary(summary, save_path=None):
    """Bar chart comparing AEO vs PCA accuracy per fault (extend similarly for FAR/MDR/F1)."""
    faults = sorted(summary["fault_number"].unique())
    width = 0.35
    x = np.arange(len(faults))

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    for offset, method, color in [(-width / 2, "AEO", "crimson"), (width / 2, "PCA", "royalblue")]:
        sub = summary[summary["method"] == method].set_index("fault_number").reindex(faults)
        ax.bar(x + offset, sub["accuracy_mean"], width, yerr=sub["accuracy_std"],
               label=method, color=color, alpha=0.8, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels([f"IDV({f})" for f in faults])
    ax.set_ylabel("Detection accuracy")
    ax.set_title("AEO vs PCA detection accuracy across TEP faults")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(PLOTS_DIR, "tep_accuracy_comparison.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[SAVED] {save_path}")
    plt.show()
    return fig


if __name__ == "__main__":
    results = run_tep_sweep()
    summary = summarize_results(results)
    print(summary)
    plot_summary(summary)


# ----------------------------------------------------------------------
# README (data layout expected by this script)
# ----------------------------------------------------------------------
# Place the Kaggle "averkij/tennessee-eastman-process-simulation-dataset"
# CSV file(s) under: Tennesse dataset/data/
#
# Expected columns (case-insensitive): faultNumber, simulationRun, sample,
# followed by process variables named like xmeas_1..xmeas_41, xmv_1..xmv_11.
#
# If the dataset ships as separate train/test CSVs, this loader auto-detects
# them by filename (looks for "train"/"test" substrings); if there's only one
# CSV, it's used directly regardless of the `split` argument's onset-sample
# assumption -- double check FAULT_ONSET_SAMPLE matches that file's actual
# fault-introduction sample before trusting detection-delay numbers.
