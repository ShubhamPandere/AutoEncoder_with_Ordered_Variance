# -*- coding: utf-8 -*-
"""
Time-series (T^2 vs. sample index) plots for the TEP faults, in the exact
same style as the DAA Scenario 1/2 comparison figures in `p_aeo (1).py`
(`plot_paeo_vs_pca`): log10 T^2 curves for AEO and PCA overlaid, control
limits, first-detection markers, and the AEO-vs-PCA lead-time annotation.

Uses the TUNED hyperparameters found by tep_tuning.py (results/tep_best_configs.csv),
evaluated on a held-out run (not the tuning run) so the plot reflects genuine
generalization, not a cherry-picked fit.
"""

import ast
import os

import numpy as np
import pandas as pd

from tep_evaluation import (
    load_tep_run, build_monitoring_sequence, fault_onset_index,
    aeo_monitoring, pca_monitoring, plot_paeo_vs_pca,
    DEFAULT_FAULTS, RESULTS_DIR, TABLES_DIR, PLOTS_DIR, DATA_DIR,
)

AEO_CONFIG_KEYS = ["latent_dim", "loss_alpha", "beta", "gamma", "lr", "alpha", "epochs"]
PCA_CONFIG_KEYS = ["Vpc", "alpha"]


def _row_to_kwargs(row, keys):
    kwargs = {}
    for k in keys:
        v = row[k]
        if pd.isna(v):
            continue
        if k == "latent_dim":
            v = int(v)
        if k == "epochs":
            v = int(v)
        kwargs[k] = v
    return kwargs


def load_best_configs(path=None):
    path = path or os.path.join(TABLES_DIR, "tep_best_configs.csv")
    df = pd.read_csv(path)
    aeo_cfgs, pca_cfgs = {}, {}
    for _, row in df.iterrows():
        fault = int(row["fault_number"])
        if row["method"] == "AEO":
            aeo_cfgs[fault] = _row_to_kwargs(row, AEO_CONFIG_KEYS)
        elif row["method"] == "PCA":
            pca_cfgs[fault] = _row_to_kwargs(row, PCA_CONFIG_KEYS)
    return aeo_cfgs, pca_cfgs


def plot_fault_timeseries(
    fault_number, aeo_cfg, pca_cfg,
    plot_run=2, fault_free_run_id=1, split="test", window_size=200,
    zoom_margin_pre=60, zoom_margin_post=400, data_dir=DATA_DIR,
):
    """Run tuned AEO + PCA on a held-out run and plot T^2 vs. time."""
    onset_idx = fault_onset_index(window_size, split=split)
    fault_free_run = load_tep_run(0, fault_free_run_id, split=split, data_dir=data_dir)
    faulty_run = load_tep_run(fault_number, plot_run, split=split, data_dir=data_dir)
    data, w = build_monitoring_sequence(fault_free_run, faulty_run, window_size)

    T2_aeo, detected_aeo, Cl_aeo = aeo_monitoring(
        data=data, window_size=w, scenario_name=f"IDV({fault_number})",
        verbose=False, **aeo_cfg,
    )
    T2_pca, detected_pca, Cl_pca = pca_monitoring(
        data=data, window_size=w, verbose=False, **pca_cfg,
    )

    # NOTE: `detected_aeo`/`detected_pca` as returned by aeo_monitoring/pca_monitoring
    # are ALL post-window_size alarms, including any pre-onset false alarms. The
    # DAA-era plot_paeo_vs_pca annotates `detected[0]` as "first detection" assuming
    # no pre-onset false alarms (true for the clean DAA runs). TEP does have pre-onset
    # false alarms for both methods (matching the false_alarm_rate we measured), so we
    # filter to onset-or-later here to keep the annotation consistent with
    # evaluate_detection()'s detection_delay_samples definition.
    detected_aeo_post_onset = [i for i in detected_aeo if i >= onset_idx]
    detected_pca_post_onset = [i for i in detected_pca if i >= onset_idx]

    zoom_range = (max(0, onset_idx - zoom_margin_pre), min(len(data), onset_idx + zoom_margin_post))
    save_path = os.path.join(PLOTS_DIR, f"tep_timeseries_fault{fault_number}.png")

    fig = plot_paeo_vs_pca(
        Ts2_pca=T2_pca, Cl_pca=Cl_pca,
        Ts2_paeo=T2_aeo, Cl_paeo=Cl_aeo,
        detected_pca=detected_pca_post_onset, detected_paeo=detected_aeo_post_onset,
        scenario_num=f"IDV({fault_number}) [held-out run {plot_run}]",
        zoom_range=zoom_range, save_path=None,  # save after annotating below
    )

    # Add: (1) a marker for the TRUE fault onset (independent of either method's
    # own "first detection" line), and (2) visible markers for any PRE-onset false
    # alarms inside the zoom window, so those aren't silently hidden.
    ax = fig.axes[0]
    ax.axvline(onset_idx, color="black", linestyle="-", linewidth=1.5, alpha=0.6,
               label="True fault onset", zorder=1)

    zoom_lo, zoom_hi = zoom_range
    fa_aeo = [i for i in detected_aeo if zoom_lo <= i < onset_idx]
    fa_pca = [i for i in detected_pca if zoom_lo <= i < onset_idx]
    if fa_aeo:
        ax.scatter(fa_aeo, np.log10(T2_aeo[fa_aeo] + 1e-6), marker="^", s=70,
                   facecolor="none", edgecolor="darkred", linewidth=1.5, zorder=6,
                   label="AEO pre-onset false alarm")
    if fa_pca:
        ax.scatter(fa_pca, np.log10(T2_pca[fa_pca] + 1e-6), marker="^", s=70,
                   facecolor="none", edgecolor="darkblue", linewidth=1.5, zorder=6,
                   label="PCA pre-onset false alarm")

    ax.legend(loc="upper left", ncol=2, fontsize=8)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[SAVED] {save_path}")

    return fig, T2_aeo, T2_pca, Cl_aeo, Cl_pca, detected_aeo, detected_pca


def plot_all(fault_numbers=DEFAULT_FAULTS, plot_run=2):
    aeo_cfgs, pca_cfgs = load_best_configs()
    for fault_number in fault_numbers:
        print(f"[plot] IDV({fault_number}) ...")
        plot_fault_timeseries(
            fault_number, aeo_cfgs[fault_number], pca_cfgs[fault_number],
            plot_run=plot_run,
        )


if __name__ == "__main__":
    plot_all()
