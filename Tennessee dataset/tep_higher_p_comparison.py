# -*- coding: utf-8 -*-
"""
Re-run the AEO vs PCA comparison using "higher p" (= full latent_dim, no
90%-cutoff truncation) as AEO's dimension-selection rule instead of the
original default. PCA's tuned config is unchanged. Evaluated across all 9
held-out validation runs (2-10) per fault.

Results are written to their own p_full_latent/ subfolders (under both
results/tables/ and results/plots/) -- the earlier baseline_90pct results
are left untouched (archived separately under results/plots/p_baseline_90pct/).
"""

import os

import numpy as np
import pandas as pd

from tep_evaluation import (
    load_tep_run, build_monitoring_sequence, fault_onset_index,
    evaluate_detection, pca_monitoring, plot_paeo_vs_pca,
    DATA_DIR, TABLES_DIR, PLOTS_DIR, DEFAULT_FAULTS,
)
from tep_timeseries import load_best_configs
from tep_p_sweep import (
    train_aeo_once, p_dims_for_strategy, build_control_stats,
    compute_T2_for_run, VALIDATION_RUNS,
)

P_STRATEGY_NAME = "p_full_latent"

NEW_TABLES_DIR = os.path.join(TABLES_DIR, P_STRATEGY_NAME)
NEW_PLOTS_DIR = os.path.join(PLOTS_DIR, P_STRATEGY_NAME)
os.makedirs(NEW_TABLES_DIR, exist_ok=True)
os.makedirs(NEW_PLOTS_DIR, exist_ok=True)


def run_higher_p_vs_pca(fault_numbers=DEFAULT_FAULTS, window_size=200, split="test",
                         fault_free_run_id=1, validation_runs=VALIDATION_RUNS):
    aeo_cfgs, pca_cfgs = load_best_configs()
    onset_idx = fault_onset_index(window_size, split=split)
    fault_free_run = load_tep_run(0, fault_free_run_id, split=split, data_dir=DATA_DIR)

    rows = []
    trained = {}
    for fault_number in fault_numbers:
        cfg = aeo_cfgs[fault_number]
        model, mu, sigma, Y0, device = train_aeo_once(
            fault_free_run, window_size,
            latent_dim=cfg["latent_dim"], loss_alpha=cfg["loss_alpha"],
            beta=cfg["beta"], gamma=cfg["gamma"], epochs=cfg["epochs"], lr=cfg["lr"],
        )
        # "higher p" = use the FULL latent_dim, i.e. don't truncate via the
        # 90%-cumulative-variance rule at all.
        p_dims = p_dims_for_strategy(Y0, ("fixed", cfg["latent_dim"]))
        Y0_selected, Y0_mean, Qy, Cl_aeo, cond_number = build_control_stats(
            Y0, p_dims, window_size, cfg["alpha"]
        )
        trained[fault_number] = dict(
            model=model, mu=mu, sigma=sigma, device=device,
            Y0_selected=Y0_selected, Y0_mean=Y0_mean, Qy=Qy,
            Cl=Cl_aeo, p_dims=p_dims, cfg=cfg,
        )
        print(f"[fault {fault_number}] higher-p = {p_dims} (latent_dim={cfg['latent_dim']}), "
              f"Cl={Cl_aeo:.3f}, cond={cond_number:.1f}")

        pca_cfg = pca_cfgs[fault_number]

        for run_id in validation_runs:
            faulty_run = load_tep_run(fault_number, run_id, split=split, data_dir=DATA_DIR)
            data, w = build_monitoring_sequence(fault_free_run, faulty_run, window_size)

            T2_aeo = compute_T2_for_run(model, mu, sigma, device, data, w,
                                         Y0_selected, Y0_mean, Qy, p_dims)
            metrics_aeo = evaluate_detection(T2_aeo, Cl_aeo, w, onset_idx)
            rows.append({"fault_number": fault_number, "method": "AEO-higher-p",
                         "run": run_id, "p_dims": p_dims, **metrics_aeo})

            T2_pca, detected_pca, Cl_pca = pca_monitoring(
                data=data, window_size=w, verbose=False, **pca_cfg
            )
            metrics_pca = evaluate_detection(T2_pca, Cl_pca, w, onset_idx)
            rows.append({"fault_number": fault_number, "method": "PCA",
                         "run": run_id, **metrics_pca})

    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(NEW_TABLES_DIR, "tep_higher_p_vs_pca_validation.csv"), index=False)
    return results, trained


def summarize(results):
    summary = results.groupby(["fault_number", "method"]).agg(
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        far_mean=("false_alarm_rate", "mean"),
        mdr_mean=("missed_detection_rate", "mean"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        delay_mean=("detection_delay_samples", "mean"), delay_std=("detection_delay_samples", "std"),
    ).reset_index()
    summary.to_csv(os.path.join(NEW_TABLES_DIR, "tep_higher_p_vs_pca_summary.csv"), index=False)
    return summary


def plot_accuracy_bar(summary):
    import matplotlib.pyplot as plt
    faults = sorted(summary["fault_number"].unique())
    width = 0.35
    x = np.arange(len(faults))
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    for offset, method, color in [(-width / 2, "AEO-higher-p", "crimson"), (width / 2, "PCA", "royalblue")]:
        sub = summary[summary["method"] == method].set_index("fault_number").reindex(faults)
        ax.bar(x + offset, sub["accuracy_mean"], width, yerr=sub["accuracy_std"],
               label=method, color=color, alpha=0.85, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"IDV({f})" for f in faults])
    ax.set_ylabel("Detection accuracy (9 held-out runs)")
    ax.set_title("AEO (higher p = full latent_dim) vs PCA - TEP faults")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    path = os.path.join(NEW_PLOTS_DIR, "tep_accuracy_comparison_higher_p.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"[SAVED] {path}")


def plot_fault_timeseries_higher_p(fault_number, trained, pca_cfg, run_id=2, window_size=200,
                                    split="test", fault_free_run_id=1,
                                    zoom_margin_pre=60, zoom_margin_post=400):
    onset_idx = fault_onset_index(window_size, split=split)
    fault_free_run = load_tep_run(0, fault_free_run_id, split=split, data_dir=DATA_DIR)
    faulty_run = load_tep_run(fault_number, run_id, split=split, data_dir=DATA_DIR)
    data, w = build_monitoring_sequence(fault_free_run, faulty_run, window_size)

    t = trained[fault_number]
    T2_aeo = compute_T2_for_run(t["model"], t["mu"], t["sigma"], t["device"], data, w,
                                 t["Y0_selected"], t["Y0_mean"], t["Qy"], t["p_dims"])
    detected_aeo = [i for i in range(w, len(T2_aeo)) if T2_aeo[i] > t["Cl"]]

    T2_pca, detected_pca, Cl_pca = pca_monitoring(data=data, window_size=w, verbose=False, **pca_cfg)

    detected_aeo_post = [i for i in detected_aeo if i >= onset_idx]
    detected_pca_post = [i for i in detected_pca if i >= onset_idx]

    zoom_range = (max(0, onset_idx - zoom_margin_pre), min(len(data), onset_idx + zoom_margin_post))
    fig = plot_paeo_vs_pca(
        Ts2_pca=T2_pca, Cl_pca=Cl_pca, Ts2_paeo=T2_aeo, Cl_paeo=t["Cl"],
        detected_pca=detected_pca_post, detected_paeo=detected_aeo_post,
        scenario_num=f"IDV({fault_number}) higher-p [held-out run {run_id}]",
        zoom_range=zoom_range, save_path=None,
    )
    ax = fig.axes[0]
    ax.axvline(onset_idx, color="black", linestyle="-", linewidth=1.5, alpha=0.6, label="True fault onset")
    fa_aeo = [i for i in detected_aeo if zoom_range[0] <= i < onset_idx]
    fa_pca = [i for i in detected_pca if zoom_range[0] <= i < onset_idx]
    if fa_aeo:
        ax.scatter(fa_aeo, np.log10(T2_aeo[fa_aeo] + 1e-6), marker="^", s=70,
                   facecolor="none", edgecolor="darkred", linewidth=1.5, zorder=6,
                   label="AEO pre-onset false alarm")
    if fa_pca:
        ax.scatter(fa_pca, np.log10(T2_pca[fa_pca] + 1e-6), marker="^", s=70,
                   facecolor="none", edgecolor="darkblue", linewidth=1.5, zorder=6,
                   label="PCA pre-onset false alarm")
    ax.legend(loc="upper left", ncol=2, fontsize=8)

    save_path = os.path.join(NEW_PLOTS_DIR, f"tep_timeseries_fault{fault_number}_higher_p.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[SAVED] {save_path}")


if __name__ == "__main__":
    results, trained = run_higher_p_vs_pca()
    summary = summarize(results)
    pd.set_option("display.width", 200)
    print("\n=== HIGHER-P AEO vs PCA SUMMARY (mean over 9 held-out runs) ===")
    print(summary.to_string())

    plot_accuracy_bar(summary)
    _, pca_cfgs = load_best_configs()
    for fault_number in trained:
        plot_fault_timeseries_higher_p(fault_number, trained, pca_cfgs[fault_number])
