# -*- coding: utf-8 -*-
"""
Full TEP pipeline (extraction -> per-fault tuning -> higher-p vs PCA
comparison -> plots) for ALL 20 TEP faults, run on Modal.

Reuses the exact same local scripts (p_aeo (1).py, tep_evaluation.py,
tep_tuning.py, tep_p_sweep.py) unmodified -- they're baked into the image
via add_local_dir and imported as-is inside each container, so there is
NO logic duplication between the local 7-fault pipeline and this one.

Data isolation: this pipeline reads/writes under /data/all20/ on the shared
"tep-data" volume, so it never touches the existing /data/tep_test_subset10.csv
(the 7-fault file) or any earlier results.

Concurrency capped at 8 containers (Modal account limit is 10, leaving headroom).

Usage:
    modal run modal_tep_all20_pipeline.py::extract        # one-time data prep
    modal run modal_tep_all20_pipeline.py                 # main pipeline + download
"""

import os

import modal

app = modal.App("tep-all20-pipeline")

LOCAL_CODING_DIR = "D:/Shubham/IIT Bombay/Coding"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("pandas", "numpy", "scipy", "torch", "matplotlib", "pyreadr")
    .add_local_dir(
        LOCAL_CODING_DIR,
        remote_path="/root/coding",
        copy=True,
        ignore=lambda p: any(
            part in {"venv", "__pycache__", ".git", "Paper"} for part in p.parts
        ) or p.suffix.lower() in {".xlsx", ".pdf", ".ipynb"},
    )
)

volume = modal.Volume.from_name("tep-data")

FAULTS_ALL20 = list(range(1, 21))
RUNS = list(range(1, 11))  # run 1 = tuning, runs 2-10 = held-out validation
WINDOW_SIZE = 200
SPLIT = "test"
ALL20_DATA_DIR = "/data/all20"


# ----------------------------------------------------------------------
# Step 1: extract all 20 faults x 10 runs into their own isolated folder
# ----------------------------------------------------------------------
@app.function(image=image, volumes={"/data": volume}, memory=16384, timeout=1800)
def extract_all20():
    import pyreadr
    import pandas as pd

    os.makedirs(ALL20_DATA_DIR, exist_ok=True)

    print("Loading TEP_Faulty_Testing.RData ...")
    res = pyreadr.read_r("/data/TEP_Faulty_Testing.RData")
    df = list(res.values())[0]
    del res
    faulty_sub = df[(df["faultNumber"].isin(FAULTS_ALL20)) & (df["simulationRun"].isin(RUNS))].copy()
    del df
    print("faulty subset shape:", faulty_sub.shape)

    print("Loading TEP_FaultFree_Testing.RData ...")
    res2 = pyreadr.read_r("/data/TEP_FaultFree_Testing.RData")
    df2 = list(res2.values())[0]
    del res2
    freefault_sub = df2[df2["simulationRun"].isin(RUNS)].copy()
    del df2

    combined = pd.concat([freefault_sub, faulty_sub], ignore_index=True)
    print("combined shape:", combined.shape)
    print(combined["faultNumber"].value_counts().sort_index())

    out_path = os.path.join(ALL20_DATA_DIR, "tep_test_subset.csv")
    combined.to_csv(out_path, index=False)
    volume.commit()
    print("Saved:", out_path)
    return combined.shape


# ----------------------------------------------------------------------
# Step 2: per-fault tuning + higher-p vs PCA comparison (one container/fault)
# ----------------------------------------------------------------------
@app.function(image=image, volumes={"/data": volume}, memory=4096,
              timeout=3600, max_containers=8)
def run_fault_pipeline(fault_number: int, seed: int = 0, out_subdir: str = "results"):
    import sys
    sys.path.insert(0, "/root/coding/Tennesse dataset")
    import matplotlib
    matplotlib.use("Agg")

    import numpy as np
    import pandas as pd

    import tep_evaluation as te
    import tep_tuning as tt
    from tep_p_sweep import train_aeo_once, p_dims_for_strategy, build_control_stats, compute_T2_for_run

    data_dir = ALL20_DATA_DIR
    onset_idx = te.fault_onset_index(WINDOW_SIZE, split=SPLIT)
    fault_free_run = te.load_tep_run(0, 1, split=SPLIT, data_dir=data_dir)

    print(f"[fault {fault_number} seed {seed}] tuning AEO ...")
    aeo_cfg, aeo_val_df, aeo_trials_df = tt.tune_aeo_for_fault(
        fault_number, fault_free_run, WINDOW_SIZE, onset_idx, SPLIT,
        tuning_run=1, validation_runs=tuple(RUNS[1:]), n_trials=15, seed=seed,
        data_dir=data_dir,
    )
    print(f"[fault {fault_number}] AEO best config: {aeo_cfg}")

    print(f"[fault {fault_number}] tuning PCA ...")
    pca_cfg, pca_val_df = tt.tune_pca_for_fault(
        fault_number, fault_free_run, WINDOW_SIZE, onset_idx, SPLIT,
        tuning_run=1, validation_runs=tuple(RUNS[1:]), data_dir=data_dir,
    )
    print(f"[fault {fault_number}] PCA best config: {pca_cfg}")

    # ---- Higher-p (full latent_dim) vs PCA, on the same held-out runs ----
    model, mu, sigma, Y0, device = train_aeo_once(
        fault_free_run, WINDOW_SIZE,
        latent_dim=aeo_cfg["latent_dim"], loss_alpha=aeo_cfg["loss_alpha"],
        beta=aeo_cfg["beta"], gamma=aeo_cfg["gamma"], epochs=aeo_cfg["epochs"], lr=aeo_cfg["lr"],
    )
    p_dims = p_dims_for_strategy(Y0, ("fixed", aeo_cfg["latent_dim"]))
    Y0_selected, Y0_mean, Qy, Cl_aeo, cond_number = build_control_stats(
        Y0, p_dims, WINDOW_SIZE, aeo_cfg["alpha"]
    )
    print(f"[fault {fault_number}] higher-p = {p_dims}, Cl={Cl_aeo:.3f}, cond={cond_number:.1f}")

    higher_p_rows = []
    for run_id in RUNS[1:]:
        faulty_run = te.load_tep_run(fault_number, run_id, split=SPLIT, data_dir=data_dir)
        data, w = te.build_monitoring_sequence(fault_free_run, faulty_run, WINDOW_SIZE)

        T2_aeo = compute_T2_for_run(model, mu, sigma, device, data, w, Y0_selected, Y0_mean, Qy, p_dims)
        metrics_aeo = te.evaluate_detection(T2_aeo, Cl_aeo, w, onset_idx)
        higher_p_rows.append({"fault_number": fault_number, "method": "AEO-higher-p",
                              "run": run_id, "p_dims": p_dims, **metrics_aeo})

        T2_pca, detected_pca, Cl_pca = te.pca_monitoring(data=data, window_size=w, verbose=False, **pca_cfg)
        metrics_pca = te.evaluate_detection(T2_pca, Cl_pca, w, onset_idx)
        higher_p_rows.append({"fault_number": fault_number, "method": "PCA",
                              "run": run_id, **metrics_pca})

    higher_p_df = pd.DataFrame(higher_p_rows)

    # ---- Timeseries plot on held-out run 2 ----
    import matplotlib.pyplot as plt
    faulty_run2 = te.load_tep_run(fault_number, 2, split=SPLIT, data_dir=data_dir)
    data2, w2 = te.build_monitoring_sequence(fault_free_run, faulty_run2, WINDOW_SIZE)
    T2_aeo2 = compute_T2_for_run(model, mu, sigma, device, data2, w2, Y0_selected, Y0_mean, Qy, p_dims)
    detected_aeo2 = [i for i in range(w2, len(T2_aeo2)) if T2_aeo2[i] > Cl_aeo]
    T2_pca2, detected_pca2, Cl_pca2 = te.pca_monitoring(data=data2, window_size=w2, verbose=False, **pca_cfg)

    detected_aeo_post = [i for i in detected_aeo2 if i >= onset_idx]
    detected_pca_post = [i for i in detected_pca2 if i >= onset_idx]
    zoom_range = (max(0, onset_idx - 60), min(len(data2), onset_idx + 400))

    fig = te.plot_paeo_vs_pca(
        Ts2_pca=T2_pca2, Cl_pca=Cl_pca2, Ts2_paeo=T2_aeo2, Cl_paeo=Cl_aeo,
        detected_pca=detected_pca_post, detected_paeo=detected_aeo_post,
        scenario_num=f"IDV({fault_number}) higher-p [held-out run 2]",
        zoom_range=zoom_range, save_path=None,
    )
    ax = fig.axes[0]
    ax.axvline(onset_idx, color="black", linewidth=1.5, alpha=0.6, label="True fault onset")
    fa_aeo = [i for i in detected_aeo2 if zoom_range[0] <= i < onset_idx]
    fa_pca = [i for i in detected_pca2 if zoom_range[0] <= i < onset_idx]
    if fa_aeo:
        ax.scatter(fa_aeo, np.log10(T2_aeo2[fa_aeo] + 1e-6), marker="^", s=70,
                  facecolor="none", edgecolor="darkred", linewidth=1.5, zorder=6,
                  label="AEO pre-onset false alarm")
    if fa_pca:
        ax.scatter(fa_pca, np.log10(T2_pca2[fa_pca] + 1e-6), marker="^", s=70,
                  facecolor="none", edgecolor="darkblue", linewidth=1.5, zorder=6,
                  label="PCA pre-onset false alarm")
    ax.legend(loc="upper left", ncol=2, fontsize=8)

    out_dir = os.path.join(ALL20_DATA_DIR, out_subdir, f"fault_{fault_number}")
    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, f"tep_timeseries_fault{fault_number}_higher_p.png")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    aeo_val_df.to_csv(os.path.join(out_dir, "aeo_validation.csv"), index=False)
    pca_val_df.to_csv(os.path.join(out_dir, "pca_validation.csv"), index=False)
    aeo_trials_df.to_csv(os.path.join(out_dir, "aeo_trials.csv"), index=False)
    higher_p_df.to_csv(os.path.join(out_dir, "higher_p_vs_pca.csv"), index=False)
    pd.DataFrame([{"fault_number": fault_number, "method": "AEO", **aeo_cfg}]).to_csv(
        os.path.join(out_dir, "aeo_best_config.csv"), index=False)
    pd.DataFrame([{"fault_number": fault_number, "method": "PCA", **pca_cfg}]).to_csv(
        os.path.join(out_dir, "pca_best_config.csv"), index=False)

    aeo_acc_mean = higher_p_df[higher_p_df["method"] == "AEO-higher-p"]["accuracy"].mean()
    pd.DataFrame([{
        "fault_number": fault_number, "seed": seed, "cond_number": cond_number,
        "p_dims": p_dims, "latent_dim": aeo_cfg["latent_dim"], "epochs": aeo_cfg["epochs"],
        "aeo_accuracy_mean": aeo_acc_mean,
    }]).to_csv(os.path.join(out_dir, "run_diagnostics.csv"), index=False)

    volume.commit()
    print(f"[fault {fault_number} seed {seed}] DONE, cond={cond_number:.1f}, "
          f"aeo_acc={aeo_acc_mean:.4f}")
    return {"fault_number": fault_number, "seed": seed, "cond_number": cond_number,
            "aeo_accuracy_mean": aeo_acc_mean}


# ----------------------------------------------------------------------
# Step 3: aggregate per-fault outputs into unified tables
# ----------------------------------------------------------------------
@app.function(image=image, volumes={"/data": volume}, memory=2048, timeout=600)
def aggregate_results():
    import pandas as pd

    configs, aeo_vals, pca_vals, higher_p_all = [], [], [], []
    for fault_number in FAULTS_ALL20:
        d = os.path.join(ALL20_DATA_DIR, "results", f"fault_{fault_number}")
        if not os.path.isdir(d):
            print(f"WARNING: no results for fault {fault_number}")
            continue
        configs.append(pd.read_csv(os.path.join(d, "aeo_best_config.csv")))
        configs.append(pd.read_csv(os.path.join(d, "pca_best_config.csv")))
        aeo_vals.append(pd.read_csv(os.path.join(d, "aeo_validation.csv")))
        pca_vals.append(pd.read_csv(os.path.join(d, "pca_validation.csv")))
        higher_p_all.append(pd.read_csv(os.path.join(d, "higher_p_vs_pca.csv")))

    configs_df = pd.concat(configs, ignore_index=True)
    validation_df = pd.concat(aeo_vals + pca_vals, ignore_index=True)
    higher_p_df = pd.concat(higher_p_all, ignore_index=True)

    out_dir = os.path.join(ALL20_DATA_DIR, "results", "_aggregated")
    os.makedirs(out_dir, exist_ok=True)
    configs_df.to_csv(os.path.join(out_dir, "tep_all20_best_configs.csv"), index=False)
    validation_df.to_csv(os.path.join(out_dir, "tep_all20_tuned_validation.csv"), index=False)
    higher_p_df.to_csv(os.path.join(out_dir, "tep_all20_higher_p_vs_pca_validation.csv"), index=False)

    summary_higher_p = higher_p_df.groupby(["fault_number", "method"]).agg(
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        far_mean=("false_alarm_rate", "mean"),
        mdr_mean=("missed_detection_rate", "mean"),
        f1_mean=("f1", "mean"),
        delay_mean=("detection_delay_samples", "mean"), delay_std=("detection_delay_samples", "std"),
    ).reset_index()
    summary_higher_p.to_csv(os.path.join(out_dir, "tep_all20_higher_p_vs_pca_summary.csv"), index=False)

    volume.commit()
    print(summary_higher_p.to_string())
    return summary_higher_p.shape


@app.local_entrypoint()
def main():
    print("Step 1: extracting all 20 faults x 10 runs ...")
    extract_all20.remote()

    print("Step 2: running per-fault pipeline (max 8 concurrent containers) ...")
    results = list(run_fault_pipeline.map(FAULTS_ALL20))
    print("Completed faults:", sorted(r["fault_number"] for r in results))

    print("Step 3: aggregating ...")
    aggregate_results.remote()


@app.local_entrypoint(name="retry_unstable")
def retry_unstable():
    """
    Re-tune faults 6 and 15 with multiple search seeds (same Modal environment
    for both, unlike the original mixed local/Modal comparison) to find a
    config that is BOTH accurate and numerically stable (reasonable covariance
    condition number), instead of trusting whichever single seed happened to
    win once. Writes each attempt to its own out_subdir so nothing overwrites
    the original all20 results until we've picked a winner.
    """
    faults_to_retry = [6, 15]
    seeds = [0, 1, 2, 3, 4]
    jobs = [(f, s, f"results_retry/fault_{f}_seed_{s}") for f in faults_to_retry for s in seeds]

    print(f"Running {len(jobs)} retry jobs (max 8 concurrent) ...")
    results = list(run_fault_pipeline.starmap(jobs))

    print("\n=== Retry results (fault, seed, cond_number, aeo_accuracy_mean) ===")
    for r in sorted(results, key=lambda r: (r["fault_number"], r["seed"])):
        flag = "  <-- UNSTABLE (cond > 10000)" if r["cond_number"] > 10000 else ""
        print(f"fault={r['fault_number']} seed={r['seed']} cond={r['cond_number']:.1f} "
              f"acc={r['aeo_accuracy_mean']:.4f}{flag}")

    print("Done. Pull results with:")
    print('  modal volume get tep-data all20/results/_aggregated <local_dir>')
    print('  modal volume get tep-data all20/results <local_dir>   # includes all plots')


@app.local_entrypoint(name="extract")
def extract_only():
    extract_all20.remote()
