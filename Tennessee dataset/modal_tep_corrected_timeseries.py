# -*- coding: utf-8 -*-
"""
Time-series (monitoring-statistic vs time) plots for all 20 TEP faults,
under the corrected protocol.
=======================================================================

Produces, per fault, a two-panel figure on a held-out validation run:

  top    T2 only        -- AEO-T2 vs PCA-T2 (Vpc <= 95), the like-for-like pair
  bottom T2 + SPE       -- AEO vs PCA with the residual statistic added

Every curve is plotted as **statistic / its own control limit**, so the
threshold is the horizontal line at 1.0 for all four monitors. That matters:
the raw statistics live on wildly different scales (on DAA the AEO T2 runs
~10^4x larger than PCA's), and plotting raw values against a shared axis is
exactly what made the original DAA "lead" look real. Normalising by each
method's own limit is the only way a shared axis is meaningful.

Also emits a 20-fault overview grid.

Configs come from the winning rows in the corrected all-20 comparison, which
this script reads off the volume. The AEO model is fault-independent, so each
unique winning config is trained once and reused for every fault that selected
it.

Usage:
    modal run modal_tep_corrected_timeseries.py
"""

import os

import modal

app = modal.App("tep-corrected-timeseries")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

CODE_FILES = [
    "tep_protocol.py",
    "tep_models.py",
    "tep_data.py",
    "tep_corrected_pipeline.py",
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("pandas", "numpy", "scipy", "torch", "matplotlib")
    .add_local_dir(_THIS_DIR, remote_path="/root/code", copy=True,
                   ignore=lambda p: p.name not in CODE_FILES)
)

volume = modal.Volume.from_name("tep-data")

DATA_CSV = "/data/all20/tep_test_subset.csv"
IN_DIR = "/data/all20/corrected"
OUT_DIR = "/data/all20/corrected/plots"
FAULTS_ALL20 = list(range(1, 21))
PLOT_RUN = 4          # first held-out validation run

C_AEO, C_PCA = "#C43E5B", "#3E6FC4"


@app.function(image=image, volumes={"/data": volume}, memory=16384, timeout=5400)
def make_timeseries():
    import sys
    sys.path.insert(0, "/root/code")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    import tep_data as td
    import tep_models as tm
    import tep_protocol as tp

    volume.reload()
    os.makedirs(OUT_DIR, exist_ok=True)
    comp = pd.read_csv(os.path.join(IN_DIR, "comparison.csv"))

    Xtr = td.get_run(0, tp.TRAIN_RUN, DATA_CSV)[:tp.WINDOW_SIZE]
    calib_X = np.vstack([td.get_run(0, r, DATA_CSV) for r in tp.CALIB_RUNS])
    onset = tp.ONSET

    # ---- train each unique winning AEO config once (model is fault-agnostic) ----
    aeo_rows = comp[comp["variant"].isin(["AEO-T2", "AEO-T2+SPE"])]
    cfg_cols = ["latent_dim", "hidden", "epochs", "q_shape",
                "loss_alpha", "beta", "gamma", "lr"]
    uniq = aeo_rows.drop_duplicates("trial")[["trial"] + cfg_cols]
    print(f"training {len(uniq)} unique AEO configs ...", flush=True)

    aeo_models = {}
    for _, r in uniq.iterrows():
        cfg = dict(latent_dim=int(r.latent_dim), hidden=int(r.hidden),
                   epochs=int(r.epochs), q_shape=str(r.q_shape),
                   loss_alpha=float(r.loss_alpha), beta=float(r.beta),
                   gamma=float(r.gamma), lr=float(r.lr))
        aeo_models[int(r.trial)] = tm.AEOMonitor(model_seed=42, **cfg).fit(Xtr)
        print(f"  trial {int(r.trial)} trained", flush=True)

    pca_models = {int(v): tm.PCAMonitor(Vpc=int(v)).fit(Xtr)
                  for v in sorted(comp[comp["variant"].str.startswith("PCA")].Vpc.unique())}

    def limits(mon):
        t2c, spec = mon.statistics(calib_X)
        cl_t2 = tp.empirical_control_limit(t2c, tp.TARGET_FAR)
        cl_t2c, cl_spec = tp.combined_control_limits(t2c, spec, tp.TARGET_FAR)
        return cl_t2, cl_t2c, cl_spec

    lim_aeo = {k: limits(m) for k, m in aeo_models.items()}
    lim_pca = {k: limits(m) for k, m in pca_models.items()}

    def curves(fault):
        """Return normalised (statistic / own limit) curves for one fault."""
        X = td.get_run(fault, PLOT_RUN, DATA_CSV)
        out = {}
        for variant, store, lims in (("AEO-T2", aeo_models, lim_aeo),
                                     ("AEO-T2+SPE", aeo_models, lim_aeo)):
            row = comp[(comp.fault_number == fault) & (comp.variant == variant)].iloc[0]
            mon = store[int(row.trial)]
            cl_t2, cl_t2c, cl_spec = lims[int(row.trial)]
            t2, spe = mon.statistics(X)
            out[variant] = t2 / cl_t2 if variant.endswith("T2") else \
                np.maximum(t2 / cl_t2c, spe / cl_spec)
        for variant in ("PCA-T2 (Vpc<=95)", "PCA-T2+SPE"):
            row = comp[(comp.fault_number == fault) & (comp.variant == variant)].iloc[0]
            mon = pca_models[int(row.Vpc)]
            cl_t2, cl_t2c, cl_spec = lim_pca[int(row.Vpc)]
            t2, spe = mon.statistics(X)
            out[variant] = t2 / cl_t2 if variant.endswith("T2)") else \
                np.maximum(t2 / cl_t2c, spe / cl_spec)
        return out

    def first_det(y):
        r = tp.first_sustained_alarm(np.asarray(y)[onset:] > 1.0, k=tp.CONSEC_K)
        return onset + r if r is not None else None

    all_curves = {}
    for fault in FAULTS_ALL20:
        cv = curves(fault)
        all_curves[fault] = cv

        fig, axes = plt.subplots(2, 1, figsize=(11, 7), dpi=150, sharex=True)
        panels = [("T² only  (like-for-like)", "AEO-T2", "PCA-T2 (Vpc<=95)"),
                  ("T² + SPE  (standard protocol)", "AEO-T2+SPE", "PCA-T2+SPE")]
        for ax, (title, ka, kp) in zip(axes, panels):
            for key, colour, label in ((ka, C_AEO, "AEO"), (kp, C_PCA, "PCA")):
                y = np.clip(cv[key], 1e-3, None)
                ax.semilogy(y, color=colour, lw=1.2, alpha=0.85, label=label)
                d = first_det(cv[key])
                if d is not None:
                    ax.axvline(d, color=colour, ls=":", lw=1.4)
                    ax.plot(d, y[d], marker="v", color=colour, ms=8, zorder=5)
                    ax.text(d, ax.get_ylim()[1], f" {label} +{d-onset}",
                            color=colour, fontsize=8, va="top", fontweight="bold")
            ax.axhline(1.0, color="black", ls="--", lw=1.3)
            ax.text(0.995, 1.0, " control limit", transform=ax.get_yaxis_transform(),
                    ha="right", va="bottom", fontsize=7.5)
            ax.axvline(onset, color="black", lw=1.6, alpha=0.6)
            ax.axvspan(0, onset, color="grey", alpha=0.08)
            ax.set_title(title, fontsize=10, loc="left")
            ax.set_ylabel("statistic / limit")
            ax.grid(True, which="major", ls="--", alpha=0.35)
            ax.legend(fontsize=8, loc="lower right")

        axes[0].text(onset / 2, axes[0].get_ylim()[1], "normal ", ha="right",
                     va="top", fontsize=8, color="dimgrey", style="italic")
        axes[1].set_xlabel(f"sample index  (fault starts at {onset}; held-out run {PLOT_RUN})")
        fig.suptitle(f"IDV({fault}) — corrected protocol, all monitors at measured FAR = 0.010",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        p = os.path.join(OUT_DIR, f"timeseries_fault{fault:02d}.png")
        fig.savefig(p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[SAVED] {p}", flush=True)

    # ---- overview grid: combined index, all 20 faults ----
    fig, axes = plt.subplots(5, 4, figsize=(18, 16), dpi=130, sharex=True)
    for ax, fault in zip(axes.ravel(), FAULTS_ALL20):
        cv = all_curves[fault]
        for key, colour, label in (("AEO-T2+SPE", C_AEO, "AEO"),
                                   ("PCA-T2+SPE", C_PCA, "PCA")):
            ax.semilogy(np.clip(cv[key], 1e-3, None), color=colour, lw=0.9,
                        alpha=0.85, label=label)
        ax.axhline(1.0, color="black", ls="--", lw=1.0)
        ax.axvline(onset, color="black", lw=1.2, alpha=0.6)
        ax.axvspan(0, onset, color="grey", alpha=0.08)
        ax.set_title(f"IDV({fault})", fontsize=10, fontweight="bold")
        ax.grid(True, ls="--", alpha=0.3)
    axes.ravel()[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("All 20 TEP faults — AEO vs PCA (T²+SPE), corrected protocol, "
                 "curves normalised so the control limit is 1.0",
                 fontsize=15, fontweight="bold")
    fig.supxlabel(f"sample index (fault starts at {onset}; held-out run {PLOT_RUN})")
    fig.supylabel("statistic / control limit")
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "timeseries_all20_grid.png")
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {p}", flush=True)

    volume.commit()
    return len(FAULTS_ALL20)


@app.local_entrypoint()
def main():
    n = make_timeseries.remote()
    print(f"done: {n} fault figures + overview grid")
    print("Pull with: modal volume get tep-data all20/corrected/plots <local_dir>")
