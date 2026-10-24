# -*- coding: utf-8 -*-
"""
Figures for the corrected TEP comparison.

Three panels, each answering one of the questions the audit raised:

  fig1  before/after delay -- shows the old PCA "speed advantage" was bought
        with a 10-40x higher false-alarm rate, and disappears at matched FAR.
  fig2  MDR by method and index -- shows the IDV(4)-class faults are a
        residual-subspace problem, fixed for AEO by adding SPE.
  fig3  the delay-vs-FAR trade-off curve that makes the confound visible.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_THIS = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(_THIS, "results", "tables")
OUT = os.path.join(_THIS, "results", "plots", "corrected")
os.makedirs(OUT, exist_ok=True)

DETECTABLE = [1, 4, 6, 8]         # 3/9/15 are undetectable by any T2/SPE method
UNDETECTABLE = [3, 9, 15]
FAULTS = DETECTABLE + UNDETECTABLE  # grouped so the shaded block is contiguous

C_AEO, C_PCA = "#C43E5B", "#3E6FC4"


def _load():
    old = pd.read_csv(os.path.join(TABLES, "all20", "tep_all20_higher_p_vs_pca_summary.csv"))
    new = pd.read_csv(os.path.join(TABLES, "corrected", "comparison.csv"))
    return old, new


def fig_before_after(old, new):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150, sharey=True)
    x = np.arange(len(DETECTABLE))
    w = 0.35

    # --- left: old protocol ---
    ax = axes[0]
    oa = [old[(old.fault_number == f) & (old.method == "AEO-higher-p")].delay_mean.iloc[0] for f in DETECTABLE]
    op = [old[(old.fault_number == f) & (old.method == "PCA")].delay_mean.iloc[0] for f in DETECTABLE]
    fa = [old[(old.fault_number == f) & (old.method == "AEO-higher-p")].far_mean.iloc[0] for f in DETECTABLE]
    fp = [old[(old.fault_number == f) & (old.method == "PCA")].far_mean.iloc[0] for f in DETECTABLE]
    ax.bar(x - w / 2, oa, w, label="AEO", color=C_AEO)
    ax.bar(x + w / 2, op, w, label="PCA", color=C_PCA)
    for i, (a, p_, ffa, ffp) in enumerate(zip(oa, op, fa, fp)):
        ax.text(i - w / 2, a + ax.get_ylim()[1]*0.035, f"{a:.1f}\nFAR {ffa:.3f}", ha="center",
                fontsize=7.5, color=C_AEO)
        ax.text(i + w / 2, p_ + ax.get_ylim()[1]*0.035, f"{p_:.1f}\nFAR {ffp:.3f}", ha="center",
                fontsize=7.5, color=C_PCA)
    ax.set_title("BEFORE — unmatched false-alarm rate\n(PCA looks faster)", fontsize=11)
    ax.set_ylabel("detection delay (samples)")

    # --- right: corrected protocol ---
    ax = axes[1]
    na = [new[(new.fault_number == f) & (new.variant == "AEO-T2")].delay_samples.iloc[0] for f in DETECTABLE]
    npc = [new[(new.fault_number == f) & (new.variant == "PCA-T2 (Vpc<=95)")].delay_samples.iloc[0] for f in DETECTABLE]
    ax.bar(x - w / 2, na, w, label="AEO-T²", color=C_AEO)
    ax.bar(x + w / 2, npc, w, label="PCA-T²", color=C_PCA)
    for i, (a, p_) in enumerate(zip(na, npc)):
        ax.text(i - w / 2, a + ax.get_ylim()[1]*0.035, f"{a:.1f}", ha="center", fontsize=7.5, color=C_AEO)
        ax.text(i + w / 2, p_ + ax.get_ylim()[1]*0.035, f"{p_:.1f}", ha="center", fontsize=7.5, color=C_PCA)
    ax.set_title("AFTER — both at measured FAR = 0.010\n"
                 "(gap closes on IDV 1/8; IDV 4 is a real AEO weakness)", fontsize=11)
    ax.set_ylabel("detection delay (samples)")

    axes[0].set_ylim(0, max(max(oa + op), max(na + npc)) * 1.28)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([f"IDV({f})" for f in DETECTABLE])
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(fontsize=9)
        ax.set_axisbelow(True)

    fig.suptitle("Most of PCA's apparent speed advantage was a false-alarm-rate artifact",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(OUT, "fig1_delay_before_after.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {p}")


def fig_mdr_by_index(new):
    variants = ["AEO-T2", "PCA-T2 (Vpc<=95)", "AEO-T2+SPE", "PCA-T2+SPE"]
    colors = [C_AEO, C_PCA, C_AEO, C_PCA]
    hatches = ["", "", "//", "//"]

    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=150)
    x = np.arange(len(FAULTS))
    w = 0.2
    for i, (v, c, h) in enumerate(zip(variants, colors, hatches)):
        vals = [new[(new.fault_number == f) & (new.variant == v)].mdr.iloc[0] for f in FAULTS]
        ax.bar(x + (i - 1.5) * w, vals, w, label=v, color=c, hatch=h,
               edgecolor="white", linewidth=0.6)

    ax.axvspan(len(DETECTABLE) - 0.5, len(FAULTS) - 0.5, color="grey", alpha=0.10)
    ax.set_ylim(0, 1.22)
    ax.text((len(DETECTABLE) + len(FAULTS) - 1) / 2, 1.08,
            "undetectable by any T²/SPE method\n(known negative control)",
            ha="center", fontsize=9, color="dimgrey", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels([f"IDV({f})" for f in FAULTS])
    ax.set_ylabel("missed-detection rate")
    ax.set_title("Adding a residual (SPE) statistic is what fixes IDV(4) for AEO\n"
                 "(all methods at measured FAR = 0.010)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(OUT, "fig2_mdr_by_index.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {p}")


def fig_delay_vs_far():
    """The confound itself: PCA delay on IDV(1) as a pure function of FAR."""
    import sys
    sys.path.insert(0, _THIS)
    import tep_data as td
    import tep_models as tm
    import tep_protocol as tp

    Xtr = td.get_run(0, tp.TRAIN_RUN)[:tp.WINDOW_SIZE]
    far_runs = {r: td.get_run(0, r) for r in tp.FAR_RUNS}

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=150)
    for Vpc, style in ((85, "--"), (95, "-")):
        mon = tm.PCAMonitor(Vpc=Vpc).fit(Xtr)
        pooled = np.concatenate([mon.statistics(v)[0] for v in far_runs.values()])
        fa = {r: mon.statistics(v)[0] for r, v in far_runs.items()}
        fl = {r: mon.statistics(td.get_run(1, r))[0] for r in tp.VAL_RUNS}
        xs, ys = [], []
        for target in np.geomspace(0.001, 0.30, 22):
            cl = tp.empirical_control_limit(pooled, target)
            m = tp.evaluate_detection_v2(fl, fa, cl)
            if m["delay_samples"] is not None:
                xs.append(m["far"])
                ys.append(m["delay_samples"])
        ax.plot(xs, ys, style, marker="o", ms=3.5, color=C_PCA,
                label=f"PCA-T² (Vpc={Vpc}, Np={mon.Np})")

    ax.axvline(0.01, color="black", linestyle=":", linewidth=1.4)
    ax.annotate("corrected protocol\noperating point", xy=(0.01, 0.04),
                xycoords=("data", "axes fraction"), xytext=(7, 0),
                textcoords="offset points", fontsize=8, va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.9))
    ax.set_xscale("log")
    ax.set_xlabel("measured false-alarm rate (log scale)")
    ax.set_ylabel("detection delay (samples), IDV(1)")
    ax.set_title("Delay falls monotonically as false alarms are allowed\n"
                 "— so delay must be compared at matched FAR",
                 fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(OUT, "fig3_delay_vs_far.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {p}")


if __name__ == "__main__":
    old, new = _load()
    fig_before_after(old, new)
    fig_mdr_by_index(new)
    fig_delay_vs_far()
