# -*- coding: utf-8 -*-
"""
Figures for the AEO vs. plain-autoencoder ablation.

    fig_ablation_p_sweep.png   mean MDR vs. retained latent dimensions p,
                               for all four arms and both monitoring indices
    fig_ablation_per_fault.png per-fault MDR at full p, showing that the mean
                               gain is concentrated in a few faults rather than
                               spread evenly

Formatted to match the manuscript figures in results/plots/paper_figures/.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_THIS = os.path.dirname(os.path.abspath(__file__))
IN_CSV = os.path.join(_THIS, "results", "tables", "ablation", "ablation_best.csv")
OUT = os.path.join(_THIS, "results", "plots", "ablation")
os.makedirs(OUT, exist_ok=True)

UNDETECTABLE = {3, 9, 15}
ARM_ORDER = ["AE", "AE-sorted", "AE+wd", "AEO"]
ARM_LABEL = {
    "AE": "AE (reconstruction only)",
    "AE-sorted": "AE + post-hoc variance sort",
    "AE+wd": "AE + weight decay",
    "AEO": "AEO (variance regularizer)",
}
ARM_COLOUR = {"AE": "#8C8C8C", "AE-sorted": "#4D9221", "AE+wd": "#2166AC", "AEO": "#B2182B"}
ARM_STYLE = {"AE": ":", "AE-sorted": "-.", "AE+wd": "--", "AEO": "-"}
P_ORDER = ["2", "4", "8", "full"]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif", "axes.linewidth": 0.9,
    "xtick.direction": "in", "ytick.direction": "in", "ytick.right": True,
})


def load():
    b = pd.read_csv(IN_CSV)
    return b[~b.fault_number.isin(UNDETECTABLE)]


def fig_p_sweep(det):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), dpi=300, sharey=True)
    titles = {"T2": r"Hotelling's $T^2$ statistic",
              "T2+SPE": r"Combined $T^2$ and SPE statistics"}

    for ax, ix in zip(axes, ["T2", "T2+SPE"]):
        sub = det[det["index"] == ix]
        for arm in ARM_ORDER:
            s = sub[sub.arm == arm].groupby("p_request").mdr.mean().reindex(P_ORDER)
            ax.plot(range(len(P_ORDER)), s.values, marker="o", ms=5,
                    color=ARM_COLOUR[arm], ls=ARM_STYLE[arm],
                    lw=2.0 if arm == "AEO" else 1.3,
                    label=ARM_LABEL[arm], zorder=4 if arm == "AEO" else 3)
        ax.set_xticks(range(len(P_ORDER)))
        ax.set_xticklabels(["2", "4", "8", "all"])
        ax.set_xlabel("Retained latent dimensions $p$", fontsize=11)
        ax.set_title(titles[ix], fontsize=11)
        ax.grid(True, ls=":", lw=0.6, color="0.75")
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Mean missed-detection rate", fontsize=11)
    axes[0].legend(fontsize=8.5, loc="upper right", framealpha=0.95, edgecolor="0.55")
    fig.suptitle("Effect of the Variance-Regularization Term on Fault Detection",
                 fontsize=13.5, fontweight="bold", y=1.03)
    fig.text(0.5, 0.955, "Mean over 17 detectable TEP faults, all arms at a common "
             "1% false-alarm rate", ha="center", fontsize=9.5, color="0.30")
    fig.tight_layout()
    p = os.path.join(OUT, "fig_ablation_p_sweep.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    fig.savefig(p.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {p}")


def fig_per_fault(det):
    f = det[(det.p_request == "full") & (det["index"] == "T2+SPE")]
    piv = f.pivot(index="fault_number", columns="arm", values="mdr")
    best_ae = piv[["AE", "AE-sorted", "AE+wd"]].min(axis=1)
    order = (best_ae - piv["AEO"]).sort_values(ascending=False).index

    fig, ax = plt.subplots(figsize=(10.5, 4.4), dpi=300)
    x = np.arange(len(order))
    w = 0.38
    ax.bar(x - w / 2, best_ae.reindex(order), w, label="best AE variant",
           color="#8C8C8C", edgecolor="white", lw=0.6)
    ax.bar(x + w / 2, piv["AEO"].reindex(order), w, label="AEO",
           color=ARM_COLOUR["AEO"], edgecolor="white", lw=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels([f"IDV({i})" for i in order], fontsize=8.5, rotation=45,
                       ha="right")
    ax.set_ylabel("Missed-detection rate", fontsize=11)
    ax.set_xlabel("Fault, ordered by AEO's advantage", fontsize=11)
    ax.grid(axis="y", ls=":", lw=0.6, color="0.75")
    ax.set_axisbelow(True)
    ax.legend(fontsize=9.5, framealpha=0.95, edgecolor="0.55")
    ax.set_title("Where the Variance Regularizer Helps",
                 fontsize=13.5, fontweight="bold", pad=22)
    ax.text(0.5, 1.02, r"Combined $T^2$ and SPE statistics, full latent space, "
            "1% false-alarm rate", transform=ax.transAxes, ha="center",
            fontsize=9.5, color="0.30")
    fig.tight_layout()
    p = os.path.join(OUT, "fig_ablation_per_fault.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    fig.savefig(p.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {p}")


if __name__ == "__main__":
    det = load()
    fig_p_sweep(det)
    fig_per_fault(det)
