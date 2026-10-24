# -*- coding: utf-8 -*-
"""
Publication-ready monitoring-chart figures for all 20 TEP faults.
=================================================================

One folder per fault, two standalone figures inside it:

    Fault_01/IDV01_T2.png       Hotelling's T^2 statistic
    Fault_01/IDV01_T2_SPE.png   Combined T^2 and SPE statistics

Both are formatted for direct inclusion in the manuscript: serif type, figure
titles rather than working labels, a secondary time axis in hours, and an
in-figure performance box giving detection delay and missed-detection rate
averaged over the held-out runs, plus an explicit statement of which method
leads and by how much.

Curves are cached to `curves.npz` on the volume the first time they are
computed, so restyling the figures afterwards costs nothing -- rerun with
`--replot` and no model is retrained.

Statistics are plotted normalised by each method's own control limit, so the
threshold is a single line at 1.0 and the two methods are directly comparable
on one axis. The raw statistics differ by orders of magnitude between methods,
so a shared raw axis would be misleading.

Usage:
    modal run modal_tep_paper_figures.py              # compute (or reuse) + plot
    modal run modal_tep_paper_figures.py --replot     # force reuse of cache
"""

import os

import modal

app = modal.App("tep-paper-figures")

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
OUT_DIR = "/data/all20/corrected/figures"
CACHE = "/data/all20/corrected/curves.npz"
FAULTS_ALL20 = list(range(1, 21))
PLOT_RUN = 4
MINUTES_PER_SAMPLE = 3

# Colour-blind-safe, print-safe pair
C_AEO, C_PCA = "#B2182B", "#2166AC"

VARIANTS = {
    "T2": ("AEO-T2", "PCA-T2 (Vpc<=95)"),
    "T2_SPE": ("AEO-T2+SPE", "PCA-T2+SPE"),
}
PANEL_TITLE = {
    "T2": r"Hotelling's $T^2$ statistic",
    "T2_SPE": r"Combined $T^2$ and SPE statistics",
}


# ----------------------------------------------------------------------
# Curve computation (cached)
# ----------------------------------------------------------------------
@app.function(image=image, volumes={"/data": volume}, memory=16384, timeout=5400)
def compute_curves():
    import sys
    sys.path.insert(0, "/root/code")
    import numpy as np
    import pandas as pd
    import tep_data as td
    import tep_models as tm
    import tep_protocol as tp

    volume.reload()
    comp = pd.read_csv(os.path.join(IN_DIR, "comparison.csv"))
    Xtr = td.get_run(0, tp.TRAIN_RUN, DATA_CSV)[:tp.WINDOW_SIZE]
    calib_X = np.vstack([td.get_run(0, r, DATA_CSV) for r in tp.CALIB_RUNS])

    cfg_cols = ["latent_dim", "hidden", "epochs", "q_shape",
                "loss_alpha", "beta", "gamma", "lr"]
    uniq = comp[comp["variant"].str.startswith("AEO")].drop_duplicates("trial")
    print(f"training {len(uniq)} unique AEO configs ...", flush=True)

    aeo, lim_aeo = {}, {}
    for _, r in uniq.iterrows():
        cfg = dict(latent_dim=int(r.latent_dim), hidden=int(r.hidden),
                   epochs=int(r.epochs), q_shape=str(r.q_shape),
                   loss_alpha=float(r.loss_alpha), beta=float(r.beta),
                   gamma=float(r.gamma), lr=float(r.lr))
        m = tm.AEOMonitor(model_seed=42, **cfg).fit(Xtr)
        aeo[int(r.trial)] = m
        t2c, spec = m.statistics(calib_X)
        lim_aeo[int(r.trial)] = (tp.empirical_control_limit(t2c, tp.TARGET_FAR),
                                 *tp.combined_control_limits(t2c, spec, tp.TARGET_FAR))
        print(f"  trial {int(r.trial)} done", flush=True)

    pca, lim_pca = {}, {}
    for v in sorted(comp[comp["variant"].str.startswith("PCA")].Vpc.unique()):
        m = tm.PCAMonitor(Vpc=int(v)).fit(Xtr)
        pca[int(v)] = m
        t2c, spec = m.statistics(calib_X)
        lim_pca[int(v)] = (tp.empirical_control_limit(t2c, tp.TARGET_FAR),
                           *tp.combined_control_limits(t2c, spec, tp.TARGET_FAR))

    out = {}
    for fault in FAULTS_ALL20:
        X = td.get_run(fault, PLOT_RUN, DATA_CSV)
        for kind, (va, vp) in VARIANTS.items():
            ra = comp[(comp.fault_number == fault) & (comp.variant == va)].iloc[0]
            m = aeo[int(ra.trial)]
            cl_t2, cl_t2c, cl_spec = lim_aeo[int(ra.trial)]
            t2, spe = m.statistics(X)
            out[f"{fault}_{kind}_AEO"] = (t2 / cl_t2 if kind == "T2"
                                          else np.maximum(t2 / cl_t2c, spe / cl_spec))

            rp = comp[(comp.fault_number == fault) & (comp.variant == vp)].iloc[0]
            m = pca[int(rp.Vpc)]
            cl_t2, cl_t2c, cl_spec = lim_pca[int(rp.Vpc)]
            t2, spe = m.statistics(X)
            out[f"{fault}_{kind}_PCA"] = (t2 / cl_t2 if kind == "T2"
                                          else np.maximum(t2 / cl_t2c, spe / cl_spec))
        print(f"  curves for IDV({fault})", flush=True)

    np.savez_compressed(CACHE, **out)
    volume.commit()
    print(f"[CACHED] {CACHE} ({len(out)} curves)")
    return len(out)


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
@app.function(image=image, volumes={"/data": volume}, memory=8192, timeout=3600)
def make_figures():
    import sys
    sys.path.insert(0, "/root/code")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import tep_protocol as tp

    volume.reload()
    comp = pd.read_csv(os.path.join(IN_DIR, "comparison.csv"))
    curves = np.load(CACHE)
    onset = tp.ONSET

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.linewidth": 0.9,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": False, "ytick.right": True,
    })

    def fmt_delay(d):
        if d is None or (isinstance(d, float) and np.isnan(d)):
            return "never detected"
        return f"{d:.1f} samples ({d * MINUTES_PER_SAMPLE:.0f} min)"

    def fmt_rate(rate, n):
        return f"detected in {int(round(rate * n))}/{int(n)} runs"

    def verdict(da, dp, ma, mp, det_a, det_p):
        """One-sentence statement of which method leads and by how much."""
        if det_a < 0.5 and det_p < 0.5:
            return ("Fault not reliably detected by either method "
                    "(caught in fewer than half the runs)")
        if det_a >= 0.5 and det_p < 0.5:
            return "Reliably detected by AEO only"
        if det_p >= 0.5 and det_a < 0.5:
            return "Reliably detected by PCA only"
        # both detect: lead on delay, then separation on MDR
        gap = dp - da
        if abs(gap) >= 0.5:
            lead, name = abs(gap), ("AEO" if gap > 0 else "PCA")
            s = (f"{name} detects {lead:.1f} samples "
                 f"({lead * MINUTES_PER_SAMPLE:.0f} min) earlier")
        else:
            s = "Detection times comparable"
        if abs(ma - mp) >= 0.005:
            better, lo, hi = (("AEO", ma, mp) if ma < mp else ("PCA", mp, ma))
            rel = (hi - lo) / hi * 100
            s += f"; {better} missed-detection rate lower by {rel:.0f}%"
        return s

    made = 0
    for fault in FAULTS_ALL20:
        fdir = os.path.join(OUT_DIR, f"Fault_{fault:02d}")
        os.makedirs(fdir, exist_ok=True)

        for kind, (va, vp) in VARIANTS.items():
            ya = curves[f"{fault}_{kind}_AEO"]
            yp = curves[f"{fault}_{kind}_PCA"]
            ra = comp[(comp.fault_number == fault) & (comp.variant == va)].iloc[0]
            rp = comp[(comp.fault_number == fault) & (comp.variant == vp)].iloc[0]

            # Fixed layout rather than tight_layout: the figure carries a title,
            # a subtitle, a secondary time axis and a performance block, and
            # automatic layout puts them on top of each other.
            fig = plt.figure(figsize=(7.6, 5.4), dpi=300)
            ax = fig.add_axes([0.115, 0.330, 0.845, 0.470])

            ax.axvspan(0, onset, color="0.92", zorder=0)
            for y, colour, label in ((ya, C_AEO, "AEO"), (yp, C_PCA, "PCA")):
                ax.semilogy(np.clip(y, 1e-3, None), color=colour, lw=1.0,
                            alpha=0.9, label=label, zorder=3)

            ax.axhline(1.0, color="black", ls="--", lw=1.1, zorder=4)
            ax.axvline(onset, color="black", ls="-", lw=1.1, alpha=0.75, zorder=4)

            # first sustained detection in this run, marked without text clutter
            for y, colour in ((ya, C_AEO), (yp, C_PCA)):
                r = tp.first_sustained_alarm(np.asarray(y)[onset:] > 1.0, k=tp.CONSEC_K)
                if r is not None:
                    ax.plot(onset + r, np.clip(y, 1e-3, None)[onset + r],
                            marker="v", color=colour, ms=7, mec="white",
                            mew=0.8, zorder=6)

            ax.set_xlim(0, len(ya))
            ax.set_xlabel(f"Sample index   (fault introduced at sample {onset})",
                          fontsize=11)
            ax.set_ylabel("Monitoring statistic / control limit", fontsize=11)
            ax.tick_params(labelsize=9.5)
            ax.grid(True, which="major", ls=":", lw=0.6, color="0.75", zorder=1)

            sec = ax.secondary_xaxis(
                "top", functions=(lambda s: s * MINUTES_PER_SAMPLE / 60.0,
                                  lambda h: h * 60.0 / MINUTES_PER_SAMPLE))
            sec.set_xlabel("Process time (h)", fontsize=10)
            sec.tick_params(labelsize=9)

            # Reference-line labels, both horizontal and boxed so they stay
            # legible wherever the curves happen to run.
            ax.text(0.995, 1.0, "Control limit", transform=ax.get_yaxis_transform(),
                    ha="right", va="bottom", fontsize=8.5, style="italic", zorder=7,
                    bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none",
                              alpha=0.85))
            ax.text(onset / 2.0, 0.035, "Normal operation", transform=
                    ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=8.5, style="italic", color="0.35", zorder=7,
                    bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none",
                              alpha=0.8))

            leg = ax.legend(fontsize=9.5, loc=_free_corner(ax, [ya, yp]),
                            frameon=True, framealpha=0.94, edgecolor="0.55")
            leg.get_frame().set_linewidth(0.8)

            # ---- titles above the axes (secondary time axis sits between) ----
            fig.text(0.5, 0.980, f"Fault Detection Performance for IDV({fault})",
                     ha="center", va="top", fontsize=13.5, fontweight="bold")
            # single subtitle line: a second one collides with the secondary
            # (process-time) axis label that sits just above the axes
            fig.text(0.5, 0.928, PANEL_TITLE[kind] +
                     " — AEO vs. PCA at a common 1% false-alarm rate",
                     ha="center", va="top", fontsize=10.5, color="0.25")

            # ---- performance block below the axes ----
            da, dp = ra.delay_samples, rp.delay_samples
            fig.text(0.5, 0.185, "Held-out performance (mean of 7 independent runs)",
                     ha="center", va="top", fontsize=9, fontweight="bold")
            rows = (f"AEO   delay {fmt_delay(da):>24s}   MDR {ra.mdr:.3f}   "
                    f"{fmt_rate(ra.detection_rate, ra.n_val_runs)}\n"
                    f"PCA   delay {fmt_delay(dp):>24s}   MDR {rp.mdr:.3f}   "
                    f"{fmt_rate(rp.detection_rate, rp.n_val_runs)}")
            fig.text(0.5, 0.152, rows, ha="center", va="top", fontsize=8.5,
                     family="monospace")
            fig.text(0.5, 0.055,
                     verdict(da, dp, ra.mdr, rp.mdr, ra.detection_rate, rp.detection_rate),
                     ha="center", va="top", fontsize=9, fontweight="bold",
                     color="0.15")
            suffix = "T2" if kind == "T2" else "T2_SPE"
            path = os.path.join(fdir, f"IDV{fault:02d}_{suffix}.png")
            fig.savefig(path, dpi=300, bbox_inches="tight")
            fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
            plt.close(fig)
            made += 1
            print(f"[SAVED] {path}", flush=True)

    volume.commit()
    return made


def _free_corner(ax, arrays):
    """
    Legend location keyword for the emptiest corner of the axes.

    The curves differ enormously in shape between faults (flat under the limit
    for IDV 3/9/15, saturated far above it for IDV 2/6), so a fixed legend
    position lands on the data for some faults. This counts how many plotted
    points fall in each corner and returns the quietest one.
    """
    import numpy as np
    y0, y1 = ax.get_ylim()
    ly0, ly1 = np.log10(y0), np.log10(y1)
    n = len(arrays[0])
    xf = np.arange(n) / max(n - 1, 1)

    candidates = {
        "upper right": ((0.55, 1.0), (0.58, 1.0)),
        "upper left": ((0.0, 0.45), (0.58, 1.0)),
        "lower right": ((0.55, 1.0), (0.0, 0.42)),
    }   # "lower left" is reserved for the "Normal operation" band label
    best, best_count = "upper right", None
    for loc, ((xlo, xhi), (ylo, yhi)) in candidates.items():
        count = 0
        for arr in arrays:
            a = np.clip(np.asarray(arr), 10 ** ly0, 10 ** ly1)
            yfrac = (np.log10(a) - ly0) / max(ly1 - ly0, 1e-12)
            count += int(((xf >= xlo) & (xf <= xhi) &
                          (yfrac >= ylo) & (yfrac <= yhi)).sum())
        if best_count is None or count < best_count:
            best, best_count = loc, count
    return best


@app.local_entrypoint()
def main(replot: bool = False):
    if not replot:
        print("Step 1/2: computing curves (cached for future restyles) ...")
        compute_curves.remote()
    else:
        print("Step 1/2: skipped -- reusing cached curves.")
    print("Step 2/2: rendering figures ...")
    n = make_figures.remote()
    print(f"done: {n} figures (PNG + PDF) in {OUT_DIR}")
    print("Pull with: modal volume get tep-data all20/corrected/figures <local_dir>")
