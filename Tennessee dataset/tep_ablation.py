# -*- coding: utf-8 -*-
"""
AEO vs. plain autoencoder ablation.
===================================

Tests what the variance-regularization term actually buys, by stripping it out
and keeping everything else identical.

Loss-term naming, because the two papers disagree (and the code follows the
first one):

    AEO paper, Eq. (19):  J = a*||X-Xhat||^2 + b*||Q^0.5[Y-Ybar]||^2 + g*||A||^2
                              a -> J1 recon      b -> J2 variance      g -> J3 weights
    Fault-detection paper, Eq. (3):
                          J =   ||X-Xhat||^2 + alpha*||Q^0.5[Y-Ybar]||^2 + beta*||theta||^2

`p_aeo (1).py` and `tep_models.py` use the AEO paper's convention:
`loss_alpha` = a (reconstruction), `beta` = b (variance), `gamma` = g (weights).
So "drop the regularizers" means **beta = 0, gamma = 0**, NOT loss_alpha = 0
(which would delete the reconstruction term and give a degenerate model).

Arms
----
    AEO          full loss; Algorithm 1 ordering
    AE           reconstruction only (beta = gamma = 0), dimensions left as-is
    AE-sorted    reconstruction only, latent dims sorted by variance post-hoc
    AE+wd        reconstruction + weight decay (beta = 0, gamma tuned), sorted

`AE-sorted` is the arm that matters. Anyone can sort a plain AE's latent
dimensions by variance after training, so comparing AEO only against unsorted
dimensions would be a strawman. AEO's real claim is that ordering is obtained
*during* training and that it concentrates variance -- if AEO beats AE-sorted,
that claim is supported; if not, the honest finding is that post-hoc sorting is
sufficient, which a reviewer would otherwise raise.

Why sweep p
-----------
At p = latent_dim, T^2 is a Mahalanobis distance over the whole latent space and
is therefore **invariant to any permutation of the latent dimensions**. Ordering
cannot matter there. It can only matter when the latent space is truncated, so
the ablation is run at p in {2, 4, 8, latent_dim}: AEO should degrade gracefully
as p shrinks, a plain AE should not, because its leading dimensions are
arbitrary.
"""

import numpy as np
import pandas as pd

import tep_data as td
import tep_models as tm
import tep_protocol as tp

ARMS = ("AEO", "AE", "AE-sorted", "AE+wd")
P_VALUES = (2, 4, 8, None)          # None = full latent_dim
INDICES = ("T2", "T2+SPE")

# Shared architecture/optimiser space -- identical across arms so the only
# difference between them is the loss.
ARCH_SPACE = {
    "latent_dim": [4, 8, 16],
    "hidden": [16, 32, 64],
    "epochs": [1000, 2000, 4000],
    "lr": (5e-4, 5e-3),             # log-uniform
}
AEO_ONLY = {
    "beta": (0.05, 1.0),
    "gamma": (1e-4, 0.3),           # log-uniform
    "q_shape": ["linear", "quadratic", "geometric"],
}


def sample_config(arm, rng):
    """Configs for a given arm. Architecture draws are shared across arms."""
    cfg = {k: (int(rng.choice(v)) if isinstance(v[0], (int, np.integer)) else str(rng.choice(v)))
           for k, v in ARCH_SPACE.items() if isinstance(v, list)}
    lo, hi = ARCH_SPACE["lr"]
    cfg["lr"] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))

    # Reconstruction weight is fixed at 1 for every arm: it is very nearly a
    # pure loss rescaling (measured ~5% effect on latent variances via Adam's
    # epsilon), and Eq. (3) of the fault-detection paper normalises it away.
    cfg["loss_alpha"] = 1.0

    if arm == "AEO":
        for k in ("beta", "gamma"):
            lo, hi = AEO_ONLY[k]
            cfg[k] = float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) if k == "gamma" \
                else float(rng.uniform(lo, hi))
        cfg["q_shape"] = str(rng.choice(AEO_ONLY["q_shape"]))
        cfg.update(enforce_ordering=True, sort_latent=False)
    elif arm == "AE+wd":
        lo, hi = AEO_ONLY["gamma"]
        cfg.update(beta=0.0, gamma=float(np.exp(rng.uniform(np.log(lo), np.log(hi)))),
                   q_shape="linear", enforce_ordering=False, sort_latent=True)
    else:                                   # AE, AE-sorted
        cfg.update(beta=0.0, gamma=0.0, q_shape="linear",
                   enforce_ordering=False, sort_latent=(arm == "AE-sorted"))
    return cfg


def build_monitor(cfg, model_seed=42):
    return tm.AEOMonitor(model_seed=model_seed, **cfg)


def evaluate_arm(monitor, faults, p_values=P_VALUES, target_far=tp.TARGET_FAR,
                 csv=td.DEFAULT_CSV):
    """
    Metrics for one fitted model across every (fault, p, index, split).

    One forward pass per run is reused for all p values -- T^2 for a given p is
    just a Mahalanobis distance on the leading p latent columns.
    """
    calib_X = np.vstack([td.get_run(0, r, csv) for r in tp.CALIB_RUNS])
    Yc, spe_c = monitor.latent_and_spe(calib_X)
    far_raw = {r: monitor.latent_and_spe(td.get_run(0, r, csv)) for r in tp.FAR_RUNS}

    fault_raw = {}
    for f in faults:
        for split, runs in (("tune", tp.TUNE_RUNS), ("val", tp.VAL_RUNS)):
            for r in runs:
                fault_raw[(f, split, r)] = monitor.latent_and_spe(td.get_run(f, r, csv))

    rows = []
    for p in p_values:
        monitor.set_p(p or monitor.cfg["latent_dim"])
        p_eff = monitor.p_dims

        t2_c = monitor.t2_from_latent(Yc)
        cl_t2 = tp.empirical_control_limit(t2_c, target_far)
        cl_t2c, cl_spec = tp.combined_control_limits(t2_c, spe_c, target_far)

        far_t2 = {r: monitor.t2_from_latent(Y) for r, (Y, _) in far_raw.items()}
        far_cmb = {r: np.maximum(monitor.t2_from_latent(Y) / cl_t2c, s / cl_spec)
                   for r, (Y, s) in far_raw.items()}
        cl_cmb_match = tp.empirical_control_limit(
            np.concatenate(list(far_cmb.values())), target_far)

        for f in faults:
            for split, runs in (("tune", tp.TUNE_RUNS), ("val", tp.VAL_RUNS)):
                fa_t2 = {r: monitor.t2_from_latent(fault_raw[(f, split, r)][0]) for r in runs}
                fa_cmb = {r: np.maximum(
                    monitor.t2_from_latent(fault_raw[(f, split, r)][0]) / cl_t2c,
                    fault_raw[(f, split, r)][1] / cl_spec) for r in runs}

                base = {"fault_number": f, "split": split, "p": p_eff,
                        "p_request": "full" if p is None else p}

                # T2 only, matched measured FAR
                cl_m = tp.empirical_control_limit(
                    np.concatenate(list(far_t2.values())), target_far)
                for lim_name, cl in (("calibrated", cl_t2), ("matched", cl_m)):
                    m = tp.evaluate_detection_v2(fa_t2, far_t2, cl)
                    rows.append({**base, "index": "T2", "limit": lim_name, **m})

                # combined T2 + SPE
                for lim_name, cl in (("calibrated", 1.0), ("matched", cl_cmb_match)):
                    m = tp.evaluate_detection_v2(fa_cmb, far_cmb, cl)
                    rows.append({**base, "index": "T2+SPE", "limit": lim_name, **m})

    for r in rows:
        r["score"] = tp.score_metrics_v2(r)
    return rows


def select_best(df, arm=None):
    """
    Best config per (arm, fault, index, p), chosen on the tuning runs under the
    no-peeking calibrated limit, reported on the validation runs at matched FAR.
    """
    sub = df if arm is None else df[df.arm == arm]
    keys = ["arm", "fault_number", "index", "p_request"]
    tune = sub[(sub.split == "tune") & (sub.limit == "calibrated")]
    val = sub[(sub.split == "val") & (sub.limit == "matched")]
    if tune.empty or val.empty:
        return val.iloc[0:0]
    win = tune.loc[tune.groupby(keys)["score"].idxmax(), keys + ["trial"]]
    return val.merge(win, on=keys + ["trial"], how="inner").reset_index(drop=True)
