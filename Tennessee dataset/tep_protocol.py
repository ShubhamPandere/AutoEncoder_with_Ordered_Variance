# -*- coding: utf-8 -*-
"""
Corrected evaluation protocol for the TEP / AEO fault-detection study.
======================================================================

This module replaces the parts of `tep_evaluation.py` + `tep_tuning.py` that the
audit (results/DISCREPANCY_REPORT.md) found to be the cause of the two results
that contradicted the DAA paper:

  * PCA appearing to detect faults *earlier* than AEO -> an artifact of
    comparing detection delay at wildly different false-alarm rates (A1), which
    in turn came from ranking configs by F1 on a 160-normal / 800-faulty split
    where an "always alarm" detector already scores F1 = 0.909 (A2).
  * AEO having lower FAR but higher MDR than PCA -> the F-distribution control
    limit is miscalibrated in *both* directions, so the nominal significance
    level `alpha` does not mean the same thing for the two methods (A5).

The three fixes here:

  1. `empirical_control_limit` -- set the limit from the (1 - target_far)
     quantile of the statistic measured on held-out FAULT-FREE data, so every
     method is compared at the *same measured* false-alarm rate instead of the
     same nominal alpha.
  2. `evaluate_detection_v2` -- FAR measured on dedicated fault-free runs over
     all 960 samples (not the 160 shared pre-onset samples, which are
     bit-identical across every fault -- audit A7c); detection delay requires
     `k` consecutive violations so a lone false alarm cannot score delay 0.
  3. `score_metrics_v2` -- rank configs by detection delay at fixed FAR, with
     MDR as the tie-breaker, instead of by F1.

Data splits (no leakage; see SPLITS below):
    fault-free run  1       -> AEO/PCA training window
    fault-free runs 2-4     -> control-limit calibration
    fault-free runs 5-10    -> held-out FAR measurement
    faulty runs     1-3     -> hyperparameter tuning
    faulty runs     4-10    -> held-out validation
"""

import numpy as np

# ----------------------------------------------------------------------
# Split definition -- single source of truth, imported by every other module
# ----------------------------------------------------------------------
TRAIN_RUN = 1                     # fault-free run used to fit the model
CALIB_RUNS = (2, 3, 4)            # fault-free runs used to set control limits
FAR_RUNS = (5, 6, 7, 8, 9, 10)    # fault-free runs used to MEASURE far
TUNE_RUNS = (1, 2, 3)             # faulty runs used for hyperparameter search
VAL_RUNS = (4, 5, 6, 7, 8, 9, 10)  # faulty runs used for reported results

WINDOW_SIZE = 480                 # training samples (audit A7k: 200 too few for 52 vars)
ONSET = 160                       # Rieth et al. testing split: fault starts at sample 160
RUN_LENGTH = 960
TARGET_FAR = 0.01                 # every method is calibrated to this measured FAR
CONSEC_K = 3                      # consecutive violations required to declare detection
MINUTES_PER_SAMPLE = 3

SPLITS = {
    "train_run": TRAIN_RUN, "calib_runs": CALIB_RUNS, "far_runs": FAR_RUNS,
    "tune_runs": TUNE_RUNS, "val_runs": VAL_RUNS,
    "window_size": WINDOW_SIZE, "onset": ONSET, "target_far": TARGET_FAR,
    "consec_k": CONSEC_K,
}


# ----------------------------------------------------------------------
# 1. Empirical control limits
# ----------------------------------------------------------------------
def empirical_control_limit(stat_calib, target_far=TARGET_FAR):
    """
    Control limit = (1 - target_far) quantile of the statistic on held-out
    fault-free data.

    This is the single change that makes FAR comparable across methods. The
    F-based limit in `p_aeo (1).py` assumes i.i.d. multivariate-normal scores;
    TEP data is strongly autocorrelated and the AEO latent covariance is the
    *in-sample* covariance of a network trained for thousands of epochs on the
    very same window, so the nominal alpha is off by up to 7x in practice.
    """
    stat_calib = np.asarray(stat_calib, dtype=float)
    if stat_calib.size == 0:
        raise ValueError("empirical_control_limit: empty calibration statistic")
    return float(np.quantile(stat_calib, 1.0 - target_far))


def combined_control_limits(t2_calib, spe_calib, target_far=TARGET_FAR,
                            tol=1e-4, max_iter=60):
    """
    Limits for a combined "alarm if T2 > Cl_T2 OR SPE > Cl_SPE" index.

    Splitting the budget naively (target_far/2 each) mis-shoots, because the two
    statistics are correlated. Instead bisect on a shared per-statistic level
    `a` so that the *joint* exceedance rate on the calibration data equals
    `target_far`. The joint rate is monotone in `a`, so bisection is exact.
    """
    t2_calib = np.asarray(t2_calib, dtype=float)
    spe_calib = np.asarray(spe_calib, dtype=float)

    def joint_rate(a):
        cl_t2 = np.quantile(t2_calib, 1.0 - a)
        cl_spe = np.quantile(spe_calib, 1.0 - a)
        return float(np.mean((t2_calib > cl_t2) | (spe_calib > cl_spe))), cl_t2, cl_spe

    lo, hi = target_far / 2.0, target_far   # joint rate is increasing in a
    mid = hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        rate, _, _ = joint_rate(mid)
        if abs(rate - target_far) < tol:
            break
        if rate > target_far:
            hi = mid
        else:
            lo = mid
    _, cl_t2, cl_spe = joint_rate(mid)
    return float(cl_t2), float(cl_spe)


# ----------------------------------------------------------------------
# 2. Metrics
# ----------------------------------------------------------------------
def first_sustained_alarm(alarms, k=CONSEC_K):
    """
    Index of the first position where `k` consecutive alarms begin, or None.

    The old `evaluate_detection` took the first single post-onset violation,
    which hands delay ~0 to any detector already alarming continuously before
    onset. Requiring k in a row is the standard TEP convention and is what makes
    the delay comparison meaningful.
    """
    alarms = np.asarray(alarms, dtype=bool)
    if alarms.size < k:
        return None
    if k == 1:
        hits = np.flatnonzero(alarms)
        return int(hits[0]) if hits.size else None
    run = alarms.copy()
    for shift in range(1, k):
        run[:-shift] &= alarms[shift:]
    run[-(k - 1):] = False
    hits = np.flatnonzero(run)
    return int(hits[0]) if hits.size else None


def evaluate_detection_v2(stat_faulty_runs, stat_far_runs, Cl,
                          onset=ONSET, k=CONSEC_K):
    """
    Metrics for one (method, fault) pair.

    Parameters
    ----------
    stat_faulty_runs : dict {run_id: 1-D array of length RUN_LENGTH}
        Detection statistic on each held-out FAULTY validation run.
    stat_far_runs : dict {run_id: 1-D array}
        Detection statistic on each held-out FAULT-FREE run. FAR is measured
        here, over all samples -- not on the 160 pre-onset samples of the faulty
        run, which are bit-identical across every fault number and so gave 20
        copies of a single measurement (audit A7c).
    Cl : float
        Control limit (from `empirical_control_limit`).
    """
    far_flags = np.concatenate([np.asarray(s) > Cl for s in stat_far_runs.values()])
    far = float(np.mean(far_flags))

    mdrs, delays, detected_flags = [], [], []
    for s in stat_faulty_runs.values():
        s = np.asarray(s, dtype=float)
        post = s[onset:] > Cl
        mdrs.append(float(np.mean(~post)))
        rel = first_sustained_alarm(post, k=k)
        detected_flags.append(rel is not None)
        delays.append(rel)

    mdr = float(np.mean(mdrs))
    detection_rate = float(np.mean(detected_flags))
    observed = [d for d in delays if d is not None]
    delay_mean = float(np.mean(observed)) if observed else None
    delay_std = float(np.std(observed, ddof=1)) if len(observed) > 1 else 0.0

    return {
        "far": far,
        "mdr": mdr,
        "tpr": 1.0 - mdr,
        "balanced_accuracy": 0.5 * ((1.0 - mdr) + (1.0 - far)),
        "detection_rate": detection_rate,   # fraction of runs ever detected
        "delay_samples": delay_mean,
        "delay_std": delay_std,
        "delay_minutes": delay_mean * MINUTES_PER_SAMPLE if delay_mean is not None else None,
        "n_val_runs": len(stat_faulty_runs),
        "Cl": float(Cl),
    }


# ----------------------------------------------------------------------
# 3. Tuning score
# ----------------------------------------------------------------------
NO_DETECTION_DELAY = RUN_LENGTH - ONSET   # 800: worst possible delay


def score_metrics_v2(metrics):
    """
    Higher is better. Rank primarily by detection delay at the (already fixed)
    measured FAR, with MDR as tie-breaker.

    Replaces `score_metrics`'s F1, which on this 160-normal / 800-faulty split
    gives an "always alarm" detector 0.909 out of a 0.995 maximum -- a 0.086
    discriminative range that pushed the tuner straight at the trigger-happy
    corner of the grid (audit A2). Because the control limit is now pinned to a
    fixed measured FAR, delay can no longer be bought with false alarms.

    Both terms are normalised to roughly [0, 1] so neither dominates; for the
    genuinely undetectable faults (IDV 3/9/15) every config saturates the delay
    term and MDR decides, which is the sensible fallback.
    """
    delay = metrics.get("delay_samples")
    if delay is None:
        delay = NO_DETECTION_DELAY
    mdr = metrics.get("mdr", 1.0)
    if mdr is None or np.isnan(mdr):
        mdr = 1.0
    return -(float(delay) / NO_DETECTION_DELAY + float(mdr))


def legacy_f1(metrics):
    """F1 kept only so corrected tables stay comparable with the old ones."""
    tpr = metrics.get("tpr")
    far = metrics.get("far")
    if tpr is None or far is None:
        return np.nan
    tp, fp = 800.0 * tpr, 160.0 * far   # old 800-faulty / 160-normal framing
    if tp + fp == 0:
        return np.nan
    prec = tp / (tp + fp)
    return float(2 * prec * tpr / (prec + tpr)) if (prec + tpr) > 0 else np.nan
