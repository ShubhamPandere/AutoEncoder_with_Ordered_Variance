# -*- coding: utf-8 -*-
"""
Regression guard: does the DAA result survive the corrected protocol?
=====================================================================

The published DAA result (Fault_detection_AEO_arXiv, Table 1) is AEO detecting
at t=502 vs PCA t=505 on Scenario 1, and t=501 vs t=505 on Scenario 2. That was
obtained with nominal-alpha control limits and a single-sample detection rule --
both of which the audit shows can manufacture a lead.

So before trusting the corrected TEP numbers, check the corrected protocol
against the case the paper already published. If AEO's DAA lead disappears once
both methods sit at the same measured false-alarm rate, the original DAA claim
was an artifact too and the paper needs revising. If it survives, the corrected
protocol is consistent with the published work and only the TEP conclusions
change.

DAA data layout (from `p_aeo (1).py`): the Excel file is variables x samples, so
it is transposed; the first 500 samples are fault-free and the disturbance is
introduced at sample 500 (Scenario 1 = impulse 500-560, Scenario 2 = step from
500 onward).

Calibration caveat: DAA provides a single run per scenario, so the only
fault-free data not used for training is samples 300-499 (200 samples). A 1%
quantile off 200 points is the 2nd-largest value -- far too noisy -- so this
guard uses a 5% target FAR. That is fine for a relative comparison: both methods
get the identical treatment, which is the entire point.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tep_models as tm
import tep_protocol as tp

DAA_DIR = r"D:/Shubham/IIT Bombay/Submitted"
SCENARIOS = {
    1: dict(path=os.path.join(DAA_DIR, "scenario1_filtered.xlsx"), onset=500),
    2: dict(path=os.path.join(DAA_DIR, "scenario2.xlsx"), onset=500),
}

TRAIN_END = 300        # samples 0-299 fit the model (Nw = 300, as in the paper)
CALIB_START = 300      # samples 300-499 are fault-free and unseen by training
TARGET_FAR = 0.05
CONSEC_K = 3


def load_daa(path):
    df = pd.read_excel(path)
    return df.to_numpy(dtype=float).T   # -> (samples, variables)


def run_scenario(scenario, aeo_kwargs=None, pca_vpc=90):
    spec = SCENARIOS[scenario]
    data = load_daa(spec["path"])
    onset = spec["onset"]
    Xtr = data[:TRAIN_END]
    Xcal = data[CALIB_START:onset]

    aeo_kwargs = dict(latent_dim=8, hidden=16, loss_alpha=3.2, beta=0.5, gamma=0.1,
                      epochs=2000, lr=2e-3, q_shape="linear") | (aeo_kwargs or {})

    rows = []
    monitors = {
        "AEO": tm.AEOMonitor(model_seed=42, **aeo_kwargs).fit(Xtr),
        "PCA": tm.PCAMonitor(Vpc=pca_vpc).fit(Xtr),
    }

    for name, mon in monitors.items():
        t2_cal, spe_cal = mon.statistics(Xcal)
        t2_all, spe_all = mon.statistics(data)

        for index, cal, full in (("T2", t2_cal, t2_all), ("SPE", spe_cal, spe_all)):
            Cl = tp.empirical_control_limit(cal, TARGET_FAR)
            far = float(np.mean(cal > Cl))
            rel = tp.first_sustained_alarm(full[onset:] > Cl, k=CONSEC_K)
            rows.append({
                "scenario": scenario, "method": name, "index": index,
                "measured_far": far, "Cl": Cl,
                "first_detection": onset + rel if rel is not None else None,
                "delay_samples": rel,
            })

        cl_t2, cl_spe = tp.combined_control_limits(t2_cal, spe_cal, TARGET_FAR)
        comb_cal = np.maximum(t2_cal / cl_t2, spe_cal / cl_spe)
        comb_all = np.maximum(t2_all / cl_t2, spe_all / cl_spe)
        rel = tp.first_sustained_alarm(comb_all[onset:] > 1.0, k=CONSEC_K)
        rows.append({
            "scenario": scenario, "method": name, "index": "T2+SPE",
            "measured_far": float(np.mean(comb_cal > 1.0)), "Cl": 1.0,
            "first_detection": onset + rel if rel is not None else None,
            "delay_samples": rel,
        })

    return pd.DataFrame(rows)


def main():
    out = []
    for s in SCENARIOS:
        print(f"=== DAA Scenario {s} ===", flush=True)
        df = run_scenario(s)
        print(df.to_string(index=False), flush=True)
        out.append(df)
        print(flush=True)

    res = pd.concat(out, ignore_index=True)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results", "tables", "corrected", "daa_regression_guard.csv")
    res.to_csv(out_path, index=False)

    print("=== AEO lead over PCA (positive = AEO earlier), at matched measured FAR ===")
    for s in SCENARIOS:
        for index in ("T2", "SPE", "T2+SPE"):
            sub = res[(res.scenario == s) & (res["index"] == index)].set_index("method")
            if {"AEO", "PCA"} <= set(sub.index):
                a, p = sub.loc["AEO", "delay_samples"], sub.loc["PCA", "delay_samples"]
                if a is not None and p is not None:
                    print(f"  Scenario {s} [{index:6s}]  AEO delay={a:>4}  "
                          f"PCA delay={p:>4}  lead={p - a:+d}")
    print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    main()
