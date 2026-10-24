# -*- coding: utf-8 -*-
"""
TEP data access for the corrected pipeline.

Thin wrapper over the existing loader in `tep_evaluation.py`, with the whole CSV
read and indexed once instead of re-scanning it for every (fault, run) pair --
the corrected pipeline touches ~170 trajectories per fault, so the original
read_csv-per-call pattern dominated runtime.

Crucially, this module does NOT stack a fault-free window in front of the faulty
run the way `build_monitoring_sequence` did. Each run is returned whole, and the
protocol module measures FAR on dedicated fault-free runs instead (audit A7c:
the 160 pre-onset samples of a faulty run are bit-identical to the fault-free
run of the same simulationRun, so they were never an independent measurement).
"""

import os
import functools

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_THIS_DIR, "data")
DEFAULT_CSV = os.path.join(DATA_DIR, "tep_test_subset.csv")


def _var_columns(df):
    meas = sorted([c for c in df.columns if str(c).lower().startswith("xmeas")],
                  key=lambda c: int(str(c).split("_")[-1]))
    mv = sorted([c for c in df.columns if str(c).lower().startswith("xmv")],
                key=lambda c: int(str(c).split("_")[-1]))
    cols = meas + mv
    if not cols:
        raise ValueError("no xmeas_*/xmv_* columns found in TEP CSV")
    return cols


@functools.lru_cache(maxsize=4)
def _load_table(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}
    fault_c, run_c, samp_c = lower["faultnumber"], lower["simulationrun"], lower["sample"]
    var_cols = _var_columns(df)
    df = df.sort_values([fault_c, run_c, samp_c])
    runs = {}
    for (f, r), g in df.groupby([fault_c, run_c]):
        runs[(int(f), int(r))] = np.ascontiguousarray(g[var_cols].to_numpy(dtype=float))
    return runs, var_cols


def get_run(fault_number, simulation_run, csv_path=DEFAULT_CSV):
    """Return one (fault, run) trajectory, shape (n_samples, n_features)."""
    runs, _ = _load_table(csv_path)
    key = (int(fault_number), int(simulation_run))
    if key not in runs:
        raise KeyError(f"fault={fault_number} run={simulation_run} not in {csv_path}")
    return runs[key]


def available_faults(csv_path=DEFAULT_CSV):
    runs, _ = _load_table(csv_path)
    return sorted({f for f, _ in runs})


def n_features(csv_path=DEFAULT_CSV):
    _, cols = _load_table(csv_path)
    return len(cols)
