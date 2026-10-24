# -*- coding: utf-8 -*-
"""
AEO vs. plain-autoencoder ablation over all 20 TEP faults, on Modal.
====================================================================

Four arms (see tep_ablation.py for the loss-term naming and the reasoning):

    AEO          full loss, Algorithm 1 ordering
    AE           reconstruction only (beta = gamma = 0)
    AE-sorted    reconstruction only, latent dims sorted by variance post-hoc
    AE+wd        reconstruction + weight decay, sorted

each swept over p in {2, 4, 8, latent_dim}, both monitoring indices, all 20
faults, every method calibrated to a measured 1% false-alarm rate.

Sharded by trial index (the models are fault-independent), 8 containers max.

Usage:
    modal run modal_tep_ae_ablation.py --n-trials 40
"""

import os

import modal

app = modal.App("tep-ae-ablation")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

CODE_FILES = [
    "tep_protocol.py",
    "tep_models.py",
    "tep_data.py",
    "tep_corrected_pipeline.py",
    "tep_ablation.py",
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("pandas", "numpy", "scipy", "torch", "matplotlib")
    .add_local_dir(_THIS_DIR, remote_path="/root/code", copy=True,
                   ignore=lambda p: p.name not in CODE_FILES)
)

volume = modal.Volume.from_name("tep-data")

DATA_CSV = "/data/all20/tep_test_subset.csv"
OUT_DIR = "/data/all20/ablation"
FAULTS_ALL20 = list(range(1, 21))
N_SHARDS = 8
MAX_CONTAINERS = 8


def _setup():
    import sys
    sys.path.insert(0, "/root/code")
    import matplotlib
    matplotlib.use("Agg")
    import tep_ablation as ab
    import tep_data as td
    import tep_protocol as tp
    return ab, td, tp


@app.function(image=image, volumes={"/data": volume}, memory=8192,
              timeout=7200, max_containers=MAX_CONTAINERS)
def run_shard(args):
    shard_idx, n_shards, n_trials, search_seed, model_seed = args
    ab, td, tp = _setup()
    import time
    import warnings
    import numpy as np
    import pandas as pd

    Xtr = td.get_run(0, tp.TRAIN_RUN, DATA_CSV)[:tp.WINDOW_SIZE]

    # Regenerate the identical config list in every container and take a
    # strided slice, so trial indices mean the same thing across shards.
    rngs = {arm: np.random.default_rng(search_seed + 1000 * i)
            for i, arm in enumerate(ab.ARMS)}
    all_cfgs = {arm: [ab.sample_config(arm, rngs[arm]) for _ in range(n_trials)]
                for arm in ab.ARMS}

    idx = list(range(shard_idx, n_trials, n_shards))
    frames = []
    for arm in ab.ARMS:
        for t in idx:
            cfg = all_cfgs[arm][t]
            t0 = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mon = ab.build_monitor(cfg, model_seed=model_seed).fit(Xtr)
            rows = ab.evaluate_arm(mon, FAULTS_ALL20, csv=DATA_CSV)
            v = mon.latent_variances
            top1 = float(v[0] / (v.sum() + 1e-12) * 100)
            for r in rows:
                r.update({"arm": arm, "trial": t, "elapsed_s": time.time() - t0,
                          "converged": mon.converged, "top1_var_pct": top1, **cfg})
            frames.append(pd.DataFrame(rows))
            print(f"[shard {shard_idx}] {arm:10s} trial {t:3d} "
                  f"{time.time()-t0:5.1f}s latent={cfg['latent_dim']:2d} "
                  f"top1={top1:5.1f}%", flush=True)

    out = pd.concat(frames, ignore_index=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"ablation_shard{shard_idx}.csv")
    out.to_csv(path, index=False)
    volume.commit()
    print(f"[shard {shard_idx}] {len(out)} rows -> {path}")
    return {"shard": shard_idx, "rows": len(out)}


@app.function(image=image, volumes={"/data": volume}, memory=16384, timeout=1800)
def aggregate(n_shards=N_SHARDS):
    ab, _, _ = _setup()
    import pandas as pd

    volume.reload()
    df = pd.concat(
        [pd.read_csv(os.path.join(OUT_DIR, f"ablation_shard{i}.csv"))
         for i in range(n_shards)
         if os.path.exists(os.path.join(OUT_DIR, f"ablation_shard{i}.csv"))],
        ignore_index=True)
    df.to_csv(os.path.join(OUT_DIR, "ablation_trials.csv"), index=False)

    best = ab.select_best(df)
    best.to_csv(os.path.join(OUT_DIR, "ablation_best.csv"), index=False)

    und = {3, 9, 15}
    det = best[~best.fault_number.isin(und)]
    summary = det.groupby(["arm", "index", "p_request"]).agg(
        mdr_mean=("mdr", "mean"), far_mean=("far", "mean"),
        delay_mean=("delay_samples", "mean"),
        detection_rate=("detection_rate", "mean"),
    ).reset_index()
    summary.to_csv(os.path.join(OUT_DIR, "ablation_summary.csv"), index=False)

    pd.set_option("display.width", 200)
    print("\n=== MEAN MDR over the 17 detectable faults ===")
    for ix in ("T2", "T2+SPE"):
        piv = summary[summary["index"] == ix].pivot(
            index="arm", columns="p_request", values="mdr_mean")
        cols = [c for c in ["2", "4", "8", "full", 2, 4, 8] if c in piv.columns]
        print(f"\n-- {ix} --")
        print(piv[cols + [c for c in piv.columns if c not in cols]].round(3).to_string())

    volume.commit()
    return summary.shape


@app.local_entrypoint()
def main(n_trials: int = 40, search_seed: int = 0, model_seed: int = 42,
         n_shards: int = N_SHARDS):
    print(f"AE ablation | trials={n_trials}/arm | arms=4 | shards={n_shards}")
    jobs = [(i, n_shards, n_trials, search_seed, model_seed) for i in range(n_shards)]
    for r in run_shard.map(jobs):
        print("  shard done:", r)
    print("\nAggregating ...")
    aggregate.remote(n_shards)
    print("\nPull with: modal volume get tep-data all20/ablation <local_dir>")
