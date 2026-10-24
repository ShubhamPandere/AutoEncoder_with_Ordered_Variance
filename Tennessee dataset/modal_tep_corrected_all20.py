# -*- coding: utf-8 -*-
"""
Corrected AEO-vs-PCA pipeline over ALL 20 TEP faults, on Modal.
===============================================================

Re-runs the protocol from `tep_corrected_pipeline.py` (empirical control limits,
matched measured FAR, leak-free splits, sustained-alarm delay, T2 + SPE for both
methods) across all 20 faults, to check whether the 7-fault conclusions hold.

Parallelisation differs from `modal_tep_all20_pipeline.py` on purpose. That
script gave each FAULT its own container, because the old code re-trained the
AEO per fault. It never needed to: both models are fitted on the fault-free
training window only, so they do not depend on the fault number at all. The
corrected pipeline trains each config ONCE and scores it against every fault, so
the thing worth sharding is the TRIAL LIST, not the faults.

Every container regenerates the identical config list from one seed and takes a
slice of it, so trial indices mean the same thing everywhere and the shards
re-aggregate cleanly.

Concurrency is capped at 8 containers.

Data: reads /data/all20/tep_test_subset.csv (already extracted on the
"tep-data" volume by the earlier run). Writes to /data/all20/corrected/ so
nothing from the previous all20 run is touched.

Usage:
    modal run modal_tep_corrected_all20.py                    # full run
    modal run modal_tep_corrected_all20.py --n-trials 16      # quick check
"""

import os

import modal

app = modal.App("tep-corrected-all20")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Only the four corrected modules are needed -- none of them import
# `p_aeo (1).py` or the old tep_evaluation/tep_tuning stack.
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

FAULTS_ALL20 = list(range(1, 21))
DATA_CSV = "/data/all20/tep_test_subset.csv"
OUT_DIR = "/data/all20/corrected"
N_SHARDS = 8
MAX_CONTAINERS = 8


def _setup():
    """Import the corrected modules inside a container."""
    import sys
    sys.path.insert(0, "/root/code")
    import matplotlib
    matplotlib.use("Agg")
    import tep_corrected_pipeline as tcp
    import tep_protocol as tp
    return tcp, tp


# ----------------------------------------------------------------------
# PCA grid (cheap -- one container)
# ----------------------------------------------------------------------
@app.function(image=image, volumes={"/data": volume}, memory=8192, timeout=1800)
def run_pca_all20():
    tcp, _ = _setup()
    import pandas as pd

    df = tcp.search_pca(faults=FAULTS_ALL20, csv=DATA_CSV)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "pca_trials.csv")
    df.to_csv(path, index=False)
    volume.commit()
    print(f"[PCA] {len(df)} rows -> {path}")
    return len(df)


# ----------------------------------------------------------------------
# AEO trial shards
# ----------------------------------------------------------------------
@app.function(image=image, volumes={"/data": volume}, memory=8192,
              timeout=7200, max_containers=MAX_CONTAINERS)
def run_aeo_shard(args):
    shard_idx, n_shards, n_trials, search_seed, model_seed = args
    tcp, _ = _setup()
    import pandas as pd

    # Identical list in every container; each takes a strided slice so the
    # shards get a comparable mix of cheap and expensive configs.
    all_cfgs = tcp.sample_config_list(n_trials, seed=search_seed)
    idx = list(range(shard_idx, n_trials, n_shards))
    cfgs = [all_cfgs[i] for i in idx]
    print(f"[shard {shard_idx}/{n_shards}] {len(cfgs)} configs: trials {idx}", flush=True)

    frames = []
    for local_i, (trial_i, cfg) in enumerate(zip(idx, cfgs)):
        df = tcp.search_aeo(faults=FAULTS_ALL20, model_seed=model_seed,
                            csv=DATA_CSV, configs=[cfg], trial_offset=trial_i,
                            log_every=1)
        frames.append(df)
        print(f"[shard {shard_idx}] done {local_i+1}/{len(cfgs)} (trial {trial_i})", flush=True)

    out = pd.concat(frames, ignore_index=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"aeo_trials_shard{shard_idx}.csv")
    out.to_csv(path, index=False)
    volume.commit()
    print(f"[shard {shard_idx}] {len(out)} rows -> {path}")
    return {"shard": shard_idx, "n_configs": len(cfgs), "n_rows": len(out)}


# ----------------------------------------------------------------------
# Aggregate
# ----------------------------------------------------------------------
@app.function(image=image, volumes={"/data": volume}, memory=8192, timeout=1800)
def aggregate(n_shards=N_SHARDS):
    tcp, _ = _setup()
    import pandas as pd

    volume.reload()
    aeo = pd.concat(
        [pd.read_csv(os.path.join(OUT_DIR, f"aeo_trials_shard{i}.csv"))
         for i in range(n_shards)
         if os.path.exists(os.path.join(OUT_DIR, f"aeo_trials_shard{i}.csv"))],
        ignore_index=True)
    pca = pd.read_csv(os.path.join(OUT_DIR, "pca_trials.csv"))

    aeo.to_csv(os.path.join(OUT_DIR, "aeo_trials.csv"), index=False)
    comp = tcp.build_comparison(aeo, pca)
    comp.to_csv(os.path.join(OUT_DIR, "comparison.csv"), index=False)

    cols = ["fault_number", "variant", "far", "mdr", "delay_samples",
            "detection_rate", "balanced_accuracy"]
    pd.set_option("display.width", 220)
    print(f"\nAEO trials aggregated: {aeo.trial.nunique()} configs, {len(aeo)} rows")
    print("\n=== CORRECTED ALL-20 COMPARISON (matched FAR, held-out runs) ===")
    print(comp[cols].sort_values(["fault_number", "variant"]).to_string(index=False))

    volume.commit()
    return comp.shape


@app.local_entrypoint()
def main(n_trials: int = 60, search_seed: int = 0, model_seed: int = 42,
         n_shards: int = N_SHARDS):
    print(f"Corrected all-20 run | trials={n_trials} shards={n_shards} "
          f"model_seed={model_seed}")

    print("\nStep 1/3: PCA grid ...")
    run_pca_all20.remote()

    print(f"\nStep 2/3: AEO search across {n_shards} shards "
          f"(max {MAX_CONTAINERS} containers) ...")
    jobs = [(i, n_shards, n_trials, search_seed, model_seed) for i in range(n_shards)]
    for r in run_aeo_shard.map(jobs):
        print("  shard done:", r)

    print("\nStep 3/3: aggregating ...")
    aggregate.remote(n_shards)

    print("\nPull results with:")
    print('  modal volume get tep-data all20/corrected <local_dir>')
