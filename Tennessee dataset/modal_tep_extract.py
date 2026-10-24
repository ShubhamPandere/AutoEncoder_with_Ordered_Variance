# -*- coding: utf-8 -*-
"""
Modal remote job: extract more TEP simulation runs from the large RData files
(they need ~7-8GB peak RAM to parse, more than was reliably free locally).

Usage:
    modal run modal_tep_extract.py
"""

import modal

app = modal.App("tep-extract")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("pandas", "pyreadr")
)

volume = modal.Volume.from_name("tep-data")

FAULTS = [1, 3, 4, 6, 8, 9, 15]
RUNS = list(range(1, 11))  # 10 runs per fault


@app.function(image=image, volumes={"/data": volume}, memory=16384, timeout=1800)
def extract_subset():
    import pyreadr
    import pandas as pd

    print("Loading TEP_Faulty_Testing.RData ...")
    res = pyreadr.read_r("/data/TEP_Faulty_Testing.RData")
    df = list(res.values())[0]
    del res
    faulty_sub = df[(df["faultNumber"].isin(FAULTS)) & (df["simulationRun"].isin(RUNS))].copy()
    del df
    print("faulty subset shape:", faulty_sub.shape)

    print("Loading TEP_FaultFree_Testing.RData ...")
    res2 = pyreadr.read_r("/data/TEP_FaultFree_Testing.RData")
    df2 = list(res2.values())[0]
    del res2
    freefault_sub = df2[df2["simulationRun"].isin(RUNS)].copy()
    del df2
    print("fault-free subset shape:", freefault_sub.shape)

    combined = pd.concat([freefault_sub, faulty_sub], ignore_index=True)
    print("combined shape:", combined.shape)
    print(combined["faultNumber"].value_counts().sort_index())

    out_path = "/data/tep_test_subset10.csv"
    combined.to_csv(out_path, index=False)
    volume.commit()
    print("Saved:", out_path)
    return combined.shape


@app.local_entrypoint()
def main():
    shape = extract_subset.remote()
    print("Done, shape:", shape)
