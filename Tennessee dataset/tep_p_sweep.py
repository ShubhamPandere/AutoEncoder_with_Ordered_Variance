# -*- coding: utf-8 -*-
"""
Sweep over how many AEO latent dimensions (`p`) are used in the T^2 statistic,
holding everything else (the tuned latent_dim/loss_alpha/beta/gamma/epochs/lr/
alpha per fault) fixed. Trains the AEO model ONCE per fault (on the same
fault-free training window used everywhere else), then re-derives Y0/cov/Cl
for each `p` strategy and evaluates across ALL held-out validation runs
(now 9 runs per fault: 2-10, since run 1 is the tuning run), so results are
averaged over far more data than the original 2-run validation.

p strategies tested:
  - the baseline auto-select-until-90%-cumulative-variance rule (as in aeo_monitoring)
  - raising that threshold to 95% / 99%
  - fixed p in {2, 4, 8, latent_dim}
"""

import os
import time

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from scipy.stats import f as f_dist

from tep_evaluation import (
    load_tep_run, build_monitoring_sequence, fault_onset_index,
    evaluate_detection, DATA_DIR, RESULTS_DIR, TABLES_DIR, PLOTS_DIR, _paeo,
)
from tep_timeseries import load_best_configs

AutoencoderAEO = _paeo.AutoencoderAEO
compute_variances = _paeo.compute_variances
is_ordered = _paeo.is_ordered
create_permutation_matrix = _paeo.create_permutation_matrix
apply_permutation_to_model = _paeo.apply_permutation_to_model
normalize = _paeo.normalize
set_seed = _paeo.set_seed

VALIDATION_RUNS = list(range(2, 11))  # runs 2-10 (run 1 was used for tuning)


def train_aeo_once(fault_free_run, window_size, latent_dim, loss_alpha, beta,
                    gamma, epochs, lr, max_iterations=10):
    """Train Algorithm-1 exactly once on the fixed fault-free training window."""
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    X0 = fault_free_run[:window_size]
    Xn0, mu, sigma = normalize(X0)
    Xt0 = torch.FloatTensor(Xn0).to(device)
    n = X0.shape[1]

    weights = torch.linspace(0, latent_dim - 1, latent_dim).to(device)
    Q = torch.diag(weights)
    model = AutoencoderAEO(n, latent_dim).to(device)

    for iteration in range(max_iterations):
        optimizer = optim.Adam(model.parameters(), lr=lr)
        model.train()
        for ep in range(epochs):
            x_recon, y = model(Xt0)
            j1 = loss_alpha * torch.sum((Xt0 - x_recon) ** 2)
            y_centered = (y - y.mean(0, keepdim=True)).T
            j2 = beta * torch.trace(y_centered.T @ Q @ y_centered)
            j3 = gamma * sum(p.pow(2).sum() for p in model.parameters())
            loss = j1 + j2 + j3
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        variances = compute_variances(model, Xt0, device)
        if is_ordered(variances):
            break
        if iteration < max_iterations - 1:
            P, _ = create_permutation_matrix(variances, device)
            apply_permutation_to_model(model, P, device)

    model.eval()
    with torch.no_grad():
        _, Y0t = model(Xt0)
    Y0 = Y0t.cpu().numpy()

    return model, mu, sigma, Y0, device


def p_dims_for_strategy(Y0, strategy):
    """strategy: ('threshold', 0.90) or ('fixed', 4)"""
    variances = Y0.var(axis=0)
    total_var = variances.sum() + 1e-10
    cum_var = variances.cumsum() / total_var
    kind, val = strategy
    if kind == "threshold":
        return max(2, int((cum_var < val).sum()) + 1)
    else:
        return int(min(val, Y0.shape[1]))


def build_control_stats(Y0, p_dims, window_size, alpha):
    Y0_selected = Y0[:, :p_dims]
    Y0_mean = Y0_selected.mean(axis=0)
    covY = np.cov((Y0_selected - Y0_mean).T) + 1e-6 * np.eye(p_dims)
    cond_number = np.linalg.cond(covY)
    Qy = np.linalg.inv(covY)

    F_crit = f_dist.ppf(1 - alpha, p_dims, window_size - p_dims)
    Cl = (p_dims * (window_size - 1) * (window_size + 1)) / \
         (window_size * (window_size - p_dims)) * F_crit
    return Y0_selected, Y0_mean, Qy, Cl, cond_number


def compute_T2_for_run(model, mu, sigma, device, data, window_size,
                        Y0_selected, Y0_mean, Qy, p_dims):
    N = data.shape[0]
    T2 = np.zeros(N)
    for i in range(window_size):
        yi_centered = Y0_selected[i] - Y0_mean
        T2[i] = yi_centered @ Qy @ yi_centered
    for k in range(window_size, N):
        xk_norm = (data[k] - mu) / sigma
        xk_tensor = torch.FloatTensor(xk_norm).unsqueeze(0).to(device)
        with torch.no_grad():
            _, yk_tensor = model(xk_tensor)
            yk = yk_tensor.cpu().numpy().flatten()
        yk_selected = yk[:p_dims]
        yk_centered = yk_selected - Y0_mean
        T2[k] = yk_centered @ Qy @ yk_centered
    return T2


def sweep_fault(fault_number, cfg, window_size=200, split="test",
                 fault_free_run_id=1, data_dir=DATA_DIR,
                 validation_runs=VALIDATION_RUNS):
    onset_idx = fault_onset_index(window_size, split=split)
    fault_free_run = load_tep_run(0, fault_free_run_id, split=split, data_dir=data_dir)

    t0 = time.time()
    model, mu, sigma, Y0, device = train_aeo_once(
        fault_free_run, window_size,
        latent_dim=cfg["latent_dim"], loss_alpha=cfg["loss_alpha"],
        beta=cfg["beta"], gamma=cfg["gamma"], epochs=cfg["epochs"], lr=cfg["lr"],
    )
    print(f"  [fault {fault_number}] trained once in {time.time()-t0:.1f}s "
          f"(latent_dim={cfg['latent_dim']}, epochs={cfg['epochs']})")

    # Pre-load all validation faulty runs once
    faulty_runs = {
        run_id: load_tep_run(fault_number, run_id, split=split, data_dir=data_dir)
        for run_id in validation_runs
    }

    strategies = [
        ("threshold", 0.90, "baseline_90pct"),
        ("threshold", 0.95, "threshold_95pct"),
        ("threshold", 0.99, "threshold_99pct"),
        ("fixed", 2, "fixed_p2"),
        ("fixed", 4, "fixed_p4"),
        ("fixed", 8, "fixed_p8"),
        ("fixed", cfg["latent_dim"], "fixed_p_full_latent"),
    ]
    # de-duplicate by (kind, val)
    seen = set()
    dedup_strategies = []
    for kind, val, label in strategies:
        key = (kind, val)
        if key not in seen:
            seen.add(key)
            dedup_strategies.append((kind, val, label))

    rows = []
    for kind, val, label in dedup_strategies:
        p_dims = p_dims_for_strategy(Y0, (kind, val))
        Y0_selected, Y0_mean, Qy, Cl, cond_number = build_control_stats(
            Y0, p_dims, window_size, cfg["alpha"]
        )
        for run_id, faulty_run in faulty_runs.items():
            data, w = build_monitoring_sequence(fault_free_run, faulty_run, window_size)
            T2 = compute_T2_for_run(model, mu, sigma, device, data, w,
                                     Y0_selected, Y0_mean, Qy, p_dims)
            metrics = evaluate_detection(T2, Cl, w, onset_idx)
            rows.append({
                "fault_number": fault_number, "p_strategy": label,
                "p_dims": p_dims, "cond_number": cond_number, "Cl": Cl,
                "run": run_id, **metrics,
            })

    return pd.DataFrame(rows)


def run_all(fault_numbers=(1, 4, 6, 8, 3, 9, 15)):
    aeo_cfgs, _ = load_best_configs()
    all_rows = []
    for fault_number in fault_numbers:
        print(f"\n=== p-sweep for fault {fault_number} ===")
        df = sweep_fault(fault_number, aeo_cfgs[fault_number])
        all_rows.append(df)
    result = pd.concat(all_rows, ignore_index=True)
    result.to_csv(os.path.join(TABLES_DIR, "tep_p_sweep_raw.csv"), index=False)
    return result


def plot_p_comparison(fault_number, cfg, baseline_strategy=("threshold", 0.90),
                       higher_strategy=("fixed", 4), run_id=2, window_size=200,
                       split="test", fault_free_run_id=1, data_dir=DATA_DIR,
                       zoom_margin_pre=60, zoom_margin_post=400):
    """
    Plot baseline-p vs higher-p AEO T^2 curves on the SAME held-out run, to
    visually show whether more latent dimensions change detection speed --
    not AEO vs PCA, just AEO(low p) vs AEO(high p) on identical everything else.
    """
    import matplotlib.pyplot as plt

    onset_idx = fault_onset_index(window_size, split=split)
    fault_free_run = load_tep_run(0, fault_free_run_id, split=split, data_dir=data_dir)
    faulty_run = load_tep_run(fault_number, run_id, split=split, data_dir=data_dir)

    model, mu, sigma, Y0, device = train_aeo_once(
        fault_free_run, window_size,
        latent_dim=cfg["latent_dim"], loss_alpha=cfg["loss_alpha"],
        beta=cfg["beta"], gamma=cfg["gamma"], epochs=cfg["epochs"], lr=cfg["lr"],
    )
    data, w = build_monitoring_sequence(fault_free_run, faulty_run, window_size)

    curves = {}
    for strategy, label in [(baseline_strategy, "baseline p"), (higher_strategy, "higher p")]:
        p_dims = p_dims_for_strategy(Y0, strategy)
        Y0_selected, Y0_mean, Qy, Cl, _ = build_control_stats(Y0, p_dims, w, cfg["alpha"])
        T2 = compute_T2_for_run(model, mu, sigma, device, data, w, Y0_selected, Y0_mean, Qy, p_dims)
        detected = [i for i in range(w, len(T2)) if T2[i] > Cl]
        curves[label] = (T2, Cl, p_dims, detected)

    zoom_lo = max(0, onset_idx - zoom_margin_pre)
    zoom_hi = min(len(data), onset_idx + zoom_margin_post)

    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    colors = {"baseline p": "royalblue", "higher p": "crimson"}
    for label, (T2, Cl, p_dims, detected) in curves.items():
        log_t2 = np.log10(T2 + 1e-6)
        ax.plot(log_t2, label=f"{label} (p={p_dims}) log10 T²", color=colors[label], linewidth=2, alpha=0.8)
        ax.axhline(np.log10(Cl), color=colors[label], linestyle="--", alpha=0.6,
                   label=f"{label} limit (p={p_dims})")
        post_onset = [i for i in detected if i >= onset_idx]
        if post_onset:
            first = post_onset[0]
            ax.axvline(first, color=colors[label], linestyle=":", linewidth=1.5)
            ax.scatter(first, log_t2[first], color=colors[label], marker="x", s=100, zorder=5)
            ax.annotate(f"{label} first detection\nt={first} (delay={first-onset_idx})",
                        xy=(first, log_t2[first]),
                        xytext=(0.05 if label == "higher p" else 0.65, 0.05 if label == "higher p" else 0.90),
                        textcoords="axes fraction", fontsize=9, fontweight="bold", color=colors[label],
                        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=colors[label], alpha=0.9))

    ax.axvline(onset_idx, color="black", linewidth=1.5, alpha=0.6, label="True fault onset")
    ax.set_xlim(zoom_lo, zoom_hi)
    ax.set_xlabel("Time-Step (Sample Index)")
    ax.set_ylabel("log10 T² Score")
    ax.set_title(f"IDV({fault_number}): AEO baseline-p vs higher-p [held-out run {run_id}]")
    ax.legend(loc="upper left", fontsize=8, ncol=1)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, f"tep_p_comparison_fault{fault_number}.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[SAVED] {save_path}")
    return fig


def summarize(result):
    summary = result.groupby(["fault_number", "p_strategy"]).agg(
        p_dims=("p_dims", "first"),
        cond_number=("cond_number", "first"),
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        far_mean=("false_alarm_rate", "mean"),
        mdr_mean=("missed_detection_rate", "mean"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        delay_mean=("detection_delay_samples", "mean"),
    ).reset_index()
    summary.to_csv(os.path.join(TABLES_DIR, "tep_p_sweep_summary.csv"), index=False)
    return summary


if __name__ == "__main__":
    result = run_all()
    summary = summarize(result)
    pd.set_option("display.width", 200)
    print("\n\n=== P-SWEEP SUMMARY (mean over 9 held-out runs) ===")
    print(summary.to_string())
