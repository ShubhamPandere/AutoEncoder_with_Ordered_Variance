# -*- coding: utf-8 -*-
"""
Diagnostics for "why is AEO underperforming on some TEP faults" -- reruns the
exact same Algorithm-1 training loop as `aeo_monitoring` in `p_aeo (1).py`
(copied here read-only for instrumentation, NOT modifying the original),
but additionally reports:
  - how many permutation iterations were needed / whether it converged at all
  - the actual latent variances at convergence (concentrated vs spread out)
  - `p`, the number of latent dims kept for T^2 (should be small; if it's
    close to latent_dim, the "ordered variance" property didn't really kick in)
  - condition number of the covariance matrix inverted for T^2 (if huge, the
    T^2 statistic is numerically unstable regardless of anything else)
"""

import numpy as np
import torch
import torch.optim as optim
import pandas as pd

from tep_evaluation import (
    load_tep_run, build_monitoring_sequence, fault_onset_index,
    evaluate_detection, DATA_DIR, RESULTS_DIR, TABLES_DIR, PLOTS_DIR, _paeo,
)

AutoencoderAEO = _paeo.AutoencoderAEO
compute_variances = _paeo.compute_variances
is_ordered = _paeo.is_ordered
create_permutation_matrix = _paeo.create_permutation_matrix
apply_permutation_to_model = _paeo.apply_permutation_to_model
normalize = _paeo.normalize
set_seed = _paeo.set_seed


def diagnostic_aeo_run(
    data, window_size, latent_dim, loss_alpha, beta, gamma, epochs, lr,
    alpha, max_iterations=10, onset_idx=None,
):
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    N, n = data.shape
    X0 = data[:window_size]
    Xn0, mu, sigma = normalize(X0)
    Xt0 = torch.FloatTensor(Xn0).to(device)

    weights = torch.linspace(0, latent_dim - 1, latent_dim).to(device)
    Q = torch.diag(weights)
    model = AutoencoderAEO(n, latent_dim).to(device)

    converged_at = None
    final_variances = None
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
        final_variances = variances
        if is_ordered(variances):
            converged_at = iteration
            break
        if iteration < max_iterations - 1:
            P, _ = create_permutation_matrix(variances, device)
            apply_permutation_to_model(model, P, device)

    # ---- Monitoring phase (identical to aeo_monitoring) ----
    model.eval()
    with torch.no_grad():
        _, Y0t = model(Xt0)
    Y0 = Y0t.cpu().numpy()

    variances = Y0.var(axis=0)
    total_var = variances.sum() + 1e-10
    cum_var = variances.cumsum() / total_var
    p_dims = max(2, int((cum_var < 0.90).sum()) + 1)

    Y0_selected = Y0[:, :p_dims]
    Y0_mean = Y0_selected.mean(axis=0)
    covY = np.cov((Y0_selected - Y0_mean).T) + 1e-6 * np.eye(p_dims)
    cond_number = np.linalg.cond(covY)
    Qy = np.linalg.inv(covY)

    from scipy.stats import f as f_dist
    F_crit = f_dist.ppf(1 - alpha, p_dims, window_size - p_dims)
    Cl = (p_dims * (window_size - 1) * (window_size + 1)) / \
         (window_size * (window_size - p_dims)) * F_crit

    T2 = np.zeros(N)
    detected = []
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
        if T2[k] > Cl:
            detected.append(k)

    diag = {
        "latent_dim": latent_dim,
        "epochs": epochs,
        "converged": converged_at is not None,
        "converged_at_iter": converged_at,
        "final_variances": np.round(final_variances, 4).tolist(),
        "variance_concentration_top1_pct": float(final_variances[0] / (final_variances.sum() + 1e-10) * 100),
        "p_dims_selected": p_dims,
        "cond_number_covY": cond_number,
        "Cl": Cl,
    }
    if onset_idx is not None:
        diag.update(evaluate_detection(T2, Cl, window_size, onset_idx))
    return diag, T2, Cl, detected


def run_diagnostics_for_fault(fault_number, base_cfg, run_id=1, window_size=200,
                               split="test", fault_free_run_id=1, data_dir=DATA_DIR,
                               epoch_variants=(2000, 5000), latent_dim_variants=(4, 8, 16)):
    onset_idx = fault_onset_index(window_size, split=split)
    fault_free_run = load_tep_run(0, fault_free_run_id, split=split, data_dir=data_dir)
    faulty_run = load_tep_run(fault_number, run_id, split=split, data_dir=data_dir)
    data, w = build_monitoring_sequence(fault_free_run, faulty_run, window_size)

    rows = []

    print(f"\n=== Fault {fault_number}: baseline tuned config ===")
    diag, *_ = diagnostic_aeo_run(data, w, onset_idx=onset_idx, **base_cfg)
    print(diag)
    rows.append({"fault_number": fault_number, "variant": "tuned_baseline", **diag})

    for ep in epoch_variants:
        if ep == base_cfg["epochs"]:
            continue
        cfg = dict(base_cfg, epochs=ep)
        print(f"\n=== Fault {fault_number}: epochs={ep} ===")
        diag, *_ = diagnostic_aeo_run(data, w, onset_idx=onset_idx, **cfg)
        print(diag)
        rows.append({"fault_number": fault_number, "variant": f"epochs_{ep}", **diag})

    for ld in latent_dim_variants:
        if ld == base_cfg["latent_dim"]:
            continue
        cfg = dict(base_cfg, latent_dim=ld)
        print(f"\n=== Fault {fault_number}: latent_dim={ld} ===")
        diag, *_ = diagnostic_aeo_run(data, w, onset_idx=onset_idx, **cfg)
        print(diag)
        rows.append({"fault_number": fault_number, "variant": f"latent_dim_{ld}", **diag})

    return pd.DataFrame(rows)


if __name__ == "__main__":
    from tep_timeseries import load_best_configs

    aeo_cfgs, _ = load_best_configs()
    all_rows = []
    for fault_number in [1, 6, 4, 9]:
        cfg = aeo_cfgs[fault_number]
        base_cfg = dict(
            latent_dim=cfg["latent_dim"], loss_alpha=cfg["loss_alpha"],
            beta=cfg["beta"], gamma=cfg["gamma"], epochs=cfg["epochs"],
            lr=cfg["lr"], alpha=cfg["alpha"],
        )
        df = run_diagnostics_for_fault(fault_number, base_cfg)
        all_rows.append(df)

    result = pd.concat(all_rows, ignore_index=True)
    result.to_csv(f"{TABLES_DIR}/tep_aeo_diagnostics.csv", index=False)
    print("\n\n=== FULL DIAGNOSTIC TABLE ===")
    cols = ["fault_number", "variant", "latent_dim", "epochs", "converged",
            "converged_at_iter", "variance_concentration_top1_pct",
            "p_dims_selected", "cond_number_covY", "accuracy", "f1",
            "false_alarm_rate", "missed_detection_rate"]
    print(result[cols].to_string())
