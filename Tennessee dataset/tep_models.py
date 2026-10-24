# -*- coding: utf-8 -*-
"""
AEO and PCA monitors with both a T^2 and a residual (SPE) statistic.
====================================================================

Why this file exists (audit A3): IDV(4)'s signature lives in principal
components 36-39 -- the directions carrying the LAST 1% of the variance. PCA
only detected it because the F1-driven tuner picked Vpc=99, which retains 39 of
52 directions and so turns its "T^2" into a whitened full-space residual test.
AEO, compressing 52 variables into latent_dim <= 16, cannot represent those
directions at ANY value of `p` -- `p` only selects among latent dims that exist,
while the discarded 36 directions live solely in the reconstruction error.

So this module gives BOTH methods the textbook pair of statistics:

    T^2  = (y - y_bar)' S_y^-1 (y - y_bar)      -- variation inside the model
    SPE  = || x - x_hat ||^2                     -- variation the model cannot explain

and lets the caller ask for T2-only, SPE-only, or the combined index. That makes
the two comparisons the plan asks for possible:

    AEO-T2        vs  PCA-T2 (Vpc capped at 95)   -- honest like-for-like
    AEO-T2+SPE    vs  PCA-T2+SPE                  -- standard TEP protocol

The AEO training loop itself is Algorithm 1 from the fault-detection paper,
unchanged in substance from `p_aeo (1).py`, with four audit fixes:
  * a real `model_seed` instead of a hardcoded set_seed(42)         (A6)
  * a tunable hidden width instead of a fixed 16 units              (A7f)
  * a tunable Q shape instead of a hardcoded linear ramp            (A7e)
  * a warning when Algorithm 1 exits without achieving ordering     (A7j)
"""

import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def normalize_fit(X):
    """Mean/std from the training window; floor tiny sigmas as in the original."""
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma = np.where(sigma < 1e-3, 1e-3, sigma)
    return mu, sigma


def _t2_stat(Y, mean, Qy):
    """Hotelling T^2 for a stack of latent/score vectors (vectorised)."""
    Z = np.asarray(Y, dtype=float) - mean
    return np.einsum("ij,jk,ik->i", Z, Qy, Z)


def q_weights(latent_dim, shape="linear"):
    """
    Diagonal of Q, satisfying the paper's requirement 0 <= q1 < q2 < ... < qm.

    `p_aeo (1).py` hardcodes the linear ramp. The AEO paper treats the shape of Q
    as a primary tuning parameter (diag(1, q^2) for m=2, diag(0.1, 0.1q, 0.5q,
    4q, 5q) for m=5, with q swept 1..10) and shows the results depend on it, so
    it belongs in the search space (audit A7e).
    """
    j = np.arange(latent_dim, dtype=float)
    if shape == "linear":
        w = j
    elif shape == "quadratic":
        w = j ** 2
    elif shape == "geometric":
        w = 2.0 ** j - 1.0
    else:
        raise ValueError(f"unknown Q shape: {shape}")
    # scale so the largest weight is latent_dim-1, keeping beta comparable across shapes
    if w.max() > 0:
        w = w / w.max() * (latent_dim - 1)
    return w


# ----------------------------------------------------------------------
# AEO
# ----------------------------------------------------------------------
class AutoencoderAEO(nn.Module):
    """Same topology as `p_aeo (1).py`, but with the hidden width exposed."""

    def __init__(self, input_dim, latent_dim=8, hidden=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(), nn.Linear(hidden, input_dim)
        )

    def forward(self, x):
        y = self.encoder(x)
        return self.decoder(y), y


def _is_ordered(v, tol=1e-6):
    return bool(np.all(v[:-1] >= v[1:] - tol))


def _permute(model, variances, device):
    """Absorb the sorting permutation P into the encoder head / decoder input."""
    order = np.argsort(variances)[::-1].copy()
    P = torch.eye(len(variances), device=device)[order, :]
    enc_last = list(model.encoder.children())[-1]
    dec_first = list(model.decoder.children())[0]
    with torch.no_grad():
        enc_last.weight.data = P @ enc_last.weight.data
        enc_last.bias.data = P @ enc_last.bias.data
        dec_first.weight.data = dec_first.weight.data @ P.T


class AEOMonitor:
    """Algorithm 1 training + T^2 / SPE monitoring statistics."""

    def __init__(self, latent_dim=8, hidden=16, loss_alpha=1.0, beta=0.5, gamma=0.1,
                 epochs=2000, lr=2e-3, q_shape="linear", p_dims=None,
                 max_iterations=10, model_seed=42, verbose=False,
                 enforce_ordering=True, sort_latent=False):
        """
        enforce_ordering
            Run Algorithm 1 (train -> permute -> retrain until variances are
            ordered). Set False for the plain-autoencoder ablation (beta = 0),
            where there is no ordering pressure in the loss, so the loop would
            never converge and would simply burn `max_iterations` trainings.
        sort_latent
            Sort the latent dimensions by descending variance *after* training.
            This is what any practitioner could do to a plain AE without the
            regularizer, so it is the fair comparison for AEO's ordering claim
            -- comparing only against unsorted dimensions would be a strawman.
        """
        self.cfg = dict(latent_dim=latent_dim, hidden=hidden, loss_alpha=loss_alpha,
                        beta=beta, gamma=gamma, epochs=epochs, lr=lr,
                        q_shape=q_shape, p_dims=p_dims, max_iterations=max_iterations,
                        model_seed=model_seed, enforce_ordering=enforce_ordering,
                        sort_latent=sort_latent)
        self.verbose = verbose
        self.converged = None
        self.latent_variances = None
        self.order = None

    def fit(self, X_train):
        c = self.cfg
        set_seed(c["model_seed"])
        device = DEVICE
        self.device = device

        self.mu, self.sigma = normalize_fit(X_train)
        Xn = (X_train - self.mu) / self.sigma
        Xt = torch.FloatTensor(Xn).to(device)
        n = X_train.shape[1]

        qvec = torch.tensor(q_weights(c["latent_dim"], c["q_shape"]),
                            dtype=torch.float32, device=device)
        model = AutoencoderAEO(n, c["latent_dim"], c["hidden"]).to(device)

        n_iter = c["max_iterations"] if c["enforce_ordering"] else 1
        converged_at = None
        for it in range(n_iter):
            opt = optim.Adam(model.parameters(), lr=c["lr"])
            model.train()
            for _ in range(c["epochs"]):
                x_recon, y = model(Xt)
                j1 = c["loss_alpha"] * torch.sum((Xt - x_recon) ** 2)
                yc = y - y.mean(0, keepdim=True)
                # tr(Yc' Q Yc) == sum_j q_j * sum_i yc_ij^2 -- same value as the
                # original N x N trace, without building an N x N matrix.
                j2 = c["beta"] * torch.sum((yc ** 2) * qvec)
                j3 = c["gamma"] * sum(p.pow(2).sum() for p in model.parameters())
                loss = j1 + j2 + j3
                opt.zero_grad()
                loss.backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                _, Y = model(Xt)
            v = Y.cpu().numpy().var(axis=0)
            if _is_ordered(v):
                converged_at = it
                break
            if it < n_iter - 1:
                _permute(model, v, device)

        self.model = model
        self.converged = converged_at is not None
        if not self.converged and c["enforce_ordering"]:
            # audit A7j: without ordering, Y0[:, :p] is NOT the top-p dims and the
            # whole variance-threshold selection rule below is meaningless.
            warnings.warn(
                f"AEO Algorithm 1 did not achieve variance ordering in "
                f"{c['max_iterations']} iterations (latent_dim={c['latent_dim']}, "
                f"epochs={c['epochs']}); T^2 dimension selection is unreliable.",
                RuntimeWarning,
            )

        model.eval()
        with torch.no_grad():
            Xrec, Y0t = model(Xt)
        Y0 = Y0t.cpu().numpy()

        # Optional post-hoc sort. AEO earns its ordering from the loss; a plain
        # AE can only be sorted afterwards, and `self.order` is applied to every
        # latent vector in `statistics` so the two are treated identically
        # downstream.
        if c["sort_latent"]:
            self.order = np.argsort(Y0.var(axis=0))[::-1].copy()
            Y0 = Y0[:, self.order]
        else:
            self.order = None

        self.Y0 = Y0
        self.latent_variances = Y0.var(axis=0)

        # --- how many latent dims enter T^2 ---
        # Audit A4: the 90%-cumulative-variance rule is degenerate here, because
        # J2 deliberately crushes every dim past the first (measured top-1
        # concentration 88-99%), so it returns p=2 almost always. Default to the
        # full latent space and let the caller override.
        self.set_p(c["p_dims"] or c["latent_dim"])
        return self

    def set_p(self, p):
        """
        Re-point T^2 at the leading `p` latent dimensions, reusing the fitted
        model. Lets a `p` sweep run off a single training, which is what makes
        the AE-vs-AEO ablation affordable.
        """
        self.p_dims = int(min(int(p), self.cfg["latent_dim"]))
        Y0s = self.Y0[:, :self.p_dims]
        self.Y0_mean = Y0s.mean(axis=0)
        covY = np.atleast_2d(np.cov(Y0s, rowvar=False)) + 1e-8 * np.eye(self.p_dims)
        self.cond_number = float(np.linalg.cond(covY))
        self.Qy = np.linalg.inv(covY)
        return self

    def latent_and_spe(self, X, batch=4096):
        """
        Return (Y, SPE) with Y the full (ordered) latent matrix.

        Separated from `statistics` so a sweep over p can reuse one forward
        pass: T^2 for any p is then just a Mahalanobis distance on Y[:, :p].
        """
        Xn = (X - self.mu) / self.sigma
        ys, spes = [], []
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(Xn), batch):
                xb = torch.FloatTensor(Xn[i:i + batch]).to(self.device)
                xrec, y = self.model(xb)
                spes.append(((xb - xrec) ** 2).sum(dim=1).cpu().numpy())
                yb = y.cpu().numpy()
                if self.order is not None:
                    yb = yb[:, self.order]
                ys.append(yb)
        return np.vstack(ys), np.concatenate(spes)

    def t2_from_latent(self, Y):
        """T^2 at the current p, from a full latent matrix."""
        return _t2_stat(Y[:, :self.p_dims], self.Y0_mean, self.Qy)

    def statistics(self, X, batch=4096):
        """Return (T2, SPE) for every row of X."""
        Xn = (X - self.mu) / self.sigma
        t2_parts, spe_parts = [], []
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(Xn), batch):
                xb = torch.FloatTensor(Xn[i:i + batch]).to(self.device)
                xrec, y = self.model(xb)
                spe_parts.append(((xb - xrec) ** 2).sum(dim=1).cpu().numpy())
                yb = y.cpu().numpy()
                if self.order is not None:
                    yb = yb[:, self.order]
                t2_parts.append(yb[:, :self.p_dims])
        Y = np.vstack(t2_parts)
        return _t2_stat(Y, self.Y0_mean, self.Qy), np.concatenate(spe_parts)


# ----------------------------------------------------------------------
# PCA
# ----------------------------------------------------------------------
class PCAMonitor:
    """
    Standard PCA monitor with T^2 on the retained components and SPE on the
    residual subspace.

    Fixes audit A7a: the original used `np.linalg.eig` on a symmetric covariance
    matrix, which returns complex values (silently `np.real`'d) and
    non-orthogonal eigenvectors for the near-degenerate tail -- precisely PCs
    32-39, the ones the Vpc=99 configs depend on. `eigh` is the correct routine.
    """

    def __init__(self, Vpc=95, n_components=None):
        self.Vpc = Vpc
        self.n_components = n_components

    def fit(self, X_train):
        self.mu, self.sigma = normalize_fit(X_train)
        Xn = (X_train - self.mu) / self.sigma
        C = np.cov(Xn, rowvar=False)
        vals, vecs = np.linalg.eigh(C)          # symmetric -> real, orthonormal
        order = np.argsort(vals)[::-1]
        self.eigvals, vecs = vals[order], vecs[:, order]

        if self.n_components is not None:
            Np = int(self.n_components)
        else:
            cum = np.cumsum(self.eigvals) / self.eigvals.sum() * 100
            Np = int(np.searchsorted(cum, self.Vpc) + 1)
        self.Np = max(1, min(Np, Xn.shape[1]))
        self.P = vecs[:, :self.Np]

        Y = Xn @ self.P
        self.Y_mean = Y.mean(axis=0)
        covY = np.cov(Y, rowvar=False) + 1e-8 * np.eye(self.Np)
        self.Qy = np.linalg.inv(np.atleast_2d(covY))
        self.cond_number = float(np.linalg.cond(np.atleast_2d(covY)))
        return self

    def statistics(self, X):
        Xn = (X - self.mu) / self.sigma
        Y = Xn @ self.P
        t2 = _t2_stat(Y, self.Y_mean, self.Qy)
        resid = Xn - Y @ self.P.T
        return t2, (resid ** 2).sum(axis=1)
