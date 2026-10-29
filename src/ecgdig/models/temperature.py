"""Temperature scaling untuk logit multi-label (sigmoid per kelas).

p_k = σ(z_k / T). T* = argmin_T NLL(T) dengan
    NLL(T) = (1/(nK)) Σ_{i,k} [softplus(z_ik/T) − y_ik z_ik/T],
dioptimasi pada u = log T (pencarian terbatas Brent, u ∈ [log_t_min, log_t_max]).
T bersama tidak mengubah urutan skor per kelas, jadi AUROC dan himpunan conformal
(M1, M2, M3) invarian; yang berubah hanya ECE/Brier/NLL.

Catatan: BCE berbobot (pos_weight w_k) menggeser logit optimum sebesar log w_k
(σ(z*) = w p / (w p + 1 − p)); pergeseran aditif ini tidak bisa dikoreksi oleh T saja.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def sigmoid(z: np.ndarray) -> np.ndarray:
    """σ(z) stabil numerik."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def bce_nll(logits: np.ndarray, y: np.ndarray) -> float:
    """Rerata BCE dari logit: mean(softplus(z) − y z)."""
    z = np.asarray(logits, dtype=np.float64)
    return float(np.mean(np.logaddexp(0.0, z) - np.asarray(y, dtype=np.float64) * z))


def fit_temperature(
    logits: np.ndarray, y: np.ndarray, log_t_min: float = -3.0, log_t_max: float = 3.0
) -> float:
    """T* bersama untuk semua kelas (minimasi NLL di VAL_A)."""
    z = np.asarray(logits, dtype=np.float64)
    res = minimize_scalar(
        lambda u: bce_nll(z / np.exp(u), y),
        bounds=(log_t_min, log_t_max),
        method="bounded",
        options={"xatol": 1e-6},
    )
    return float(np.exp(res.x))


def fit_temperature_per_class(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    """T*_k per kelas (ablation)."""
    z, yy = np.asarray(logits), np.asarray(y)
    return np.array([fit_temperature(z[:, [k]], yy[:, [k]]) for k in range(z.shape[1])])


def apply_temperature(logits: np.ndarray, temperature: float | np.ndarray) -> np.ndarray:
    """Probabilitas terkalibrasi σ(z / T)."""
    return sigmoid(np.asarray(logits, dtype=np.float64) / np.asarray(temperature))
