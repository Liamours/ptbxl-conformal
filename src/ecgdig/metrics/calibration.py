"""Metrik kalibrasi per kelas: ECE (lebar sama / massa sama) dan dekomposisi Brier.

ECE_k = Σ_{m=1}^{M} (|B_{k,m}| / n) · |ȳ_{k,m} − p̄_{k,m}|.
- Lebar sama: B_m = [(m−1)/M, m/M) (bin terakhir tertutup di 1).
- Massa sama: B_m = blok berurutan berukuran ≈ n/M setelah p diurutkan.
macro-ECE = rerata ECE_k.

Dekomposisi Brier dengan binning (identitas eksak, diturunkan dari
p_i − y_i = (p_i − p̄_m) + (p̄_m − ȳ_m) + (ȳ_m − y_i)):
    BS = REL − RES + UNC + WBV − 2·WBC,
    REL = Σ_m (n_m/n)(p̄_m − ȳ_m)²,  RES = Σ_m (n_m/n)(ȳ_m − ȳ)²,  UNC = ȳ(1 − ȳ),
    WBV = (1/n) Σ_m Σ_{i∈m} (p_i − p̄_m)²,  WBC = (1/n) Σ_m Σ_{i∈m} (p_i − p̄_m)(y_i − ȳ_m).
REL kecil = terkalibrasi; RES besar = diskriminatif; UNC hanya bergantung prevalensi.
"""

from __future__ import annotations

import numpy as np


def _bin_index(p: np.ndarray, n_bins: int, scheme: str) -> np.ndarray:
    if scheme == "width":
        return np.clip(np.floor(p * n_bins).astype(np.int64), 0, n_bins - 1)
    if scheme == "mass":
        order = np.argsort(p, kind="mergesort")
        idx = np.empty(p.size, dtype=np.int64)
        idx[order] = (np.arange(p.size) * n_bins) // p.size
        return idx
    raise ValueError(f"skema bin tidak dikenal: {scheme}")


def ece_binary(p: np.ndarray, y: np.ndarray, n_bins: int = 15, scheme: str = "width") -> float:
    """ECE untuk satu kelas biner."""
    p = np.asarray(p, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    idx = _bin_index(p, n_bins, scheme)
    cnt = np.bincount(idx, minlength=n_bins)
    sum_p = np.bincount(idx, weights=p, minlength=n_bins)
    sum_y = np.bincount(idx, weights=y, minlength=n_bins)
    nz = cnt > 0
    return float(np.sum(np.abs(sum_y[nz] - sum_p[nz])) / p.size)


def ece_per_class(
    probs: np.ndarray, y: np.ndarray, n_bins: int = 15, scheme: str = "width"
) -> np.ndarray:
    """ECE per kelas, shape (K,)."""
    return np.array(
        [ece_binary(probs[:, k], y[:, k], n_bins, scheme) for k in range(probs.shape[1])]
    )


def brier_decomposition(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> dict[str, float]:
    """Brier satu kelas dan suku-sukunya (bin lebar sama); ``bs`` = rel − res + unc + wbv − 2wbc."""
    p = np.asarray(p, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = p.size
    idx = _bin_index(p, n_bins, "width")
    cnt = np.bincount(idx, minlength=n_bins).astype(np.float64)
    nz = cnt > 0
    pbar = np.bincount(idx, weights=p, minlength=n_bins)[nz] / cnt[nz]
    ybar = np.bincount(idx, weights=y, minlength=n_bins)[nz] / cnt[nz]
    w = cnt[nz] / n
    yall = y.mean()
    dp = p - _scatter(pbar, nz, n_bins)[idx]
    dy = y - _scatter(ybar, nz, n_bins)[idx]
    return {
        "bs": float(np.mean((p - y) ** 2)),
        "rel": float(np.sum(w * (pbar - ybar) ** 2)),
        "res": float(np.sum(w * (ybar - yall) ** 2)),
        "unc": float(yall * (1 - yall)),
        "wbv": float(np.mean(dp**2)),
        "wbc": float(np.mean(dp * dy)),
    }


def _scatter(values: np.ndarray, mask: np.ndarray, size: int) -> np.ndarray:
    out = np.zeros(size)
    out[mask] = values
    return out
