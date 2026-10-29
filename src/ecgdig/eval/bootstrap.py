"""Bootstrap tingkat pasien (resample pasien, bukan rekaman).

Untuk P pasien unik, satu replika b menarik P pasien dengan pengembalian; banyaknya tarikan
pasien j, c_bj ~ Multinomial(P, 1/P), menjadi bobot semua rekamannya: w_bi = c_{b, j(i)}.
Statistik replika dihitung dengan bobot w_b (setara menduplikasi rekaman).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def patient_bootstrap_weights(
    patient_ids: np.ndarray, n_resamples: int, rng: np.random.Generator
) -> np.ndarray:
    """Bobot integer (B, n) untuk n rekaman."""
    _, inv = np.unique(np.asarray(patient_ids), return_inverse=True)
    n_pat = int(np.max(inv)) + 1
    counts = rng.multinomial(n_pat, np.full(n_pat, 1.0 / n_pat), size=n_resamples)
    return counts[:, inv].astype(np.int32)


def percentile_ci(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """CI persentil (abaikan NaN)."""
    lo = (1.0 - level) / 2.0
    s = np.asarray(samples, dtype=np.float64)
    return float(np.nanquantile(s, lo)), float(np.nanquantile(s, 1.0 - lo))


def bootstrap_stat(
    stat: Callable[[np.ndarray], float], weights: np.ndarray, level: float = 0.95
) -> tuple[float, float, np.ndarray]:
    """Terapkan ``stat(w)`` ke setiap replika; return (ci_lo, ci_hi, sampel)."""
    vals = np.array([stat(w) for w in weights], dtype=np.float64)
    lo, hi = percentile_ci(vals, level)
    return lo, hi, vals


def weighted_rate(hit: np.ndarray, mask: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Proporsi berbobot Σ w·hit·mask / Σ w·mask untuk setiap replika, shape (B,)."""
    m = np.asarray(mask, dtype=np.float64)
    h = np.asarray(hit, dtype=np.float64) * m
    w = np.asarray(weights, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (w @ h) / (w @ m)
