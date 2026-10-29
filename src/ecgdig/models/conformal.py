"""Split conformal prediction per label untuk klasifikasi multi-label (K kelas).

Notasi: p_ik = probabilitas kelas k untuk rekaman i; y_ik ∈ {0,1}. Untuk setiap kelas k,
himpunan prediksi C_k(x) ⊆ {0, 1}. Representasi: dua matriks bool ``has1`` (1 ∈ C_k) dan
``has0`` (0 ∈ C_k), shape (N, K).

Kuantil konformal (Angelopoulos & Bates, arXiv:2107.07511): untuk skor kalibrasi s_1..s_n,
    q̂ = s_(r),   r = ⌈(n+1)(1−α)⌉,   q̂ = +∞ jika r > n (termasuk n = 0).
Di bawah exchangeability, P(s_test ≤ q̂) ≥ 1−α, dan ≤ 1−α + 1/(n+1) jika skor tanpa ties.

Metode:
- M0: ambang tetap 0,5 (tanpa jaminan): has1 = p ≥ 0,5; has0 = ¬has1.
- M1: satu q̂ gabungan dari skor 1−p semua pasangan (i,k) positif: has1 = 1−p ≤ q̂.
- M2: Mondrian per kelas, positif saja: has1_k = 1−p_k ≤ q̂_k; menjamin
      P(1 ∈ C_k | y_k = 1) ≥ 1−α (sensitivitas per kelas). has0 = ¬has1.
- M3: Mondrian per kelas dan per nilai label: skor s_k(x,1) = 1−p_k,
      s_k(x,0) = p_k; has1 = 1−p ≤ q̂_{k,1}, has0 = p ≤ q̂_{k,0}. C_k = {0,1} berarti
      "ragu, rujuk"; C_k = ∅ mungkin terjadi.
Untuk M0–M2, C_k selalu singleton, jadi coverage pada y=0 = spesifisitas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

METHODS: tuple[str, ...] = ("M0", "M1", "M2", "M3")


def conformal_rank(n: int, alpha: float) -> int:
    """r = ⌈(n+1)(1−α)⌉ (1-indeks). Toleransi 1e-9 melawan galat floating point."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha harus di (0,1): {alpha}")
    return int(np.ceil((n + 1) * (1.0 - alpha) - 1e-9))


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Kuantil konformal: skor terkecil ke-r, atau +∞ jika r > n."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    n = s.size
    r = conformal_rank(n, alpha)
    if r > n:
        return float("inf")
    r = max(r, 1)
    return float(np.partition(s, r - 1)[r - 1])


@dataclass(frozen=True)
class Calibration:
    """Ambang hasil kalibrasi. ``q1``/``q0`` shape (K,); NaN jika tidak dipakai metode."""

    method: str
    alpha: float
    q1: np.ndarray
    q0: np.ndarray
    n1: np.ndarray  # jumlah skor kalibrasi positif per kelas (M1: total di semua kelas)
    n0: np.ndarray


def calibrate(method: str, probs: np.ndarray, y: np.ndarray, alpha: float) -> Calibration:
    """Hitung ambang conformal dari data kalibrasi (VAL_B)."""
    p = np.asarray(probs, dtype=np.float64)
    yb = np.asarray(y).astype(bool)
    K = p.shape[1]
    nan = np.full(K, np.nan)
    n1 = yb.sum(axis=0).astype(np.int64)
    n0 = (~yb).sum(axis=0).astype(np.int64)
    if method == "M0":
        return Calibration(method, alpha, nan, nan, n1, n0)
    if method == "M1":
        q = conformal_quantile(1.0 - p[yb], alpha)
        tot = np.full(K, int(yb.sum()), dtype=np.int64)
        return Calibration(method, alpha, np.full(K, q), nan, tot, n0)
    if method in ("M2", "M3"):
        q1 = np.array([conformal_quantile(1.0 - p[yb[:, k], k], alpha) for k in range(K)])
        if method == "M2":
            return Calibration(method, alpha, q1, nan, n1, n0)
        q0 = np.array([conformal_quantile(p[~yb[:, k], k], alpha) for k in range(K)])
        return Calibration(method, alpha, q1, q0, n1, n0)
    raise ValueError(f"Metode tidak dikenal: {method}")


def predict_sets(cal: Calibration, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kembalikan (has1, has0), masing-masing bool (N, K)."""
    p = np.asarray(probs, dtype=np.float64)
    if cal.method == "M0":
        has1 = p >= 0.5
        return has1, ~has1
    has1 = (1.0 - p) <= cal.q1[None, :]
    if cal.method in ("M1", "M2"):
        return has1, ~has1
    has0 = p <= cal.q0[None, :]
    return has1, has0


def set_metrics(has1: np.ndarray, has0: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """Metrik per kelas (array (K,)) + agregat per rekaman.

    cov1 = P(1 ∈ C_k | y_k=1), cov0 = P(0 ∈ C_k | y_k=0), abstain = P(C_k = {0,1}),
    empty = P(C_k = ∅), decisive_acc = akurasi pada C_k singleton,
    labels_per_record = rerata |{k : 1 ∈ C_k}|, all_pos_covered = P(semua label positif ∈ C).
    """
    yb = np.asarray(y).astype(bool)
    covered = np.where(yb, has1, has0)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov1 = (covered & yb).sum(0) / yb.sum(0)
        cov0 = (covered & ~yb).sum(0) / (~yb).sum(0)
        single = has1 ^ has0
        decisive_acc = (single & covered).sum(0) / single.sum(0)
    return {
        "cov1": cov1,
        "cov0": cov0,
        "abstain": (has1 & has0).mean(0),
        "empty": (~has1 & ~has0).mean(0),
        "decisive_acc": decisive_acc,
        "labels_per_record": np.array(has1.sum(1).mean()),
        "all_pos_covered": np.array(np.all(has1 | ~yb, axis=1).mean()),
    }
