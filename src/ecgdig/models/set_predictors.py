"""Prediktor himpunan label pembanding untuk E6 (melengkapi M0–M3 di ``conformal.py``).

Semua mengembalikan ``has1`` bool (N, K): label k ditandai untuk rekaman i.

- ``f1``    : ambang per kelas t_k = argmax_t F1_k(t) di VAL_A (praktik umum, TANPA jaminan).
- ``m2pac`` : Mondrian per kelas dengan rank PAC k_δ (lihat ``eval/validity.py``):
              P_kalibrasi(P(1 ∈ C_k | y_k = 1) ≥ 1 − α) ≥ 1 − δ.
- ``crc``   : Conformal Risk Control (Angelopoulos dkk., arXiv 2208.02814, Thm 1) untuk
              loss FNR pasien  L_i(λ) = 1 − |Y_i ∩ C_λ(x_i)| / |Y_i|,  C_λ(x) = {k : p_k ≥ 1 − λ}.
              L_i tak-naik dan kontinu-kanan dalam λ, 0 ≤ L_i ≤ B = 1, L_i(1) = 0. Dengan
              λ̂ = inf{λ : n/(n+1) · R̂_n(λ) + B/(n+1) ≤ α},  R̂_n = rerata L_i di kalibrasi,
              dijamin E[L_{n+1}(λ̂)] ≤ α: rata-rata fraksi label benar yang terlewat ≤ α.
              Satu ambang untuk semua kelas; tidak menjamin sensitivitas per kelas.
"""

from __future__ import annotations

import numpy as np

from ecgdig.eval.validity import pac_rank


def fnr_loss(has1: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fraksi label positif yang tidak ditandai, per rekaman (N,). Rekaman tanpa label → 0."""
    yb = np.asarray(y).astype(bool)
    n_pos = yb.sum(axis=1)
    missed = (yb & ~has1).sum(axis=1)
    return np.where(n_pos > 0, missed / np.maximum(n_pos, 1), 0.0)


def crc_lambda(probs: np.ndarray, y: np.ndarray, alpha: float, bound: float = 1.0) -> float:
    """λ̂ CRC untuk loss FNR. Kandidat λ = 0 dan semua titik loncat 1 − p_ik (y_ik = 1)."""
    p = np.asarray(probs, np.float64)
    yb = np.asarray(y).astype(bool)
    n = p.shape[0]
    cands = np.unique(np.r_[0.0, 1.0 - p[yb], 1.0])
    for lam in cands:  # R̂ tak-naik dalam λ: kandidat pertama yang memenuhi = infimum
        risk = fnr_loss(p >= 1.0 - lam, yb).mean()
        if n / (n + 1) * risk + bound / (n + 1) <= alpha:
            return float(lam)
    return 1.0  # λ_max: semua label ditandai


def crc_sets(probs: np.ndarray, lam: float) -> np.ndarray:
    """C_λ(x) = {k : p_k ≥ 1 − λ}."""
    return np.asarray(probs, np.float64) >= 1.0 - lam


def f1_thresholds(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """t_k yang memaksimalkan F1 kelas k (positif jika p ≥ t); dicari di semua nilai p unik."""
    p = np.asarray(probs, np.float64)
    yb = np.asarray(y).astype(bool)
    out = np.empty(p.shape[1])
    for k in range(p.shape[1]):
        order = np.argsort(-p[:, k], kind="mergesort")
        ps, ys = p[order, k], yb[order, k]
        tp = np.cumsum(ys)
        fp = np.cumsum(~ys)
        f1 = 2 * tp / (tp + fp + ys.sum())
        last = np.r_[ps[1:] != ps[:-1], True]  # hanya posisi akhir tiap nilai p (ties)
        best = int(np.argmax(np.where(last, f1, -1.0)))
        out[k] = ps[best]
    return out


def pac_thresholds(probs: np.ndarray, y: np.ndarray, alpha: float, delta: float) -> np.ndarray:
    """Ambang skor q_k (skor 1 − p) per kelas dengan rank PAC; +∞ jika n tidak cukup."""
    p = np.asarray(probs, np.float64)
    yb = np.asarray(y).astype(bool)
    q = np.empty(p.shape[1])
    for k in range(p.shape[1]):
        s = np.sort(1.0 - p[yb[:, k], k])
        r = pac_rank(s.size, alpha, delta)
        q[k] = np.inf if r > s.size else s[r - 1]
    return q
