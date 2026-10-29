"""Metrik klasifikasi multi-label: AUROC/AUPRC per kelas dan makro, Brier, triage.

- AUROC_k: ``sklearn.metrics.roc_auc_score`` per kelas; makro = rerata tak berbobot.
- AUPRC_k: average precision (``average_precision_score``).
- Brier = (1/(nK)) Σ (p_ik − y_ik)².
- Triage: skor s = 1 − p_NORM, label abnormal a_i = 1 jika ada superclass selain NORM.
  Ambang t dipilih di VAL_A sebagai kuantil spesifisitas target dari skor negatif;
  sensitivitas/spesifisitas lalu diukur di set lain dengan t yang sama (positif jika s > t).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def per_class_auroc(
    y: np.ndarray, p: np.ndarray, sample_weight: np.ndarray | None = None
) -> np.ndarray:
    """AUROC per kelas, shape (K,). NaN jika kelas hanya punya satu nilai label."""
    out = np.full(p.shape[1], np.nan)
    for k in range(p.shape[1]):
        yk = y[:, k]
        w = sample_weight
        pos = (yk == 1) if w is None else ((yk == 1) & (w > 0))
        neg = (yk == 0) if w is None else ((yk == 0) & (w > 0))
        if pos.any() and neg.any():
            out[k] = roc_auc_score(yk, p[:, k], sample_weight=w)
    return out


def per_class_auprc(
    y: np.ndarray, p: np.ndarray, sample_weight: np.ndarray | None = None
) -> np.ndarray:
    """Average precision per kelas, shape (K,)."""
    out = np.full(p.shape[1], np.nan)
    for k in range(p.shape[1]):
        if (y[:, k] == 1).any():
            out[k] = average_precision_score(y[:, k], p[:, k], sample_weight=sample_weight)
    return out


def brier(y: np.ndarray, p: np.ndarray) -> float:
    """Skor Brier rerata atas semua (i, k)."""
    return float(np.mean((np.asarray(p, np.float64) - np.asarray(y, np.float64)) ** 2))


def triage_labels(y: np.ndarray, classes: Sequence[str]) -> tuple[int, np.ndarray]:
    """Kembalikan (indeks kolom NORM, label abnormal bool (N,))."""
    k_norm = list(classes).index("NORM")
    others = [k for k in range(len(classes)) if k != k_norm]
    return k_norm, np.asarray(y)[:, others].any(axis=1)


def threshold_at_specificity(scores_neg: np.ndarray, specificity: float) -> float:
    """Ambang t = kuantil-``specificity`` skor negatif (metode 'higher'); positif jika s > t."""
    return float(np.quantile(np.asarray(scores_neg, np.float64), specificity, method="higher"))


def sens_spec(scores: np.ndarray, labels: np.ndarray, t: float) -> tuple[float, float]:
    """(sensitivitas, spesifisitas) untuk aturan positif jika s > t."""
    pred = np.asarray(scores) > t
    lab = np.asarray(labels).astype(bool)
    return float(pred[lab].mean()), float((~pred[~lab]).mean())
