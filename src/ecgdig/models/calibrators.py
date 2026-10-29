"""Kalibrator post-hoc per kelas untuk logit multi-label (E5). Semua di-fit di VAL_A.

Notasi: z = logit satu kelas, y ∈ {0,1}, p = probabilitas terkalibrasi.

- ``none``      : p = σ(z).
- ``temp``      : p = σ(z/T), T bersama semua kelas (lihat ``temperature.py``).
- ``temp_k``    : p = σ(z/T_k) per kelas.
- ``platt``     : p = σ(a z + b) per kelas (vector scaling). (a, b) = argmin NLL, masalah
                  konveks; diselesaikan dengan Newton–Raphson:
                  θ ← θ − H⁻¹∇,  ∇ = Xᵀ(p − y),  H = Xᵀ diag(p(1−p)) X + εI,  X = [z, 1].
- ``isotonic``  : p = f(z), f tak-turun, f = argmin Σ (y_i − f(z_i))² (proyeksi L2 ke kerucut
                  monoton), dihitung dengan Pool-Adjacent-Violators (PAV). Prediksi: fungsi
                  tangga kontinu-kanan pada titik z latih (blok PAV).
- ``hist``      : histogram binning massa-sama (B bin dari kuantil z latih); p = rerata y bin.

Probabilitas di-clip ke [1e-6, 1 − 1e-6] agar NLL terdefinisi.
Invariansi conformal: temp, temp_k, platt (a > 0), dan isotonic adalah transformasi tak-turun
per kelas, sehingga urutan skor per kelas, dan karenanya himpunan M2/M3, tidak berubah (kecuali
efek ties dari isotonic). ``hist`` TIDAK dijamin monoton: frekuensi bin bertetangga bisa turun
karena derau sampel, sehingga M2/M3 bisa berubah. M1 (ambang gabungan antar-kelas) tidak invarian
terhadap transformasi per kelas apa pun kecuali yang sama untuk semua kelas (mis. ``temp``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ecgdig.models.temperature import fit_temperature, fit_temperature_per_class, sigmoid

EPS = 1e-6
METHODS: tuple[str, ...] = ("none", "temp", "temp_k", "platt", "isotonic", "hist")


def pav(y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Regresi isotonik (tak-turun) untuk barisan y yang sudah terurut menurut z.

    Pool-Adjacent-Violators: gabungkan blok bertetangga selama rerata berbobotnya menurun.
    Hasil = argmin_{f_1 ≤ … ≤ f_n} Σ w_i (y_i − f_i)². Kompleksitas O(n).
    """
    y = np.asarray(y, dtype=np.float64)
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=np.float64)
    means: list[float] = []
    weights: list[float] = []
    sizes: list[int] = []
    for yi, wi in zip(y, w, strict=True):
        means.append(float(yi))
        weights.append(float(wi))
        sizes.append(1)
        while len(means) > 1 and means[-2] > means[-1]:
            m2, w2, s2 = means.pop(), weights.pop(), sizes.pop()
            m1, w1, s1 = means.pop(), weights.pop(), sizes.pop()
            wt = w1 + w2
            means.append((w1 * m1 + w2 * m2) / wt)
            weights.append(wt)
            sizes.append(s1 + s2)
    return np.repeat(means, sizes)


def fit_platt(z: np.ndarray, y: np.ndarray, iters: int = 50, ridge: float = 1e-8) -> np.ndarray:
    """(a, b) MLE untuk p = σ(a z + b) dengan Newton–Raphson (konveks; konvergen kuadratik)."""
    X = np.c_[np.asarray(z, np.float64), np.ones(len(z))]
    yy = np.asarray(y, np.float64)
    theta = np.array([1.0, 0.0])
    for _ in range(iters):
        p = sigmoid(X @ theta)
        grad = X.T @ (p - yy)
        H = (X * (p * (1 - p))[:, None]).T @ X + ridge * np.eye(2)
        step = np.linalg.solve(H, grad)
        theta -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    return theta


@dataclass
class Calibrator:
    """Kalibrator per kelas; ``fit`` pada (logit, label) VAL_A, ``transform`` → probabilitas."""

    method: str
    params: dict[str, Any] = field(default_factory=dict)

    def fit(self, z: np.ndarray, y: np.ndarray, n_bins: int = 15) -> Calibrator:
        z, y = np.asarray(z, np.float64), np.asarray(y, np.float64)
        K = z.shape[1]
        m = self.method
        if m == "temp":
            self.params["T"] = fit_temperature(z, y)
        elif m == "temp_k":
            self.params["T"] = fit_temperature_per_class(z, y)
        elif m == "platt":
            self.params["ab"] = np.array([fit_platt(z[:, k], y[:, k]) for k in range(K)])
        elif m == "isotonic":
            steps = []
            for k in range(K):
                order = np.argsort(z[:, k], kind="mergesort")
                steps.append((z[order, k], pav(y[order, k])))
            self.params["steps"] = steps
        elif m == "hist":
            bins = []
            for k in range(K):
                edges = np.quantile(z[:, k], np.linspace(0, 1, n_bins + 1)[1:-1])
                idx = np.searchsorted(edges, z[:, k], side="right")
                cnt = np.bincount(idx, minlength=n_bins)
                rate = np.bincount(idx, weights=y[:, k], minlength=n_bins) / np.maximum(cnt, 1)
                bins.append((edges, rate))
            self.params["bins"] = bins
        elif m != "none":
            raise ValueError(f"Kalibrator tidak dikenal: {m}")
        return self

    def transform(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, np.float64)
        m = self.method
        if m == "none":
            p = sigmoid(z)
        elif m in ("temp", "temp_k"):
            p = sigmoid(z / np.asarray(self.params["T"]))
        elif m == "platt":
            ab = np.asarray(self.params["ab"])
            p = sigmoid(z * ab[:, 0] + ab[:, 1])
        elif m == "isotonic":
            p = np.empty_like(z)
            for k, (zs, fs) in enumerate(self.params["steps"]):
                i = np.searchsorted(zs, z[:, k], side="right") - 1
                p[:, k] = np.where(i < 0, fs[0], fs[np.clip(i, 0, len(fs) - 1)])
        elif m == "hist":
            p = np.empty_like(z)
            for k, (edges, rate) in enumerate(self.params["bins"]):
                p[:, k] = rate[np.searchsorted(edges, z[:, k], side="right")]
        else:
            raise ValueError(f"Kalibrator tidak dikenal: {m}")
        return np.clip(p, EPS, 1 - EPS)
