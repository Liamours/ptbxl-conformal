"""Fitur EKG klasik ber-rumus eksplisit untuk model pembanding (E7, ``features_logreg``).

Masukan x ∈ ℝ^{12×T} (mV, f_s = 100 Hz, tanpa standarisasi). Per lead ℓ:

1. Statistik (7): rerata μ, simpangan baku σ, skewness E[(x−μ)³]/σ³, kurtosis E[(x−μ)⁴]/σ⁴,
   min, maks, puncak-ke-puncak (maks − min; terkait kriteria voltase HYP).
2. Hjorth (2): mobilitas M(x) = sqrt(Var(x′)/Var(x)), kompleksitas C(x) = M(x′)/M(x),
   dengan x′ = beda maju (turunan diskret).
3. Spektrum Welch (7): PSD S(f) (segmen 256, Hann, overlap 50%). Daya relatif pita
   [0,5;3) [3;8) [8;15) [15;30) [30;50] Hz = ∫_pita S / ∫_{0,5}^{50} S, entropi spektral
   ternormalisasi H = −Σ s_j log s_j / log J (s_j = S_j/ΣS pada 0,5–50 Hz), dan log daya total.
Global (2), dari rerata autokorelasi ternormalisasi antar-lead
   ρ(τ) = Σ_n (x_n − μ)(x_{n+τ} − μ) / Σ_n (x_n − μ)²  (dihitung via FFT, Wiener–Khinchin):
   τ* = argmax_{0,3 s ≤ τ ≤ 2,0 s} ρ̄(τ) → HR = 60/τ* bpm, dan keteraturan ritme ρ̄(τ*).
Total 12 × 16 + 2 = 194 fitur.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis, skew

BANDS_HZ: tuple[tuple[float, float], ...] = ((0.5, 3), (3, 8), (8, 15), (15, 30), (30, 50.01))
PER_LEAD = (
    "mean",
    "std",
    "skew",
    "kurt",
    "min",
    "max",
    "ptp",
    "hj_mob",
    "hj_comp",
    "bp_0.5-3",
    "bp_3-8",
    "bp_8-15",
    "bp_15-30",
    "bp_30-50",
    "spec_entropy",
    "log_power",
)


def feature_names(leads: tuple[str, ...]) -> list[str]:
    """Nama 194 fitur dalam urutan kolom."""
    return [f"{lead}_{f}" for lead in leads for f in PER_LEAD] + ["hr_bpm", "rhythm_reg"]


def hjorth(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(mobilitas, kompleksitas) sepanjang sumbu terakhir."""
    d1 = np.diff(x, axis=-1)
    d2 = np.diff(d1, axis=-1)
    v0, v1, v2 = x.var(axis=-1), d1.var(axis=-1), d2.var(axis=-1)
    mob = np.sqrt(v1 / np.maximum(v0, 1e-12))
    mob_d = np.sqrt(v2 / np.maximum(v1, 1e-12))
    return mob, mob_d / np.maximum(mob, 1e-12)


def band_features(x: np.ndarray, fs_hz: float) -> np.ndarray:
    """(…, 7): 5 daya pita relatif, entropi spektral ternormalisasi, log daya total."""
    f, S = welch(x, fs=fs_hz, nperseg=min(256, x.shape[-1]), axis=-1)
    keep = (f >= 0.5) & (f <= fs_hz / 2)
    Sk = S[..., keep]
    total = np.maximum(Sk.sum(axis=-1), 1e-20)
    fk = f[keep]
    rel = [S[..., (f >= lo) & (f < hi)].sum(axis=-1) / total for lo, hi in BANDS_HZ]
    s = Sk / total[..., None]
    ent = -(s * np.log(np.maximum(s, 1e-20))).sum(axis=-1) / np.log(fk.size)
    return np.stack([*rel, ent, np.log(total)], axis=-1)


def autocorr_rhythm(x: np.ndarray, fs_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """(HR bpm, ρ̄(τ*)) dari rerata autokorelasi ternormalisasi antar-lead; x (N, L, T)."""
    xc = x - x.mean(axis=-1, keepdims=True)
    T = x.shape[-1]
    F = np.fft.rfft(xc, n=2 * T, axis=-1)
    ac = np.fft.irfft(F * np.conj(F), axis=-1)[..., :T]
    ac = ac / np.maximum(ac[..., :1], 1e-12)
    rho = ac.mean(axis=-2)  # rerata antar-lead
    lo, hi = int(round(0.3 * fs_hz)), int(round(2.0 * fs_hz))
    tau = lo + np.argmax(rho[..., lo : hi + 1], axis=-1)
    return 60.0 * fs_hz / tau, np.take_along_axis(rho, tau[..., None], axis=-1)[..., 0]


def extract_features(x_mv: np.ndarray, fs_hz: float = 100.0, chunk: int = 2048) -> np.ndarray:
    """Matriks fitur float32 (N, 194) dari sinyal (N, 12, T) mV."""
    out = []
    for i in range(0, x_mv.shape[0], chunk):
        x = x_mv[i : i + chunk].astype(np.float64)
        stats = np.stack(
            [
                x.mean(-1),
                x.std(-1),
                skew(x, axis=-1),
                kurtosis(x, axis=-1, fisher=False),
                x.min(-1),
                x.max(-1),
                np.ptp(x, axis=-1),
            ],
            axis=-1,
        )
        mob, comp = hjorth(x)
        per_lead = np.concatenate(
            [stats, mob[..., None], comp[..., None], band_features(x, fs_hz)], axis=-1
        )
        hr, reg = autocorr_rhythm(x, fs_hz)
        feats = np.c_[per_lead.reshape(len(x), -1), hr, reg]
        out.append(np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))
    return np.concatenate(out)
