"""Statistik inferensial post-hoc untuk hasil conformal (tanpa melatih ulang, tanpa tuning).

Notasi: satu kelas, m positif uji, X = banyaknya positif uji yang tercakup (1 ∈ C_k).

- Clopper–Pearson 95% untuk coverage X/m (eksak, berbasis kuantil Beta):
      lo = B⁻¹(0,025; X, m−X+1) (0 jika X = 0),  hi = B⁻¹(0,975; X+1, m−X) (1 jika X = m).
- Uji binomial eksak satu sisi, H0: coverage ≥ 1−α vs H1: coverage < 1−α, ambang q̂ dianggap
  tetap:  p = P(Bin(m, 1−α) ≤ X).
- Uji marginal di bawah exchangeability (q̂ = skor kalibrasi terkecil ke-k dari n, skor
  kontinu): X ~ BetaBin(m, k, n+1−k) secara eksak, sehingga  p = P(BetaBin(m, k, n+1−k) ≤ X).
  E[X] = m·k/(n+1),  Var[X] = m·a·b·(a+b+m) / ((a+b)²(a+b+1)),  a = k, b = n+1−k.
- Uji kecocokan χ² untuk R hitungan X_1..X_R terhadap pmf teori: sel dukungan {0..m} digabung
  berurutan sampai frekuensi harapan ≥ e_min; χ² = Σ (O_j − E_j)²/E_j, df = J − 1 (tidak ada
  parameter yang diestimasi dari data).
- Holm: p_(i) diurutkan naik, p̃_(i) = max_{j≤i} min(1, (M − j + 1) p_(j)).
- Uji berpasangan untuk selisih d_i = a_i − b_i: Wilcoxon signed-rank (selisih nol dibuang,
  ``zero_method="wilcox"``), uji-t berpasangan, dan CI bootstrap pasien untuk rerata d.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


def clopper_pearson(x: int, m: int, level: float = 0.95) -> tuple[float, float]:
    """Interval eksak Clopper–Pearson untuk proporsi x/m."""
    if m <= 0:
        return float("nan"), float("nan")
    a = (1.0 - level) / 2.0
    lo = 0.0 if x == 0 else float(stats.beta.ppf(a, x, m - x + 1))
    hi = 1.0 if x == m else float(stats.beta.ppf(1.0 - a, x + 1, m - x))
    return lo, hi


def binom_p_lower(x: int, m: int, p0: float) -> float:
    """p-value satu sisi P(Bin(m, p0) ≤ x) (H1: proporsi < p0)."""
    return float(stats.binom.cdf(x, m, p0))


def betabinom_params(n_cal: int, rank: int) -> tuple[int, int] | None:
    """(a, b) = (k, n+1−k) untuk X ~ BetaBin(m, a, b); None jika q̂ = +∞ (k > n)."""
    if rank > n_cal:
        return None
    return rank, n_cal + 1 - rank


def betabinom_moments(m: int, n_cal: int, rank: int) -> tuple[float, float]:
    """(E[X], Var[X]) untuk banyaknya skor uji ≤ q̂; (m, 0) jika q̂ = +∞."""
    ab = betabinom_params(n_cal, rank)
    if ab is None:
        return float(m), 0.0
    a, b = ab
    mean = m * a / (a + b)
    var = m * a * b * (a + b + m) / ((a + b) ** 2 * (a + b + 1))
    return float(mean), float(var)


def betabinom_p_lower(x: int, m: int, n_cal: int, rank: int) -> float:
    """p-value marginal P(BetaBin(m, k, n+1−k) ≤ x); 0 jika q̂ = +∞ dan x < m."""
    ab = betabinom_params(n_cal, rank)
    if ab is None:
        return 1.0 if x >= m else 0.0
    return float(stats.betabinom.cdf(x, m, *ab))


def merge_bins(expected: np.ndarray, e_min: float = 5.0) -> list[tuple[int, int]]:
    """Gabungkan sel berurutan [lo, hi] sampai Σ harapan ≥ e_min; sisa ekor masuk sel terakhir."""
    bins: list[tuple[int, int]] = []
    start, acc = 0, 0.0
    for j, e in enumerate(expected):
        acc += float(e)
        if acc >= e_min:
            bins.append((start, j))
            start, acc = j + 1, 0.0
    if start < len(expected):
        if bins:
            lo, _ = bins.pop()
            bins.append((lo, len(expected) - 1))
        else:
            bins.append((0, len(expected) - 1))
    return bins


def chisq_gof(counts: np.ndarray, pmf: np.ndarray, e_min: float = 5.0) -> dict[str, Any]:
    """χ² kecocokan hitungan diskret ``counts`` (nilai 0..len(pmf)−1) terhadap ``pmf``."""
    c = np.asarray(counts, dtype=np.int64)
    p = np.asarray(pmf, dtype=np.float64)
    observed = np.bincount(c, minlength=p.size).astype(np.float64)
    expected = c.size * p / p.sum()
    bins = merge_bins(expected, e_min)
    obs = np.array([observed[lo : hi + 1].sum() for lo, hi in bins])
    exp = np.array([expected[lo : hi + 1].sum() for lo, hi in bins])
    df = len(bins) - 1
    chi2 = float(np.sum((obs - exp) ** 2 / exp)) if df > 0 else float("nan")
    pval = float(stats.chi2.sf(chi2, df)) if df > 0 else float("nan")
    return {
        "chi2": chi2,
        "df": df,
        "p": pval,
        "n_bins": len(bins),
        "bins": "|".join(f"{lo}" if lo == hi else f"{lo}-{hi}" for lo, hi in bins),
        "observed": "|".join(str(int(o)) for o in obs),
        "expected": "|".join(f"{e:.2f}" for e in exp),
    }


def holm(pvals: np.ndarray) -> np.ndarray:
    """p-value terkoreksi Holm (step-down); NaN dibiarkan NaN dan tidak dihitung dalam M."""
    p = np.asarray(pvals, dtype=np.float64)
    out = np.full_like(p, np.nan)
    ok = np.flatnonzero(~np.isnan(p))
    if ok.size == 0:
        return out
    order = ok[np.argsort(p[ok], kind="mergesort")]
    M = ok.size
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, min(1.0, (M - i) * p[idx]))
        out[idx] = running
    return out


def paired_tests(
    a: np.ndarray, b: np.ndarray, weights: np.ndarray, level: float = 0.95
) -> dict[str, float]:
    """Bandingkan a vs b per rekaman: rerata d = a − b, CI bootstrap pasien, Wilcoxon, uji-t.

    ``weights`` (B, n) = bobot bootstrap pasien (lihat ``eval/bootstrap.py``).
    """
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    boot = (w @ d) / w.sum(axis=1)
    lo, hi = np.quantile(boot, [(1 - level) / 2, 1 - (1 - level) / 2])
    out = {
        "n_records": int(d.size),
        "mean_diff": float(d.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n_higher": int(np.sum(d > 0)),
        "n_lower": int(np.sum(d < 0)),
        "n_equal": int(np.sum(d == 0)),
        "wilcoxon_stat": float("nan"),
        "wilcoxon_p": float("nan"),
        "t_stat": float("nan"),
        "t_p": float("nan"),
    }
    if np.any(d != 0):
        wres = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
        out["wilcoxon_stat"], out["wilcoxon_p"] = float(wres.statistic), float(wres.pvalue)
        if np.std(d) > 0:
            tres = stats.ttest_rel(a, b)
            out["t_stat"], out["t_p"] = float(tres.statistic), float(tres.pvalue)
    return out
