"""Validitas sampel-hingga split conformal (eksperimen E3 / "T2").

Setup: N skor nonkonformitas berbeda (positif satu kelas), dipecah acak menjadi kalibrasi
n dan uji m = N − n. Ambang q̂ = skor kalibrasi terkecil ke-k, k = ⌈(n+1)(1−α)⌉.

Teori: urutan rank skor dalam pool hasil permutasi acak berdistribusi seragam, sama seperti
skor i.i.d. kontinu. Maka coverage bersyarat kalibrasi F(q̂) ~ Beta(k, n+1−k), dan banyaknya
skor uji ≤ q̂ berdistribusi Beta-Binomial(m, k, n+1−k) secara eksak (tanpa ties), dengan
E[coverage] = k/(n+1) ∈ [1−α, 1−α + 1/(n+1)].

Varian PAC (training-conditional): k_δ = min{k' ≤ n : P(Beta(k', n+1−k') ≥ 1−α) ≥ 1−δ},
sehingga P_kalibrasi(coverage ≥ 1−α) ≥ 1−δ. k_δ > n berarti q̂ = +∞.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from ecgdig.models.conformal import conformal_rank


@dataclass(frozen=True)
class SplitCoverage:
    """Hasil simulasi split acak untuk satu (kelas, α, aturan rank)."""

    n_cal: int
    n_test: int
    rank: int
    counts: np.ndarray  # (R,) banyaknya skor uji ≤ q̂


def pac_rank(n: int, alpha: float, delta: float) -> int:
    """k_δ terkecil sehingga P(Beta(k, n+1−k) ≥ 1−α) ≥ 1−δ; n+1 jika tidak ada (q̂ = +∞)."""
    for k in range(1, n + 1):
        if stats.beta.sf(1.0 - alpha, k, n + 1 - k) >= 1.0 - delta:
            return k
    return n + 1


def simulate_split_coverage(
    scores: np.ndarray, n_cal: int, rank: int, n_splits: int, rng: np.random.Generator
) -> SplitCoverage:
    """Ulangi split acak kalibrasi/uji; ambang = skor kalibrasi terkecil ke-``rank``."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    n_test = s.size - n_cal
    if n_cal < 1 or n_test < 1:
        raise ValueError("n_cal dan n_test harus ≥ 1")
    counts = np.empty(n_splits, dtype=np.int64)
    for r in range(n_splits):
        perm = rng.permutation(s.size)
        cal, test = s[perm[:n_cal]], s[perm[n_cal:]]
        q = np.inf if rank > n_cal else np.partition(cal, rank - 1)[rank - 1]
        counts[r] = int(np.count_nonzero(test <= q))
    return SplitCoverage(n_cal, n_test, rank, counts)


def theory_distribution(n_cal: int, n_test: int, rank: int) -> stats.rv_discrete | None:
    """Beta-Binomial(m, k, n+1−k) untuk banyaknya skor uji ≤ q̂; None jika q̂ = +∞."""
    if rank > n_cal:
        return None
    return stats.betabinom(n_test, rank, n_cal + 1 - rank)


def ks_discrete(counts: np.ndarray, dist: stats.rv_discrete, m: int) -> float:
    """sup_x |F_emp(x) − F(x)| pada dukungan {0..m}."""
    support = np.arange(m + 1)
    emp = np.searchsorted(np.sort(counts), support, side="right") / counts.size
    return float(np.max(np.abs(emp - dist.cdf(support))))


def compare_with_theory(
    sim: SplitCoverage, alpha: float, rng: np.random.Generator, n_mc: int = 500
) -> dict[str, float]:
    """Bandingkan distribusi coverage empiris dengan teori Beta-Binomial.

    p-value KS dihitung Monte Carlo: n_mc himpunan R tarikan dari distribusi teori.
    """
    m, cov = sim.n_test, sim.counts / sim.n_test
    below = int(np.ceil(m * (1.0 - alpha) - 1e-9)) - 1  # coverage < 1−α ⇔ count ≤ below
    out: dict[str, float] = {
        "n_cal": sim.n_cal,
        "n_test": m,
        "rank": sim.rank,
        "emp_mean": float(cov.mean()),
        "emp_sd": float(cov.std(ddof=1)),
        "emp_p_below": float(np.mean(sim.counts <= below)),
        "emp_q05": float(np.quantile(cov, 0.05)),
    }
    dist = theory_distribution(sim.n_cal, m, sim.rank)
    if dist is None:
        out.update(theory_mean=1.0, theory_sd=0.0, theory_p_below=0.0, theory_cond_p_below=0.0)
        out.update(ks=np.nan, ks_p=np.nan)
        return out
    # P(F(q̂) < 1−α) untuk coverage bersyarat (yang dijamin PAC); berbeda dari theory_p_below
    # yang memuat derau binomial m titik uji.
    out["theory_cond_p_below"] = float(
        stats.beta.cdf(1.0 - alpha, sim.rank, sim.n_cal + 1 - sim.rank)
    )
    d_obs = ks_discrete(sim.counts, dist, m)
    R = sim.counts.size
    d_mc = np.array([ks_discrete(dist.rvs(R, random_state=rng), dist, m) for _ in range(n_mc)])
    out.update(
        theory_mean=float(dist.mean() / m),
        theory_sd=float(dist.std() / m),
        theory_p_below=float(dist.cdf(below)) if below >= 0 else 0.0,
        theory_q05=float(dist.ppf(0.05) / m),
        ks=d_obs,
        ks_p=float((1 + np.sum(d_mc >= d_obs)) / (1 + n_mc)),
    )
    return out


def standard_rank(n_cal: int, alpha: float) -> int:
    """k = ⌈(n+1)(1−α)⌉ (split conformal standar)."""
    return conformal_rank(n_cal, alpha)
