"""Test statistik post-hoc (``eval/inference.py``) dan orkestrasi ``posthoc.py``."""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ecgdig.eval.inference import (
    betabinom_moments,
    betabinom_p_lower,
    binom_p_lower,
    chisq_gof,
    clopper_pearson,
    holm,
    merge_bins,
    paired_tests,
)
from ecgdig.eval.validity import simulate_split_coverage
from ecgdig.models.conformal import conformal_rank
from ecgdig.posthoc import S1_METHODS, class_rows, method_sets, run_posthoc

SMOKE = Path(__file__).resolve().parents[1] / "results" / "smoke" / "smoke_c0_crop__local"


def test_clopper_pearson_closed_forms() -> None:
    lo, hi = clopper_pearson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(1 - 0.025 ** (1 / 10))
    lo, hi = clopper_pearson(10, 10)
    assert hi == 1.0 and lo == pytest.approx(0.025 ** (1 / 10))
    lo, hi = clopper_pearson(45, 50)
    assert lo < 0.9 < hi


def test_binom_p_lower_matches_sum() -> None:
    direct = sum(stats.binom.pmf(j, 40, 0.9) for j in range(34))
    assert binom_p_lower(33, 40, 0.9) == pytest.approx(direct)


def test_betabinom_moments_and_infinite_threshold() -> None:
    m, n, k = 37, 50, conformal_rank(50, 0.1)
    mean, var = betabinom_moments(m, n, k)
    ref = stats.betabinom(m, k, n + 1 - k)
    assert mean == pytest.approx(ref.mean()) and var == pytest.approx(ref.var())
    assert mean / m == pytest.approx(k / (n + 1))
    assert betabinom_moments(m, 5, 6) == (float(m), 0.0)  # k > n → q̂ = +∞
    assert betabinom_p_lower(m, m, 5, 6) == 1.0 and betabinom_p_lower(m - 1, m, 5, 6) == 0.0


def test_merge_bins_contiguous_and_large_enough() -> None:
    exp = 1000 * stats.betabinom(60, 50, 6).pmf(np.arange(61))
    bins = merge_bins(exp)
    assert bins[0][0] == 0 and bins[-1][1] == 60
    assert all(b[0] == a[1] + 1 for a, b in zip(bins[:-1], bins[1:], strict=True))
    assert all(exp[lo : hi + 1].sum() >= 5 for lo, hi in bins)


def test_chisq_accepts_true_law_on_conformal_splits() -> None:
    rng = np.random.default_rng(0)
    s = rng.random(200)  # skor kontinu exchangeable
    n, k = 100, conformal_rank(100, 0.1)
    sim = simulate_split_coverage(s, n, k, 1000, rng)
    res = chisq_gof(sim.counts, stats.betabinom(100, k, n + 1 - k).pmf(np.arange(101)))
    assert res["df"] == res["n_bins"] - 1 > 2
    assert res["p"] > 0.001


def test_chisq_rejects_wrong_law() -> None:
    rng = np.random.default_rng(1)
    x = rng.binomial(100, 0.8, size=1000)
    res = chisq_gof(x, stats.binom(100, 0.9).pmf(np.arange(101)))
    assert res["p"] < 1e-6


def test_holm_known_example_and_nan() -> None:
    adj = holm(np.array([0.01, 0.04, 0.03, np.nan]))
    assert np.allclose(adj[:3], [0.03, 0.06, 0.06]) and np.isnan(adj[3])


def test_paired_tests_degenerate_and_shift() -> None:
    a = np.array([1.0, 2, 3, 2, 1, 0, 2, 3])
    w = np.ones((20, a.size), dtype=np.int32)
    same = paired_tests(a, a, w)
    assert same["mean_diff"] == 0 and same["n_equal"] == a.size and np.isnan(same["wilcoxon_p"])
    up = paired_tests(a + 1, a, w)
    assert up["mean_diff"] == 1 and up["ci_lo"] == up["ci_hi"] == 1 and up["n_higher"] == a.size
    assert up["wilcoxon_p"] < 0.05 and np.isnan(up["t_p"])  # selisih konstan: uji-t tak terdefinisi


def test_method_sets_and_class_rows_synthetic() -> None:
    rng = np.random.default_rng(3)
    yB = rng.random((300, 2)) < 0.4
    pB = np.clip(yB * 0.4 + rng.random((300, 2)) * 0.6, 0, 1)
    yT = rng.random((200, 2)) < 0.4
    pT = np.clip(yT * 0.4 + rng.random((200, 2)) * 0.6, 0, 1)
    for method in S1_METHODS:
        h1, h0, rank, q1 = method_sets(method, pB, yB, pT, 0.1, 0.1)
        rows = class_rows(h1, h0, yT, yB.sum(0), rank, q1, 0.1, method, ["A", "B"])
        for r in rows:
            assert 0 <= r["coverage"] <= 1 and r["cp95_lo"] <= r["coverage"] <= r["cp95_hi"]
            if method != "M3":  # M0–M2, PAC: C_k selalu singleton
                assert r["singleton_rate"] == 1.0 and r["empty_rate"] == 0.0
    h2 = method_sets("M2", pB, yB, pT, 0.1, 0.1)[0]
    hp = method_sets("M2_PAC", pB, yB, pT, 0.1, 0.1)[0]
    assert np.all(hp >= h2)  # k_δ ≥ k → himpunan PAC tidak lebih kecil


@pytest.mark.skipif(not (SMOKE / "e4").exists(), reason="run smoke lokal tidak tersedia")
def test_run_posthoc_on_smoke(tmp_path: Path) -> None:
    run = tmp_path / SMOKE.name
    shutil.copytree(SMOKE, run)
    paths = run_posthoc(run, tmp_path / "out", tmp_path / "TEST_LOG.md", log=lambda _: None)
    assert all(p.exists() for p in paths.values())
    v = pd.read_csv(paths["s3_validity_counts"])
    assert v["matches_e3_validity_csv"].all()
    s2 = pd.read_csv(paths["s2_set_size_paired_by_seed"])
    ok = s2["wilcoxon_p"].notna()
    assert (s2.loc[ok, "wilcoxon_p_holm"] >= s2.loc[ok, "wilcoxon_p"] - 1e-12).all()
    assert "posthoc" in (tmp_path / "TEST_LOG.md").read_text(encoding="utf-8")
