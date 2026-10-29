import numpy as np
import pytest
from scipy import stats

from ecgdig.eval.bootstrap import patient_bootstrap_weights, percentile_ci, weighted_rate
from ecgdig.eval.validity import (
    compare_with_theory,
    pac_rank,
    simulate_split_coverage,
    standard_rank,
    theory_distribution,
)


def test_bootstrap_weights_are_shared_within_patient() -> None:
    pid = np.array([5, 5, 7, 9, 9, 9])
    W = patient_bootstrap_weights(pid, 200, np.random.default_rng(0))
    assert W.shape == (200, 6)
    assert np.all(W[:, 0] == W[:, 1]) and np.all(W[:, 3] == W[:, 5])
    # setiap replika menarik tepat P = 3 pasien
    assert np.all(W[:, [0, 2, 3]].sum(axis=1) == 3)


def test_weighted_rate_with_unit_weights_equals_plain_rate() -> None:
    hit = np.array([1, 0, 1, 1])
    mask = np.array([1, 1, 0, 1])
    r = weighted_rate(hit, mask, np.ones((1, 4)))
    assert r[0] == pytest.approx(2 / 3)
    assert percentile_ci(np.arange(101.0)) == pytest.approx((2.5, 97.5))


def test_theory_mean_is_k_over_n_plus_one() -> None:
    n, m, alpha = 100, 80, 0.1
    k = standard_rank(n, alpha)
    dist = theory_distribution(n, m, k)
    assert dist is not None
    assert dist.mean() / m == pytest.approx(k / (n + 1))
    assert theory_distribution(5, 10, 6) is None


def test_simulation_matches_beta_binomial() -> None:
    rng = np.random.default_rng(3)
    scores = rng.uniform(size=150)  # tanpa ties
    n = 75
    sim = simulate_split_coverage(scores, n, standard_rank(n, 0.1), 2000, rng)
    res = compare_with_theory(sim, 0.1, np.random.default_rng(4), n_mc=200)
    assert res["emp_mean"] == pytest.approx(res["theory_mean"], abs=0.004)
    assert res["emp_sd"] == pytest.approx(res["theory_sd"], rel=0.1)
    assert res["ks_p"] > 0.01


def test_pac_rule_bounds_conditional_miscoverage() -> None:
    rng = np.random.default_rng(5)
    n, alpha, delta = 120, 0.1, 0.1
    sim = simulate_split_coverage(rng.uniform(size=240), n, pac_rank(n, alpha, delta), 200, rng)
    res = compare_with_theory(sim, alpha, np.random.default_rng(6), n_mc=20)
    assert res["theory_cond_p_below"] <= delta


def test_pac_rank_meets_delta_and_is_minimal() -> None:
    n, alpha, delta = 200, 0.1, 0.1
    k = pac_rank(n, alpha, delta)
    assert stats.beta.sf(1 - alpha, k, n + 1 - k) >= 1 - delta
    assert stats.beta.sf(1 - alpha, k - 1, n + 2 - k) < 1 - delta
    assert k >= standard_rank(n, alpha)
    assert pac_rank(5, 0.01, 0.1) == 6  # tidak ada rank yang cukup → +∞
