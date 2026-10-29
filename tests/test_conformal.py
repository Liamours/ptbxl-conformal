import numpy as np
import pytest

from ecgdig.models.conformal import (
    calibrate,
    conformal_quantile,
    conformal_rank,
    predict_sets,
    set_metrics,
)


def test_worked_example_from_notebook() -> None:
    # n = 9, α = 0.2 → r = ⌈10 · 0.8⌉ = 8 → elemen terkecil ke-8 = 0.8
    s = np.array([0.9, 0.2, 0.5, 0.1, 0.8, 0.4, 0.7, 0.3, 0.6])
    assert conformal_rank(9, 0.2) == 8
    assert conformal_quantile(s, 0.2) == pytest.approx(0.8)


def test_rank_exceeds_n_gives_inf_and_empty_gives_inf() -> None:
    assert conformal_quantile(np.array([0.1, 0.2]), 0.01) == np.inf
    assert conformal_quantile(np.array([]), 0.1) == np.inf


def test_rank_is_robust_to_float_error() -> None:
    # (n+1)(1−α) = 9.0 persis secara matematis, tetapi 10 * 0.9 di float bisa 9.000000000000002
    assert conformal_rank(9, 0.1) == 9


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_marginal_coverage_uniform_scores(alpha: float) -> None:
    rng = np.random.default_rng(0)
    n, m, reps = 200, 200, 3000
    cov = [
        np.mean(rng.uniform(size=m) <= conformal_quantile(rng.uniform(size=n), alpha))
        for _ in range(reps)
    ]
    lo, hi = 1 - alpha, 1 - alpha + 1 / (n + 1)
    assert lo - 0.005 <= np.mean(cov) <= hi + 0.005


def _synthetic(n: int, K: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    y = rng.uniform(size=(n, K)) < 0.3
    logits = rng.normal(size=(n, K)) + 2.0 * (y - 0.5)
    return 1 / (1 + np.exp(-logits)), y.astype(np.int8)


def test_m2_class_conditional_coverage_on_positives() -> None:
    rng = np.random.default_rng(1)
    alpha, covs = 0.1, []
    for _ in range(200):
        pc, yc = _synthetic(400, 3, rng)
        pt, yt = _synthetic(400, 3, rng)
        has1, has0 = predict_sets(calibrate("M2", pc, yc, alpha), pt)
        assert np.array_equal(has0, ~has1)
        covs.append(set_metrics(has1, has0, yt)["cov1"])
    assert np.all(np.mean(covs, axis=0) >= 1 - alpha - 0.01)


def test_m3_covers_both_label_values_and_can_abstain() -> None:
    rng = np.random.default_rng(2)
    alpha, c1, c0, ab = 0.1, [], [], []
    for _ in range(200):
        pc, yc = _synthetic(400, 2, rng)
        pt, yt = _synthetic(400, 2, rng)
        has1, has0 = predict_sets(calibrate("M3", pc, yc, alpha), pt)
        met = set_metrics(has1, has0, yt)
        c1.append(met["cov1"])
        c0.append(met["cov0"])
        ab.append(met["abstain"])
    assert np.all(np.mean(c1, 0) >= 1 - alpha - 0.01)
    assert np.all(np.mean(c0, 0) >= 1 - alpha - 0.01)
    assert np.all(np.mean(ab, 0) > 0)


def test_m0_and_m1_shapes_and_m1_pooled_threshold() -> None:
    p = np.array([[0.9, 0.2], [0.4, 0.7], [0.6, 0.1]])
    y = np.array([[1, 0], [0, 1], [1, 0]])
    h1, h0 = predict_sets(calibrate("M0", p, y, 0.1), p)
    assert h1.tolist() == [[True, False], [False, True], [True, False]]
    c = calibrate("M1", p, y, 0.5)  # skor positif 1−p: [0.1, 0.4, 0.3]; r = ⌈4·0.5⌉ = 2 → 0.3
    assert c.q1 == pytest.approx([0.3, 0.3])


def test_set_metrics_hand_example() -> None:
    has1 = np.array([[True], [True], [False], [False]])
    has0 = np.array([[False], [True], [True], [False]])
    y = np.array([[1], [0], [0], [1]])
    met = set_metrics(has1, has0, y)
    assert met["cov1"][0] == pytest.approx(0.5)
    assert met["cov0"][0] == pytest.approx(1.0)
    assert met["abstain"][0] == pytest.approx(0.25)
    assert met["empty"][0] == pytest.approx(0.25)
    assert met["decisive_acc"][0] == pytest.approx(1.0)
