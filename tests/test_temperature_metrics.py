import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from ecgdig.metrics.calibration import ece_binary, ece_per_class
from ecgdig.metrics.classification import (
    brier,
    per_class_auroc,
    sens_spec,
    threshold_at_specificity,
    triage_labels,
)
from ecgdig.models.temperature import (
    apply_temperature,
    bce_nll,
    fit_temperature,
    fit_temperature_per_class,
    sigmoid,
)


def test_sigmoid_stable_at_extremes() -> None:
    p = sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    assert p.tolist() == [0.0, 0.5, 1.0]


def test_fit_temperature_recovers_known_t() -> None:
    rng = np.random.default_rng(0)
    t0 = 2.5
    z = rng.normal(scale=4.0, size=(40000, 2))
    y = (rng.uniform(size=z.shape) < sigmoid(z / t0)).astype(np.int8)
    assert fit_temperature(z, y) == pytest.approx(t0, rel=0.05)
    assert fit_temperature_per_class(z, y) == pytest.approx([t0, t0], rel=0.08)


def test_temperature_keeps_auroc_and_lowers_nll() -> None:
    rng = np.random.default_rng(1)
    z = rng.normal(scale=4.0, size=(5000, 1))
    y = (rng.uniform(size=z.shape) < sigmoid(z / 3.0)).astype(np.int8)
    T = fit_temperature(z, y)
    p = apply_temperature(z, T)
    assert roc_auc_score(y[:, 0], p[:, 0]) == pytest.approx(roc_auc_score(y[:, 0], z[:, 0]))
    assert bce_nll(z / T, y) < bce_nll(z, y)


def test_ece_zero_when_bin_frequency_matches() -> None:
    p = np.full(10, 0.3)
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    assert ece_binary(p, y) == pytest.approx(0.0)


def test_ece_hand_example() -> None:
    # bin [0.8, 0.866) berisi 4 sampel p=0.8 dengan 2 positif: |0.5 − 0.8| · 4/4
    p = np.full(4, 0.8)
    y = np.array([1, 1, 0, 0])
    assert ece_binary(p, y, n_bins=15) == pytest.approx(0.3)
    assert ece_per_class(np.c_[p, p], np.c_[y, y]) == pytest.approx([0.3, 0.3])


def test_per_class_auroc_and_weights_equal_duplication() -> None:
    rng = np.random.default_rng(2)
    y = (rng.uniform(size=(300, 3)) < 0.4).astype(int)
    p = rng.uniform(size=(300, 3)) + 0.3 * y
    ref = [roc_auc_score(y[:, k], p[:, k]) for k in range(3)]
    assert per_class_auroc(y, p) == pytest.approx(ref)
    w = rng.integers(0, 3, size=300)
    dup = np.repeat(np.arange(300), w)
    assert per_class_auroc(y, p, w) == pytest.approx(per_class_auroc(y[dup], p[dup]))


def test_brier_and_triage_helpers() -> None:
    assert brier(np.array([[1, 0]]), np.array([[0.5, 0.5]])) == pytest.approx(0.25)
    y = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 1]])
    k, abn = triage_labels(y, ["NORM", "MI", "STTC", "CD", "HYP"])
    assert k == 0 and abn.tolist() == [False, True, True]
    neg = np.arange(100) / 100.0
    t = threshold_at_specificity(neg, 0.9)
    sens, spec = sens_spec(np.r_[neg, [0.95, 0.5]], np.r_[np.zeros(100), [1, 1]], t)
    assert spec >= 0.9 and sens == pytest.approx(0.5)
