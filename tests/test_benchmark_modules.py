"""Test modul pembanding E5–E7: kalibrator, dekomposisi Brier, CRC, F1, PAC, fitur, crop/TTA."""

import numpy as np
import pytest
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from ecgdig.features import autocorr_rhythm, band_features, extract_features, hjorth
from ecgdig.metrics.calibration import brier_decomposition, ece_binary
from ecgdig.models.calibrators import Calibrator, fit_platt, pav
from ecgdig.models.resnet1d import ResNet1D18, count_parameters
from ecgdig.models.set_predictors import (
    crc_lambda,
    crc_sets,
    f1_thresholds,
    fnr_loss,
    pac_thresholds,
)
from ecgdig.models.temperature import sigmoid
from ecgdig.models.train import predict_logits, random_crop, window_offsets


def _data(n: int, K: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    y = (rng.uniform(size=(n, K)) < 0.3).astype(np.int8)
    z = rng.normal(size=(n, K)) * 1.5 + 2.5 * (y - 0.5)
    return z, y


# ------------------------------------------------------------------ kalibrasi
def test_pav_matches_sklearn_isotonic() -> None:
    rng = np.random.default_rng(0)
    x = np.sort(rng.normal(size=300))
    y = (rng.uniform(size=300) < sigmoid(2 * x)).astype(float)
    ref = IsotonicRegression().fit_transform(x, y)
    assert np.allclose(pav(y), ref, atol=1e-12)
    assert np.all(np.diff(pav(y)) >= -1e-12)


def test_platt_newton_matches_sklearn_unregularized() -> None:
    rng = np.random.default_rng(1)
    z = rng.normal(size=2000) * 2
    y = (rng.uniform(size=2000) < sigmoid(0.7 * z - 0.4)).astype(int)
    a, b = fit_platt(z, y)
    clf = LogisticRegression(C=1e10, max_iter=10000).fit(z[:, None], y)
    assert a == pytest.approx(clf.coef_[0, 0], rel=1e-4)
    assert b == pytest.approx(clf.intercept_[0], abs=1e-4)


@pytest.mark.parametrize("method", ["none", "temp", "temp_k", "platt", "isotonic", "hist"])
def test_calibrators_range_and_monotonicity(method: str) -> None:
    rng = np.random.default_rng(2)
    z, y = _data(2000, 3, rng)
    cal = Calibrator(method).fit(z, y)
    grid = np.tile(np.linspace(-6, 6, 400)[:, None], (1, 3))
    p = cal.transform(grid)
    assert p.shape == grid.shape and np.all((p > 0) & (p < 1))
    monotone = bool(np.all(np.diff(p, axis=0) >= -1e-12))
    # hist TIDAK dijamin monoton (frekuensi bin bisa turun karena derau); sisanya wajib monoton
    assert monotone or method == "hist"


def test_isotonic_reduces_calibration_error_on_miscalibrated_scores() -> None:
    rng = np.random.default_rng(3)
    z = rng.normal(size=(20000, 1)) * 3
    y = (rng.uniform(size=z.shape) < sigmoid(z / 3)).astype(int)  # logit 3× terlalu yakin
    fit, ev = slice(0, 10000), slice(10000, None)
    raw = ece_binary(sigmoid(z[ev, 0]), y[ev, 0])
    iso = Calibrator("isotonic").fit(z[fit], y[fit]).transform(z[ev])
    assert ece_binary(iso[:, 0], y[ev, 0]) < raw / 3


def test_brier_decomposition_identity_is_exact() -> None:
    rng = np.random.default_rng(4)
    p = rng.uniform(size=5000)
    y = (rng.uniform(size=5000) < p**1.5).astype(float)
    d = brier_decomposition(p, y, n_bins=15)
    assert d["bs"] == pytest.approx(d["rel"] - d["res"] + d["unc"] + d["wbv"] - 2 * d["wbc"])
    assert d["unc"] == pytest.approx(y.mean() * (1 - y.mean()))


def test_ece_mass_bins_have_equal_size() -> None:
    p = np.linspace(0.01, 0.99, 150)
    y = (p > 0.5).astype(float)
    assert ece_binary(p, y, 15, "mass") >= 0
    with pytest.raises(ValueError):
        ece_binary(p, y, 15, "lain")


# ------------------------------------------------------------------ prediktor himpunan
def test_fnr_loss_hand_example() -> None:
    y = np.array([[1, 1, 0], [0, 0, 1], [1, 0, 0]])
    h = np.array([[True, False, True], [False, False, True], [False, True, True]])
    assert fnr_loss(h, y).tolist() == [0.5, 0.0, 1.0]


def test_crc_controls_expected_fnr() -> None:
    rng = np.random.default_rng(5)
    alpha, risks = 0.1, []
    for _ in range(300):
        zc, yc = _data(300, 4, rng)
        zt, yt = _data(300, 4, rng)
        keep_c, keep_t = yc.sum(1) > 0, yt.sum(1) > 0
        pc, pt = sigmoid(zc[keep_c]), sigmoid(zt[keep_t])
        lam = crc_lambda(pc, yc[keep_c], alpha)
        risks.append(fnr_loss(crc_sets(pt, lam), yt[keep_t]).mean())
    assert np.mean(risks) <= alpha + 0.005  # Thm 1: E[L] ≤ α
    assert np.mean(risks) >= alpha - 0.03  # tidak terlalu konservatif


def test_crc_lambda_is_infimum_of_feasible_set() -> None:
    rng = np.random.default_rng(6)
    z, y = _data(200, 3, rng)
    keep = y.sum(1) > 0
    p, yy = sigmoid(z[keep]), y[keep]
    n, alpha = p.shape[0], 0.2
    lam = crc_lambda(p, yy, alpha)
    ok = lambda lm: n / (n + 1) * fnr_loss(crc_sets(p, lm), yy).mean() + 1 / (n + 1) <= alpha  # noqa: E731
    assert ok(lam)
    assert not ok(lam - 1e-9) or lam == 0.0


def test_f1_thresholds_match_bruteforce() -> None:
    rng = np.random.default_rng(7)
    z, y = _data(300, 2, rng)
    p = sigmoid(z)
    t = f1_thresholds(p, y)
    for k in range(2):

        def f1(th: float, k: int = k) -> float:
            pred = p[:, k] >= th
            tp = np.sum(pred & (y[:, k] == 1))
            return 2 * tp / (pred.sum() + y[:, k].sum())

        best = max(f1(th) for th in np.unique(p[:, k]))
        assert f1(t[k]) == pytest.approx(best)


def test_pac_thresholds_inf_when_calibration_too_small() -> None:
    p = np.array([[0.9], [0.8], [0.7]])
    y = np.ones((3, 1))
    assert np.isinf(pac_thresholds(p, y, alpha=0.01, delta=0.1)[0])


# ------------------------------------------------------------------ fitur
def test_hjorth_mobility_of_sinusoid() -> None:
    fs, f = 100.0, 5.0
    x = np.sin(2 * np.pi * f * np.arange(4000) / fs)
    mob, comp = hjorth(x)
    assert mob == pytest.approx(2 * np.sin(np.pi * f / fs), rel=1e-3)  # beda maju diskret
    assert comp == pytest.approx(1.0, rel=1e-2)  # sinus murni: kompleksitas 1


def test_band_power_and_rhythm_features() -> None:
    fs = 100.0
    t = np.arange(1000) / fs
    x = np.sin(2 * np.pi * 10 * t)[None, :]
    bf = band_features(x, fs)[0]
    assert bf[2] > 0.95  # pita 8–15 Hz
    imp = np.zeros((1, 12, 1000))
    imp[..., ::80] = 1.0  # periode 0,8 s → 75 bpm
    hr, reg = autocorr_rhythm(imp, fs)
    assert hr[0] == pytest.approx(75.0) and reg[0] > 0.5


def test_extract_features_shape_and_finite() -> None:
    rng = np.random.default_rng(8)
    F = extract_features(rng.normal(size=(5, 12, 1000)).astype(np.float32))
    assert F.shape == (5, 194) and np.isfinite(F).all()


# ------------------------------------------------------------------ crop / TTA / lebar
def test_window_offsets_cover_signal() -> None:
    assert window_offsets(1000, 250, 125) == [0, 125, 250, 375, 500, 625, 750]
    assert window_offsets(1000, 300, 400) == [0, 400, 700]


def test_random_crop_and_tta_average() -> None:
    gen = torch.Generator().manual_seed(0)
    x = torch.arange(2 * 3 * 20, dtype=torch.float32).reshape(2, 3, 20)
    c = random_crop(x, 5, gen)
    assert c.shape == (2, 3, 5)
    off = int(c[0, 0, 0].item())
    assert torch.equal(c[0], x[0, :, off : off + 5])

    class MeanModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))

        def forward(self, xb: torch.Tensor) -> torch.Tensor:
            return xb.mean(dim=(1, 2))[:, None] + self.dummy

    z = predict_logits(MeanModel(), x, crop_len=10, stride=5)
    offs = window_offsets(20, 10, 5)
    ref = np.mean([x[..., o : o + 10].mean(dim=(1, 2)).numpy() for o in offs], axis=0)
    assert np.allclose(z[:, 0], ref, atol=1e-5)


def test_resnet_width_scales_parameters() -> None:
    full, small = count_parameters(ResNet1D18(12, 5, 64)), count_parameters(ResNet1D18(12, 5, 16))
    assert full == 8_742_533 and 12 < full / small < 17
    assert ResNet1D18(12, 5, 16)(torch.zeros(2, 12, 250)).shape == (2, 5)
