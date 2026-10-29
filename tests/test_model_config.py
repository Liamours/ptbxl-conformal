from pathlib import Path

import numpy as np
import pytest
import torch

from ecgdig.config import apply_overrides, experiment_hash, load_config
from ecgdig.data.preprocess import apply_standardizer, fit_standardizer, highpass
from ecgdig.models.resnet1d import ResNet1D18, count_parameters
from ecgdig.models.train import (
    TrainSettings,
    cpu_latency_ms,
    pos_weight_vector,
    predict_logits,
    train_model,
)

ROOT = Path(__file__).resolve().parents[1]


def test_resnet_shape_and_param_count_matches_notebook() -> None:
    m = ResNet1D18(12, 5)
    assert count_parameters(m) == 8_742_533
    assert m(torch.zeros(3, 12, 1000)).shape == (3, 5)


def test_standardizer_and_highpass() -> None:
    rng = np.random.default_rng(0)
    x = (rng.normal(size=(50, 12, 1000)) * 2 + 3).astype(np.float32)
    mean, std = fit_standardizer(x)
    z = apply_standardizer(x, mean, std)
    assert z.dtype == np.float32
    assert np.allclose(z.mean(axis=(0, 2)), 0, atol=1e-4)
    assert np.allclose(z.std(axis=(0, 2)), 1, atol=1e-3)
    hp = highpass(np.full((1, 1, 1000), 5.0, np.float32), 100.0, 0.5)
    assert np.abs(hp).max() < 1e-3  # DC hilang


def test_pos_weight_vector() -> None:
    y = np.array([[1, 0], [0, 0], [1, 1], [0, 0]])
    assert pos_weight_vector(y, "balanced") == pytest.approx([1.0, 3.0])
    assert pos_weight_vector(y, "none") is None


def test_tiny_training_learns_separable_signal() -> None:
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=(256, 2)) < 0.5).astype(np.float32)
    x = rng.normal(size=(256, 12, 200)).astype(np.float32)
    x[:, 0, :] += 2.0 * y[:, [0]]
    x[:, 1, :] += 2.0 * y[:, [1]]
    torch.manual_seed(0)
    model = ResNet1D18(12, 2)
    s = TrainSettings(epochs=3, batch_size=64, lr=1e-3, weight_decay=0.0, amp="none")
    res = train_model(
        model,
        torch.from_numpy(x),
        torch.from_numpy(y),
        torch.from_numpy(x),
        y,
        s,
        seed=0,
        log=lambda _: None,
    )
    assert len(res.history) == 3 and res.best_score > 0.9
    assert predict_logits(model, torch.from_numpy(x[:5])).shape == (5, 2)
    assert cpu_latency_ms(model, (1, 12, 200), threads=1, n_runs=3) > 0


def test_config_profiles_hash_and_overrides(tmp_path: Path) -> None:
    cfg = load_config(ROOT / "configs" / "stage3_c0.yaml", "local", base_dir=tmp_path)
    assert cfg.paths.raw_dir == tmp_path / "data" / "raw" / "ptbxl"
    colab = load_config(ROOT / "configs" / "stage3_c0.yaml", "colab")
    assert colab.experiment_hash == cfg.experiment_hash  # profil tidak masuk hash
    assert colab.paths.runs_dir.as_posix().endswith("ptbxl_conformal/results/runs")
    p2 = apply_overrides(cfg.params, ["training.epochs=5"])
    assert p2["training"]["epochs"] == 5 and experiment_hash(p2) != cfg.experiment_hash
    with pytest.raises(KeyError):
        apply_overrides(cfg.params, ["training.tidak_ada=1"])
