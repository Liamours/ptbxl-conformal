"""Loop pelatihan multi-label dengan seleksi epoch pada VAL_A.

Loss: BCE per kelas, opsional berbobot (``pos_weight``):
    L = −(1/(nK)) Σ_{i,k} [w_k y_ik log σ(z_ik) + (1 − y_ik) log(1 − σ(z_ik))],
    w_k = N⁻_k / N⁺_k (dari TRAIN) jika ``pos_weight == "balanced"``, selain itu w_k = 1.
Optimasi: AdamW, jadwal LR cosine per langkah (opsional), autocast bf16 di CUDA (opsional,
hanya forward pelatihan; semua inferensi fp32).
Seleksi: bobot pada epoch dengan macro-AUROC VAL_A tertinggi (tanpa early termination).
Random crop (opsional, ``crop_len`` = L): tiap sampel latih dipotong ke jendela [o, o+L) dengan
o ~ Uniform{0..T−L}. Inferensi memakai rerata logit atas jendela geser o_j = j·s (+ jendela
terakhir di T−L): ẑ = (1/J) Σ_j z(x[o_j : o_j+L]). Jika galat logit antar-jendela berkorelasi
ρ dengan varians σ², Var(ẑ) = σ²(1 + (J−1)ρ)/J ≤ σ².
Data latih disimpan di device sebagai satu tensor; urutan batch dari ``torch.Generator(seed)``.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from ecgdig.metrics.classification import per_class_auroc


@dataclass(frozen=True)
class TrainSettings:
    """Hiperparameter pelatihan (dari ``training`` di config)."""

    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    scheduler: str = "cosine"
    pos_weight: str = "none"
    amp: str = "bf16"
    eval_batch_size: int = 1024
    cudnn_benchmark: bool = True
    crop_len: int | None = None  # sampel; None = sinyal penuh
    tta_stride: int | None = None  # langkah jendela saat inferensi (default crop_len // 2)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainSettings:
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in keys})


@dataclass
class TrainResult:
    """Riwayat per epoch, epoch terbaik (1-indeks), dan state_dict terbaik (CPU)."""

    history: list[dict[str, float]]
    best_epoch: int
    best_score: float
    best_state: dict[str, torch.Tensor]


def set_seed(seed: int) -> None:
    """Seed global torch (CPU + CUDA). Keacakan numpy di paket ini selalu lewat Generator."""
    torch.manual_seed(seed)


def pos_weight_vector(y_train: np.ndarray, mode: str) -> np.ndarray | None:
    """w_k = N⁻_k / N⁺_k untuk ``balanced``; None untuk ``none``."""
    if mode == "none":
        return None
    if mode == "balanced":
        pos = y_train.sum(axis=0).astype(np.float64)
        return (y_train.shape[0] - pos) / np.maximum(pos, 1.0)
    raise ValueError(f"pos_weight tidak dikenal: {mode}")


def _autocast(device: torch.device, amp: str) -> torch.autocast:
    enabled = device.type == "cuda" and amp == "bf16"
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)


def window_offsets(total: int, length: int, stride: int) -> list[int]:
    """Offset jendela geser o_j = j·stride, ditambah jendela terakhir yang berakhir di ``total``."""
    offs = list(range(0, total - length + 1, stride))
    if offs[-1] != total - length:
        offs.append(total - length)
    return offs


def random_crop(xb: torch.Tensor, length: int, gen: torch.Generator) -> torch.Tensor:
    """Potong tiap sampel (B, C, T) ke jendela acak sepanjang ``length``."""
    B, C, T = xb.shape
    off = torch.randint(0, T - length + 1, (B,), generator=gen).to(xb.device)
    idx = off[:, None] + torch.arange(length, device=xb.device)[None, :]
    return torch.gather(xb, 2, idx[:, None, :].expand(B, C, length))


@torch.no_grad()
def predict_logits(
    model: nn.Module,
    x: torch.Tensor,
    batch_size: int = 1024,
    crop_len: int | None = None,
    stride: int | None = None,
) -> np.ndarray:
    """Logit fp32 (N, K); jika ``crop_len``: rerata logit atas jendela geser (lihat modul)."""
    if crop_len is None or crop_len >= x.shape[-1]:
        return _predict_full(model, x, batch_size)
    offs = window_offsets(x.shape[-1], crop_len, stride or max(1, crop_len // 2))
    zs = [_predict_full(model, x[..., o : o + crop_len], batch_size) for o in offs]
    return np.mean(zs, axis=0, dtype=np.float64).astype(np.float32)


@torch.no_grad()
def _predict_full(model: nn.Module, x: torch.Tensor, batch_size: int = 1024) -> np.ndarray:
    """Logit float32 (N, K) untuk masukan (N, C, T), selalu presisi penuh fp32.

    Inferensi bf16 membulatkan logit ke ±3 digit signifikan sehingga banyak skor kembar
    (ties); ties membuat kuantil conformal konservatif dan merusak eksaknya teori E3.
    """
    model.eval()
    device = next(model.parameters()).device
    outs = []
    for i in range(0, x.shape[0], batch_size):
        xb = x[i : i + batch_size].to(device, non_blocking=True)
        outs.append(model(xb).float().cpu())
    return torch.cat(outs).numpy()


def train_model(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: np.ndarray,
    settings: TrainSettings,
    seed: int,
    log: Callable[[str], None] = print,
) -> TrainResult:
    """Latih ``model`` (sudah di device) dan kembalikan bobot epoch terbaik di VAL_A."""
    device = next(model.parameters()).device
    torch.backends.cudnn.benchmark = settings.cudnn_benchmark
    pw = pos_weight_vector(y_train.cpu().numpy(), settings.pos_weight)
    crit = nn.BCEWithLogitsLoss(
        pos_weight=None if pw is None else torch.tensor(pw, dtype=torch.float32, device=device)
    )
    opt = torch.optim.AdamW(model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)
    n = x_train.shape[0]
    steps_per_epoch = (n + settings.batch_size - 1) // settings.batch_size
    sched = None
    if settings.scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=settings.epochs * steps_per_epoch
        )
    elif settings.scheduler != "none":
        raise ValueError(f"scheduler tidak dikenal: {settings.scheduler}")
    gen = torch.Generator().manual_seed(seed)
    history: list[dict[str, float]] = []
    best_score, best_epoch, best_state = -np.inf, 0, copy.deepcopy(model.state_dict())
    for epoch in range(1, settings.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        perm = torch.randperm(n, generator=gen).to(device)
        loss_sum = 0.0
        for s in range(steps_per_epoch):
            idx = perm[s * settings.batch_size : (s + 1) * settings.batch_size]
            xb, yb = x_train.index_select(0, idx), y_train.index_select(0, idx)
            if settings.crop_len:
                xb = random_crop(xb, settings.crop_len, gen)
            opt.zero_grad(set_to_none=True)
            with _autocast(device, settings.amp):
                logits = model(xb)
            loss = crit(logits.float(), yb)
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
            loss_sum += float(loss.detach()) * idx.numel()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_train = time.perf_counter() - t0
        z_val = predict_logits(
            model, x_val, settings.eval_batch_size, settings.crop_len, settings.tta_stride
        )
        auroc = per_class_auroc(y_val, z_val)
        macro = float(np.nanmean(auroc))
        val_loss = float(np.mean(np.logaddexp(0.0, z_val) - y_val * z_val))
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / n,
            "val_a_loss": val_loss,
            "val_a_macro_auroc": macro,
            "lr": opt.param_groups[0]["lr"],
            "time_train_s": t_train,
            "time_epoch_s": time.perf_counter() - t0,
        }
        history.append(row)
        if macro > best_score:
            best_score, best_epoch = macro, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        log(
            f"  epoch {epoch:3d} | loss {row['train_loss']:.4f} | VAL_A loss {val_loss:.4f} "
            f"AUROC {macro:.4f} | {row['time_epoch_s']:.1f}s"
        )
    return TrainResult(history, best_epoch, float(best_score), best_state)


@torch.no_grad()
def cpu_latency_ms(
    model: nn.Module, input_shape: tuple[int, ...], threads: int = 1, n_runs: int = 50
) -> float:
    """Median latensi inferensi CPU (ms) untuk satu masukan ``input_shape``."""
    m = copy.deepcopy(model).cpu().eval()
    old = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        x = torch.zeros(input_shape)
        for _ in range(5):
            m(x)
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            m(x)
            times.append((time.perf_counter() - t0) * 1e3)
    finally:
        torch.set_num_threads(old)
    return float(np.median(times))
