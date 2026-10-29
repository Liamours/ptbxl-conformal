"""ResNet1D-18 untuk EKG 12-lead 100 Hz (arsitektur sama dengan ``logikaui.ipynb``).

Masukan ``(B, C_in, T)``; keluaran logit ``(B, K)``. Stem: Conv(k=15, s=2) → BN → ReLU →
MaxPool(3, s=2). Empat stage BasicBlock (k=7) dengan lebar 64/128/256/512 dan stride 1/2/2/2.
Pooling: konkatenasi rerata dan maksimum global sepanjang waktu → Linear(1024, K).
Lebar stage = (w, 2w, 4w, 8w), default w = 64. Untuk C_in = 12, K = 5, w = 64: 8.742.533
parameter (sama dengan notebook); jumlah parameter ≈ ∝ w² (w = 16 untuk varian kecil).
"""

from __future__ import annotations

import torch
from torch import nn


class BasicBlock1D(nn.Module):
    """Blok residual: y = ReLU(BN(Conv(ReLU(BN(Conv(x))))) + shortcut(x))."""

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, 7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, 7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm1d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class ResNet1D18(nn.Module):
    """ResNet-18 versi 1D dengan pooling avg+max."""

    def __init__(self, in_channels: int = 12, num_classes: int = 5, width: int = 64) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, width, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        widths, strides = (width, 2 * width, 4 * width, 8 * width), (1, 2, 2, 2)
        layers: list[nn.Module] = []
        in_planes = width
        for w, s in zip(widths, strides, strict=True):
            layers += [BasicBlock1D(in_planes, w, s), BasicBlock1D(w, w, 1)]
            in_planes = w
        self.body = nn.Sequential(*layers)
        self.fc = nn.Linear(2 * in_planes, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(self.stem(x))
        feat = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=1)
        return self.fc(feat)


def build_model(name: str, in_channels: int, num_classes: int, width: int = 64) -> nn.Module:
    """Pabrik model berdasarkan nama di config (``width`` = lebar stage pertama)."""
    if name == "resnet1d18":
        return ResNet1D18(in_channels, num_classes, width)
    raise ValueError(f"Model tidak dikenal: {name}")


def count_parameters(model: nn.Module) -> int:
    """Jumlah parameter yang dilatih."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
