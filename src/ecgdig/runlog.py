"""Utilitas I/O run: manifest, info lingkungan, JSON, dan log akses TEST."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np


def now_iso() -> str:
    """Waktu lokal ISO-8601 dengan zona."""
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def to_jsonable(obj: Any) -> Any:
    """Konversi rekursif numpy/Path ke tipe JSON."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        f = float(obj)
        return None if np.isnan(f) else f
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: Path, obj: Any) -> None:
    """Tulis JSON terindentasi (UTF-8)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(obj), indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> Any:
    """Baca JSON (UTF-8)."""
    return json.loads(path.read_text(encoding="utf-8"))


def ids_digest(ids: np.ndarray) -> str:
    """SHA-256 dari ID terurut (bukti himpunan sumber, untuk test anti-leakage)."""
    arr = np.sort(np.asarray(ids, dtype=np.int64))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def env_info() -> dict[str, Any]:
    """Versi Python/paket dan perangkat keras."""
    info: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform()}
    import pandas as pd
    import scipy
    import sklearn

    info.update(
        numpy=np.__version__,
        pandas=pd.__version__,
        scipy=scipy.__version__,
        sklearn=sklearn.__version__,
    )
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["torch_cuda"] = torch.version.cuda
    except ImportError:
        info["torch"] = None
    return info


def log_test_access(
    log_path: Path, run_dir: Path, train_hash: str, code: str, purpose: str
) -> None:
    """Tambahkan satu baris ke log akses TEST (Markdown)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new = not log_path.exists()
    with log_path.open("a", encoding="utf-8") as f:
        if new:
            f.write("# TEST_ACCESS_LOG\n\nSetiap evaluasi pada fold 10 (TEST) dicatat di sini.\n\n")
            f.write("| waktu | run | train_hash | code | tujuan |\n|---|---|---|---|---|\n")
        f.write(f"| {now_iso()} | {run_dir.name} | {train_hash[:12]} | {code[:12]} | {purpose} |\n")
