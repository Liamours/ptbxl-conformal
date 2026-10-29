"""Muat config eksperimen YAML, pilih profil path (local/colab), dan hitung hash.

Kontrak:
- File YAML berisi parameter eksperimen + blok ``profiles`` (path per mesin).
- ``experiment_hash`` = SHA-256 dari config **tanpa** ``profiles``, jadi run lokal dan Colab
  dengan parameter sama punya hash sama (untuk membandingkan hardware).
- Path relatif di profil di-resolve terhadap ``base_dir`` (default: direktori kerja).
- ``code_fingerprint`` = SHA-256 isi ``src/ecgdig/**/*.py`` (CRLF dinormalisasi ke LF),
  pengganti hash commit selama folder belum menjadi repo git.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Paths:
    """Lokasi data, cache, dan hasil untuk satu mesin."""

    raw_dir: Path
    cache_dir: Path
    runs_dir: Path
    test_access_log: Path


@dataclass(frozen=True)
class RunConfig:
    """Config ter-resolve: parameter eksperimen (``params``) + path profil."""

    params: dict[str, Any]
    profile: str
    paths: Paths
    device: str
    experiment_hash: str

    @property
    def run_name(self) -> str:
        return str(self.params["run_name"])

    @property
    def run_dir(self) -> Path:
        return self.paths.runs_dir / f"{self.run_name}__{self.profile}"


def canonical_json(obj: Any) -> str:
    """JSON deterministik (kunci terurut) untuk hashing."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def experiment_hash(params: dict[str, Any]) -> str:
    """SHA-256 parameter eksperimen tanpa blok ``profiles``."""
    p = {k: v for k, v in params.items() if k != "profiles"}
    return hashlib.sha256(canonical_json(p).encode("utf-8")).hexdigest()


def code_fingerprint(root: Path = PACKAGE_ROOT) -> str:
    """SHA-256 atas path relatif + isi semua file .py di paket."""
    h = hashlib.sha256()
    for f in sorted(root.rglob("*.py")):
        h.update(f.relative_to(root).as_posix().encode("utf-8"))
        h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def apply_overrides(params: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Terapkan override ``a.b.c=value`` (value di-parse sebagai YAML). Kunci harus sudah ada."""
    out = copy.deepcopy(params)
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"Override harus berbentuk kunci=nilai: {item!r}")
        node = out
        parts = key.strip().split(".")
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                raise KeyError(f"Kunci override tidak ada: {key}")
            node = node[part]
        if parts[-1] not in node:
            raise KeyError(f"Kunci override tidak ada: {key}")
        node[parts[-1]] = yaml.safe_load(raw)
    return out


def load_config(
    path: str | Path,
    profile: str,
    overrides: list[str] | None = None,
    base_dir: str | Path | None = None,
) -> RunConfig:
    """Baca YAML, terapkan override, dan resolve path profil."""
    with Path(path).open(encoding="utf-8") as f:
        params: dict[str, Any] = yaml.safe_load(f)
    params = apply_overrides(params, overrides or [])
    profiles = params.get("profiles", {})
    if profile not in profiles:
        raise KeyError(f"Profil {profile!r} tidak ada; tersedia: {sorted(profiles)}")
    prof = profiles[profile]
    base = Path(base_dir) if base_dir is not None else Path.cwd()

    def resolve(p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else (base / q)

    paths = Paths(
        raw_dir=resolve(prof["raw_dir"]),
        cache_dir=resolve(prof["cache_dir"]),
        runs_dir=resolve(prof["runs_dir"]),
        test_access_log=resolve(prof["test_access_log"]),
    )
    return RunConfig(
        params=params,
        profile=profile,
        paths=paths,
        device=str(prof.get("device", "cuda")),
        experiment_hash=experiment_hash(params),
    )
