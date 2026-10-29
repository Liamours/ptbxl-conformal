"""Lengkapi/verifikasi file PTB-XL v1.0.3 dari mirror S3 PhysioNet, per file dengan SHA-256.

Kontrak:
- Daftar file dan hash diambil dari ``SHA256SUMS.txt`` resmi di ``--root``.
- Hanya path yang diawali salah satu ``--prefix`` yang diproses (default ``records100/``).
- File lokal yang hash-nya cocok dibiarkan; yang hilang/tidak cocok diunduh ulang dari
  ``https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/<path>``, diverifikasi, baru ditulis.
- Path disanitasi: harus relatif dan tetap di dalam ``--root``.

Contoh: ``python -I scripts/download_ptbxl.py --root data/raw/ptbxl --prefix records100/``
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

BASE_URL = "https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/"


def parse_sha256sums(text: str) -> dict[str, str]:
    """Parse baris ``<sha256> <relpath>`` menjadi {relpath: sha256}."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2 and len(parts[0]) == 64:
            out[parts[1].strip()] = parts[0].lower()
    return out


def safe_target(root: Path, rel: str) -> Path:
    """Tolak path absolut atau yang keluar dari ``root``."""
    p = PurePosixPath(rel)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"path tidak aman: {rel!r}")
    target = (root / Path(*p.parts)).resolve()
    if root.resolve() not in target.parents:
        raise ValueError(f"path keluar dari root: {rel!r}")
    return target


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_verified(rel: str, expected: str, root: Path, retries: int = 3) -> str:
    """Unduh satu file; tulis hanya jika SHA-256 cocok. Mengembalikan status."""
    target = safe_target(root, rel)
    if target.exists() and sha256_file(target) == expected:
        return "ok"
    last_err = ""
    for _ in range(retries):
        try:
            with urllib.request.urlopen(BASE_URL + rel, timeout=60) as resp:
                data = resp.read()
            if hashlib.sha256(data).hexdigest() != expected:
                last_err = "hash mismatch"
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(target.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(target)
            return "downloaded"
        except OSError as exc:
            last_err = str(exc)
    return f"failed: {last_err}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--prefix", action="append", default=None)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args(argv)
    prefixes = tuple(args.prefix or ["records100/"])
    sums = parse_sha256sums((args.root / "SHA256SUMS.txt").read_text(encoding="utf-8"))
    todo = {k: v for k, v in sums.items() if k.startswith(prefixes)}
    counts: dict[str, int] = {}
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_verified, rel, sha, args.root): rel for rel, sha in todo.items()}
        for i, fut in enumerate(as_completed(futs), 1):
            status = fut.result()
            key = status.split(":")[0]
            counts[key] = counts.get(key, 0) + 1
            if key == "failed":
                failures.append(f"{futs[fut]}: {status}")
            if i % 5000 == 0:
                print(f"{i}/{len(todo)} {counts}", flush=True)
    print(f"selesai: {len(todo)} file, {counts}")
    for line in failures[:20]:
        print("GAGAL", line)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
