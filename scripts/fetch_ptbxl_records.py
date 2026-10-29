"""Lengkapi file PTB-XL v1.0.3 lokal dari mirror S3 PhysioNet, terverifikasi SHA-256.

Hanya stdlib. Jalankan dengan ``python -I`` karena membaca file hasil unduhan.

Kontrak:
- Daftar file acuan = baris ``<sha256> <path>`` di ``SHA256SUMS.txt`` yang path-nya
  diawali salah satu ``--prefix`` (default ``records100/``).
- File lokal yang sudah ada dan hash-nya cocok dilewati. File hilang atau hash salah
  diunduh ulang ke ``*.part``, diverifikasi, lalu di-rename. File yang gagal verifikasi
  tidak disimpan.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

BASE_URL = "https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/"


def sha256_file(path: Path) -> str:
    """SHA-256 heksadesimal isi file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_sums(sums_path: Path, prefixes: tuple[str, ...]) -> list[tuple[str, str]]:
    """Baca pasangan (sha256, path relatif POSIX) yang lolos sanitasi dan filter prefix."""
    entries: list[tuple[str, str]] = []
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue
        digest, rel = parts[0].lower(), parts[1].strip()
        p = PurePosixPath(rel)
        if p.is_absolute() or ".." in p.parts or len(digest) != 64:
            raise ValueError(f"Entri SHA256SUMS mencurigakan: {line!r}")
        if rel.startswith(prefixes):
            entries.append((digest, rel))
    return entries


def ensure_file(root: Path, digest: str, rel: str, retries: int = 3) -> str:
    """Pastikan ``root/rel`` ada dengan hash ``digest``. Return status: ok/downloaded/failed."""
    dst = root / Path(*PurePosixPath(rel).parts)
    if dst.exists() and sha256_file(dst) == digest:
        return "ok"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f"{dst.name}.{os.getpid()}.part")  # unik per proses
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(BASE_URL + rel, timeout=60) as r:
                data = r.read()
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError("hash mismatch")
            tmp.write_bytes(data)
            tmp.replace(dst)
            return "downloaded"
        except Exception:  # noqa: BLE001 - jaringan/IO: coba ulang lalu laporkan
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
            time.sleep(1.0 + attempt)
    return "failed"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True, help="direktori data/raw/ptbxl")
    ap.add_argument("--prefix", action="append", default=None)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    prefixes = tuple(args.prefix or ["records100/"])
    entries = parse_sums(args.root / "SHA256SUMS.txt", prefixes)
    print(f"{len(entries)} file acuan untuk prefix {prefixes}", flush=True)
    counts = {"ok": 0, "downloaded": 0, "failed": 0}
    failed: list[str] = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(ensure_file, args.root, d, r): r for d, r in entries}
        for i, fut in enumerate(as_completed(futs), 1):
            status = fut.result()
            counts[status] += 1
            if status == "failed":
                failed.append(futs[fut])
            if i % 2000 == 0:
                print(f"  {i}/{len(entries)} {counts}", flush=True)
    print(f"Selesai dalam {time.perf_counter() - t0:.1f} s: {counts}")
    for rel in failed[:20]:
        print("GAGAL:", rel)
    return 0 if counts["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
