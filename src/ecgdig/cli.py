"""CLI eksperimen.

Contoh (dari akar proyek atau folder kode di Colab):
    python -m ecgdig.cli cache   --config configs/stage3_c0.yaml --profile local
    python -m ecgdig.cli e1      --config configs/stage3_c0.yaml --profile colab --seeds 42 43
    python -m ecgdig.cli all     --config configs/stage3_smoke.yaml --profile local   # E1–E6
    python -m ecgdig.cli compare --config configs/stage3_c0.yaml configs/stage3_feat_lr.yaml \\
                                 --profile colab
Override parameter: ``--set training.epochs=5`` (boleh berulang; mengubah hash).
"""

from __future__ import annotations

import argparse
import sys
import time

from ecgdig import experiments as ex
from ecgdig.config import code_fingerprint, load_config
from ecgdig.runlog import now_iso

COMMANDS = ["cache", "e1", "e2", "e3", "e4", "e5", "e6", "all", "compare"]
ALL_STEPS = ["e1", "e2", "e3", "e4", "e5", "e6"]


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):  # konsol Windows (cp1252) tidak bisa mencetak α, →
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(
        prog="ecgdig.cli", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("command", choices=COMMANDS)
    ap.add_argument("--config", required=True, nargs="+", help=">1 hanya untuk compare")
    ap.add_argument("--profile", required=True, help="nama profil path di config (local/colab)")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VAL")
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="subset seed untuk E1")
    ap.add_argument("--base-dir", default=None, help="basis path relatif (default: cwd)")
    args = ap.parse_args(argv)
    if args.command != "compare" and len(args.config) != 1:
        ap.error("perintah selain compare menerima tepat satu --config")
    cfgs = [load_config(c, args.profile, args.overrides, args.base_dir) for c in args.config]
    cfg = cfgs[0]

    def log(msg: str) -> None:
        print(msg, flush=True)

    log(
        f"=== {now_iso()} [{args.command}] run={cfg.run_name} profil={cfg.profile} "
        f"exp_hash={cfg.experiment_hash[:12]} train_hash={ex.train_hash(cfg.params)[:12]} "
        f"code={code_fingerprint()[:12]}"
    )
    t0 = time.perf_counter()
    if args.command == "compare":
        ex.run_compare(cfgs, cfg.paths.runs_dir.parent / "benchmark", log)
        log(f"Selesai dalam {time.perf_counter() - t0:.1f} s.")
        return 0
    steps = ALL_STEPS if args.command == "all" else [args.command]
    for step in steps:
        if step == "cache":
            ex.cmd_cache(cfg, log)
        elif step == "e1":
            log(ex.run_e1(cfg, args.seeds, log).to_string(index=False))
        elif step == "e2":
            tab = ex.run_e2(cfg, log)["classification"]
            sel = tab["metric"].isin(["auroc", "auprc", "ece_raw", "ece_T"])
            log(tab[sel].to_string(index=False))
        elif step == "e3":
            ex.run_e3(cfg, log)
        elif step == "e4":
            ex.run_e4(cfg, log)
        elif step == "e5":
            ex.run_e5(cfg, log)
        elif step == "e6":
            ex.run_e6(cfg, log)
    log(f"Selesai dalam {time.perf_counter() - t0:.1f} s. Hasil: {cfg.run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
