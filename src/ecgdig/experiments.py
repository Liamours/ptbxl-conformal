"""Eksperimen E1–E4 (sinyal PTB-XL 100 Hz). Sama persis untuk profil local dan colab.

E1  Latih ResNet1D-18 per seed; seleksi epoch di VAL_A; simpan logit VAL_A/VAL_B/TEST.
E2  Temperature (VAL_A) + metrik TEST (AUROC/AUPRC/ECE/Brier/triage, CI bootstrap pasien)
    + conformal M0–M3 dikalibrasi di VAL_B, dievaluasi di TEST, untuk semua α.
E3  Validitas sampel-hingga ("T2"): split acak kalibrasi/uji berulang pada positif TEST,
    distribusi coverage vs teori Beta-Binomial; varian PAC.
E4  Coverage per subgrup (jenis kelamin, kelompok umur, flag kualitas sinyal):
    Mondrian marginal vs Mondrian per (kelas, subgrup), CI bootstrap pasien.

Penjaga anti-leakage: partisi per pasien (overlap harus 0), statistik normalisasi dari TRAIN,
temperature dari VAL_A, kalibrasi conformal dari VAL_B, setiap evaluasi TEST dicatat.
E2–E4 menolak berjalan jika ``train_hash`` config ≠ ``train_hash`` saat E1.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ecgdig.config import RunConfig, canonical_json, code_fingerprint
from ecgdig.data.preprocess import apply_standardizer, fit_standardizer, highpass
from ecgdig.data.ptbxl import FS_LR_HZ, age_band, build_cache, load_cache, load_metadata
from ecgdig.data.splits import assign_partitions, patient_overlap, splits_table
from ecgdig.eval.bootstrap import patient_bootstrap_weights, percentile_ci, weighted_rate
from ecgdig.eval.validity import compare_with_theory, pac_rank, simulate_split_coverage
from ecgdig.metrics.calibration import ece_per_class
from ecgdig.metrics.classification import (
    brier,
    per_class_auprc,
    per_class_auroc,
    sens_spec,
    threshold_at_specificity,
    triage_labels,
)
from ecgdig.models.conformal import calibrate, conformal_rank, predict_sets, set_metrics
from ecgdig.models.temperature import (
    apply_temperature,
    bce_nll,
    fit_temperature,
    fit_temperature_per_class,
    sigmoid,
)
from ecgdig.runlog import env_info, ids_digest, log_test_access, now_iso, read_json, write_json

Log = Callable[[str], None]
EVAL_PARTITIONS = ("VAL_A", "VAL_B", "TEST")


# ---------------------------------------------------------------- run directory & manifest
def train_hash(params: dict[str, Any]) -> str:
    """Hash bagian config yang menentukan model (tanpa daftar seed)."""
    sub = {k: params[k] for k in ("data", "split", "preprocess", "model", "training")}
    sub["training"] = {k: v for k, v in sub["training"].items() if k != "seeds"}
    return hashlib.sha256(canonical_json(sub).encode("utf-8")).hexdigest()


def prepare_run_dir(cfg: RunConfig, create: bool) -> Path:
    """Buat/verifikasi ``run_dir``; ``manifest.json`` mengunci train_hash."""
    run_dir, th = cfg.run_dir, train_hash(cfg.params)
    man_path = run_dir / "manifest.json"
    if man_path.exists():
        man = read_json(man_path)
        if man["train_hash"] != th:
            raise RuntimeError(
                f"train_hash config ({th[:12]}) ≠ manifest run ({man['train_hash'][:12]}). "
                "Ganti run_name untuk config baru; jangan campur hasil."
            )
        return run_dir
    if not create:
        raise FileNotFoundError(f"Run belum ada: {run_dir}. Jalankan E1 dulu.")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        man_path,
        {
            "run_name": cfg.run_name,
            "profile": cfg.profile,
            "created": now_iso(),
            "train_hash": th,
            "experiment_hash": cfg.experiment_hash,
            "code_fingerprint": code_fingerprint(),
            "env": env_info(),
        },
    )
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(cfg.params, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return run_dir


def stage_record(run_dir: Path, stage: str, cfg: RunConfig, extra: dict[str, Any]) -> None:
    """Catat provenance tiap tahap (E1–E4) di ``<stage>/stage.json``."""
    write_json(
        run_dir / stage / "stage.json",
        {
            "stage": stage,
            "finished": now_iso(),
            "experiment_hash": cfg.experiment_hash,
            "code_fingerprint": code_fingerprint(),
            "env": env_info(),
            **extra,
        },
    )


# ---------------------------------------------------------------- data
def cmd_cache(cfg: RunConfig, log: Log = print) -> None:
    """Bangun cache sinyal 100 Hz dari WFDB mentah (jika belum ada)."""
    classes = cfg.params["data"]["classes"]
    if (cfg.paths.cache_dir / "X_100hz_float32.npy").exists():
        log(f"Cache sudah ada di {cfg.paths.cache_dir}; tidak dibangun ulang.")
        return
    sig, meta = build_cache(cfg.paths.raw_dir, cfg.paths.cache_dir, classes)
    log(f"Cache ditulis: {sig} dan {meta}")


def load_inputs(cfg: RunConfig, log: Log = print) -> tuple[np.ndarray, pd.DataFrame, pd.Series]:
    """Sinyal (N, 12, 1000) mV, metadata, dan partisi (setelah subset opsional)."""
    p = cfg.params
    X, meta = load_cache(cfg.paths.raw_dir, cfg.paths.cache_dir, p["data"]["classes"])
    s = p["split"]
    part = assign_partitions(
        meta, s["train_folds"], s["val_fold"], s["test_fold"], s["val_a_fraction"], s["seed"]
    )
    overlap = patient_overlap(meta, part)
    if any(overlap.values()):
        raise RuntimeError(f"Overlap pasien antar-partisi: {overlap}")
    cap = p["data"].get("max_records_per_partition")
    if cap:
        keep = part.groupby(part).cumcount().to_numpy() < int(cap)
        X, meta, part = X[keep], meta[keep], part[keep]
        log(f"[subset] {int(cap)} rekaman pertama per partisi → {len(meta)} rekaman")
    log("Rekaman per partisi: " + ", ".join(f"{k}={v}" for k, v in part.value_counts().items()))
    return X, meta, part


def _preds_path(run_dir: Path, seed: int) -> Path:
    return run_dir / "e1" / f"seed_{seed}" / "preds.npz"


def load_preds(run_dir: Path, seed: int) -> dict[str, np.ndarray]:
    """Isi ``preds.npz`` satu seed."""
    with np.load(_preds_path(run_dir, seed), allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def available_seeds(cfg: RunConfig) -> list[int]:
    """Seed dari config yang sudah punya preds.npz."""
    seeds = [int(s) for s in cfg.params["training"]["seeds"]]
    have = [s for s in seeds if _preds_path(cfg.run_dir, s).exists()]
    if not have:
        raise FileNotFoundError("Belum ada seed E1 yang selesai.")
    return have


# ---------------------------------------------------------------- E1
def _pack_preds(
    logits: dict[str, np.ndarray],
    Y: np.ndarray,
    meta: pd.DataFrame,
    masks: dict[str, np.ndarray],
    classes: list[str],
) -> dict[str, np.ndarray]:
    """Isi preds.npz: logit, label, ecg_id, patient_id, dan partisi untuk VAL_A/VAL_B/TEST."""
    out: dict[str, np.ndarray] = {"classes": np.array(classes)}
    for P in EVAL_PARTITIONS:
        m = masks[P]
        out[f"logits_{P}"] = logits[P].astype(np.float32)
        out[f"y_{P}"] = Y[m].astype(np.int8)
        out[f"ecg_id_{P}"] = meta.index[m].to_numpy(np.int64)
        out[f"patient_id_{P}"] = meta["patient_id"].to_numpy(np.int64)[m]
        out[f"partition_{P}"] = np.array([P])
    return out


def _finish_seed(
    sd: Path, seed: int, out: dict[str, np.ndarray], summary: dict[str, Any], classes: list[str]
) -> None:
    auc_a = per_class_auroc(out["y_VAL_A"], out["logits_VAL_A"])
    summary["val_a_auroc_per_class"] = dict(zip(classes, auc_a, strict=True))
    summary["finished"] = now_iso()
    write_json(sd / "summary.json", {"seed": seed, **summary})
    # Ditulis terakhir: keberadaan preds.npz menandai seed selesai (E1 resumable).
    np.savez_compressed(sd / "preds.npz", **out)  # type: ignore[arg-type]  # stub numpy


def run_e1(cfg: RunConfig, seeds: list[int] | None = None, log: Log = print) -> pd.DataFrame:
    """Latih semua seed yang belum selesai (resumable). Model: ``model.name`` di config."""
    p = cfg.params
    run_dir = prepare_run_dir(cfg, create=True)
    seeds = [int(s) for s in (seeds or p["training"]["seeds"])]
    todo = [s for s in seeds if not _preds_path(run_dir, s).exists()]
    done = sorted(set(seeds) - set(todo))
    log(f"E1 run_dir={run_dir}\n  seed selesai: {done}, akan dilatih: {todo}")
    if todo:
        classes = list(p["data"]["classes"])
        X, meta, part = load_inputs(cfg, log)
        splits_table(meta, part).to_csv(run_dir / "splits.csv", index=False)
        pp = p["preprocess"]
        if pp.get("highpass_hz"):
            X = highpass(X, FS_LR_HZ, float(pp["highpass_hz"]), int(pp.get("highpass_order", 2)))
        masks = {P: (part == P).to_numpy() for P in ("TRAIN", *EVAL_PARTITIONS)}
        Y = meta[classes].to_numpy(np.float32)
        if p["model"]["name"] == "features_logreg":
            _e1_features(cfg, run_dir, todo, X, Y, meta, masks, classes, log)
        else:
            _e1_deep(cfg, run_dir, todo, X, Y, meta, masks, classes, log)
    rows = [read_json(run_dir / "e1" / f"seed_{s}" / "summary.json") for s in available_seeds(cfg)]
    keep = ("seed", "best_epoch", "val_a_macro_auroc", "train_time_s", "finished")
    df = pd.DataFrame([{k: r.get(k) for k in keep} for r in rows])
    df.to_csv(run_dir / "e1" / "summary.csv", index=False)
    stage_record(run_dir, "e1", cfg, {"seeds_done": df["seed"].tolist()})
    return df


def _e1_deep(
    cfg: RunConfig,
    run_dir: Path,
    todo: list[int],
    X: np.ndarray,
    Y: np.ndarray,
    meta: pd.DataFrame,
    masks: dict[str, np.ndarray],
    classes: list[str],
    log: Log,
) -> None:
    import torch

    from ecgdig.models.resnet1d import build_model, count_parameters
    from ecgdig.models.train import (
        TrainSettings,
        cpu_latency_ms,
        predict_logits,
        set_seed,
        train_model,
        window_offsets,
    )

    p = cfg.params
    mean, std = fit_standardizer(X[masks["TRAIN"]])
    write_json(
        run_dir / "norm_stats.json",
        {
            "mean_mv": mean,
            "std_mv": std,
            "source": "TRAIN",
            "n_records": int(masks["TRAIN"].sum()),
            "ecg_ids_sha256": ids_digest(meta.index[masks["TRAIN"]].to_numpy()),
        },
    )
    X = apply_standardizer(X, mean, std)
    want = cfg.device
    device = torch.device(want if (want != "cuda" or torch.cuda.is_available()) else "cpu")
    gpu = f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""
    log(f"Device: {device}{gpu}")
    settings = TrainSettings.from_dict(p["training"])
    width = int(p["model"].get("width", 64))
    x_tr = torch.from_numpy(X[masks["TRAIN"]]).to(device)
    y_tr = torch.from_numpy(Y[masks["TRAIN"]]).to(device)
    x_va = torch.from_numpy(X[masks["VAL_A"]]).to(device)
    model = None
    for seed in todo:
        sd = run_dir / "e1" / f"seed_{seed}"
        sd.mkdir(parents=True, exist_ok=True)
        log(f"--- seed {seed} ---")
        set_seed(seed)
        model = build_model(p["model"]["name"], 12, len(classes), width).to(device)
        res = train_model(model, x_tr, y_tr, x_va, Y[masks["VAL_A"]], settings, seed, log)
        model.load_state_dict(res.best_state)
        logits = {
            P: predict_logits(
                model,
                torch.from_numpy(X[masks[P]]),
                settings.eval_batch_size,
                settings.crop_len,
                settings.tta_stride,
            )
            for P in EVAL_PARTITIONS
        }
        torch.save(res.best_state, sd / "model_best.pt")
        pd.DataFrame(res.history).to_csv(sd / "history.csv", index=False)
        summary = {
            "best_epoch": res.best_epoch,
            "val_a_macro_auroc": res.best_score,
            "train_time_s": float(sum(h["time_epoch_s"] for h in res.history)),
        }
        _finish_seed(sd, seed, _pack_preds(logits, Y, meta, masks, classes), summary, classes)
        log(f"seed {seed}: epoch terbaik {res.best_epoch}, VAL_A macro-AUROC {res.best_score:.4f}")
    eff_path = run_dir / "e1" / "efficiency.json"
    if model is not None and not eff_path.exists():
        T = X.shape[-1]
        L = settings.crop_len or T
        n_win = len(window_offsets(T, L, settings.tta_stride or max(1, L // 2))) if L < T else 1
        shape = (n_win, 12, L)  # TTA: semua jendela satu rekaman diproses sebagai satu batch
        size = (run_dir / "e1" / f"seed_{todo[-1]}" / "model_best.pt").stat().st_size
        write_json(
            eff_path,
            {
                "n_parameters": count_parameters(model),
                "state_dict_bytes": size,
                "inference_input_shape": shape,
                "cpu_latency_ms_1thread": cpu_latency_ms(model, shape, threads=1),
                "cpu_latency_ms_4threads": cpu_latency_ms(model, shape, threads=4),
                "env": env_info(),
            },
        )


def _e1_features(
    cfg: RunConfig,
    run_dir: Path,
    todo: list[int],
    X: np.ndarray,
    Y: np.ndarray,
    meta: pd.DataFrame,
    masks: dict[str, np.ndarray],
    classes: list[str],
    log: Log,
) -> None:
    """Model pembanding klasik: 194 fitur ber-rumus + regresi logistik L2 per kelas.

    min_w Σ_i log(1 + exp(−s_i wᵀφ_i)) + ‖w‖²/(2C), s_i ∈ {±1}; C per kelas dipilih dari grid
    dengan AUROC VAL_A. Fitur distandarisasi dengan statistik TRAIN. Deterministik.
    """
    import time

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    from ecgdig.features import extract_features

    t0 = time.perf_counter()
    F = extract_features(X, FS_LR_HZ)
    t_feat = time.perf_counter() - t0
    log(f"Fitur: {F.shape} dalam {t_feat:.1f} s")
    tr, va = masks["TRAIN"], masks["VAL_A"]
    mu = F[tr].mean(axis=0)
    sigma = np.maximum(F[tr].std(axis=0), 1e-6)
    write_json(
        run_dir / "norm_stats.json",
        {
            "feature_mean": mu,
            "feature_std": sigma,
            "source": "TRAIN",
            "n_records": int(tr.sum()),
            "ecg_ids_sha256": ids_digest(meta.index[tr].to_numpy()),
        },
    )
    Z = (F - mu) / sigma
    grid = [float(c) for c in cfg.params["model"].get("c_grid", [0.01, 0.1, 1.0, 10.0])]
    for seed in todo:
        sd = run_dir / "e1" / f"seed_{seed}"
        sd.mkdir(parents=True, exist_ok=True)
        t1 = time.perf_counter()
        logits = {P: np.zeros((int(masks[P].sum()), len(classes))) for P in EVAL_PARTITIONS}
        chosen: dict[str, float] = {}
        coefs = []
        for k, c in enumerate(classes):
            best_auc, best_c, best_clf = -1.0, grid[0], None
            for C in grid:
                clf = LogisticRegression(C=C, max_iter=5000).fit(Z[tr], Y[tr, k])
                auc = float(roc_auc_score(Y[va, k], clf.decision_function(Z[va])))
                if auc > best_auc:
                    best_auc, best_c, best_clf = auc, C, clf
            assert best_clf is not None
            chosen[c] = best_c
            coefs.append(np.r_[best_clf.coef_.ravel(), best_clf.intercept_])
            for P in EVAL_PARTITIONS:
                logits[P][:, k] = best_clf.decision_function(Z[masks[P]])
            log(f"  {c}: C={best_c} AUROC VAL_A={best_auc:.4f}")
        W = np.array(coefs)
        np.save(sd / "coef.npy", W)
        out = _pack_preds(logits, Y, meta, masks, classes)
        auc_a = float(np.mean(per_class_auroc(out["y_VAL_A"], out["logits_VAL_A"])))
        summary = {
            "best_epoch": 0,
            "val_a_macro_auroc": auc_a,
            "C": chosen,
            "train_time_s": time.perf_counter() - t1 + t_feat,
        }
        _finish_seed(sd, seed, out, summary, classes)
        log(f"seed {seed}: VAL_A macro-AUROC {auc_a:.4f}")
        eff_path = run_dir / "e1" / "efficiency.json"
        if not eff_path.exists():
            reps = []
            for _ in range(20):
                t2 = time.perf_counter()
                z1 = (extract_features(X[:1], FS_LR_HZ) - mu) / sigma
                _ = z1 @ W[:, :-1].T + W[:, -1]
                reps.append((time.perf_counter() - t2) * 1e3)
            write_json(
                eff_path,
                {
                    "n_parameters": int(W.size),
                    "state_dict_bytes": int((sd / "coef.npy").stat().st_size),
                    "inference_input_shape": [1, 12, X.shape[-1]],
                    "cpu_latency_ms_1thread": float(np.median(reps)),
                    "cpu_latency_ms_4threads": None,
                    "env": env_info(),
                },
            )


# ---------------------------------------------------------------- E2
def _calibrated(d: dict[str, np.ndarray]) -> tuple[float, np.ndarray, dict[str, np.ndarray]]:
    """T bersama dan T_k per kelas dari VAL_A; probabilitas terkalibrasi (T bersama)."""
    T = fit_temperature(d["logits_VAL_A"], d["y_VAL_A"])
    Tk = fit_temperature_per_class(d["logits_VAL_A"], d["y_VAL_A"])
    probs = {P: apply_temperature(d[f"logits_{P}"], T) for P in EVAL_PARTITIONS}
    return T, Tk, probs


def run_e2(cfg: RunConfig, log: Log = print) -> dict[str, pd.DataFrame]:
    """Kalibrasi + metrik TEST + conformal M0–M3."""
    p, ev, cp = cfg.params, cfg.params["evaluation"], cfg.params["conformal"]
    run_dir = prepare_run_dir(cfg, create=False)
    seeds = available_seeds(cfg)
    classes = list(p["data"]["classes"])
    log_test_access(
        cfg.paths.test_access_log,
        run_dir,
        train_hash(p),
        code_fingerprint(),
        f"E2 metrik+conformal TEST, seeds {seeds}",
    )
    ref = load_preds(run_dir, seeds[0])
    W = patient_bootstrap_weights(
        ref["patient_id_TEST"],
        int(ev["bootstrap_resamples"]),
        np.random.default_rng(int(ev["bootstrap_seed"])),
    )
    cls_rows: list[dict[str, Any]] = []
    conf_rows: list[dict[str, Any]] = []
    for seed in seeds:
        d = load_preds(run_dir, seed)
        if not np.array_equal(d["ecg_id_TEST"], ref["ecg_id_TEST"]):
            raise RuntimeError("Urutan TEST berbeda antar-seed.")
        T, Tk, pr = _calibrated(d)
        zT, yT = d["logits_TEST"], d["y_TEST"].astype(np.int64)
        raw, cal_k = sigmoid(zT), apply_temperature(zT, Tk)

        def add(
            metric: str,
            cls: str,
            value: float,
            lo: float = np.nan,
            hi: float = np.nan,
            seed: int = seed,
        ) -> None:
            cls_rows.append(
                {
                    "seed": seed,
                    "metric": metric,
                    "class": cls,
                    "value": float(value),
                    "ci_lo": lo,
                    "ci_hi": hi,
                }
            )

        auc, ap = per_class_auroc(yT, raw), per_class_auprc(yT, raw)
        boot = np.array(
            [np.r_[per_class_auroc(yT, raw, w), per_class_auprc(yT, raw, w)] for w in W]
        )
        K = len(classes)
        for k, c in enumerate(classes):
            add("auroc", c, auc[k], *percentile_ci(boot[:, k]))
            add("auprc", c, ap[k], *percentile_ci(boot[:, K + k]))
        add("auroc", "macro", np.nanmean(auc), *percentile_ci(np.nanmean(boot[:, :K], axis=1)))
        add("auprc", "macro", np.nanmean(ap), *percentile_ci(np.nanmean(boot[:, K:], axis=1)))
        add("temperature", "shared", T)
        for k, c in enumerate(classes):
            add("temperature", c, Tk[k])
        for tag, prob, logits in (
            ("raw", raw, zT),
            ("T", pr["TEST"], zT / T),
            ("Tk", cal_k, zT / Tk),
        ):
            ece = ece_per_class(prob, yT, int(ev["ece_bins"]))
            for k, c in enumerate(classes):
                add(f"ece_{tag}", c, ece[k])
            add(f"ece_{tag}", "macro", ece.mean())
            add(f"brier_{tag}", "all", brier(yT, prob))
            add(f"nll_{tag}", "all", bce_nll(logits, yT))
        k_norm, abn_A = triage_labels(d["y_VAL_A"], classes)
        _, abn_T = triage_labels(yT, classes)
        sA, sT = 1.0 - pr["VAL_A"][:, k_norm], 1.0 - pr["TEST"][:, k_norm]
        t = threshold_at_specificity(sA[~abn_A], float(ev["triage_specificity"]))
        sens, spec = sens_spec(sT, abn_T, t)
        add("triage_auroc", "abnormal", per_class_auroc(abn_T[:, None].astype(int), sT[:, None])[0])
        add("triage_sens_at_spec", "abnormal", sens)
        add("triage_spec_realized", "abnormal", spec)
        for method in cp["methods"]:
            for alpha in cp["alphas"]:
                c_ = calibrate(method, pr["VAL_B"], d["y_VAL_B"], float(alpha))
                has1, has0 = predict_sets(c_, pr["TEST"])
                met = set_metrics(has1, has0, yT)
                for k, c in enumerate(classes):
                    n1 = int(c_.n1[k])
                    conf_rows.append(
                        {
                            "seed": seed,
                            "method": method,
                            "alpha": float(alpha),
                            "class": c,
                            "cov1": met["cov1"][k],
                            "cov0": met["cov0"][k],
                            "abstain": met["abstain"][k],
                            "empty": met["empty"][k],
                            "decisive_acc": met["decisive_acc"][k],
                            "n1_cal": n1,
                            "n0_cal": int(c_.n0[k]),
                            "q1": c_.q1[k],
                            "q0": c_.q0[k],
                            "n1_test": int(yT[:, k].sum()),
                            "cov1_upper_theory": 1 - float(alpha) + 1 / (n1 + 1),
                        }
                    )
                conf_rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "alpha": float(alpha),
                        "class": "ALL",
                        "labels_per_record": float(met["labels_per_record"]),
                        "all_pos_covered": float(met["all_pos_covered"]),
                    }
                )
        log(f"seed {seed}: T={T:.3f}, macro-AUROC TEST={np.nanmean(auc):.4f}")
    out_dir = run_dir / "e2"
    out_dir.mkdir(parents=True, exist_ok=True)
    cls_df, conf_df = pd.DataFrame(cls_rows), pd.DataFrame(conf_rows)
    cls_df.to_csv(out_dir / "classification_by_seed.csv", index=False)
    conf_df.to_csv(out_dir / "conformal_by_seed.csv", index=False)
    cls_sum = (
        cls_df.groupby(["metric", "class"])["value"].agg(["mean", "std", "count"]).reset_index()
    )
    cls_sum.to_csv(out_dir / "classification_summary.csv", index=False)
    num = [c for c in conf_df.columns if c not in ("seed", "method", "alpha", "class")]
    conf_sum = conf_df.groupby(["method", "alpha", "class"])[num].mean().reset_index()
    conf_sum.to_csv(out_dir / "conformal_summary_mean.csv", index=False)
    stage_record(run_dir, "e2", cfg, {"seeds": seeds})
    return {"classification": cls_sum, "conformal": conf_sum}


# ---------------------------------------------------------------- E3
def run_e3(cfg: RunConfig, log: Log = print) -> pd.DataFrame:
    """Distribusi coverage pada split acak berulang vs teori Beta-Binomial (+ PAC)."""
    p, vp = cfg.params, cfg.params["validity"]
    run_dir = prepare_run_dir(cfg, create=False)
    seeds = available_seeds(cfg)
    seed_model = int(vp["model_seed"]) if vp.get("model_seed") is not None else seeds[0]
    classes = list(p["data"]["classes"])
    log_test_access(
        cfg.paths.test_access_log,
        run_dir,
        train_hash(p),
        code_fingerprint(),
        f"E3 validitas split acak pada positif TEST, seed {seed_model}",
    )
    d = load_preds(run_dir, seed_model)
    _, _, pr = _calibrated(d)
    rows = []
    for k, c in enumerate(classes):
        scores = 1.0 - pr["TEST"][d["y_TEST"][:, k] == 1, k]
        N = scores.size
        n_cal = N // 2
        ties = N - np.unique(scores).size
        for alpha in vp["alphas"]:
            a = float(alpha)
            for rule, rank in (
                ("standard", conformal_rank(n_cal, a)),
                ("pac", pac_rank(n_cal, a, float(vp["pac_delta"]))),
            ):
                sim = simulate_split_coverage(
                    scores, n_cal, rank, int(vp["n_splits"]), np.random.default_rng(int(vp["seed"]))
                )
                res = compare_with_theory(
                    sim, a, np.random.default_rng(int(vp["seed"]) + 1), int(vp["n_mc"])
                )
                rows.append(
                    {
                        "class": c,
                        "alpha": a,
                        "rule": rule,
                        "n_pool": N,
                        "ties": ties,
                        "target": 1 - a,
                        **res,
                    }
                )
                log(
                    f"{c:5s} α={a:.2f} {rule:8s} n={n_cal} k={rank}: mean {res['emp_mean']:.4f} "
                    f"(teori {res['theory_mean']:.4f}), sd {res['emp_sd']:.4f} "
                    f"(teori {res['theory_sd']:.4f}), P(cov<1−α) {res['emp_p_below']:.3f} "
                    f"(teori {res['theory_p_below']:.3f}), KS p={res['ks_p']:.3f}"
                )
    df = pd.DataFrame(rows)
    (run_dir / "e3").mkdir(parents=True, exist_ok=True)
    df.to_csv(run_dir / "e3" / "validity.csv", index=False)
    stage_record(run_dir, "e3", cfg, {"model_seed": seed_model})
    return df


# ---------------------------------------------------------------- E4
def subgroup_table(cfg: RunConfig, ecg_ids: np.ndarray) -> pd.DataFrame:
    """Variabel subgrup per rekaman: sex, age_band, quality (noisy/clean)."""
    sp = cfg.params["subgroups"]
    meta = load_metadata(cfg.paths.raw_dir, cfg.params["data"]["classes"])
    raw = pd.read_csv(cfg.paths.raw_dir / "ptbxl_database.csv", index_col="ecg_id")
    raw.index = raw.index.astype(np.int64)
    qcols = list(sp["quality_columns"])
    noisy = raw[qcols].notna() & (raw[qcols].astype(str).apply(lambda s: s.str.strip()) != "")
    edges = list(sp["age_edges"])
    out = pd.DataFrame(index=pd.Index(ecg_ids, name="ecg_id"))
    out["sex"] = "sex=" + meta.loc[ecg_ids, "sex"].astype(int).astype(str).to_numpy()
    out["age_band"] = [age_band(a, edges) for a in meta.loc[ecg_ids, "age"]]
    out["quality"] = np.where(noisy.loc[ecg_ids].any(axis=1), "noisy", "clean")
    return out


def run_e4(cfg: RunConfig, log: Log = print) -> pd.DataFrame:
    """Coverage conformal per subgrup: Mondrian marginal (M2, M3) vs per-subgrup (M2-group)."""
    p, sp = cfg.params, cfg.params["subgroups"]
    ev = p["evaluation"]
    run_dir = prepare_run_dir(cfg, create=False)
    seeds = available_seeds(cfg)
    classes = list(p["data"]["classes"])
    alpha = float(sp["alpha"])
    log_test_access(
        cfg.paths.test_access_log,
        run_dir,
        train_hash(p),
        code_fingerprint(),
        f"E4 coverage subgrup TEST α={alpha}, seeds {seeds}",
    )
    rows: list[dict[str, Any]] = []
    for i_seed, seed in enumerate(seeds):
        d = load_preds(run_dir, seed)
        _, _, pr = _calibrated(d)
        yB, yT = d["y_VAL_B"].astype(bool), d["y_TEST"].astype(bool)
        gB, gT = subgroup_table(cfg, d["ecg_id_VAL_B"]), subgroup_table(cfg, d["ecg_id_TEST"])
        c2, c3 = calibrate("M2", pr["VAL_B"], yB, alpha), calibrate("M3", pr["VAL_B"], yB, alpha)
        h2, _ = predict_sets(c2, pr["TEST"])
        h31, h30 = predict_sets(c3, pr["TEST"])
        W = None
        if i_seed == 0:
            W = patient_bootstrap_weights(
                d["patient_id_TEST"],
                int(ev["bootstrap_resamples"]),
                np.random.default_rng(int(ev["bootstrap_seed"])),
            )
        for var in ("sex", "age_band", "quality"):
            for g in sorted(gT[var].unique()):
                inB, inT = (gB[var] == g).to_numpy(), (gT[var] == g).to_numpy()
                cg = calibrate("M2", pr["VAL_B"][inB], yB[inB], alpha)
                hg, _ = predict_sets(cg, pr["TEST"])
                for k, c in enumerate(classes):
                    pos = yT[:, k] & inT
                    neg = ~yT[:, k] & inT
                    row: dict[str, Any] = {
                        "seed": seed,
                        "variable": var,
                        "group": g,
                        "class": c,
                        "n_test_pos": int(pos.sum()),
                        "n_cal_pos": int((yB[:, k] & inB).sum()),
                        "too_small": bool(pos.sum() < int(sp["min_positives"])),
                        "cov1_M2": h2[pos, k].mean() if pos.any() else np.nan,
                        "cov1_M2group": hg[pos, k].mean() if pos.any() else np.nan,
                        "q1_M2": c2.q1[k],
                        "q1_M2group": cg.q1[k],
                        "flag_rate_M2_neg": h2[neg, k].mean() if neg.any() else np.nan,
                        "flag_rate_M2group_neg": hg[neg, k].mean() if neg.any() else np.nan,
                        "cov1_M3": h31[pos, k].mean() if pos.any() else np.nan,
                        "cov0_M3": h30[neg, k].mean() if neg.any() else np.nan,
                        "abstain_M3": (h31[inT, k] & h30[inT, k]).mean() if inT.any() else np.nan,
                    }
                    if W is not None and pos.any():
                        for tag, hit in (("M2", h2[:, k]), ("M2group", hg[:, k])):
                            lo, hi = percentile_ci(weighted_rate(hit, pos, W))
                            row[f"cov1_{tag}_ci_lo"], row[f"cov1_{tag}_ci_hi"] = lo, hi
                    rows.append(row)
        log(f"seed {seed}: subgrup selesai")
    df = pd.DataFrame(rows)
    out_dir = run_dir / "e4"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "subgroups_by_seed.csv", index=False)
    keys = ["variable", "group", "class"]
    num = [c for c in df.columns if c.startswith(("cov", "flag", "abstain")) and "_ci_" not in c]
    summ = (
        df.groupby(keys)
        .agg(
            n_test_pos=("n_test_pos", "first"),
            n_cal_pos=("n_cal_pos", "first"),
            too_small=("too_small", "first"),
            **{f"{c}_mean": (c, "mean") for c in num},
            **{f"{c}_sd": (c, "std") for c in ("cov1_M2", "cov1_M2group")},
        )
        .reset_index()
    )
    ci_cols = [c for c in df.columns if "_ci_" in c]
    if ci_cols:  # CI bootstrap hanya untuk seed pertama
        first = df[df["seed"] == seeds[0]][keys + ci_cols]
        summ = summ.merge(
            first.rename(columns={c: f"{c}_seed{seeds[0]}" for c in ci_cols}), on=keys, how="left"
        )
    summ.to_csv(out_dir / "subgroups_summary.csv", index=False)
    stage_record(run_dir, "e4", cfg, {"seeds": seeds, "alpha": alpha})
    return summ


# ---------------------------------------------------------------- E5
E5_DEFAULTS: dict[str, Any] = {
    "methods": ["none", "temp", "temp_k", "platt", "isotonic", "hist"],
    "n_bins": 15,
    "ref_alpha": 0.10,
}


def run_e5(cfg: RunConfig, log: Log = print) -> pd.DataFrame:
    """Pembanding kalibrasi post-hoc (fit VAL_A, evaluasi TEST) + uji invariansi conformal.

    Untuk setiap kalibrator: ECE lebar-sama dan massa-sama, Brier + dekomposisi
    (REL, RES, UNC, WBV, WBC), NLL, AUROC. Lalu M1 (ambang gabungan) dan M2 (Mondrian per
    kelas) dikalibrasi di VAL_B dengan probabilitas kalibrator itu: M2 seharusnya invarian
    (transformasi tak-turun per kelas), M1 tidak.
    """
    from ecgdig.metrics.calibration import brier_decomposition
    from ecgdig.models.calibrators import Calibrator

    p = cfg.params
    cp = {**E5_DEFAULTS, **p.get("calibration", {})}
    run_dir = prepare_run_dir(cfg, create=False)
    seeds = available_seeds(cfg)
    classes = list(p["data"]["classes"])
    nb, a_ref = int(cp["n_bins"]), float(cp["ref_alpha"])
    log_test_access(
        cfg.paths.test_access_log,
        run_dir,
        train_hash(p),
        code_fingerprint(),
        f"E5 pembanding kalibrasi TEST, seeds {seeds}",
    )
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        d = load_preds(run_dir, seed)
        zA, yA = d["logits_VAL_A"], d["y_VAL_A"].astype(np.float64)
        zB, yB = d["logits_VAL_B"], d["y_VAL_B"]
        zT, yT = d["logits_TEST"], d["y_TEST"].astype(np.int64)
        for method in cp["methods"]:
            cal = Calibrator(method).fit(zA, yA, n_bins=nb)
            pT, pB = cal.transform(zT), cal.transform(zB)

            def add(
                metric: str, cls: str, value: float, method: str = method, seed: int = seed
            ) -> None:
                rows.append(
                    {
                        "seed": seed,
                        "calibrator": method,
                        "class": cls,
                        "metric": metric,
                        "value": float(value),
                    }
                )

            ew = ece_per_class(pT, yT, nb, "width")
            em = ece_per_class(pT, yT, nb, "mass")
            auc = per_class_auroc(yT, pT)
            nll_k = -np.mean(yT * np.log(pT) + (1 - yT) * np.log(1 - pT), axis=0)
            for k, c in enumerate(classes):
                dec = brier_decomposition(pT[:, k], yT[:, k], nb)
                for name, val in dec.items():
                    add(name, c, val)
                add("ece_width", c, ew[k])
                add("ece_mass", c, em[k])
                add("auroc", c, auc[k])
                add("nll", c, nll_k[k])
            add("ece_width", "macro", ew.mean())
            add("ece_mass", "macro", em.mean())
            add("auroc", "macro", np.nanmean(auc))
            add("nll", "macro", nll_k.mean())
            add("bs", "macro", np.mean((pT - yT) ** 2))
            for cm in ("M1", "M2"):
                has1, has0 = predict_sets(calibrate(cm, pB, yB, a_ref), pT)
                met = set_metrics(has1, has0, yT)
                for k, c in enumerate(classes):
                    add(f"cov1_{cm}", c, met["cov1"][k])
                add(f"labels_per_record_{cm}", "ALL", float(met["labels_per_record"]))
        log(f"seed {seed}: {len(cp['methods'])} kalibrator dievaluasi")
    df = pd.DataFrame(rows)
    out = run_dir / "e5"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "calibration_by_seed.csv", index=False)
    summ = (
        df.groupby(["calibrator", "class", "metric"])["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summ.to_csv(out / "calibration_summary.csv", index=False)
    macro = summ[summ["class"].isin(["macro", "ALL"])].pivot_table(
        index="calibrator", columns="metric", values="mean"
    )
    log(macro.round(4).to_string())
    stage_record(run_dir, "e5", cfg, {"seeds": seeds, **cp})
    return summ


# ---------------------------------------------------------------- E6
E6_DEFAULTS: dict[str, Any] = {"alphas": [0.05, 0.10, 0.20], "pac_delta": 0.10}
E6_METHODS = ("M0_0.5", "F1_valA", "M1", "M2", "M2_PAC", "M3", "CRC_FNR")


def _e6_sets(
    method: str, pr: dict[str, np.ndarray], d: dict[str, np.ndarray], alpha: float, delta: float
) -> tuple[np.ndarray, np.ndarray | None, dict[str, float]]:
    """(has1 TEST, has0 TEST atau None, info) untuk satu metode."""
    from ecgdig.models.set_predictors import crc_lambda, crc_sets, f1_thresholds, pac_thresholds

    pA, pB, pT = pr["VAL_A"], pr["VAL_B"], pr["TEST"]
    yA, yB = d["y_VAL_A"], d["y_VAL_B"]
    if method == "M0_0.5":
        return pT >= 0.5, None, {}
    if method == "F1_valA":
        return pT >= f1_thresholds(pA, yA)[None, :], None, {}
    if method in ("M1", "M2", "M3"):
        c = calibrate(method, pB, yB, alpha)
        h1, h0 = predict_sets(c, pT)
        return h1, (h0 if method == "M3" else None), {}
    if method == "M2_PAC":
        q = pac_thresholds(pB, yB, alpha, delta)
        return (1.0 - pT) <= q[None, :], None, {}
    if method == "CRC_FNR":
        lam = crc_lambda(pB, yB, alpha)
        return crc_sets(pT, lam), None, {"lambda": lam, "threshold_p": 1.0 - lam}
    raise ValueError(method)


def run_e6(cfg: RunConfig, log: Log = print) -> pd.DataFrame:
    """Pembanding prediktor himpunan label di TEST (kalibrasi VAL_B; ambang F1 di VAL_A).

    Metrik per kelas: sensitivitas P(k ∈ C | y_k = 1) dan laju flag pada negatif.
    Metrik per rekaman: FNR pasien (fraksi label benar yang terlewat), P(semua label benar
    tercakup), rerata |C|, laju himpunan kosong. M3: laju "ragu, rujuk" per kelas.
    CI 95% bootstrap pasien untuk seed pertama (FNR dan |C|).
    """
    from ecgdig.models.set_predictors import fnr_loss

    p = cfg.params
    cp = {**E6_DEFAULTS, **p.get("conformal_benchmark", {})}
    ev = p["evaluation"]
    run_dir = prepare_run_dir(cfg, create=False)
    seeds = available_seeds(cfg)
    classes = list(p["data"]["classes"])
    log_test_access(
        cfg.paths.test_access_log,
        run_dir,
        train_hash(p),
        code_fingerprint(),
        f"E6 pembanding prediktor himpunan TEST, seeds {seeds}",
    )
    rows: list[dict[str, Any]] = []
    for i_seed, seed in enumerate(seeds):
        d = load_preds(run_dir, seed)
        _, _, pr = _calibrated(d)
        yT = d["y_TEST"].astype(bool)
        W = None
        if i_seed == 0:
            W = patient_bootstrap_weights(
                d["patient_id_TEST"],
                int(ev["bootstrap_resamples"]),
                np.random.default_rng(int(ev["bootstrap_seed"])),
            )
        for alpha in [float(a) for a in cp["alphas"]]:
            for method in E6_METHODS:
                h1, h0, info = _e6_sets(method, pr, d, alpha, float(cp["pac_delta"]))
                fnr = fnr_loss(h1, yT)
                size = h1.sum(axis=1).astype(np.float64)
                base = {"seed": seed, "alpha": alpha, "method": method}
                rec: dict[str, Any] = {
                    **base,
                    "class": "ALL",
                    "fnr_mean": fnr.mean(),
                    "all_pos_covered": float(np.mean(fnr == 0)),
                    "labels_per_record": size.mean(),
                    "empty_rate": float(np.mean(size == 0)),
                    **info,
                }
                if W is not None:
                    w = W.astype(np.float64)
                    tot = w.sum(axis=1)
                    rec["fnr_ci_lo"], rec["fnr_ci_hi"] = percentile_ci((w @ fnr) / tot)
                    rec["size_ci_lo"], rec["size_ci_hi"] = percentile_ci((w @ size) / tot)
                rows.append(rec)
                for k, c in enumerate(classes):
                    pos, neg = yT[:, k], ~yT[:, k]
                    r = {
                        **base,
                        "class": c,
                        "sensitivity": h1[pos, k].mean(),
                        "flag_rate_neg": h1[neg, k].mean(),
                    }
                    if h0 is not None:
                        r["abstain"] = float(np.mean(h1[:, k] & h0[:, k]))
                        r["cov0"] = h0[neg, k].mean()
                    rows.append(r)
        log(f"seed {seed}: {len(E6_METHODS)} metode × {len(cp['alphas'])} α")
    df = pd.DataFrame(rows)
    out = run_dir / "e6"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "sets_by_seed.csv", index=False)
    num = [
        c for c in df.columns if c not in ("seed", "alpha", "method", "class") and "_ci_" not in c
    ]
    summ = df.groupby(["alpha", "method", "class"])[num].mean().reset_index()
    summ.to_csv(out / "sets_summary.csv", index=False)
    show = summ[summ["class"] == "ALL"][
        ["alpha", "method", "fnr_mean", "all_pos_covered", "labels_per_record", "empty_rate"]
    ]
    log(show.round(4).to_string(index=False))
    stage_record(run_dir, "e6", cfg, {"seeds": seeds, **cp})
    return summ


# ---------------------------------------------------------------- compare (benchmark E7)
# Acuan literatur: README repo/ecg_ptbxl_benchmarking (clone lokal), tabel
# "4. PTB-XL: Diagnostic superclasses", macro-AUC fold 10; angka dalam kurung ditulis apa adanya.
LITERATURE_SUPERDIAGNOSTIC = (
    ("resnet1d_wang", "0.930(05)"),
    ("xresnet1d101", "0.928(05)"),
    ("lstm", "0.927(05)"),
    ("fcn_wang", "0.925(06)"),
    ("inception1d", "0.921(06)"),
    ("lstm_bidir", "0.921(06)"),
    ("Wavelet+NN", "0.874(07)"),
)


def run_compare(cfgs: list[RunConfig], out_dir: Path, log: Log = print) -> pd.DataFrame:
    """Tabel benchmark lintas run (hanya membaca hasil E1/E2/E6 yang sudah ada)."""
    rows: list[dict[str, Any]] = []
    for cfg in cfgs:
        rd = cfg.run_dir
        row: dict[str, Any] = {
            "run": rd.name,
            "model": cfg.params["model"]["name"],
            "train_hash": train_hash(cfg.params)[:12],
        }
        if not (rd / "e2" / "classification_by_seed.csv").exists():
            log(f"[compare] {rd.name}: E2 belum ada, dilewati")
            continue
        c = pd.read_csv(rd / "e2" / "classification_by_seed.csv")
        for metric, cls, name in (
            ("auroc", "macro", "macro_auroc"),
            ("auprc", "macro", "macro_auprc"),
            ("ece_T", "macro", "macro_ece_T"),
            ("brier_T", "all", "brier_T"),
            ("triage_auroc", "abnormal", "triage_auroc"),
        ):
            v = c[(c["metric"] == metric) & (c["class"] == cls)]["value"]
            row[f"{name}_mean"], row[f"{name}_sd"] = v.mean(), v.std()
        row["n_seeds"] = int(c["seed"].nunique())
        eff = rd / "e1" / "efficiency.json"
        if eff.exists():
            e = read_json(eff)
            row.update(
                n_parameters=e["n_parameters"],
                size_mb=e["state_dict_bytes"] / 1e6,
                cpu_ms_1thread=e["cpu_latency_ms_1thread"],
            )
        s6 = rd / "e6" / "sets_summary.csv"
        if s6.exists():
            s = pd.read_csv(s6)
            s = s[(s["class"] == "ALL") & (np.isclose(s["alpha"], 0.10))].set_index("method")
            for m in ("M2", "CRC_FNR"):
                if m in s.index:
                    row[f"{m}_a0.10_labels"] = s.loc[m, "labels_per_record"]
                    row[f"{m}_a0.10_fnr"] = s.loc[m, "fnr_mean"]
        rows.append(row)
    for name, auc in LITERATURE_SUPERDIAGNOSTIC:
        rows.append({"run": f"[literatur] {name}", "model": name, "macro_auroc_reported": auc})
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "benchmark.csv", index=False)
    log(df.round(4).to_string(index=False))
    return df
