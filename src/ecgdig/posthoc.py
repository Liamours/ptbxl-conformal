"""Analisis post-hoc S1–S4 dari artefak run yang sudah ada (tanpa melatih ulang, tanpa tuning).

Membaca ``config_resolved.yaml`` dan ``e1/seed_*/preds.npz`` milik run (logit VAL_A/VAL_B/TEST),
memakai probabilitas yang sama dengan E2 (temperature bersama T dari VAL_A), lalu:

S1  Coverage positif per kelas untuk M0, M1, M2, M3, M2-PAC di semua α config: X/m, CI
    Clopper–Pearson, p-value binomial eksak P(Bin(m, 1−α) ≤ X), p-value marginal
    P(BetaBin(m, k, n+1−k) ≤ X) (M2, M3, M2-PAC), ukuran himpunan, himpunan kosong/singleton,
    laju flag pada negatif. Lihat ``eval/inference.py``.
S2  Ukuran himpunan label per rekaman |Ĉ(x)| = #{k : 1 ∈ C_k}: M2 vs M1 dan M2 vs M0
    (rerata selisih, CI bootstrap pasien, Wilcoxon, uji-t berpasangan, Holm per seed atas
    keluarga |α| × 2 perbandingan).
S3  E3 diulang dengan seed RNG yang sama (split identik, per rekaman) dan dilaporkan sebagai
    hitungan X: rerata/varians empiris vs Beta-Binomial, uji χ² (sel digabung, harapan ≥ 5).
S4  Tabel subgrup E4 (tanpa akses TEST baru) dengan label jenis kelamin:
    0 = laki-laki, 1 = perempuan (Wagner dkk., Sci Data 7:154, 2020, Tabel 2).

Setiap akses TEST dicatat di log akses TEST.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ecgdig.config import code_fingerprint
from ecgdig.eval.bootstrap import patient_bootstrap_weights
from ecgdig.eval.inference import (
    betabinom_moments,
    betabinom_p_lower,
    binom_p_lower,
    chisq_gof,
    clopper_pearson,
    holm,
    paired_tests,
)
from ecgdig.eval.validity import pac_rank, simulate_split_coverage, theory_distribution
from ecgdig.experiments import _calibrated, load_preds, train_hash
from ecgdig.models.conformal import calibrate, conformal_rank, predict_sets
from ecgdig.models.set_predictors import pac_thresholds
from ecgdig.runlog import env_info, log_test_access, now_iso, read_json, write_json

S1_METHODS = ("M0", "M1", "M2", "M3", "M2_PAC")
SEX_LABEL = {"sex=0": "male", "sex=1": "female"}  # Wagner dkk. 2020, Tabel 2


def run_seeds(run_dir: Path) -> list[int]:
    """Seed yang punya preds.npz, urut naik."""
    return sorted(int(p.parent.name.split("_")[1]) for p in run_dir.glob("e1/seed_*/preds.npz"))


def method_sets(
    method: str, pB: np.ndarray, yB: np.ndarray, pT: np.ndarray, alpha: float, delta: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(has1, has0, rank per kelas, ambang q1 pada skor 1−p per kelas) untuk satu metode."""
    K = pB.shape[1]
    nB = yB.astype(bool).sum(axis=0)
    if method == "M2_PAC":
        q = pac_thresholds(pB, yB, alpha, delta)
        has1 = (1.0 - pT) <= q[None, :]
        rank = np.array([pac_rank(int(n), alpha, delta) for n in nB], dtype=float)
        return has1, ~has1, rank, q
    cal = calibrate(method, pB, yB, alpha)
    has1, has0 = predict_sets(cal, pT)
    if method == "M0":
        return has1, has0, np.full(K, np.nan), np.full(K, 0.5)
    if method == "M1":
        return has1, has0, np.full(K, float(conformal_rank(int(nB.sum()), alpha))), cal.q1
    rank = np.array([conformal_rank(int(n), alpha) for n in nB], dtype=float)
    return has1, has0, rank, cal.q1


def class_rows(
    has1: np.ndarray,
    has0: np.ndarray,
    yT: np.ndarray,
    nB: np.ndarray,
    rank: np.ndarray,
    q1: np.ndarray,
    alpha: float,
    method: str,
    classes: list[str],
) -> list[dict[str, Any]]:
    """Baris S1 per kelas untuk satu (seed, metode, α)."""
    rows = []
    size_k = has1.astype(int) + has0.astype(int)  # |C_k| ∈ {0, 1, 2}
    single = size_k == 1
    correct = np.where(yT, has1, has0)
    for k, c in enumerate(classes):
        pos, neg = yT[:, k], ~yT[:, k]
        m, x = int(pos.sum()), int(has1[pos, k].sum())
        lo, hi = clopper_pearson(x, m)
        r = int(rank[k]) if np.isfinite(rank[k]) else None
        bb = method in ("M2", "M3", "M2_PAC") and r is not None
        mean_bb, var_bb = betabinom_moments(m, int(nB[k]), r) if bb and r else (np.nan, np.nan)
        rows.append(
            {
                "method": method,
                "alpha": alpha,
                "class": c,
                "n_cal_pos": int(nB[k]),
                "n_cal_used": int(nB.sum()) if method == "M1" else int(nB[k]),
                "m_test_pos": m,
                "rank_k": r,
                "q1_score_threshold": float(q1[k]),
                "covered_x": x,
                "coverage": x / m if m else np.nan,
                "cp95_lo": lo,
                "cp95_hi": hi,
                "p_binom_lower": binom_p_lower(x, m, 1.0 - alpha),
                "p_betabinom_lower": (
                    betabinom_p_lower(x, m, int(nB[k]), r) if bb and r else np.nan
                ),
                "betabinom_mean_x": mean_bb,
                "betabinom_var_x": var_bb,
                "mean_binary_set_size": float(size_k[:, k].mean()),
                "empty_rate": float(np.mean(size_k[:, k] == 0)),
                "singleton_rate": float(single[:, k].mean()),
                "singleton_accuracy": (
                    float(correct[single[:, k], k].mean()) if single[:, k].any() else np.nan
                ),
                "both_rate": float(np.mean(size_k[:, k] == 2)),
                "neg_flag_rate": float(has1[neg, k].mean()),
                "neg_coverage": float(has0[neg, k].mean()),
            }
        )
    return rows


def record_row(has1: np.ndarray, yT: np.ndarray, method: str, alpha: float) -> dict[str, Any]:
    """Baris S1 tingkat rekaman: ukuran himpunan label |Ĉ(x)| dan turunannya."""
    size = has1.sum(axis=1)
    one = size == 1
    n_pos = yT.sum(axis=1)
    missed = (yT & ~has1).sum(axis=1)
    return {
        "method": method,
        "alpha": alpha,
        "class": "ALL",
        "mean_label_set_size": float(size.mean()),
        "label_set_empty_rate": float(np.mean(size == 0)),
        "label_set_singleton_rate": float(one.mean()),
        "singleton_label_is_true": float((has1 & yT)[one].any(axis=1).mean())
        if one.any()
        else np.nan,
        "singleton_equals_truth": float(np.all(has1 == yT, axis=1)[one].mean())
        if one.any()
        else np.nan,
        "all_pos_covered": float(np.mean(missed == 0)),
        "fnr_mean": float(np.mean(np.where(n_pos > 0, missed / np.maximum(n_pos, 1), 0.0))),
    }


def s1_s2(run_dir: Path, cfg: dict[str, Any], log: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    """S1 (coverage per kelas) dan S2 (uji berpasangan ukuran himpunan) untuk semua seed."""
    classes = list(cfg["data"]["classes"])
    alphas = [float(a) for a in cfg["conformal"]["alphas"]]
    delta = float(cfg.get("conformal_benchmark", {}).get("pac_delta", 0.10))
    ev = cfg["evaluation"]
    seeds = run_seeds(run_dir)
    ref = load_preds(run_dir, seeds[0])
    W = patient_bootstrap_weights(
        ref["patient_id_TEST"],
        int(ev["bootstrap_resamples"]),
        np.random.default_rng(int(ev["bootstrap_seed"])),
    )
    s1: list[dict[str, Any]] = []
    s2: list[dict[str, Any]] = []
    for seed in seeds:
        d = load_preds(run_dir, seed)
        if not np.array_equal(d["ecg_id_TEST"], ref["ecg_id_TEST"]):
            raise RuntimeError("Urutan TEST berbeda antar-seed.")
        _, _, pr = _calibrated(d)
        yB, yT = d["y_VAL_B"].astype(bool), d["y_TEST"].astype(bool)
        nB = yB.sum(axis=0)
        fam: list[dict[str, Any]] = []
        for alpha in alphas:
            sizes: dict[str, np.ndarray] = {}
            for method in S1_METHODS:
                h1, h0, rank, q1 = method_sets(method, pr["VAL_B"], yB, pr["TEST"], alpha, delta)
                for r in class_rows(h1, h0, yT, nB, rank, q1, alpha, method, classes):
                    s1.append({"seed": seed, **r})
                s1.append({"seed": seed, **record_row(h1, yT, method, alpha)})
                sizes[method] = h1.sum(axis=1).astype(np.float64)
            for other in ("M1", "M0"):
                res = paired_tests(sizes["M2"], sizes[other], W)
                fam.append({"seed": seed, "alpha": alpha, "comparison": f"M2-{other}", **res})
        for key in ("wilcoxon_p", "t_p"):
            adj = holm(np.array([r[key] for r in fam]))
            for r, a in zip(fam, adj, strict=True):
                r[f"{key}_holm"] = float(a)
        s2.extend(fam)
        log(f"seed {seed}: S1/S2 selesai")
    return pd.DataFrame(s1), pd.DataFrame(s2)


def summarize_s1(s1: pd.DataFrame, ref_seed: int) -> pd.DataFrame:
    """Ringkasan antar-seed per (metode, α, kelas); uji dan CI eksak ditampilkan untuk ref_seed."""
    cls = s1[s1["class"] != "ALL"]
    keys = ["method", "alpha", "class"]
    g = cls.groupby(keys, sort=False)
    out = g.agg(
        n_cal_pos=("n_cal_pos", "first"),
        n_cal_used=("n_cal_used", "first"),
        m_test_pos=("m_test_pos", "first"),
        rank_k=("rank_k", "first"),
        coverage_mean=("coverage", "mean"),
        coverage_sd=("coverage", "std"),
        coverage_min=("coverage", "min"),
        coverage_max=("coverage", "max"),
        p_binom_min=("p_binom_lower", "min"),
        p_binom_max=("p_binom_lower", "max"),
        n_seeds_p_binom_lt_0_05=("p_binom_lower", lambda p: int(np.sum(p < 0.05))),
        p_betabinom_min=("p_betabinom_lower", "min"),
        p_betabinom_max=("p_betabinom_lower", "max"),
        mean_binary_set_size=("mean_binary_set_size", "mean"),
        empty_rate=("empty_rate", "mean"),
        singleton_rate=("singleton_rate", "mean"),
        singleton_accuracy=("singleton_accuracy", "mean"),
        both_rate=("both_rate", "mean"),
        neg_flag_rate=("neg_flag_rate", "mean"),
        n_seeds=("seed", "nunique"),
    ).reset_index()
    r = cls[cls["seed"] == ref_seed][
        keys + ["covered_x", "coverage", "cp95_lo", "cp95_hi", "p_binom_lower", "p_betabinom_lower"]
    ].rename(columns=lambda c: c if c in keys else f"{c}_seed{ref_seed}")
    return out.merge(r, on=keys, how="left")


def summarize_records(s1: pd.DataFrame) -> pd.DataFrame:
    """Rerata antar-seed metrik tingkat rekaman (class = ALL)."""
    rec = s1[s1["class"] == "ALL"].dropna(axis=1, how="all")
    num = [c for c in rec.columns if c not in ("seed", "method", "alpha", "class")]
    agg = rec.groupby(["method", "alpha"], sort=False)[num].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reset_index()


def summarize_s2(s2: pd.DataFrame, ref_seed: int) -> pd.DataFrame:
    """Rerata selisih antar-seed + hasil lengkap seed acuan."""
    g = s2.groupby(["alpha", "comparison"], sort=False)
    out = g.agg(
        mean_diff_mean=("mean_diff", "mean"),
        mean_diff_sd=("mean_diff", "std"),
        wilcoxon_p_holm_max=("wilcoxon_p_holm", "max"),
        t_p_holm_max=("t_p_holm", "max"),
        n_seeds=("seed", "nunique"),
    ).reset_index()
    r = s2[s2["seed"] == ref_seed].drop(columns="seed")
    r = r.rename(columns=lambda c: c if c in ("alpha", "comparison") else f"{c}_seed{ref_seed}")
    return out.merge(r, on=["alpha", "comparison"], how="left")


def s3(run_dir: Path, cfg: dict[str, Any], log: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    """E3 diulang (RNG sama) dalam bentuk hitungan X; dibandingkan dengan e3/validity.csv."""
    vp = cfg["validity"]
    classes = list(cfg["data"]["classes"])
    seeds = run_seeds(run_dir)
    seed_model = int(vp["model_seed"]) if vp.get("model_seed") is not None else seeds[0]
    d = load_preds(run_dir, seed_model)
    _, _, pr = _calibrated(d)
    e3_path = run_dir / "e3" / "validity.csv"
    e3 = pd.read_csv(e3_path) if e3_path.exists() else None
    rows, raw = [], []
    for k, c in enumerate(classes):
        scores = 1.0 - pr["TEST"][d["y_TEST"][:, k] == 1, k]
        N = scores.size
        n_cal = N // 2
        ties = N - np.unique(scores).size
        for alpha in [float(a) for a in vp["alphas"]]:
            for rule, rank in (
                ("standard", conformal_rank(n_cal, alpha)),
                ("pac", pac_rank(n_cal, alpha, float(vp["pac_delta"]))),
            ):
                sim = simulate_split_coverage(
                    scores, n_cal, rank, int(vp["n_splits"]), np.random.default_rng(int(vp["seed"]))
                )
                m, xs = sim.n_test, sim.counts
                mean_t, var_t = betabinom_moments(m, n_cal, rank)
                dist = theory_distribution(n_cal, m, rank)
                gof = (
                    chisq_gof(xs, dist.pmf(np.arange(m + 1)))
                    if dist is not None
                    else {"chi2": np.nan, "df": 0, "p": np.nan, "n_bins": 0, "bins": ""}
                )
                below = int(np.ceil(m * (1.0 - alpha) - 1e-9)) - 1
                match = np.nan
                if e3 is not None:
                    e = e3[
                        (e3["class"] == c) & np.isclose(e3["alpha"], alpha) & (e3["rule"] == rule)
                    ]
                    if len(e):
                        match = bool(abs(float(e["emp_mean"].iloc[0]) - xs.mean() / m) < 1e-12)
                rows.append(
                    {
                        "model_seed": seed_model,
                        "class": c,
                        "alpha": alpha,
                        "rule": rule,
                        "n_pool": N,
                        "ties_in_pool": ties,
                        "n_cal": n_cal,
                        "m_test": m,
                        "rank_k": rank,
                        "n_splits": xs.size,
                        "x_mean_obs": float(xs.mean()),
                        "x_mean_theory": mean_t,
                        "x_var_obs": float(xs.var(ddof=1)),
                        "x_var_theory": var_t,
                        "p_cov_below_target_obs": float(np.mean(xs <= below)),
                        "p_cov_below_target_theory": float(dist.cdf(below))
                        if dist and below >= 0
                        else 0.0,
                        **{f"chi2_{key}": v for key, v in gof.items()},
                        "matches_e3_validity_csv": match,
                    }
                )
                raw.append(
                    pd.DataFrame(
                        {
                            "class": c,
                            "alpha": alpha,
                            "rule": rule,
                            "split": np.arange(xs.size),
                            "x": xs,
                        }
                    )
                )
        log(f"S3 {c}: N={N}, n={n_cal}, m={N - n_cal}, ties={ties}")
    return pd.DataFrame(rows), pd.concat(raw, ignore_index=True)


def s4(run_dir: Path) -> pd.DataFrame:
    """Tabel E4 dengan label subgrup yang bisa dibaca (tidak menyentuh TEST lagi)."""
    s = pd.read_csv(run_dir / "e4" / "subgroups_summary.csv")
    s.insert(2, "group_label", s["group"].map(lambda g: SEX_LABEL.get(g, g)))
    return s


def run_posthoc(run_dir: Path, out: Path, test_log: Path, log: Any = print) -> dict[str, Path]:
    """Jalankan S1–S4 untuk satu run dan tulis CSV ke ``out``."""
    cfg = yaml.safe_load((run_dir / "config_resolved.yaml").read_text(encoding="utf-8"))
    man = read_json(run_dir / "manifest.json")
    th = train_hash(cfg)
    if th != man["train_hash"]:
        raise RuntimeError("config_resolved.yaml tidak cocok dengan manifest run (train_hash).")
    seeds = run_seeds(run_dir)
    ref_seed = 42 if 42 in seeds else seeds[0]
    log_test_access(
        test_log,
        run_dir,
        th,
        code_fingerprint(),
        f"posthoc S1-S3 (statistik dari logit tersimpan, tanpa latih ulang), seeds {seeds}",
    )
    out.mkdir(parents=True, exist_ok=True)
    s1, s2 = s1_s2(run_dir, cfg, log)
    v, raw = s3(run_dir, cfg, log)
    files = {
        "s1_coverage_by_seed": s1[s1["class"] != "ALL"].dropna(axis=1, how="all"),
        "s1_coverage_summary": summarize_s1(s1, ref_seed),
        "s1_label_sets_by_seed": s1[s1["class"] == "ALL"].dropna(axis=1, how="all"),
        "s1_label_sets_summary": summarize_records(s1),
        "s2_set_size_paired_by_seed": s2,
        "s2_set_size_paired_summary": summarize_s2(s2, ref_seed),
        "s3_validity_counts": v,
        "s3_validity_split_counts": raw,
        "s4_subgroups": s4(run_dir),
    }
    paths = {}
    for name, df in files.items():
        paths[name] = out / f"{name}.csv"
        df.to_csv(paths[name], index=False)
    write_json(
        out / "posthoc_provenance.json",
        {
            "run": run_dir.name,
            "train_hash": man["train_hash"],
            "run_code_fingerprint": man["code_fingerprint"],
            "posthoc_code_fingerprint": code_fingerprint(),
            "finished": now_iso(),
            "seeds": seeds,
            "reference_seed": ref_seed,
            "env": env_info(),
            "note": "Dihitung dari preds.npz tersimpan; tidak ada pelatihan ulang atau tuning.",
        },
    )
    return paths


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(prog="ecgdig.posthoc", description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--test-access-log", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None, help="default: <run-dir>/posthoc")
    args = ap.parse_args(argv)
    out = args.out or args.run_dir / "posthoc"
    for name, p in run_posthoc(args.run_dir, out, args.test_access_log).items():
        print(f"{name}: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
