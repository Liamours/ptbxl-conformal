"""Visualisasi hasil E1–E7: PNG (200 dpi) + PDF vektor + caption dari data.

Pemakaian:
    python -m ecgdig.viz --results <dir results> --profile colab --out <dir figures>

Gambar hanya membaca hasil run yang tercatat (``results/runs/<run>__<profile>/``); caption
disusun dari angka yang sama (tidak ada angka yang diketik tangan). Fig. 6 dan 8 menghitung
ulang kalibrator/simulasi dari ``preds.npz`` seed acuan (42 jika ada) dengan seed RNG dari
config run.

Gaya: palet acuan dataviz (mode terang). Slot kategori tetap per kelas
NORM/MI/STTC/CD/HYP (biru, oranye, aqua, kuning, magenta; lolos validator CVD untuk pasangan
berdekatan), tinta abu untuk teks, grid hairline solid, legenda untuk ≥ 2 seri.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

CLASSES = ("NORM", "MI", "STTC", "CD", "HYP")
SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
CLASS_COLOR = dict(zip(CLASSES, SLOTS, strict=False))
INK, INK2, MUTED, GRID, AXIS, NEUTRAL = (
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
    "#f0efec",
)
BLUE_RAMP = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")


@dataclass(frozen=True)
class Model:
    key: str
    run: str
    label: str


MODELS = (
    Model("main", "c0_resnet1d18_crop250_v1", "ResNet1D-18 + crop/TTA"),
    Model("base", "c0_resnet1d18_100hz_v1", "ResNet1D-18"),
    Model("small", "c0_resnet1d18_w16_v1", "ResNet1D-18 kecil (w=16)"),
    Model("feat", "feat194_logreg_v1", "194 fitur + regresi logistik"),
)
LITERATURE = (  # README repo/ecg_ptbxl_benchmarking, superdiagnostic fold 10 (terverifikasi)
    ("resnet1d_wang", 0.930),
    ("xresnet1d101", 0.928),
    ("lstm", 0.927),
    ("fcn_wang", 0.925),
    ("inception1d", 0.921),
    ("lstm_bidir", 0.921),
    ("Wavelet+NN", 0.874),
)


def f(x: float, d: int = 3) -> str:
    """Angka format Indonesia (koma desimal)."""
    return f"{x:.{d}f}".replace(".", ",")


def ni(n: float) -> str:
    """Bilangan bulat format Indonesia (titik ribuan)."""
    return f"{int(n):,}".replace(",", ".")


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "axes.labelcolor": INK2,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK2,
            "ytick.labelcolor": INK2,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 1.5,
            "lines.solid_capstyle": "round",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "savefig.bbox": "tight",
            "figure.constrained_layout.use": True,
        }
    )


@dataclass
class Ctx:
    results: Path
    profile: str

    def run_dir(self, m: Model) -> Path:
        return self.results / "runs" / f"{m.run}__{self.profile}"

    def csv(self, m: Model, rel: str) -> pd.DataFrame:
        return pd.read_csv(self.run_dir(m) / rel)

    def json(self, m: Model, rel: str) -> Any:
        return json.loads((self.run_dir(m) / rel).read_text(encoding="utf-8"))

    def config(self, m: Model) -> dict[str, Any]:
        return yaml.safe_load((self.run_dir(m) / "config_resolved.yaml").read_text("utf-8"))

    def seeds(self, m: Model) -> list[int]:
        return sorted(int(p.name.split("_")[1]) for p in (self.run_dir(m) / "e1").glob("seed_*"))

    def ref_seed(self, m: Model) -> int:
        """Seed acuan untuk gambar seed tunggal: 42 jika ada, selain itu seed pertama."""
        seeds = self.seeds(m)
        return 42 if 42 in seeds else seeds[0]

    def preds(self, m: Model) -> dict[str, np.ndarray]:
        path = self.run_dir(m) / "e1" / f"seed_{self.ref_seed(m)}" / "preds.npz"
        with np.load(path) as z:
            return {k: z[k] for k in z.files}


MAIN, BASE = MODELS[0], MODELS[1]
FigFn = Callable[[Ctx], tuple[plt.Figure, str, str]]


def macro(ctx: Ctx, m: Model, metric: str = "auroc", cls: str = "macro") -> pd.Series:
    c = ctx.csv(m, "e2/classification_by_seed.csv")
    return c[(c["metric"] == metric) & (c["class"] == cls)]["value"]


# --------------------------------------------------------------------------- Fig 1
def fig_protocol(ctx: Ctx) -> tuple[plt.Figure, str, str]:
    sp = ctx.csv(MAIN, "splits.csv")
    n = sp["partition"].value_counts()
    pats = sp.groupby("partition")["patient_id"].nunique()
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.set_axis_off()
    ax.set_xlim(-0.05, 10.05)
    ax.set_ylim(0.15, 4.1)

    def box(x: float, y: float, w: float, h: float, text: str, bold: bool = False) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                fc=NEUTRAL,
                ec=AXIS,
                lw=0.8,
            )
        )
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=7.2,
            color=INK,
            fontweight="bold" if bold else "normal",
            linespacing=1.35,
        )

    def arrow(x0: float, y0: float, x1: float, y1: float) -> None:
        ax.annotate(
            "", (x1, y1), (x0, y0), arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 0.9}
        )

    total = int(n.sum())
    box(
        0.0,
        1.45,
        2.3,
        1.3,
        f"PTB-XL v1.0.3\n{ni(total)} rekaman berlabel\n5 superclass, 12 lead,\n100 Hz, 10 s",
    )
    rows = [
        ("TRAIN", "fold 1–8", "latih model, statistik\nstandarisasi"),
        ("VAL_A", "fold 9, ½ pasien", "seleksi epoch, temperature /\nPlatt, ambang F1"),
        ("VAL_B", "fold 9, ½ pasien", "kalibrasi conformal\n(M1–M3, PAC, CRC)"),
        ("TEST", "fold 10", "evaluasi akhir sekali,\ntiap akses dicatat"),
    ]
    for i, (p, fold, role) in enumerate(rows):
        y = 3.25 - i * 1.0
        box(
            2.85,
            y,
            3.15,
            0.75,
            f"{p} · {fold}\n{ni(n[p])} rekaman · {ni(pats[p])} pasien",
            bold=True,
        )
        box(6.5, y, 3.5, 0.75, role)
        arrow(2.32, 2.1, 2.83, y + 0.375)
        arrow(6.02, y + 0.375, 6.48, y + 0.375)
    cap = (
        f"Alur data dan pembagian per pasien. {ni(total)} rekaman PTB-XL berlabel "
        f"(411 tanpa superclass dibuang) dibagi menurut strat_fold; fold 9 dibagi per pasien "
        f"menjadi VAL_A ({ni(n['VAL_A'])}) dan VAL_B ({ni(n['VAL_B'])}). Overlap pasien "
        "antar-partisi = 0 (diuji otomatis)."
    )
    return fig, "Alur eksperimen dan partisi data", cap


# --------------------------------------------------------------------------- Fig 2
def fig_training(ctx: Ctx) -> tuple[plt.Figure, str, str]:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    info = {}
    for m, color in ((BASE, SLOTS[0]), (MAIN, SLOTS[1])):
        hs = [ctx.csv(m, f"e1/seed_{s}/history.csv") for s in ctx.seeds(m)]
        best = [int(h.loc[h["val_a_macro_auroc"].idxmax(), "epoch"]) for h in hs]
        for i, h in enumerate(hs):
            axes[0].plot(
                h["epoch"],
                h["val_a_macro_auroc"],
                color=color,
                lw=0.9,
                alpha=0.75,
                label=m.label if i == 0 else None,
            )
            b = h.loc[h["val_a_macro_auroc"].idxmax()]
            axes[0].plot(
                b["epoch"], b["val_a_macro_auroc"], "o", ms=4.5, color=color, mec="white", mew=1.0
            )
        ax = axes[1] if m is BASE else axes[2]
        E = min(len(h) for h in hs)
        tr = np.array([h["train_loss"].to_numpy()[:E] for h in hs])
        va = np.array([h["val_a_loss"].to_numpy()[:E] for h in hs])
        ep = np.arange(1, E + 1)
        for arr, c, lab in ((tr, MUTED, "BCE latih"), (va, color, "BCE VAL_A")):
            ax.fill_between(ep, arr.min(0), arr.max(0), color=c, alpha=0.12, lw=0)
            ax.plot(ep, arr.mean(0), color=c, label=lab)
        ax.set_title(m.label, loc="left")
        ax.set_xlabel("epoch")
        ax.legend(loc="upper left")
        info[m.key] = (
            min(best),
            max(best),
            [h["val_a_macro_auroc"].max() for h in hs],
            float(np.min(va.mean(axis=0))),
            va.mean(0)[-1],
            tr.mean(0)[-1],
        )
    axes[0].set_title("Macro-AUROC VAL_A (5 seed/model)", loc="left")
    axes[0].set_xlabel("epoch")
    axes[0].legend(loc="lower right")
    axes[1].set_ylabel("loss")
    axes[1].sharey(axes[2])
    b, m_ = info["base"], info["main"]
    cap = (
        f"Kurva pelatihan. Model dasar mencapai puncak VAL_A pada epoch {b[0]}–{b[1]} "
        f"(macro-AUROC {f(min(b[2]))}–{f(max(b[2]))}) lalu overfit: BCE latih turun ke "
        f"{f(b[5], 4)} sementara BCE VAL_A naik dari minimum {f(b[3])} ke {f(b[4])}. "
        f"Random crop 2,5 s menunda overfit (epoch terbaik {m_[0]}–{m_[1]}) dan menaikkan "
        f"macro-AUROC VAL_A ke {f(min(m_[2]))}–{f(max(m_[2]))}. Titik = epoch terpilih "
        "(seleksi di VAL_A). Pita = rentang 5 seed."
    )
    return fig, "Kurva pelatihan: overfit dan efek random crop", cap


# --------------------------------------------------------------------------- Fig 3
def fig_benchmark(ctx: Ctx) -> tuple[plt.Figure, str, str]:
    rows = []
    for m in MODELS:
        v = macro(ctx, m)
        rows.append((m.label, v.mean(), v.std() if len(v) > 1 else np.nan, len(v), True))
    rows += [(f"{n} (literatur)", a, np.nan, 0, False) for n, a in LITERATURE]
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for i, (_, mu, sd, k, ours) in enumerate(rows):
        if ours:
            ax.errorbar(
                mu,
                i,
                xerr=None if np.isnan(sd) else sd,
                fmt="o",
                ms=6,
                color=SLOTS[0],
                mec="white",
                mew=1.0,
                ecolor=SLOTS[0],
                elinewidth=1.2,
                capsize=0,
            )
            ax.text(
                mu + 0.0035,
                i,
                f"{f(mu)}" + ("" if k < 2 else f" ± {f(sd)}"),
                va="center",
                fontsize=7.5,
                color=INK,
            )
        else:
            ax.plot(mu, i, "o", ms=5, mfc="white", mec=MUTED, mew=1.2)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlim(0.85, 0.955)
    ax.set_xlabel("macro-AUROC TEST (fold 10)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Benchmark 5 superclass PTB-XL", loc="left")
    ax.plot([], [], "o", color=SLOTS[0], label="eksperimen ini (rerata ± SD seed)")
    ax.plot([], [], "o", mfc="white", mec=MUTED, label="dilaporkan literatur")
    ax.legend(loc="lower right")
    mm = macro(ctx, MAIN)
    lo_lit = min(a for _, a in LITERATURE[:-1])  # tanpa Wavelet+NN (bukan deep learning)
    hi_lit = max(a for _, a in LITERATURE)
    pos = (
        "di dalam"
        if lo_lit <= mm.mean() <= hi_lit
        else ("di bawah" if mm.mean() < lo_lit else "di atas")
    )
    spread = f" ± {f(mm.std())}" if len(mm) > 1 else ""
    cap = (
        f"Macro-AUROC pada TEST. Model utama (ResNet1D-18 + crop/TTA; dipilih karena VAL_A "
        f"tertinggi) {f(mm.mean())}{spread} ({len(mm)} seed), {pos} rentang model deep learning "
        f"literatur {f(lo_lit)}–{f(hi_lit)} "
        "(README ecg_ptbxl_benchmarking, protokol fold yang sama). Angka literatur dari "
        "pelatihan dan pipeline yang berbeda, sehingga hanya pembanding kasar."
    )
    return fig, "Benchmark terhadap literatur", cap


# --------------------------------------------------------------------------- Fig 4
def fig_efficiency(ctx: Ctx) -> tuple[plt.Figure, str, str]:
    from matplotlib.ticker import FuncFormatter, NullFormatter

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    pts: list[tuple[str, float, float, float, float]] = []
    for m in MODELS:
        e = ctx.json(m, "e1/efficiency.json")
        v = macro(ctx, m)
        sd = float(v.std()) if len(v) > 1 else 0.0
        pts.append(
            (
                m.label,
                float(e["cpu_latency_ms_1thread"]),
                float(e["n_parameters"]),
                float(v.mean()),
                sd,
            )
        )
    panels = (
        (axes[0], 1, "latensi CPU per rekaman 10 s (ms, 1 thread)", [2, 5, 10, 20, 50]),
        (axes[1], 2, "jumlah parameter", [1e3, 1e4, 1e5, 1e6, 1e7]),
    )
    for ax, idx, xlab, ticks in panels:
        xv = [float(p[idx]) for p in pts]
        xmax = max(xv)
        for p, x in zip(pts, xv, strict=True):
            ax.errorbar(
                x,
                p[3],
                yerr=p[4],
                fmt="o",
                ms=6,
                color=SLOTS[0],
                mec="white",
                mew=1.0,
                elinewidth=1.2,
                capsize=0,
            )
            right = x >= xmax * 0.6
            ax.annotate(
                p[0],
                (x, p[3]),
                textcoords="offset points",
                xytext=(-7, -3) if right else (7, -3),
                ha="right" if right else "left",
                fontsize=7.2,
                color=INK2,
            )
        ax.set_xscale("log")
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda v, _: f"{v:g}" if v < 1e3 else f"{v:.0e}".replace("e+0", "e"))
        )
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(min(xv) / 1.8, xmax * 1.8)
        ax.set_xlabel(xlab + " · skala log")
    axes[0].set_ylabel("macro-AUROC TEST")
    axes[0].set_title("Akurasi vs biaya inferensi", loc="left")
    axes[1].set_title("Akurasi vs ukuran model", loc="left")
    env = ctx.json(MAIN, "e1/efficiency.json")["env"]
    d = {p[0]: p for p in pts}
    cap = (
        f"Trade-off akurasi–efisiensi (latensi diukur di CPU runtime {env.get('platform', '?')}, "
        f"1 thread). Model utama memerlukan {f(d[MAIN.label][1], 1)} ms karena merata-rata 7 "
        f"jendela; ResNet kecil {ni(d[MODELS[2].label][2])} parameter, "
        f"{f(d[MODELS[2].label][1], 1)} ms, macro-AUROC {f(d[MODELS[2].label][3])}; model fitur "
        "klasik hanya "
        f"{ni(d[MODELS[3].label][2])} parameter dengan macro-AUROC {f(d[MODELS[3].label][3])}."
    )
    return fig, "Akurasi vs efisiensi", cap


# --------------------------------------------------------------------------- Fig 5
def fig_per_class(ctx: Ctx) -> tuple[plt.Figure, str, str]:
    fig, axes = plt.subplots(1, 5, figsize=(7.2, 2.3), sharey=True, sharex=True)
    hyp = {}
    for ax, c in zip(axes, CLASSES, strict=True):
        for i, m in enumerate(MODELS[::-1]):
            v = macro(ctx, m, "auroc", c)
            ax.errorbar(
                v.mean(),
                i,
                xerr=v.std() if len(v) > 1 else None,
                fmt="o",
                ms=5,
                color=CLASS_COLOR[c],
                mec="white",
                mew=0.8,
                elinewidth=1.2,
                capsize=0,
            )
            hyp[(c, m.key)] = v.mean()
        ax.set_title(c, loc="left")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(MODELS)), [m.label for m in MODELS[::-1]])
    axes[2].set_xlabel("AUROC TEST (rerata ± SD seed)")
    worst = min(CLASSES, key=lambda c: hyp[(c, "main")])
    cap = (
        f"AUROC per kelas. Pada model utama, kelas tersulit adalah {worst} "
        f"({f(hyp[(worst, 'main')])}); NORM tertinggi ({f(hyp[('NORM', 'main')])}). Urutan "
        "model konsisten di semua kelas."
    )
    return fig, "AUROC per kelas", cap


# --------------------------------------------------------------------------- Fig 6
def fig_calibration(ctx: Ctx) -> tuple[plt.Figure, str, str]:
    from ecgdig.metrics.calibration import _bin_index
    from ecgdig.models.calibrators import Calibrator

    d = ctx.preds(MAIN)
    zA, yA, zT, yT = d["logits_VAL_A"], d["y_VAL_A"], d["logits_TEST"], d["y_TEST"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1, 1.1]})
    ax = axes[0]
    ax.axhline(0, color=INK, lw=0.7)
    for color, meth, lab in (
        (SLOTS[0], "none", "tanpa kalibrasi"),
        (SLOTS[1], "temp", "temperature"),
        (SLOTS[2], "platt", "Platt (Newton)"),
        (SLOTS[3], "isotonic", "isotonic (PAV)"),
    ):
        p = Calibrator(meth).fit(zA, yA).transform(zT).ravel()
        y = yT.ravel().astype(float)
        idx = _bin_index(p, 15, "mass")
        pm = np.bincount(idx, p) / np.bincount(idx)
        ym = np.bincount(idx, y) / np.bincount(idx)
        ax.plot(pm, ym - pm, "-o", color=color, ms=3.5, mec="white", mew=0.6, label=lab)
    ax.set_xlabel("probabilitas prediksi p̄ (rerata bin, 15 bin massa-sama)")
    ax.set_ylabel("ȳ − p̄ (frekuensi teramati − prediksi)")
    ax.text(
        0.36, 0.97, "↑ model kurang yakin", transform=ax.transAxes, va="top", fontsize=7, color=INK2
    )
    ax.text(
        0.02,
        0.03,
        "↓ model terlalu yakin",
        transform=ax.transAxes,
        va="bottom",
        fontsize=7,
        color=INK2,
    )
    ax.set_title("Selisih kalibrasi, TEST", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    s = ctx.csv(MAIN, "e5/calibration_summary.csv")
    e = s[(s["class"] == "macro") & (s["metric"] == "ece_width")].set_index("calibrator")
    order = ["none", "temp", "temp_k", "platt", "isotonic", "hist"]
    names = {
        "none": "tanpa",
        "temp": "temperature",
        "temp_k": "temperature per kelas",
        "platt": "Platt",
        "isotonic": "isotonic (PAV)",
        "hist": "histogram binning",
    }
    ax = axes[1]
    for i, k in enumerate(order[::-1]):
        ax.errorbar(
            e.loc[k, "mean"],
            i,
            xerr=e.loc[k, "std"],
            fmt="o",
            ms=6,
            color=SLOTS[0],
            mec="white",
            mew=1.0,
            elinewidth=1.2,
            capsize=0,
        )
        ax.text(
            e.loc[k, "mean"] + e.loc[k, "std"] + 0.001,
            i,
            f(e.loc[k, "mean"]),
            va="center",
            fontsize=7.2,
            color=INK2,
        )
    ax.set_yticks(range(len(order)), [names[k] for k in order[::-1]])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("macro-ECE TEST (fit di VAL_A; rerata ± SD seed)")
    ax.set_title("macro-ECE per kalibrator", loc="left")
    lab = s[(s["class"] == "ALL") & (s["metric"] == "labels_per_record_M2")].set_index("calibrator")
    inv = sorted(
        k
        for k in ("none", "temp", "temp_k", "platt")
        if np.isclose(lab.loc[k, "mean"], lab.loc["none", "mean"])
    )
    best = e["mean"].idxmin()
    cap = (
        f"Kalibrasi model utama. Kiri: selisih frekuensi teramati − probabilitas prediksi per "
        f"bin (5 kelas digabung, seed {ctx.ref_seed(MAIN)}); garis 0 = terkalibrasi sempurna. "
        "Kanan: "
        f"macro-ECE; terbaik {names[best]} ({f(e.loc[best, 'mean'])}) vs tanpa kalibrasi "
        f"({f(e.loc['none', 'mean'])}). Ukuran himpunan conformal M2 identik untuk "
        f"{', '.join(names[k] for k in inv)} ({f(lab.loc['none', 'mean'], 3)} label/rekaman), "
        "sesuai proposisi invariansi terhadap transformasi monoton per kelas."
    )
    return fig, "Kalibrasi probabilitas", cap


# --------------------------------------------------------------------------- Fig 7
def fig_coverage(ctx: Ctx) -> tuple[plt.Figure, str, str]:
    c = ctx.csv(MAIN, "e2/conformal_by_seed.csv")
    c = c[(c["method"] == "M2") & (c["class"] != "ALL")]
    fig, axes = plt.subplots(1, 5, figsize=(7.2, 2.3), sharey=True, sharex=True)
    below = 0
    for ax, k in zip(axes, CLASSES, strict=True):
        g = c[c["class"] == k].groupby("alpha")
        nom = 1 - np.array(sorted(c["alpha"].unique()))
        mu, lo, hi = g["cov1"].mean(), g["cov1"].min(), g["cov1"].max()
        up = g["cov1_upper_theory"].mean()
        ax.plot([0.78, 1.0], [0.78, 1.0], color=INK, lw=0.7, label="target 1 − α")
        ax.plot(nom, up.to_numpy(), color=MUTED, lw=0.9, label="batas atas 1 − α + 1/(n+1)")
        ax.errorbar(
            nom,
            mu.to_numpy(),
            yerr=[mu - lo, hi - mu],
            fmt="o",
            ms=4.5,
            color=CLASS_COLOR[k],
            mec="white",
            mew=0.8,
            elinewidth=1.1,
            capsize=0,
        )
        below += int(np.sum(mu.to_numpy() < nom - 0.01))
        ax.set_title(k, loc="left")
        ax.set_xlim(0.78, 1.0)
        ax.set_ylim(0.70, 1.0)
    axes[0].set_ylabel("coverage TEST P(k ∈ C | y_k=1)")
    axes[0].plot([], [], "o", color=MUTED, label="coverage TEST (titik: warna kelas)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=3)
    axes[2].set_xlabel("target 1 − α")
    cap = (
        f"Coverage conformal Mondrian per kelas (M2, kalibrasi VAL_B) vs target. Garis diagonal "
        "= 1 − α; garis abu = batas atas teoretis 1 − α + 1/(n+1). Titik = rerata 5 seed, "
        f"batang = min–maks. {below} dari 25 titik lebih dari 1 poin persentase di bawah target "
        "(VAL_B yang sama dipakai semua seed; fluktuasi satu set kalibrasi, lihat Gbr. 8)."
    )
    return fig, "Coverage conformal per kelas", cap


# --------------------------------------------------------------------------- Fig 8
def fig_validity(ctx: Ctx) -> tuple[plt.Figure, str, str]:
    from ecgdig.eval.validity import pac_rank, simulate_split_coverage, theory_distribution
    from ecgdig.models.conformal import conformal_rank

    cfg = ctx.config(MAIN)["validity"]
    d = ctx.preds(MAIN)
    alpha, delta = 0.10, float(cfg["pac_delta"])
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), sharey="row")
    notes = []
    for r, k in enumerate(("NORM", "HYP")):
        j = CLASSES.index(k)
        s = 1.0 - 1.0 / (1.0 + np.exp(-d["logits_TEST"][d["y_TEST"][:, j] == 1, j]))
        n = s.size // 2
        for col, (rule, rank) in enumerate(
            (
                ("standar", conformal_rank(n, alpha)),
                ("PAC δ=" + f(delta, 2), pac_rank(n, alpha, delta)),
            )
        ):
            sim = simulate_split_coverage(
                s, n, rank, int(cfg["n_splits"]), np.random.default_rng(int(cfg["seed"]))
            )
            m = sim.n_test
            dist = theory_distribution(n, m, rank)
            ax = axes[r, col]
            cnt = np.bincount(sim.counts, minlength=m + 1) / sim.counts.size
            x = np.arange(m + 1) / m
            lo = max(0, int(np.min(sim.counts)) - 3)
            ax.bar(
                x[lo:],
                cnt[lo:],
                width=0.8 / m,
                color=CLASS_COLOR[k],
                alpha=0.55,
                lw=0,
                label="empiris (1.000 split)",
            )
            if dist is not None:
                ax.plot(
                    x[lo:],
                    dist.pmf(np.arange(m + 1))[lo:],
                    color=INK,
                    lw=1.1,
                    label="teori Beta-Binomial",
                )
            ax.axvline(1 - alpha, color=MUTED, lw=0.9)
            ax.set_xticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
            cov = sim.counts / m
            ax.set_title(f"{k} (n = {n}, m = {m}) · aturan {rule}", loc="left")
            ax.text(
                0.02,
                0.95,
                f"rerata {f(cov.mean())} (teori {f(dist.mean() / m if dist else 1)})"
                f"\nP(cov < 0,9) = {f(np.mean(cov < 1 - alpha), 2)}",
                transform=ax.transAxes,
                va="top",
                fontsize=7.2,
                color=INK2,
            )
            notes.append((k, rule, cov.mean(), np.mean(cov < 1 - alpha)))
    axes[1, 0].set_xlabel("coverage pada set uji")
    axes[1, 1].set_xlabel("coverage pada set uji")
    axes[0, 0].set_ylabel("proporsi split")
    axes[1, 0].set_ylabel("proporsi split")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    proxies = [
        Line2D([], [], color=INK, lw=1.1, label="teori Beta-Binomial"),
        Patch(color=MUTED, alpha=0.55, label="empiris, 1.000 split (warna batang = kelas)"),
    ]
    fig.legend(handles=proxies, loc="outside lower center", ncol=2)
    cap = (
        "Validitas sampel-hingga split conformal (N positif TEST satu kelas dipecah acak per "
        "rekaman 1.000 kali menjadi kalibrasi n = ⌊N/2⌋ dan uji m = N − n, α = 0,10). "
        "Distribusi coverage empiris berimpit dengan "
        "teori Beta-Binomial(m, k, n+1−k), k = ⌈(n+1)(1−α)⌉. Aturan PAC (rank k_δ) menurunkan "
        "peluang coverage bersyarat < 1−α ke ≤ δ dengan biaya himpunan lebih besar; garis abu = "
        "1 − α."
    )
    return fig, "Validitas teori conformal (Beta-Binomial)", cap


# --------------------------------------------------------------------------- Fig 9
def fig_subgroups(ctx: Ctx) -> tuple[plt.Figure, str, str]:
    s = ctx.csv(MAIN, "e4/subgroups_summary.csv")
    groups = [
        ("age_band", "<50", "umur <50"),
        ("age_band", "50-64", "umur 50–64"),
        ("age_band", "65-79", "umur 65–79"),
        ("age_band", "80+", "umur 80+"),
        ("sex", "sex=0", "laki-laki"),
        ("sex", "sex=1", "perempuan"),
        ("quality", "clean", "sinyal bersih"),
        ("quality", "noisy", "sinyal ber-flag noise"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(7.2, 3.1), sharey=True, sharex=True)
    worst = []
    for ax, k in zip(axes, CLASSES, strict=True):
        for i, (var, g, _) in enumerate(groups[::-1]):
            r = s[(s["variable"] == var) & (s["group"] == g) & (s["class"] == k)]
            if r.empty:
                continue
            r = r.iloc[0]
            for dy, tag, color in ((0.17, "M2", SLOTS[0]), (-0.17, "M2group", SLOTS[1])):
                mu = r[f"cov1_{tag}_mean"]
                rs = ctx.ref_seed(MAIN)
                lo = r.get(f"cov1_{tag}_ci_lo_seed{rs}", np.nan)
                hi = r.get(f"cov1_{tag}_ci_hi_seed{rs}", np.nan)
                ax.plot([lo, hi], [i + dy, i + dy], color=color, lw=1.1)
                small = bool(r["too_small"])
                ax.plot(
                    mu,
                    i + dy,
                    "o",
                    ms=4.6 if small else 4.2,
                    color=color,
                    mec=color if small else "white",
                    mew=1.1 if small else 0.7,
                    mfc="white" if small else color,
                )
            worst.append((r["cov1_M2_mean"], k, groups[::-1][i][2], int(r["n_test_pos"])))
        ax.axvline(0.9, color=INK, lw=0.7)
        ax.set_title(k, loc="left")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0.4, 1.02)
    axes[0].set_yticks(range(len(groups)), [g[2] for g in groups[::-1]])
    axes[2].set_xlabel("coverage TEST pada positif (α = 0,10)")
    axes[0].plot([], [], "o", color=SLOTS[0], label="Mondrian per kelas (marginal)")
    axes[0].plot([], [], "o", color=SLOTS[1], label="Mondrian per (kelas, subgrup)")
    axes[0].plot([], [], "o", mfc="white", mec=MUTED, label="< 30 positif")
    fig.legend(loc="outside lower center", ncol=3)
    w = sorted(x for x in worst if x[3] >= 30)[:3]
    cap = (
        f"Coverage per subgrup pada model utama. Titik = rerata {len(ctx.seeds(MAIN))} seed; "
        f"garis = CI 95% bootstrap pasien (seed {ctx.ref_seed(MAIN)}); garis tegak = target 0,90. "
        "Under-coverage terbesar (≥ 30 positif): "
        + "; ".join(f"{k} {g} {f(v)} ({n} positif TEST)" for v, k, g, n in w)
        + ". Kalibrasi per subgrup memulihkan coverage tetapi menaikkan flag pada negatif. "
        "Kode sex PTB-XL: 0 = laki-laki, 1 = perempuan (Wagner dkk. 2020, Tabel 2)."
    )
    return fig, "Coverage per subgrup", cap


# --------------------------------------------------------------------------- Fig 10
def fig_sets(ctx: Ctx) -> tuple[plt.Figure, str, str]:
    s = ctx.csv(MAIN, "e6/sets_summary.csv")
    a = s[s["class"] == "ALL"]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    names = {
        "M1": "M1 gabungan",
        "M2": "M2 per kelas",
        "M2_PAC": "M2-PAC",
        "CRC_FNR": "CRC (FNR)",
        "M0_0.5": "ambang 0,5",
        "F1_valA": "ambang F1",
    }
    for color, meth in zip(SLOTS, ("M2", "M2_PAC", "CRC_FNR", "M1"), strict=False):
        g = a[a["method"] == meth].sort_values("alpha")
        ax.plot(
            g["labels_per_record"],
            g["fnr_mean"],
            "-o",
            zorder=3,
            color=color,
            ms=4.5,
            mec="white",
            mew=0.8,
            label=names[meth],
        )
    for meth in ("M0_0.5", "F1_valA"):
        g = a[(a["method"] == meth)].iloc[0]
        ax.plot(g["labels_per_record"], g["fnr_mean"], "s", ms=4.5, color=MUTED, zorder=2)
        ax.annotate(
            names[meth] + " (tanpa jaminan)",
            (g["labels_per_record"], g["fnr_mean"]),
            textcoords="offset points",
            xytext=(6, -2),
            fontsize=7,
            color=INK2,
        )
    ax.set_xlabel("rerata label ditandai per rekaman |C|")
    ax.set_ylabel("FNR pasien (fraksi label benar terlewat)")
    ax.set_title("Ukuran himpunan vs label terlewat", loc="left")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=2,
        title="titik kiri→kanan: α = 0,20 · 0,10 · 0,05",
        title_fontsize=7,
        alignment="left",
    )
    ax = axes[1]
    order = ["M0_0.5", "F1_valA", "M1", "CRC_FNR", "M2", "M2_PAC"]
    h = (
        s[np.isclose(s["alpha"], 0.10) & (s["class"] != "ALL")]
        .pivot_table(index="method", columns="class", values="sensitivity")
        .loc[order, list(CLASSES)]
    )
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    im = ax.imshow(h.to_numpy(), cmap=cmap, vmin=0.4, vmax=1.0, aspect="auto")
    for (i, j), v in np.ndenumerate(h.to_numpy()):
        ax.text(
            j, i, f(v, 2), ha="center", va="center", fontsize=7, color="white" if v > 0.78 else INK
        )
    ax.set_xticks(range(5), CLASSES)
    ax.set_yticks(range(len(order)), [names[m] for m in order])
    ax.grid(False)
    ax.set_title("Sensitivitas per kelas, α = 0,10", loc="left")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).outline.set_visible(False)
    r = a[np.isclose(a["alpha"], 0.10)].set_index("method")
    c20 = a[np.isclose(a["alpha"], 0.20) & (a["method"] == "CRC_FNR")].iloc[0]
    f1 = a[a["method"] == "F1_valA"].iloc[0]
    cap = (
        f"Pembanding prediktor himpunan (model utama, kalibrasi VAL_B, TEST). Pada α = 0,10: "
        f"CRC menghasilkan {f(r.loc['CRC_FNR', 'labels_per_record'], 2)} label/rekaman dengan "
        f"FNR {f(r.loc['CRC_FNR', 'fnr_mean'])}, M2 {f(r.loc['M2', 'labels_per_record'], 2)} "
        f"label dengan FNR {f(r.loc['M2', 'fnr_mean'])}; ambang 0,5 hanya "
        f"{f(r.loc['M0_0.5', 'labels_per_record'], 2)} label tetapi melewatkan "
        f"{f(r.loc['M0_0.5', 'fnr_mean'])} label. Sensitivitas HYP: CRC "
        f"{f(h.loc['CRC_FNR', 'HYP'], 2)} vs M2 {f(h.loc['M2', 'HYP'], 2)}: mengontrol "
        "rata-rata per pasien tidak menjamin kelas minoritas. "
        f"Pada α = 0,20 CRC ({f(c20['labels_per_record'], 2)} label, FNR {f(c20['fnr_mean'])}) "
        f"hampir berimpit dengan ambang F1 ({f(f1['labels_per_record'], 2)} label, FNR "
        f"{f(f1['fnr_mean'])}), tetapi hanya CRC yang membawa jaminan."
    )
    return fig, "Jenis jaminan conformal: ukuran himpunan vs sensitivitas", cap


FIGURES: tuple[tuple[str, FigFn], ...] = (
    ("fig01_protocol", fig_protocol),
    ("fig02_training", fig_training),
    ("fig03_benchmark", fig_benchmark),
    ("fig04_efficiency", fig_efficiency),
    ("fig05_per_class", fig_per_class),
    ("fig06_calibration", fig_calibration),
    ("fig07_coverage", fig_coverage),
    ("fig08_validity", fig_validity),
    ("fig09_subgroups", fig_subgroups),
    ("fig10_sets", fig_sets),
)


def summary_table(ctx: Ctx) -> pd.DataFrame:
    """Tabel ringkas per model (rerata antar-seed), untuk notebook/paper."""
    rows = []
    for m in MODELS:
        e = ctx.json(m, "e1/efficiency.json")
        s6 = ctx.csv(m, "e6/sets_summary.csv")
        r = s6[(s6["class"] == "ALL") & np.isclose(s6["alpha"], 0.10)].set_index("method")
        rows.append(
            {
                "model": m.label,
                "seed": len(macro(ctx, m)),
                "macro_AUROC": macro(ctx, m).mean(),
                "macro_AUROC_SD": macro(ctx, m).std(),
                "macro_AUPRC": macro(ctx, m, "auprc").mean(),
                "macro_ECE_T": macro(ctx, m, "ece_T").mean(),
                "triage_AUROC": macro(ctx, m, "triage_auroc", "abnormal").mean(),
                "parameter": e["n_parameters"],
                "latensi_CPU_ms": e["cpu_latency_ms_1thread"],
                "M2_label_per_rekaman_a0.10": r.loc["M2", "labels_per_record"],
                "CRC_label_per_rekaman_a0.10": r.loc["CRC_FNR", "labels_per_record"],
                "CRC_FNR_a0.10": r.loc["CRC_FNR", "fnr_mean"],
            }
        )
    return pd.DataFrame(rows)


def make_all(
    results: Path, profile: str, out: Path, log: Callable[[str], None] = print
) -> list[dict[str, str]]:
    """Buat semua gambar; gambar yang gagal dicatat dan dilewati (tidak menghentikan yang lain)."""
    style()
    ctx = Ctx(results, profile)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, fn in FIGURES:
        try:
            fig, title, cap = fn(ctx)
        except (FileNotFoundError, KeyError, IndexError, ValueError) as exc:
            log(f"[viz] {name} dilewati: {type(exc).__name__}: {exc}")
            continue
        fig.savefig(out / f"{name}.png", dpi=200)
        fig.savefig(out / f"{name}.pdf")
        plt.close(fig)
        manifest.append({"file": name, "title": title, "caption": cap})
        log(f"[viz] {name}: {title}")
    try:
        summary_table(ctx).to_csv(out / "summary_table.csv", index=False)
    except (FileNotFoundError, KeyError) as exc:
        log(f"[viz] summary_table dilewati: {exc}")
    (out / "figures_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--profile", default="colab")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    man = make_all(args.results, args.profile, args.out)
    print(f"{len(man)}/{len(FIGURES)} gambar ditulis ke {args.out}")
    return 0 if man else 1


if __name__ == "__main__":
    sys.exit(main())
