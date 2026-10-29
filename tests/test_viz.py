"""Smoke test visualisasi: pohon hasil tiruan dari run smoke lokal (dilewati jika tidak ada)."""

import json
import re
import shutil
from pathlib import Path

import pytest

from ecgdig.viz import FIGURES, MODELS, f, make_all, ni

SMOKE = Path(__file__).resolve().parents[1] / "results" / "smoke"
SOURCES = {  # run model di viz ← run smoke lokal yang punya E1–E6
    "main": "smoke_c0_crop__local",
    "base": "smoke_c0_small__local",
    "small": "smoke_c0_small__local",
    "feat": "smoke_feat_lr__local",
}


def test_number_format_is_indonesian() -> None:
    assert f(0.9257) == "0,926" and f(1.5, 1) == "1,5" and ni(21388) == "21.388"


@pytest.mark.skipif(
    not all((SMOKE / s / "e6").exists() for s in SOURCES.values()),
    reason="run smoke lokal E1–E6 tidak tersedia",
)
def test_make_all_from_smoke_runs(tmp_path: Path) -> None:
    runs = tmp_path / "results" / "runs"
    for m in MODELS:
        shutil.copytree(SMOKE / SOURCES[m.key], runs / f"{m.run}__local")
    out = tmp_path / "fig"
    man = make_all(tmp_path / "results", "local", out, log=lambda _: None)
    assert len(man) == len(FIGURES)
    for item in man:
        assert (out / f"{item['file']}.png").stat().st_size > 10_000
        assert (out / f"{item['file']}.pdf").exists()
        assert item["caption"] and not re.search(r"\bnan\b", item["caption"].lower())
    assert json.loads((out / "figures_manifest.json").read_text(encoding="utf-8")) == man
    assert (out / "summary_table.csv").exists()
