"""Loader PTB-XL. Test berbasis data nyata dilewati jika ``data/raw/ptbxl`` tidak ada."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ecgdig.data.ptbxl import (
    CANONICAL_LEADS,
    age_band,
    aggregate_superclasses,
    load_metadata,
    read_record,
)

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "ptbxl"
needs_raw = pytest.mark.skipif(
    not (RAW / "ptbxl_database.csv").exists(), reason="PTB-XL lokal tidak ada"
)


def test_aggregation_ignores_likelihood_and_non_diagnostic() -> None:
    dmap = {"NORM": "NORM", "IMI": "MI", "ASMI": "MI", "LVH": "HYP"}
    codes = {"IMI": 0.0, "ASMI": 100.0, "SR": 0.0, "LVH": 50.0}
    assert aggregate_superclasses(codes, dmap, ["NORM", "MI", "HYP"]) == {"MI", "HYP"}
    assert aggregate_superclasses({"SR": 0.0}, dmap, ["NORM"]) == set()


def test_age_band() -> None:
    edges = [50, 65, 80]
    assert [age_band(a, edges) for a in (20, 50, 64, 65, 79, 80, 300)] == [
        "<50",
        "50-64",
        "50-64",
        "65-79",
        "65-79",
        "80+",
        "80+",
    ]
    assert age_band(float("nan"), edges) == "Unknown"


@needs_raw
def test_counts_match_physionet_page() -> None:
    meta = load_metadata(RAW)
    assert len(meta) == 21388  # 21.799 − 411 tanpa superclass
    assert meta["patient_id"].nunique() <= 18869
    expected = {"NORM": 9514, "MI": 5469, "STTC": 5235, "CD": 4898, "HYP": 2649}
    assert {c: int(meta[c].sum()) for c in expected} == expected
    assert meta.index.is_monotonic_increasing


@needs_raw
def test_all_headers_use_canonical_lead_order() -> None:
    meta = load_metadata(RAW)
    bad = []
    for rel in meta["filename_lr"]:
        lines = (RAW / f"{rel}.hea").read_text().splitlines()
        names = tuple(line.split()[-1].upper() for line in lines[1:13])
        if names != CANONICAL_LEADS:
            bad.append(rel)
    assert not bad, f"{len(bad)} header dengan urutan lead lain, mis. {bad[:3]}"


@needs_raw
def test_read_record_contract() -> None:
    meta = load_metadata(RAW)
    x = read_record(RAW, str(meta["filename_lr"].iloc[0]))
    assert x.shape == (12, 1000) and x.dtype == np.float32 and np.isfinite(x).all()
    assert isinstance(meta, pd.DataFrame)
