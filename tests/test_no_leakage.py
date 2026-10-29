"""Test anti-leakage.

1. Pasien saling lepas antar-partisi (data sintetis + PTB-XL nyata jika ada).
2. Untuk setiap run di ``results/runs`` (jika ada): partisi di preds.npz konsisten dengan
   splits.csv; statistik normalisasi berasal dari TRAIN saja; TEST tidak tumpang tindih dengan
   VAL_A/VAL_B per pasien.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ecgdig.data.ptbxl import load_metadata
from ecgdig.data.splits import PARTITIONS, assign_partitions, patient_overlap
from ecgdig.runlog import ids_digest, read_json

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "ptbxl"
RUNS = sorted((ROOT / "results" / "runs").glob("*/splits.csv"))


def _toy_meta() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    pid = np.repeat(np.arange(200), rng.integers(1, 4, size=200))
    fold = (pid % 10) + 1
    return pd.DataFrame(
        {"patient_id": pid, "strat_fold": fold},
        index=pd.Index(np.arange(pid.size) + 1, name="ecg_id"),
    )


def test_partitions_disjoint_by_patient_toy() -> None:
    meta = _toy_meta()
    part = assign_partitions(meta, range(1, 9), 9, 10, 0.5, seed=1)
    assert set(part) == set(PARTITIONS)
    assert all(v == 0 for v in patient_overlap(meta, part).values())
    a = meta.loc[part == "VAL_A", "patient_id"].nunique()
    b = meta.loc[part == "VAL_B", "patient_id"].nunique()
    assert abs(a - b) <= 1


def test_split_is_deterministic_given_seed() -> None:
    meta = _toy_meta()
    p1 = assign_partitions(meta, range(1, 9), 9, 10, 0.5, seed=7)
    p2 = assign_partitions(meta, range(1, 9), 9, 10, 0.5, seed=7)
    assert p1.equals(p2)


def test_patient_in_two_folds_is_rejected() -> None:
    meta = _toy_meta().copy()
    meta.iloc[0, meta.columns.get_loc("strat_fold")] = 1
    meta.iloc[1, meta.columns.get_loc("strat_fold")] = 10
    meta.iloc[1, meta.columns.get_loc("patient_id")] = meta.iloc[0]["patient_id"]
    with pytest.raises(ValueError, match="lebih dari satu"):
        assign_partitions(meta, range(1, 9), 9, 10, 0.5, seed=1)


@pytest.mark.skipif(not (RAW / "ptbxl_database.csv").exists(), reason="PTB-XL lokal tidak ada")
def test_real_ptbxl_partitions_disjoint() -> None:
    meta = load_metadata(RAW)
    part = assign_partitions(meta, range(1, 9), 9, 10, 0.5, seed=20261028)
    assert all(v == 0 for v in patient_overlap(meta, part).values())
    assert (part == "TEST").sum() == (meta["strat_fold"] == 10).sum()


@pytest.mark.parametrize("splits_csv", RUNS, ids=[p.parent.name for p in RUNS])
def test_run_artifacts_consistent(splits_csv: Path) -> None:
    run = splits_csv.parent
    sp = pd.read_csv(splits_csv).set_index("ecg_id")
    by_part = {P: set(sp.index[sp["partition"] == P]) for P in PARTITIONS}
    pats = {P: set(sp.loc[sp["partition"] == P, "patient_id"]) for P in PARTITIONS}
    for i, a in enumerate(PARTITIONS):
        for b in PARTITIONS[i + 1 :]:
            assert not (pats[a] & pats[b]), f"overlap pasien {a}/{b}"
    norm = read_json(run / "norm_stats.json")
    assert norm["source"] == "TRAIN"
    assert norm["ecg_ids_sha256"] == ids_digest(np.array(sorted(by_part["TRAIN"])))
    for preds in sorted(run.glob("e1/seed_*/preds.npz")):
        with np.load(preds) as f:
            for P in ("VAL_A", "VAL_B", "TEST"):
                ids = f[f"ecg_id_{P}"]
                assert set(ids.tolist()) == by_part[P], f"{preds}: partisi {P} tidak cocok"
                assert np.array_equal(f[f"patient_id_{P}"], sp.loc[ids, "patient_id"].to_numpy())
