# ptbxl-conformal

Kode eksperimen klasifikasi superclass PTB-XL dengan ResNet1D-18 dan conformal prediction per kelas (ClassCover). Penulis: Farhan Hamzah dan M. Rifqi Dzaky Azhad.

## Isi
| Folder | Isi |
|---|---|
| `src/ecgdig/` | Data dan pembagian fold, model, kalibrasi, conformal, evaluasi, statistik post-hoc |
| `configs/` | Konfigurasi tiap run (`stage3_c0_crop.yaml` adalah model utama) |
| `scripts/` | Pengunduhan dan verifikasi file PTB-XL |
| `tests/` | Pengujian, termasuk pemeriksaan tidak ada kebocoran data antar partisi |
| `docs/EXPERIMENTS.md` | Rancangan eksperimen E1-E7 dan definisi metode |

## Menjalankan
```bash
uv venv --python 3.12
uv pip install -e . pytest
uv run python scripts/download_ptbxl.py --root data/raw/ptbxl --prefix records100/
uv run python -m ecgdig.cli all --config configs/stage3_c0_crop.yaml --profile local
uv run pytest -q
```

Data PTB-XL versi 1.0.3: https://doi.org/10.13026/kfzx-aw45. Pembagian: fold 1-8 latih, fold 9 dibagi per pasien menjadi validasi dan kalibrasi, fold 10 uji.
