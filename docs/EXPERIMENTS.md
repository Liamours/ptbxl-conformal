# EXPERIMENTS: eksperimen E1–E7, lokal dan Colab

Status: dijalankan (2026-10-28). Kode di `src/ecgdig/`, config di `configs/stage3_c0.yaml`. Angka hasil hanya dikutip dari `results/runs/<run>/` (lihat §6).

## 1. Ringkasan

| ID | Pertanyaan | Partisi | Keluaran utama |
|---|---|---|---|
| E1 | Seberapa baik ResNet1D-18 pada sinyal PTB-XL 100 Hz (C0), dan seberapa stabil antar-seed? | latih TRAIN, seleksi epoch VAL_A | `e1/seed_*/preds.npz` (logit VAL_A/VAL_B/TEST + `ecg_id`, `patient_id`, partisi), `history.csv`, `efficiency.json` |
| E2 | Setelah temperature scaling, seberapa terkalibrasi probabilitasnya, dan apakah conformal per label memberi coverage nominal pada TEST? | T di VAL_A, conformal di VAL_B, evaluasi TEST | `e2/classification_*.csv`, `e2/conformal_*.csv` |
| E3 | Apakah distribusi coverage pada satu set kalibrasi berhingga cocok dengan teori Beta-Binomial, dan berapa ukuran kalibrasi yang perlu untuk jaminan PAC? | positif TEST (split acak berulang) | `e3/validity.csv` |
| E4 | Apakah coverage marginal per kelas tetap terjaga per subgrup (jenis kelamin, umur, kualitas sinyal), dan apakah Mondrian per subgrup memulihkannya? | kalibrasi VAL_B, evaluasi TEST | `e4/subgroups_*.csv` |
| E5 | Kalibrator post-hoc mana yang paling baik, dan apakah himpunan conformal invarian terhadapnya? | fit VAL_A, evaluasi TEST | `e5/calibration_*.csv` |
| E6 | Bagaimana jenis jaminan (tanpa, per kelas, PAC, risiko FNR pasien) menukar ukuran himpunan dengan sensitivitas per kelas? | ambang F1 di VAL_A, kalibrasi VAL_B, TEST | `e6/sets_*.csv` |
| E7 | Bagaimana model klasik ber-rumus, ResNet + crop/TTA, dan ResNet kecil dibandingkan dengan ResNet1D-18 dan literatur (akurasi, kalibrasi, ukuran himpunan, biaya)? | sama dengan E1–E6 | `results/benchmark/benchmark.csv` |

E1–E4 melanjutkan desain di notebook awal (config lama: `alphas`, `t2_alphas`, `t2_num_splits`, `bootstrap_resamples`, `min_subgroup_positives`, `age_bands`). Keempatnya berguna untuk opsi K8 mana pun. Eksperimen di bawah pergeseran sinyal (inti K8-a) **belum** dibuat karena K8 belum diputuskan (§7).

## 2. Partisi dan anti-leakage
- TRAIN = fold 1–8 (17.084 rekaman), VAL_A/VAL_B = fold 9 dibagi per pasien (seed 20261028; 1.072/1.074 rekaman), TEST = fold 10 (2.158). 411 rekaman tanpa superclass dibuang (aturan `example_physionet.py`).
- **Perubahan dari notebook lama:** fold 9 tidak lagi dipakai sekaligus untuk validasi dan kalibrasi. VAL_A untuk seleksi epoch dan temperature, VAL_B hanya untuk kalibrasi conformal. Konsekuensi: n kalibrasi per kelas kira-kira setengah dari rencana T1 lama (mis. HYP ±134, bukan 268).
- Statistik standarisasi per lead dari TRAIN saja (`norm_stats.json` menyimpan SHA-256 ID sumbernya).
- E2–E4 menolak berjalan jika `train_hash` config berbeda dari saat E1 (`manifest.json`).
- Setiap evaluasi TEST ditambahkan ke `TEST_ACCESS_LOG.md` (lokal: `results/`; Colab: `ptbxl_conformal/results/`).
- `tests/test_no_leakage.py` memeriksa overlap pasien = 0 dan konsistensi `preds.npz` ↔ `splits.csv` ↔ `norm_stats.json` untuk setiap run di `results/runs/`.

## 3. Formulasi
**Model (E1).** ResNet1D-18 (8.742.533 parameter, sama dengan notebook), input $x \in \mathbb{R}^{12\times1000}$ terstandarisasi per lead. BCE multi-label tanpa bobot; AdamW (lr $10^{-3}$, wd $10^{-2}$), cosine per langkah, 30 epoch, batch 256, bf16 hanya pada forward pelatihan. Bobot diambil dari epoch dengan macro-AUROC VAL_A tertinggi.

*Kenapa BCE tanpa bobot:* minimizer populasi BCE berbobot $w_k$ memenuhi $\sigma(z^*) = \frac{w p}{w p + 1 - p}$, yaitu $z^* = \operatorname{logit}(p) + \log w_k$. Ini pergeseran aditif yang tidak bisa dikoreksi temperature scaling (yang hanya mengalikan logit). Karena E2 mengukur kalibrasi, default-nya `pos_weight: none`; `balanced` tersedia sebagai ablation.

**Temperature (E2).** $p_k = \sigma(z_k/T)$, $T^* = \arg\min_T \frac{1}{nK}\sum_{i,k}[\operatorname{softplus}(z_{ik}/T) - y_{ik} z_{ik}/T]$ di VAL_A (cari pada $\log T$). $T$ bersama tidak mengubah AUROC maupun himpunan conformal M1–M3, karena keduanya hanya bergantung pada urutan skor.

**Conformal (E2).** Kuantil $\hat q = s_{(k)}$, $k = \lceil (n+1)(1-\alpha) \rceil$, $\hat q = +\infty$ jika $k > n$.
- M0: ambang 0,5 (pembanding tanpa jaminan).
- M1: satu $\hat q$ gabungan dari skor $1-p$ semua pasangan positif.
- M2: Mondrian per kelas, positif saja: $\Pr(1 \in C_k \mid y_k = 1) \ge 1-\alpha$.
- M3: Mondrian per kelas dan per nilai label ($s_k(x,1) = 1-p_k$, $s_k(x,0) = p_k$). $C_k = \{0,1\}$ berarti "ragu, rujuk".

Batas atas teoretis: $\mathbb{E}[\text{coverage}] \le 1 - \alpha + 1/(n+1)$ tanpa ties.

**Validitas sampel-hingga (E3, "T2").** Untuk pool $N$ skor berbeda yang dipecah acak menjadi $n$ kalibrasi dan $m$ uji, banyaknya skor uji $\le \hat q$ berdistribusi Beta-Binomial$(m, k, n+1-k)$ secara eksak. Alasannya, rank dalam pool hasil permutasi seragam, sama seperti kasus i.i.d. kontinu; identitas ini dicek numerik terhadap Negative-Hypergeometric (selisih $2{,}8\cdot10^{-16}$). Diukur rerata, SD, $\Pr(\text{cov} < 1-\alpha)$, dan statistik KS diskret dengan p-value Monte Carlo.

Varian PAC: $k_\delta = \min\{k : \Pr(\mathrm{Beta}(k, n+1-k) \ge 1-\alpha) \ge 1-\delta\}$ menjamin $\Pr_{\text{kalibrasi}}(\text{coverage} \ge 1-\alpha) \ge 1-\delta$ ($\delta = 0{,}1$).

**Subgrup (E4).** Coverage $\Pr(1 \in C_k \mid y_k = 1, G = g)$ untuk M2 marginal vs M2 per (kelas, subgrup) (kalibrasi Mondrian di VAL_B per sel), α = 0,10. Subgrup: `sex` (0 = laki-laki, 1 = perempuan; terverifikasi 2026-10-29 dari Wagner dkk., *Sci Data* 7:154 (2020), Tabel 2 "sex (male 0, female 1)"), kelompok umur (<50, 50–64, 65–79, 80+; umur 300 masuk 80+), dan kualitas sinyal (`noisy` jika salah satu dari `baseline_drift`, `static_noise`, `burst_noise`, `electrodes_problems` terisi). CI 95% dari bootstrap pasien (B = 1000) untuk seed pertama. Sel dengan < 30 positif ditandai `too_small`.

## 4. Cara menjalankan
**Lokal** (venv `.venv`, Python 3.13.9, torch 2.11.0+cu130, RTX 4060):
```bash
.venv/Scripts/python.exe -m ecgdig.cli cache --config configs/stage3_c0.yaml --profile local   # sekali (±70 s)
.venv/Scripts/python.exe -m ecgdig.cli all   --config configs/stage3_smoke.yaml --profile local # smoke (±15 s)
.venv/Scripts/python.exe -m ecgdig.cli all   --config configs/stage3_c0.yaml --profile local    # E1–E4
```
**Colab:** sel B0-B6 menjalankan kode yang sama dari bundel zip; hasilnya disimpan di `results/runs/<run>/` dan `TEST_ACCESS_LOG.md`.

## 5. Asumsi dan keterbatasan
- Sinyal `records100` dipakai apa adanya (tanpa high-pass), sama seperti notebook dan benchmark. Opsi `preprocess.highpass_hz` tersedia.
- Bit-exact reproducibility antar-GPU tidak diharapkan (cuDNN benchmark, bf16). Variasi antar-seed dilaporkan terpisah dari CI bootstrap.
- Inferensi selalu fp32. Logit bf16 menghasilkan banyak ties (terdeteksi di smoke test: 113 ties dari 158 skor), yang membuat conformal konservatif dan merusak eksaknya teori E3.
- E3 memakai positif TEST sesuai desain T2 notebook. Analisis ini tidak men-tuning apa pun, tetapi tetap tercatat sebagai akses TEST.

## 6. Hasil run lokal `c0_resnet1d18_100hz_v1__local` (2026-10-28)
Provenance: `results/runs/c0_resnet1d18_100hz_v1__local/` (`manifest.json`: train_hash `60af3f909522`; kode E1/E2/E4 `56c5fbe6782b`, E3 dijalankan ulang dengan kode `f7a8280bb7ba` setelah kolom `theory_cond_p_below` ditambahkan; `stage.json` per tahap). Lingkungan: RTX 4060, torch 2.11.0+cu130, Python 3.13.9. Akses TEST: `results/TEST_ACCESS_LOG.md`.

**Replikasi Colab** (`ptbxl_conformal/results/runs/c0_resnet1d18_100hz_v1__colab/`, A100-SXM4-40GB, Python 3.13.16, torch 2.11.0+cu130, kode `f7a8280bb7ba`, train_hash sama `60af3f909522`, dijalankan pemilik proyek 2026-10-28 17:31–17:38 WIB):
- E1: epoch terbaik 6–9, VAL_A 0,9233–0,9250, ±29 s per seed.
- E2: macro-AUROC TEST 0,9163 ± 0,0014 (lokal 0,9172 ± 0,0020), macro-AUPRC 0,801 (lokal 0,803), macro-ECE setelah T 0,043 (lokal 0,032), T 1,01–1,51, M2 α = 0,10 cov1 MI 0,887 / HYP 0,905.
- Selisih lintas hardware lebih kecil dari variasi antar-seed.
- E3 Colab masih berjalan saat dicatat.

**E1.** Epoch terbaik 6–9 (dari 30). Macro-AUROC VAL_A 0,9234 / 0,9241 / 0,9261 / 0,9284 / 0,9250 (seed 42–46); ±84 s per seed. Model overfit setelah ±epoch 8: loss latih turun ke ~5·10⁻⁴, loss VAL_A naik ke ~0,8. Efisiensi: 8.742.533 parameter, state_dict 35,0 MB (fp32), latensi CPU i7-13700F per rekaman 10 s **8,0 ms (1 thread) / 3,5 ms (4 thread)**.

**E2 (TEST, rerata ± SD antar 5 seed; CI 95% bootstrap pasien B=1000 untuk seed 42).**

| Metrik | Nilai |
|---|---|
| macro-AUROC | **0,9172 ± 0,0020** (seed 42: 0,9139, CI 0,9058–0,9209) |
| AUROC per kelas | NORM 0,940 · STTC 0,928 · CD 0,912 · MI 0,904 · HYP 0,902 |
| macro-AUPRC | 0,803 (0,794–0,808 antar-seed) |
| T* (VAL_A) | 1,04–1,28 (model terlalu yakin) |
| macro-ECE raw → T → T_k | 0,039 → 0,032 → 0,030 |
| Brier raw → T | 0,0897 → 0,0889; NLL 0,298 → 0,291 |
| Triage 1−p_NORM | AUROC 0,930–0,934; target spesifisitas 0,90 dari VAL_A memberi spesifisitas TEST 0,86–0,92 dan sensitivitas 0,80–0,84 |

Sanity check terhadap benchmark (README `repo/ecg_ptbxl_benchmarking`, superdiagnostic fold 10): resnet1d_wang 0,930(05), xresnet1d101 0,928(05). **Model kita ±0,013 di bawahnya, jadi kriteria "C0 dalam rentang benchmark" belum terpenuhi.** Kemungkinan penyebab: tanpa augmentasi/random crop dan overfit (usulan E1b, §7).

Conformal, coverage pada positif (cov1), rerata 5 seed:

| α | M0 (0,5) | M1 gabungan | M2 Mondrian per kelas (min–maks kelas) |
|---|---|---|---|
| 0,01 | 0,49–0,88 | 0,969–0,999 | 0,986–0,997 |
| 0,05 | sama | 0,860–0,992 | 0,931–0,963 |
| 0,10 | sama | 0,766–0,974 | 0,878–0,917 |
| 0,20 | sama | 0,596–0,933 | 0,763–0,825 |

- M0 (ambang 0,5) hanya memberi sensitivitas HYP 0,485 dan MI 0,598.
- M1 gabungan tidak menjaga coverage per kelas: HYP 0,766 pada α = 0,10.
- M2 pada α = 0,10: NORM 0,917, CD 0,905, HYP 0,894, STTC 0,891, **MI 0,878** (antar-seed 0,851–0,889). Kelas lain dekat nominal. VAL_B sama untuk semua seed, jadi rerata antar-seed tidak menghapus variasi akibat pemilihan set kalibrasi. Menurut E3, SD coverage untuk n ≈ 275 adalah ±0,025, sehingga MI masih dalam ±1 SD. Ini bukan bukti pelanggaran jaminan, tetapi juga tidak boleh ditulis "coverage tercapai" untuk MI.
- M3 (α = 0,10): cov0 0,88–0,91; laju "ragu, rujuk" NORM 7%, STTC 10%, MI 15%, CD 20%, HYP 20%; himpunan kosong 0%.
- M2 pada α = 0,10 memberi rerata 2,10 label per rekaman, dan semua label positif tercakup pada 87,4% rekaman.

**E3 (positif TEST, seed 42, 1.000 split, n ≈ m ≈ N/2, tanpa ties).**
- Rerata dan SD coverage empiris cocok dengan teori Beta-Binomial di semua 20 konfigurasi. Selisih rerata ≤ 0,0022, contohnya NORM α = 0,10: 0,8993 vs 0,9004 dan SD 0,0195 vs 0,0193. Untuk HYP (n = 131), SD coverage 0,036 (α = 0,10).
- KS: 19/20 p > 0,05. Satu konfigurasi (MI, α = 0,20, standard) p = 0,002. Pada uji ulang dengan 20.000 split × 3 seed hasilnya p = 0,36 / 0,50 / 0,61, jadi itu fluktuasi kebetulan dari satu set 1.000 split.
- PAC (δ = 0,10): P(F(q̂) < 1−α) teoretis 0,068–0,098 ≤ δ di semua kelas. Coverage pada m titik uji < 1−α terjadi pada 13–18% split (teori Beta-Binomial 13–18%) karena derau binomial titik uji. Biaya PAC: rerata coverage naik ±0,02–0,05.

**E4 (α = 0,10, TEST; M2 marginal vs M2 per-subgrup; eksploratori, 40 sel tanpa koreksi multipel).**
- **Umur < 50 tahun under-coverage untuk kelas abnormal** (M2 marginal): CD 0,771 (CI seed 42: 0,563–0,814), MI 0,774 (0,581–0,868), STTC 0,814, HYP 0,848 (n = 25, `too_small`). NORM < 50 justru 0,953.
- NORM ≥ 80 tahun: 0,695 (n = 21, `too_small`).
- Kalibrasi per subgrup memulihkan coverage, tetapi dengan biaya flag palsu yang besar. Untuk < 50: MI 0,774 → 0,974 dengan laju flag pada negatif 0,085 → 0,330; CD 0,771 → 0,857 dengan flag 0,205 → 0,396. n kalibrasi per sel kecil (10–24).
- Kualitas sinyal: coverage *noisy* vs *clean* mirip, kecuali MI (0,846 vs 0,889).
- Jenis kelamin: tidak ada selisih besar (0,87–0,93 di kedua grup).

**Pemeriksaan run Colab (2026-10-28).** Tidak ada error di sel B0–B6. Cache lama di Drive lolos validasi (urutan `ecg_id`, label, dan 8 rekaman acak identik dengan WFDB). `splits.csv` Colab identik dengan lokal (21.388 baris); hash file berbeda hanya karena CRLF vs LF. Statistik normalisasi identik (selisih 0,0). Satu seed yang sama menghasilkan model berbeda antar-hardware (epoch terbaik 6 vs 9; korelasi logit TEST 0,925), sehingga laporan wajib memakai rerata ± SD antar-seed, bukan seed tunggal. E3 Colab: satu konfigurasi dengan KS p = 0,020 (CD α = 0,20 PAC), konsisten dengan uji ganda (lokal: konfigurasi lain, p = 0,002, hilang saat diuji ulang 20.000 split).

**Catatan lintas eksperimen:** semua metode conformal sedikit di bawah target pada TEST (M2 cov1 rerata 0,897; FNR CRC 0,106 pada α = 0,10). Masih dalam fluktuasi satu set kalibrasi (E3), tetapi konsisten di dua hardware karena VAL_B dan TEST sama. Usulan diagnostik: uji exchangeability VAL_B vs TEST (KS dua sampel pada skor per kelas).

## 7. Pembanding E5–E7 (ditambahkan 2026-10-28)
Kode: `models/calibrators.py`, `models/set_predictors.py`, `features.py`, `metrics/calibration.py` (dekomposisi Brier), dan opsi crop/TTA/lebar di `models/train.py` dan `models/resnet1d.py`. Config E7: `configs/stage3_feat_lr.yaml`, `stage3_c0_crop.yaml`, `stage3_c0_small.yaml`. Di Colab: Bagian C (C1–C5).

**E5, pembanding kalibrasi post-hoc** (fit VAL_A, evaluasi TEST): none, temperature bersama, temperature per kelas, Platt/vector scaling (Newton–Raphson, H = Xᵀdiag(p(1−p))X), isotonic (PAV = proyeksi L2 ke kerucut monoton), histogram binning massa-sama.
- Metrik: ECE lebar-sama dan massa-sama, NLL, AUROC, dan dekomposisi Brier eksak BS = REL − RES + UNC + WBV − 2·WBC (identitas diuji di `tests/test_benchmark_modules.py`).
- **Proposisi invariansi:** himpunan Mondrian per kelas (M2/M3) invarian terhadap transformasi tak-turun per kelas; M1 hanya invarian terhadap transformasi yang sama untuk semua kelas. Histogram binning tidak dijamin monoton (ditunjukkan test).

Hasil lokal (5 seed, rerata):

| Kalibrator | ECE lebar | ECE massa | NLL | Brier | REL | AUROC | label/rekaman M2 α = 0,10 |
|---|---|---|---|---|---|---|---|
| none | 0,0391 | 0,0377 | 0,2977 | 0,0897 | 0,0038 | 0,9172 | 2,0986 |
| temp | 0,0317 | 0,0318 | 0,2906 | 0,0889 | 0,0030 | 0,9172 | 2,0986 |
| temp_k | 0,0295 | 0,0303 | 0,2898 | 0,0887 | 0,0028 | 0,9172 | 2,0986 |
| **platt** | **0,0213** | **0,0195** | **0,2858** | **0,0872** | **0,0013** | 0,9172 | 2,0986 |
| isotonic | 0,0261 | 0,0261 | 0,3123 | 0,0888 | 0,0018 | 0,9144 | 2,3109 |
| hist | 0,0245 | 0,0273 | 0,3107 | 0,0894 | 0,0015 | 0,9104 | 2,4088 |

- Platt terbaik di semua metrik kalibrasi tanpa mengubah AUROC.
- Ukuran himpunan M2 identik untuk none/temp/temp_k/platt (invariansi terkonfirmasi), tetapi berubah untuk isotonic (ties) dan hist (tidak monoton).
- Isotonic/hist: REL turun, tetapi NLL dan AUROC memburuk, karena fungsi tangga menghasilkan ties dan probabilitas ekstrem.

**E6, pembanding prediktor himpunan** (T bersama dari VAL_A; kalibrasi VAL_B; TEST), hasil lokal pada α = 0,10, rerata 5 seed (CI seed 42 dalam kurung):

| Metode | Jaminan | FNR pasien | Label/rekaman | Sensitivitas HYP / CD / MI |
|---|---|---|---|---|
| Ambang 0,5 | tidak ada | 0,259 | 1,18 | 0,485 / 0,655 / 0,598 |
| Ambang F1 (VAL_A) | tidak ada | 0,207 | 1,37 | 0,660 / 0,704 / 0,708 |
| M1 gabungan | coverage marginal atas pasangan positif | 0,100 | 1,91 | 0,766 / 0,829 / 0,854 |
| M2 Mondrian | per kelas | 0,103 (seed 42: 0,112 [0,100; 0,125]) | 2,10 | 0,894 / 0,905 / 0,878 |
| M2-PAC (δ = 0,1) | per kelas, bersyarat kalibrasi | 0,080 | 2,31 | 0,919 / 0,933 / 0,904 |
| **CRC-FNR** | E[FNR pasien] ≤ α | 0,106 (seed 42: 0,110 [0,099; 0,123]) | **1,87** | 0,750 / 0,821 / 0,846 |

Temuan: CRC memberi himpunan terkecil untuk target FNR yang sama, tetapi menurunkan sensitivitas kelas minoritas (HYP 0,750). Mengontrol rata-rata per pasien tidak sama dengan menjamin tiap kelas. Jenis jaminan adalah pilihan desain klinis.

**E7, pembanding model** (masing-masing menjalankan E1–E6; ditabulasi oleh `cli compare` ke `results/benchmark/benchmark.csv` bersama angka literatur):
- `feat194_logreg_v1`: 194 fitur ber-rumus (`features.py`: statistik orde 1–4, Hjorth, PSD Welch, autokorelasi → HR) + regresi logistik L2, C per kelas dipilih via AUROC VAL_A. 975 parameter.
- `c0_resnet1d18_crop250_v1`: random crop 2,5 s + rerata logit 7 jendela geser (Var(ẑ) = σ²(1+(J−1)ρ)/J), 50 epoch. Latensi CPU naik karena 7 forward per rekaman (smoke: ±41 ms).
- `c0_resnet1d18_w16_v1`: lebar 16 (551.333 parameter, ≈ 1/16) untuk kurva akurasi vs biaya.
- Status: smoke test lokal lulus untuk ketiganya (angka smoke tidak dipakai); run penuh menunggu Colab (Bagian C).

## 7b. Hasil E5–E7 di Colab dan visualisasi (2026-10-28)
Run Colab: `c0_resnet1d18_crop250_v1__colab`, `c0_resnet1d18_w16_v1__colab`, `feat194_logreg_v1__colab` (E1–E6), ditambah E5/E6 pada `c0_resnet1d18_100hz_v1__colab`; kode `319734c46168`. Percobaan pertama C1/C2 gagal karena B0 belum dijalankan ulang (kode lama di `/content`; log `e5.log`, `e6.log`, `e7_feat_lr.log` baris awal). Setelah B0, semua sukses. Sejak kode `69880fa1780c`, setiap sel `!` mengecek penanda versi yang ditulis B0.

| Model | macro-AUROC TEST | macro-ECE (T) | Parameter | CPU ms (1 thread) | M2 label/rek. α = 0,10 | CRC label/rek. (FNR) |
|---|---|---|---|---|---|---|
| ResNet1D-18 + crop/TTA (model utama; VAL_A tertinggi 0,933–0,937) | **0,926 ± 0,001** | 0,028 | 8.742.533 | 25,5 | 2,03 | 1,81 (0,099) |
| ResNet1D-18 | 0,916 ± 0,001 | 0,043 | 8.742.533 | 12,6 | 2,14 | 1,94 (0,103) |
| ResNet1D-18 kecil (w = 16) | 0,904 ± 0,003 | 0,041 | 551.333 | 2,9 | 2,24 | 1,98 (0,110) |
| 194 fitur + regresi logistik (1 run) | 0,862 | 0,024 | 975 | 4,4 | 2,72 | 2,30 (0,103) |
| Literatur (README benchmark) | 0,921–0,930 (deep) · 0,874 (Wavelet+NN) | – | – | – | – | – |

Sumber: `results/figures/summary_table.csv` dan `results/benchmark/benchmark.csv` di Drive. Latensi diukur di CPU runtime Colab, sehingga tidak sebanding langsung dengan angka lokal (i7-13700F).

**Visualisasi** (`src/ecgdig/viz.py`, `python -m ecgdig.viz --results <dir> --profile colab --out <dir>`): 10 gambar (PNG 200 dpi + PDF vektor) + `figures_manifest.json` (judul dan caption yang disusun otomatis dari angka run) + `summary_table.csv`.
- Lokasi: lokal `dist/figures/`, Drive `ptbxl_conformal/results/figures/`.
- Isi: alur dan partisi; kurva latih; benchmark vs literatur; akurasi vs efisiensi; AUROC per kelas; kalibrasi (selisih ȳ − p̄ dan ECE per kalibrator); coverage conformal vs target; validitas Beta-Binomial (standar vs PAC); coverage subgrup; trade-off jenis jaminan.
- Gaya: palet acuan skill dataviz (lolos validator CVD untuk 5 slot berdekatan); setiap gambar diperiksa visual (tabrakan label, legenda, teks terpotong) sebelum dipakai.
- Notebook Bagian D menanam ke-10 gambar sebagai output sel; D0 untuk membuat ulang.

## 7c. Analisis post-hoc (2026-10-29)
**Tanpa latih ulang dan tanpa tuning**: `python -m ecgdig.posthoc --run-dir <run> --test-access-log <log>` membaca `config_resolved.yaml` + `preds.npz` run, memakai probabilitas yang sama dengan E2 (T bersama dari VAL_A), dan menulis `<run>/posthoc/`. Kode: `src/ecgdig/posthoc.py`, `src/ecgdig/eval/inference.py`, test `tests/test_posthoc.py`.
- **S1** coverage positif per kelas untuk M0, M1, M2, M3, M2-PAC di α ∈ {0,01; 0,025; 0,05; 0,10; 0,20}: CI Clopper–Pearson, p binomial eksak P(Bin(m, 1−α) ≤ X) (q̂ dianggap tetap) dan p marginal P(BetaBin(m, k, n+1−k) ≤ X), ukuran himpunan, kosong/singleton, flag pada negatif.
- **S2** ukuran himpunan |Ĉ(x)| per rekaman: M2 vs M1 dan M2 vs M0 (rerata selisih, CI bootstrap pasien, Wilcoxon, uji-t berpasangan, Holm per seed atas 5 α × 2 perbandingan).
- **S3** E3 diulang dengan RNG yang sama (split identik dengan run Colab; dicek otomatis) dalam bentuk hitungan X, + uji χ² (sel digabung sampai harapan ≥ 5, df = sel − 1).
- **S4** tabel E4 dengan label jenis kelamin (0 = laki-laki, 1 = perempuan; Wagner dkk. 2020 Tabel 2).

Hasil model utama (`c0_resnet1d18_crop250_v1__colab`, 125 uji per metode):
- p binomial < 0,05: M0 117, M1 49, M2 11, M2-PAC 0.
- p Beta-Binomial < 0,05 untuk M2: 3; minimum 0,0089 (STTC α = 0,025).
- χ² S3: 20 uji, p 0,0457–0,9418.
- |Ĉ| M2 − M1 pada α = 0,10, seed 42: 0,1265, CI [0,1029; 0,1494].
