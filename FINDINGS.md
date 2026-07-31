# Temuan racik

Ringkasan hasil terukur, beserta batasannya. Semua angka bisa direproduksi
dengan perintah yang tercantum; artefak mentah ada di `results_server/`.

Ringkas: **tiga instrumen kalibrasi** yang murah dan hampir tak dipakai orang,
plus **tiga kasus di mana kesimpulan berbalik** ketika instrumen itu dipakai —
dua di antaranya membatalkan klaim kami sendiri.

---

## Bagian 1 — Instrumen: cara mengetahui kapan sebuah angka tidak berarti

### 1.1 Lantai noise (noise floor)

Jalankan **dua implementasi dari algoritme yang identik** dan ukur selisihnya.
Mereka seharusnya seri, jadi selisih terukurnya adalah batas bawah klaim yang
sah pada protokol itu.

| Protokol | Pasangan identik | Lantai noise |
|---|---|---|
| Objektif sintetis, budget 30, 40 seed | `random` vs `optuna_random` | **0.0254** |
| Data nyata, budget 12, 20 seed | `random` vs `optuna_random` | **0.0092** |

Lantai menyempit seiring bertambahnya seed (≈1/√n), jadi ia sekaligus
mengukur seberapa banyak seed yang dibutuhkan untuk klaim sebesar apa pun.

### 1.2 Lantai daya uji (power floor)

Uji sign-flip 2-sisi atas n pasang punya 2ⁿ susunan tanda, jadi p terkecil
yang **mungkin** dicapai adalah 2/2ⁿ — berapa pun besar efeknya:

| n seed | 3 | 4 | 5 | **6** | 10 | 20 |
|---|---|---|---|---|---|---|
| p minimum | 0.250 | 0.125 | 0.063 | **0.031** | 0.002 | 2e-6 |

**n < 6 tidak bisa mencapai p<0.05, titik.** Praktik 3–5 seed yang lazim di
makalah NAS/HPO berada di bawah batas ini. Kami membuktikannya secara tak
sengaja: validasi 3-seed kami sendiri menghasilkan enam perbandingan yang
semuanya mentok di p=0.25–0.75. `scripts/paired_test.py` kini memperingatkan
otomatis.

### 1.3 Budget efektif

**budget efektif = budget − warmup.** Pencari berbasis model menghabiskan
warmup untuk sampling acak; membandingkannya pada budget yang tidak jauh
lebih besar dari warmup berarti membandingkan random search bernama lain.

Terukur — fraksi seed yang kurvanya masih **identik** dengan random:

| Pencari (warmup) | t=5 | t=10 | t=12 |
|---|---|---|---|
| tpe2 (10) | 20/20 | **20/20** | 12/20 |
| tpe v1 (5) | 20/20 | 7/20 | 4/20 |
| evolution (populasi 8) | 20/20 | 8/20 | 6/20 |

Pada budget 12, tpe2 hanya membuat **2** keputusan berbasis model.

---

### 1.4 Indeks ketidakstabilan papan peringkat (LII)

Instrumen §1.1–1.3 memberi tahu kapan **satu** perbandingan tidak sah. LII
menjawab pertanyaan yang lebih langsung: **seberapa sering seluruh papan
peringkat menunjuk pemenang yang salah?** Caranya: subsampel k seed dari run
ber-seed banyak (2000 ulangan), lalu hitung berapa kali juaranya berbeda dari
juara pada seluruh seed.

| k seed | P(juara berubah) — sintetis (n=40) | — data nyata (n=20) |
|---|---|---|
| 3 | 60.1% | **71.4%** |
| 5 | 51.3% | 67.0% |
| 10 | 37.4% | 54.4% |
| 15 | 28.5% | 38.0% |
| 20 | 20.0% | 0% (= acuan) |
| 30 | 7.7% | — |

**Sebuah studi 3-seed menobatkan pemenang yang salah pada 6–7 dari 10
kesempatan.** Itu bukan pengukuran, itu lempar koin dengan tabel.

Dan dari lengan plasebo (§1.1) kita bisa menghitung harga kepastian:

| Perbaikan yang ingin diklaim | Seed minimum (data nyata) |
|---|---|
| 0.05 | 16 |
| 0.02 | **105** |
| 0.01 | **>200** |

Implementasi: [`racik/validity.py`](racik/validity.py);
laporan: `AUDIT_dummy40.md`, `AUDIT_real20.md`;
panduan pakai untuk tim: [INDUSTRY.md](INDUSTRY.md).

---

## Bagian 2 — Tiga kesimpulan yang berbalik

### 2.1 Korelasi zero-cost proxy berbalik tanda dua kali

Proxy hanya melihat arsitektur. Memvalidasinya pada sampel yang
hyperparameternya ikut berubah adalah perbandingan yang tidak sah.

| Protokol validasi | NASWOT | synflow |
|---|---|---|
| Tercemar (n=9, lr/optimizer/augment bervariasi) | −0.786 | +0.449 |
| Terkontrol (n=14, hyperparameter dikunci, 1 epoch) | **+0.449** | **−0.275** |
| Terkontrol, fidelity naik (3 epoch, data 5× lebih besar) | **−0.525** | −0.305 |

Berbalik pertama karena kontaminasi hyperparameter, kedua karena perubahan
fidelity. Konsekuensi praktisnya: gerbang NASWOT di successive halving
meloloskan **9/9 mobilenet** dari 60 kandidat (bias per keluarga arsitektur),
dan hasil pencarian jadi **lebih buruk** daripada tanpa gerbang:
0.553 vs 0.645 pada seed dan budget training identik.

### 2.2 Peringkat algoritme berbalik saat seed ditambah

| Pencari | 15 seed | 40 seed |
|---|---|---|
| optuna_tpe | 0.7506 (ke-3) | **0.7637 (ke-1)** |
| evolution (kami) | **0.7689 (ke-1)** | 0.7385 (ke-3) |
| tpe v1 (kami) | 0.7375 | 0.7077 (terakhir) |

Pada 15 seed kami "menang"; pada 40 seed TPE Optuna **signifikan** lebih baik
dari TPE kami (Δ=0.056, p=0.029). Seluruh papan peringkat 15-seed berada di
bawah lantai noise — kami hampir menerbitkan klaim yang salah arah.

### 2.3 Permukaan sintetis melebih-lebihkan perbedaan antar-algoritme

| Protokol | Hasil |
|---|---|
| Sintetis, budget 30, 40 seed | tpe2 & evolution menang, p ≤ 0.01 |
| **Data nyata, budget 12, 20 seed** | **0 dari 15 perbandingan signifikan** |

Di data nyata, papan peringkatnya membentang hanya 0.020 (dua kali lantai
noise 0.0092), dan evolution vs random hanya p=0.186 — padahal daya uji
memadai (p minimum terjangkau = 2e-6). Sebagian penyebabnya adalah budget
efektif (§1.3); uji budget 40 sedang berjalan untuk memisahkan dua sebab itu.

---

## Bagian 3 — Hasil positif

**TPE v2 (`racik/tpe2.py`) menutup celah ke rujukan industri.** Defisit v1
signifikan, jadi layak dikejar; penyebabnya algoritmik, bukan tuning:
bandwidth kernel adaptif per titik (jarak tetangga terdekat) menggantikan
satu bandwidth global, plus magic clip, kernel prior eksplisit, gamma
adaptif, dan bobot usia — semuanya dari Bergstra dkk. (2011).

Sintetis, budget 30, 40 seed: **tpe2 0.7844** vs tpe v1 0.7077
(Δ=+0.077, **p=0.000**, tiga kali lantai noise) dan vs optuna_tpe 0.7637
(Δ=+0.021, p=0.389 — di bawah lantai noise, jadi **seri**).
Klaim yang sah: *setara rujukan industri*, bukan mengalahkannya.

**Arsitektur rakitan sendiri bersaing.** Pada successive halving di data
nyata, CNN `tiny` (residual, depth 3–4) mencapai 0.598 @3 epoch mengalahkan
resnet18 terbaik 0.540 — dengan budget 18 epoch-unit vs 27 bila semua
dilatih penuh.

---

## Batasan (jujur)

1. **Satu dataset** (Intel Image Classification, 6 kelas). Klaim lintas-tugas
   butuh minimal 3–5 dataset dengan karakter berbeda.
2. **Fidelity rendah**: 1–9 epoch, akurasi belum konvergen. Semua pernyataan
   soal proxy dan peringkat terikat pada rezim ini.
3. **Model kecil** (0.2–11 juta parameter), gambar 32–64 px.
4. **Validasi proxy n=14** per protokol — cukup untuk melihat pembalikan
   tanda, tidak cukup untuk memperkirakan besarnya korelasi dengan presisi.
5. **Ruang pencarian tunggal**, dengan gerbang `arch` yang dominan
   (mobilenet konsisten buruk di rezim epoch rendah). Ruang yang lebih rata
   bisa berperilaku lain.
6. Uji sign-flip mengasumsikan simetri di bawah H0; ia kuat tetapi
   konservatif dibandingkan uji-t berpasangan bila datanya normal.

Karena itu bukti ini setara **workshop paper**, bukan konferensi utama. Jalan
naiknya jelas dan tertulis di README: multi-dataset, fidelity konvergen,
lebih banyak seed, dan satu pembanding lagi yang lengang di literatur
(pencari berbasis LLM) pada protokol yang sama.

---

## Reproduksi

```bash
pip install -r requirements.txt && pip install optuna cmaes

# Instrumen: lantai noise + lantai daya uji + peringkat
py run.py bench sweep_dummy.yaml \
  --searchers random,optuna_random,tpe,tpe2,optuna_tpe,evolution \
  --budget 30 --seeds 40
py scripts/paired_test.py bench.json

# Validasi proxy terkontrol (hyperparameter dikunci)
py scripts/validate_proxy.py sweep_kaggle.yaml

# Gerbang proxy di halving: polos vs bergerbang, budget identik
py run.py halving sweep_kaggle.yaml --n0 9 --eta 3 --r0 1 --rmax 9 --seed 11
py run.py halving sweep_kaggle.yaml --n0 9 --eta 3 --r0 1 --rmax 9 --seed 11 \
  --pool 60 --proxy naswot

# Di GPU T4 Kaggle (butuh KAGGLE_API_TOKEN): lihat scripts/pow_suite.py,
# scripts/b40_suite.py, scripts/t4_suite.py
```
