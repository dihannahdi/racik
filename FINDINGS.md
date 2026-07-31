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

| k seed | sintetis (n=40) | data nyata b12 (n=20) | data nyata b40 (n=20) |
|---|---|---|---|
| 3 | 60.1% | **71.4%** | 66.7% |
| 5 | 51.3% | 67.0% | 63.1% |
| 10 | 37.4% | 54.4% | 53.2% |
| 15 | 28.5% | 38.0% | **49.0%** |
| 30 | 7.7% | — | — |

**Sebuah studi 3-seed menobatkan pemenang yang salah pada 6–7 dari 10
kesempatan.** Itu bukan pengukuran, itu lempar koin dengan tabel.

Perhatikan kolom terakhir: pada budget 40, LII tetap **49% di 15 seed** —
lebih tinggi daripada protokol lain. Sebabnya justru karena dua kandidat
teratas hampir seri (tpe2 vs optuna_tpe, Δ=0.0003). **LII memuncak tepat
ketika perlombaan ketat** — yaitu kondisi yang paling sering diklaim sebagai
kemenangan di makalah. Karena itu LII layak dilaporkan berdampingan dengan
papan peringkat: ia mengukur seberapa banyak papan peringkat itu boleh
dipercaya.

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

### 2.3 Pada budget kecil tak ada yang mengalahkan random — pada budget cukup, TPE menang

Dua protokol, **satu variabel** yang berbeda (budget 12 → 40), data dan seed
sama, 20 seed di GPU T4:

| | budget 12 (efektif ~2) | budget 40 (efektif ~30) |
|---|---|---|
| Perbandingan signifikan | **0 dari 15** | **2 dari 15** |
| Lantai noise | 0.0092 | **0.0013** |
| tpe2 vs random | −0.0038, p=0.736 | **+0.0238, p=0.028** ✔ |
| optuna_tpe vs random | −0.0077, p=0.634 | **+0.0242, p=0.046** ✔ |
| evolution vs random | +0.0145, p=0.183 | +0.0006, p=0.952 (di bawah lantai) |
| tpe2 vs optuna_tpe | −0.0038, p=0.799 | **+0.0003, p=0.978** (seri persis) |

Tiga hal yang bisa disimpulkan, semuanya baru:

1. **Hasil nol di budget 12 memang artefak budget efektif**, bukan bukti
   bahwa algoritme tak berguna. Begitu diberi 30 keputusan berbasis model
   (bukan 2), keluarga TPE memisahkan diri dari random secara signifikan.
   Pesan praktisnya presisi: *algoritme tidak penting pada budget kecil* —
   dan budget kecil adalah kondisi paling umum di lapangan.
2. **Kemenangan evolution di budget 12 (0.4328, "juara") ternyata noise.**
   Di budget 40 ia tepat setara random (Δ=0.0006, di bawah lantai noise).
   Papan peringkat budget-12 menobatkan pemenang yang salah — persis yang
   diramalkan LII.
3. **TPE v2 kami setara persis TPE Optuna di data nyata**: Δ=0.0003, p=0.978.
   Di objektif sintetis selisihnya 0.021 (juga tidak signifikan); di data
   nyata praktis nol. Klaim *setara rujukan industri* kini punya dua bukti
   independen.

Besar efeknya tetap sederhana: **+0.024 akurasi** dari mengganti random
dengan TPE pada budget 40. Itu nyata, tetapi jauh lebih kecil daripada yang
biasa disiratkan makalah — dan butuh ~75 seed untuk diklaim andal bila
lengan-lengannya tidak berbagi keacakan (lihat catatan berikut).

**Replikasi di dataset kedua (CIFAR-10, fidelity terkalibrasi 5 epoch/5000,
budget 12, 20 seed).** Polanya sama seperti Intel Image di budget 12: dari 15
perbandingan, hanya **1** yang signifikan (`optuna_tpe` vs `optuna_random`,
Δ=0.027, p=0.001), sedangkan `optuna_tpe` vs `random` sendiri belum signifikan
(Δ=0.022, p=0.126). Papan peringkatnya membentang 0.028 saja. Jadi temuan
"budget kecil = pilihan algoritme hampir tak berpengaruh" **bertahan di dua
dataset**, bukan khas satu tugas.

Catatan penting: nilai ini diperoleh setelah fidelity CIFAR dikalibrasi
(akurasi ~0.50, jauh di atas lantai tebakan 0.10). Tanpa kalibrasi itu,
angkanya akan tampak mendukung kesimpulan yang sama karena alasan yang salah.

### 2.4 Catatan halus: lengan plasebo bisa terlalu konservatif

Audit budget-40 memperkirakan MDE=0.0414 pada 20 seed, tetapi kami *berhasil*
mendeteksi Δ=0.0238 dengan p=0.028. Bukan kontradiksi — penyebabnya penting:

Pencari racik **berbagi RNG warmup**, jadi `random` dan `tpe2` menyampel
konfigurasi awal yang identik pada seed yang sama. Selisih berpasangannya
karena itu berkorelasi tinggi dan variansnya kecil. Sementara lengan plasebo
kami (`random` vs `optuna_random`) memakai dua aliran RNG yang **independen**,
sehingga variansnya lebih besar.

Konsekuensinya, ada dua jenis lengan plasebo:

| Jenis | Sifat | Kapan dipakai |
|---|---|---|
| RNG independen | batas **atas** varians null → MDE konservatif | bila arm-mu tidak berbagi keacakan |
| RNG bersama (seed dipasangkan) | varians null lebih ketat, daya uji lebih tinggi | bila arm-mu berbagi warmup/data order |

MDE dari lengan plasebo independen aman dipakai sebagai batas konservatif —
klaim di atasnya pasti terdukung. Tetapi klaim di bawahnya belum otomatis
gugur bila arm-nya berbagi keacakan; ukur variansnya langsung.

**Pelajaran praktis untuk industri:** *pasangkan* keacakan antar arm bila
bisa (seed sama, urutan data sama, warmup sama). Itu menaikkan daya uji
tanpa satu pun training tambahan — cara termurah memperbaiki keandalan
keputusan sweep.

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

## Perangkap yang kami temukan pada diri sendiri: efek lantai lintas-dataset

Saat menyiapkan dataset kedua (CIFAR-10) untuk uji lintas-tugas, protokolnya
kami samakan persis dengan yang pertama — termasuk **1 epoch dan 1500 sampel
train**. Smoke-test-nya menghasilkan akurasi **0.1233**, nyaris lantai tebakan
acak untuk 10 kelas (0.10).

Di rezim itu benchmark akan melaporkan "tak ada pencari yang berbeda" — tetapi
karena **tidak ada sinyal untuk dibedakan**, bukan karena algoritmenya setara.
Kesimpulan yang benar dengan alasan yang salah tetap kesimpulan yang cacat,
dan sulit dibedakan dari temuan asli §2.3.

Pelajarannya berlaku umum untuk siapa pun yang membandingkan lintas dataset:

> Menyamakan **jumlah epoch** antar dataset yang kesulitannya berbeda bukan
> perbandingan yang adil. Yang harus disamakan adalah **rezim akurasinya** —
> kalibrasi fidelity tiap dataset sampai akurasi acuannya berada di band yang
> sebanding, lalu bandingkan di sana.

`scripts/calibrate_fidelity.py` melakukan kalibrasi itu dari bukti: ia mencoba
beberapa setelan (epoch, ukuran subset) dan melaporkan akurasi acuan tiap
setelan, sehingga protokol dataset kedua dipilih dari pengukuran, bukan tebakan.

## Karya terdahulu — apa yang BUKAN baru dari kami

Diperiksa setelah instrumen di Bagian 1 dibangun. Hasilnya menuntut kami
mengecilkan klaim kebaruan, dan itu dicatat di sini alih-alih disembunyikan.

| Instrumen kami | Karya terdahulu |
|---|---|
| Lantai daya uji, MDE, jumlah seed | **Colas dkk. (2018)**, *How Many Random Seeds? Statistical Power Analysis in Deep RL* — analisis daya uji untuk seed sudah dibahas tuntas di sana. Batas p minimum uji berpasangan pada n kecil juga sudah dilaporkan orang lain (mis. Wilcoxon pada 5 seed). |
| Varians benchmark & rekomendasi protokol | **Bouthillier dkk. (MLSys 2021)**, *Accounting for Variance in ML Benchmarks* — memodelkan sumber varians (sampling data, inisialisasi, hyperparameter) dan memberi rekomendasi perbandingan. |
| Ketidakstabilan peringkat via resampling | Sudah ada beberapa: bootstrap peringkat pada MS MARCO; audit benchmark deteksi depresi yang melaporkan peluang bootstrap konfigurasi terbaik benar-benar rank-1 hanya **0.323**; *Quantifying Ranking Uncertainty in LLM Benchmarks* (2026); *Unstable Rankings in Bayesian Deep Learning Evaluation* (2026). |
| Kerangka membandingkan banyak optimizer | **CARP-S** (2025) — kerangka membandingkan N optimizer HPO pada M benchmark. |

**Jadi tiga instrumen kami bukan penemuan baru.** Yang tersisa sebagai
kontribusi, dan lebih sempit dari kesan awal:

1. **Lengan plasebo sebagai arm kontrol eksplisit** — menjalankan dua
   *implementasi berbeda dari algoritme yang sama* dan memakai selisihnya
   sebagai lantai noise sekaligus taksiran varians null. Karya terdahulu
   memodelkan varians dari sumbernya; kami mengukurnya langsung dari satu
   arm kontrol, seperti kelompok plasebo. Belum kami temukan padanannya,
   tetapi belum bisa kami pastikan tidak ada.
2. **Diagnosis budget efektif** (= budget − warmup), lengkap dengan
   pengukuran fraksi seed yang kurvanya masih identik dengan random.
   Praktis, dan menjelaskan hasil nol yang tanpa itu akan disalahartikan.
3. **Pengemasan**: audit yang menerima CSV `arm,seed,score` dari perkakas
   apa pun tanpa mengubah pipeline. Ini rekayasa, bukan sains — tetapi
   justru itu yang menentukan apakah metodenya dipakai orang.
4. **Hasil empiris spesifik** pada NAS/HPO vision nyata: TPE mengalahkan
   random hanya di atas budget efektif ~30; evolution tidak pernah; dan
   ketiga pembalikan di Bagian 2.

Konsekuensi untuk penulisan: makalah yang mengabaikan Bouthillier dkk. dan
Colas dkk. akan ditolak sebelum ditinjau. Posisi kami yang jujur adalah
**melanjutkan** garis itu ke ranah NAS/HPO vision dengan arm kontrol dan
alat yang bisa langsung dipakai, bukan mengklaim membuka jalan baru.

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
