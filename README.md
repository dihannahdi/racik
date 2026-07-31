# racik

Kerangka riset kecil untuk **algoritme pencarian arsitektur & hyperparameter model vision** — dibangun dari nol, bukan pembungkus library lain. Nama dari kata *meracik*: mencampur bahan sampai ketemu takaran yang pas.

> **[INDUSTRY.md](INDUSTRY.md) — audit sweep-mu dalam 2 menit.** Untuk tim yang
> harus memutuskan "apakah pemenang sweep ini nyata atau kebetulan". Menerima
> CSV `arm,seed,score` dari Optuna/W&B/Ray/MLflow. Angka pembukanya: sweep
> 3-seed menobatkan pemenang yang **salah 71%** dari waktu.
>
> **[FINDINGS.md](FINDINGS.md) — ringkasan temuan beserta batasannya.** Tiga
> instrumen kalibrasi (lantai noise, lantai daya uji, budget efektif) dan tiga
> kasus di mana kesimpulan berbalik ketika instrumen itu dipakai — dua di
> antaranya membatalkan klaim kami sendiri.

## Pertanyaan riset

Menyetel model vision itu kerja *mix and match*: coba kombinasi (arsitektur, learning rate, optimizer, augmentasi), lihat hasil, ubah, coba lagi. Di literatur ini disebut **hyperparameter optimization** dan **neural architecture search (NAS)**. Pertanyaan yang bisa dijawab kerangka ini:

> Pada budget percobaan yang sama, algoritme pencari mana yang paling cepat menemukan konfigurasi CNN terbaik?

## Algoritme yang diimplementasikan (dari nol, `racik/searchers.py`)

| Algoritme | Ide inti | Rujukan |
|---|---|---|
| `random` | Sampling acak murni — baseline yang wajib dikalahkan | Bergstra & Bengio, *Random Search for Hyper-Parameter Optimization*, JMLR 2012 |
| `evolution` | Regularized evolution: populasi FIFO (aging), seleksi turnamen, mutasi satu gen | Real dkk., *Regularized Evolution for Image Classifier Architecture Search*, AAAI 2019 |
| `tpe` | Tree-structured Parzen Estimator: modelkan distribusi konfigurasi "bagus" vs "buruk", pilih kandidat dengan rasio kepadatan tertinggi | Bergstra dkk., *Algorithms for Hyper-Parameter Optimization*, NeurIPS 2011 |

Semuanya memakai antarmuka **ask/tell** yang sama: `ask()` mengusulkan konfigurasi, `tell(config, skor)` melaporkan hasilnya. Menambah algoritme baru = menulis satu kelas dengan dua metode itu.

## Ruang pencarian

Didefinisikan di `sweep.yaml` (`racik/space.py` menangani sampling & mutasi):

- **Hyperparameter**: `lr` (log-uniform), `optimizer`, `augment`
- **Arsitektur** — bagian mix & match yang sesungguhnya: `arch` memilih antara
  CNN rakitan sendiri (`tiny`) atau model torchvision (`resnet18`, `mobilenet_v3_small`).
  Untuk `tiny`, dirakit dari: `depth`, `width`, `block` (basic / residual / depthwise),
  dan `use_se` (Squeeze-and-Excitation).

Catatan penting: ruang ini **bersyarat** — parameter tiny hanya berpengaruh jika `arch=tiny`. Ini disengaja, karena ruang bersyarat adalah kesulitan nyata di NAS.

## Fungsi objektif (dua backend)

1. **`dummy`** (`racik/dummy_backend.py`) — permukaan skor sintetis yang optimumnya kita ketahui (≈0.90). Gunanya: memverifikasi implementasi algoritme dalam hitungan detik, tanpa GPU. Prinsip riset: jangan bakar GPU untuk algoritme yang belum terbukti benar.
2. **`vision`** (`racik/vision_backend.py`) — training sungguhan: bangun CNN sesuai konfigurasi, latih di CIFAR-10 (atau folder gambar sendiri), skor = akurasi validasi. Default memakai subset kecil + 1 epoch supaya satu trial selesai dalam menit; naikkan setelah kandidat juara ketemu.

## Hasil eksperimen (dummy, budget 30 trial)

Riwayat perbaikan — tiap versi diukur ulang dengan protokol yang sama:

| Versi | random | tpe | evolution | Catatan |
|---|---|---|---|---|
| v0.2 (5 seed) | 0.6997 | 0.7060 | 0.7776 | TPE ≈ random: buta struktur bersyarat |
| v0.3 + ruang bersyarat (5 seed) | 0.6690 | 0.7034 | 0.7816 | belum cukup |
| v0.3 + TPE prior-mixture (10 seed) | 0.7050 | **0.7418** | **0.7838** | urutan sesuai teori |

Dua temuan:

1. **Evolution unggul konsisten di ruang bersyarat yang menipu** (plateau
   `arch=resnet18` di 0.55 menjebak pencari yang exploit terlalu dini).
   Mutasi mewarisi "gen" arch induk, jadi evolution jarang terjebak.
2. **TPE butuh dua perbaikan dari literatur untuk sekadar bekerja di sini**:
   (a) kepadatan dihitung hanya atas observasi yang memuat parameter itu
   (sadar-kondisi), dan (b) prior seragam ikut sebagai komponen campuran
   berbobot satu observasi — eksplorasi otomatis besar di awal, mengecil sendiri.
   Tanpa keduanya, TPE tak terbedakan dari random.

**Successive halving** (n0=27, eta=3, r0=1, rmax=9 pada dummy): menyaring
27 kandidat dengan **81 epoch-unit**, dibanding 243 kalau semua dilatih penuh —
compute 3× lebih hemat untuk cakupan kandidat yang sama.

## Hasil di data nyata (Kaggle: Intel Image Classification)

Dataset: 6 kelas pemandangan, 800 train + 150 val per kelas, gambar 32px,
1500 sampel train, 1 epoch per trial, CPU.

**Benchmark antar-algoritme** (budget 10 trial, 2 seed — dijalankan di VPS,
arsip di `results_server/`):

| Algoritme | Rata-rata skor terbaik |
|---|---|
| evolution | **0.4592 ± 0.081** |
| random | 0.3850 ± 0.024 |
| tpe | 0.3641 ± 0.006 |

Konsisten dengan hasil dummy: evolution unggul. TPE tenggelam di budget
sekecil ini (5 dari 10 trialnya habis untuk warmup acak) — sesuai ekspektasi
teori, bukan anomali. Catatan: 2 seed masih terlalu tipis untuk klaim keras.

**Successive halving di data nyata** (n0=9, eta=3, r0=1→3 epoch, 18 vs 27
epoch-unit): juara akhirnya justru **arsitektur `tiny` rakitan sendiri**
(residual, depth 4, width 16) — akurasi **0.598** @3 epoch, mengalahkan
resnet18 terbaik (0.540). Arsitektur hasil mix-and-match menang atas baseline
torchvision pada budget yang sama.

## Cara pakai

```bash
pip install -r requirements.txt

# 1. Sanity-check algoritme (tanpa GPU, detik)
py run.py bench sweep_dummy.yaml --budget 30 --seeds 5

# 2. Pencarian sungguhan di CIFAR-10
py run.py search sweep.yaml --searcher evolution --budget 20

# 3. Successive halving: saring banyak kandidat dengan budget bertingkat
py run.py halving sweep.yaml --n0 9 --eta 3 --r0 1 --rmax 4

# 4. Benchmark antar-algoritme pada training sungguhan (lama; siapkan kesabaran)
py run.py bench sweep.yaml --budget 15 --seeds 3

# 5. Data Kaggle sendiri (folder-per-kelas di data/kaggle/train|val)
py run.py search sweep_kaggle.yaml --searcher evolution --budget 8
```

Keluaran: `REPORT.md` + `results.json` (per pencarian), `BENCH.md` + `bench.json` (per benchmark). Hasil evaluasi di-cache di `.racik_cache/` — konfigurasi yang sama tidak pernah dilatih dua kali.

## Posisi terhadap alat yang sudah ada

Optuna, Ray Tune, dan W&B Sweeps jauh lebih matang untuk produksi (ratusan trial, terdistribusi, pruning). racik sengaja kecil: seluruh algoritmenya ditulis sendiri dan bisa dibaca habis, karena tujuannya **riset dan pemahaman**, bukan menggantikan mereka. Kalau butuh skala, hasil di sini tetap valid sebagai prototipe sebelum pindah ke Optuna.

## Zero-cost proxies: menilai arsitektur tanpa training (v0.4)

Pain point terbesar yang tersisa: mengetahui sebuah arsitektur jelek saja
harus membayar 1 epoch training. `racik/proxies.py` mengimplementasikan dua
proxy tanpa-training dari nol — **NASWOT** (logdet pola aktivasi ReLU;
Mellor dkk., ICML 2021) dan **synflow** (Tanaka dkk., NeurIPS 2020) — lalu
memakainya sebagai **rung -1 gratis** di successive halving
(`py run.py halving ... --pool 60 --proxy naswot`).

**Disiplin validasinya adalah temuannya.** Proxy tidak dipercaya buta;
diadili dulu terhadap akurasi hasil training sendiri (`py run.py proxycheck`),
dan hasilnya mengejutkan:

| Proxy | Validasi tercemar (n=9, hyperparam bervariasi) | Validasi terkontrol (n=14, hyperparam dikunci) |
|---|---|---|
| NASWOT | −0.786 | **+0.449** |
| synflow | +0.449 | **−0.275** |

Kedua proxy **berbalik tanda** saat eksperimen dibersihkan (proxy hanya
melihat arsitektur — membandingkannya pada sampel yang hyperparameternya
ikut berubah itu tidak sah). Validasi yang tercemar bukan sekadar kurang
akurat: dia menunjuk proxy yang salah. Kualitas gerbang NASWOT terukur:
top-7 dari 14 arsitektur memuat juara sejati (acc 0.44) dengan rata-rata
kandidat 0.337 vs 0.248 di bottom-7 — kenaikan kualitas kandidat gratis.

Catatan fidelity: rho +0.45 diukur terhadap akurasi-1-epoch (didominasi
kecepatan belajar). Diuji ulang di GPU — lihat verdict di bawah.

## Verdict GPU T4 (Kaggle, data 5x lebih besar, 3-9 epoch)

Suite `scripts/t4_suite.py` (2x Tesla T4, ~19 menit, ~80 evaluasi;
arsip di `results_server/t4/`):

**Benchmark algoritme** (12 trial x 2 seed @3 epoch):

| Algoritme | Rata-rata skor terbaik |
|---|---|
| **tpe** | **0.7171 ± 0.061** |
| evolution | 0.7100 ± 0.077 |
| random | 0.6275 ± 0.026 |

TPE terbayar begitu budget melewati warmup-nya (5 trial acak) — konsisten
dengan teorinya. Kedua algoritme model-based kini jelas di atas random.

**Temuan utama — gerbang proxy GAGAL, dan kegagalannya informatif:**

| Eksperimen | Hasil |
|---|---|
| Halving polos (seed 11, 27 epoch-unit) | terbaik **0.6450** |
| Halving + gerbang NASWOT (pool 60, seed & budget sama) | terbaik **0.5533** |
| Validasi ulang NASWOT @3 epoch | rho **−0.525** (lokal @1 epoch: +0.449) |

Gerbang NASWOT meloloskan **9/9 mobilenet** dari 60 kandidat — keluarga yang
paling lambat belajar di rezim epoch rendah (semua 0.18 di rung pertama).
Kesimpulan yang bisa dipertahankan: **korelasi zero-cost proxy tidak stabil**
— berbalik tanda karena (a) kontaminasi hyperparameter, (b) perubahan
fidelity/ukuran data — dan biasnya sistematik per keluarga arsitektur.
Di ruang pencarian campuran (torchvision + custom) pada fidelity praktis
(1–9 epoch), gerbang proxy lebih buruk daripada tanpa gerbang.

Ini *negative result* dengan protokol reproducible — kandidat tesis paper:
*"kapan dan mengapa zero-cost proxies menyesatkan"*. Bukti masih level
workshop (1 dataset, n=14 per validasi, model kecil); untuk klaim penuh
perlu: beberapa dataset, lebih banyak seed, korelasi vs akurasi konvergen.

## Adu langsung vs Optuna — dan temuan terpenting proyek ini (v0.5)

Pertanyaan yang wajar: *apakah implementasi kita lebih baik dari yang sudah ada?*
`racik/baselines.py` membungkus sampler Optuna 4.9 ke antarmuka ask/tell yang
sama, jadi TPE/evolution kita diadu pada ruang, budget, dan seed identik.

**Kalibrasi lantai noise.** `random` dan `optuna_random` adalah algoritme yang
*persis sama* (sampling seragam) — mereka harus seri. Selisih terukurnya
adalah lantai noise protokol: **0.025**. Selisih apa pun di bawah itu tidak
boleh diklaim sebagai kemenangan. Kalibrasi ini hampir tak pernah dilakukan
di literatur NAS/HPO, padahal murah.

**Peringkat berbalik saat seed ditambah** (dummy, budget 30, uji sign-flip):

| Pencari | 15 seed | 40 seed |
|---|---|---|
| optuna_tpe | 0.7506 (ke-3) | **0.7637 (ke-1)** |
| optuna_cmaes | 0.7508 (ke-2) | 0.7494 (ke-2) |
| **evolution (kita)** | **0.7689 (ke-1)** | 0.7385 (ke-3) |
| optuna_random | 0.7401 | 0.7336 |
| random (kita) | 0.7144 | 0.7082 |
| **tpe (kita)** | 0.7375 | 0.7077 (terakhir) |

Pada 15 seed kita "menang". Pada 40 seed **TPE Optuna signifikan lebih baik
dari TPE kita** (Δ=0.056, p=0.029), dan evolution kita turun ke posisi tiga
tanpa selisih signifikan terhadap siapa pun. Jawaban jujurnya: **tidak,
framework kita belum lebih baik** — dan papan peringkat 15-seed itu noise.

**Ini temuan sentral proyek, dan ia konsisten dengan temuan proxy kita:**
di rezim ini *jumlah seed menentukan siapa juaranya*. Fenomena sign-flip yang
kita temukan pada validasi zero-cost proxy (kontaminasi hyperparameter,
lalu fidelity) muncul lagi pada **peringkat algoritme**. Dengan 15 seed —
lebih banyak dari yang lazim dipakai orang — kita hampir menerbitkan klaim
yang salah arah. Kalibrasi lantai noise adalah penangkalnya.

**Lantai daya uji (power floor) — konsekuensi yang bisa dihitung, bukan opini.**
Uji sign-flip 2-sisi atas n pasang hanya punya 2^n susunan tanda, jadi p
terkecil yang *mungkin* dicapai adalah 2/2^n — berapa pun besar efeknya:

| n seed | 3 | 4 | 5 | **6** | 10 | 20 |
|---|---|---|---|---|---|---|
| p minimum | 0.250 | 0.125 | 0.063 | **0.031** | 0.002 | ~1e-6 |

Artinya **dengan kurang dari 6 seed, p<0.05 mustahil dicapai** — sementara
3–5 seed adalah praktik lazim di makalah NAS/HPO. Validasi kita sendiri di
data nyata dengan 3 seed membuktikannya: keenam perbandingan mentok di
p=0.25–0.75, tak satu pun bisa signifikan. `scripts/paired_test.py` kini
mencetak peringatan otomatis bila n terlalu kecil.

Kandidat tesis paper: *"Rank instability in NAS/HPO benchmarks: a noise-floor
calibration protocol"* — tiga pilar, semuanya terukur di repo ini:
(1) lantai noise dari dua implementasi algoritme identik, (2) lantai daya uji
2/2^n, (3) dua studi kasus sign-flip (validasi proxy dan peringkat searcher).
Reproduksi: `py run.py bench ... --seeds 40` lalu
`py scripts/paired_test.py bench.json`.

## Menutup celah: TPE v2 (`racik/tpe2.py`)

Defisit v1 signifikan (bukan noise), jadi bisa diperbaiki dan diverifikasi.
Bedah sumber Optuna menunjukkan penyebabnya **algoritmik, bukan tuning** —
empat hal dari Bergstra dkk. (2011) yang hilang di v1:

| Aspek | v1 | v2 (formulasi asli) |
|---|---|---|
| Bandwidth kernel | satu global `rentang/√n` | **adaptif per titik**: `max(jarak tetangga kiri, kanan)` |
| Batas bawah sigma | tidak ada | magic clip `rentang/min(100, n+2)` |
| Prior | campuran seragam ad hoc | kernel Gaussian eksplisit di tengah domain |
| Pembagi good/bad | gamma tetap 0.25 | `min(⌈0.1n⌉, 25)` + bobot meluruh menurut usia |

Bandwidth adaptif adalah yang paling berpengaruh: daerah padat observasi
mendapat kernel tajam (eksploitasi presisi), daerah renggang mendapat kernel
lebar (tetap menjelajah). Satu bandwidth global memaksa kompromi buruk di
kedua rezim. Kernelnya normal **terpancung** pada domain, jadi massa
probabilitas tidak bocor keluar batas.

**Hasil (dummy, budget 30, 40 seed, protokol identik):**

| Pencari | Skor | vs tpe2 |
|---|---|---|
| **tpe2 (kita)** | **0.7844** | — |
| optuna_tpe | 0.7637 | −0.021, p=0.389 (di bawah lantai noise → **seri**) |
| optuna_cmaes | 0.7494 | −0.035, p=0.026 (signifikan) |
| evolution (kita) | 0.7385 | −0.046, p=0.010 (signifikan) |
| optuna_random | 0.7336 | −0.051, p=0.011 (signifikan) |
| random | 0.7082 | −0.076, p=0.000 (signifikan) |
| tpe v1 (kita) | 0.7077 | **−0.077, p=0.000 (signifikan)** |

Bacaan yang jujur: **v2 mengalahkan v1 dengan sangat tegas** (Δ=0.077, tiga
kali lantai noise, p=0.000) — perbaikannya nyata, bukan keberuntungan. Terhadap
TPE Optuna, v2 **nominal di depan tapi statistik seri** (Δ=0.021 di bawah
lantai noise 0.025): celah yang tadinya signifikan tertinggal kini tertutup.
Klaim yang boleh ditulis: *implementasi kami setara rujukan industri* — bukan
"mengalahkan". Terhadap CMA-ES dan evolution kita sendiri, v2 signifikan unggul.

## Temuan terkuat: di data nyata, tak ada pencari yang mengalahkan random —
## dan penyebabnya bisa dihitung (v0.7)

Benchmark data nyata dengan **daya uji memadai** (Intel Image, 6 pencari,
budget 12, **20 seed**, GPU T4; `p` minimum yang mungkin = 0.00002, jadi
signifikansi benar-benar terjangkau):

| Pencari | Skor |
|---|---|
| evolution | 0.3757 |
| optuna_random | 0.3703 |
| optuna_tpe | 0.3688 |
| tpe2 | 0.3650 |
| random | 0.3612 |
| tpe v1 | 0.3558 |

Lantai noise (random vs optuna_random): **0.0092**.
**Dari 15 perbandingan berpasangan, NOL yang signifikan.** Bahkan
evolution vs random hanya p=0.186. Seluruh rentang papan peringkat (0.020)
hanya dua kali lantai noise.

Bandingkan dengan objektif sintetis (budget 30, 40 seed) di mana tpe2 dan
evolution menang dengan p≤0.01. **Permukaan sintetis melebih-lebihkan
perbedaan antar-algoritme; tugas nyata meratakannya.** Itu temuan tentang
validitas benchmark, bukan tentang algoritmenya.

**Diagnosis "budget efektif" — terukur, bukan spekulasi.** Kami hitung
berapa banyak seed yang kurvanya masih identik dengan random di tiap trial:

| Pencari (warmup) | t=5 | t=10 | t=12 |
|---|---|---|---|
| tpe2 (10) | 20/20 identik | 20/20 identik | 12/20 identik |
| tpe v1 (5) | 20/20 | 7/20 | 4/20 |
| evolution (populasi 8) | 20/20 | 8/20 | 6/20 |

Pada budget 12, **tpe2 hanya membuat 2 keputusan berbasis model** — sisanya
warmup acak. Jadi benchmark itu sebagian besar mengukur warmup, bukan
algoritmenya. Rumusnya sederhana dan layak jadi aturan praktis:

> **budget efektif = budget − warmup.** Membandingkan pencari berbasis model
> pada budget yang tidak jauh lebih besar dari warmup-nya berarti
> membandingkan random search dengan nama yang berbeda.

Uji lanjutan (satu variabel: budget 12 → **40**, jadi budget efektif 30)
sedang berjalan di T4: `scripts/b40_suite.py`.

## Fitur v0.3

- **Ruang bersyarat** (`when:` di sweep.yaml) — parameter mati dibuang dari
  bentuk kanonik konfigurasi, jadi dua konfigurasi yang hanya beda di parameter
  mati tidak membuang budget dua kali.
- **Successive halving** (`py run.py halving ...`) — Jamieson & Talwalkar 2016;
  fidelity = jumlah epoch, cache sadar-fidelity.
- **TPE sadar-kondisi + prior-mixture** — lihat tabel hasil di atas.

## Arah lanjut (berurut dari yang paling bernilai)

1. **ASHA asinkron** (Li dkk., 2018) — versi paralel dari halving; plus
   Hyperband (beberapa bracket halving sekaligus).
2. **Searcher keempat: LLM-guided search** — agen membaca riwayat hasil dan mengusulkan konfigurasi berikutnya (pola GEPA dari dunia prompt, dipindah ke NAS), lalu diadu lawan ketiga algoritme klasik pada budget sama. Ini celah riset yang masih lengang.
3. **Weight sharing / supernet** (ENAS, DARTS) — kalau mau masuk NAS modern sungguhan.
