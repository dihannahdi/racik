# racik

Kerangka riset kecil untuk **algoritme pencarian arsitektur & hyperparameter model vision** — dibangun dari nol, bukan pembungkus library lain. Nama dari kata *meracik*: mencampur bahan sampai ketemu takaran yang pas.

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
kecepatan belajar). Literatur melaporkan korelasi lebih tinggi terhadap
akurasi final — diuji di `scripts/t4_suite.py` (GPU T4 Kaggle, 3+ epoch).

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
