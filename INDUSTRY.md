# Audit sweep-mu dalam 2 menit

Untuk tim yang menjalankan hyperparameter/architecture sweep dan harus
memutuskan: **"apakah pemenang sweep ini nyata, atau kebetulan?"**

Bukan alat pengganti Optuna/Ray/W&B — alat pendamping. Ia menerima hasil
sweep yang **sudah kamu punya** dan mengeluarkan verdict.

---

## Kenapa ini penting sekarang

Tiga angka dari eksperimen kami sendiri (protokol lengkap di
[FINDINGS.md](FINDINGS.md), audit mentah di `AUDIT_dummy40.md` dan
`AUDIT_real20.md`):

| Temuan | Angka |
|---|---|
| Peluang sweep 3-seed menobatkan pemenang yang **salah** | **71%** (data nyata), 60% (sintetis) |
| Peluang sweep 5-seed menobatkan pemenang yang salah | 67% / 51% |
| Seed yang dibutuhkan untuk mengklaim perbaikan 2% | **105** (data nyata) |
| Seed yang dibutuhkan untuk mengklaim perbaikan 1% | **>200** |

Artinya: keputusan "sampler A lebih baik dari sampler B" yang diambil dari
3–5 seed itu **lebih dekat ke lempar koin daripada ke pengukuran**. Kalau
keputusan itu dipakai untuk memilih pipeline produksi, biaya kekeliruannya
bukan cuma akurasi — tapi juga waktu engineer dan tagihan GPU untuk
mengejar perbaikan yang tidak ada.

---

## Tiga aturan yang bisa dipakai besok

### 1. Tambahkan lengan plasebo (biaya: satu arm)

Jalankan algoritme **yang sama** dua kali dengan RNG berbeda — beri nama
`baseline_a` dan `baseline_b`. Keduanya seharusnya seri. Selisih yang
terukur adalah **lantai noise** protokolmu.

Aturannya: **klaim di bawah lantai noise tidak boleh disebut kemenangan.**

Ini padanan kelompok plasebo dalam uji klinis. Murah, dan sekali dipasang
ia menjaga setiap keputusan sweep berikutnya.

### 2. Hitung MDE sebelum menjanjikan angka

Dari lengan plasebo, alat ini menghitung **MDE** (minimum detectable
effect): perbaikan terkecil yang mungkin terdeteksi pada jumlah seed-mu.
Kalau target OKR-mu "naik 1%" sementara MDE protokolmu 4%, target itu
tidak bisa diverifikasi — perbaiki protokolnya dulu, atau ganti targetnya.

### 3. Jangan pakai kurang dari 6 seed

Uji berpasangan atas n seed hanya punya 2ⁿ susunan tanda, jadi p terkecil
yang mungkin adalah 2/2ⁿ. Di bawah 6 seed, **p<0.05 mustahil** — seberapa
pun besar perbaikanmu. Angka "signifikan" dari 3 seed adalah kekeliruan
aritmetika, bukan temuan.

---

## Pakai

```bash
git clone https://github.com/dihannahdi/racik.git && cd racik
pip install pyyaml            # audit tidak butuh torch

# Ekspor hasil sweep jadi CSV panjang: arm,seed,score
#   arm   = nama varian yang dibandingkan (sampler, config, model)
#   seed  = ulangan; WAJIB dipasangkan (arm yang sama, seed yang sama)
#   score = hasil akhir run itu (akurasi/AUC/loss terbaik yang dicapai)
py scripts/audit.py hasil.csv --sham baseline_a,baseline_b --out AUDIT.md
```

Contoh CSV:

```csv
arm,seed,score
baseline_a,1,0.812
baseline_b,1,0.809
metode_baru,1,0.828
baseline_a,2,0.795
...
```

Ekspor sudah tersedia di hampir semua perkakas: Optuna (`study.trials_dataframe()`),
W&B (Export CSV di UI runs), Ray Tune (`ExperimentAnalysis.dataframe()`),
MLflow (`mlflow.search_runs()`). Ganti nama kolomnya jadi `arm,seed,score`.

Isi laporannya:

1. **Papan peringkat** — skor rata-rata per arm.
2. **Lengan plasebo** — lantai noise + varians null; plus peringatan bila dua
   algoritme identik justru berbeda signifikan (tanda protokolmu bocor).
3. **MDE** pada 3/5/10/20/40 seed, dan **jumlah seed yang dibutuhkan** untuk
   mengklaim delta 0.005 / 0.01 / 0.02 / 0.05.
4. **LII** — peluang studi k-seed menobatkan pemenang yang berbeda.
5. **Verdict per perbandingan** — signifikan / tidak / di bawah lantai noise.

---

## Yang harus dan tidak boleh disimpulkan

- Verdict "tidak signifikan" **bukan** berarti kedua arm sama. Ia berarti
  protokolmu belum bisa membedakannya. Perbaikannya: tambah seed, kurangi
  varians (fidelity lebih tinggi, subset lebih besar), atau terima bahwa
  pilihan itu tidak penting dan alihkan compute ke hal lain.
- Kalau semua arm-mu tidak signifikan, **itu informasi berharga**: pilih
  yang paling murah/sederhana, lalu belanjakan compute pada variabel yang
  benar-benar berpengaruh (ukuran data, fidelity, ruang pencarian).
- Alat ini mengukur **reliabilitas keputusan**, bukan kualitas model. Model
  terbaik tetap ditentukan oleh eval-mu; ini menjawab apakah peringkatnya
  bisa dipercaya.

## Catatan metode

Uji: sign-flip berpasangan 2-sisi (nonparametrik; eksak untuk n<=14, sampling
di atasnya). MDE dan jumlah seed: simulasi daya uji memakai varians null dari
lengan plasebo. LII: subsampling k dari n seed (2000 ulangan), dibandingkan
dengan pemenang pada seluruh seed. Asumsi: seed saling bebas dan berpasangan
lintas arm. Implementasi: [`racik/validity.py`](racik/validity.py) —
tanpa dependensi selain pustaka standar Python.
