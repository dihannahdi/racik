# Audit validitas benchmark — Intel Image, budget 12, 20 seed

n seed = **20**  |  p terkecil yang mungkin = **0.00000**

## Papan peringkat

| Pencari | Skor akhir rata-rata |
|---|---|
| evolution | 0.3757 |
| optuna_random | 0.3703 |
| optuna_tpe | 0.3688 |
| tpe2 | 0.3650 |
| random | 0.3612 |
| tpe | 0.3558 |

## Lengan plasebo (dua implementasi algoritme identik)

Pasangan: **random vs optuna_random** — keduanya seharusnya seri.

- Lantai noise (|Δ| terukur): **0.0092**
- Simpangan baku null (bahan analisis daya uji): **0.0700**
- p uji plasebo: 0.566 (sesuai harapan, tidak signifikan)

Klaim apa pun di bawah lantai noise tidak boleh disebut kemenangan.

## Efek terkecil yang terdeteksi (MDE, alpha=0.05, daya 0.8)

| n seed | MDE |
|---|---|
| 3 | mustahil (lantai daya uji) |
| 5 | mustahil (lantai daya uji) |
| 10 | 0.0700 |
| 20 | 0.0560 |
| 40 | 0.0420 |

Klaim yang lebih kecil dari MDE pada n seed kamu tidak terdukung — bukan karena efeknya tidak ada, tetapi karena protokolnya tidak mampu melihatnya.

## Seed yang dibutuhkan untuk mengklaim efek sebesar delta

| delta | n seed minimum |
|---|---|
| 0.005 | >200 |
| 0.010 | >200 |
| 0.020 | 105 |
| 0.050 | 16 |

## Indeks ketidakstabilan papan peringkat (LII)

Acuan: juara pada seluruh 20 seed = **evolution**. Kolom kedua: peluang sebuah studi dengan k seed menobatkan juara yang BERBEDA.

| k seed | P(juara berubah) | Spearman rata-rata vs acuan |
|---|---|---|
| 3 | **71.4%** | 0.259 |
| 5 | **67.0%** | 0.346 |
| 8 | **59.3%** | 0.463 |
| 10 | **54.4%** | 0.524 |
| 15 | **38.0%** | 0.761 |
| 20 | **0.0%** | 1.000 |

## Verdict per perbandingan

| A | B | delta | p | Verdict |
|---|---|---|---|---|
| tpe | evolution | -0.0198 | 0.130 | tidak signifikan |
| random | evolution | -0.0145 | 0.183 | tidak signifikan |
| tpe2 | evolution | -0.0107 | 0.394 | tidak signifikan |
| optuna_random | tpe | +0.0145 | 0.437 | tidak signifikan |
| tpe | optuna_tpe | -0.0130 | 0.452 | tidak signifikan |
| tpe | tpe2 | -0.0092 | 0.470 | tidak signifikan |
| random | optuna_random | -0.0092 | 0.566 | tidak signifikan |
| random | optuna_tpe | -0.0077 | 0.634 | di bawah lantai noise |
| optuna_tpe | evolution | -0.0068 | 0.640 | di bawah lantai noise |
| random | tpe | +0.0053 | 0.654 | di bawah lantai noise |
| optuna_random | optuna_tpe | +0.0015 | 0.731 | di bawah lantai noise |
| random | tpe2 | -0.0038 | 0.736 | di bawah lantai noise |
| optuna_random | evolution | -0.0053 | 0.743 | di bawah lantai noise |
| optuna_random | tpe2 | +0.0053 | 0.761 | di bawah lantai noise |
| tpe2 | optuna_tpe | -0.0038 | 0.799 | di bawah lantai noise |
