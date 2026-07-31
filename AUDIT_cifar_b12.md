# Audit validitas benchmark — CIFAR-10 terkalibrasi, budget 12, 20 seed

n seed = **20**  |  p terkecil yang mungkin = **0.00000**

## Papan peringkat

| Pencari | Skor akhir rata-rata |
|---|---|
| optuna_tpe | 0.5219 |
| evolution | 0.5081 |
| tpe2 | 0.5075 |
| random | 0.5001 |
| optuna_random | 0.4947 |
| tpe | 0.4938 |

## Lengan plasebo (dua implementasi algoritme identik)

Pasangan: **random vs optuna_random** — keduanya seharusnya seri.

- Lantai noise (|Δ| terukur): **0.0055**
- Simpangan baku null (bahan analisis daya uji): **0.0495**
- p uji plasebo: 0.625 (sesuai harapan, tidak signifikan)

Klaim apa pun di bawah lantai noise tidak boleh disebut kemenangan.

## Efek terkecil yang terdeteksi (MDE, alpha=0.05, daya 0.8)

| n seed | MDE |
|---|---|
| 3 | mustahil (lantai daya uji) |
| 5 | mustahil (lantai daya uji) |
| 10 | 0.0495 |
| 20 | 0.0396 |
| 40 | 0.0297 |

Klaim yang lebih kecil dari MDE pada n seed kamu tidak terdukung — bukan karena efeknya tidak ada, tetapi karena protokolnya tidak mampu melihatnya.

## Seed yang dibutuhkan untuk mengklaim efek sebesar delta

| delta | n seed minimum |
|---|---|
| 0.005 | >200 |
| 0.010 | >200 |
| 0.020 | 54 |
| 0.050 | 12 |

## Indeks ketidakstabilan papan peringkat (LII)

Acuan: juara pada seluruh 20 seed = **optuna_tpe**. Kolom kedua: peluang sebuah studi dengan k seed menobatkan juara yang BERBEDA.

| k seed | P(juara berubah) | Spearman rata-rata vs acuan |
|---|---|---|
| 3 | **51.7%** | 0.411 |
| 5 | **43.1%** | 0.533 |
| 8 | **30.8%** | 0.659 |
| 10 | **23.1%** | 0.728 |
| 15 | **6.9%** | 0.882 |
| 20 | **0.0%** | 1.000 |

## Verdict per perbandingan

| A | B | delta | p | Verdict |
|---|---|---|---|---|
| optuna_random | optuna_tpe | -0.0272 | 0.001 | signifikan |
| tpe | optuna_tpe | -0.0281 | 0.088 | tidak signifikan |
| random | optuna_tpe | -0.0217 | 0.126 | tidak signifikan |
| tpe | tpe2 | -0.0137 | 0.148 | tidak signifikan |
| tpe | evolution | -0.0143 | 0.216 | tidak signifikan |
| optuna_random | evolution | -0.0135 | 0.311 | tidak signifikan |
| tpe2 | optuna_tpe | -0.0144 | 0.318 | tidak signifikan |
| optuna_random | tpe2 | -0.0129 | 0.319 | tidak signifikan |
| optuna_tpe | evolution | +0.0137 | 0.356 | tidak signifikan |
| random | tpe2 | -0.0074 | 0.391 | tidak signifikan |
| random | evolution | -0.0080 | 0.454 | tidak signifikan |
| random | tpe | +0.0064 | 0.579 | tidak signifikan |
| random | optuna_random | +0.0055 | 0.625 | tidak signifikan |
| optuna_random | tpe | +0.0009 | 0.947 | di bawah lantai noise |
| tpe2 | evolution | -0.0006 | 0.949 | di bawah lantai noise |
