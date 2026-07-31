# Audit validitas benchmark — Intel Image, budget 40, 20 seed

n seed = **20**  |  p terkecil yang mungkin = **0.00000**

## Papan peringkat

| Pencari | Skor akhir rata-rata |
|---|---|
| optuna_tpe | 0.4354 |
| tpe2 | 0.4351 |
| tpe | 0.4218 |
| optuna_random | 0.4125 |
| evolution | 0.4118 |
| random | 0.4112 |

## Lengan plasebo (dua implementasi algoritme identik)

Pasangan: **random vs optuna_random** — keduanya seharusnya seri.

- Lantai noise (|Δ| terukur): **0.0013**
- Simpangan baku null (bahan analisis daya uji): **0.0517**
- p uji plasebo: 0.908 (sesuai harapan, tidak signifikan)

Klaim apa pun di bawah lantai noise tidak boleh disebut kemenangan.

## Efek terkecil yang terdeteksi (MDE, alpha=0.05, daya 0.8)

| n seed | MDE |
|---|---|
| 3 | mustahil (lantai daya uji) |
| 5 | mustahil (lantai daya uji) |
| 10 | 0.0517 |
| 20 | 0.0414 |
| 40 | 0.0310 |

Klaim yang lebih kecil dari MDE pada n seed kamu tidak terdukung — bukan karena efeknya tidak ada, tetapi karena protokolnya tidak mampu melihatnya.

## Seed yang dibutuhkan untuk mengklaim efek sebesar delta

| delta | n seed minimum |
|---|---|
| 0.005 | >200 |
| 0.010 | >200 |
| 0.020 | 75 |
| 0.050 | 12 |

## Indeks ketidakstabilan papan peringkat (LII)

Acuan: juara pada seluruh 20 seed = **optuna_tpe**. Kolom kedua: peluang sebuah studi dengan k seed menobatkan juara yang BERBEDA.

| k seed | P(juara berubah) | Spearman rata-rata vs acuan |
|---|---|---|
| 3 | **66.7%** | 0.415 |
| 5 | **63.1%** | 0.526 |
| 8 | **56.8%** | 0.687 |
| 10 | **53.2%** | 0.746 |
| 15 | **49.0%** | 0.843 |
| 20 | **0.0%** | 1.000 |

## Verdict per perbandingan

| A | B | delta | p | Verdict |
|---|---|---|---|---|
| random | tpe2 | -0.0238 | 0.028 | signifikan |
| random | optuna_tpe | -0.0242 | 0.046 | signifikan |
| optuna_random | optuna_tpe | -0.0229 | 0.089 | tidak signifikan |
| optuna_random | tpe2 | -0.0226 | 0.121 | tidak signifikan |
| optuna_tpe | evolution | +0.0236 | 0.127 | tidak signifikan |
| tpe2 | evolution | +0.0232 | 0.132 | tidak signifikan |
| tpe | optuna_tpe | -0.0137 | 0.236 | tidak signifikan |
| random | tpe | -0.0105 | 0.282 | tidak signifikan |
| tpe | tpe2 | -0.0133 | 0.304 | tidak signifikan |
| optuna_random | tpe | -0.0093 | 0.419 | tidak signifikan |
| tpe | evolution | +0.0099 | 0.477 | tidak signifikan |
| random | optuna_random | -0.0013 | 0.908 | tidak signifikan |
| random | evolution | -0.0006 | 0.952 | di bawah lantai noise |
| optuna_random | evolution | +0.0007 | 0.966 | di bawah lantai noise |
| tpe2 | optuna_tpe | -0.0003 | 0.978 | di bawah lantai noise |
