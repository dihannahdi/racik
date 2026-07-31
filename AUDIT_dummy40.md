# Audit validitas benchmark — objektif sintetis, budget 30, 40 seed

n seed = **40**  |  p terkecil yang mungkin = **0.00000**

## Papan peringkat

| Pencari | Skor akhir rata-rata |
|---|---|
| tpe2 | 0.7844 |
| optuna_tpe | 0.7637 |
| optuna_cmaes | 0.7494 |
| evolution | 0.7385 |
| optuna_random | 0.7336 |
| random | 0.7082 |
| tpe | 0.7077 |

## Lengan plasebo (dua implementasi algoritme identik)

Pasangan: **random vs optuna_random** — keduanya seharusnya seri.

- Lantai noise (|Δ| terukur): **0.0254**
- Simpangan baku null (bahan analisis daya uji): **0.0890**
- p uji plasebo: 0.076 (sesuai harapan, tidak signifikan)

Klaim apa pun di bawah lantai noise tidak boleh disebut kemenangan.

## Efek terkecil yang terdeteksi (MDE, alpha=0.05, daya 0.8)

| n seed | MDE |
|---|---|
| 3 | mustahil (lantai daya uji) |
| 5 | mustahil (lantai daya uji) |
| 10 | 0.0890 |
| 20 | 0.0712 |
| 40 | 0.0534 |

Klaim yang lebih kecil dari MDE pada n seed kamu tidak terdukung — bukan karena efeknya tidak ada, tetapi karena protokolnya tidak mampu melihatnya.

## Seed yang dibutuhkan untuk mengklaim efek sebesar delta

| delta | n seed minimum |
|---|---|
| 0.005 | >200 |
| 0.010 | >200 |
| 0.020 | >200 |
| 0.050 | 39 |

## Indeks ketidakstabilan papan peringkat (LII)

Acuan: juara pada seluruh 40 seed = **tpe2**. Kolom kedua: peluang sebuah studi dengan k seed menobatkan juara yang BERBEDA.

| k seed | P(juara berubah) | Spearman rata-rata vs acuan |
|---|---|---|
| 3 | **60.1%** | 0.492 |
| 5 | **51.3%** | 0.599 |
| 8 | **42.1%** | 0.688 |
| 10 | **37.4%** | 0.750 |
| 15 | **28.5%** | 0.824 |
| 20 | **20.0%** | 0.889 |
| 30 | **7.7%** | 0.952 |
| 40 | **0.0%** | 1.000 |

## Verdict per perbandingan

| A | B | delta | p | Verdict |
|---|---|---|---|---|
| random | tpe2 | -0.0762 | 0.000 | signifikan |
| tpe | tpe2 | -0.0767 | 0.000 | signifikan |
| tpe2 | evolution | +0.0460 | 0.008 | signifikan |
| optuna_random | tpe2 | -0.0508 | 0.011 | signifikan |
| random | optuna_tpe | -0.0555 | 0.015 | signifikan |
| random | optuna_cmaes | -0.0412 | 0.022 | signifikan |
| tpe | optuna_tpe | -0.0560 | 0.024 | signifikan |
| tpe2 | optuna_cmaes | +0.0350 | 0.026 | signifikan |
| tpe | optuna_cmaes | -0.0417 | 0.036 | signifikan |
| random | evolution | -0.0303 | 0.072 | tidak signifikan |
| random | optuna_random | -0.0254 | 0.076 | tidak signifikan |
| optuna_random | optuna_tpe | -0.0301 | 0.110 | tidak signifikan |
| tpe | evolution | -0.0308 | 0.155 | tidak signifikan |
| optuna_random | tpe | +0.0259 | 0.187 | tidak signifikan |
| optuna_tpe | evolution | +0.0252 | 0.290 | di bawah lantai noise |
| optuna_random | optuna_cmaes | -0.0158 | 0.369 | di bawah lantai noise |
| tpe2 | optuna_tpe | +0.0208 | 0.379 | di bawah lantai noise |
| optuna_cmaes | evolution | +0.0109 | 0.504 | di bawah lantai noise |
| optuna_tpe | optuna_cmaes | +0.0143 | 0.512 | di bawah lantai noise |
| optuna_random | evolution | -0.0048 | 0.795 | di bawah lantai noise |
| random | tpe | +0.0005 | 0.980 | di bawah lantai noise |
