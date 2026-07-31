# Benchmark algoritme pencari (racik)

Tugas: Klasifikasi gambar dataset Kaggle: temukan konfigurasi CNN dengan akurasi validasi tertinggi pada budget training singkat (CPU-friendly).
Backend: vision  |  Budget: 12 trial  |  Seed: 20

Angka = rata-rata skor-terbaik-sejauh-ini pada trial ke-t (± simpangan baku).

| Trial | random | optuna_random | tpe | tpe2 | optuna_tpe | evolution |
|-------|------|------|------|------|------|------|
| 1 | 0.2303 ± 0.0705 | 0.2338 ± 0.0718 | 0.2303 ± 0.0705 | 0.2303 ± 0.0705 | 0.2338 ± 0.0718 | 0.2303 ± 0.0705 |
| 5 | 0.3144 ± 0.0434 | 0.3108 ± 0.0549 | 0.3144 ± 0.0434 | 0.3144 ± 0.0434 | 0.3108 ± 0.0549 | 0.3144 ± 0.0434 |
| 10 | 0.3511 ± 0.0560 | 0.3576 ± 0.0442 | 0.3401 ± 0.0427 | 0.3511 ± 0.0560 | 0.3576 ± 0.0442 | 0.3701 ± 0.0523 |
| 12 | 0.3612 ± 0.0528 | 0.3703 ± 0.0473 | 0.3558 ± 0.0569 | 0.3650 ± 0.0547 | 0.3688 ± 0.0453 | 0.3757 ± 0.0514 |

**Terbaik pada budget penuh: `evolution` (rata-rata 0.3757).**
