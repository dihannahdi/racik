# Benchmark algoritme pencari (racik)

Tugas: Klasifikasi gambar dataset Kaggle: temukan konfigurasi CNN dengan akurasi validasi tertinggi pada budget training singkat (CPU-friendly).
Backend: vision  |  Budget: 10 trial  |  Seed: 2

Angka = rata-rata skor-terbaik-sejauh-ini pada trial ke-t (± simpangan baku).

| Trial | random | evolution | tpe |
|-------|------|------|------|
| 1 | 0.3641 ± 0.0059 | 0.3641 ± 0.0059 | 0.3641 ± 0.0059 |
| 5 | 0.3641 ± 0.0059 | 0.3641 ± 0.0059 | 0.3641 ± 0.0059 |
| 10 | 0.3850 ± 0.0236 | 0.4592 ± 0.0813 | 0.3641 ± 0.0059 |

**Terbaik pada budget penuh: `evolution` (rata-rata 0.4592).**
