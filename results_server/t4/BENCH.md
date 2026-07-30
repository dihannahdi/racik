# Benchmark algoritme pencari (racik)

Tugas: Klasifikasi Intel Image (Kaggle, T4): benchmark algoritme pencari dan zero-cost proxy pada budget lebih serius.
Backend: vision  |  Budget: 12 trial  |  Seed: 2

Angka = rata-rata skor-terbaik-sejauh-ini pada trial ke-t (± simpangan baku).

| Trial | random | evolution | tpe |
|-------|------|------|------|
| 1 | 0.5962 ± 0.0183 | 0.5962 ± 0.0183 | 0.5962 ± 0.0183 |
| 5 | 0.5975 ± 0.0165 | 0.5975 ± 0.0165 | 0.5975 ± 0.0165 |
| 10 | 0.6275 ± 0.0259 | 0.7100 ± 0.0767 | 0.6175 ± 0.0117 |
| 12 | 0.6275 ± 0.0259 | 0.7100 ± 0.0767 | 0.7171 ± 0.0607 |

**Terbaik pada budget penuh: `tpe` (rata-rata 0.7171).**
