# Benchmark algoritme pencari (racik)

Tugas: Klasifikasi CIFAR-10: replikasi protokol benchmark pencari pada dataset kedua, untuk menguji apakah temuan lintas-tugas.
Backend: vision  |  Budget: 12 trial  |  Seed: 20

Angka = rata-rata skor-terbaik-sejauh-ini pada trial ke-t (± simpangan baku).

| Trial | random | optuna_random | tpe | tpe2 | optuna_tpe | evolution |
|-------|------|------|------|------|------|------|
| 1 | 0.3165 ± 0.1601 | 0.2947 ± 0.1581 | 0.3165 ± 0.1601 | 0.3165 ± 0.1601 | 0.2947 ± 0.1581 | 0.3165 ± 0.1601 |
| 5 | 0.4671 ± 0.0552 | 0.4637 ± 0.0535 | 0.4671 ± 0.0552 | 0.4671 ± 0.0552 | 0.4637 ± 0.0535 | 0.4671 ± 0.0552 |
| 10 | 0.4930 ± 0.0381 | 0.4911 ± 0.0474 | 0.4877 ± 0.0527 | 0.4930 ± 0.0381 | 0.4911 ± 0.0474 | 0.4981 ± 0.0468 |
| 12 | 0.5001 ± 0.0426 | 0.4947 ± 0.0454 | 0.4938 ± 0.0504 | 0.5075 ± 0.0395 | 0.5219 ± 0.0400 | 0.5081 ± 0.0458 |

**Terbaik pada budget penuh: `optuna_tpe` (rata-rata 0.5219).**
