# Hasil pencarian racik

Tugas: Klasifikasi Intel Image (Kaggle, T4): benchmark algoritme pencari dan zero-cost proxy pada budget lebih serius.
Backend: vision  |  Algoritme: halving(random)  |  Trial: 13

| # | Skor | Konfigurasi | Catatan |
|---|------|-------------|---------|
| 1 | 0.5533 | arch=mobilenet_v3_small lr=9.21e-03 optimizer=sgd augment=none | 38.2s, 1.52M param @9 epoch |
| 2 | 0.1800 | arch=mobilenet_v3_small lr=9.21e-03 optimizer=sgd augment=none | 5.3s, 1.52M param @1 epoch |
| 3 | 0.1800 | arch=mobilenet_v3_small lr=1.07e-04 optimizer=adam augment=none | 5.3s, 1.52M param @1 epoch |
| 4 | 0.1800 | arch=mobilenet_v3_small lr=2.93e-04 optimizer=sgd augment=basic | 5.7s, 1.52M param @1 epoch |
| 5 | 0.1800 | arch=mobilenet_v3_small lr=1.31e-03 optimizer=sgd augment=none | 5.3s, 1.52M param @1 epoch |
| 6 | 0.1800 | arch=mobilenet_v3_small lr=3.46e-03 optimizer=sgd augment=basic | 5.8s, 1.52M param @1 epoch |
| 7 | 0.1800 | arch=mobilenet_v3_small lr=9.21e-03 optimizer=sgd augment=none | 13.5s, 1.52M param @3 epoch |
| 8 | 0.1800 | arch=mobilenet_v3_small lr=2.93e-04 optimizer=sgd augment=basic | 15.0s, 1.52M param @3 epoch |
| 9 | 0.1783 | arch=mobilenet_v3_small lr=3.46e-04 optimizer=adam augment=basic | 5.9s, 1.52M param @1 epoch |
| 10 | 0.1783 | arch=mobilenet_v3_small lr=1.07e-04 optimizer=adam augment=none | 13.8s, 1.52M param @3 epoch |
| 11 | 0.1683 | arch=mobilenet_v3_small lr=5.13e-03 optimizer=adam augment=none | 5.5s, 1.52M param @1 epoch |
| 12 | 0.1683 | arch=mobilenet_v3_small lr=8.94e-04 optimizer=adam augment=none | 5.3s, 1.52M param @1 epoch |
| 13 | 0.1683 | arch=mobilenet_v3_small lr=6.24e-04 optimizer=adam augment=none | 6.2s, 1.52M param @1 epoch |

## Konfigurasi terbaik

```json
{
  "arch": "mobilenet_v3_small",
  "lr": 0.009212698142984177,
  "optimizer": "sgd",
  "augment": "none"
}
```
