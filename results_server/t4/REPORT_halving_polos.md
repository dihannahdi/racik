# Hasil pencarian racik

Tugas: Klasifikasi Intel Image (Kaggle, T4): benchmark algoritme pencari dan zero-cost proxy pada budget lebih serius.
Backend: vision  |  Algoritme: halving(random)  |  Trial: 13

| # | Skor | Konfigurasi | Catatan |
|---|------|-------------|---------|
| 1 | 0.6450 | arch=tiny lr=2.11e-03 optimizer=sgd augment=basic depth=3 width=48 block=basic use_se=False | 12.5s, 0.23M param @3 epoch |
| 2 | 0.6425 | arch=tiny lr=2.11e-03 optimizer=sgd augment=basic depth=3 width=48 block=basic use_se=False | 35.1s, 0.23M param @9 epoch |
| 3 | 0.6392 | arch=tiny lr=2.40e-04 optimizer=adam augment=none depth=3 width=32 block=residual use_se=True | 11.8s, 0.31M param @3 epoch |
| 4 | 0.6200 | arch=resnet18 lr=5.39e-03 optimizer=sgd augment=basic | 13.5s, 11.18M param @3 epoch |
| 5 | 0.5717 | arch=tiny lr=2.11e-03 optimizer=sgd augment=basic depth=3 width=48 block=basic use_se=False | 4.9s, 0.23M param @1 epoch |
| 6 | 0.5550 | arch=tiny lr=2.40e-04 optimizer=adam augment=none depth=3 width=32 block=residual use_se=True | 4.7s, 0.31M param @1 epoch |
| 7 | 0.4350 | arch=resnet18 lr=5.39e-03 optimizer=sgd augment=basic | 6.1s, 11.18M param @1 epoch |
| 8 | 0.3175 | arch=resnet18 lr=4.04e-04 optimizer=adam augment=none | 4.8s, 11.18M param @1 epoch |
| 9 | 0.1800 | arch=mobilenet_v3_small lr=9.21e-03 optimizer=sgd augment=none | 5.3s, 1.52M param @1 epoch |
| 10 | 0.1800 | arch=mobilenet_v3_small lr=1.07e-04 optimizer=adam augment=none | 5.3s, 1.52M param @1 epoch |
| 11 | 0.1800 | arch=mobilenet_v3_small lr=2.93e-04 optimizer=sgd augment=basic | 5.7s, 1.52M param @1 epoch |
| 12 | 0.1683 | arch=mobilenet_v3_small lr=5.13e-03 optimizer=adam augment=none | 5.5s, 1.52M param @1 epoch |
| 13 | 0.1683 | arch=mobilenet_v3_small lr=8.94e-04 optimizer=adam augment=none | 5.3s, 1.52M param @1 epoch |

## Konfigurasi terbaik

```json
{
  "arch": "tiny",
  "lr": 0.0021129978919778567,
  "optimizer": "sgd",
  "augment": "basic",
  "depth": 3,
  "width": 48,
  "block": "basic",
  "use_se": false
}
```
