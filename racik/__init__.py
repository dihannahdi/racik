"""racik — kerangka riset untuk algoritme pencarian arsitektur & hyperparameter vision.

Struktur:
  space.py          -> definisi ruang pencarian (sample & mutate)
  searchers.py      -> algoritme pencari: random, regularized evolution, TPE
  vision_backend.py -> fungsi objektif: latih CNN, kembalikan akurasi validasi
  dummy_backend.py  -> fungsi objektif sintetis untuk sanity-check algoritme
  runner.py         -> loop ask/tell + cache hasil
  report.py         -> laporan per-run dan laporan benchmark antar-algoritme
"""

__version__ = "0.2.0"
