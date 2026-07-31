"""Kalibrasi fidelity: cari (epoch, ukuran subset) agar dataset kedua berada
di rezim akurasi yang sebanding dengan dataset pertama.

Kenapa perlu: menyamakan JUMLAH EPOCH antar dataset yang kesulitannya berbeda
bukan perbandingan yang adil. CIFAR-10 pada 1 epoch/1500 sampel tertahan di
dekat lantai tebakan acak (0.10) — di rezim itu semua algoritme akan tampak
sama karena tidak ada sinyal untuk dibedakan, bukan karena temuan.

Yang disamakan seharusnya rezimnya: akurasi acuan (random search) berada di
band yang sebanding. Intel Image pada protokol kami: ~0.36. Skrip ini mencoba
beberapa setelan dan melaporkan akurasi rata-rata 3 konfigurasi acak, supaya
setelan untuk benchmark lintas-dataset dipilih dari bukti, bukan tebakan.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from racik.config import load_sweep  # noqa: E402
from racik.runner import evaluate  # noqa: E402
from racik.space import Space  # noqa: E402
from racik.vision_backend import VisionBackend  # noqa: E402

TARGET = 0.36  # band akurasi acuan dari protokol Intel Image

# (epochs, limit_train, limit_val)
KANDIDAT = [
    (1, 1500, 600),
    (3, 3000, 1000),
    (5, 5000, 2000),
    (8, 8000, 2000),
]

cfg = load_sweep(sys.argv[1] if len(sys.argv) > 1 else "sweep_cifar.yaml")
space = Space(cfg["space"])
rng = random.Random(123)
configs = [space.sample(rng) for _ in range(3)]

print(f"Target band akurasi acuan: ~{TARGET:.2f}\n")
for epochs, lim_tr, lim_va in KANDIDAT:
    trial_cfg = dict(cfg)
    trial_cfg["epochs"] = epochs
    trial_cfg["dataset"] = dict(cfg["dataset"],
                                limit_train=lim_tr, limit_val=lim_va)
    backend = VisionBackend(trial_cfg)
    accs, secs = [], []
    for c in configs:
        row = evaluate(trial_cfg, c, backend, use_cache=True)
        accs.append(row["score"])
        secs.append((row.get("metrics") or {}).get("durasi_detik", 0))
    mean = sum(accs) / len(accs)
    print(f"epochs={epochs:<2} train={lim_tr:<5} -> akurasi rata2 {mean:.4f} "
          f"(maks {max(accs):.4f}), ~{sum(secs) / len(secs):.0f}s per training",
          flush=True)
    if mean >= TARGET * 0.8:
        print(f"\n=> Setelan ini sudah masuk band target. Pakai: "
              f"epochs={epochs}, limit_train={lim_tr}, limit_val={lim_va}")
        break
else:
    print("\n=> Tidak ada kandidat yang mencapai band; naikkan epoch/data lagi.")
