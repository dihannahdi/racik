"""Fungsi objektif sintetis untuk sanity-check algoritme pencari.

Permukaan skornya kita rancang sendiri sehingga optimum diketahui (≈0.90 untuk
tiny + lr≈3e-3 + adam + residual + depth 3 + width 48 + SE + augment basic).
Dengan begitu kita bisa memverifikasi: algoritme yang benar harus mendekati
optimum itu lebih cepat daripada random search — tanpa perlu GPU sama sekali.
Skor deterministik per-konfigurasi (noise kecil di-seed dari isi konfigurasi).
"""

import hashlib
import json
import math
import random


class DummyBackend:
    def __init__(self, cfg):
        self.noise = float((cfg.get("dummy") or {}).get("noise", 0.02))

    def run(self, config, fidelity=None):
        # fidelity meniru jumlah epoch: makin tinggi, makin kecil noise
        # estimasi skornya (sigma ~ 1/sqrt(epoch)) — persis perilaku training asli.
        s = 0.0
        arch = config.get("arch", "tiny")

        if arch == "tiny":
            lr = config.get("lr")
            if lr:
                s += 0.30 * math.exp(-((math.log10(lr) - math.log10(3e-3)) / 0.5) ** 2)
            if config.get("optimizer") == "adam":
                s += 0.10
            s += {"residual": 0.10, "depthwise": 0.05}.get(config.get("block"), 0.0)
            depth = config.get("depth")
            if depth is not None:
                s += max(0.0, 0.10 - 0.05 * abs(int(depth) - 3))
            width = config.get("width")
            if width:
                s += min(float(width) / 48.0, 1.0) * 0.10
            if config.get("use_se"):
                s += 0.05
            if config.get("augment") == "basic":
                s += 0.05
            s += 0.10  # bonus keluarga tiny — optimumnya memang di sini
        elif arch == "resnet18":
            s = 0.55
        else:
            s = 0.50

        sigma = self.noise / math.sqrt(fidelity) if fidelity else self.noise
        seed = int(hashlib.sha256(
            json.dumps({"c": config, "f": fidelity}, sort_keys=True,
                       default=str).encode()
        ).hexdigest(), 16) % (2 ** 32)
        s += random.Random(seed).gauss(0, sigma)
        return {"score": round(min(max(s, 0.0), 1.0), 4)}
