"""Eksperimen terkontrol: validasi zero-cost proxy pada ARSITEKTUR MURNI.

Motivasi: validasi pertama (proxycheck atas cache) tercemar — proxy hanya
melihat arsitektur, sedangkan baris cache bervariasi di lr/optimizer/augment
dan cuma memuat 3 arsitektur berbeda. Di sini hyperparameter DIKUNCI
(lr=3e-3, adam, augment basic) dan hanya arsitektur yang divariasikan:
12 varian tiny acak + resnet18 + mobilenet_v3_small, masing-masing dilatih
1 epoch pada data Kaggle. Lalu ukur Spearman proxy vs akurasi.
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from racik.config import config_label, load_sweep
from racik.proxies import ProxyScorer, spearman
from racik.runner import evaluate
from racik.space import Space
from racik.vision_backend import VisionBackend

FIXED = {"lr": 0.003, "optimizer": "adam", "augment": "basic"}

SWEEP = sys.argv[1] if len(sys.argv) > 1 else "sweep_kaggle.yaml"
cfg = load_sweep(SWEEP)
arch_space = Space({
    "depth": {"type": "int", "low": 2, "high": 4},
    "width": {"type": "choice", "options": [16, 32, 48]},
    "block": {"type": "choice", "options": ["basic", "residual", "depthwise"]},
    "use_se": {"type": "choice", "options": [True, False]},
})

rng = random.Random(7)
configs, seen = [], set()
while len(configs) < 12:
    c = {"arch": "tiny", **FIXED, **arch_space.sample(rng)}
    k = json.dumps(c, sort_keys=True)
    if k not in seen:
        seen.add(k)
        configs.append(c)
configs += [{"arch": "resnet18", **FIXED},
            {"arch": "mobilenet_v3_small", **FIXED}]

backend = VisionBackend(cfg)
rows = []
for i, c in enumerate(configs, 1):
    row = evaluate(cfg, c, backend, use_cache=True)
    print(f"[{i}/{len(configs)}] acc={row['score']:.4f}  {config_label(c)}",
          flush=True)
    rows.append(row)

accs = [r["score"] for r in rows]
out = {"fixed_hparams": FIXED, "rows": rows}
print()
for proxy in ("naswot", "synflow"):
    scorer = ProxyScorer(cfg, proxy=proxy)
    scores = [scorer.score(r["config"]) for r in rows]
    rho = spearman(scores, accs)
    out[proxy] = {"scores": scores, "spearman": rho}
    print(f"{proxy}: Spearman rho vs akurasi (arsitektur murni) = {rho:+.3f}",
          flush=True)

with open("proxy_validation.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, default=str)
print("\nTersimpan: proxy_validation.json")
