"""Loop eksperimen ask/tell, dengan cache hasil di disk.

Protokolnya sederhana dan sama untuk semua algoritme:
  untuk t = 1..budget:
    config = searcher.ask()        # algoritme mengusulkan konfigurasi
    skor   = objective(config)     # latih & ukur (atau fungsi sintetis)
    searcher.tell(config, skor)    # algoritme belajar dari hasilnya
"""

import hashlib
import json
from pathlib import Path

from .config import config_label

CACHE_DIR = Path(".racik_cache")


def _cache_key(cfg, config, fidelity=None):
    ctx = {
        "backend": cfg.get("backend"),
        "dataset": cfg.get("dataset"),
        "epochs": cfg.get("epochs"),
        "seed": cfg.get("seed"),
        "fidelity": fidelity,
        "config": config,
    }
    raw = json.dumps(ctx, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def evaluate(cfg, config, backend, use_cache=True, fidelity=None):
    CACHE_DIR.mkdir(exist_ok=True)
    key = _cache_key(cfg, config, fidelity)
    cache_file = CACHE_DIR / f"{key}.json"
    if use_cache and cache_file.exists():
        row = json.loads(cache_file.read_text(encoding="utf-8"))
        row["dari_cache"] = True
        return row
    out = backend.run(config, fidelity=fidelity)
    row = {"config": config, "score": out.get("score", 0.0)}
    if fidelity:
        row["fidelity"] = fidelity
    for extra in ("metrics", "error"):
        if out.get(extra):
            row[extra] = out[extra]
    cache_file.write_text(json.dumps(row, ensure_ascii=False, indent=1),
                          encoding="utf-8")
    return row


def run_search(cfg, searcher, backend, budget, use_cache=True, verbose=True):
    """Jalankan satu pencarian penuh; kembalikan (riwayat, kurva-terbaik)."""
    history, curve, best = [], [], 0.0
    for t in range(1, budget + 1):
        config = searcher.ask()
        row = evaluate(cfg, config, backend, use_cache=use_cache)
        searcher.tell(config, row["score"])
        best = max(best, row["score"])
        history.append(row)
        curve.append(best)
        if verbose:
            tag = " (cache)" if row.get("dari_cache") else ""
            err = f"  ERROR: {row['error']}" if row.get("error") else ""
            print(f"  trial {t:>3}: skor={row['score']:.4f}  "
                  f"terbaik={best:.4f}{tag}  {config_label(config)}{err}")
    return history, curve
