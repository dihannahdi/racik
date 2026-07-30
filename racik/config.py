"""Muat konfigurasi sweep dan format label konfigurasi."""

import yaml


def load_sweep(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("backend", "vision")
    if not cfg.get("space"):
        raise ValueError("sweep.yaml harus punya bagian `space`.")
    return cfg


def config_label(config):
    """Label ringkas satu konfigurasi, untuk log dan laporan."""
    parts = []
    for k, v in config.items():
        if isinstance(v, float):
            v = f"{v:.2e}" if abs(v) < 0.01 else f"{v:.3f}"
        parts.append(f"{k}={v}")
    return " ".join(parts)
