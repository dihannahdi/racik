"""Laporan: papan peringkat satu run + tabel benchmark antar-algoritme."""

import json
import statistics
from pathlib import Path

from .config import config_label


def write_report(history, cfg, searcher_name, out_dir="."):
    out = Path(out_dir)
    rows = sorted(history, key=lambda r: r["score"], reverse=True)

    lines = [
        "# Hasil pencarian racik",
        "",
        f"Tugas: {str(cfg.get('task', '-')).strip()}",
        f"Backend: {cfg.get('backend')}  |  Algoritme: {searcher_name}  |  Trial: {len(rows)}",
        "",
        "| # | Skor | Konfigurasi | Catatan |",
        "|---|------|-------------|---------|",
    ]
    for i, r in enumerate(rows, 1):
        note = r.get("error", "")
        if not note and r.get("metrics"):
            m = r["metrics"]
            note = f"{m.get('durasi_detik', '?')}s, {m.get('params_juta', '?')}M param"
        if r.get("fidelity"):
            note = (note + " " if note else "") + f"@{r['fidelity']} epoch"
        lines.append(f"| {i} | {r['score']:.4f} | {config_label(r['config'])} | {note} |")

    if rows:
        lines += ["", "## Konfigurasi terbaik", "", "```json",
                  json.dumps(rows[0]["config"], ensure_ascii=False, indent=2,
                             default=str),
                  "```"]

    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    print(f"\nLaporan tersimpan: {out / 'REPORT.md'} dan {out / 'results.json'}")
    return rows


def write_bench(curves, cfg, budget, out_dir="."):
    """curves: {nama_searcher: [kurva_seed0, kurva_seed1, ...]}
    Tiap kurva = skor-terbaik-sejauh-ini per trial (panjang = budget)."""
    out = Path(out_dir)
    names = list(curves.keys())

    def mean_std(name, t):
        vals = [c[t] for c in curves[name]]
        m = statistics.mean(vals)
        s = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return m, s

    checkpoints = sorted({1, budget} | {t for t in range(5, budget + 1, 5)})

    lines = [
        "# Benchmark algoritme pencari (racik)",
        "",
        f"Tugas: {str(cfg.get('task', '-')).strip()}",
        f"Backend: {cfg.get('backend')}  |  Budget: {budget} trial  |  "
        f"Seed: {len(next(iter(curves.values())))}",
        "",
        "Angka = rata-rata skor-terbaik-sejauh-ini pada trial ke-t (± simpangan baku).",
        "",
        "| Trial | " + " | ".join(names) + " |",
        "|-------|" + "|".join(["------"] * len(names)) + "|",
    ]
    for t in checkpoints:
        cells = []
        for n in names:
            m, s = mean_std(n, t - 1)
            cells.append(f"{m:.4f} ± {s:.4f}")
        lines.append(f"| {t} | " + " | ".join(cells) + " |")

    finals = {n: mean_std(n, budget - 1)[0] for n in names}
    winner = max(finals, key=finals.get)
    lines += ["", f"**Terbaik pada budget penuh: `{winner}` "
                  f"(rata-rata {finals[winner]:.4f}).**"]

    (out / "BENCH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "bench.json").write_text(
        json.dumps({n: curves[n] for n in names}, indent=1), encoding="utf-8")
    print(f"\nBenchmark tersimpan: {out / 'BENCH.md'} dan {out / 'bench.json'}")
    return winner, finals
