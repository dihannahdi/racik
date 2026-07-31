"""Audit validitas benchmark: lengan plasebo -> daya uji -> verdict + LII.

Bekerja pada hasil sweep MANA PUN, bukan hanya racik. Dua format masukan:

  # format racik (bench.json: {nama: [kurva_per_seed, ...]})
  py scripts/audit.py bench.json --sham random,optuna_random

  # format universal (CSV panjang: arm,seed,score) — ekspor dari Optuna,
  # W&B, Ray Tune, MLflow, atau spreadsheet apa pun
  py scripts/audit.py hasil.csv --sham baseline_a,baseline_b

`score` adalah hasil akhir satu run (mis. akurasi terbaik yang dicapai).
Baris wajib berpasangan: setiap arm harus punya seed yang sama.

Keluaran: laporan markdown yang bisa dilampirkan ke paper, PR, atau tiket
keputusan sebagai bukti bahwa klaim di dalamnya berada di atas lantai noise
dan di atas MDE — atau bukti bahwa ia tidak.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from racik.validity import audit  # noqa: E402


def load_curves(path):
    """Kembalikan {arm: [kurva_per_seed]} dari bench.json atau CSV panjang."""
    p = Path(path)
    if p.suffix.lower() == ".json":
        return json.loads(p.read_text(encoding="utf-8"))

    rows = list(csv.DictReader(p.read_text(encoding="utf-8-sig").splitlines()))
    if not rows:
        raise SystemExit("CSV kosong.")
    cols = {c.lower(): c for c in rows[0]}
    for need in ("arm", "seed", "score"):
        if need not in cols:
            raise SystemExit(f"CSV harus punya kolom arm, seed, score "
                             f"(ditemukan: {list(rows[0])})")

    by_arm = {}
    for r in rows:
        arm = r[cols["arm"]].strip()
        by_arm.setdefault(arm, {})[r[cols["seed"]].strip()] = float(r[cols["score"]])

    seed_sets = {arm: set(d) for arm, d in by_arm.items()}
    common = set.intersection(*seed_sets.values())
    for arm, s in seed_sets.items():
        if s != common:
            print(f"  catatan: arm '{arm}' punya {len(s)} seed; "
                  f"dipakai {len(common)} seed yang dimiliki semua arm.")
    if len(common) < 2:
        raise SystemExit("Butuh minimal 2 seed yang sama di semua arm.")
    order = sorted(common)
    # setiap 'kurva' cukup satu titik: skor akhir run itu
    return {arm: [[by_arm[arm][s]] for s in order] for arm in by_arm}


def fmt(x, nd=4):
    return "—" if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bench", nargs="?", default="bench.json")
    ap.add_argument("--sham", default=None,
                    help="dua pencari identik, dipisah koma (lengan plasebo)")
    ap.add_argument("--out", default="AUDIT.md")
    ap.add_argument("--label", default=None, help="judul protokol")
    args = ap.parse_args()

    curves = load_curves(args.bench)
    sham = tuple(s.strip() for s in args.sham.split(",")) if args.sham else None
    rep = audit(curves, sham=sham)

    n = rep["n_seed"]
    L = [f"# Audit validitas benchmark — {args.label or args.bench}", "",
         f"n seed = **{n}**  |  p terkecil yang mungkin = **{fmt(rep['p_floor'], 5)}**",
         ""]
    if rep["p_floor"] > 0.05:
        L += [f"> **PERINGATAN:** dengan {n} seed, p<0.05 mustahil dicapai "
              "seberapa pun besar efeknya (lantai daya uji 2/2^n).", ""]

    L += ["## Papan peringkat", "", "| Pencari | Skor akhir rata-rata |",
          "|---|---|"]
    for name in sorted(rep["finals"], key=lambda m: -sum(rep["finals"][m]) / n):
        L.append(f"| {name} | {fmt(sum(rep['finals'][name]) / n)} |")

    pl = rep["placebo"]
    L += ["", "## Lengan plasebo (dua implementasi algoritme identik)", ""]
    if pl:
        L += [f"Pasangan: **{pl['pasangan']}** — keduanya seharusnya seri.", "",
              f"- Lantai noise (|Δ| terukur): **{fmt(pl['lantai_noise'])}**",
              f"- Simpangan baku null (bahan analisis daya uji): "
              f"**{fmt(pl['sd_null'])}**",
              f"- p uji plasebo: {fmt(pl['p'], 3)} "
              f"({'sesuai harapan, tidak signifikan' if pl['p'] >= 0.05 else 'MENCURIGAKAN: dua algoritme identik berbeda signifikan'})",
              "",
              "Klaim apa pun di bawah lantai noise tidak boleh disebut kemenangan."]
    else:
        L += ["_Tidak ada lengan plasebo pada benchmark ini._", "",
              "Tanpa dua implementasi algoritme identik, lantai noise dan MDE "
              "tidak bisa ditaksir. Tambahkan satu pasangan sham "
              "(mis. `random` + implementasi random dari library lain)."]

    if rep.get("mde"):
        L += ["", "## Efek terkecil yang terdeteksi (MDE, alpha=0.05, daya 0.8)",
              "", "| n seed | MDE |", "|---|---|"]
        for row in rep["mde"]:
            v = fmt(row["mde"]) if row["mde"] else "mustahil (lantai daya uji)"
            L.append(f"| {row['n']} | {v} |")
        L += ["", "Klaim yang lebih kecil dari MDE pada n seed kamu tidak "
              "terdukung — bukan karena efeknya tidak ada, tetapi karena "
              "protokolnya tidak mampu melihatnya.",
              "", "## Seed yang dibutuhkan untuk mengklaim efek sebesar delta",
              "", "| delta | n seed minimum |", "|---|---|"]
        for row in rep["required_n"]:
            L.append(f"| {row['effect']:.3f} | "
                     f"{row['n'] if row['n'] else '>200'} |")

    lii = rep["lii"]
    L += ["", "## Indeks ketidakstabilan papan peringkat (LII)", "",
          f"Acuan: juara pada seluruh {lii['acuan_n']} seed = "
          f"**{lii['juara_acuan']}**. Kolom kedua: peluang sebuah studi dengan "
          "k seed menobatkan juara yang BERBEDA.", "",
          "| k seed | P(juara berubah) | Spearman rata-rata vs acuan |",
          "|---|---|---|"]
    for row in lii["kurva"]:
        L.append(f"| {row['k']} | **{row['p_juara_berubah']:.1%}** | "
                 f"{fmt(row['spearman_rata2'], 3)} |")

    L += ["", "## Verdict per perbandingan", "",
          "| A | B | delta | p | Verdict |", "|---|---|---|---|---|"]
    for c in sorted(rep["comparisons"], key=lambda r: r["p"]):
        if c["signifikan"]:
            v = "signifikan"
        elif c["di_bawah_lantai"]:
            v = "di bawah lantai noise"
        else:
            v = "tidak signifikan"
        L.append(f"| {c['a']} | {c['b']} | {c['delta']:+.4f} | "
                 f"{c['p']:.3f} | {v} |")

    Path(args.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:40]))
    print(f"\n... laporan lengkap: {args.out}")


if __name__ == "__main__":
    main()
