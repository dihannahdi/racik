"""Uji berpasangan atas hasil bench: apakah selisihnya nyata atau noise?

Dua pencari dibandingkan pada seed yang sama (paired), lalu:
  - selisih rata-rata,
  - uji permutasi tanda (exact sign-flip test, 2-sisi) — tanpa scipy.

Kalibrasi noise: `random` vs `optuna_random` secara algoritmik IDENTIK
(dua-duanya sampling seragam), jadi selisih terukur antara keduanya adalah
lantai noise protokol ini. Selisih apa pun yang lebih kecil dari itu tidak
boleh diklaim sebagai kemenangan.
"""

import itertools
import json
import sys

with open(sys.argv[1] if len(sys.argv) > 1 else "bench.json", encoding="utf-8") as f:
    curves = json.load(f)

names = list(curves)
finals = {n: [c[-1] for c in curves[n]] for n in names}
n_seed = len(next(iter(finals.values())))


def sign_flip_p(diffs):
    """p-value 2-sisi: peluang |mean| sebesar ini kalau tanda acak (H0)."""
    obs = abs(sum(diffs) / len(diffs))
    if len(diffs) > 20:  # ruang 2^n terlalu besar — pakai subsampel deterministik
        import random as _r
        rng = _r.Random(0)
        signs = (tuple(rng.choice([1, -1]) for _ in diffs) for _ in range(20000))
    else:
        signs = itertools.product([1, -1], repeat=len(diffs))
    total = hits = 0
    for s in signs:
        total += 1
        if abs(sum(si * di for si, di in zip(s, diffs)) / len(diffs)) >= obs - 1e-12:
            hits += 1
    return hits / total


print(f"n_seed = {n_seed}\n")
print("Skor akhir rata-rata:")
for n in sorted(names, key=lambda k: -sum(finals[k]) / n_seed):
    m = sum(finals[n]) / n_seed
    print(f"  {n:<15} {m:.4f}")

if "random" in finals and "optuna_random" in finals:
    d = [a - b for a, b in zip(finals["random"], finals["optuna_random"])]
    floor = abs(sum(d) / n_seed)
    print(f"\nLANTAI NOISE (random vs optuna_random, algoritma identik): "
          f"{floor:.4f}  (p={sign_flip_p(d):.3f})")
else:
    floor = None

print("\nPerbandingan berpasangan (selisih rata-rata, p sign-flip):")
for a, b in itertools.combinations(names, 2):
    d = [x - y for x, y in zip(finals[a], finals[b])]
    m = sum(d) / n_seed
    p = sign_flip_p(d)
    verdict = "signifikan" if p < 0.05 else "TIDAK signifikan"
    note = ""
    if floor is not None and abs(m) < floor:
        note = "  [di bawah lantai noise]"
    print(f"  {a:<15} - {b:<15} {m:+.4f}  p={p:.3f}  {verdict}{note}")
