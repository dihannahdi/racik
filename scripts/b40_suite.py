"""racik — uji hipotesis "budget efektif" di data nyata (GPU T4).

Temuan sebelumnya: pada budget 12, TIDAK ADA pencari yang berbeda signifikan
dari random (20 seed, daya uji memadai). Diagnosisnya terukur: tpe2 dengan
warmup 10 hanya membuat 2 keputusan berbasis model dari 12 trial — kurvanya
identik dengan random sampai trial 10 di 20/20 seed. Jadi benchmark itu
sebagian besar mengukur warmup, bukan algoritmenya.

Hipotesis: naikkan budget menjadi 40 (budget efektif 30 keputusan berbasis
model) dan perbedaan akan muncul. Satu variabel saja yang diubah — fidelity
tetap 1 epoch, data tetap sama, seed tetap 20 — agar sebabnya jelas.
"""

import os
import random
import shutil
import subprocess
import sys

REPO = "https://github.com/dihannahdi/racik.git"
WORK = "/kaggle/working/racik"

subprocess.run(["git", "clone", "--depth", "1", REPO, WORK], check=True)
os.chdir(WORK)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "optuna", "cmaes"],
               check=True)


def find_split(root, name):
    for dirpath, _dirnames, _files in os.walk(root):
        if os.path.basename(dirpath) == name:
            inner = os.path.join(dirpath, name)
            cand = inner if os.path.isdir(inner) else dirpath
            subs = [d for d in os.listdir(cand)
                    if os.path.isdir(os.path.join(cand, d))]
            if subs and not any(s.startswith("seg_") for s in subs):
                return cand
    raise FileNotFoundError(f"{name} tidak ditemukan di {root}")


ROOT = "/kaggle/input"
if os.path.isdir(ROOT) and os.listdir(ROOT):
    SRC = ROOT
else:
    import kagglehub
    SRC = kagglehub.dataset_download("puneet6060/intel-image-classification")

rng = random.Random(42)
for split, srcname, cap in [("train", "seg_train", 800), ("val", "seg_test", 150)]:
    base = find_split(SRC, srcname)
    for cls in sorted(os.listdir(base)):
        files = sorted(os.listdir(os.path.join(base, cls)))
        rng.shuffle(files)
        dst = os.path.join("data/kaggle", split, cls)
        os.makedirs(dst, exist_ok=True)
        for f in files[:cap]:
            shutil.copy(os.path.join(base, cls, f), dst)
print("data siap", flush=True)

env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONUTF8="1")
SEARCHERS = "random,optuna_random,tpe,tpe2,optuna_tpe,evolution"

subprocess.run([sys.executable, "run.py", "bench", "sweep_kaggle.yaml",
                "--searchers", SEARCHERS, "--budget", "40", "--seeds", "20"],
               check=False, env=env)

print("\n" + "=" * 60 + "\n== UJI BERPASANGAN + LANTAI NOISE (budget 40)\n"
      + "=" * 60, flush=True)
subprocess.run([sys.executable, "scripts/paired_test.py", "bench.json"],
               check=False, env=env)

for name in ["BENCH.md", "bench.json"]:
    if os.path.exists(name):
        shutil.copy(name, f"/kaggle/working/{name}")
shutil.rmtree("data", ignore_errors=True)
shutil.rmtree(".racik_cache", ignore_errors=True)   # jangan ikut ke output
shutil.rmtree(".git", ignore_errors=True)
print("\nSELESAI", flush=True)
