"""racik — benchmark daya-uji-layak di data nyata (GPU T4 Kaggle).

Kenapa GPU, bukan CPU: kita butuh 20 seed x 6 pencari x 12 trial = 1440
training. Di VPS itu belasan jam; di T4 sekitar 2-3 jam.
Kenapa bukan TPU: XLA meng-compile graf per bentuk arsitektur, sedangkan
pencarian arsitektur mengganti bentuk model setiap trial — waktu kompilasi
akan menenggelamkan waktu training.

Pertanyaan yang dijawab: apakah keunggulan TPE v2 (terbukti di objektif
sintetis, delta +0.077 p=0.000) bertahan di data gambar sungguhan, pada
jumlah seed yang cukup untuk menyimpulkan? Pasangan kalibrasi
random/optuna_random disertakan untuk mengukur lantai noise data nyata.
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

# data disamakan dengan sweep_kaggle.yaml agar hasil CPU & GPU sebanding
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
print("data siap:", sum(len(fs) for _, _, fs in os.walk("data/kaggle")), "file",
      flush=True)

env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONUTF8="1")
SEARCHERS = "random,optuna_random,tpe,tpe2,optuna_tpe,evolution"

subprocess.run([sys.executable, "run.py", "bench", "sweep_kaggle.yaml",
                "--searchers", SEARCHERS, "--budget", "12", "--seeds", "20"],
               check=False, env=env)

print("\n" + "=" * 60 + "\n== UJI BERPASANGAN + LANTAI NOISE\n" + "=" * 60,
      flush=True)
subprocess.run([sys.executable, "scripts/paired_test.py", "bench.json"],
               check=False, env=env)

for name in ["BENCH.md", "bench.json"]:
    if os.path.exists(name):
        shutil.copy(name, f"/kaggle/working/{name}")
shutil.rmtree("data", ignore_errors=True)
print("\nSELESAI", flush=True)
