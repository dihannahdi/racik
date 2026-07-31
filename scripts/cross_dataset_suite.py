"""racik — uji lintas-tugas: apakah temuan bertahan di dataset kedua? (GPU T4)

Menyerang batasan nomor 1 di FINDINGS.md. Protokol disamakan persis dengan
benchmark data nyata pertama (budget 12 & 40, 20 seed, 1 epoch, ruang
pencarian sama); satu-satunya variabel yang berubah adalah datasetnya
(CIFAR-10, terunduh otomatis oleh torchvision).

Dijalankan pada dua budget sekaligus supaya diagnosis "budget efektif" ikut
teruji di tugas kedua, bukan cuma di tugas pertama.
"""

import os
import shutil
import subprocess
import sys

REPO = "https://github.com/dihannahdi/racik.git"
WORK = "/kaggle/working/racik"

subprocess.run(["git", "clone", "--depth", "1", REPO, WORK], check=True)
os.chdir(WORK)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "optuna", "cmaes"],
               check=True)

env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONUTF8="1")
SEARCHERS = "random,optuna_random,tpe,tpe2,optuna_tpe,evolution"

for budget in (12, 40):
    print("\n" + "=" * 60, flush=True)
    print(f"== CIFAR-10 | budget {budget} | 20 seed", flush=True)
    print("=" * 60, flush=True)
    subprocess.run([sys.executable, "run.py", "bench", "sweep_cifar.yaml",
                    "--searchers", SEARCHERS,
                    "--budget", str(budget), "--seeds", "20"],
                   check=False, env=env)
    subprocess.run([sys.executable, "scripts/paired_test.py", "bench.json"],
                   check=False, env=env)
    for name in ("BENCH.md", "bench.json"):
        if os.path.exists(name):
            shutil.copy(name, f"/kaggle/working/{name.split('.')[0]}"
                              f"_cifar_b{budget}.{name.split('.')[1]}")

shutil.rmtree("data", ignore_errors=True)
shutil.rmtree(".racik_cache", ignore_errors=True)
shutil.rmtree(".git", ignore_errors=True)
print("\nSELESAI", flush=True)
