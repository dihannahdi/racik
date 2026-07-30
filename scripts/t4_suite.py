"""racik — suite eksperimen T4 di Kaggle.

Alur: clone repo GitHub -> susun data dari dataset Intel yang ter-mount ->
jalankan (1) benchmark antar-algoritme, (2) validasi zero-cost proxy pada
arsitektur murni, (3) halving polos vs halving ber-prefilter proxy.
Semua keluaran tertulis ke /kaggle/working (otomatis jadi output kernel).
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

# --- temukan data: mount /kaggle/input, atau unduh sendiri (mount bisa flaky)
ROOT = "/kaggle/input"
if os.path.isdir(ROOT) and os.listdir(ROOT):
    SRC = ROOT
    print("isi /kaggle/input:", os.listdir(ROOT), flush=True)
else:
    import kagglehub
    SRC = kagglehub.dataset_download("puneet6060/intel-image-classification")
    print("mount kosong — unduh via kagglehub:", SRC, flush=True)


def find_split(root, name):
    """Cari folder split (seg_train/seg_test) di mana pun dia berada di mount,
    kembalikan direktori yang langsung berisi subfolder kelas."""
    for dirpath, dirnames, _files in os.walk(root):
        if os.path.basename(dirpath) == name:
            inner = os.path.join(dirpath, name)
            cand = inner if os.path.isdir(inner) else dirpath
            subs = [d for d in os.listdir(cand)
                    if os.path.isdir(os.path.join(cand, d))]
            if subs and not any(s.startswith("seg_") for s in subs):
                return cand
    raise FileNotFoundError(f"{name} tidak ditemukan di {root}")


rng = random.Random(42)
for split, srcname, cap in [("train", "seg_train", 2000),
                            ("val", "seg_test", 400)]:
    base = find_split(SRC, srcname)
    print(f"{split} <- {base}", flush=True)
    for cls in sorted(os.listdir(base)):
        files = sorted(os.listdir(os.path.join(base, cls)))
        rng.shuffle(files)
        dst = os.path.join("data/kaggle", split, cls)
        os.makedirs(dst, exist_ok=True)
        for f in files[:cap]:
            shutil.copy(os.path.join(base, cls, f), dst)
print("data siap:", sum(len(fs) for _, _, fs in os.walk("data/kaggle")), "file")

# --- sweep untuk T4: data lebih besar, 3 epoch per trial --------------------
with open("sweep_t4.yaml", "w") as f:
    f.write("""backend: vision
task: >
  Klasifikasi Intel Image (Kaggle, T4): benchmark algoritme pencari dan
  zero-cost proxy pada budget lebih serius.
dataset:
  name: folder
  dir: ./data/kaggle
  image_size: 64
  limit_train: 4000
  limit_val: 1200
epochs: 3
seed: 42
space:
  arch:      {type: choice, options: [tiny, resnet18, mobilenet_v3_small]}
  lr:        {type: loguniform, low: 0.0001, high: 0.01}
  optimizer: {type: choice, options: [adam, sgd]}
  augment:   {type: choice, options: [none, basic]}
  depth:     {type: int, low: 2, high: 4, when: {arch: tiny}}
  width:     {type: choice, options: [16, 32, 48], when: {arch: tiny}}
  block:     {type: choice, options: [basic, residual, depthwise], when: {arch: tiny}}
  use_se:    {type: choice, options: [true, false], when: {arch: tiny}}
""")

env = dict(os.environ, PYTHONUNBUFFERED="1")


def run(label, cmd):
    print(f"\n{'=' * 60}\n== {label}\n{'=' * 60}", flush=True)
    subprocess.run(cmd, check=False, env=env)


# 1) validasi proxy pada arsitektur murni (14 model @ 3 epoch)
run("VALIDASI PROXY", [sys.executable, "scripts/validate_proxy.py", "sweep_t4.yaml"])

# 2) benchmark antar-algoritme (12 trial x 2 seed x 3 algoritme)
run("BENCH ALGORITME", [sys.executable, "run.py", "bench", "sweep_t4.yaml",
                        "--budget", "12", "--seeds", "2"])

# 3) halving polos vs ber-prefilter proxy (seed sama, budget training sama)
run("HALVING POLOS", [sys.executable, "run.py", "halving", "sweep_t4.yaml",
                      "--n0", "9", "--eta", "3", "--r0", "1", "--rmax", "9",
                      "--seed", "11"])
if os.path.exists("REPORT.md"):
    shutil.copy("REPORT.md", "/kaggle/working/REPORT_halving_polos.md")

run("HALVING + PREFILTER", [sys.executable, "run.py", "halving", "sweep_t4.yaml",
                            "--n0", "9", "--eta", "3", "--r0", "1", "--rmax", "9",
                            "--seed", "11", "--pool", "60", "--proxy", "naswot"])
if os.path.exists("REPORT.md"):
    shutil.copy("REPORT.md", "/kaggle/working/REPORT_halving_prefilter.md")

# --- kumpulkan keluaran -----------------------------------------------------
for name in ["BENCH.md", "bench.json", "proxy_validation.json"]:
    if os.path.exists(name):
        shutil.copy(name, f"/kaggle/working/{name}")
shutil.rmtree("data", ignore_errors=True)  # jangan ikutkan data di output kernel
print("\nSELESAI — keluaran di /kaggle/working", flush=True)
