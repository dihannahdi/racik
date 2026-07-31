"""Audit validitas benchmark — metode "lengan plasebo" untuk benchmark ML.

Gagasan pokoknya dipinjam dari uji klinis, bukan dari literatur ML:

  LENGAN PLASEBO. Uji klinis menyertakan kelompok plasebo agar tahu seberapa
  besar perbaikan yang muncul TANPA obat. Padanannya di sini: dua implementasi
  independen dari algoritme yang IDENTIK (mis. `random` vs `optuna_random`).
  Keduanya seharusnya seri. Selisih terukurnya adalah (a) lantai noise —
  batas bawah klaim yang sah, dan (b) taksiran VARIANS NULL protokol itu.

  (b) itu kuncinya: varians null adalah bahan baku analisis daya uji. Dari
  satu lengan plasebo kita bisa menghitung, untuk protokol itu:
    - MDE (minimum detectable effect): efek terkecil yang MUNGKIN terdeteksi
      pada n seed tertentu — kalau klaim seseorang lebih kecil dari MDE-nya,
      klaim itu tidak terdukung apa pun jumlah datanya.
    - n yang dibutuhkan untuk mengklaim efek sebesar Δ.

  INDEKS KETIDAKSTABILAN PAPAN PERINGKAT (LII). Dari satu run ber-seed banyak,
  kita subsampel k seed berulang kali dan menghitung seberapa sering juaranya
  BERBEDA dari juara pada seluruh seed. Ini menjawab pertanyaan yang jarang
  ditanyakan: "kalau eksperimen ini diulang dengan k seed, berapa peluang
  papan peringkatnya menunjuk pemenang yang lain?"

Semua nonparametrik (uji sign-flip + subsampling), jadi tidak mengasumsikan
normalitas skor. Yang diasumsikan: seed saling bebas dan dipasangkan lintas
pencari (protokol racik menjamin ini — seed yang sama dipakai semua pencari).
"""

import itertools
import math
import random


# --- uji sign-flip berpasangan ------------------------------------------------

def sign_flip_p(diffs, rng=None, reps=4000, exact_limit=14):
    """p 2-sisi: peluang |rata-rata| sebesar ini bila tanda acak (H0 simetri).

    Enumerasi eksak untuk n kecil; sampling untuk n besar (2^n terlalu besar).
    """
    n = len(diffs)
    if n == 0:
        return 1.0
    obs = abs(sum(diffs) / n)
    if n <= exact_limit:
        signs_iter = itertools.product((1, -1), repeat=n)
        total = 2 ** n
    else:
        rng = rng or random.Random(0)
        signs_iter = (tuple(rng.choice((1, -1)) for _ in range(n))
                      for _ in range(reps))
        total = reps
    hits = sum(1 for s in signs_iter
               if abs(sum(si * di for si, di in zip(s, diffs)) / n) >= obs - 1e-12)
    return hits / total


def p_floor(n):
    """p terkecil yang mungkin dicapai uji ini pada n pasang: 2/2^n."""
    return 2.0 / (2 ** n) if n < 60 else 0.0


# --- lengan plasebo -----------------------------------------------------------

def placebo_arm(finals, name_a, name_b):
    """Ukur lantai noise dan varians null dari dua algoritme identik."""
    diffs = [a - b for a, b in zip(finals[name_a], finals[name_b])]
    n = len(diffs)
    mean = sum(diffs) / n
    sd = (sum((d - mean) ** 2 for d in diffs) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return {
        "pasangan": f"{name_a} vs {name_b}",
        "n": n,
        "lantai_noise": abs(mean),
        "sd_null": sd,          # simpangan baku selisih berpasangan di bawah H0
        "p": sign_flip_p(diffs),
    }


# --- analisis daya uji dari varians null --------------------------------------

def power_at(effect, sd_null, n, rng, alpha=0.05, sims=200, reps=600):
    """Peluang uji sign-flip menolak H0 bila efek sebenarnya = `effect`."""
    hits = 0
    for _ in range(sims):
        diffs = [rng.gauss(effect, sd_null) for _ in range(n)]
        if sign_flip_p(diffs, rng=rng, reps=reps) < alpha:
            hits += 1
    return hits / sims


def mde(sd_null, n, rng, alpha=0.05, target_power=0.8, grid=None):
    """Minimum detectable effect: efek terkecil dengan daya >= target_power."""
    if p_floor(n) > alpha:
        return None                      # mustahil signifikan pada n ini
    grid = grid or [sd_null * f for f in
                    (0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0)]
    for eff in grid:
        if power_at(eff, sd_null, n, rng, alpha=alpha) >= target_power:
            return eff
    return None                          # di luar jangkauan grid


def required_n(effect, sd_null, rng, alpha=0.05, target_power=0.8, n_max=200):
    """Jumlah seed minimum untuk mengklaim efek sebesar `effect`."""
    n = 6                                # di bawah ini p<0.05 mustahil
    while n <= n_max:
        if power_at(effect, sd_null, n, rng, alpha=alpha) >= target_power:
            return n
        n = n + 2 if n < 20 else int(n * 1.4)
    return None


# --- indeks ketidakstabilan papan peringkat ----------------------------------

def leaderboard_instability(finals, k_values, rng, reps=2000):
    """P(juara pada k seed != juara pada seluruh seed), plus kesepakatan
    peringkat rata-rata (Spearman) antara subsampel dan acuan."""
    names = list(finals)
    n = len(finals[names[0]])

    def mean_of(name, idx):
        return sum(finals[name][i] for i in idx) / len(idx)

    ref_idx = list(range(n))
    ref_rank = _ranking(names, {m: mean_of(m, ref_idx) for m in names})
    ref_winner = ref_rank[0]

    out = []
    for k in k_values:
        if k > n:
            continue
        wrong, rhos = 0, []
        for _ in range(reps):
            idx = rng.sample(range(n), k)
            means = {m: mean_of(m, idx) for m in names}
            rank = _ranking(names, means)
            if rank[0] != ref_winner:
                wrong += 1
            rhos.append(_spearman_rank(ref_rank, rank))
        out.append({
            "k": k,
            "p_juara_berubah": wrong / reps,
            "spearman_rata2": sum(rhos) / len(rhos),
        })
    return {"acuan_n": n, "juara_acuan": ref_winner, "kurva": out}


def _ranking(names, means):
    return sorted(names, key=lambda m: -means[m])


def _spearman_rank(rank_a, rank_b):
    """Spearman antara dua urutan (daftar nama terurut)."""
    pos_a = {m: i for i, m in enumerate(rank_a)}
    pos_b = {m: i for i, m in enumerate(rank_b)}
    n = len(rank_a)
    if n < 2:
        return 1.0
    d2 = sum((pos_a[m] - pos_b[m]) ** 2 for m in rank_a)
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


# --- laporan lengkap ----------------------------------------------------------

def audit(curves, sham=None, seed=0, k_values=(3, 5, 8, 10, 15, 20, 30, 40)):
    """curves: {nama_pencari: [kurva_seed0, ...]} dari bench.json."""
    rng = random.Random(seed)
    finals = {n: [c[-1] for c in curves[n]] for n in curves}
    n_seed = len(next(iter(finals.values())))

    # lengan plasebo: pakai pasangan yang diberikan, atau tebak dari nama
    if sham is None:
        cands = [(a, b) for a, b in itertools.combinations(finals, 2)
                 if a.endswith("random") and b.endswith("random")]
        sham = cands[0] if cands else None

    rep = {"n_seed": n_seed, "p_floor": p_floor(n_seed), "finals": finals}
    rep["placebo"] = placebo_arm(finals, *sham) if sham else None

    if rep["placebo"]:
        sd = rep["placebo"]["sd_null"]
        rep["mde"] = [{"n": k, "mde": mde(sd, k, rng)}
                      for k in (3, 5, 10, 20, 40) if k <= max(40, n_seed)]
        rep["required_n"] = [{"effect": e, "n": required_n(e, sd, rng)}
                             for e in (0.005, 0.01, 0.02, 0.05)]

    rep["lii"] = leaderboard_instability(
        finals, [k for k in k_values if k <= n_seed], rng)

    rep["comparisons"] = []
    floor = rep["placebo"]["lantai_noise"] if rep["placebo"] else None
    for a, b in itertools.combinations(finals, 2):
        d = [x - y for x, y in zip(finals[a], finals[b])]
        m = sum(d) / n_seed
        p = sign_flip_p(d)
        rep["comparisons"].append({
            "a": a, "b": b, "delta": m, "p": p,
            "signifikan": p < 0.05,
            "di_bawah_lantai": floor is not None and abs(m) < floor,
        })
    return rep
