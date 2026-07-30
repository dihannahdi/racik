"""Algoritme pencari, diimplementasikan dari nol dengan antarmuka ask/tell.

  ask()          -> usulkan satu konfigurasi untuk dievaluasi
  tell(cfg, s)   -> laporkan skor konfigurasi itu (0..1, makin tinggi makin baik)

Tiga algoritme:
  RandomSearcher    : baseline — sampling acak murni (Bergstra & Bengio, 2012).
  EvolutionSearcher : regularized evolution — turnamen + mutasi + aging
                      (Real dkk., 2019, "Regularized Evolution for Image
                      Classifier Architecture Search").
  TPESearcher       : Tree-structured Parzen Estimator — modelkan distribusi
                      konfigurasi "bagus" vs "buruk", pilih kandidat dengan
                      rasio kepadatan l(x)/g(x) tertinggi (Bergstra dkk., 2011,
                      "Algorithms for Hyper-Parameter Optimization").
"""

import math
import random
from collections import deque


class Searcher:
    name = "base"

    def __init__(self, space, seed=0):
        self.space = space
        self.rng = random.Random(seed)
        self.history = []  # daftar (config, score)

    def ask(self) -> dict:
        raise NotImplementedError

    def tell(self, config: dict, score: float):
        self.history.append((config, score))


class RandomSearcher(Searcher):
    name = "random"

    def ask(self):
        return self.space.sample(self.rng)


class EvolutionSearcher(Searcher):
    """Regularized evolution: populasi FIFO (aging), seleksi turnamen, mutasi 1 gen."""

    name = "evolution"

    def __init__(self, space, seed=0, population=8, tournament=3):
        super().__init__(space, seed)
        self.population = deque(maxlen=population)
        self.tournament = tournament

    def ask(self):
        if len(self.population) < self.population.maxlen:
            return self.space.sample(self.rng)
        contenders = self.rng.sample(list(self.population),
                                     min(self.tournament, len(self.population)))
        parent = max(contenders, key=lambda cs: cs[1])[0]
        return self.space.mutate(parent, self.rng)

    def tell(self, config, score):
        super().tell(config, score)
        self.population.append((config, score))  # yang tertua otomatis "mati"


class TPESearcher(Searcher):
    """TPE sederhana: per-parameter, independen, KDE Gaussian untuk numerik
    dan frekuensi (dengan smoothing) untuk kategorikal."""

    name = "tpe"

    def __init__(self, space, seed=0, n_init=5, n_candidates=24, gamma=0.25):
        super().__init__(space, seed)
        self.n_init = n_init
        self.n_candidates = n_candidates
        self.gamma = gamma

    # --- kepadatan per parameter -------------------------------------------
    def _bounds(self, p):
        if p["type"] == "loguniform":
            return math.log(p["low"]), math.log(p["high"])
        return float(p["low"]), float(p["high"])

    def _to_x(self, p, v):
        return math.log(v) if p["type"] == "loguniform" else float(v)

    def _log_dens_numeric(self, p, value, points):
        # KDE Gaussian DICAMPUR prior seragam berbobot satu observasi
        # (resep TPE asli) — mencegah model mengunci diri pada data awal.
        lo, hi = self._bounds(p)
        x = self._to_x(p, value)
        pts = [self._to_x(p, v) for v in points]
        prior = 1.0 / (hi - lo + 1e-12)
        if not pts:
            return math.log(prior)
        h = max((hi - lo) / (len(pts) ** 0.5), 1e-6)  # bandwidth heuristik
        kde = sum(math.exp(-0.5 * ((x - m) / h) ** 2) for m in pts)
        kde /= h * math.sqrt(2 * math.pi)
        dens = (prior + kde) / (len(pts) + 1)
        return math.log(dens + 1e-12)

    def _log_dens_choice(self, p, value, points):
        k = len(p["options"])
        count = sum(1 for v in points if v == value)
        return math.log((count + 1) / (len(points) + k))  # add-one smoothing

    def _log_dens(self, name, value, observations):
        # sadar-kondisi: hanya belajar dari observasi yang memuat parameter ini
        p = self.space.spec[name]
        points = [c[name] for c, _ in observations if name in c]
        if p["type"] == "choice":
            return self._log_dens_choice(p, value, points)
        return self._log_dens_numeric(p, value, points)

    # --- sampling kandidat dari model "bagus" ------------------------------
    def _sample_from(self, name, good):
        # peluang eksplorasi = bobot prior dalam campuran: 1/(n+1) —
        # besar saat data sedikit, mengecil sendiri saat data bertambah
        p = self.space.spec[name]
        points = [c[name] for c, _ in good if name in c]
        if not points or self.rng.random() < 1.0 / (len(points) + 1):
            return self.space._sample_one(name, self.rng)
        if p["type"] == "choice":
            return self.rng.choice(points)
        lo, hi = self._bounds(p)
        h = max((hi - lo) / (len(points) ** 0.5), 1e-6)
        x = self._to_x(p, self.rng.choice(points)) + self.rng.gauss(0, h)
        x = min(max(x, lo), hi)
        v = math.exp(x) if p["type"] == "loguniform" else x
        return int(round(v)) if p["type"] == "int" else v

    def _candidate(self, good):
        """Bangun kandidat menghormati struktur bersyarat ruang pencarian."""
        cand = {k: self._sample_from(k, good) for k in self.space.unconditional}
        for k in self.space.conditional:
            if self.space.active(k, cand):
                cand[k] = self._sample_from(k, good)
        return cand

    def ask(self):
        if len(self.history) < self.n_init:
            return self.space.sample(self.rng)

        ranked = sorted(self.history, key=lambda cs: cs[1], reverse=True)
        n_good = max(1, int(len(ranked) * self.gamma))
        good, bad = ranked[:n_good], ranked[n_good:]

        best_cfg, best_ei = None, -math.inf
        for _ in range(self.n_candidates):
            cand = self._candidate(good)
            # EI dihitung hanya atas parameter yang aktif di kandidat ini,
            # dinormalkan per-parameter agar kandidat beda panjang tetap adil
            ei = sum(self._log_dens(k, cand[k], good)
                     - self._log_dens(k, cand[k], bad)
                     for k in cand) / max(len(cand), 1)
            if ei > best_ei:
                best_cfg, best_ei = cand, ei
        return best_cfg


SEARCHERS = {
    RandomSearcher.name: RandomSearcher,
    EvolutionSearcher.name: EvolutionSearcher,
    TPESearcher.name: TPESearcher,
}


def all_searchers():
    """Semua pencari: milik racik + baseline eksternal (kalau optuna terpasang)."""
    out = dict(SEARCHERS)
    from .tpe2 import TPE2Searcher
    out[TPE2Searcher.name] = TPE2Searcher
    try:
        from .baselines import OPTUNA_SEARCHERS
        out.update(OPTUNA_SEARCHERS)
    except ImportError:
        pass  # optuna opsional
    return out


def make_searcher(name, space, seed=0):
    registry = all_searchers()
    if name not in registry:
        raise ValueError(f"searcher tidak dikenal: {name} "
                         f"(pilihan: {sorted(registry)})")
    return registry[name](space, seed=seed)
