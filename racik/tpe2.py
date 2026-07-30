"""TPE v2 — Parzen estimator setia pada formulasi asli, ditulis dari nol.

Latar: adu langsung menunjukkan TPE v1 kita signifikan tertinggal dari TPE
Optuna (delta 0.056, p=0.029 pada 40 seed). Bedah sumbernya menunjukkan
penyebabnya bukan tuning, melainkan empat hal algoritmik — semuanya ada di
Bergstra dkk. (2011) dan hilang di v1 kita:

  1. BANDWIDTH ADAPTIF PER TITIK (yang terbesar pengaruhnya).
     v1: satu bandwidth global h = rentang / sqrt(n) untuk semua kernel.
     v2: sigma_i = max(jarak ke tetangga kiri, jarak ke tetangga kanan) pada
         daftar mu yang diurutkan + sentinel [low, high]. Efeknya: daerah yang
         padat observasi dapat kernel tajam (eksploitasi presisi), daerah
         renggang dapat kernel lebar (tetap menjelajah). Satu bandwidth global
         memaksa kompromi buruk di dua-duanya.
  2. MAGIC CLIP: sigma dibatasi bawah pada rentang/min(100, n_kernel+1) agar
     kernel tidak menjadi paku (delta) saat dua observasi hampir sama.
  3. KERNEL PRIOR EKSPLISIT: satu Gaussian di titik tengah domain dengan
     sigma = rentang penuh dan bobot 1 — bukan campuran seragam ad hoc.
  4. GAMMA ADAPTIF + BOBOT USIA: n_good = min(ceil(0.1n), 25) (v1: 0.25n
     tetap); observasi lebih dari 25 terlama bobotnya meluruh linear, jadi
     model mengikuti wilayah yang sedang dieksplorasi.

Distribusi kernelnya normal TERPANCUNG pada [low, high] — massa probabilitas
tidak bocor keluar domain seperti pada Gaussian biasa.
Sadar-kondisi: tiap parameter hanya belajar dari observasi yang memuatnya.
"""

import math

from .searchers import Searcher

SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)


def _trunc_norm_logpdf(x, mu, sigma, lo, hi):
    """log pdf normal terpancung pada [lo, hi]."""
    z = (x - mu) / sigma
    mass = 0.5 * (math.erf((hi - mu) / (sigma * SQRT2))
                  - math.erf((lo - mu) / (sigma * SQRT2)))
    return -0.5 * z * z - math.log(sigma * SQRT2PI) - math.log(mass + 1e-12)


def _logsumexp(vals):
    m = max(vals)
    if m == -math.inf:
        return -math.inf
    return m + math.log(sum(math.exp(v - m) for v in vals))


def default_weights(n):
    """Bobot observasi menurut usia: 25 terbaru penuh, sisanya meluruh linear."""
    if n <= 25:
        return [1.0] * n
    ramp = [(1.0 + i * (n - 1.0) / max(n - 26, 1)) / n for i in range(n - 25)]
    return ramp + [1.0] * 25


class ParzenNumeric:
    """Campuran normal terpancung: kernel per observasi + satu kernel prior."""

    def __init__(self, values, lo, hi, weights=None):
        n = len(values)
        w = list(weights) if weights else [1.0] * n

        order = sorted(range(n), key=lambda i: values[i])
        smus = [values[i] for i in order]
        aug = [lo] + smus + [hi]
        ssig = [max(aug[i + 1] - aug[i], aug[i + 2] - aug[i + 1]) for i in range(n)]
        # consider_endpoints=False: kernel ujung memakai jarak ke tetangga
        # dalam, bukan ke batas domain (batas sering jauh dan melebarkannya)
        if n >= 2:
            ssig[0] = aug[2] - aug[1]
            ssig[-1] = aug[-2] - aug[-3]

        span = hi - lo
        minsigma = span / min(100.0, n + 2.0)   # magic clip (n_kernel = n + 1)
        sigmas = [0.0] * n
        for rank, i in enumerate(order):
            sigmas[i] = min(max(ssig[rank], minsigma), span)

        self.mus = list(values) + [0.5 * (lo + hi)]   # + kernel prior
        self.sigmas = sigmas + [span]
        self.lo, self.hi = lo, hi
        total = sum(w) + 1.0                          # bobot prior = 1
        self.logw = [math.log(x / total) for x in w] + [math.log(1.0 / total)]

    def logpdf(self, x):
        return _logsumexp([
            lw + _trunc_norm_logpdf(x, mu, s, self.lo, self.hi)
            for lw, mu, s in zip(self.logw, self.mus, self.sigmas)
        ])

    def sample(self, rng):
        r, acc = rng.random(), 0.0
        idx = len(self.mus) - 1
        for i, lw in enumerate(self.logw):
            acc += math.exp(lw)
            if r <= acc:
                idx = i
                break
        mu, s = self.mus[idx], self.sigmas[idx]
        for _ in range(50):                           # rejection sampling
            v = rng.gauss(mu, s)
            if self.lo <= v <= self.hi:
                return v
        return min(max(mu, self.lo), self.hi)


class ParzenCategorical:
    """Frekuensi berbobot + prior seragam (satu observasi setara)."""

    def __init__(self, values, options, weights=None):
        w = list(weights) if weights else [1.0] * len(values)
        k = len(options)
        counts = {o: 1.0 / k for o in options}         # prior
        for v, wi in zip(values, w):
            if v in counts:
                counts[v] += wi
        total = sum(counts.values())
        self.options = list(options)
        self.probs = [counts[o] / total for o in self.options]

    def logpdf(self, value):
        for o, p in zip(self.options, self.probs):
            if o == value:
                return math.log(p + 1e-12)
        return math.log(1e-12)

    def sample(self, rng):
        r, acc = rng.random(), 0.0
        for o, p in zip(self.options, self.probs):
            acc += p
            if r <= acc:
                return o
        return self.options[-1]


class TPE2Searcher(Searcher):
    name = "tpe2"

    def __init__(self, space, seed=0, n_startup=10, n_candidates=24):
        super().__init__(space, seed)
        self.n_startup = n_startup
        self.n_candidates = n_candidates

    # --- ruang x: loguniform dimodelkan di kawasan log -----------------------
    def _bounds(self, p):
        if p["type"] == "loguniform":
            return math.log(p["low"]), math.log(p["high"])
        return float(p["low"]), float(p["high"])

    def _to_x(self, p, v):
        return math.log(v) if p["type"] == "loguniform" else float(v)

    def _from_x(self, p, x):
        if p["type"] == "loguniform":
            return math.exp(x)
        if p["type"] == "int":
            return int(round(x))
        return x

    def _estimator(self, name, obs):
        """obs: daftar (config, skor) berurut usia (lama -> baru)."""
        p = self.space.spec[name]
        vals = [c[name] for c, _ in obs if name in c]          # sadar-kondisi
        w = default_weights(len(vals))
        if p["type"] == "choice":
            return ParzenCategorical(vals, p["options"], w)
        lo, hi = self._bounds(p)
        return ParzenNumeric([self._to_x(p, v) for v in vals], lo, hi, w)

    def ask(self):
        if len(self.history) < self.n_startup:
            return self.space.sample(self.rng)

        # gamma adaptif: n_good = min(ceil(0.1n), 25), diambil dari peringkat
        # skor, tetapi tiap subset dipertahankan urut usia untuk bobot usia
        n = len(self.history)
        n_good = min(math.ceil(0.1 * n), 25)
        ranked = sorted(range(n), key=lambda i: self.history[i][1], reverse=True)
        good_idx = set(ranked[:n_good])
        good = [self.history[i] for i in range(n) if i in good_idx]
        bad = [self.history[i] for i in range(n) if i not in good_idx]

        gm = {k: self._estimator(k, good) for k in self.space.keys}
        bm = {k: self._estimator(k, bad) for k in self.space.keys}

        best_cfg, best_ei = None, -math.inf
        for _ in range(self.n_candidates):
            cand, ei = {}, 0.0
            for k in self.space.unconditional:
                cand[k] = self._draw(k, gm[k])
            for k in self.space.conditional:
                if self.space.active(k, cand):
                    cand[k] = self._draw(k, gm[k])
            for k in cand:                     # EI = log l(x) - log g(x)
                p = self.space.spec[k]
                x = cand[k] if p["type"] == "choice" else self._to_x(p, cand[k])
                ei += gm[k].logpdf(x) - bm[k].logpdf(x)
            ei /= max(len(cand), 1)            # normalisasi: kandidat beda panjang
            if ei > best_ei:
                best_cfg, best_ei = cand, ei
        return best_cfg

    def _draw(self, name, model):
        p = self.space.spec[name]
        if p["type"] == "choice":
            return model.sample(self.rng)
        v = self._from_x(p, model.sample(self.rng))
        if p["type"] == "int":
            v = min(max(v, p["low"]), p["high"])
        return v
