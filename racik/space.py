"""Ruang pencarian: definisi parameter, sampling, mutasi — dengan dukungan
parameter BERSYARAT (conditional), kesulitan nyata di NAS.

Tipe parameter (di sweep.yaml bagian `space`):
  choice     : {type: choice, options: [a, b, c]}
  uniform    : {type: uniform, low: 0.0, high: 1.0}
  loguniform : {type: loguniform, low: 1e-4, high: 1e-2}
  int        : {type: int, low: 2, high: 5}

Syarat aktif (opsional):
  depth: {type: int, low: 2, high: 4, when: {arch: tiny}}
  -> depth hanya ada di konfigurasi jika arch == tiny. Konfigurasi disimpan
     dalam bentuk kanonik: parameter tak aktif TIDAK muncul, sehingga dua
     konfigurasi yang hanya beda di parameter mati dianggap identik
     (tidak buang budget dua kali untuk eksperimen yang sama).

Batasan: syarat hanya boleh merujuk parameter yang tidak bersyarat.
"""

import math


class Space:
    def __init__(self, spec: dict):
        if not spec:
            raise ValueError("sweep.yaml harus punya bagian `space` yang tidak kosong.")
        self.spec = spec
        self.keys = list(spec.keys())
        self.unconditional = [k for k in self.keys if "when" not in spec[k]]
        self.conditional = [k for k in self.keys if "when" in spec[k]]

    def active(self, name, config) -> bool:
        when = self.spec[name].get("when")
        if not when:
            return True
        for k, v in when.items():
            allowed = v if isinstance(v, list) else [v]
            if config.get(k) not in allowed:
                return False
        return True

    def _sample_one(self, name, rng):
        p = self.spec[name]
        t = p["type"]
        if t == "choice":
            return rng.choice(p["options"])
        if t == "uniform":
            return rng.uniform(p["low"], p["high"])
        if t == "loguniform":
            return math.exp(rng.uniform(math.log(p["low"]), math.log(p["high"])))
        if t == "int":
            return rng.randint(p["low"], p["high"])
        raise ValueError(f"tipe parameter tidak dikenal: {name}: {t}")

    def sample(self, rng) -> dict:
        config = {k: self._sample_one(k, rng) for k in self.unconditional}
        for k in self.conditional:
            if self.active(k, config):
                config[k] = self._sample_one(k, rng)
        return config

    def canonicalize(self, config, rng) -> dict:
        """Bentuk kanonik: buang parameter mati, sampel parameter yang baru hidup."""
        out = {}
        for k in self.unconditional:
            out[k] = config[k] if k in config else self._sample_one(k, rng)
        for k in self.conditional:
            if self.active(k, out):
                out[k] = config[k] if k in config else self._sample_one(k, rng)
        return out

    def _mutate_value(self, name, value, rng):
        p = self.spec[name]
        t = p["type"]
        if t == "choice":
            others = [o for o in p["options"] if o != value]
            return rng.choice(others) if others else value
        if t == "int":
            return min(max(value + rng.choice([-1, 1]), p["low"]), p["high"])
        if t == "uniform":
            span = p["high"] - p["low"]
            return min(max(value + rng.gauss(0, 0.2 * span), p["low"]), p["high"])
        if t == "loguniform":
            lo, hi = math.log(p["low"]), math.log(p["high"])
            x = math.log(value) + rng.gauss(0, 0.2 * (hi - lo))
            return math.exp(min(max(x, lo), hi))
        return value

    def mutate(self, config: dict, rng) -> dict:
        """Ubah tepat satu parameter AKTIF, lalu kanonikalisasi
        (kalau parameter gerbang berubah, anak yang baru hidup ikut disampel)."""
        child = dict(config)
        name = rng.choice(list(child.keys()))
        child[name] = self._mutate_value(name, child[name], rng)
        return self.canonicalize(child, rng)
