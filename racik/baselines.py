"""Baseline eksternal: bungkus sampler Optuna ke antarmuka ask/tell racik.

Gunanya untuk kejujuran ilmiah: TPE dan evolution rakitan sendiri diadu
langsung dengan implementasi rujukan industri, pada ruang pencarian, budget,
dan seed yang sama persis. Kalau implementasi kita menempel, klaimnya sah;
kalau tertinggal, kita tahu di mana lubangnya.

Ruang bersyarat ditangani lewat define-by-run Optuna: parameter anak hanya
di-`suggest` bila gerbangnya aktif — persis semantik `when:` milik racik.
"""

from .searchers import Searcher


class OptunaSearcher(Searcher):
    """name: optuna_tpe | optuna_random | optuna_cmaes (lihat SAMPLERS)."""

    name = "optuna_tpe"
    sampler_kind = "tpe"

    def __init__(self, space, seed=0, **kwargs):
        super().__init__(space, seed)
        import logging

        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        logging.getLogger("optuna").setLevel(logging.WARNING)

        if self.sampler_kind == "random":
            sampler = optuna.samplers.RandomSampler(seed=seed)
        elif self.sampler_kind == "cmaes":
            # CMA-ES butuh parameter numerik; kategorikal ditangani sampler
            # independen (default TPE) — inilah perilaku bawaan Optuna.
            sampler = optuna.samplers.CmaEsSampler(seed=seed)
        else:
            sampler = optuna.samplers.TPESampler(seed=seed)

        self.study = optuna.create_study(direction="maximize", sampler=sampler)
        self._pending = {}  # kunci konfigurasi -> trial Optuna

    def _suggest(self, trial, name):
        p = self.space.spec[name]
        t = p["type"]
        if t == "choice":
            # nilai kategorikal harus hashable & stabil; str() agar bool/int aman
            return trial.suggest_categorical(name, [str(o) for o in p["options"]])
        if t == "int":
            return trial.suggest_int(name, p["low"], p["high"])
        if t == "loguniform":
            return trial.suggest_float(name, p["low"], p["high"], log=True)
        return trial.suggest_float(name, p["low"], p["high"])

    def _decode(self, name, value):
        """Balikkan str() kategorikal ke tipe aslinya sesuai opsi di ruang."""
        p = self.space.spec[name]
        if p["type"] != "choice":
            return value
        for o in p["options"]:
            if str(o) == value:
                return o
        return value

    def ask(self):
        import json

        trial = self.study.ask()
        config = {}
        for k in self.space.unconditional:
            config[k] = self._decode(k, self._suggest(trial, k))
        for k in self.space.conditional:
            if self.space.active(k, config):
                config[k] = self._decode(k, self._suggest(trial, k))
        self._pending[json.dumps(config, sort_keys=True, default=str)] = trial
        return config

    def tell(self, config, score):
        import json

        super().tell(config, score)
        trial = self._pending.pop(
            json.dumps(config, sort_keys=True, default=str), None)
        if trial is not None:
            self.study.tell(trial, score)


class OptunaRandomSearcher(OptunaSearcher):
    name = "optuna_random"
    sampler_kind = "random"


class OptunaCmaesSearcher(OptunaSearcher):
    name = "optuna_cmaes"
    sampler_kind = "cmaes"


OPTUNA_SEARCHERS = {
    OptunaSearcher.name: OptunaSearcher,
    OptunaRandomSearcher.name: OptunaRandomSearcher,
    OptunaCmaesSearcher.name: OptunaCmaesSearcher,
}
