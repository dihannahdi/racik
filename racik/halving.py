"""Successive halving: alokasikan budget training hanya ke kandidat yang menjanjikan.

Ide (Jamieson & Talwalkar, AISTATS 2016; dasar dari Hyperband dan ASHA,
Li dkk., 2018): evaluasi banyak kandidat dengan budget kecil (fidelity rendah,
mis. 1 epoch), buang yang jelek, naikkan budget hanya untuk yang lolos.
Dengan budget total yang sama, jauh lebih banyak kandidat bisa dijajaki
daripada melatih semuanya sampai tuntas.

  rung 0: n0 kandidat  @ r0 epoch
  rung 1: n0/eta       @ r0*eta epoch
  rung 2: n0/eta^2     @ r0*eta^2 epoch  ... dst sampai rmax atau tersisa satu
"""

import math

from .config import config_label
from .runner import evaluate


def prefilter(cfg, searcher, pool, keep, proxy="naswot", verbose=True):
    """Rung -1, gratis: saring `pool` kandidat dengan zero-cost proxy (tanpa
    training), loloskan `keep` terbaik ke halving. Biaya: detik CPU per kandidat."""
    import time

    from .proxies import ProxyScorer

    scorer = ProxyScorer(cfg, proxy=proxy)
    t0 = time.time()
    scored = []
    for _ in range(pool):
        c = searcher.ask()
        scored.append((scorer.score(c), c))
    scored.sort(key=lambda sc: sc[0], reverse=True)
    if verbose:
        print(f"-- rung -1 (gratis): {pool} kandidat dinilai proxy `{proxy}` "
              f"dalam {time.time() - t0:.1f}s, lolos {keep} --")
        for s, c in scored[:keep]:
            print(f"  {proxy}={s:.3f}  {config_label(c)}")
    return [c for _s, c in scored[:keep]]


def run_halving(cfg, searcher, backend, n0=9, eta=3, r0=1, rmax=None,
                use_cache=True, verbose=True, initial_configs=None):
    configs = initial_configs or [searcher.ask() for _ in range(n0)]
    r = r0
    rung = 0
    history = []
    spent = 0  # budget dalam satuan epoch

    while True:
        if verbose:
            print(f"\n-- rung {rung}: {len(configs)} kandidat @ {r} epoch --")
        rows = []
        for c in configs:
            row = evaluate(cfg, c, backend, use_cache=use_cache, fidelity=r)
            searcher.tell(c, row["score"])
            rows.append(row)
            history.append(row)
            spent += r
            if verbose:
                tag = " (cache)" if row.get("dari_cache") else ""
                print(f"  skor={row['score']:.4f}{tag}  {config_label(c)}")

        rows.sort(key=lambda x: x["score"], reverse=True)
        if len(configs) == 1 or (rmax and r >= rmax):
            break
        keep = max(1, math.ceil(len(rows) / eta))
        configs = [x["config"] for x in rows[:keep]]
        r = min(r * eta, rmax) if rmax else r * eta
        rung += 1

    best = max(history, key=lambda x: x["score"])
    if verbose:
        print(f"\nBudget terpakai: {spent} epoch-unit "
              f"(bandingkan: {n0} kandidat x {r} epoch penuh = {n0 * r})")
    return history, best, spent
