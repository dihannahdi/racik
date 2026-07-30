#!/usr/bin/env python
"""racik — riset algoritme pencarian arsitektur & hyperparameter vision.

Pakai:
  python run.py search sweep.yaml --searcher tpe --budget 20
  python run.py bench  sweep_dummy.yaml --budget 30 --seeds 5
"""

import argparse

from racik.config import config_label, load_sweep
from racik.halving import run_halving
from racik.report import write_bench, write_report
from racik.runner import run_search
from racik.searchers import all_searchers, make_searcher
from racik.space import Space


def make_backend(cfg):
    if cfg["backend"] == "dummy":
        from racik.dummy_backend import DummyBackend
        return DummyBackend(cfg)
    from racik.vision_backend import VisionBackend
    return VisionBackend(cfg)


def cmd_search(args):
    cfg = load_sweep(args.sweep)
    space = Space(cfg["space"])
    print(f"Backend : {cfg['backend']}  |  Algoritme: {args.searcher}  |  "
          f"Budget: {args.budget} trial")
    print(f"Ruang   : {', '.join(space.keys)}")
    if args.dry_run:
        import random
        print("Contoh 3 sampel dari ruang pencarian:")
        rng = random.Random(args.seed)
        for _ in range(3):
            print("  -", config_label(space.sample(rng)))
        return

    backend = make_backend(cfg)
    searcher = make_searcher(args.searcher, space, seed=args.seed)
    history, _ = run_search(cfg, searcher, backend, args.budget,
                            use_cache=not args.no_cache)
    rows = write_report(history, cfg, args.searcher)
    if rows:
        print(f"\nTerbaik: [{rows[0]['score']:.4f}] {config_label(rows[0]['config'])}")


def cmd_halving(args):
    cfg = load_sweep(args.sweep)
    space = Space(cfg["space"])
    backend = make_backend(cfg)
    searcher = make_searcher(args.searcher, space, seed=args.seed)
    print(f"Successive halving: n0={args.n0}, eta={args.eta}, "
          f"r0={args.r0}, rmax={args.rmax}"
          + (f", prefilter proxy={args.proxy} pool={args.pool}" if args.pool else ""))

    initial = None
    if args.pool:
        from racik.halving import prefilter
        initial = prefilter(cfg, searcher, pool=args.pool, keep=args.n0,
                            proxy=args.proxy)

    history, best, spent = run_halving(cfg, searcher, backend,
                                       n0=args.n0, eta=args.eta,
                                       r0=args.r0, rmax=args.rmax,
                                       use_cache=not args.no_cache,
                                       initial_configs=initial)
    write_report(history, cfg, f"halving({args.searcher})")
    print(f"\nTerbaik: [{best['score']:.4f}] {config_label(best['config'])}")


def cmd_proxycheck(args):
    """Validasi zero-cost proxy terhadap akurasi hasil training kita sendiri
    (baris cache yang punya metrics = hasil backend vision sungguhan)."""
    import json
    from pathlib import Path

    from racik.proxies import ProxyScorer, spearman

    cfg = load_sweep(args.sweep)
    rows, seen = [], set()
    for f in Path(".racik_cache").glob("*.json"):
        row = json.loads(f.read_text(encoding="utf-8"))
        if "metrics" not in row or row.get("fidelity") not in (None, 1):
            continue  # hanya hasil training nyata pada fidelity setara
        key = json.dumps(row["config"], sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    if len(rows) < 5:
        print(f"Baru {len(rows)} hasil training di cache — terlalu sedikit "
              "untuk korelasi yang berarti. Jalankan search/halving dulu.")
        return

    print(f"{len(rows)} konfigurasi terlatih ditemukan di cache. "
          "Menghitung skor proxy...")
    accs = [r["score"] for r in rows]
    for proxy in ("naswot", "synflow"):
        scorer = ProxyScorer(cfg, proxy=proxy)
        scores = [scorer.score(r["config"]) for r in rows]
        rho = spearman(scores, accs)
        print(f"\n== {proxy}: Spearman rho vs akurasi = {rho:+.3f} ==")
        ranked = sorted(zip(scores, accs, rows), key=lambda t: t[0], reverse=True)
        for s, a, r in ranked:
            print(f"  proxy={s:>9.3f}  acc={a:.4f}  {config_label(r['config'])}")


def cmd_bench(args):
    cfg = load_sweep(args.sweep)
    space = Space(cfg["space"])
    names = [n.strip() for n in args.searchers.split(",")]
    print(f"Benchmark {names} | backend={cfg['backend']} | "
          f"budget={args.budget} | seeds={args.seeds}")

    backend = make_backend(cfg)
    curves = {}
    for name in names:
        curves[name] = []
        for seed in range(args.seeds):
            print(f"\n== {name} (seed {seed}) ==")
            searcher = make_searcher(name, space, seed=seed)
            _, curve = run_search(cfg, searcher, backend, args.budget,
                                  use_cache=not args.no_cache,
                                  verbose=args.verbose)
            curves[name].append(curve)
            print(f"  selesai — terbaik: {curve[-1]:.4f}")

    winner, finals = write_bench(curves, cfg, args.budget)
    print("\nSkor akhir rata-rata per algoritme:")
    for n, v in sorted(finals.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {n:<12} {v:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", help="jalankan satu algoritme pencari")
    s.add_argument("sweep", nargs="?", default="sweep.yaml")
    s.add_argument("--searcher", default="tpe", choices=sorted(all_searchers()))
    s.add_argument("--budget", type=int, default=20)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--no-cache", action="store_true")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=cmd_search)

    h = sub.add_parser("halving", help="successive halving: budget bertingkat")
    h.add_argument("sweep", nargs="?", default="sweep.yaml")
    h.add_argument("--searcher", default="random", choices=sorted(all_searchers()))
    h.add_argument("--n0", type=int, default=9, help="jumlah kandidat awal")
    h.add_argument("--eta", type=int, default=3, help="faktor pemangkasan")
    h.add_argument("--r0", type=int, default=1, help="epoch rung pertama")
    h.add_argument("--rmax", type=int, default=None, help="epoch maksimum")
    h.add_argument("--seed", type=int, default=0)
    h.add_argument("--no-cache", action="store_true")
    h.add_argument("--pool", type=int, default=None,
                   help="prefilter: nilai POOL kandidat dengan zero-cost proxy "
                        "(tanpa training), loloskan n0 terbaik")
    h.add_argument("--proxy", default="naswot", choices=["naswot", "synflow"])
    h.set_defaults(fn=cmd_halving)

    p = sub.add_parser("proxycheck",
                       help="validasi zero-cost proxy vs akurasi cache kita")
    p.add_argument("sweep", nargs="?", default="sweep_kaggle.yaml")
    p.set_defaults(fn=cmd_proxycheck)

    b = sub.add_parser("bench", help="bandingkan beberapa algoritme, budget sama")
    b.add_argument("sweep", nargs="?", default="sweep_dummy.yaml")
    b.add_argument("--searchers", default="random,evolution,tpe")
    b.add_argument("--budget", type=int, default=30)
    b.add_argument("--seeds", type=int, default=5)
    b.add_argument("--no-cache", action="store_true")
    b.add_argument("--verbose", action="store_true")
    b.set_defaults(fn=cmd_bench)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
