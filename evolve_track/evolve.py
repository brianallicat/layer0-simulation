"""(mu+lambda) evolutionary search over the Layer 0 rule space, scored by the honest protocol."""
from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field

import numpy as np

from . import protocol as P
from . import space as S


@dataclass
class EvoConfig:
    lam: float = 5.0                  # training body scale (smaller than the published lam=7, for budget)
    grow_steps: int = 160
    regen_steps: int = 200
    train_seeds: tuple = (1, 2)
    cuts: tuple = (P.CORE, P.WEDGE)
    mu_err: float = 1e-2              # moderate copy-error rate for the tumour test
    mut_margin: int = 12
    mut_turnover: float = 0.01
    mut_steps: int = 200
    min_tissue: int = 5
    mu: int = 4                       # parents kept
    lam_off: int = 14                 # offspring per generation
    generations: int = 10
    p_cross: float = 0.3
    init_strength: float = 2.0
    include_wildtype: bool = True
    fidelity_tol: float = 0.02        # growth floor = wild-type training growth - fidelity_tol (None = off)
    rng_seed: int = 2026
    workers: int = 6
    held_out: tuple = (4, 6, 7, 42, 10000)

    def __post_init__(self):
        bad = set(self.train_seeds) & (set(self.held_out) | {P.REF_SEED})
        assert not bad, f"training seeds overlap held-out/reference: {bad}"


def _init_worker():
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def _job(args):
    cand, seed, ec, ref, mref = args
    mc = dict(lam=ec.lam, margin=ec.mut_margin, turnover=ec.mut_turnover, mu=ec.mu_err, steps=ec.mut_steps)
    r = P.evaluate_body(cand, seed, ec.lam, ec.grow_steps, ec.regen_steps, list(ec.cuts), ref, mc, mref,
                        min_tissue=ec.min_tissue)
    return cand.key(), seed, r


def _pmap(fn, jobs, workers):
    if workers <= 1:
        return [fn(j) for j in jobs]
    with ProcessPoolExecutor(workers, initializer=_init_worker) as ex:
        return list(ex.map(fn, jobs))


def evaluate_population(cands, ec, ref, mref, cache, growth_floor=None):
    todo = [c for c in dict.fromkeys(cands) if c.key() not in cache]
    jobs = [(c, s, ec, ref, mref) for c in todo for s in ec.train_seeds]
    by = {}
    for key, seed, r in _pmap(_job, jobs, ec.workers):
        by.setdefault(key, []).append(r)
    for c in todo:
        rows = sorted(by[c.key()], key=lambda r: r["seed"])
        F, parts = P.fitness_from_bodies(rows, ec.regen_steps, growth_floor=growth_floor)
        F_ung, _ = P.fitness_from_bodies(rows, ec.regen_steps)
        cache[c.key()] = dict(fitness=F, fitness_unguarded=F_ung, parts=parts, rows=rows)
    return [cache[c.key()]["fitness"] for c in cands]


def run_evolution(ec: EvoConfig, log=print):
    rng = np.random.default_rng(ec.rng_seed)
    ref = P.ref_body(ec.lam, ec.grow_steps)
    mref = P.ref_body(ec.lam, ec.mut_steps, margin=ec.mut_margin, turnover=ec.mut_turnover)
    cache = {}
    t0 = time.time()
    # wild type is always scored first (it sets the fidelity floor); it joins the population only
    # if include_wildtype
    wt_cache = {}
    evaluate_population([S.WILDTYPE], ec, ref, mref, wt_cache)
    growth_floor = None if ec.fidelity_tol is None else wt_cache[S.WILDTYPE.key()]["parts"]["growth"] - ec.fidelity_tol
    if growth_floor is not None:
        e = wt_cache[S.WILDTYPE.key()]
        e["fitness"], e["parts"] = P.fitness_from_bodies(e["rows"], ec.regen_steps, growth_floor=growth_floor)
    pop = [S.WILDTYPE] if ec.include_wildtype else []
    if ec.include_wildtype:
        cache.update(wt_cache)
    while len(pop) < ec.lam_off:
        pop.append(S.mutate(S.WILDTYPE, rng, ec.init_strength))
    evaluate_population(pop, ec, ref, mref, cache, growth_floor)
    history = []

    def rank(cs):
        uniq = list({c.key(): c for c in cs}.values())
        return sorted(uniq, key=lambda c: -cache[c.key()]["fitness"])

    parents = rank(pop)[:ec.mu]
    wt_fit = wt_cache[S.WILDTYPE.key()]["fitness"]
    history.append(dict(gen=0, best=cache[parents[0].key()]["fitness"], wildtype=wt_fit,
                        mean_pop=float(np.mean([cache[c.key()]["fitness"] for c in pop])),
                        elapsed_s=time.time() - t0))
    log(f"gen 0 best {history[-1]['best']:.4f} wt {wt_fit} ({history[-1]['elapsed_s']:.0f}s)")
    for g in range(1, ec.generations + 1):
        kids = []
        while len(kids) < ec.lam_off:
            a = parents[rng.integers(len(parents))]
            if rng.random() < ec.p_cross and len(parents) > 1:
                b = parents[rng.integers(len(parents))]
                child = S.mutate(S.crossover(a, b, rng), rng, 0.5)
            else:
                child = S.mutate(a, rng, 1.0)
            if child.key() not in cache and child not in kids:
                kids.append(child)
        evaluate_population(kids, ec, ref, mref, cache, growth_floor)
        parents = rank(parents + kids)[:ec.mu]
        history.append(dict(gen=g, best=cache[parents[0].key()]["fitness"],
                            mean_pop=float(np.mean([cache[c.key()]["fitness"] for c in kids])),
                            best_parts=cache[parents[0].key()]["parts"], elapsed_s=time.time() - t0))
        log(f"gen {g} best {history[-1]['best']:.4f} parts {history[-1]['best_parts']} ({history[-1]['elapsed_s']:.0f}s)")
    return dict(parents=parents, cache=cache, wt=wt_cache[S.WILDTYPE.key()], growth_floor=growth_floor,
                history=history, config=asdict(ec),
                n_evaluated=len(cache), elapsed_s=time.time() - t0)
