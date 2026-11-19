"""Rescore v4: ONE common held-out seed set, ALL cuts, BOTH honest references.

Addition only: genome.py, sim.py, experiments.py, metrics.py, eval_v2.py, evolve_track/, ext_v3.py,
eval_ext_v3.py and every existing results file are untouched (eval_v2 is imported read-only).

Seeds: union of the eval_v2 held-out set (1-19, 42, 10000) and the evolve-track held-out set
(4, 6, 7, 42, 10000, 13, 21, 77, 314, 2026) = 25 bodies; tuned seed 0 is run separately and also
serves as the seed-0 reference body. Default simulator (Config defaults: positional memory,
organizer re-derivation, proofreading, redundancy ON), hand-written genome, lambda=7, grown 200 steps,
400 steps after each cut, twin rng 12345 (identical to eval_v2 and evolve_track).

Cuts: the eval_v2 damage catalog (`eval_v2.catalog_masks`), which contains the four structured
cuts (anterior, large anterior, core excision, posterior wedge) plus 5 angled/depth amputations,
3 two-organizer losses and 4 random discs = 16 cuts per body.

Every step of every run is scored against TWO references:
  * "own"  : the body's own pre-cut pattern P0; floor = its undamaged twin vs P0 (eval_v2 convention)
  * "ref0" : the seed-0 hand-written body (grown 200 steps, separate run); floor = undamaged twin vs
             ref0 (evolve_track convention). Not defined for seed 0 itself (self-reference).
Verdict = eval_v2.sustained (last-50 mean >= floor-0.03 AND >= 40/50 steps >= floor-0.03).

Run:  python3 rescore_v4.py [workers]   -> results_rescore_v4.json, results_rescore_v4_curves.npz
Then: python3 rescore_v4_report.py       -> RESCORE_V4.md
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import genome as gn
from eval_v2 import (CONDS, TWIN_RNG, WINDOW, boot_stats, catalog_masks, sustained, wilson, _j)
from experiments import GROW_STEPS, _orgs
from metrics import class_iou, similarity
from sim import Config, World

HERE = os.path.dirname(os.path.abspath(__file__))
LAM = 7.0
REGEN_STEPS = 400
REF_SEED = 0
EVAL_V2_SEEDS = list(range(1, 20)) + [42, 10000]
EVOLVE_SEEDS = [4, 6, 7, 42, 10000, 13, 21, 77, 314, 2026]
PAPER_GROWTH_SEEDS = list(range(1, 11))
HELDOUT = sorted(set(EVAL_V2_SEEDS) | set(EVOLVE_SEEDS))
ALL_SEEDS = [REF_SEED] + HELDOUT
STRUCT = ["anterior amputation (incl. organizer A)", "large anterior amputation (incl. A)",
          "core excision (incl. organizer C)", "posterior wedge (incl. organizer P)"]


def _grow(seed):
    return seed, World(Config(seed=seed, lam=LAM)).run(GROW_STEPS)


def _job(args):
    base, seed, name, cat, mask, ref0, cond = args
    w = copy.deepcopy(base)
    w.rng = np.random.default_rng(TWIN_RNG)
    for k, v in CONDS[cond].items():
        setattr(w.cfg, k, v)
    P0 = base.pattern()
    lost = sorted(k for k, v in _orgs(base).items() if any(mask[p] for p in v))
    if mask.any():
        w.kill(mask)
    c_own, c_ref = [], []
    for _ in range(REGEN_STEPS):
        w.step()
        p = w.pattern()
        c_own.append(similarity(p, P0))
        c_ref.append(similarity(p, ref0))
    p = w.pattern()
    return dict(seed=seed, name=name, category=cat, cond=cond,
                frac_removed=float(mask.sum() / base.alive.sum()), organizers_lost=lost,
                cells_final=int(w.alive.sum()), orgs_after=_orgs(w),
                tissue_iou_vs_own_final={gn.TISSUE_NAMES[t]: v for t, v in class_iou(p, P0).items()},
                tissue_iou_vs_ref0_final={gn.TISSUE_NAMES[t]: v for t, v in class_iou(p, ref0).items()},
                curve_own=c_own, curve_ref0=c_ref, final_pattern=p)


def decompose(final, P0, ref0):
    """Where do the cut body's errors sit? Union-of-bodies cell accounting at the final step."""
    u_own = (final > 0) | (P0 > 0)
    u3 = u_own | (ref0 > 0)
    miss_own = u3 & (final != P0)          # cells where regrown body != its own pre-cut body
    pre_diff = u3 & (P0 != ref0)            # cells where the pre-cut body already differed from ref0
    lands_ref = miss_own & (final == ref0)  # changed cells that now agree with ref0
    return dict(cells_changed_vs_own=int(miss_own.sum()),
                changed_where_precut_already_differed_from_ref0=int((miss_own & pre_diff).sum()),
                changed_cells_now_matching_ref0=int(lands_ref.sum()),
                new_errors_vs_ref0=int((miss_own & ~pre_diff).sum()))


def main(workers=6):
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        bases = dict(ex.map(_grow, ALL_SEEDS, chunksize=1))
    ref0 = bases[REF_SEED].pattern()
    pats = {s: b.pattern() for s, b in bases.items()}
    # ---------------- growth reconciliation (final step t=GROW_STEPS) ----------------
    vs0 = {s: similarity(pats[s], ref0) for s in HELDOUT}
    def sub(seeds):
        return dict(seeds=seeds, **boot_stats([vs0[s] for s in seeds]))
    eval_v2_bodies = [REF_SEED] + EVAL_V2_SEEDS
    pairs = {a: float(np.mean([similarity(pats[a], pats[b]) for b in eval_v2_bodies if b != a]))
             for a in eval_v2_bodies}
    pairs_all = {a: float(np.mean([similarity(pats[a], pats[b]) for b in ALL_SEEDS if b != a]))
                 for a in ALL_SEEDS}
    growth = dict(label=f"plain similarity at final growth step t={GROW_STEPS}, lambda={LAM}",
                  vs_seed0_by_seed={str(k): v for k, v in vs0.items()},
                  paper_seeds_1_10_vs_seed0=sub(PAPER_GROWTH_SEEDS),
                  eval_v2_heldout_vs_seed0=sub(EVAL_V2_SEEDS),
                  evolve_heldout_vs_seed0=sub(EVOLVE_SEEDS),
                  union_heldout_vs_seed0=sub(HELDOUT),
                  eval_v2_all_pairs_excl_self_22_bodies=boot_stats(pairs.values()),
                  union_all_pairs_excl_self_26_bodies=boot_stats(pairs_all.values()),
                  cells={str(s): int((p > 0).sum()) for s, p in pats.items()})
    # ---------------- damage battery ----------------
    jobs = []
    for s in ALL_SEEDS:
        b = bases[s]
        jobs.append((b, s, "undamaged twin", "twin", np.zeros_like(b.alive), ref0, "ON"))
        for name, cat, m in catalog_masks(b, s):
            jobs.append((b, s, name, cat, m, ref0, "ON"))
    print(f"{len(jobs)} runs on {workers} workers ...", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        raw = list(ex.map(_job, jobs, chunksize=1))
    print(f"  runs done {time.time()-t0:.0f}s", flush=True)
    twins = {r["seed"]: r for r in raw if r["name"] == "undamaged twin"}
    floors = {s: dict(own=float(np.mean(t["curve_own"][-WINDOW:])),
                      ref0=float(np.mean(t["curve_ref0"][-WINDOW:]))) for s, t in twins.items()}
    rows = []
    for r in raw:
        if r["name"] == "undamaged twin":
            continue
        s = r["seed"]
        row = {k: v for k, v in r.items() if k not in ("curve_own", "curve_ref0", "final_pattern")}
        row["own"] = dict(floor=floors[s]["own"], **sustained(r["curve_own"], floors[s]["own"]))
        if s != REF_SEED:
            row["ref0"] = dict(floor=floors[s]["ref0"], **sustained(r["curve_ref0"], floors[s]["ref0"]))
            row["decomposition_final"] = decompose(r["final_pattern"], pats[s], ref0)
        else:
            row["ref0"] = None   # seed 0 vs seed-0 reference would be self-reference
        rows.append(row)
    # pass tables
    def table(seeds, key):
        out = {}
        groups = [(n, lambda r, n=n: r["name"] == n) for n in STRUCT]
        groups += [(c, lambda r, c=c: r["category"] == c) for c in sorted({r["category"] for r in rows})]
        groups += [("ALL 16 cuts", lambda r: True)]
        for g, fn in groups:
            rr = [r for r in rows if r["seed"] in seeds and fn(r) and r[key] is not None]
            k = sum(r[key]["recovered_sustained"] for r in rr)
            out[g] = dict(n=len(rr), k=k, rate=k / len(rr) if rr else None, wilson95=wilson(k, len(rr)),
                          failing=[r["seed"] for r in rr if not r[key]["recovered_sustained"]]
                          if g in STRUCT else None)
        return out
    sets = dict(union_25=HELDOUT, eval_v2_21=EVAL_V2_SEEDS, evolve_10=EVOLVE_SEEDS)
    passes = {sn: {key: table(ss, key) for key in ("own", "ref0")} for sn, ss in sets.items()}
    seed0 = {r["name"]: r["own"] for r in rows if r["seed"] == REF_SEED}
    res = dict(protocol=dict(lam=LAM, grow_steps=GROW_STEPS, regen_steps=REGEN_STEPS, twin_rng=TWIN_RNG,
                             condition="ON (Config defaults)", genome="hand-written WILDTYPE",
                             heldout_union=HELDOUT, eval_v2_seeds=EVAL_V2_SEEDS, evolve_seeds=EVOLVE_SEEDS,
                             ref0="seed 0, hand-written, grown 200 steps",
                             rule="eval_v2.sustained: last-50 mean >= floor-0.03 AND >= 40/50 steps >= floor-0.03"),
               growth=growth, floors={str(k): v for k, v in floors.items()},
               floor_stats=dict(own=boot_stats([floors[s]["own"] for s in HELDOUT]),
                                ref0=boot_stats([floors[s]["ref0"] for s in HELDOUT])),
               rows=rows, passes=passes, tuned_seed0_own=seed0, runtime_s=time.time() - t0)
    with open(os.path.join(HERE, "results_rescore_v4.json"), "w") as f:
        json.dump(_j(res), f, indent=1)
    np.savez_compressed(os.path.join(HERE, "results_rescore_v4_curves.npz"),
                        **{f"{r['seed']}|{r['name']}|own": np.asarray(r["curve_own"], np.float32) for r in raw},
                        **{f"{r['seed']}|{r['name']}|ref0": np.asarray(r["curve_ref0"], np.float32) for r in raw},
                        **{f"pattern|{s}": p for s, p in pats.items()},
                        **{f"final|{r['seed']}|{r['name']}": r["final_pattern"] for r in raw})
    print(f"done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
