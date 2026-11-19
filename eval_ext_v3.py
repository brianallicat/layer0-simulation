"""Evaluate the v3 extensions (ext_v3.py) with the eval_v2 protocol, ON vs OFF, same seeds/catalog.

Re-uses eval_v2's scoring code unchanged (sustained rule, bootstrap CIs, Wilson intervals, catalog
masks, no-self-reference scaling pairs). The baseline ON/OFF numbers for seeds 0-19, 42, 10000 are
read from results_eval_v2.json (the default code path is bit-identical; checked in tests and here).

Stages (python3 eval_ext_v3.py <stage> ...):  catalog | fresh | scaling | mutation | md
Outputs: results_ext_v3.json (merged per stage), EXT_V3.md
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time

import numpy as np

import eval_v2 as ev
import genome as gn
from experiments import (CONDITIONS as MUT_CONDS, GROW_STEPS, MUT_LAM, MUT_MARGIN, MUT_STEPS,
                         MUT_TURNOVER, _orgs)
from ext_v3 import ConfigV3, WorldV3
from metrics import aligned_similarity, class_iou, overgrowth, resample, similarity

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_ext_v3.json")
WORKERS = int(os.environ.get("EXT_WORKERS", "7"))

EXT = {
    "EXT2 memory": dict(mem_v3=True),
    "EXT3 scaling": dict(scale_field=True),
    "EXT3 scaling+orgs": dict(scale_field=True, scale_orgs=True),
    "EXT2+3": dict(mem_v3=True, scale_field=True),
}
BASE_CONDS = {"ON": dict(), "OFF": dict(organizer_regen=False, pos_memory=False)}
FRESH_SEEDS = list(range(20, 30))      # never looked at while developing the extensions


def _grow(args):
    seed, flags, steps = args
    return seed, WorldV3(ConfigV3(seed=seed, lam=ev.LAM, **flags)).run(steps)


def _cut(args):
    base, seed, name, cat, mask, cond, setattrs, steps = args
    w = copy.deepcopy(base)
    w.rng = np.random.default_rng(ev.TWIN_RNG)
    for k, v in setattrs.items():
        setattr(w.cfg, k, v)
    P0 = base.pattern()
    lost = sorted(k for k, v in _orgs(base).items() if any(mask[p] for p in v))
    if mask.any():
        w.kill(mask)
    curve = []
    for _ in range(steps):
        w.step()
        curve.append(similarity(w.pattern(), P0))
    p = w.pattern()
    return dict(seed=seed, name=name, category=cat, cond=cond,
                frac_removed=float(mask.sum() / base.alive.sum()), organizers_lost=lost,
                aligned_pm2_final_step=aligned_similarity(p, P0),
                tissue_iou_final_step={gn.TISSUE_NAMES[t]: v for t, v in class_iou(p, P0).items()},
                orgs_after=_orgs(w), curve=curve)


def _jobs(bases, cond, setattrs, steps=ev.REGEN_STEPS, twin=True):
    jobs = []
    for seed, base in bases.items():
        if twin:
            jobs.append((base, seed, "undamaged twin", "twin", np.zeros_like(base.alive), cond, setattrs, steps))
        for name, cat, m in ev.catalog_masks(base, seed):
            jobs.append((base, seed, name, cat, m, cond, setattrs, steps))
    return jobs


def _score(raw, floors):
    rows = []
    for r in raw:
        if r["name"] == "undamaged twin":
            continue
        s = ev.sustained(r["curve"], floors[r["seed"]])
        rows.append(dict({k: v for k, v in r.items() if k != "curve"}, twin_floor_last50_mean=floors[r["seed"]], **s))
    return rows


def _twin_floors(raw, cond):
    return {r["seed"]: float(np.mean(r["curve"][-ev.WINDOW:])) for r in raw
            if r["name"] == "undamaged twin" and r["cond"] == cond}


def stage_catalog(conds=("EXT2 memory",)):
    """Tuned seed 0 + held-out seeds, extension ON, scored against eval_v2's ON twin floors."""
    v2 = json.load(open(os.path.join(HERE, "results_eval_v2.json")))
    v2_floors = {int(k): v for k, v in v2["twin_floors_last50_mean"].items()}
    seeds = [ev.TUNED_SEED] + ev.HELDOUT_SEEDS
    out = {}
    default_bases = None
    for cond in conds:
        t0 = time.time()
        bases = dict(ev._pmap(_grow, [(s, EXT[cond], GROW_STEPS) for s in seeds], WORKERS))
        if default_bases is None:
            default_bases = dict(ev._pmap(_grow, [(s, {}, GROW_STEPS) for s in seeds], WORKERS))
        same = {s: bool(np.array_equal(bases[s].pattern(), default_bases[s].pattern())) for s in seeds}
        raw = ev._pmap(_cut, _jobs(bases, cond, {}), WORKERS)
        ext_floors = _twin_floors(raw, cond)
        # if growth changed (item 3), the v2 floors no longer apply; use the extension's own twins
        floors = v2_floors if all(same.values()) else ext_floors
        rows = _score(raw, floors)
        out[cond] = dict(rows=rows, body_identical_to_default=same,
                         floors_used="eval_v2 ON twin floors" if all(same.values()) else "own twin floors (body changed)",
                         ext_twin_floors={str(k): v for k, v in ext_floors.items()},
                         pass_rates_heldout=ev.pass_table(rows, ev.HELDOUT_SEEDS),
                         pass_rates_tuned_seed0=ev.pass_table(rows, [ev.TUNED_SEED]),
                         runtime_s=time.time() - t0)
        print(cond, f"{time.time()-t0:.0f}s", flush=True)
    return out


def stage_fresh(ext_cond="EXT2 memory"):
    """Fresh seeds 20-29 (not used during development): current system ON vs extension."""
    t0 = time.time()
    seeds = FRESH_SEEDS
    bases = dict(ev._pmap(_grow, [(s, {}, GROW_STEPS) for s in seeds], WORKERS))
    jobs = _jobs(bases, "ON", {}) + _jobs(bases, "OFF", BASE_CONDS["OFF"], twin=False)
    # the extension must be ON while the body grows (its memories are laid down during growth)
    ext_bases = dict(ev._pmap(_grow, [(s, EXT[ext_cond], GROW_STEPS) for s in seeds], WORKERS))
    assert all(np.array_equal(ext_bases[s].pattern(), bases[s].pattern()) for s in seeds)
    jobs += _jobs(ext_bases, ext_cond, {}, twin=True)
    raw = ev._pmap(_cut, jobs, WORKERS)
    floors = _twin_floors(raw, "ON")
    rows = _score(raw, floors)
    return dict(seeds=seeds, ext_cond=ext_cond, rows=rows, twin_floors_ON={str(k): v for k, v in floors.items()},
                ext_twin_floors={str(k): v for k, v in _twin_floors(raw, ext_cond).items()},
                pass_rates=ev.pass_table(rows, seeds), runtime_s=time.time() - t0)


def _scale_job(args):
    lam, seed, steps, twin_steps, flags = args
    w = WorldV3(ConfigV3(lam=lam, seed=seed, **flags)).run(steps)
    p = w.pattern()
    pr = resample(p, ev.RS)
    s_end = w.s
    w.rng = np.random.default_rng(ev.TWIN_RNG)
    nat, rsm = [], []
    for _ in range(twin_steps):
        w.step()
        q = w.pattern()
        nat.append(similarity(q, p))
        rsm.append(similarity(resample(q, ev.RS), pr))
    return lam, seed, p, nat, rsm, s_end


def scaling_eval(flags, lams=ev.SCALE_LAMS, seeds=ev.SCALE_SEEDS, steps=ev.SCALE_STEPS,
                 twin_steps=ev.REGEN_STEPS, workers=WORKERS):
    """Same analysis as eval_v2.scaling_eval, with the extension flags applied to every body."""
    out = ev._pmap(_scale_job, [(l, s, steps, twin_steps, flags) for l in lams for s in seeds], workers)
    P = {(l, s): p for l, s, p, *_ in out}
    R = {k: resample(v, ev.RS) for k, v in P.items()}
    twin = {(l, s): (n, r) for l, s, _, n, r, _ in out}
    s_end = {(l, s): x for l, s, *_, x in out}
    rows = []
    for lam in lams:
        row = dict(lam=lam, body_cells=ev.boot_stats([int((P[(lam, s)] > 0).sum()) for s in seeds]),
                   expander_s_final={str(s): s_end[(lam, s)] for s in seeds})
        v = {a[1]: similarity(R[a], R[b]) for a, b in ev.scaling_pairs([lam], seeds, mode="heldout_ref")}
        row["shape_vs_heldout_ref_final_step"] = dict(by_seed={str(k): x for k, x in v.items()}, **ev.boot_stats(v.values()))
        rsm = {}
        for a, b in ev.scaling_pairs([lam], seeds, mode="same_size"):
            rsm.setdefault(a[1], []).append(similarity(R[a], R[b]))
        row["same_size_cross_seed_resampled64_final_step"] = ev.boot_stats([np.mean(x) for x in rsm.values()])
        row["twin_floor_resampled64_last50_mean"] = ev.boot_stats([np.mean(twin[(lam, s)][1][-ev.WINDOW:]) for s in seeds])
        row["tissue_fractions"] = {gn.TISSUE_NAMES[t]: float(np.mean([(P[(lam, s)] == t).sum() / (P[(lam, s)] > 0).sum() for s in seeds])) for t in range(1, 7)}
        rows.append(row)
    return dict(flags=flags, rows=rows)


def stage_scaling(conds=("EXT3 scaling",)):
    out = {}
    for c in conds:
        t0 = time.time()
        out[c] = scaling_eval(EXT[c])
        out[c]["runtime_s"] = time.time() - t0
        print(c, f"{time.time()-t0:.0f}s", flush=True)
    return out


def _mut_job(args):
    cond, mu, seed, ref, flags = args
    kw = MUT_CONDS[cond]
    w = WorldV3(ConfigV3(lam=MUT_LAM, seed=seed, mu=mu, margin=MUT_MARGIN, turnover=MUT_TURNOVER, **kw, **flags))
    w.run(MUT_STEPS)
    p = w.pattern()
    og = overgrowth(p, ref)
    nref = int((ref > 0).sum())
    return dict(condition=cond, mu=mu, seed=seed, overgrowth_frac=og / nref, tumor=bool(og > 0.05 * nref),
                fidelity=similarity(p, ref), extinct=bool(w.alive.sum() < 0.2 * nref))


def stage_mutation(mus=(0.0, 1e-2, 3e-2, 1e-1), seeds=range(4), conds=("all 3 layers ON", "ALL OFF")):
    t0 = time.time()
    variants = {"default": {}, "EXT2 memory": EXT["EXT2 memory"], "EXT3 scaling": EXT["EXT3 scaling"]}
    out = {}
    for vname, flags in variants.items():
        ref = WorldV3(ConfigV3(lam=MUT_LAM, seed=0, margin=MUT_MARGIN, turnover=MUT_TURNOVER, **flags)).run(MUT_STEPS).pattern()
        rows = ev._pmap(_mut_job, [(c, mu, s, ref, flags) for c in conds for mu in mus for s in seeds], WORKERS)
        tab = []
        for c in conds:
            for mu in mus:
                rr = [r for r in rows if r["condition"] == c and r["mu"] == mu]
                tab.append(dict(condition=c, mu=mu, n=len(rr), tumors=int(sum(r["tumor"] for r in rr)),
                                mean_overgrowth_frac=float(np.mean([r["overgrowth_frac"] for r in rr])),
                                mean_fidelity=float(np.mean([r["fidelity"] for r in rr])),
                                extinct=int(sum(r["extinct"] for r in rr))))
        out[vname] = dict(table=tab)
        print(vname, f"{time.time()-t0:.0f}s", flush=True)
    out["runtime_s"] = time.time() - t0
    return out


def _merge(key, val):
    d = json.load(open(OUT)) if os.path.exists(OUT) else {}
    d[key] = ev._j(val)
    with open(OUT, "w") as f:
        json.dump(d, f, indent=1)


if __name__ == "__main__":
    for st in sys.argv[1:]:
        t0 = time.time()
        if st == "catalog":
            _merge("catalog", stage_catalog())
        elif st == "catalog23":
            d = json.load(open(OUT)) if os.path.exists(OUT) else {}
            cat = d.get("catalog", {})
            cat.update(ev._j(stage_catalog(("EXT2+3",))))
            _merge("catalog", cat)
        elif st == "fresh":
            _merge("fresh", stage_fresh())
        elif st == "scaling":
            _merge("scaling", stage_scaling())
        elif st == "scaling2":
            d = json.load(open(OUT)) if os.path.exists(OUT) else {}
            sc = d.get("scaling", {})
            sc.update(ev._j(stage_scaling(("EXT2 memory", "EXT2+3"))))
            _merge("scaling", sc)
        elif st == "fresh_ext":
            # re-run only the extension rows on the fresh seeds (bug fix: bases must grow with the
            # extension ON); ON/OFF rows and ON twin floors are kept from the 'fresh' stage
            d = json.load(open(OUT))
            f = d["fresh"]
            seeds, cond = f["seeds"], f["ext_cond"]
            t1 = time.time()
            ext_bases = dict(ev._pmap(_grow, [(s, EXT[cond], GROW_STEPS) for s in seeds], WORKERS))
            def_bases = dict(ev._pmap(_grow, [(s, {}, GROW_STEPS) for s in seeds], WORKERS))
            assert all(np.array_equal(ext_bases[s].pattern(), def_bases[s].pattern()) for s in seeds)
            raw = ev._pmap(_cut, _jobs(ext_bases, cond, {}, twin=True), WORKERS)
            floors = {int(k): v for k, v in f["twin_floors_ON"].items()}
            rows = [r for r in f["rows"] if r["cond"] != cond] + _score(raw, floors)
            f["rows_buggy_ext_memory_not_grown"] = [r for r in f["rows"] if r["cond"] == cond]
            f["rows"] = rows
            f["ext_twin_floors"] = {str(k): v for k, v in _twin_floors(raw, cond).items()}
            f["pass_rates"] = ev.pass_table(rows, seeds)
            f["runtime_ext_rerun_s"] = time.time() - t1
            _merge("fresh", f)
        elif st == "mutation":
            _merge("mutation", stage_mutation())
        print(f"stage {st}: {time.time()-t0:.0f}s", flush=True)
