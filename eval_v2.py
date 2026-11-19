"""Evaluation protocol v2 for Layer 0 (scoring only; the science is untouched).

This module does NOT change genome.py, sim.py, or any rule. It re-uses World/Config and the
existing experiment helpers, and only changes HOW outcomes are scored:

  1. Honest scoring: no body is ever scored against itself; held-out body seeds; seed-wise
     mean, SD and bootstrap 95% CIs; every statistic carries an explicit label
     (last-50 mean / final step / peak / +-2 aligned). Scaling is recomputed against an
     independent reference, with a same-size noise floor per scale.
  2. Sustained recovery: "recovered" only if the last-50 mean is within TOL of the floor AND
     at least SUSTAIN_MIN of the last WINDOW steps are within TOL. First crossing and the
     settle step (first step after which it never drops below) are also reported.
     Per-tissue IoU of the final body is reported for every case.
  3. Cut catalog: existing structured cuts + angled/depth amputations + two organizers lost
     at once + random disc damage (NCA-style), across held-out body seeds, with organizer
     re-derivation + positional memory ON (current system) and OFF (controls).

Run:  python3 eval_v2.py      -> results_eval_v2.json, results_eval_v2_curves.npz, EVAL_V2.md
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
from experiments import GROW_STEPS, _orgs, damage_masks
from metrics import aligned_similarity, class_iou, resample, similarity
from sim import Config, World

HERE = os.path.dirname(os.path.abspath(__file__))
WORKERS = 8

# ---- protocol constants -------------------------------------------------------------
TOL = 0.03            # recovery tolerance below the floor (same as the report)
WINDOW = 50           # "last 50 steps"
SUSTAIN_MIN = 40      # steps of the last WINDOW that must be within TOL
REGEN_STEPS = 400     # same as regeneration_test / core_regen_seeds
TWIN_RNG = 12345      # same rng convention as experiments.py (_regen_job / _core_seed_job)
LAM = 7.0
TUNED_SEED = 0        # the seed the system was developed on (reported separately)
HELDOUT_SEEDS = list(range(1, 20)) + [42, 10000]
BOOT_N = 10000
BOOT_SEED = 0

SCALE_LAMS = (4.0, 7.0, 11.0)
SCALE_STEPS = 300     # same as scaling_test
SCALE_SEEDS = list(range(10)) + [42, 10000]
SCALE_REF = (7.0, 0)  # held-out reference body: mid scale, seed 0 (never itself scored)
RS = 64               # resample size used by the shape-similarity metric


def _pmap(fn, jobs, workers=WORKERS):
    if workers <= 1:
        return [fn(j) for j in jobs]
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(fn, jobs, chunksize=1))


# ======================================================================== statistics
def boot_stats(x, n=BOOT_N, seed=BOOT_SEED):
    """Seed-wise mean, SD (ddof=1) and percentile bootstrap 95% CI of the mean."""
    x = np.asarray(list(x), float)
    out = dict(n=int(len(x)), mean=float(x.mean()) if len(x) else None,
               sd=float(x.std(ddof=1)) if len(x) > 1 else None,
               min=float(x.min()) if len(x) else None, max=float(x.max()) if len(x) else None)
    if len(x) > 1:
        rng = np.random.default_rng(seed)
        m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
        out["ci95"] = [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
    else:
        out["ci95"] = None
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(max(0.0, c - h)), float(min(1.0, c + h))]


# ======================================================================== 2. sustained rule
def sustained(curve, floor, tol=TOL, window=WINDOW, min_in=SUSTAIN_MIN):
    """Score a recovery curve (similarity to the pre-damage body, one value per step) against
    a floor. Steps are 1-indexed. Returns all labelled quantities."""
    c = np.asarray(curve, float)
    thr = floor - tol
    ok = c >= thr
    last = c[-window:]
    n_in = int((last >= thr).sum())
    first = int(np.argmax(ok)) + 1 if ok.any() else None
    if ok[-1]:
        bad = np.nonzero(~ok)[0]
        settled = int(bad[-1]) + 2 if len(bad) else 1
    else:
        settled = None
    mean_last = float(last.mean())
    return dict(threshold=float(thr), last50_mean=mean_last, final_step=float(c[-1]),
                peak=float(c.max()), peak_step=int(c.argmax()) + 1,
                steps_within_last50=n_in, first_crossing_step=first, settled_step=settled,
                recovered_mean_only=bool(mean_last >= thr),
                recovered_sustained=bool(mean_last >= thr and n_in >= min_in))


# ======================================================================== 1. scaling pairs
def scaling_pairs(lams, seeds, ref=SCALE_REF, mode="heldout_ref"):
    """List of ((lam_a, seed_a), (lam_b, seed_b)) comparisons. Guaranteed free of self-reference.
    heldout_ref : every body except the reference seed vs the single held-out reference body.
    all_pairs   : every body vs every mid-scale body of a DIFFERENT seed.
    same_size   : every body vs every body of the same scale and a different seed."""
    out = []
    if mode == "heldout_ref":
        out = [((l, s), ref) for l in lams for s in seeds if s != ref[1]]
    elif mode == "all_pairs":
        out = [((l, s), (ref[0], t)) for l in lams for s in seeds for t in seeds if t != s]
    elif mode == "same_size":
        out = [((l, s), (l, t)) for l in lams for s in seeds for t in seeds if t != s]
    for a, b in out:
        if a == b:
            raise AssertionError(f"self-reference {a}")
    return out


def _scale_body_job(args):
    lam, seed, steps, twin_steps = args
    w = World(Config(lam=lam, seed=seed)).run(steps)
    p = w.pattern()
    pr = resample(p, RS)
    w.rng = np.random.default_rng(TWIN_RNG)          # twin: same body, different noise
    nat, rsm = [], []
    for _ in range(twin_steps):
        w.step()
        q = w.pattern()
        nat.append(similarity(q, p))
        rsm.append(similarity(resample(q, RS), pr))
    return lam, seed, p, nat, rsm


def scaling_eval(lams=SCALE_LAMS, seeds=SCALE_SEEDS, steps=SCALE_STEPS, twin_steps=REGEN_STEPS,
                 workers=WORKERS):
    seeds = list(seeds)
    out = _pmap(_scale_body_job, [(l, s, steps, twin_steps) for l in lams for s in seeds], workers)
    P = {(l, s): p for l, s, p, _, _ in out}
    R = {k: resample(v, RS) for k, v in P.items()}
    twin = {(l, s): (n, r) for l, s, _, n, r in out}

    def shp(a, b):
        return similarity(R[a], R[b])

    rows = []
    for lam in lams:
        row = dict(lam=lam, seeds=seeds,
                   body_cells_by_seed={str(s): int((P[(lam, s)] > 0).sum()) for s in seeds})
        row["body_cells"] = boot_stats([int((P[(lam, s)] > 0).sum()) for s in seeds])
        # (a) shape similarity (resampled 64x64, final step t=steps) vs the held-out reference
        pr = [p for p in scaling_pairs([lam], seeds, mode="heldout_ref")]
        v = {a[1]: shp(a, b) for a, b in pr}
        row["shape_vs_heldout_ref_final_step"] = dict(by_seed={str(k): x for k, x in v.items()},
                                                      **boot_stats(v.values()))
        # (b) shape similarity vs ALL mid-scale bodies of other seeds (per-seed mean, then stats)
        per = {}
        for a, b in scaling_pairs([lam], seeds, mode="all_pairs"):
            per.setdefault(a[1], []).append(shp(a, b))
        row["shape_vs_all_mid_scale_other_seeds_final_step"] = dict(
            by_seed={str(k): float(np.mean(x)) for k, x in per.items()},
            **boot_stats([np.mean(x) for x in per.values()]))
        # (c) same-size cross-seed floors: native similarity and resampled-64 similarity
        nat, rsm = {}, {}
        for a, b in scaling_pairs([lam], seeds, mode="same_size"):
            nat.setdefault(a[1], []).append(similarity(P[a], P[b]))
            rsm.setdefault(a[1], []).append(shp(a, b))
        row["same_size_cross_seed_native_final_step"] = boot_stats([np.mean(x) for x in nat.values()])
        row["same_size_cross_seed_resampled64_final_step"] = boot_stats([np.mean(x) for x in rsm.values()])
        # (d) twin noise floor at this scale: same body, different noise, mean of last 50 of 400
        row["twin_floor_native_last50_mean"] = boot_stats([np.mean(twin[(lam, s)][0][-WINDOW:]) for s in seeds])
        row["twin_floor_resampled64_last50_mean"] = boot_stats([np.mean(twin[(lam, s)][1][-WINDOW:]) for s in seeds])
        # (e) reproduction of the OLD (inflated) number: seeds (0,1) vs seed 0 at mid scale
        if 0 in seeds and 1 in seeds:
            old = [similarity(R[(lam, 0)], R[SCALE_REF]), similarity(R[(lam, 1)], R[SCALE_REF])]
            row["old_protocol_seeds01_vs_mid_seed0_INCLUDES_SELF_at_mid"] = float(np.mean(old))
        rows.append(row)
    return dict(protocol=dict(steps=steps, resample=RS, reference=list(SCALE_REF),
                              twin=f"same body, rng {TWIN_RNG}, {twin_steps} steps, mean of last {WINDOW}",
                              note="the reference seed is excluded from the scored set at every scale"),
                rows=rows)


# ======================================================================== 3. cut catalog
def _disc(w, cy, cx, r):
    return w.alive & (np.hypot(w.yy - cy, w.xx - cx) <= r)


def catalog_masks(w, body_seed):
    """Ordered list of (name, category, mask). Masks depend only on the body and body_seed."""
    c = w.N // 2
    R = float(np.sqrt(w.alive.sum() / np.pi))
    out = [(n, "structured (original 4)", m) for n, m in damage_masks(w).items()]
    for ang, f in ((90, 0.35), (180, 0.35), (270, 0.35), (180, 0.2), (45, 0.5)):
        t = np.radians(ang)
        proj = (w.xx - c) * np.cos(t) + (w.yy - c) * np.sin(t)
        out.append((f"amputation {ang} deg, plane at {f}R", "angled/depth amputation",
                    w.alive & (proj > f * R)))
    o = {k: v[0] for k, v in _orgs(w).items() if v}
    for a, b in (("A", "C"), ("C", "P"), ("A", "P")):
        if a in o and b in o:
            m = _disc(w, *o[a], 0.3 * R) | _disc(w, *o[b], 0.3 * R)
            out.append((f"two organizers lost: {a}+{b} (discs 0.3R)", "two organizers lost", m))
    rng = np.random.default_rng([int(body_seed), 777])
    ys, xs = np.nonzero(w.alive)
    k = 0
    for rf in (0.2, 0.35):
        for _ in range(2):
            i = rng.integers(len(ys))
            out.append((f"random disc {k} (r={rf}R at {int(ys[i])},{int(xs[i])})", "random disc (NCA-style)",
                        _disc(w, ys[i], xs[i], rf * R)))
            k += 1
    return out


CONDS = {
    "ON": dict(),                                            # current fixed system
    "OFF": dict(organizer_regen=False, pos_memory=False),    # no re-derivation, no memory
    "pre-fix (memory OFF)": dict(pos_memory=False),          # used for core excision only
}


def _grow_base(args):
    seed, lam, steps = args
    return seed, World(Config(seed=seed, lam=lam)).run(steps)


def _cut_job(args):
    base, seed, name, cat, mask, cond, steps = args
    w = copy.deepcopy(base)
    w.rng = np.random.default_rng(TWIN_RNG)
    for k, v in CONDS[cond].items():
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
    iou = class_iou(p, P0)
    return dict(seed=seed, name=name, category=cat, cond=cond,
                frac_removed=float(mask.sum() / base.alive.sum()), organizers_lost=lost,
                aligned_pm2_final_step=aligned_similarity(p, P0),
                tissue_iou_final_step={gn.TISSUE_NAMES[t]: v for t, v in iou.items()},
                orgs_after=_orgs(w), curve=curve)


def catalog_jobs(bases, steps=REGEN_STEPS, conds=("ON", "OFF"), prefix_core=True):
    jobs = []
    for seed, base in bases.items():
        jobs.append((base, seed, "undamaged twin", "twin", np.zeros_like(base.alive), "ON", steps))
        for name, cat, m in catalog_masks(base, seed):
            for cond in conds:
                jobs.append((base, seed, name, cat, m, cond, steps))
            if prefix_core and name.startswith("core excision"):
                jobs.append((base, seed, name, cat, m, "pre-fix (memory OFF)", steps))
    return jobs


def score_catalog(raw):
    floors = {r["seed"]: float(np.mean(r["curve"][-WINDOW:])) for r in raw if r["name"] == "undamaged twin"}
    rows = []
    for r in raw:
        if r["name"] == "undamaged twin":
            continue
        s = sustained(r["curve"], floors[r["seed"]])
        rows.append(dict({k: v for k, v in r.items() if k != "curve"}, twin_floor_last50_mean=floors[r["seed"]], **s))
    return floors, rows


def pass_table(rows, seeds):
    tab = []
    keys = sorted({(r["category"], r["cond"]) for r in rows})
    for cat, cond in keys + [(None, c) for c in sorted({r["cond"] for r in rows})]:
        rr = [r for r in rows if r["seed"] in seeds and r["cond"] == cond and (cat is None or r["category"] == cat)]
        if cat is None and cond == "pre-fix (memory OFF)":
            continue
        if not rr:
            continue
        k = sum(r["recovered_sustained"] for r in rr)
        km = sum(r["recovered_mean_only"] for r in rr)
        tab.append(dict(category=cat or "ALL", cond=cond, n=len(rr), passed_sustained=k,
                        pass_rate_sustained=k / len(rr), wilson95=wilson(k, len(rr)),
                        passed_mean_only=km,
                        failing=[(r["seed"], r["name"]) for r in rr if not r["recovered_sustained"]]))
    return tab


def growth_eval(bases):
    """Cross-seed reproducibility of the grown body (final step t=GROW_STEPS), no self-pairs."""
    pats = {s: b.pattern() for s, b in bases.items()}
    seeds = sorted(pats)
    per, per_al, vs0 = {}, {}, {}
    for a in seeds:
        for b in seeds:
            if a == b:
                continue
            per.setdefault(a, []).append(similarity(pats[a], pats[b]))
            per_al.setdefault(a, []).append(aligned_similarity(pats[a], pats[b]))
        if a != TUNED_SEED and TUNED_SEED in pats:
            vs0[a] = similarity(pats[a], pats[TUNED_SEED])
    return dict(label=f"similarity at final step t={GROW_STEPS}",
                all_pairs_excluding_self=boot_stats([np.mean(v) for v in per.values()]),
                all_pairs_excluding_self_aligned_pm2=boot_stats([np.mean(v) for v in per_al.values()]),
                heldout_vs_seed0=dict(by_seed={str(k): v for k, v in vs0.items()}, **boot_stats(vs0.values())),
                cells=boot_stats([int((p > 0).sum()) for p in pats.values()]))


# ======================================================================== driver
def run_all(workers=WORKERS):
    t0 = time.time()
    res = {}
    print("scaling ...", flush=True)
    res["scaling"] = scaling_eval(workers=workers)
    print(f"  {time.time()-t0:.0f}s", flush=True)
    seeds = [TUNED_SEED] + HELDOUT_SEEDS
    bases = dict(_pmap(_grow_base, [(s, LAM, GROW_STEPS) for s in seeds], workers))
    res["growth"] = growth_eval(bases)
    jobs = catalog_jobs(bases)
    print(f"catalog: {len(jobs)} runs ...", flush=True)
    raw = _pmap(_cut_job, jobs, workers)
    print(f"  {time.time()-t0:.0f}s", flush=True)
    floors, rows = score_catalog(raw)
    res["twin_floors_last50_mean"] = {str(k): v for k, v in floors.items()}
    res["twin_floor_stats_heldout"] = boot_stats([floors[s] for s in HELDOUT_SEEDS])
    res["catalog_rows"] = rows
    res["pass_rates_heldout"] = pass_table(rows, HELDOUT_SEEDS)
    res["pass_rates_tuned_seed0"] = pass_table(rows, [TUNED_SEED])
    res["protocol"] = dict(tol=TOL, window=WINDOW, sustain_min=SUSTAIN_MIN, regen_steps=REGEN_STEPS,
                           twin_rng=TWIN_RNG, lam=LAM, grow_steps=GROW_STEPS, tuned_seed=TUNED_SEED,
                           heldout_seeds=HELDOUT_SEEDS, bootstrap=f"{BOOT_N} percentile resamples over seeds, rng {BOOT_SEED}",
                           conditions={k: v for k, v in CONDS.items()},
                           recovered_rule=f"last-{WINDOW} mean >= floor-{TOL} AND >= {SUSTAIN_MIN} of last {WINDOW} steps >= floor-{TOL}")
    res["runtime_s"] = time.time() - t0
    curves = {f"{r['seed']}|{r['name']}|{r['cond']}": np.asarray(r["curve"], np.float32) for r in raw}
    return res, curves


def _j(o):
    if isinstance(o, dict):
        return {str(k): _j(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_j(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# ======================================================================== report
def _f(x, d=4):
    return "n/a" if x is None else f"{x:.{d}f}"


def _st(s, d=4):
    if s is None or s.get("mean") is None:
        return "n/a"
    ci = s.get("ci95")
    sd = "" if s.get("sd") is None else f" ± {s['sd']:.{d}f} SD"
    return f"{s['mean']:.{d}f}{sd}" + (f" [95% CI {ci[0]:.{d}f}–{ci[1]:.{d}f}]" if ci else "") + f" (n={s['n']})"


def write_md(res, path):
    P = res["protocol"]
    L = ["# Layer 0 — evaluation protocol v2 (scoring only; genome and rules unchanged)", "",
         "Generated by `python3 eval_v2.py` from `results_eval_v2.json` (per-step curves in "
         "`results_eval_v2_curves.npz`). Every statistic is labelled: **last-50 mean** = mean of the "
         "last 50 of 400 steps; **final step** = value at the last step; **peak** = max over the run; "
         "**±2 aligned** = best similarity over rigid shifts |dy|,|dx| ≤ 2, at the final step. "
         "\"±\" is the seed-wise SD; CIs are percentile bootstrap over seeds "
         f"({P['bootstrap']}).", "",
         "## Protocol", "",
         f"- Body: λ={P['lam']}, grown {P['grow_steps']} steps. Tuned seed {P['tuned_seed']} is reported "
         f"separately; held-out body seeds: {', '.join(map(str, P['heldout_seeds']))} ({len(P['heldout_seeds'])} seeds).",
         f"- Every damaged body is scored against its own pre-damage body; its floor is its own undamaged twin "
         f"(same body copy, rng {P['twin_rng']}, {P['regen_steps']} steps, last-50 mean) — the experiments.py convention.",
         f"- **Recovered (sustained)** = {P['recovered_rule']}. `mean-only` = the old rule (last-50 mean only).",
         "- Conditions: **ON** = current fixed system (organizer re-derivation + positional memory); "
         "**OFF** = organizer_regen=False and pos_memory=False; **pre-fix** = pos_memory=False only (run for core excision).",
         f"- Runtime {res['runtime_s']:.0f} s on 8 workers.", ""]
    g = res["growth"]
    L += ["## 1a. Growth reproducibility (no self-pairs; similarity at final step t=200)", "",
          f"- All pairs of the 22 bodies, excluding self (per-seed mean, then stats): {_st(g['all_pairs_excluding_self'])}",
          f"- Same, ±2 aligned: {_st(g['all_pairs_excluding_self_aligned_pm2'])}",
          f"- Held-out seeds vs seed 0: {_st(g['heldout_vs_seed0'])}",
          f"- Body size (cells): {_st(g['cells'], 1)}", ""]
    sc = res["scaling"]
    L += ["## 1b. Scaling, recomputed without self-reference", "",
          f"Bodies grown {sc['protocol']['steps']} steps at each λ, seeds {', '.join(map(str, sc['rows'][0]['seeds']))}. "
          f"Shape similarity = tissue-by-tissue similarity after resampling each body to {RS}×{RS}; reference = "
          f"λ={SCALE_REF[0]:g} seed {SCALE_REF[1]}, which is excluded from the scored set at every scale. All values are final-step values (t=300).", "",
          "| λ | cells | shape vs held-out ref (seed 0 excluded) | shape vs all mid-scale bodies of other seeds | same-size cross-seed, resampled (floor for shape) | same-size cross-seed, native | twin floor native (last-50 mean) | twin floor resampled (last-50 mean) | old protocol (inflated) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in sc["rows"]:
        L.append(f"| {r['lam']:g} | {_st(r['body_cells'], 1)} | {_st(r['shape_vs_heldout_ref_final_step'])} | "
                 f"{_st(r['shape_vs_all_mid_scale_other_seeds_final_step'])} | {_st(r['same_size_cross_seed_resampled64_final_step'])} | "
                 f"{_st(r['same_size_cross_seed_native_final_step'])} | {_st(r['twin_floor_native_last50_mean'])} | "
                 f"{_st(r['twin_floor_resampled64_last50_mean'])} | {_f(r.get('old_protocol_seeds01_vs_mid_seed0_INCLUDES_SELF_at_mid'))} |")
    mid = [r for r in sc["rows"] if r["lam"] == SCALE_REF[0]][0]
    L += ["", "Per-seed shape similarity vs the held-out reference at λ=7: " +
          ", ".join(f"s{k}={v:.4f}" for k, v in mid["shape_vs_heldout_ref_final_step"]["by_seed"].items()), ""]
    L += ["## 2. Twin floors (undamaged twin, last-50 mean)", "",
          ", ".join(f"s{k}={v:.4f}" for k, v in res["twin_floors_last50_mean"].items()), "",
          f"Held-out seeds: {_st(res['twin_floor_stats_heldout'])}", ""]
    rows = res["catalog_rows"]

    def tab(rr):
        out = ["| seed | cut | cond | frac removed | organizers lost | floor | last-50 mean | final step | peak | ±2 aligned (final) | steps within (of 50) | first crossing | settled step | mean-only | **sustained** |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rr:
            out.append(f"| {r['seed']} | {r['name']} | {r['cond']} | {r['frac_removed']:.2f} | {'+'.join(r['organizers_lost']) or '-'} | "
                       f"{r['twin_floor_last50_mean']:.4f} | {r['last50_mean']:.4f} | {r['final_step']:.4f} | {r['peak']:.4f} | "
                       f"{r['aligned_pm2_final_step']:.4f} | {r['steps_within_last50']} | {r['first_crossing_step'] or 'never'} | "
                       f"{r['settled_step'] or 'never'} | {'yes' if r['recovered_mean_only'] else 'no'} | **{'YES' if r['recovered_sustained'] else 'no'}** |")
        return out
    L += ["## 2a. Original four cuts, tuned seed 0, sustained rule", ""]
    L += tab([r for r in rows if r["seed"] == TUNED_SEED and r["category"].startswith("structured")])
    core = [r for r in rows if r["name"].startswith("core excision")]
    L += ["", "## 2b. Core excision across all body seeds", ""]
    for cond in ("ON", "pre-fix (memory OFF)", "OFF"):
        rr = [r for r in core if r["cond"] == cond and r["seed"] in HELDOUT_SEEDS]
        k = sum(r["recovered_sustained"] for r in rr)
        L.append(f"- **{cond}**, held-out seeds: {k}/{len(rr)} pass (sustained); failing seeds: "
                 f"{', '.join(str(r['seed']) for r in rr if not r['recovered_sustained']) or 'none'}")
    L += [""] + tab(sorted(core, key=lambda r: (r["cond"] != "ON", r["cond"], r["seed"])))
    wedge = [r for r in rows if r["name"].startswith("posterior wedge")]
    L += ["", "## 2c. Posterior wedge across all body seeds", ""]
    for cond in ("ON", "OFF"):
        rr = [r for r in wedge if r["cond"] == cond and r["seed"] in HELDOUT_SEEDS]
        k = sum(r["recovered_sustained"] for r in rr)
        L.append(f"- **{cond}**, held-out seeds: {k}/{len(rr)} pass (sustained); "
                 f"last-50 mean {_st(boot_stats([r['last50_mean'] for r in rr]))}; gap to floor "
                 f"{_st(boot_stats([r['twin_floor_last50_mean'] - r['last50_mean'] for r in rr]))}")
    L += [""] + tab(sorted(wedge, key=lambda r: (r["cond"], r["seed"])))
    L += ["", "## 3. Cut catalog pass rates (sustained rule)", "",
          "| set | category | cond | passed / n | pass rate | Wilson 95% | passed under old mean-only rule |", "|---|---|---|---|---|---|---|"]
    for name, key in (("held-out", "pass_rates_heldout"), ("tuned seed 0", "pass_rates_tuned_seed0")):
        for t in res[key]:
            w = t["wilson95"]
            L.append(f"| {name} | {t['category']} | {t['cond']} | {t['passed_sustained']}/{t['n']} | {100*t['pass_rate_sustained']:.1f}% | "
                     f"{100*w[0]:.1f}–{100*w[1]:.1f}% | {t['passed_mean_only']}/{t['n']} |")
    L += ["", "### Per-cut pass counts, held-out seeds (ON / OFF)", "", "| cut | category | ON pass | OFF pass | ON last-50 mean (gap to floor) | failing seeds (ON) |", "|---|---|---|---|---|---|"]
    names = []
    for r in rows:
        if r["name"] not in names and not r["name"].startswith("random disc"):
            names.append(r["name"])
    groups = [(n, lambda r, n=n: r["name"] == n) for n in names]
    for rf in ("0.2R", "0.35R"):
        groups.append((f"random discs r={rf} (2 per body)", lambda r, rf=rf: r["name"].startswith("random disc") and f"r={rf}" in r["name"]))
    for n, fn in groups:
        on = [r for r in rows if fn(r) and r["cond"] == "ON" and r["seed"] in HELDOUT_SEEDS]
        off = [r for r in rows if fn(r) and r["cond"] == "OFF" and r["seed"] in HELDOUT_SEEDS]
        gap = boot_stats([r["twin_floor_last50_mean"] - r["last50_mean"] for r in on])
        L.append(f"| {n} | {on[0]['category']} | {sum(r['recovered_sustained'] for r in on)}/{len(on)} | "
                 f"{sum(r['recovered_sustained'] for r in off)}/{len(off)} | "
                 f"{np.mean([r['last50_mean'] for r in on]):.4f} ({gap['mean']:.4f}) | "
                 f"{', '.join(sorted({str(r['seed']) for r in on if not r['recovered_sustained']}, key=int)) or 'none'} |")
    L += ["", "## Per-tissue IoU at the final step (mean over held-out seeds, ON)", "",
          "| cut | " + " | ".join(gn.TISSUE_NAMES[1:]) + " |", "|---|" + "---|" * 6]
    for n, fn in groups:
        on = [r for r in rows if fn(r) and r["cond"] == "ON" and r["seed"] in HELDOUT_SEEDS]
        L.append(f"| {n} | " + " | ".join(f"{np.mean([r['tissue_iou_final_step'][t] for r in on]):.3f}" for t in gn.TISSUE_NAMES[1:]) + " |")
    L += ["", "Per-case IoU for every seed/cut/condition is in `results_eval_v2.json` → `catalog_rows[*].tissue_iou_final_step`.", ""]
    with open(path, "w") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    if "--md-only" in sys.argv:
        with open(os.path.join(HERE, "results_eval_v2.json")) as f:
            write_md(json.load(f), os.path.join(HERE, "EVAL_V2.md"))
        sys.exit(0)
    res, curves = run_all()
    with open(os.path.join(HERE, "results_eval_v2.json"), "w") as f:
        json.dump(_j(res), f, indent=1)
    np.savez_compressed(os.path.join(HERE, "results_eval_v2_curves.npz"), **curves)
    write_md(json.loads(json.dumps(_j(res))), os.path.join(HERE, "EVAL_V2.md"))
    print(f"done in {res['runtime_s']:.0f}s")
