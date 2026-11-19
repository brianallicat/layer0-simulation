"""All Layer-0 experiments. Each returns plain dicts (JSON-serialisable) plus, where useful,
patterns for plotting. The scoring reference patterns are produced by the experimenter
from separate runs; the simulated body never has access to them."""
from __future__ import annotations

import copy
import hashlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import genome as gn
from metrics import (aligned_similarity, blueprint_bytes, class_iou, overgrowth, resample,
                     similarity)
from sim import Config, World

GROW_STEPS = 200
WORKERS = 8


def _pmap(fn, jobs, workers=WORKERS):
    if workers <= 1:
        return [fn(j) for j in jobs]
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(fn, jobs))


def _orgs(w):
    return {k: [(int(a), int(b)) for a, b in v] for k, v in w.organizer_positions().items()}


# ---------------------------------------------------------------- 1. growth / reproducibility
def _grow_job(args):
    seed, lam, steps, snaps = args
    w = World(Config(seed=seed, lam=lam))
    frames = {}
    for k in range(1, steps + 1):
        w.step()
        if k in snaps:
            frames[k] = (w.pattern(), _orgs(w))
    return seed, w.pattern(), _orgs(w), int(w.alive.sum()), frames, w.fields.copy()


def growth_test(seeds=range(1, 11), lam=7.0, steps=GROW_STEPS, workers=WORKERS):
    snaps = (5, 15, 30, 50, 80, steps)
    res = _pmap(_grow_job, [(s, lam, steps, snaps) for s in [0, *seeds]], workers)
    ref_seed, ref, ref_orgs, ref_n, ref_frames, ref_fields = res[0]
    rows = []
    for seed, p, orgs, n, _, _ in res[1:]:
        rows.append(dict(seed=seed, cells=n, similarity=similarity(p, ref),
                         aligned=aligned_similarity(p, ref),
                         mean_class_iou=float(np.mean(list(class_iou(p, ref).values()))),
                         organizers=orgs))
    sims = [r["similarity"] for r in rows]
    # noise floor: the same body against itself 50 steps later
    w = World(Config(seed=0, lam=lam)).run(steps)
    a = w.pattern()
    w.run(50)
    floor = similarity(w.pattern(), a)
    summary = dict(n_seeds=len(rows), ref_cells=ref_n, mean_similarity=float(np.mean(sims)),
                   min_similarity=float(np.min(sims)), max_similarity=float(np.max(sims)),
                   mean_aligned=float(np.mean([r["aligned"] for r in rows])),
                   self_similarity_50_steps_later=floor,
                   tissue_counts={gn.TISSUE_NAMES[i]: int((ref == i).sum()) for i in range(1, 7)})
    return dict(summary=summary, rows=rows), dict(ref=ref, ref_orgs=ref_orgs, frames=ref_frames,
                                                   fields=ref_fields)


# ---------------------------------------------------------------- 2. regeneration
def damage_masks(w):
    c = w.N // 2
    R = float(np.sqrt(w.alive.sum() / np.pi))
    d = np.hypot(w.yy - c, w.xx - c)
    ang = np.degrees(np.arctan2(w.yy - c, w.xx - c))
    return {
        "anterior amputation (incl. organizer A)": w.alive & (w.xx > c + 0.35 * R),
        "large anterior amputation (incl. A)": w.alive & (w.xx > c + 0.2 * R),
        "core excision (incl. organizer C)": w.alive & (d < 0.5 * R),
        "posterior wedge (incl. organizer P)": w.alive & (np.abs(ang) > 120),
    }


def _regen_job(args):
    base, name, regen, steps = args
    w = copy.deepcopy(base)
    w.rng = np.random.default_rng(12345)
    P0 = base.pattern()
    mask = damage_masks(base)[name] if name != "undamaged twin" else np.zeros_like(base.alive)
    lost = {k: [p for p in v if mask[p]] for k, v in _orgs(base).items()}
    w.cfg.organizer_regen = regen
    w.kill(mask)
    after_cut = w.pattern()
    curve, acurve = [], []
    for k in range(steps):
        w.step()
        p = w.pattern()
        curve.append(similarity(p, P0))
        if k % 10 == 9 or k == steps - 1:
            acurve.append(aligned_similarity(p, P0))
    return dict(name=name, regen=regen, frac_removed=float(mask.sum() / base.alive.sum()),
                organizers_destroyed=[k for k, v in lost.items() if v], curve=curve,
                aligned_final=acurve[-1], final=curve[-1], cells=int(w.alive.sum()),
                orgs_after=_orgs(w)), after_cut, w.pattern()


def regeneration_test(lam=7.0, seed=0, steps=400, workers=WORKERS):
    base = World(Config(seed=seed, lam=lam)).run(GROW_STEPS)
    names = list(damage_masks(base))
    jobs = [(base, "undamaged twin", True, steps)]
    jobs += [(base, n, True, steps) for n in names]
    jobs += [(base, n, False, steps) for n in names]   # controls: organizer re-derivation disabled
    out = _pmap(_regen_job, jobs, workers)
    twin = out[0][0]
    floor = float(np.mean(twin["curve"][-50:]))
    rows, pics = [], {}
    for (r, cut, final) in out[1:]:
        tgt = floor - 0.03
        hit = next((i + 1 for i, v in enumerate(r["curve"]) if v >= tgt), None)
        r["final_mean_last50"] = float(np.mean(r["curve"][-50:]))
        r["steps_to_recover"] = hit
        r["recovered"] = bool(r["final_mean_last50"] >= tgt)
        rows.append(r)
        pics[(r["name"], r["regen"])] = (cut, final)
    return dict(noise_floor=floor, twin_aligned=twin["aligned_final"], rows=rows), \
        dict(base=base.pattern(), base_orgs=_orgs(base), pics=pics,
             curves={(r["name"], r["regen"]): r["curve"] for r in rows}, twin_curve=twin["curve"])


def _core_seed_job(args):
    seed, steps, memory = args
    base = World(Config(seed=seed, lam=7.0)).run(GROW_STEPS)
    out = {}
    for name in ("undamaged twin", "core excision (incl. organizer C)"):
        w = copy.deepcopy(base)
        w.rng = np.random.default_rng(12345)
        w.cfg.pos_memory = memory
        P0 = base.pattern()
        if name != "undamaged twin":
            w.kill(damage_masks(base)[name])
        curve = []
        for _ in range(steps):
            w.step()
            curve.append(similarity(w.pattern(), P0))
        out[name] = dict(final=float(np.mean(curve[-50:])), orgs=_orgs(w), curve=curve)
    floor = out["undamaged twin"]["final"]
    c = out["core excision (incl. organizer C)"]
    hit = next((i + 1 for i, v in enumerate(c["curve"]) if v >= floor - 0.03), None)
    return dict(seed=seed, memory=memory, noise_floor=floor, core_final=c["final"],
                recovered=bool(c["final"] >= floor - 0.03), steps_to_recover=hit,
                C_before=_orgs(base)["C"], C_after=c["orgs"]["C"])


def core_regen_seeds(seeds=(0, 1, 2, 3), steps=400, memory=(True, False), workers=WORKERS):
    """Robustness of core regeneration across independently grown bodies, with positional
    memory ON and OFF (ablation)."""
    return _pmap(_core_seed_job, [(s, steps, m) for m in memory for s in seeds], workers)


# ---------------------------------------------------------------- 3. mutation / error correction
CONDITIONS = {   # full 2x2x2 factorial over the three layers, plus one sub-ablation
    "all 3 layers ON": dict(),
    "no proofreading": dict(proofread=False),
    "no redundancy": dict(redundancy=1),
    "no apoptosis/clearance": dict(apoptosis=False, neighbor_clear=False),
    "proofreading only": dict(redundancy=1, apoptosis=False, neighbor_clear=False),
    "redundancy only": dict(proofread=False, apoptosis=False, neighbor_clear=False),
    "apoptosis/clearance only": dict(proofread=False, redundancy=1),
    "ALL OFF": dict(proofread=False, redundancy=1, apoptosis=False, neighbor_clear=False),
    "all ON except neighbour clearance": dict(neighbor_clear=False),
}
MUT_LAM, MUT_STEPS, MUT_TURNOVER, MUT_MARGIN = 6.0, 300, 0.01, 16


def _mut_ref():
    return World(Config(lam=MUT_LAM, seed=0, margin=MUT_MARGIN, turnover=MUT_TURNOVER)).run(MUT_STEPS).pattern()


def _mut_job(args):
    cond, mu, seed, ref = args
    kw = CONDITIONS[cond]
    w = World(Config(lam=MUT_LAM, seed=seed, mu=mu, margin=MUT_MARGIN, turnover=MUT_TURNOVER, **kw))
    w.run(MUT_STEPS)
    p = w.pattern()
    og = overgrowth(p, ref)
    nref = int((ref > 0).sum())
    return dict(condition=cond, mu=mu, seed=seed, overgrowth_cells=og, overgrowth_frac=og / nref,
                tumor=bool(og > 0.05 * nref), fidelity=similarity(p, ref), cells=int(w.alive.sum()),
                mutant_frac=w.mutant_fraction(), extinct=bool(w.alive.sum() < 0.2 * nref),
                **{k: int(v) for k, v in w.stats.items()})


def mutation_test(mus=(0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1), seeds=range(4), conditions=None,
                  workers=WORKERS):
    conditions = conditions or list(CONDITIONS)
    ref = _mut_ref()
    jobs = [(c, mu, s, ref) for c in conditions for mu in mus for s in seeds]
    rows = _pmap(_mut_job, jobs, workers)
    table = []
    for c in conditions:
        for mu in mus:
            rr = [r for r in rows if r["condition"] == c and r["mu"] == mu]
            table.append(dict(condition=c, mu=mu, n=len(rr),
                              tumor_incidence=float(np.mean([r["tumor"] for r in rr])),
                              mean_overgrowth_frac=float(np.mean([r["overgrowth_frac"] for r in rr])),
                              mean_fidelity=float(np.mean([r["fidelity"] for r in rr])),
                              mean_mutant_frac=float(np.mean([r["mutant_frac"] for r in rr])),
                              extinct=int(sum(r["extinct"] for r in rr))))
    return dict(table=table, runs=rows, ref_cells=int((ref > 0).sum())), dict(ref=ref)


# ---------------------------------------------------------------- 4. scaling
def _scale_job(args):
    lam, seed, steps = args
    w = World(Config(lam=lam, seed=seed)).run(steps)
    return lam, seed, w.pattern(), _orgs(w), hashlib.sha256(w.cfg.genome).hexdigest()[:16]


def scaling_test(lams=(4.0, 7.0, 11.0), seeds=(0, 1), steps=300, workers=WORKERS):
    out = _pmap(_scale_job, [(l, s, steps) for l in lams for s in seeds], workers)
    by = {}
    for lam, seed, p, orgs, h in out:
        by.setdefault(lam, []).append((seed, p, orgs, h))
    mid = lams[len(lams) // 2]
    ref_r = resample(by[mid][0][1], 64)
    rows, pics = [], {}
    for lam in lams:
        ps = [p for _, p, _, _ in by[lam]]
        n = [int((p > 0).sum()) for p in ps]
        frac = {gn.TISSUE_NAMES[i]: float(np.mean([(p == i).sum() / (p > 0).sum() for p in ps])) for i in range(1, 7)}
        bp = blueprint_bytes(ps[0])
        rows.append(dict(lam=lam, body_cells=float(np.mean(n)),
                         body_diameter=float(2 * np.sqrt(np.mean(n) / np.pi)),
                         genome_bytes=gn.GENOME_LEN, genome_bytes_per_block_x3=gn.GENOME_LEN * 3,
                         genome_sha=by[lam][0][3], tissue_fractions=frac,
                         shape_similarity_to_mid_scale=float(np.mean([similarity(resample(p, 64), ref_r) for p in ps])),
                         blueprint_raw_bytes=bp["raw"], blueprint_zlib_bytes=bp["zlib"],
                         blueprint_1KB_per_block=bp["per_block_1KB"]))
        pics[lam] = (by[lam][0][1], by[lam][0][2])
    return dict(rows=rows, same_genome=len({r["genome_sha"] for r in rows}) == 1), dict(pics=pics)


# ---------------------------------------------------------------- 5. morph (shift a field source)
def morph_test(lam=7.0, seed=0, steps=400, shift_frac=0.3):
    base = World(Config(seed=seed, lam=lam)).run(GROW_STEPS)
    P0 = base.pattern()
    c = base.N // 2
    R = float(np.sqrt(base.alive.sum() / np.pi))
    sh = int(round(shift_frac * R))
    w = copy.deepcopy(base)
    w.org[c, c] = 0                      # graft: move the core organizer (a field source)
    w.org[c, c + sh] = 1
    w.org_born[c, c + sh] = -10 ** 6
    target = np.roll(P0, sh, 1)          # what the body *should* become: same plan, re-centred
    curve_new, curve_bp, frames = [], [], {}
    for k in range(1, steps + 1):
        w.step()
        p = w.pattern()
        curve_new.append(similarity(p, target))
        curve_bp.append(similarity(p, P0))
        if k in (40, 100, steps):
            frames[k] = (p, _orgs(w))
    return dict(shift_cells=sh, final_similarity_to_shifted_plan=curve_new[-1],
                stored_blueprint_similarity_to_shifted_plan=similarity(P0, target),
                body_similarity_to_old_blueprint=curve_bp[-1],
                steps_to_090=next((i + 1 for i, v in enumerate(curve_new) if v >= 0.9), None),
                orgs_before=_orgs(base), orgs_after=_orgs(w)), \
        dict(P0=P0, target=target, frames=frames, curve=curve_new, curve_bp=curve_bp, base_orgs=_orgs(base))


# ---------------------------------------------------------------- 6. materials / physics
def physics_test(regen_art=None, seeds=range(0, 6), workers=WORKERS):
    """Map tissues to materials and check heat, power and load/strain limits on grown bodies
    (several seeds) and on every regenerated body. Environments: still air (nominal) and vacuum."""
    import materials as M
    grown = _pmap(_grow_pattern, [(s, 7.0) for s in seeds], workers)
    rows = []
    for s, p in grown:
        for env, cfg in (("air", M.PhysConfig()), ("vacuum", M.PhysConfig(h_conv=0.0))):
            ev = M.evaluate(p, cfg)
            rows.append(dict(body=f"grown seed {s}", env=env, **M.summary(ev)))
    if regen_art is not None:
        for (name, regen), (cut, fin) in regen_art["pics"].items():
            if regen and fin.any():
                ev = M.evaluate(fin, M.PhysConfig())
                rows.append(dict(body=f"regrown: {name}", env="air", **M.summary(ev)))
    p0 = grown[0][1]
    pmax_air = M.max_core_power(p0, M.PhysConfig())
    pmax_vac = M.max_core_power(p0, M.PhysConfig(h_conv=0.0))
    # power ablation: no copper traces
    cu = M.CU
    M.set_copper(0.0)
    ev_nocu = M.evaluate(p0, M.PhysConfig())
    M.set_copper(cu)
    ev0 = M.evaluate(p0, M.PhysConfig())
    # heat-pipe ablation: channels filled with plain composite instead
    khp = M.MAT["heat_pipe"]["k"]
    M.MAT["heat_pipe"]["k"] = M.MAT["graphene_epoxy"]["k"]
    M.PROPS.update({t: M.block_props(t) for t in M.TISSUE_STACK})
    ev_nohp = M.evaluate(p0, M.PhysConfig())
    pmax_nohp = M.max_core_power(p0, M.PhysConfig())
    M.MAT["heat_pipe"]["k"] = khp
    M.PROPS.update({t: M.block_props(t) for t in M.TISSUE_STACK})
    props = {gn.TISSUE_NAMES[t]: {k: v for k, v in pr.items()} for t, pr in M.PROPS.items()}
    return dict(rows=rows, max_core_W_air=pmax_air, max_core_W_vacuum=pmax_vac,
                no_copper=dict(max_droop=ev_nocu["power"]["max_droop"], collapsed=ev_nocu["power"]["collapsed"],
                               power_ok=ev_nocu["power"]["ok"]),
                no_heat_pipes=dict(T_max_C=ev_nohp["thermal"]["Tmax"], thermal_ok=ev_nohp["thermal"]["ok"],
                                   violations=ev_nohp["thermal"]["violations"], max_core_W_air=pmax_nohp),
                block_props=props, cfg=vars(M.PhysConfig())), \
        dict(pattern=p0, T=ev0["thermal"]["T"], V=ev0["power"]["V"], ev=ev0, T_nohp=ev_nohp["thermal"]["T"],
             T_vac=M.evaluate(p0, M.PhysConfig(h_conv=0.0))["thermal"]["T"])


def _grow_pattern(args):
    s, lam = args
    return s, World(Config(seed=s, lam=lam)).run(GROW_STEPS).pattern()
