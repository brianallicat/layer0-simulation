"""Honest scoring protocol used by the evolved-rule track.

Same definitions as the item-1 protocol (eval_v2), implemented locally because eval_v2.py was not
available/importable when this track was written:

  * independent reference: a body is NEVER scored against itself. The growth/regeneration target
    is the hand-written (wild-type) genome grown from reference seed 0 (REF_SEED) in a separate
    run. Scored bodies always use other seeds; REF_SEED is excluded from training and held-out sets.
  * undamaged-twin floor: after the cut time the undamaged twin (same body, same post-cut RNG,
    no cut) is scored against the same independent reference; floor = mean of its last 50 steps.
  * sustained recovery: a cut body counts as recovered only if, over the last 50 steps,
    mean >= floor - 0.03 AND at least 40 of the 50 steps are >= floor - 0.03.
  * steps to sustained recovery: first step k that is itself >= threshold and from which the rest of
    the run is sustained (mean of steps k..end >= threshold and >= 80 % of those steps >= threshold);
    None if not recovered. (Different from the original 'first crossing' number.)
  * tumour: cells outside the 2-dilated outline of the independent reference body > 5 % of the
    reference body's cell count, at a fixed copy-error rate (moderate: mu = 1e-2 per byte per copy).
  * anti-exploit guards: body cell count within [0.6, 1.5] x reference, every one of the six tissues
    present (>= MIN_TISSUE cells).
A self-referenced number (cut body vs its own pre-cut pattern, first threshold crossing - the
ORIGINAL protocol) is also returned for context only; it never enters the fitness.
"""
from __future__ import annotations

import copy
import zlib

import numpy as np

import genome as gn
from experiments import damage_masks
from metrics import blueprint_bytes, overgrowth, similarity
from sim import Config, World

try:   # reuse the item-1 protocol's sustained rule when the other worker's module is importable
    from eval_v2 import sustained as _v2_sustained  # noqa: E402
    V2_AVAILABLE = True
except Exception:  # pragma: no cover - eval_v2 is developed concurrently
    _v2_sustained, V2_AVAILABLE = None, False

REF_SEED = 0
SUSTAIN_WIN, SUSTAIN_MIN, SUSTAIN_TOL = 50, 40, 0.03
TUMOR_FRAC = 0.05
CORE = "core excision (incl. organizer C)"
WEDGE = "posterior wedge (incl. organizer P)"
ANT = "anterior amputation (incl. organizer A)"
LARGE_ANT = "large anterior amputation (incl. A)"


def ref_body(lam, steps, margin=10, turnover=0.004):
    """The independent reference: wild-type genome, REF_SEED, separate run."""
    return World(Config(seed=REF_SEED, lam=lam, margin=margin, turnover=turnover)).run(steps).pattern()


def sustained(curve, floor):
    """Sustained-recovery verdict + steps to sustained recovery (1-indexed)."""
    c = np.asarray(curve, float)
    thr = floor - SUSTAIN_TOL
    last = c[-SUSTAIN_WIN:]
    ok = bool(last.mean() >= thr and (last >= thr).sum() >= SUSTAIN_MIN)
    steps = None
    if ok:
        above = (c >= thr).astype(float)
        n = len(c)
        suf_sum = np.cumsum(c[::-1])[::-1]
        suf_above = np.cumsum(above[::-1])[::-1]
        cnt = np.arange(n, 0, -1)
        good = (c >= thr) & (suf_sum / cnt >= thr) & (suf_above / cnt >= 0.8)
        good[n - SUSTAIN_WIN + 1:] = False       # need at least a full window after k
        idx = np.nonzero(good)[0]
        steps = int(idx[0] + 1) if len(idx) else n - SUSTAIN_WIN + 1
    return dict(sustained=ok, last50_mean=float(last.mean()), threshold=float(thr),
                n_above_last50=int((last >= thr).sum()), steps_to_sustained=steps)


def guards(p, ref, min_tissue):
    n, nref = int((p > 0).sum()), int((ref > 0).sum())
    counts = {gn.TISSUE_NAMES[i]: int((p == i).sum()) for i in range(1, 7)}
    size_ok = 0.6 * nref <= n <= 1.5 * nref
    tissues_ok = all(v >= min_tissue for v in counts.values())
    return dict(cells=n, ref_cells=nref, size_ok=bool(size_ok), tissues_ok=bool(tissues_ok),
                tissue_counts=counts)


def evaluate_body(cand, seed, lam, grow_steps, regen_steps, cuts, ref, mut_cfg=None, mut_ref=None,
                  min_tissue=10, keep_pattern=False):
    """Full honest evaluation of one candidate on one body seed. Returns a JSON-able dict."""
    assert seed != REF_SEED, "scored bodies must never use the reference seed (no self-reference)"
    cfg = Config(seed=seed, lam=lam, genome=cand.genome, neighbor_clear=cand.neighbor_clear)
    base = World(cfg).run(grow_steps)
    P0 = base.pattern()
    out = dict(seed=seed, growth_similarity=similarity(P0, ref), **guards(P0, ref, min_tissue))
    masks = damage_masks(base) if base.alive.any() else {}
    runs = {}
    for name in ["undamaged twin", *cuts]:
        w = copy.deepcopy(base)
        w.rng = np.random.default_rng(12345)
        if name != "undamaged twin":
            m = masks.get(name)
            if m is None:
                runs[name] = None
                continue
            w.kill(m)
        cv_ref, cv_self = [], []
        for _ in range(regen_steps):
            w.step()
            p = w.pattern()
            cv_ref.append(similarity(p, ref))      # honest: independent reference
            cv_self.append(similarity(p, P0))      # context only (original, self-referenced)
        runs[name] = (cv_ref, cv_self, int(w.alive.sum()))
    twin_ref, twin_self, _ = runs["undamaged twin"]
    floor = float(np.mean(twin_ref[-SUSTAIN_WIN:]))
    floor_self = float(np.mean(twin_self[-SUSTAIN_WIN:]))
    out["twin_floor_vs_ref"] = floor
    out["twin_floor_self_context"] = floor_self
    out["cuts"] = {}
    for name in cuts:
        if runs[name] is None:
            out["cuts"][name] = dict(sustained=False, last50_mean=0.0, threshold=floor - SUSTAIN_TOL,
                                     n_above_last50=0, steps_to_sustained=None, error="no body")
            continue
        cv_ref, cv_self, cells = runs[name]
        s = sustained(cv_ref, floor)
        thr_self = floor_self - SUSTAIN_TOL
        s["context_self_last50"] = float(np.mean(cv_self[-SUSTAIN_WIN:]))
        s["context_self_first_crossing"] = next((i + 1 for i, v in enumerate(cv_self) if v >= thr_self), None)
        s["cells_final"] = cells
        if V2_AVAILABLE:
            v2h = _v2_sustained(cv_ref, floor)
            assert v2h["recovered_sustained"] == s["sustained"], "verdict differs from eval_v2.sustained"
            v2 = _v2_sustained(cv_self, floor_self)          # eval_v2 convention (vs pre-cut body)
            s["v2_convention_vs_precut_sustained"] = v2["recovered_sustained"]
            s["v2_convention_vs_precut_settled_step"] = v2["settled_step"]
            s["v2_honest_settled_step"] = v2h["settled_step"]
        out["cuts"][name] = s
    if mut_cfg is not None:
        mc = dict(mut_cfg)
        msteps = mc.pop("steps")
        w = World(Config(seed=seed, genome=cand.genome, neighbor_clear=cand.neighbor_clear, **mc))
        w.run(msteps)
        p = w.pattern()
        og = overgrowth(p, mut_ref)
        nref = int((mut_ref > 0).sum())
        out["tumour"] = dict(mu=mut_cfg["mu"], overgrowth_cells=int(og), tumour=bool(og > TUMOR_FRAC * nref),
                             cells=int(w.alive.sum()), mutant_frac=float(w.mutant_fraction()),
                             extinct=bool(w.alive.sum() < 0.2 * nref))
    if P0.any():
        bp = blueprint_bytes(P0)
        out["blueprint_zlib_bytes"] = bp["zlib"]
        out["blueprint_raw_bytes"] = bp["raw"]
    out["genome_bytes"] = len(cand.genome)
    out["genome_zlib_bytes"] = len(zlib.compress(cand.genome, 9))
    if keep_pattern:
        out["_pattern"] = P0
    return out


def fitness_from_bodies(rows, regen_steps, weights=None, growth_floor=None):
    """Scalar fitness from honest per-seed evaluations (higher is better).
    F = growth_sim + 0.5*pass_rate + 0.5*mean(clipped margin) - 0.1*mean(step fraction)
        - 0.5*tumour_rate - 1.0*guard_violation_rate - 5*max(0, growth_floor - growth_sim)
    growth_floor (fidelity guard): the regeneration criterion is relative to the candidate's OWN
    undamaged twin, so a body that matches the designed plan less precisely can pass it as easily.
    An unguarded first search traded ~0.08 growth similarity for regeneration passes; the guard
    stops that trade (set growth_floor=None for the unguarded score)."""
    w = dict(pass_=0.5, margin=0.5, steps=0.1, tumour=0.5, guard=1.0, fidelity=5.0)
    if weights:
        w.update(weights)
    g = np.mean([r["growth_similarity"] for r in rows])
    passes, margins, stepf = [], [], []
    for r in rows:
        for name, s in r["cuts"].items():
            passes.append(float(s["sustained"]))
            margins.append(float(np.clip(s["last50_mean"] - s["threshold"], -0.3, 0.05)))
            stepf.append(s["steps_to_sustained"] / regen_steps if s["sustained"] else 1.0)
    tum = np.mean([r["tumour"]["tumour"] for r in rows if "tumour" in r]) if any("tumour" in r for r in rows) else 0.0
    guard = np.mean([not (r["size_ok"] and r["tissues_ok"]) for r in rows])
    F = (g + w["pass_"] * np.mean(passes) + w["margin"] * np.mean(margins) - w["steps"] * np.mean(stepf)
         - w["tumour"] * tum - w["guard"] * guard)
    fid = 0.0 if growth_floor is None else max(0.0, growth_floor - g)
    F -= w["fidelity"] * fid
    return float(F), dict(fidelity_shortfall=float(fid), growth=float(g), pass_rate=float(np.mean(passes)), margin=float(np.mean(margins)),
                          step_frac=float(np.mean(stepf)), tumour_rate=float(tum), guard_violation=float(guard))
