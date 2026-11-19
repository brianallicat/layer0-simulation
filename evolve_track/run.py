"""Full evolved-rule comparison run: search on training seeds (lam=5), then evaluate the
hand-written genome and the top evolved genomes on HELD-OUT body seeds at the published scale
(lam=7), with the honest protocol. Writes evolve_results.json (and the numbers used by EVOLVE_TRACK.md).

    python3 -m evolve_track.run            # full budget
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

from . import protocol as P  # noqa: E402
from . import space as S  # noqa: E402
from .evolve import EvoConfig, run_evolution  # noqa: E402
import genome as gn  # noqa: E402
from metrics import similarity  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HELD_OUT = (4, 6, 7, 42, 10000, 13, 21, 77, 314, 2026)
EVAL = dict(lam=7.0, grow_steps=200, regen_steps=400,
            cuts=[P.CORE, P.WEDGE, P.ANT, P.LARGE_ANT],
            mut=dict(lam=6.0, margin=16, turnover=0.01, mu=1e-2, steps=300))


def _eval_job(args):
    label, cand, seed, ref, mref = args
    r = P.evaluate_body(cand, seed, EVAL["lam"], EVAL["grow_steps"], EVAL["regen_steps"], EVAL["cuts"],
                        ref, EVAL["mut"], mref, min_tissue=10, keep_pattern=True)
    return label, seed, r


def t_ci(x):
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return [float(x.mean())] * 2
    tq = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31, 10: 2.26}.get(n, 2.0)
    h = tq * x.std(ddof=1) / math.sqrt(n)
    return [float(x.mean() - h), float(x.mean() + h)]


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(max(0, c - h)), float(min(1, c + h))]


def summarize(rows):
    g = [r["growth_similarity"] for r in rows]
    pats = [r["_pattern"] for r in rows]
    pair = [similarity(pats[i], pats[j]) for i in range(len(pats)) for j in range(i + 1, len(pats))]
    out = dict(n=len(rows), growth_mean=float(np.mean(g)), growth_sd=float(np.std(g, ddof=1)),
               growth_ci95=t_ci(g), growth_min=float(np.min(g)),
               reproducibility_pairwise_mean=float(np.mean(pair)),
               cells_mean=float(np.mean([r["cells"] for r in rows])),
               guard_violations=int(sum(not (r["size_ok"] and r["tissues_ok"]) for r in rows)),
               twin_floor_mean=float(np.mean([r["twin_floor_vs_ref"] for r in rows])),
               blueprint_zlib_mean=float(np.mean([r["blueprint_zlib_bytes"] for r in rows])),
               blueprint_raw_mean=float(np.mean([r["blueprint_raw_bytes"] for r in rows])),
               genome_bytes=rows[0]["genome_bytes"], genome_zlib_bytes=rows[0]["genome_zlib_bytes"])
    out["cuts"] = {}
    for c in EVAL["cuts"]:
        ss = [r["cuts"][c] for r in rows]
        k = sum(s["sustained"] for s in ss)
        st = [s["steps_to_sustained"] for s in ss if s["sustained"]]
        out["cuts"][c] = dict(pass_k=int(k), pass_rate=k / len(ss), pass_ci95=wilson(k, len(ss)),
                              steps_to_sustained_median=float(np.median(st)) if st else None,
                              steps_to_sustained_range=[int(min(st)), int(max(st))] if st else None,
                              last50_mean=float(np.mean([s["last50_mean"] for s in ss])),
                              failing_seeds=[r["seed"] for r, s in zip(rows, ss) if not s["sustained"]],
                              context_self_first_crossing_median=float(np.median(
                                  [s["context_self_first_crossing"] for s in ss if s["context_self_first_crossing"]]))
                              if any(s["context_self_first_crossing"] for s in ss) else None)
    tum = [r["tumour"] for r in rows]
    out["tumour_incidence"] = float(np.mean([t["tumour"] for t in tum]))
    out["tumour_k"] = int(sum(t["tumour"] for t in tum))
    out["mutant_frac_mean"] = float(np.mean([t["mutant_frac"] for t in tum]))
    out["extinct"] = int(sum(t["extinct"] for t in tum))
    return out


def main(ec: EvoConfig | None = None, n_eval_evolved=1, out_json="evolve_results.json"):
    t0 = time.time()
    ec = ec or EvoConfig(generations=8)
    evo = run_evolution(ec, log=lambda s: print(s, flush=True))
    cache = evo["cache"]
    cache.setdefault(S.WILDTYPE.key(), evo["wt"])
    finalists = [c for c in evo["parents"] if c.key() != S.WILDTYPE.key()][:n_eval_evolved]
    # post-hoc: the candidate the UNGUARDED fitness would have picked from the same evaluations
    all_c = {k: S.Candidate(*k) for k in cache}
    ung = max(all_c.values(), key=lambda c: cache[c.key()]["fitness_unguarded"])
    print(f"search done in {evo['elapsed_s']:.0f}s, {evo['n_evaluated']} candidates", flush=True)
    # held-out evaluation at the published scale
    t1 = time.time()
    ref = P.ref_body(EVAL["lam"], EVAL["grow_steps"])
    m = EVAL["mut"]
    mref = P.ref_body(m["lam"], m["steps"], margin=m["margin"], turnover=m["turnover"])
    labels = [("hand-written (wild type)", S.WILDTYPE)] + [(f"evolved #{i+1}", c) for i, c in enumerate(finalists)]
    if ung.key() not in {c.key() for _, c in labels}:
        labels.append(("unguarded pick (post hoc)", ung))
    jobs = [(lab, c, s, ref, mref) for lab, c in labels for s in HELD_OUT]
    with ProcessPoolExecutor(ec.workers) as ex:
        res = list(ex.map(_eval_job, jobs))
    by = {}
    for lab, seed, r in res:
        by.setdefault(lab, []).append(r)
    held = {}
    for lab, c in labels:
        rows = by[lab]
        held[lab] = dict(genome_hex=c.genome.hex(), neighbor_clear=c.neighbor_clear,
                         train_fitness=cache[c.key()]["fitness"] if c.key() in cache else None,
                         train_fitness_unguarded=cache[c.key()]["fitness_unguarded"] if c.key() in cache else None,
                         hamming_vs_wt=S.hamming(c.genome, gn.WILDTYPE), diffs_vs_wt=S.describe_diff(S.WILDTYPE, c),
                         train_parts=cache[c.key()]["parts"] if c.key() in cache else None,
                         summary=summarize(rows),
                         per_seed=[{k: v for k, v in r.items() if k != "_pattern"} for r in rows])
    out = dict(
        note="Evolved-rule comparison track. ADDITION; original model/results unchanged. Post-fix simulator "
             "hardware (positional memory + intercalation ON, all error-correction layers per candidate).",
        budget=dict(search_s=evo["elapsed_s"], heldout_s=time.time() - t1, total_s=time.time() - t0,
                    candidates_evaluated=evo["n_evaluated"], workers=ec.workers,
                    sim_runs_per_candidate=len(ec.train_seeds) * (2 + len(ec.cuts))),
        evo_config=evo["config"], eval_config={k: v for k, v in EVAL.items()}, held_out_seeds=list(HELD_OUT),
        reference=dict(seed=P.REF_SEED, genome="wild type", note="independent run; never a scored body"),
        history=evo["history"],
        final_parents=[dict(genome_hex=c.genome.hex(), neighbor_clear=c.neighbor_clear,
                            fitness=cache[c.key()]["fitness"], parts=cache[c.key()]["parts"],
                            hamming_vs_wt=S.hamming(c.genome, gn.WILDTYPE),
                            diffs_vs_wt=S.describe_diff(S.WILDTYPE, c)) for c in evo["parents"]],
        wildtype_train=dict(fitness=evo["wt"]["fitness"], fitness_unguarded=evo["wt"]["fitness_unguarded"],
                            parts=evo["wt"]["parts"]),
        growth_floor_train=evo["growth_floor"],
        all_evaluated=sorted([dict(genome_hex=k[0].hex(), neighbor_clear=k[1], fitness=v["fitness"],
                                   fitness_unguarded=v["fitness_unguarded"], parts=v["parts"])
                              for k, v in cache.items()], key=lambda d: -d["fitness"]),
        protocol_note=dict(eval_v2_sustained_reused=P.V2_AVAILABLE),
        held_out=held,
        decoded={lab: gn.decode(c.genome) for lab, c in labels},
    )
    with open(os.path.join(ROOT, out_json), "w") as f:
        json.dump(out, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"wrote {out_json} total {time.time()-t0:.0f}s", flush=True)
    return out


if __name__ == "__main__":
    main()
