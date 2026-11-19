"""Diagnostic: is the evolved genome's held-out core-excision failure a SCALE effect (trained at
lam=5, tested at lam=7) or a SEED effect? Core excision only, honest protocol. Writes evolve_diag.json."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import json  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

from . import protocol as P  # noqa: E402
from . import space as S  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETS = {"lam7_training_seeds": (7.0, 200, 400, (1, 2)),
        "lam5_heldout_seeds": (5.0, 160, 200, (4, 6, 7, 42, 10000))}


def _job(a):
    label, cand, setname, seed, ref = a
    lam, g, rs, _ = SETS[setname]
    r = P.evaluate_body(cand, seed, lam, g, rs, [P.CORE], ref, min_tissue=5)
    c = r["cuts"][P.CORE]
    return dict(genome=label, set=setname, seed=seed, lam=lam, sustained=c["sustained"],
                last50_vs_ref=c["last50_mean"], threshold=c["threshold"], cells_final=c["cells_final"],
                context_self_last50=c["context_self_last50"])


if __name__ == "__main__":
    res = json.load(open(os.path.join(ROOT, "evolve_results.json")))
    ev = res["held_out"]["evolved #1"]
    cands = {"hand-written (wild type)": S.WILDTYPE,
             "evolved #1": S.Candidate(bytes.fromhex(ev["genome_hex"]), ev["neighbor_clear"])}
    refs = {k: P.ref_body(v[0], v[1]) for k, v in SETS.items()}
    jobs = [(l, c, s, seed, refs[s]) for l, c in cands.items() for s, v in SETS.items() for seed in v[3]]
    with ProcessPoolExecutor(8) as ex:
        rows = list(ex.map(_job, jobs))
    json.dump(rows, open(os.path.join(ROOT, "evolve_diag.json"), "w"), indent=1)
    for r in rows:
        print(r)
