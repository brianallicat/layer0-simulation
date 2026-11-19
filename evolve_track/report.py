"""Render EVOLVE_TRACK.md tables from evolve_results.json (numbers are copied, never typed)."""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def f3(x):
    return "n/a" if x is None else f"{x:.3f}"


def f4(x):
    return "n/a" if x is None else f"{x:.4f}"


def table(res):
    from .protocol import ANT, CORE, LARGE_ANT, WEDGE
    labs = list(res["held_out"])
    S = {l: res["held_out"][l]["summary"] for l in labs}
    rows = []
    hdr = "| metric (held-out seeds, lam=7, n=%d) | " % S[labs[0]]["n"] + " | ".join(labs) + " |"
    rows.append(hdr)
    rows.append("|" + "---|" * (len(labs) + 1))

    def add(name, fn):
        rows.append(f"| {name} | " + " | ".join(fn(S[l]) for l in labs) + " |")
    add("growth similarity vs independent ref (mean ± SD)", lambda s: f"{s['growth_mean']:.4f} ± {s['growth_sd']:.4f}")
    add("growth similarity 95% CI (t)", lambda s: f"[{s['growth_ci95'][0]:.4f}, {s['growth_ci95'][1]:.4f}]")
    add("growth similarity min", lambda s: f4(s["growth_min"]))
    add("reproducibility (pairwise, same genome, target-free)", lambda s: f4(s["reproducibility_pairwise_mean"]))
    add("body cells (mean)", lambda s: f"{s['cells_mean']:.0f}")
    add("anti-exploit guard violations", lambda s: str(s["guard_violations"]))
    add("undamaged-twin floor vs ref (mean)", lambda s: f4(s["twin_floor_mean"]))
    for cut, short in ((CORE, "core excision"), (WEDGE, "posterior wedge"), (ANT, "anterior amputation"),
                       (LARGE_ANT, "large anterior amputation")):
        add(f"{short}: sustained pass (k/n, Wilson 95% CI)",
            lambda s, c=cut: f"{s['cuts'][c]['pass_k']}/{s['n']} [{s['cuts'][c]['pass_ci95'][0]:.2f}, {s['cuts'][c]['pass_ci95'][1]:.2f}]")
        add(f"{short}: steps to sustained recovery (median [range])",
            lambda s, c=cut: "none" if s['cuts'][c]['steps_to_sustained_median'] is None else
            f"{s['cuts'][c]['steps_to_sustained_median']:.0f} {s['cuts'][c]['steps_to_sustained_range']}")
        add(f"{short}: failing seeds", lambda s, c=cut: ", ".join(map(str, s['cuts'][c]['failing_seeds'])) or "none")
    add("tumour incidence at mu=1e-2 (k/n)", lambda s: f"{s['tumour_k']}/{s['n']}")
    add("mutant fraction at mu=1e-2 (mean)", lambda s: f4(s["mutant_frac_mean"]))
    add("genome size (bytes)", lambda s: str(s["genome_bytes"]))
    add("genome zlib-9 (bytes)", lambda s: str(s["genome_zlib_bytes"]))
    add("zlib-9 blueprint of grown body (mean bytes)", lambda s: f"{s['blueprint_zlib_mean']:.1f}")
    add("raw blueprint (bbox bytes, mean)", lambda s: f"{s['blueprint_raw_mean']:.0f}")
    return "\n".join(rows)


if __name__ == "__main__":
    res = json.load(open(os.path.join(ROOT, "evolve_results.json")))
    print(table(res))
