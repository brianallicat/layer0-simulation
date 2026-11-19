"""Run every Layer-0 experiment, print a results table, write results.json / RESULTS.md and figures.

    python run_all.py            # full run (~5-8 min on 8 cores)
"""
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiments as ex  # noqa: E402
import genome as gn  # noqa: E402
import viz  # noqa: E402
from viz import plt, show, legend  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)


def crop_of(p, pad=3):
    ys, xs = np.nonzero(p)
    lo = min(ys.min(), xs.min()) - pad
    hi = max(ys.max(), xs.max()) + pad
    return (lo, hi)


def fig_growth(g):
    art = g
    frames = art["frames"]
    ks = sorted(frames)
    cr = crop_of(art["ref"], 4)
    fig, axs = plt.subplots(2, 6, figsize=(18, 6.6), gridspec_kw=dict(height_ratios=[1, 1]))
    for ax, k in zip(axs[0], ks):
        p, o = frames[k]
        show(ax, p, o, f"step {k}: {int((p > 0).sum())} blocks", crop=cr)
    names = ["C (core organizer)", "A (anterior organizer)", "P (posterior organizer)"]
    for i in range(3):
        ax = axs[1, i]
        f = np.ma.masked_where(art["ref"] == 0, np.log10(art["fields"][i] + 1e-4))
        im = ax.imshow(f, cmap="magma", vmin=-2.5, vmax=0)
        ax.set_xlim(*cr); ax.set_ylim(cr[1], cr[0]); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"field {names[i]}  (log10)", fontsize=9)
    fig.colorbar(im, ax=axs[1, 2], fraction=0.046)
    ax = axs[1, 3]
    ax.axis("off")
    ax.text(0, 1, gn.decode(gn.WILDTYPE).replace("(core)", "").replace("(anterior)", "").replace("(posterior)", ""),
            fontsize=6.3, family="monospace", va="top")
    ax.set_title(f"the whole genome: {gn.GENOME_LEN} bytes", fontsize=9)
    for ax in axs[1, 4:]:
        ax.axis("off")
    legend(fig)
    fig.suptitle("Layer 0 growth from one seed block: local rules + morphogen fields -> regional body plan", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    path = os.path.join(FIG, "1_growth_sequence.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def fig_regen(res, art):
    names = [r["name"] for r in res["rows"] if r["regen"]]
    cr = crop_of(art["base"], 4)
    fig, axs = plt.subplots(len(names), 5, figsize=(17, 3.4 * len(names)))
    for i, n in enumerate(names):
        r = next(x for x in res["rows"] if x["name"] == n and x["regen"])
        rc = next(x for x in res["rows"] if x["name"] == n and not x["regen"])
        cut, fin = art["pics"][(n, True)]
        _, finc = art["pics"][(n, False)]
        show(axs[i, 0], art["base"], art["base_orgs"], "before", crop=cr)
        show(axs[i, 1], cut, None, f"{n}\n{r['frac_removed']*100:.0f}% removed", crop=cr)
        show(axs[i, 2], fin, r["orgs_after"], f"regrown (rules+gradients)\nsim {r['final_mean_last50']:.3f}", crop=cr)
        show(axs[i, 3], finc, rc["orgs_after"], f"control: organizer re-derivation OFF\nsim {rc['final_mean_last50']:.3f}", crop=cr)
        ax = axs[i, 4]
        ax.plot(art["twin_curve"], color="0.6", label="undamaged twin (noise floor)")
        ax.plot(art["curves"][(n, True)], color="C2", label="regenerating")
        ax.plot(art["curves"][(n, False)], color="C3", label="control")
        ax.set_ylim(0, 1.02); ax.set_xlabel("steps after damage"); ax.set_ylabel("similarity to pre-damage")
        if i == 0:
            ax.legend(fontsize=7, loc="lower right")
    legend(fig)
    fig.suptitle("Regeneration by re-derivation: no stored blueprint, only gradients + the same rules", fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    path = os.path.join(FIG, "2_regeneration.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def fig_mut(res):
    tab = res["table"]
    conds = list(dict.fromkeys(r["condition"] for r in tab))
    mus = sorted({r["mu"] for r in tab})
    xs = [m if m > 0 else 3e-4 for m in mus]
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.2))
    styles = ["-o", "-s", "-^", "-v", "--o", "--s", "--^", "-x", ":d"]
    for c, st in zip(conds, styles):
        rr = [next(r for r in tab if r["condition"] == c and r["mu"] == m) for m in mus]
        lw = 3 if c in ("all 3 layers ON", "ALL OFF", "no apoptosis/clearance") else 1.4
        axs[0].plot(xs, [r["tumor_incidence"] for r in rr], st, lw=lw, label=c)
        axs[1].plot(xs, [r["mean_fidelity"] for r in rr], st, lw=lw, label=c)
        axs[2].plot(xs, [r["mean_mutant_frac"] for r in rr], st, lw=lw, label=c)
    for ax, t in zip(axs, ["tumor / overgrowth incidence (>5% extra blocks)", "pattern fidelity (similarity to wild type)", "fraction of blocks with a mutated genome"]):
        ax.set_xscale("log"); ax.set_xlabel("copy error rate per byte per copy (leftmost point = 0)"); ax.set_title(t, fontsize=10)
        ax.set_ylim(-0.03, 1.03); ax.grid(alpha=0.3)
    axs[0].legend(fontsize=7.5, loc="upper left")
    fig.suptitle("Error correction ablation: proofreading x redundancy x apoptosis/clearance", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = os.path.join(FIG, "3_error_correction_ablation.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def fig_mut_examples(examples, ref):
    fig, axs = plt.subplots(1, len(examples) + 1, figsize=(4 * (len(examples) + 1), 4.4))
    show(axs[0], ref, None, "wild-type reference")
    for ax, (t, p) in zip(axs[1:], examples):
        show(ax, p, None, t)
    legend(fig)
    fig.suptitle("Example phenotypes after 300 steps of wear + copy errors", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    path = os.path.join(FIG, "3b_mutation_examples.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def fig_scaling(res, art):
    rows = res["rows"]
    fig, axs = plt.subplots(1, 4, figsize=(19, 4.8), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.3]))
    for ax, r in zip(axs[:3], rows):
        p, o = art["pics"][r["lam"]]
        show(ax, p, o, f"lambda={r['lam']:g}: {r['body_cells']:.0f} blocks, genome {r['genome_bytes']} B\n(sha {r['genome_sha']})", crop=crop_of(p, 3))
    ax = axs[3]
    x = np.arange(len(rows))
    w = 0.2
    ax.bar(x - 1.5 * w, [r["genome_bytes"] for r in rows], w, label="genome (bytes)")
    ax.bar(x - 0.5 * w, [r["blueprint_zlib_bytes"] for r in rows], w, label="blueprint, zlib-compressed raster")
    ax.bar(x + 0.5 * w, [r["blueprint_raw_bytes"] for r in rows], w, label="blueprint, raw 1 B/cell raster")
    ax.bar(x + 1.5 * w, [r["blueprint_1KB_per_block"] for r in rows], w, label="blueprint, 1 KB/block (doc)")
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([f"{r['body_cells']:.0f} blocks" for r in rows])
    ax.set_ylabel("bytes"); ax.legend(fontsize=7.5); ax.set_title("same recipe, bigger body", fontsize=10)
    legend(fig)
    fig.suptitle("Scaling: one unchanged genome grows bodies of different size with the same proportions", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    path = os.path.join(FIG, "4_scaling.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def fig_morph(res, art):
    fig, axs = plt.subplots(1, 6, figsize=(22, 4.2))
    cr = crop_of(art["P0"] | art["target"], 3)
    show(axs[0], art["P0"], art["base_orgs"], "before (core organizer at *)", crop=cr)
    for ax, k in zip(axs[1:4], sorted(art["frames"])):
        p, o = art["frames"][k]
        show(ax, p, o, f"{k} steps after moving the core source", crop=cr)
    show(axs[4], art["target"], None, "same plan re-centred (scoring target)", crop=cr)
    ax = axs[5]
    ax.plot(art["curve"], label="body vs re-centred plan")
    ax.plot(art["curve_bp"], label="body vs old blueprint")
    ax.axhline(res["stored_blueprint_similarity_to_shifted_plan"], ls="--", color="k", lw=1, label="stored blueprint vs new plan")
    ax.set_ylim(0, 1.02); ax.legend(fontsize=7); ax.set_xlabel("steps")
    legend(fig)
    fig.suptitle("Morph: move the core organizer (a field source) 0.3 R to the right - the body re-derives its whole plan around it", fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    path = os.path.join(FIG, "5_morph.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def fig_physics(res, art):
    import materials as M
    p = art["pattern"]
    cr = crop_of(p, 3)
    fig, axs = plt.subplots(1, 5, figsize=(24, 4.9))
    show(axs[0], p, None, "tissues (materials: see table)", crop=cr)
    for ax, T, t in zip(axs[1:4], [art["T"], art["T_vac"], art["T_nohp"]],
                        ["still air, 0.30 W core (nominal)", "vacuum (radiation only), 0.30 W", "still air, NO heat pipes"]):
        im = ax.imshow(np.ma.masked_invalid(T), cmap="inferno")
        ax.set_xlim(*cr); ax.set_ylim(cr[1], cr[0]); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"temperature {t}\nmax {np.nanmax(T):.1f} C", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, label="degC")
    V = art["V"]
    im = axs[4].imshow(np.ma.masked_invalid((1 - V / M.PhysConfig().V_bus) * 100), cmap="viridis")
    axs[4].set_xlim(*cr); axs[4].set_ylim(cr[1], cr[0]); axs[4].set_xticks([]); axs[4].set_yticks([])
    axs[4].set_title("supply voltage droop (%)\nconduit blocks hold the bus", fontsize=9)
    fig.colorbar(im, ax=axs[4], fraction=0.046)
    legend(fig)
    fig.suptitle("Materials physics on the grown body: heat (core+sensors -> scaffold heat pipes -> radiator) and power delivery", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    path = os.path.join(FIG, "6_physics.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def main():
    t0 = time.time()
    results, figs = {}, []
    print(f"genome: {gn.GENOME_LEN} bytes, {gn.MAX_RULES} rules\n{gn.decode(gn.WILDTYPE)}\n")

    print("[1/7] growth reproducibility ...", flush=True)
    g, gart = ex.growth_test()
    results["growth"] = g
    figs.append(fig_growth(gart))

    print("[2/7] regeneration ...", flush=True)
    r, rart = ex.regeneration_test()
    for row in r["rows"]:
        row.pop("curve")
    results["regeneration"] = r
    figs.append(fig_regen(r, rart))

    print("[3/7] mutation / error correction ...", flush=True)
    m, mart = ex.mutation_test()
    results["mutation"] = {"table": m["table"], "ref_cells": m["ref_cells"]}
    figs.append(fig_mut(m))
    # example phenotypes
    exs = []
    for cond, mu, s in [("ALL OFF", 1e-2, 0), ("no apoptosis/clearance", 1e-1, 0), ("all 3 layers ON", 1e-1, 0)]:
        w = ex.World(ex.Config(lam=ex.MUT_LAM, seed=s, mu=mu, margin=ex.MUT_MARGIN, turnover=ex.MUT_TURNOVER, **ex.CONDITIONS[cond]))
        w.run(ex.MUT_STEPS)
        exs.append((f"{cond}\nmu={mu:g}: {int(w.alive.sum())} blocks", w.pattern()))
    figs.append(fig_mut_examples(exs, mart["ref"]))

    print("[4/7] scaling ...", flush=True)
    s, sart = ex.scaling_test()
    results["scaling"] = s
    figs.append(fig_scaling(s, sart))

    print("[5/7] morph ...", flush=True)
    mo, moart = ex.morph_test()
    results["morph"] = mo
    figs.append(fig_morph(mo, moart))

    print("[6/7] core regeneration robustness (4 bodies, memory on/off) ...", flush=True)
    results["core_robustness"] = ex.core_regen_seeds()
    for row in results["core_robustness"]:
        pass

    print("[7/7] materials physics ...", flush=True)
    ph, phart = ex.physics_test(rart)
    results["physics"] = ph
    figs.append(fig_physics(ph, phart))
    with open(os.path.join(HERE, "physics_art.pkl"), "wb") as f:
        pickle.dump(dict(phys=phart, regen=rart, growth=gart, morph=moart), f)

    results["figures"] = figs
    results["runtime_s"] = time.time() - t0
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(results, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    md = report(results)
    with open(os.path.join(HERE, "RESULTS.md"), "w") as f:
        f.write(md)
    print(md)


def report(R):
    L = []
    g = R["growth"]["summary"]
    L.append("# Layer 0 results\n")
    L.append(f"Genome: **{gn.GENOME_LEN} bytes** ({gn.MAX_RULES} local rules), same bytes in every block (x3 redundant copies = {3*gn.GENOME_LEN} B/block).\n")
    L.append("## 1. Growth from one seed (lambda=7, 200 steps)\n")
    L.append("| metric | value |\n|---|---|")
    L.append(f"| seeds compared to reference seed 0 | {g['n_seeds']} |")
    L.append(f"| body size (reference) | {g['ref_cells']} blocks |")
    L.append(f"| pattern similarity mean / min / max | {g['mean_similarity']:.3f} / {g['min_similarity']:.3f} / {g['max_similarity']:.3f} |")
    L.append(f"| similarity allowing +-2 cell offset (mean) | {g['mean_aligned']:.3f} |")
    L.append(f"| noise floor: same body vs itself 50 steps later | {g['self_similarity_50_steps_later']:.3f} |")
    L.append(f"| tissue counts (reference) | {g['tissue_counts']} |\n")
    r = R["regeneration"]
    L.append(f"## 2. Regeneration by re-derivation (noise floor = undamaged twin: {r['noise_floor']:.3f}; 'recovered' = within 0.03 of it)\n")
    L.append("| damage | removed | organizers destroyed | organizer re-derivation | similarity (mean last 50 steps) | +-2 cell aligned | steps to recover | recovered |")
    L.append("|---|---|---|---|---|---|---|---|")
    for x in r["rows"]:
        L.append(f"| {x['name']} | {x['frac_removed']*100:.0f}% | {','.join(x['organizers_destroyed']) or '-'} | {'ON' if x['regen'] else 'OFF (control)'} | "
                 f"{x['final_mean_last50']:.3f} | {x['aligned_final']:.3f} | {x['steps_to_recover'] or 'not reached'} | {'YES' if x['recovered'] else 'no'} |")
    m = R["mutation"]
    L.append(f"\n## 3. Copy errors vs error correction (lambda=6, 300 steps, wear-driven turnover 1%/step, 4 seeds each; tumor = >5% extra blocks outside the normal outline; ref body {m['ref_cells']} blocks)\n")
    L.append("| condition | mu (per byte per copy) | tumor incidence | mean overgrowth (x body) | pattern fidelity | mutant blocks | extinct runs |")
    L.append("|---|---|---|---|---|---|---|")
    for x in m["table"]:
        L.append(f"| {x['condition']} | {x['mu']:g} | {x['tumor_incidence']*100:.0f}% | {x['mean_overgrowth_frac']:.2f} | {x['mean_fidelity']:.3f} | {x['mean_mutant_frac']*100:.1f}% | {x['extinct']} |")
    s = R["scaling"]
    L.append(f"\n## 4. Scaling (same genome bytes at every size: {s['same_genome']})\n")
    L.append("| lambda | body blocks | diameter | genome B | blueprint zlib B | blueprint raw B | blueprint 1KB/block | shape similarity to mid size (resampled) | core / sensor / radiator / conduit / shell / scaffold fractions |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for x in s["rows"]:
        f = x["tissue_fractions"]
        L.append(f"| {x['lam']:g} | {x['body_cells']:.0f} | {x['body_diameter']:.1f} | {x['genome_bytes']} | {x['blueprint_zlib_bytes']} | {x['blueprint_raw_bytes']} | {x['blueprint_1KB_per_block']:,} | {x['shape_similarity_to_mid_scale']:.3f} | "
                 f"{f['core']:.2f} / {f['sensor']:.2f} / {f['radiator']:.2f} / {f['conduit']:.2f} / {f['shell']:.2f} / {f['scaffold']:.2f} |")
    mo = R["morph"]
    L.append("\n## 5. Morph: move the core field source by %d cells\n" % mo["shift_cells"])
    L.append("| metric | value |\n|---|---|")
    L.append(f"| body vs re-centred plan after 400 steps | {mo['final_similarity_to_shifted_plan']:.3f} |")
    L.append(f"| steps to reach 0.90 | {mo['steps_to_090']} |")
    L.append(f"| a stored blueprint (unchanged) vs re-centred plan | {mo['stored_blueprint_similarity_to_shifted_plan']:.3f} |")
    L.append(f"| organizers before -> after | {mo['orgs_before']} -> {mo['orgs_after']} |")
    cr = R["core_robustness"]
    L.append("\n## 6. Core regeneration across independently grown bodies (positional memory ON vs OFF)\n")
    L.append("| body seed | positional memory | noise floor | core excision (mean last 50) | recovered | steps to recover | C before -> after |")
    L.append("|---|---|---|---|---|---|---|")
    for x in cr:
        L.append(f"| {x['seed']} | {'ON' if x['memory'] else 'OFF'} | {x['noise_floor']:.3f} | {x['core_final']:.3f} | {'YES' if x['recovered'] else 'no'} | {x['steps_to_recover'] or 'not reached'} | {x['C_before']} -> {x['C_after']} |")
    ph = R["physics"]
    L.append("\n## 7. Materials physics (1 mm blocks, sheet one block thick; air: h=10 W/m2K + radiation, 25 C; vacuum: radiation only)\n")
    L.append("| body | env | all OK | T max C | conduit / core / sensor T max C | heat in W | max droop | bend util (100 g) | thermal-mismatch stress MPa (util) | mass g |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for x in ph["rows"]:
        pt = x["per_tissue_T"]
        L.append(f"| {x['body']} | {x['env']} | {'YES' if x['ok'] else 'NO'} | {x['T_max_C']:.1f} | {pt.get('conduit', float('nan')):.1f} / {pt.get('core', float('nan')):.1f} / {pt.get('sensor', float('nan')):.1f} | {x['heat_in_W']:.3f} | {x['max_droop']:.4f} | {x['bend_util']:.3f} | {x['thermal_stress_MPa']:.2f} ({x['thermal_stress_util']:.2f}) | {x['mass_g']:.2f} |")
    L.append(f"\nMax total core power before any block exceeds its limit: air {ph['max_core_W_air']:.3f} W, vacuum {ph['max_core_W_vacuum']:.3f} W, air without heat pipes {ph['no_heat_pipes']['max_core_W_air']:.3f} W (limit set by the 40 C vanadium electrolyte).")
    L.append(f"Power ablation, no copper traces: max droop {ph['no_copper']['max_droop']:.2f}, collapsed={ph['no_copper']['collapsed']}, power OK={ph['no_copper']['power_ok']}.")
    L.append(f"No heat pipes (air, 0.30 W): T max {ph['no_heat_pipes']['T_max_C']:.1f} C, OK={ph['no_heat_pipes']['thermal_ok']}, violations {ph['no_heat_pipes']['violations']}.")
    fx = os.path.join(HERE, "CORE_FIX.md")
    if os.path.exists(fx):
        L.append("\n" + open(fx).read())
    L.append(f"\nFigures: {', '.join(os.path.relpath(p, HERE) for p in R['figures'])}\n\nRuntime {R['runtime_s']:.0f} s")
    return "\n".join(L)


if __name__ == "__main__":
    main()
