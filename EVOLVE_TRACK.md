# Evolved-rule comparison track (item 10)

An **addition**: `genome.py`, `sim.py`, `experiments.py`, `metrics.py`, the original tests, `results.json` and
deliverables are untouched. Rules are found by evolution over the existing 131-byte genome format and run by the
**unchanged, current (post-fix) simulator** (positional memory + intercalation ON, organizer re-derivation ON,
proofreading + triple redundancy ON). All numbers below are copied from `evolve_results.json` / `evolve_diag.json`.

Files: `evolve_track/` (`space.py` rule space, `protocol.py` honest scoring, `evolve.py` search, `run.py` full run,
`diagnose.py`, `report.py`), `tests/test_evolve_track.py`, `evolve_results.json`, `evolve_diag.json`, logs in `evolve_track/`.
Re-run: `python3 -m evolve_track.run` then `python3 -m evolve_track.diagnose`.

## 1. Rule space
A candidate = 131-byte genome + the `neighbor_clear` hardware switch (not in the genome; carried as an on/off gene).
Mutable genes: every non-TRUE condition's threshold byte (main axis), rare op/field/action changes, `p_divide`
(growth), `starve_T` (starvation self-destruct timer), apoptosis flag (0xA5 on / 0x00 off), neighbour clearance on/off.
Frozen: magic, version, n_rules=12. CRC32 is recomputed, so every candidate is a valid genome
(`space.is_valid`). The hand-written genome decodes and re-encodes byte-for-byte (it is a point in the space).

## 2. Honest protocol (same definitions as item 1 / `eval_v2.py`)
- **No self-reference.** Target = hand-written genome grown from reference seed 0 in a separate run. Reference seed is
  barred from training and held-out sets (asserted). Regeneration curves are also scored against that reference.
- **Floor** = undamaged twin (same body, rng 12345) vs reference, mean of last 50 steps.
- **Sustained recovery** = last-50 mean >= floor-0.03 AND >= 40/50 steps at/above it. The verdict is cross-checked
  against `eval_v2.sustained` on every cut (asserted equal). Steps to sustained recovery = first step at/above threshold
  from which the rest of the run stays sustained (not the old "first crossing").
- **Tumour** = cells outside the 2-dilated reference outline > 5% of reference cells, at copy-error rate 1e-2/byte/copy.
- **Anti-gaming guards:** body size 0.6-1.5x reference, all six tissues present (>= 5 cells train, >= 10 held-out),
  and a **fidelity floor** (growth similarity >= hand-written training value - 0.02, penalty 5x shortfall).
- For context only, the eval_v2 convention (cut body vs its own pre-cut body) is also recorded; it never enters fitness.

## 3. Budget
(mu+lambda) = (4+14), 8 generations, rng seed 2026, wild type in the initial population, rule-wise
crossover p=0.3. Training: lambda=5.0 (smaller body, ~613 blocks) for speed, grow 160 steps, cuts = core excision +
posterior wedge, 200 steps after the cut, tumour run 200 steps; training body seeds [1, 2].
126 candidates evaluated, 6 workers. Search 980 s, held-out evaluation 315 s, total 1301 s
(about 22 min, while another worker's jobs shared the CPUs).
Held-out evaluation: the published scale (lambda=7, grow 200, 400 steps after each cut, 4 cuts; tumour at lambda=6, 300 steps,
turnover 1%), body seeds [4, 6, 7, 42, 10000, 13, 21, 77, 314, 2026].

## 4. Best evolved genome
Fitness on training: evolved 1.3943 vs hand-written 1.2746
(growth 0.9238 vs 0.9462; regeneration pass rate 1.00 vs 0.75).
Hamming distance from the hand-written genome: 4 rule bytes (+ CRC).

`4e30010cac96a502004603003f0000000b01013201003f0000000901013a01003f0200c70a010051000000000000010200550300550000000101015d000000000000020100140000000000000602001f000000000000070201650200c70000000804003d0000000000000501014500000000000004000000000000000000039201fec6`

Changes from the hand-written genome, in plain terms:
- division probability p_divide 0.50 -> 0.67
- R01 (ORGANIZE_C) condition 1: A(anterior) > 0.140 -> A(anterior) > 0.112
- R03 (ORGANIZE_P) condition 1: C(core) < 0.210 -> C(core) < 0.163
- R08 (SENSOR) condition 1: A(anterior) > 0.368 -> A(anterior) > 0.380
- apoptosis, neighbour clearance, starve_T and all other rules: unchanged.

In words: blocks divide more often (0.50 -> 0.67 per step when allowed), a core organizer may form where the anterior
signal is somewhat weaker, the posterior organizer's allowed band is narrowed toward the outer edge, and the sensor
zone shrinks slightly.

## 5. Side by side on held-out seeds (lambda=7, honest protocol)
| metric (held-out seeds, lam=7, n=10) | hand-written (wild type) | evolved #1 |
|---|---|---|
| growth similarity vs independent ref (mean ± SD) | 0.9284 ± 0.0321 | 0.9124 ± 0.0040 |
| growth similarity 95% CI (t) | [0.9054, 0.9513] | [0.9095, 0.9153] |
| growth similarity min | 0.8952 | 0.9065 |
| reproducibility (pairwise, same genome, target-free) | 0.9322 | 0.9547 |
| body cells (mean) | 1139 | 1141 |
| anti-exploit guard violations | 0 | 0 |
| undamaged-twin floor vs ref (mean) | 0.9290 | 0.9117 |
| core excision: sustained pass (k/n, Wilson 95% CI) | 1/10 [0.02, 0.40] | 0/10 [0.00, 0.28] |
| core excision: steps to sustained recovery (median [range]) | 94 [94, 94] | none |
| core excision: failing seeds | 4, 6, 7, 42, 10000, 13, 21, 77, 314 | 4, 6, 7, 42, 10000, 13, 21, 77, 314, 2026 |
| posterior wedge: sustained pass (k/n, Wilson 95% CI) | 7/10 [0.40, 0.89] | 10/10 [0.72, 1.00] |
| posterior wedge: steps to sustained recovery (median [range]) | 37 [36, 49] | 43 [40, 52] |
| posterior wedge: failing seeds | 6, 314, 2026 | none |
| anterior amputation: sustained pass (k/n, Wilson 95% CI) | 10/10 [0.72, 1.00] | 10/10 [0.72, 1.00] |
| anterior amputation: steps to sustained recovery (median [range]) | 54 [48, 105] | 52 [44, 66] |
| anterior amputation: failing seeds | none | none |
| large anterior amputation: sustained pass (k/n, Wilson 95% CI) | 10/10 [0.72, 1.00] | 10/10 [0.72, 1.00] |
| large anterior amputation: steps to sustained recovery (median [range]) | 66 [60, 160] | 64 [53, 101] |
| large anterior amputation: failing seeds | none | none |
| tumour incidence at mu=1e-2 (k/n) | 0/10 | 0/10 |
| mutant fraction at mu=1e-2 (mean) | 0.0000 | 0.0000 |
| genome size (bytes) | 131 | 131 |
| genome zlib-9 (bytes) | 96 | 99 |
| zlib-9 blueprint of grown body (mean bytes) | 164.4 | 164.7 |
| raw blueprint (bbox bytes, mean) | 1369 | 1369 |


eval_v2 convention (vs own pre-cut body, context only): core 1/10 hand-written vs 0/10 evolved; posterior wedge
2/10 vs 7/10. Hand-written core-excision bodies that dissolved completely: seeds [7, 13, 314]; evolved: all 10 ([4, 6, 7, 42, 10000, 13, 21, 77, 314, 2026]).
The hand-written per-seed core results match `EVAL_V2.md` exactly on the overlapping seeds (4, 6, 7, 13, 42, 10000).

Context numbers for the hand-written genome on the ORIGINAL protocol (tuned seed 0, self-referenced): pre-fix core 0.833
(mean last 50), post-fix 0.9744 recovered at step 105 (first crossing), twin floor 0.9757, drift floor 0.9808.
They are not comparable to the table above (different seeds, reference and rule).

## 6. Diagnosis: why core excision gets worse
Core excision only, honest protocol (`evolve_diag.json`):

| set | hand-written | evolved |
|---|---|---|
| lambda=7, the two TRAINING seeds (1, 2) | 2/2 | 0/2 |
| lambda=5 (training scale), held-out seeds 4, 6, 7, 42, 10000 | 5/5 | 1/5 |

The evolved genome passed core excision on both training bodies at lambda=5, but fails at lambda=7 even on those same
seeds, and passes only 1 of 5 new bodies at lambda=5 (all four failures end within 0.002 of the threshold; three have a last-50 mean above it
but miss the 40/50 rule). So the core "gain" was **overfitting to two training bodies at one scale**. At lambda=5, bodies from
different seeds are sometimes identical (e.g. hand-written seeds 1 and 6 give the same 0.9525), so two training seeds
were effectively very few independent bodies. The hand-written genome's core repair is far more reliable at lambda=5
(5/5) than at lambda=7 (1/10).

## 7. Metric-gaming check
- **Trivial bodies:** none got through. 25 of 126 evaluated candidates broke the size/tissue guards and were penalised.
  Held-out body sizes match (1139 vs 1141 blocks, 0 guard violations).
- **Fidelity trade (found, partly contained):** an unguarded first search (`evolve_run_unguarded_aborted.log`, stopped
  after generation 3) dropped growth similarity from 0.964 to 0.888 to reach a 100% regeneration pass rate. The
  criterion is relative to the candidate's own undamaged twin, so a body that matches the plan less closely can pass
  as easily. With the fidelity floor the winner still gives up some fidelity: training growth 0.9238 vs 0.9462,
  held-out 0.9124 vs 0.9284. The unguarded score, applied to the same
  evaluations, picks the same genome.
- **Home advantage:** the reference is the hand-written body, so growth similarity favours the hand-written genome by
  construction. The target-free measure (pairwise similarity between bodies of the same genome) favours the evolved one
  (0.9547 vs 0.9322), and its growth spread is much smaller (SD 0.0040 vs 0.0321).
- **Tumours:** none for either genome at 1e-2. Evolution never turned off apoptosis or clearance in the final parents.
  At this error rate the test barely separates genomes (3 of 126 candidates showed any tumour).

## 8. Verdict
The evolved rule set does **not** beat the hand-written genome overall. It is better at the posterior wedge
(10/10 vs 7/10 sustained, honest; 7/10 vs 2/10 eval_v2 convention), equal on both anterior amputations (10/10)
and tumours (0/10), and slightly more reproducible. It is worse on growth fidelity to the designed plan and much worse on
core excision (0/10 vs 1/10; every evolved body dissolves). Same 131-byte size; genome zlib 99 vs 96 B; grown-body zlib
blueprint about 165 B for both.
The useful lead is the wedge: narrowing the posterior organizer's band (R03 C < 0.210 -> < 0.163) is a single-rule
change worth testing alone, with `p_divide` held at 0.50, as a candidate fix for the wedge.

## 9. Caveats
- Small search (126 candidates, 2 training bodies, trained at lambda=5), local around the hand-written genome. A larger
  search across several scales and more seeds could do better; this run does not show that it can.
- The fitness weights are a design choice; other weights pick other trade-offs.
- n=10 held-out bodies: confidence intervals are wide (Wilson 95% for 7/10 is 0.40-0.89).
- The neighbour-clearance switch is hardware, not genome. It stayed ON in the winner.
