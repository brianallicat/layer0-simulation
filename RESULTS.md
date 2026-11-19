# Layer 0 results

Genome: **131 bytes** (12 local rules), same bytes in every block (x3 redundant copies = 393 B/block).

## 1. Growth from one seed (lambda=7, 200 steps)

| metric | value |
|---|---|
| seeds compared to reference seed 0 | 10 |
| body size (reference) | 1140 blocks |
| pattern similarity mean / min / max | 0.946 / 0.913 / 0.976 |
| similarity allowing +-2 cell offset (mean) | 0.946 |
| noise floor: same body vs itself 50 steps later | 0.981 |
| tissue counts (reference) | {'scaffold': 315, 'shell': 178, 'conduit': 103, 'core': 60, 'sensor': 141, 'radiator': 343} |

## 2. Regeneration by re-derivation (noise floor = undamaged twin: 0.976; 'recovered' = within 0.03 of it)

| damage | removed | organizers destroyed | organizer re-derivation | similarity (mean last 50 steps) | +-2 cell aligned | steps to recover | recovered |
|---|---|---|---|---|---|---|---|
| anterior amputation (incl. organizer A) | 29% | A | ON | 0.972 | 0.975 | 49 | YES |
| large anterior amputation (incl. A) | 39% | A | ON | 0.968 | 0.975 | 85 | YES |
| core excision (incl. organizer C) | 26% | C | ON | 0.974 | 0.974 | 105 | YES |
| posterior wedge (incl. organizer P) | 33% | P | ON | 0.944 | 0.944 | 156 | no |
| anterior amputation (incl. organizer A) | 29% | A | OFF (control) | 0.477 | 0.478 | not reached | no |
| large anterior amputation (incl. A) | 39% | A | OFF (control) | 0.478 | 0.479 | not reached | no |
| core excision (incl. organizer C) | 26% | C | OFF (control) | 0.000 | 0.000 | not reached | no |
| posterior wedge (incl. organizer P) | 33% | P | OFF (control) | 0.890 | 0.889 | not reached | no |

## 3. Copy errors vs error correction (lambda=6, 300 steps, wear-driven turnover 1%/step, 4 seeds each; tumor = >5% extra blocks outside the normal outline; ref body 855 blocks)

| condition | mu (per byte per copy) | tumor incidence | mean overgrowth (x body) | pattern fidelity | mutant blocks | extinct runs |
|---|---|---|---|---|---|---|
| all 3 layers ON | 0 | 0% | 0.00 | 0.929 | 0.0% | 0 |
| all 3 layers ON | 0.001 | 0% | 0.00 | 0.922 | 0.0% | 0 |
| all 3 layers ON | 0.003 | 0% | 0.00 | 0.916 | 0.0% | 0 |
| all 3 layers ON | 0.01 | 0% | 0.00 | 0.939 | 0.0% | 0 |
| all 3 layers ON | 0.03 | 0% | 0.00 | 0.947 | 0.0% | 0 |
| all 3 layers ON | 0.1 | 0% | 0.00 | 0.889 | 0.1% | 0 |
| no proofreading | 0 | 0% | 0.00 | 0.929 | 0.0% | 0 |
| no proofreading | 0.001 | 0% | 0.00 | 0.911 | 0.0% | 0 |
| no proofreading | 0.003 | 0% | 0.00 | 0.931 | 0.0% | 0 |
| no proofreading | 0.01 | 0% | 0.00 | 0.923 | 0.1% | 0 |
| no proofreading | 0.03 | 0% | 0.00 | 0.881 | 0.2% | 0 |
| no proofreading | 0.1 | 100% | 3.77 | 0.046 | 99.9% | 0 |
| no redundancy | 0 | 0% | 0.00 | 0.929 | 0.0% | 0 |
| no redundancy | 0.001 | 0% | 0.00 | 0.904 | 0.0% | 0 |
| no redundancy | 0.003 | 0% | 0.00 | 0.912 | 0.0% | 0 |
| no redundancy | 0.01 | 0% | 0.00 | 0.914 | 0.2% | 0 |
| no redundancy | 0.03 | 0% | 0.00 | 0.896 | 0.6% | 0 |
| no redundancy | 0.1 | 100% | 3.77 | 0.053 | 88.0% | 0 |
| no apoptosis/clearance | 0 | 0% | 0.00 | 0.887 | 0.0% | 0 |
| no apoptosis/clearance | 0.001 | 0% | 0.00 | 0.927 | 0.0% | 0 |
| no apoptosis/clearance | 0.003 | 0% | 0.00 | 0.907 | 0.0% | 0 |
| no apoptosis/clearance | 0.01 | 0% | 0.00 | 0.908 | 0.1% | 0 |
| no apoptosis/clearance | 0.03 | 0% | 0.00 | 0.905 | 1.2% | 0 |
| no apoptosis/clearance | 0.1 | 25% | 0.08 | 0.856 | 32.2% | 0 |
| proofreading only | 0 | 0% | 0.00 | 0.887 | 0.0% | 0 |
| proofreading only | 0.001 | 25% | 0.04 | 0.842 | 19.0% | 0 |
| proofreading only | 0.003 | 100% | 3.34 | 0.194 | 86.5% | 0 |
| proofreading only | 0.01 | 75% | 2.79 | 0.301 | 88.7% | 0 |
| proofreading only | 0.03 | 100% | 3.76 | 0.090 | 99.5% | 0 |
| proofreading only | 0.1 | 100% | 3.76 | 0.063 | 100.0% | 0 |
| redundancy only | 0 | 0% | 0.00 | 0.887 | 0.0% | 0 |
| redundancy only | 0.001 | 0% | 0.00 | 0.903 | 0.0% | 0 |
| redundancy only | 0.003 | 0% | 0.00 | 0.913 | 2.0% | 0 |
| redundancy only | 0.01 | 25% | 0.94 | 0.707 | 41.2% | 0 |
| redundancy only | 0.03 | 75% | 2.82 | 0.207 | 95.9% | 0 |
| redundancy only | 0.1 | 100% | 3.77 | 0.045 | 100.0% | 0 |
| apoptosis/clearance only | 0 | 0% | 0.00 | 0.929 | 0.0% | 0 |
| apoptosis/clearance only | 0.001 | 0% | 0.00 | 0.901 | 0.1% | 0 |
| apoptosis/clearance only | 0.003 | 0% | 0.00 | 0.899 | 0.8% | 0 |
| apoptosis/clearance only | 0.01 | 100% | 3.76 | 0.061 | 91.0% | 0 |
| apoptosis/clearance only | 0.03 | 100% | 3.76 | 0.038 | 100.0% | 0 |
| apoptosis/clearance only | 0.1 | 100% | 3.76 | 0.036 | 100.0% | 0 |
| ALL OFF | 0 | 0% | 0.00 | 0.887 | 0.0% | 0 |
| ALL OFF | 0.001 | 100% | 3.71 | 0.139 | 97.0% | 0 |
| ALL OFF | 0.003 | 100% | 3.77 | 0.057 | 99.6% | 0 |
| ALL OFF | 0.01 | 100% | 3.77 | 0.043 | 99.9% | 0 |
| ALL OFF | 0.03 | 100% | 3.77 | 0.043 | 100.0% | 0 |
| ALL OFF | 0.1 | 100% | 3.76 | 0.037 | 100.0% | 0 |
| all ON except neighbour clearance | 0 | 0% | 0.00 | 0.887 | 0.0% | 0 |
| all ON except neighbour clearance | 0.001 | 0% | 0.00 | 0.927 | 0.0% | 0 |
| all ON except neighbour clearance | 0.003 | 0% | 0.00 | 0.907 | 0.0% | 0 |
| all ON except neighbour clearance | 0.01 | 0% | 0.00 | 0.911 | 0.0% | 0 |
| all ON except neighbour clearance | 0.03 | 0% | 0.00 | 0.933 | 0.0% | 0 |
| all ON except neighbour clearance | 0.1 | 0% | 0.00 | 0.904 | 0.0% | 0 |

## 4. Scaling (same genome bytes at every size: True)

| lambda | body blocks | diameter | genome B | blueprint zlib B | blueprint raw B | blueprint 1KB/block | shape similarity to mid size (resampled) | core / sensor / radiator / conduit / shell / scaffold fractions |
|---|---|---|---|---|---|---|---|---|
| 4 | 414 | 22.9 | 131 | 98 | 529 | 423,936 | 0.770 | 0.05 / 0.11 / 0.30 / 0.08 / 0.19 / 0.26 |
| 7 | 1140 | 38.1 | 131 | 166 | 1369 | 1,167,360 | 0.987 | 0.05 / 0.12 / 0.30 / 0.09 / 0.16 / 0.27 |
| 11 | 2690 | 58.5 | 131 | 265 | 3481 | 2,755,584 | 0.858 | 0.05 / 0.13 / 0.30 / 0.09 / 0.14 / 0.29 |

## 5. Morph: move the core field source by 6 cells

| metric | value |
|---|---|
| body vs re-centred plan after 400 steps | 0.956 |
| steps to reach 0.90 | 155 |
| a stored blueprint (unchanged) vs re-centred plan | 0.330 |
| organizers before -> after | {'C': [[29, 29]], 'A': [[29, 42]], 'P': [[27, 18]]} -> {'C': [[29, 35]], 'A': [[29, 48]], 'P': [[26, 24]]} |

## 6. Core regeneration across independently grown bodies (positional memory ON vs OFF)

| body seed | positional memory | noise floor | core excision (mean last 50) | recovered | steps to recover | C before -> after |
|---|---|---|---|---|---|---|
| 0 | ON | 0.976 | 0.974 | YES | 105 | [[29, 29]] -> [[29, 29]] |
| 1 | ON | 0.974 | 0.970 | YES | 102 | [[29, 29]] -> [[29, 29]] |
| 2 | ON | 0.977 | 0.976 | YES | 123 | [[29, 29]] -> [[29, 29]] |
| 3 | ON | 0.974 | 0.963 | YES | 188 | [[29, 29]] -> [[29, 29]] |
| 0 | OFF | 0.976 | 0.833 | no | not reached | [[29, 29]] -> [[28, 30]] |
| 1 | OFF | 0.974 | 0.834 | no | not reached | [[29, 29]] -> [[28, 30]] |
| 2 | OFF | 0.977 | 0.824 | no | not reached | [[29, 29]] -> [[30, 30]] |
| 3 | OFF | 0.974 | 0.756 | no | not reached | [[29, 29]] -> [[27, 30]] |

## 7. Materials physics (1 mm blocks, sheet one block thick; air: h=10 W/m2K + radiation, 25 C; vacuum: radiation only)

| body | env | all OK | T max C | conduit / core / sensor T max C | heat in W | max droop | bend util (100 g) | thermal-mismatch stress MPa (util) | mass g |
|---|---|---|---|---|---|---|---|---|---|
| grown seed 0 | air | YES | 36.0 | 35.9 / 36.0 / 35.9 | 0.370 | 0.0000 | 0.161 | 3.36 (0.34) | 2.21 |
| grown seed 0 | vacuum | NO | 56.8 | 56.7 / 56.8 / 56.7 | 0.370 | 0.0000 | 0.161 | 9.81 (0.98) | 2.21 |
| grown seed 1 | air | YES | 35.9 | 35.9 / 35.9 / 35.9 | 0.370 | 0.0000 | 0.161 | 3.36 (0.34) | 2.21 |
| grown seed 1 | vacuum | NO | 56.7 | 56.7 / 56.7 / 56.7 | 0.370 | 0.0000 | 0.161 | 9.80 (0.98) | 2.21 |
| grown seed 2 | air | YES | 35.9 | 35.8 / 35.9 / 35.8 | 0.369 | 0.0000 | 0.161 | 3.35 (0.33) | 2.21 |
| grown seed 2 | vacuum | NO | 56.6 | 56.6 / 56.6 / 56.5 | 0.369 | 0.0000 | 0.161 | 9.77 (0.98) | 2.21 |
| grown seed 3 | air | YES | 35.9 | 35.9 / 35.9 / 35.8 | 0.370 | 0.0000 | 0.162 | 3.35 (0.34) | 2.22 |
| grown seed 3 | vacuum | NO | 56.7 | 56.6 / 56.7 / 56.6 | 0.370 | 0.0000 | 0.162 | 9.79 (0.98) | 2.22 |
| grown seed 4 | air | YES | 35.9 | 35.9 / 35.9 / 35.8 | 0.368 | 0.0000 | 0.160 | 3.35 (0.33) | 2.20 |
| grown seed 4 | vacuum | NO | 56.5 | 56.5 / 56.5 / 56.5 | 0.368 | 0.0000 | 0.160 | 9.75 (0.97) | 2.20 |
| grown seed 5 | air | YES | 36.0 | 35.9 / 36.0 / 35.9 | 0.371 | 0.0000 | 0.162 | 3.37 (0.34) | 2.22 |
| grown seed 5 | vacuum | NO | 56.9 | 56.8 / 56.9 / 56.8 | 0.371 | 0.0000 | 0.162 | 9.84 (0.98) | 2.22 |
| regrown: anterior amputation (incl. organizer A) | air | YES | 35.8 | 35.8 / 35.8 / 35.7 | 0.368 | 0.0000 | 0.161 | 3.33 (0.33) | 2.21 |
| regrown: large anterior amputation (incl. A) | air | YES | 35.9 | 35.8 / 35.9 / 35.8 | 0.370 | 0.0000 | 0.162 | 3.36 (0.34) | 2.21 |
| regrown: core excision (incl. organizer C) | air | YES | 35.9 | 35.9 / 35.9 / 35.8 | 0.370 | 0.0000 | 0.161 | 3.35 (0.34) | 2.21 |
| regrown: posterior wedge (incl. organizer P) | air | YES | 35.9 | 35.8 / 35.9 / 35.8 | 0.370 | 0.0000 | 0.162 | 3.36 (0.34) | 2.22 |

Max total core power before any block exceeds its limit: air 0.442 W, vacuum 0.091 W, air without heat pipes 0.373 W (limit set by the 40 C vanadium electrolyte).
Power ablation, no copper traces: max droop 1.14, collapsed=True, power OK=False.
No heat pipes (air, 0.30 W): T max 37.8 C, OK=True, violations {}.

## Core regeneration fix

**Root cause (diagnosed, not guessed).** After the core is cut out, organizer C was re-derived from the A/P balance point (score min(A, P)). That point is not where the original core was: A and P sit on the ring band at slightly different radii (A 13 blocks out, P about 11 blocks out and 2 rows off the axis), so the balance point is about 1.4 blocks away from the original core block, at (28,30) instead of (29,29). Every C-dependent tissue (core, shell, the body outline) then shifts with it, while the A-dependent ones (sensor, radiator) stay put, so neither raw nor +/-2-cell-aligned similarity can recover (0.833 / 0.879). Instrumented oracle runs settle it. Re-placing C by hand right after the cut at the original block gives 0.974, which is the noise floor. Placing it at (28,30) gives 0.854, and at (27,30) 0.766. So the delay, hole closure, wrong tissue locking in and symmetry breaking are not the problem. The problem is *position information*: A and P do not carry enough of it.

**Change (local only, genome unchanged).**
1. *Positional memory*: every block keeps a slow memory of the core-signal level it reads (time constant ~330 steps). When the core signal disappears the memory freezes, like positional memory in salamander limb cells.
2. *Intercalation*: a block with no memory of its own (born into the wound, or a wear replacement) takes `min over neighbours of memory x exp(+step/lambda)`, which means "I can be at most one step closer to the core than my neighbour says". It is a local relay, the mirror image of the existing signal relay. Filling a convex wound inward from a rim of remembering blocks recovers exp(-distance/lambda) to the *old* core position, and its peak is the old core block.
3. A missing or settling core organizer forms at, and hill-climbs to, the peak of that estimate (with min(A, P) kept as a 1e-3 tie-breaker/fallback).
No coordinates, no stored pattern, and no target are involved: each block stores one scalar, its own last reading. The genome stays **131 bytes, byte-for-byte identical** (sha unchanged), and the three error-correction layers are untouched.

Two bugs were found and fixed on the way. (a) The memory first read the slow organizer average, which starts at 3% of the true value for a newborn block, so it now takes the first raw reading. (b) Remembering blocks must not be overwritten by the intercalated minimum, or a few blocks with low readings drag the whole estimate down.

**Before -> after** (same seeds, same damage):

| case | before | after |
|---|---|---|
| core excision, body seed 0 (sim, mean last 50 steps) | 0.833 (aligned 0.879), C re-formed at (28,30) | **0.974** (aligned 0.974), C at (29,29); noise floor 0.976 -> recovered in 105 steps |
| core excision, body seeds 1 / 2 / 3 | 0.834 / 0.824 / 0.756 (memory OFF) | **0.970 / 0.976 / 0.963** (all within 0.03 of their floors) |
| posterior wedge (P lost) | 0.944, not recovered | 0.944, not recovered (unchanged, see limitation) |
| anterior / large anterior amputation | 0.972 / 0.968 | 0.972 / 0.968 (identical) |
| growth, scaling, morph | see sections 1, 4, 5 | identical numbers |
| error-correction ablation | tumour incidence table | identical tumour incidence in all 54 cells; fidelity differs by <=0.01 only in runs that were already tumours |

**Posterior wedge: same family of problem, not fixed.** An oracle run that re-places P on its original block gives 0.976, and one cell off gives 0.952. The re-derived P lands 1 block off because its score ln C - ln A is flat along the A-core axis, so its radial position is degenerate. Positional memory cannot fix this the same way. The wound is open to the outside and the remembering blocks are all on one side of P, so the intercalated estimate becomes a plateau instead of a peak. Two alternatives were tried and rejected. A band-midpoint term in P's score put P on the axis but broke hole closure after core excision (0.0). Making P "farthest from A" gave 0.931. A longer P maturation window wrecked the large anterior amputation (0.746). The wedge stays at 0.944, 0.03 short of the floor.


Figures: figures/1_growth_sequence.png, figures/2_regeneration.png, figures/3_error_correction_ablation.png, figures/3b_mutation_examples.png, figures/4_scaling.png, figures/5_morph.png, figures/6_physics.png

Runtime 349 s