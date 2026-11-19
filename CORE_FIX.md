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
