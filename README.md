# Layer 0: the "source code" of a self-building body

This folder is a working simulation of Layer 0 of Brian's architecture. It covers how identical blocks, all carrying the same tiny genome, grow a body from one seed, repair it, resist copy errors, and change shape, using only local rules. A first materials and physics layer now checks whether that body would stay within thermal, power and strength limits if it were built from real materials.

Everything here is **simulation**. Nothing has been fabricated.

## How it works, in plain language

* **One recipe, not a blueprint.** Every block carries the same 131-byte genome: a 7-byte header, 12 rules of 10 bytes each, and a 4-byte checksum. No block knows its coordinates, the body's size, or what the finished body looks like.
* **Signals instead of addresses.** Three blocks act as *organizers* and broadcast a signal that fades with distance (relayed block to block through living tissue only): **C** (core), **A** (front) and **P** (back). Each block reads the three strengths where it sits.
* **The 12 rules** (`python3 genome.py` prints them):
  * R01-R03: when to become an organizer. C forms where A and P are both strong. A and P form in a ring band around C; A goes toward a weak outside cue, and P goes to the far end of the A-core axis.
  * R04-R06: grow when the core signal is strong. A wound-healing rule divides in the A/P mid-zone. Blocks starve when the core signal is lost.
  * R07-R12, where the first match wins: **core** (strong C), **sensor** (strong A), **radiator** (far from A), **conduit** (between A and P), **shell** (weak C, at the edge), and otherwise **scaffold**.
* **Error correction, three layers.** (1) Each copy is proofread when a block divides, fixing 90% of byte errors. (2) Each block holds 3 copies and runs their majority vote. (3) A CRC-32 "self marker": a block whose marker fails self-destructs, and if its self-destruct gene is broken, healthy neighbours clear it.
* **Positional memory (new).** Each block slowly remembers the core-signal level it reads. If the core is destroyed, blocks keep that memory. New blocks in the wound fill in a value from their neighbours ("at most one step closer to the core than my neighbour"), and the core organizer re-forms at the peak of that estimate, which is the original core block. This is local, stores one number per block, and is modelled on positional memory in salamander regeneration.
* **Materials (new, `materials.py`).**

| tissue | material stack (volume fractions are design assumptions) | limiting factor |
|---|---|---|
| scaffold | graphene-nanoplatelet/epoxy composite + 15% micro heat-pipe channels + 3% Cu traces | composite strength 60 MPa, 120 C |
| shell | same composite + heat pipes + 10% ta-C diamond-like-carbon skin | 120 C |
| conduit | silica optical fibre (data) + vanadium redox electrolyte channel (charge, heat) + Cu current collectors | **electrolyte limit 40 C** |
| core | silicon / photonic compute + heat-pipe spreader | Si 85 C |
| sensor | Si (VIS/NIR) + InGaAs (SWIR) + VOx microbolometer behind Ge/ZnSe optics | InGaAs 70 C, ZnSe 55 MPa |
| radiator | pyrolytic graphite sheet, high-emissivity face | PGS 20 MPa |

Blocks are taken to be 1 mm cubes, one block thick, so the body is about 38 mm across and weighs 2.2 g. Literature values and sources are in `materials.py` (`MAT`, `REFS`).

## How to run

```
python3 -m pytest -q tests/        # 18 tests, ~2 min
python3 run_all.py                 # every experiment, figures/, results.json, RESULTS.md (~6 min, 8 cores)
python3 deliverables/make_illustrations.py
```

## Results, sorted by how much we actually know

### Verified (by this simulation; numbers from `RESULTS.md`)
* A 131-byte genome grows a 1,140-block, six-tissue body from one seed. Mean pattern similarity across 10 random seeds is 0.946, against a noise floor of 0.981.
* **Regeneration.** Front amputations (29% and 39%, organizer A included) recover to 0.972 and 0.968, against a 0.976 floor. **Core excision (26%, organizer C included) now recovers to 0.974** (it was 0.833 before the fix). That holds on 4 of 4 independently grown bodies (0.963-0.976). With positional memory switched off, all four fail again (0.756-0.834).
* Without organizer re-derivation, regeneration fails (0.48 front; 0.00 core, where the body starves).
* **Error correction.** With all three layers on there were 0 tumours at every copy-error rate up to 10% per byte. With all three off, every run became a tumour from 0.1%. Any single layer on its own fails by 1%.
* **Scaling.** The same 131 bytes grow 414-, 1,140- and 2,690-block bodies with the same proportions (core about 5%, radiator about 30%).
* **Morph.** Moving the core source by 6 blocks makes the body rebuild around it (0.956). A stored blueprint scores 0.330.
* **Physics** (1 mm blocks, still air at 25 C): with a 0.30 W core and 0.07 W of sensors, every grown body (6 seeds) and every regrown body stays within its limits. The hottest point is 36 C, supply droop is under 0.1%, bending at 100 g is at 16% of allowable, and thermal-mismatch stress is at 34% of allowable. The maximum core power in air is 0.44 W, limited by the 40 C electrolyte.
* **Physics failures we found, kept as tests.** (1) Without the copper traces, the graphene composite (~10 S/m) cannot carry the power and the supply collapses. Copper was not in the original material list; it was added. (2) In vacuum (radiation only) the body reaches 57 C and the electrolyte goes over its limit. The vacuum ceiling is 0.09 W of core power. (3) The electrolyte in the conduits holds only about 10 s of energy at full load, so it can carry energy but cannot store it.

### Roadmap (the next things to build or test)
* Fix the posterior-wedge case (0.944, 0.03 short of the floor). The cause is known: P's position along its axis is not uniquely defined.
* Couple the physics back into growth, for example by letting heat or voltage readings act as a fourth signal so radiators grow where heat builds up.
* Model heat and power in 3D, with real latch contact resistance and heat-pipe capillary limits.
* Work out an external charging path or reservoir for the electrolyte.
* Model a moving body: dynamic loads, fatigue, and latch wear.

### Concept (sound in principle, but untested here)
* Gradient fields made physical as chemical concentration, light intensity at one wavelength, or a magnetic bias.
* Sub-millimetre blocks that latch together and pass power, data and heat.
* A fixed, air-gapped core inside a body that changes shape.

### Speculative
* Building blocks this small with all of these materials combined in one block.
* Biological-style regeneration of a full machine in the field.
* Scaling the same genome up to large bodies, beyond the ~2,700 blocks tested here.

## Limitations (honest list)
* 2D grid, discrete time, idealised signal relay; the materials physics is steady-state and one block thick.
* Physics does not influence growth, it is evaluated afterward.
* Positional memory, intercalation and organizer settling are modelled as block "hardware" mechanics outside the genome, as the existing signal relay already was. They add no genome bytes, but they are extra machinery a real block would need.
* Volume fractions, the 1 mm block size, the latch strength (20 MPa) and the copper traces are design assumptions.
* Thermal limits use typical commercial ratings.
* The posterior wedge does not fully recover.

## Files
`sim.py` world and organizer mechanics · `genome.py` the 131-byte genome · `experiments.py` all experiments · `materials.py` materials and physics · `metrics.py` scoring (never visible to blocks) · `tests/test_layer0.py` 18 tests · `RESULTS.md` all tables, including the "Core regeneration fix" section · `figures/` research figures · `deliverables/` illustrations, the Word report and the explainer video.
