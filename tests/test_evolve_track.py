"""Tests for the evolved-rule comparison track (evolve_track/). Small, fast configurations."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import genome as gn  # noqa: E402
from evolve_track import protocol as P  # noqa: E402
from evolve_track import space as S  # noqa: E402
from evolve_track.evolve import EvoConfig, run_evolution  # noqa: E402

TINY = dict(lam=3.0, grow_steps=50, regen_steps=60, train_seeds=(1,), cuts=(P.CORE,),
            mut_steps=40, mut_margin=8, min_tissue=1, mu=2, lam_off=3, generations=2, workers=1)


def test_wildtype_decode_roundtrip_and_is_point_in_space():
    p = S.decode_params(gn.WILDTYPE)
    assert S.encode_params(p) == gn.WILDTYPE
    assert S.is_valid(gn.WILDTYPE)
    assert len(p.rules) == 12 and p.apoptosis == gn.APOPTOSIS_ON
    assert S.WILDTYPE.neighbor_clear is True
    assert S.describe_diff(S.WILDTYPE, S.WILDTYPE) == []


def test_mutation_and_crossover_preserve_validity_and_length():
    rng = np.random.default_rng(0)
    c = S.WILDTYPE
    seen_change = False
    for _ in range(500):
        c2 = S.mutate(c, rng, strength=2.0)
        assert len(c2.genome) == gn.GENOME_LEN == 131
        assert S.is_valid(c2.genome) and gn.crc_ok(c2.genome)
        seen_change |= c2.genome != gn.WILDTYPE
        x = S.crossover(c2, S.WILDTYPE, rng)
        assert S.is_valid(x.genome)
        c = c2
    assert seen_change


def test_mutation_is_deterministic_for_fixed_seed():
    a = [S.mutate(S.WILDTYPE, np.random.default_rng(7)) for _ in range(3)]
    assert len({x.key() for x in a}) == 1


def test_sustained_rule():
    floor = 0.95
    good = [0.5] * 50 + [0.94] * 100
    s = P.sustained(good, floor)
    assert s["sustained"] and s["steps_to_sustained"] == 51
    blip = [0.5] * 100 + [0.95] * 5 + [0.5] * 45           # brief crossing only
    assert not P.sustained(blip, floor)["sustained"]
    flick = [0.5] * 100 + [0.95, 0.5] * 25                  # 25/50 above -> not sustained
    assert not P.sustained(flick, floor)["sustained"]


def test_fitness_never_uses_self_reference(monkeypatch):
    """Every similarity() call made while scoring must compare against the independent reference
    (wild type, REF_SEED) or the pre-cut pattern for the context-only number; and the reference seed
    can never be a scored body."""
    ref = P.ref_body(3.0, 50)
    calls = []
    real = P.similarity

    def spy(p, r):
        calls.append(r is ref)
        return real(p, r)
    monkeypatch.setattr(P, "similarity", spy)
    r = P.evaluate_body(S.WILDTYPE, 1, 3.0, 50, 60, [P.CORE], ref, min_tissue=1)
    # growth + twin/cut honest curves use ref; context curves are the same count again
    n_ref = sum(calls)
    assert n_ref == 1 + 2 * 60 and len(calls) == n_ref + 2 * 60
    with pytest.raises(AssertionError):
        P.evaluate_body(S.WILDTYPE, P.REF_SEED, 3.0, 50, 60, [P.CORE], ref, min_tissue=1)
    with pytest.raises(AssertionError):
        EvoConfig(train_seeds=(0, 1))
    with pytest.raises(AssertionError):
        EvoConfig(train_seeds=(42,))
    # fitness is computed only from the reference-based fields
    F1, _ = P.fitness_from_bodies([r], 60)
    r2 = dict(r, cuts={k: dict(v, context_self_last50=0.0, context_self_first_crossing=None)
                       for k, v in r["cuts"].items()}, twin_floor_self_context=0.0)
    F2, _ = P.fitness_from_bodies([r2], 60)
    assert F1 == F2


def test_guards_flag_trivial_bodies():
    ref = P.ref_body(3.0, 50)
    tiny = np.zeros_like(ref); tiny[ref.shape[0] // 2, ref.shape[1] // 2] = 1
    g = P.guards(tiny, ref, 1)
    assert not g["size_ok"] and not g["tissues_ok"]


def test_evaluation_deterministic_for_fixed_seed():
    ref = P.ref_body(3.0, 50)
    a = P.evaluate_body(S.WILDTYPE, 2, 3.0, 50, 60, [P.CORE], ref, min_tissue=1)
    b = P.evaluate_body(S.WILDTYPE, 2, 3.0, 50, 60, [P.CORE], ref, min_tissue=1)
    assert P.fitness_from_bodies([a], 60) == P.fitness_from_bodies([b], 60)


def test_tiny_end_to_end_evolution_runs_and_returns_valid_genome():
    ec = EvoConfig(**TINY)
    r1 = run_evolution(ec, log=lambda s: None)
    best = r1["parents"][0]
    assert S.is_valid(best.genome) and len(best.genome) == 131
    assert len(r1["history"]) == ec.generations + 1
    hist = [h["best"] for h in r1["history"]]
    assert all(b2 >= b1 - 1e-12 for b1, b2 in zip(hist, hist[1:]))   # (mu+lambda) is elitist
    r2 = run_evolution(ec, log=lambda s: None)
    assert r2["parents"][0].key() == best.key()                      # deterministic
    assert r2["history"][-1]["best"] == r1["history"][-1]["best"]
