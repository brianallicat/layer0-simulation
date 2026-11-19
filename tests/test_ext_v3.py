"""Tests for the optional v3 extensions (ext_v3.py). Switches OFF must reproduce the baseline exactly."""
import copy
import hashlib
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import genome as gn  # noqa: E402
from experiments import damage_masks  # noqa: E402
from ext_v3 import ConfigV3, WorldV3  # noqa: E402
from metrics import similarity  # noqa: E402
from sim import Config, World  # noqa: E402

# sha256 of the baseline files at the time ext_v3 was written: the extension must not edit them
BASELINE_SHA = {
    "sim.py": "699d7ba514a6d17b9d1739af16cde97b9461f29156176e8dc8cfa7ecf765337f",
    "genome.py": "2cc610d0a4554f0a3bfab6abbe4626ddeeca43938482892675eef3e8380bf304",
    "eval_v2.py": "0cdba8dfb2564494c73a4f9da20b77531050b40d5249b06779b883e7cdec9fd8",
    "experiments.py": "677c99e457688cd1e73d15b35d13d640896d1b3f13e19ef99622dd0cd47b81d3",
    "results.json": "648c8946e5691ffa1cf2b441663fab59beafa83d01d4517341ff947c8d93dfaa",
    "results_eval_v2.json": "e05814308ae501599168d4cc000ff4643027584093cb32aa3140ea5bb0f2e11b",
}


@pytest.mark.parametrize("fname", sorted(BASELINE_SHA))
def test_baseline_files_untouched(fname):
    with open(os.path.join(HERE, fname), "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == BASELINE_SHA[fname]


def test_switches_default_off_and_genome_unchanged():
    c = ConfigV3()
    assert c.mem_v3 is False and c.scale_field is False
    assert hashlib.sha256(gn.WILDTYPE).hexdigest()[:16] == "df78d46b265b2269"
    w = WorldV3(c)
    assert not any("ref" in a or "target" in a or "blueprint" in a for a in vars(w))


def _state(w):
    return (w.alive.copy(), w.tissue.copy(), w.org.copy(), w.G.copy(), w.mem.copy(), w.t)


@pytest.mark.parametrize("lam,seed", [(4.0, 3), (7.0, 0)])
def test_switch_off_reproduces_baseline_exactly(lam, seed):
    a = World(Config(lam=lam, seed=seed)).run(130)
    b = WorldV3(ConfigV3(lam=lam, seed=seed)).run(130)
    for x, y in zip(_state(a), _state(b)):
        assert np.array_equal(x, y)
    # and through a damage + regeneration run (core excision, the memory code path)
    m = damage_masks(a)["core excision (incl. organizer C)"]
    ca, cb = [], []
    for base, curve in ((a, ca), (b, cb)):
        w = copy.deepcopy(base)
        w.rng = np.random.default_rng(12345)
        w.kill(m)
        for _ in range(40):
            w.step()
            curve.append(similarity(w.pattern(), base.pattern()))
    assert ca == cb


def test_switch_off_with_copy_errors_reproduces_baseline():
    kw = dict(lam=4.0, seed=1, mu=0.03, turnover=0.01)
    a = World(Config(**kw)).run(80)
    b = WorldV3(ConfigV3(**kw)).run(80)
    assert np.array_equal(a.pattern(), b.pattern()) and a.stats == b.stats


def test_item2_memory_does_not_change_growth():
    a = World(Config(lam=7.0, seed=4)).run(200)
    b = WorldV3(ConfigV3(lam=7.0, seed=4, mem_v3=True)).run(200)
    assert np.array_equal(a.pattern(), b.pattern())


def test_item2_runs_end_to_end_and_reforms_core():
    base = WorldV3(ConfigV3(lam=7.0, seed=4, mem_v3=True)).run(200)
    m = damage_masks(base)["core excision (incl. organizer C)"]
    w = copy.deepcopy(base)
    w.rng = np.random.default_rng(12345)
    w.kill(m)
    w.run(150)
    assert w.alive.sum() > 0.8 * base.alive.sum()
    assert (w.org == 1).sum() == 1
    assert np.isfinite(w.m3[0][w.alive]).mean() > 0.9     # memories exist (one scalar per field per block)


def test_item3_runs_end_to_end():
    w = WorldV3(ConfigV3(lam=4.0, seed=1, scale_field=True)).run(150)
    assert w.alive.sum() > 300
    assert w.cfg.s_min <= w.s <= w.cfg.s_max and w.s > 1.0   # expander has expanded the patterning range
    assert len(w.s_trace) == 150
    both = WorldV3(ConfigV3(lam=4.0, seed=1, scale_field=True, mem_v3=True)).run(60)
    assert both.alive.sum() > 0
