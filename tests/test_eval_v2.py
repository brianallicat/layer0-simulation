"""Tests for the v2 evaluation protocol (scoring only; science untouched)."""
import hashlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eval_v2 as ev  # noqa: E402
import genome as gn  # noqa: E402


# ---------------------------------------------------------------- 1. no self-reference
@pytest.mark.parametrize("mode", ["heldout_ref", "all_pairs", "same_size"])
def test_scaling_pairs_never_self_reference(mode):
    pairs = ev.scaling_pairs(ev.SCALE_LAMS, ev.SCALE_SEEDS, mode=mode)
    assert pairs
    assert all(a != b for a, b in pairs)
    if mode == "heldout_ref":
        # the reference body is never itself scored, at any scale
        assert all(a[1] != ev.SCALE_REF[1] for a, _ in pairs)
        assert all(b == ev.SCALE_REF for _, b in pairs)
    else:
        assert all(a[1] != b[1] for a, b in pairs)   # never the same seed, even across scales


def test_scaling_eval_small_has_no_self_scores():
    s = ev.scaling_eval(lams=(4.0, 7.0), seeds=[0, 1, 2], steps=30, twin_steps=55, workers=1)
    for row in s["rows"]:
        assert str(ev.SCALE_REF[1]) not in row["shape_vs_heldout_ref_final_step"]["by_seed"]
        vals = row["shape_vs_heldout_ref_final_step"]["by_seed"].values()
        assert all(v < 1.0 for v in vals)


# ---------------------------------------------------------------- 2. sustained rule
def test_sustained_rule_clean_recovery():
    c = [0.5] * 100 + [0.96] * 300
    r = ev.sustained(c, 0.97)
    assert r["recovered_sustained"] and r["recovered_mean_only"]
    assert r["first_crossing_step"] == 101 and r["settled_step"] == 101
    assert r["steps_within_last50"] == 50


def test_sustained_rule_brief_crossing_fails():
    # crosses once at step 156 then falls back (the posterior-wedge pattern)
    c = [0.90] * 155 + [0.95] * 10 + [0.92] * 235
    r = ev.sustained(c, 0.97)
    assert r["first_crossing_step"] == 156
    assert r["settled_step"] is None
    assert not r["recovered_sustained"] and not r["recovered_mean_only"]


def test_sustained_rule_mean_ok_but_flickering_fails():
    # last-50 mean above threshold, but only 30 of 50 steps within: mean-only passes, sustained fails
    last = [0.99] * 30 + [0.92] * 20          # thr 0.94; mean = 0.962
    c = [0.5] * 350 + last
    r = ev.sustained(c, 0.97)
    assert r["recovered_mean_only"]
    assert r["steps_within_last50"] == 30
    assert not r["recovered_sustained"]


def test_sustained_rule_settle_step_after_dip():
    c = [0.95] * 100 + [0.90] + [0.95] * 299
    r = ev.sustained(c, 0.97)
    assert r["first_crossing_step"] == 1 and r["settled_step"] == 102
    assert r["recovered_sustained"]


def test_sustained_rule_exactly_40_of_50_passes():
    c = [0.5] * 350 + [0.97] * 40 + [0.939] * 10   # thr 0.94; mean 0.9638
    r = ev.sustained(c, 0.97)
    assert r["steps_within_last50"] == 40 and r["recovered_sustained"]


def test_boot_stats_labels_and_ci():
    s = ev.boot_stats([0.9, 0.95, 0.97, 0.99])
    assert s["n"] == 4 and s["ci95"][0] <= s["mean"] <= s["ci95"][1]
    assert abs(s["sd"] - np.std([0.9, 0.95, 0.97, 0.99], ddof=1)) < 1e-12


# ---------------------------------------------------------------- 3. catalog determinism
def test_catalog_deterministic_for_fixed_seed():
    def run():
        bases = dict([ev._grow_base((5, 4.0, 60))])
        jobs = ev.catalog_jobs(bases, steps=25)
        return [ev._cut_job(j) for j in jobs]
    a, b = run(), run()
    assert len(a) == len(b) > 10
    for x, y in zip(a, b):
        assert (x["name"], x["cond"]) == (y["name"], y["cond"])
        assert x["curve"] == y["curve"]
        assert x["tissue_iou_final_step"] == y["tissue_iou_final_step"]


def test_catalog_has_all_categories_and_nonempty_masks():
    seed, base = ev._grow_base((2, 7.0, 120))
    m = ev.catalog_masks(base, seed)
    cats = {c for _, c, _ in m}
    assert cats == {"structured (original 4)", "angled/depth amputation", "two organizers lost",
                    "random disc (NCA-style)"}
    assert all(mask.any() and not mask.all() for _, _, mask in m)
    assert len({n for n, _, _ in m}) == len(m)


def test_science_untouched():
    # genome bytes and sim rules are not modified by eval_v2
    assert hashlib.sha256(gn.WILDTYPE).hexdigest()[:16] == "df78d46b265b2269"
    src = open(os.path.join(os.path.dirname(ev.__file__), "eval_v2.py")).read()
    assert "setattr(World" not in src and "World." not in src.replace("World(", "")
