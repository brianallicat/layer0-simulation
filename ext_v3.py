"""Layer 0 extensions v3 (items 2 and 3) - OPTIONAL switches, OFF by default.

This module subclasses sim.World; it does not edit genome.py or sim.py. With both switches
off, WorldV3 runs exactly the base World code path (bit-identical patterns; see tests).

Item 2 - `mem_v3`: positional memory for all three organizers (C, A, P) + intercalation
  * every block keeps one remembered level per organizer field (C, A, P). The memory is set from
    the mean of its first `mem_init_n` readings (instead of a single noisy reading) and then
    follows the reading at `mem_rate_v3` (faster than sim.py's 0.003, which lagged the true level
    by ~12% on average for P at t=200; safe because the memory is frozen during re-derivation). Memory of field t is frozen while
    organizer t is being re-derived, so a misplaced settling organizer cannot overwrite it.
  * when a mature organizer of type t is lost, blocks with no memory (wound fillers) take an
    intercalated estimate  min_nb mem(nb) * exp(+step/lambda)  - 'at most one step closer to the
    source than my neighbour'. For an open wound (posterior wedge) this runs OUTWARD from the wound
    edge and is left UNCAPPED, so it crosses 1.0 exactly at the old source position.
  * the re-derived organizer t forms at, and hill-climbs to, the block whose estimate is closest
    to 1 (score -|ln est|): along the A-core axis this has a unique optimum, which removes the
    radial degeneracy of P's original score (ln C - ln A, kept as a 1e-3 tie-breaker).
  * a block whose remembered/estimated core level is >= `mem_qual_c` may qualify as core organizer
    even if the A/P overlap rule does not fire (so the core can re-form when A/P sit off-axis).
  Each block still stores only scalars (its own past readings); nothing stores a pattern.

Item 3 - `scale_field`: expansion-repression scaling (Ben-Zvi & Barkai 2010, well-mixed limit)
  * an expander level s (one scalar: the expander is assumed to diffuse much faster than the
    morphogens) is produced by blocks whose patterning-C reading is below theta (repressed by the
    morphogen) and removed at a constant rate: ds/dt = eta * (frac_low - phi).
  * the expander widens the PATTERNING readings: the fate and organizer rules read C,A,P with
    range lambda*s (computed as reading**(1/s), exact for the relayed exp(-path/lambda) field);
    the growth rules (DIVIDE / STARVE) keep reading the raw field, so body SIZE is still set by
    lambda and only proportions are adapted.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import genome as gn
from sim import Config, World, shift

_L = float(np.log(1 / 0.08))   # genome units: body radius / lambda


@dataclass
class ConfigV3(Config):
    # ---- item 2 ----
    mem_v3: bool = False
    mem_init_n: int = 16          # memory = mean of the first n readings, then slow tracking
    mem_qual_c: float = 0.6       # remembered core level that qualifies a block as core organizer
    mem_conf_n: int = 40          # a memory counts as an intercalation anchor only after this many readings
    mem_floor: float = 0.0        # ... and only if it remembers at least this level (low levels are unreliable)
    over_w: float = 4.0           # penalty weight on estimates ABOVE 1 (beyond the old source the
                                  # outward estimate rises slowly, so -|ln est| alone is biased outward)
    mem_rate_v3: float = 0.05     # tracking rate while the organizer is in place (frozen while re-deriving)
    c_score: str = "unit"         # core score while re-deriving: "unit" = -|ln est|, "peak" = min(est, 1)
    # ---- item 3 ----
    scale_field: bool = False
    edge_frac: float = 0.95       # patterning-C at fraction edge_frac of the radius should be f(edge_frac)
    scale_eta: float = 0.05       # expander production/removal rate
    s_min: float = 1.0
    scale_orgs: bool = False      # also feed the scaled readings to the ORGANIZE_* rules (tried: worse)
    s_max: float = 1.6


class WorldV3(World):
    def __init__(self, cfg: ConfigV3):
        super().__init__(cfg)
        N = self.N
        self.m3 = np.full((3, N, N), np.inf)        # remembered C, A, P levels (inf = none)
        self.n3 = np.zeros((3, N, N), np.int16)     # readings averaged into the memory so far
        self.est3 = np.full((3, N, N), np.inf)      # intercalated estimates (only while re-deriving)
        self.rederiving = [False, False, False]     # organizer t lost after maturity, not yet re-matured
        self.ever_mature = [True, False, False]     # founder core is mature from the start
        self.s = 1.0                                # expander level (item 3)
        self.s_trace = []

    # ------------------------------------------------------------------ item 2: memory
    def _place(self, y, x, template, clean=False):
        super()._place(y, x, template, clean)
        if hasattr(self, "m3"):
            self.m3[:, y, x] = np.inf
            self.n3[:, y, x] = 0

    def kill(self, mask):
        super().kill(mask)
        if hasattr(self, "m3"):
            self.m3[:, mask] = np.inf
            self.n3[:, mask] = 0

    def sense(self):
        super().sense()
        if not self.cfg.mem_v3:
            return
        cfg = self.cfg
        for t in range(3):
            m, n = self.m3[t], self.n3[t]
            alive = self.alive
            if self.rederiving[t]:
                # frozen; wound fillers adopt the intercalated estimate
                adopt = alive & ~np.isfinite(m) & np.isfinite(self.est3[t])
                m[adopt] = self.est3[t][adopt]
                n[adopt] = cfg.mem_init_n          # adopted, not yet an anchor
                continue
            heard = alive & (self.fields[t] > 0)
            r = self.read[t]
            # first readings: running mean (the fast EMA reading `read` is used, as in sim.py)
            first = heard & (n < cfg.mem_init_n)
            k = n[first].astype(float) + 1
            m0 = np.where(np.isfinite(m[first]), m[first], 0.0)
            m[first] = m0 + (r[first] - m0) / k
            n[first] += 1
            upd = heard & ~first & np.isfinite(m)
            m[upd] += cfg.mem_rate_v3 * (r[upd] - m[upd])
            n[upd] = np.minimum(n[upd] + 1, 10000)
            m[~alive] = np.inf

    def _anchors(self, t):
        cfg = self.cfg
        return (self.alive & np.isfinite(self.m3[t]) & (self.n3[t] >= cfg.mem_conf_n)
                & (self.m3[t] >= cfg.mem_floor))

    def _estimate(self, t):
        """Anchors use their own memory; every other block (wound fillers, young blocks) uses the
        current intercalated estimate; a block with neither falls back to its own (weak) memory."""
        e = np.where(np.isfinite(self.est3[t]), self.est3[t], self.m3[t])
        return np.where(self._anchors(t), self.m3[t], e)

    def _intercalate3(self, t):
        alive = self.alive
        anc = self._anchors(t)
        R = np.where(anc, self.m3[t], np.inf)
        free = alive & ~anc
        if not np.isfinite(R).any():
            return np.full_like(R, np.inf)
        grow = {o: 1.0 / d for o, d in self.decay.items()}
        for _ in range(4 * self.N):
            new = R.copy()
            for o, g in grow.items():
                np.minimum(new, shift(R, *o, fill=np.inf) * g, out=new)
            new = np.where(free, new, R)
            if np.allclose(np.nan_to_num(new, posinf=9.0), np.nan_to_num(R, posinf=9.0), atol=1e-9):
                R = new
                break
            R = new
        return R

    def _update_states(self):
        cfg = self.cfg
        for t in range(3):
            present = (self.org == t + 1)
            win = cfg.mature_core if t == 0 else cfg.mature
            if present.any():
                age = self.t - self.org_born[present].max()
                if age > win:
                    self.ever_mature[t] = True
                    self.rederiving[t] = False
            elif self.ever_mature[t]:
                self.rederiving[t] = True

    # ------------------------------------------------------------------ item 3: patterning scale
    def _update_scale(self):
        cfg = self.cfg
        if not self.alive.any():
            return
        theta = 0.08 ** cfg.edge_frac
        phi = 1.0 - cfg.edge_frac ** 2
        cp = self.read[0][self.alive] ** (1.0 / self.s)
        frac_low = float((cp < theta).mean())
        self.s = float(np.clip(self.s + cfg.scale_eta * (frac_low - phi), cfg.s_min, cfg.s_max))
        self.s_trace.append(self.s)

    def evaluate(self):
        if not self.cfg.scale_field:
            return super().evaluate()
        self._update_scale()
        raw = self.read
        _, divide, starve, qual_raw = super().evaluate()    # growth rules: raw field (size set by lambda)
        self.read = np.where(raw > 0, raw, 0.0) ** (1.0 / self.s)
        try:
            fate, _, _, qual = super().evaluate()           # fate + organizer rules: patterning field
        finally:
            self.read = raw
        return fate, divide, starve, (qual if self.cfg.scale_orgs else qual_raw)

    # ------------------------------------------------------------------ organizers
    def organizers(self, qual):
        if self.cfg.mem_v3:
            self._update_states()
            for t in range(3):
                self.est3[t] = self._intercalate3(t) if self.rederiving[t] else np.inf
            if self.rederiving[0]:
                own = self._estimate(0)
                memq = self.alive & np.isfinite(own) & (own >= self.cfg.mem_qual_c)
                qual = dict(qual)
                qual[gn.ORG_C] = qual[gn.ORG_C] | memq
        return super().organizers(qual)

    def scores(self):
        sc = super().scores()
        if not self.cfg.mem_v3:
            return sc
        for t in range(3):
            if not self.rederiving[t]:
                continue
            # own memory where the block has one (anchor or not); intercalated estimate for wound fillers
            est = self._estimate(t)
            have = self.alive & np.isfinite(est) & (est > 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                le = np.log(np.where(have, est, 1.0))
                closeness = np.where(have, -np.maximum(-le, self.cfg.over_w * le), -50.0)
            base = sc[t + 1]
            base = np.where(np.isfinite(base), base, -50.0)
            if t == 0 and self.cfg.c_score == "peak":
                closeness = np.where(have, np.minimum(np.where(have, est, 0.0), 1.0), -50.0)
            sc[t + 1] = closeness + 1e-3 * np.clip(base, -50, 50)
        return sc


def make_world(cfg_kwargs=None, **kw):
    d = dict(cfg_kwargs or {})
    d.update(kw)
    return WorldV3(ConfigV3(**d))
