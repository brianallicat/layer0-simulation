"""Layer 0 world: a 2D grid of blocks that grow from one seed using only the genome
rules, local morphogen readings, and local neighbour interactions.

What is "hardware/physics" (fixed, not in the genome, honest list):
  * field relay: every living block re-emits the strongest neighbouring signal,
    attenuated by exp(-distance/lambda) (lambda = medium attenuation length).
    Signals only travel through living tissue (a wound blocks them).
  * sensor noise on readings + a short exponential moving average (local memory).
  * organizer mechanics: de-novo formation needs the block to be the local maximum of
    a score within R_org (lateral inhibition), organizers hand their identity to a
    better-scoring neighbour (hill-climb), and two same-type organizers within
    R_inhib resolve to one. Scores: A = external cue E, P = -A, C = min(A, P).
  * positional memory + intercalation (added for core regeneration): each block keeps a slow
    memory of the core-signal level it reads; when the core is lost, blocks with no memory
    (wound fillers, wear replacements) intercalate a value from neighbours (see intercalate()).
    A missing / settling core organizer forms at, and climbs to, the peak of that estimate.
  * random wear (turnover), copy errors at division, the error-correction machinery.
Nothing in the world ever stores, consults, or copies a target pattern.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import numpy as np

import genome as gn

# relay neighbourhood: primitive offsets within radius 5 (near-Euclidean path lengths)
_RELAY_R = 5
_OFFS = [(dy, dx) for dy in range(-_RELAY_R, _RELAY_R + 1) for dx in range(-_RELAY_R, _RELAY_R + 1)
         if (dy, dx) != (0, 0) and dy * dy + dx * dx <= _RELAY_R ** 2 and np.gcd(abs(dy), abs(dx)) == 1]
_NB8 = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
ORG_NAMES = {1: "C", 2: "A", 3: "P"}


def shift(a: np.ndarray, dy: int, dx: int, fill=0):
    """out[y,x] = a[y-dy, x-dx]  (value arriving from the neighbour at offset -(dy,dx))."""
    out = np.full_like(a, fill)
    H, W = a.shape[-2:]
    ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    out[..., yd, xd] = a[..., ys, xs]
    return out


@dataclass
class Config:
    lam: float = 7.0              # medium attenuation length (cells) -> sets body scale
    size: int | None = None       # grid side; default from lam
    margin: int = 10
    seed: int = 0
    noise: float = 0.03           # multiplicative sensor noise
    ema: float = 0.8              # reading smoothing (local memory)
    ema_slow: float = 0.97        # organizer input smoothing
    turnover: float = 0.004       # random wear death prob / block / step
    mu: float = 0.0               # copy error rate per byte per copy
    redundancy: int = 3           # genome copies per block (1 = ablated)
    proofread: bool = True
    proof_eff: float = 0.9          # fraction of copy errors the proofreader catches
    apoptosis: bool = True        # self-check suicide of blocks with a corrupted genome
    neighbor_clear: bool = True   # neighbours clear blocks displaying a bad self-marker
    p_clear: float = 0.25
    R_org: int = 3
    R_climb: int | None = None    # default = ceil(lam)
    R_inhib: float | None = None  # default = lam
    p_form: float = 0.05
    hysteresis: float = 0.005     # relative score gain needed before an organizer moves
    mature: int = 60              # steps a newly formed A/P organizer may settle (hill-climb) before it fixes
    mature_core: int = 150        # same for a re-formed core organizer
    cue_angle: float = 0.0        # direction of the weak external cue E (radians)
    p_mid_beta: float = 0.5
    p_score: str = "axis"         # P organizer score: "axis" = ln C - ln A (default); "far" = -ln A (tried, worse)
    mem_rate: float = 0.003       # how fast positional memory follows the current core signal
    pos_memory: bool = True       # positional memory + intercalation for re-deriving the core organizer
    organizer_regen: bool = True  # allow de-novo organizer formation after the founder stage
    genome: bytes = field(default_factory=lambda: gn.WILDTYPE)


class World:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        R_est = cfg.lam * np.log(1 / 0.08) + 2
        self.N = cfg.size or int(2 * R_est + 2 * cfg.margin)
        N, L, k = self.N, gn.GENOME_LEN, cfg.redundancy
        self.alive = np.zeros((N, N), bool)
        self.tissue = np.zeros((N, N), np.int8)
        self.org = np.zeros((N, N), np.int8)
        self.copies = np.zeros((N, N, k, L), np.uint8)
        self.G = np.zeros((N, N, L), np.uint8)       # expressed (consensus) genome
        self.valid = np.zeros((N, N), bool)          # self-marker (CRC) intact
        self.starve = np.zeros((N, N), np.int16)
        self.read = np.zeros((3, N, N))              # smoothed readings C, A, P
        self.slow = np.zeros((3, N, N))              # slow-averaged readings used by organizers
        self.fields = np.zeros((3, N, N))
        self.org_born = np.zeros((N, N), np.int32)
        self.org_off = np.zeros((N, N), np.int16)
        self.qstab = np.zeros((3, N, N))
        self.mem = np.full((N, N), np.inf)          # positional memory: last slow C level (inf = none)
        self.recon = np.full((N, N), np.inf)        # intercalated C estimate (see intercalate())
        self._formed = {1: True}
        self.t = 0
        self.wt = np.frombuffer(cfg.genome, np.uint8).copy()
        yy, xx = np.mgrid[0:N, 0:N]
        self.yy, self.xx = yy, xx
        self.jitter = self.rng.random((N, N)) * 1e-9
        self.set_cue(cfg.cue_angle)
        self.stats = dict(divisions=0, copy_errors_expressed=0, apoptosis=0, cleared=0)
        c = N // 2
        self._place(c, c, self.wt, clean=True)      # the single (error-free) seed ...
        self.org[c, c] = 1                          # ... is the founder core organizer
        self.org_born[c, c] = -10**6               # founder: fixed from the start
        if cfg.R_climb is None:
            cfg.R_climb = int(np.ceil(cfg.lam))
        if cfg.R_inhib is None:
            cfg.R_inhib = cfg.lam
        # middle (geometric) of the C band in which this genome's ORGANIZE_P rule fires
        self.p_band_mid = 1.0
        for i in range(gn.MAX_RULES):
            o = gn.HDR + gn.RULE_BYTES * i
            if self.wt[o + 9] % gn.N_ACTIONS == gn.ORG_P:
                ts = [gn.THR_TABLE[self.wt[o + 3 * c + 2]] for c in range(3) if self.wt[o + 3 * c] % gn.N_FIELDS == gn.F_C]
                if ts:
                    self.p_band_mid = float(np.exp(np.mean(np.log(ts))))
        self.decay = {o: np.exp(-np.hypot(*o) / cfg.lam) for o in _OFFS}

    # ------------------------------------------------------------------ helpers
    def set_cue(self, angle):
        """Weak external cue E (think: a distant light source), only used to break the
        initial symmetry when choosing where the anterior organizer sits. It is a point
        source 3 grid-widths away, so on a ring it has a single closest point."""
        N = self.N
        c = N // 2
        sy, sx = c + 3 * N * np.sin(angle), c + 3 * N * np.cos(angle)
        d = np.hypot(self.yy - sy, self.xx - sx)
        self.E = 1.0 - d / (6 * N)

    def _place(self, y, x, template, clean=False):
        cfg = self.cfg
        copies, cons = gn.copy_genome(template, cfg.redundancy, 0.0 if clean else cfg.mu,
                                      cfg.proofread, cfg.proof_eff, self.rng)
        self.copies[y, x] = copies
        self.G[y, x] = cons
        self.valid[y, x] = gn.crc_ok(cons)
        if not np.array_equal(cons, self.wt):
            self.stats["copy_errors_expressed"] += 1
        self.alive[y, x] = True
        self.starve[y, x] = 0
        self.tissue[y, x] = gn.SCAFFOLD
        self.read[:, y, x] = self.fields[:, y, x]
        self.mem[y, x] = np.inf                    # newborn: no positional memory yet

    def kill(self, mask):
        self.alive &= ~mask
        self.tissue[mask] = 0
        self.org[mask] = 0
        self.G[mask] = 0
        self.valid[mask] = False
        self.read[:, mask] = 0
        self.mem[mask] = np.inf

    # ------------------------------------------------------------------ physics
    def relay(self):
        alive = self.alive
        src = np.stack([(self.org == i) & alive for i in (1, 2, 3)]).astype(float)
        F = src.copy()
        for _ in range(4 * self.N):
            new = F.copy()
            for o, d in self.decay.items():
                np.maximum(new, shift(F, *o) * d, out=new)
            new *= alive
            np.maximum(new, src, out=new)
            if np.allclose(new, F, atol=1e-7):
                F = new
                break
            F = new
        self.fields = F

    def sense(self):
        cfg = self.cfg
        noisy = self.fields * (1 + cfg.noise * self.rng.standard_normal(self.fields.shape))
        noisy = np.clip(noisy, 0, None)
        self.read = np.where(self.alive, cfg.ema * self.read + (1 - cfg.ema) * noisy, 0)
        # fields that are truly absent read as zero (no false signal from noise memory)
        self.read[self.fields == 0] = 0
        # organizers integrate their inputs over a longer window (slow memory)
        self.slow = np.where(self.alive, cfg.ema_slow * self.slow + (1 - cfg.ema_slow) * noisy, 0)
        fresh = self.alive & (self.slow == 0)
        self.slow[fresh] = noisy[fresh]
        self.slow[self.fields == 0] = 0
        # positional memory: while the core signal is heard a block remembers its slow C level;
        # when the signal is lost (core destroyed) the last value is kept (frozen), like the
        # positional memory of salamander limb cells. A new block born with no C signal has none.
        # Memory is slow (time constant ~1/mem_rate steps). A block born with no memory of its own
        # adopts the value intercalated from its neighbours if there is one (see intercalate),
        # otherwise its first reading of the core signal.
        heard = self.alive & (self.fields[0] > 0)
        have = np.isfinite(self.mem)
        adopt = self.alive & ~have & np.isfinite(self.recon)
        self.mem[adopt] = self.recon[adopt]
        first = heard & ~have & ~adopt
        self.mem[first] = noisy[0][first]
        upd = heard & have
        self.mem[upd] += cfg.mem_rate * (self.read[0][upd] - self.mem[upd])
        self.mem[~self.alive] = np.inf

    def intercalate(self):
        """Local intercalation of positional values (hardware, like the relay). A block that still
        holds its own positional memory keeps it; a block with none (born in the wound, or a
        replacement after wear) takes the smallest value consistent with its neighbours,
            recon(x) = min_nb recon(nb) * exp(+|step| / lambda),
        i.e. 'I am at most one step closer to the core than my neighbour'. Filling a convex wound
        inward from a rim of remembering blocks recovers exp(-distance/lambda) to the OLD core
        position, so its peak marks where the core organizer was. Runs only through living
        tissue; it stores no pattern, only each block's own last reading."""
        if not self.cfg.pos_memory:
            self.recon[:] = np.inf
            return
        alive = self.alive
        R = np.where(alive, self.mem, np.inf)
        free = alive & ~np.isfinite(self.mem)         # blocks with no memory of their own
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
        self.recon = R

    # ------------------------------------------------------------------ genome interpreter
    def evaluate(self):
        G = self.G
        C, A, P = self.read
        AP = np.sqrt(A * P)
        vals = np.stack([np.ones_like(C), C, A, P, AP])           # field table
        n_rules = np.minimum(G[..., 3], gn.MAX_RULES)
        fate = np.zeros(C.shape, np.int8)
        divide = np.zeros(C.shape, bool)
        starve = np.zeros(C.shape, bool)
        qual = {gn.ORG_C: np.zeros(C.shape, bool), gn.ORG_A: np.zeros(C.shape, bool),
                gn.ORG_P: np.zeros(C.shape, bool)}
        yy, xx = self.yy, self.xx
        for i in range(gn.MAX_RULES):
            o = gn.HDR + gn.RULE_BYTES * i
            match = self.alive & (n_rules > i)
            for c in range(3):
                fid = G[..., o + 3 * c] % gn.N_FIELDS
                op = G[..., o + 3 * c + 1] % 2
                thr = gn.THR_TABLE[G[..., o + 3 * c + 2]]
                v = vals[fid, yy, xx]
                ok = np.where(op == gn.GT, v > thr, v < thr)
                match &= (fid == gn.F_TRUE) | ok
            act = G[..., o + 9] % gn.N_ACTIONS
            divide |= match & (act == gn.DIVIDE)
            starve |= match & (act == gn.STARVE)
            for a in qual:
                qual[a] |= match & (act == a)
            for a, t in gn.FATE_OF_ACTION.items():
                sel = match & (act == a) & (fate == 0)
                fate[sel] = t
        return fate, divide, starve, qual

    # ------------------------------------------------------------------ organizers
    def scores(self):
        """Organizer scores, computed from a block's own (slow-averaged) readings.
        C: min(A, P)        -> the point midway between the two poles
        A: external cue E   -> the point of the ring nearest the cue
        P: ln C - ln A      -> distance-to-A minus distance-to-core, which is maximal where
                               A, core and the block are collinear: the far end of the A-core axis
        """
        C, A, P = self.slow
        with np.errstate(divide="ignore"):
            if self.cfg.p_score == "axis":
                sp = np.log(C + 1e-12) - np.log(A + 1e-12)
            elif self.cfg.p_score == "axis_mid":
                # same axis term, plus a preference for the middle of the genome's P band, so
                # the radial position along the A-core axis is no longer a flat (degenerate) ridge
                sp = (np.log(C + 1e-12) - np.log(A + 1e-12)
                      - self.cfg.p_mid_beta * np.abs(np.log(C + 1e-12) - np.log(self.p_band_mid)))
            else:
                sp = -np.log(A + 1e-12)
        sc = np.minimum(A, P)
        if self.cfg.pos_memory:
            # core score: the intercalated positional estimate where one exists (capped at 1, the
            # source strength), with the A/P balance as a tiny tie-breaker / fallback
            rc = np.where(np.isfinite(self.recon), np.minimum(self.recon, 1.0), 0.0)
            sc = rc + 1e-3 * sc
        return {1: sc, 2: self.E.copy(), 3: sp}

    def organizers(self, qual):
        """Organizer mechanics (hardware): formation, settling (hill-climb), niche loss,
        lateral inhibition. A newly formed organizer settles by hill-climbing its score for a
        maturation window, then fixes in place. The founder core organizer (the seed) is fixed
        from the start - the doc's 'fixed core'. An A/P organizer that is told by the core
        signal that it sits outside its band for 30 steps dissolves (and is re-derived)."""
        cfg = self.cfg
        sc = self.scores()
        amap = {1: gn.ORG_C, 2: gn.ORG_A, 3: gn.ORG_P}
        for t in (1, 2, 3):
            # qualification is smoothed per block (local memory) so noise at a band edge
            # neither lures an organizer outward nor dissolves it
            self.qstab[t - 1] = np.where(self.alive, 0.8 * self.qstab[t - 1] + 0.2 * qual[amap[t]], 0)
            q = (self.qstab[t - 1] > 0.9) & self.alive
            s = sc[t] + self.jitter
            for (y, x) in zip(*np.nonzero(self.org == t)):
                young = self.t - self.org_born[y, x] <= (cfg.mature_core if t == 1 else cfg.mature)
                best, by, bx = s[y, x], y, x
                rc = cfg.R_climb
                for dy in range(-rc, rc + 1):
                    for dx in range(-rc, rc + 1):
                        yy, xx = y + dy, x + dx
                        if young and 0 <= yy < self.N and 0 <= xx < self.N and q[yy, xx] and self.org[yy, xx] == 0:
                            if s[yy, xx] > best + (0.0 if t == 2 else cfg.hysteresis * abs(best)) + 1e-9:
                                best, by, bx = s[yy, xx], yy, xx
                if (by, bx) != (y, x):
                    self.org[y, x] = 0
                    self.org[by, bx] = t
                    self.org_born[by, bx] = self.org_born[y, x]
                    self.org_off[by, bx] = 0
                elif t != 1:
                    # niche rule: an A/P organizer that no longer qualifies dissolves
                    # (only when the core signal is present: with no core there is no niche to judge)
                    lost = self.qstab[t - 1, y, x] <= 0.1 and self.fields[0, y, x] > 0
                    self.org_off[y, x] = self.org_off[y, x] + 1 if lost else 0
                    if self.org_off[y, x] >= 30:
                        self.org[y, x] = 0
            # de-novo formation where this organizer's signal is absent
            # (organizers only form in intact tissue: no empty space within radius 2)
            cand = q & (self.fields[t - 1] < 1e-3) & (self.org == 0) & self.intact
            if not cfg.organizer_regen and self._formed.get(t):
                cand[:] = False
            if cand.any():
                ms = np.where(cand, s, -np.inf)
                lm = ms.copy()
                ro = cfg.R_org if t != 1 else int(np.ceil(cfg.lam))
                for dy in range(-ro, ro + 1):
                    for dx in range(-ro, ro + 1):
                        if dy * dy + dx * dx <= ro ** 2:
                            np.maximum(lm, shift(ms, dy, dx, -np.inf), out=lm)
                new = cand & (ms >= lm) & (self.rng.random(cand.shape) < cfg.p_form)
                self.org[new] = t
                self.org_born[new] = self.t
                self.org_off[new] = 0
                if new.any():
                    self._formed[t] = True
            # lateral inhibition between same-type organizers
            pos = list(zip(*np.nonzero(self.org == t)))
            pos.sort(key=lambda p: -s[p])
            kept = []
            for p in pos:
                if any(np.hypot(p[0] - k[0], p[1] - k[1]) <= cfg.R_inhib for k in kept):
                    self.org[p] = 0
                else:
                    kept.append(p)

    # ------------------------------------------------------------------ main step
    def step(self):
        cfg, rng = self.cfg, self.rng
        self.relay()
        self.sense()
        fate, divide, starve, qual = self.evaluate()
        self.tissue = np.where(self.alive, np.where(fate > 0, fate, np.maximum(self.tissue, 1)), 0).astype(np.int8)
        empty = (~self.alive).astype(np.int8)
        near_empty = np.zeros_like(self.alive)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                near_empty |= shift(empty, dy, dx, 1).astype(bool)
        self.intact = self.alive & ~near_empty
        if not (self.org == 1).any() or self.t - self.org_born[self.org == 1].max() <= cfg.mature_core:
            self.intercalate()      # only needed while a core organizer is missing or settling
        else:
            self.recon[:] = np.inf
        self.organizers(qual)

        # ---- deaths
        self.starve = np.where(starve & self.alive, self.starve + 1, 0)
        starve_T = np.maximum(self.G[..., 5], 1)
        dead = self.alive & (self.starve >= starve_T)
        bad = self.alive & ~self.valid
        if cfg.apoptosis:   # layer 3a: self-check (only works if the apoptosis gene is intact)
            apo = bad & (self.G[..., 6] == gn.APOPTOSIS_ON)
            self.stats["apoptosis"] += int(apo.sum())
            dead |= apo
        if cfg.neighbor_clear:  # layer 3b: neighbours with valid markers clear a bad one
            good = (self.alive & self.valid).astype(np.int8)
            ngood = sum(shift(good, dy, dx) for dy, dx in _NB8)
            clr = bad & ~dead & (ngood >= 3) & (rng.random(bad.shape) < cfg.p_clear)
            self.stats["cleared"] += int(clr.sum())
            dead |= clr
        # random wear; organizer blocks are treated as hardened founder blocks (exempt)
        dead |= self.alive & (self.org == 0) & (rng.random(self.alive.shape) < cfg.turnover)
        if dead.any():
            self.kill(dead)

        # ---- divisions (into an empty 8-neighbour)
        p_div = self.G[..., 4] / 255.0
        dv = divide & self.alive & (rng.random(divide.shape) < p_div)
        ys, xs = np.nonzero(dv)
        order = rng.permutation(len(ys))
        N = self.N
        for i in order:
            y, x = ys[i], xs[i]
            empties = [(y + dy, x + dx) for dy, dx in _NB8
                       if 0 <= y + dy < N and 0 <= x + dx < N and not self.alive[y + dy, x + dx]]
            if not empties:
                continue
            ty, tx = empties[rng.integers(len(empties))]
            self._place(ty, tx, self.G[y, x])
            self.stats["divisions"] += 1
        self.t += 1

    def run(self, steps, callback=None):
        for _ in range(steps):
            self.step()
            if callback:
                callback(self)
        return self

    # ------------------------------------------------------------------ outputs
    def pattern(self):
        return np.where(self.alive, self.tissue, 0).astype(np.int8)

    def organizer_positions(self):
        return {ORG_NAMES[t]: list(zip(*np.nonzero(self.org == t))) for t in (1, 2, 3)}

    def mutant_fraction(self):
        if not self.alive.any():
            return 0.0
        return float((self.G[self.alive] != self.wt).any(axis=1).mean())


def grow(cfg: Config, steps: int = 140):
    return World(cfg).run(steps)
