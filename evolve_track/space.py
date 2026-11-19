"""Searchable rule space over Layer 0's existing genome components.

A candidate is (genome: 131 bytes, neighbor_clear: bool).
  * genome bytes are the ONLY thing evolution writes into the body; they keep the exact format
    of genome.py (7-byte header, 12 rules x 10 bytes, CRC32 tail) and the CRC is recomputed, so
    every candidate is a valid, self-consistent genome the unchanged simulator runs as-is.
  * neighbor_clear is a hardware switch of sim.Config (layer 3b of the error correction). It is not
    in the genome; it is carried as a separate on/off gene because the brief asked for it. Turning
    it off can only remove a protection, never add an ability.

Genes (what mutation may change):
  header   p_divide byte (division probability /255), starve_T (steps before a starving block
           dies = growth limit / self-destruct timer), apoptosis flag (0xA5 = self-check suicide ON,
           0x00 = OFF).  magic, version and n_rules (=12) are frozen.
  rules    per condition: field id (0..4), op (GT/LT), threshold byte (log-coded exp(-b/32));
           per rule: action (0..11: NOP, DIVIDE (growth), STARVE (apoptosis), 6 fates, 3 organizers).
Thresholds are the main search axis; field/op/action changes are rare structural mutations.
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field

import numpy as np

import genome as gn

N_RULES = 12
assert gn.MAX_RULES == N_RULES and gn.GENOME_LEN == 131


@dataclass
class Params:
    p_divide: int
    starve_T: int
    apoptosis: int                    # raw byte (0xA5 = ON)
    rules: list = field(default_factory=list)   # [ [(fid,op,tbyte)x3], action ]
    header_frozen: bytes = b"N0\x01\x0c"

    def copy(self):
        return Params(self.p_divide, self.starve_T, self.apoptosis,
                      [[list(map(tuple, r[0])), r[1]] for r in self.rules], self.header_frozen)


def decode_params(g: bytes) -> Params:
    g = bytes(g)
    assert len(g) == gn.GENOME_LEN
    rules = []
    for i in range(N_RULES):
        o = gn.HDR + gn.RULE_BYTES * i
        conds = [(g[o + 3 * c], g[o + 3 * c + 1], g[o + 3 * c + 2]) for c in range(3)]
        rules.append([conds, g[o + 9]])
    return Params(g[4], g[5], g[6], rules, g[:4])


def encode_params(p: Params) -> bytes:
    body = bytearray(p.header_frozen)
    body += bytes([p.p_divide, p.starve_T, p.apoptosis])
    for conds, action in p.rules:
        for (fid, op, tb) in conds:
            body += bytes([fid, op, tb])
        body += bytes([action])
    crc = zlib.crc32(bytes(body)) & 0xFFFFFFFF
    out = bytes(body) + struct.pack("<I", crc)
    assert len(out) == gn.GENOME_LEN
    return out


def is_valid(g: bytes) -> bool:
    """Valid point of the space: right length, frozen header, CRC intact, all ids in range."""
    g = bytes(g)
    if len(g) != gn.GENOME_LEN or g[:4] != gn.WILDTYPE[:4] or not gn.crc_ok(g):
        return False
    p = decode_params(g)
    for conds, a in p.rules:
        if a >= gn.N_ACTIONS:
            return False
        for fid, op, _ in conds:
            if fid >= gn.N_FIELDS or op > 1:
                return False
    return 1 <= p.p_divide <= 255 and 1 <= p.starve_T <= 255 and p.apoptosis in (0x00, gn.APOPTOSIS_ON)


@dataclass(frozen=True)
class Candidate:
    genome: bytes
    neighbor_clear: bool = True

    def key(self):
        return (self.genome, self.neighbor_clear)


WILDTYPE = Candidate(gn.WILDTYPE, True)


def mutate(cand: Candidate, rng: np.random.Generator, strength: float = 1.0) -> Candidate:
    """Return a mutated copy. Every output is valid (is_valid() == True)."""
    p = decode_params(cand.genome).copy()
    nc = cand.neighbor_clear
    n_events = 1 + rng.poisson(1.0 * strength)
    for _ in range(n_events):
        r = rng.random()
        if r < 0.62:                                   # threshold of a non-TRUE condition
            live = [(i, c) for i in range(N_RULES) for c in range(3) if p.rules[i][0][c][0] != gn.F_TRUE]
            i, c = live[rng.integers(len(live))]
            fid, op, tb = p.rules[i][0][c]
            step = int(np.round(rng.normal(0, 6 * strength))) or int(rng.choice([-1, 1]))
            p.rules[i][0][c] = (fid, op, int(np.clip(tb + step, 0, 255)))
        elif r < 0.72:                                 # p_divide
            p.p_divide = int(np.clip(p.p_divide + np.round(rng.normal(0, 20 * strength)), 1, 255))
        elif r < 0.80:                                 # starve_T (self-destruct timer)
            p.starve_T = int(np.clip(p.starve_T + np.round(rng.normal(0, 25 * strength)), 1, 255))
        elif r < 0.84:                                 # apoptosis flag on/off
            p.apoptosis = 0x00 if p.apoptosis == gn.APOPTOSIS_ON else gn.APOPTOSIS_ON
        elif r < 0.87:                                 # neighbour clearance on/off (hardware switch)
            nc = not nc
        elif r < 0.92:                                 # op flip
            i, c = rng.integers(N_RULES), rng.integers(3)
            fid, op, tb = p.rules[i][0][c]
            if fid != gn.F_TRUE:
                p.rules[i][0][c] = (fid, 1 - op, tb)
        elif r < 0.96:                                 # field change (TRUE <-> field adds/removes a condition)
            i, c = rng.integers(N_RULES), rng.integers(3)
            fid, op, tb = p.rules[i][0][c]
            nf = int(rng.integers(gn.N_FIELDS))
            p.rules[i][0][c] = (nf, op if nf != gn.F_TRUE else 0, (tb if nf != gn.F_TRUE else 0) if fid != gn.F_TRUE else int(rng.integers(0, 200)))
        else:                                          # action change
            i = rng.integers(N_RULES)
            p.rules[i][1] = int(rng.integers(gn.N_ACTIONS))
    out = Candidate(encode_params(p), nc)
    assert is_valid(out.genome)
    return out


def crossover(a: Candidate, b: Candidate, rng: np.random.Generator) -> Candidate:
    """Rule-wise uniform crossover (whole rules and header genes are inherited intact)."""
    pa, pb = decode_params(a.genome), decode_params(b.genome)
    c = pa.copy()
    for i in range(N_RULES):
        if rng.random() < 0.5:
            c.rules[i] = [list(pb.rules[i][0]), pb.rules[i][1]]
    if rng.random() < 0.5:
        c.p_divide = pb.p_divide
    if rng.random() < 0.5:
        c.starve_T = pb.starve_T
    if rng.random() < 0.5:
        c.apoptosis = pb.apoptosis
    nc = a.neighbor_clear if rng.random() < 0.5 else b.neighbor_clear
    return Candidate(encode_params(c), nc)


def _cond_txt(fid, op, tb):
    if fid % gn.N_FIELDS == gn.F_TRUE:
        return "always"
    return f"{gn.FIELD_NAMES[fid % gn.N_FIELDS]} {'>' if op % 2 == gn.GT else '<'} {gn.THR_TABLE[tb]:.3f}"


def describe_diff(a: Candidate, b: Candidate) -> list:
    """Plain-language list of what changed from a (e.g. wild type) to b."""
    pa, pb = decode_params(a.genome), decode_params(b.genome)
    out = []
    if pa.p_divide != pb.p_divide:
        out.append(f"division probability p_divide {pa.p_divide/255:.2f} -> {pb.p_divide/255:.2f}")
    if pa.starve_T != pb.starve_T:
        out.append(f"starvation self-destruct timer starve_T {pa.starve_T} -> {pb.starve_T} steps")
    if pa.apoptosis != pb.apoptosis:
        out.append(f"self-check apoptosis {'ON' if pa.apoptosis == gn.APOPTOSIS_ON else 'OFF'} -> "
                   f"{'ON' if pb.apoptosis == gn.APOPTOSIS_ON else 'OFF'}")
    if a.neighbor_clear != b.neighbor_clear:
        out.append(f"neighbour clearance (hardware switch) {'ON' if a.neighbor_clear else 'OFF'} -> "
                   f"{'ON' if b.neighbor_clear else 'OFF'}")
    for i in range(N_RULES):
        (ca, aa), (cb, ab) = pa.rules[i], pb.rules[i]
        name = f"R{i+1:02d} ({gn.ACTION_NAMES[aa % gn.N_ACTIONS]})"
        if aa != ab:
            out.append(f"{name}: action {gn.ACTION_NAMES[aa % gn.N_ACTIONS]} -> {gn.ACTION_NAMES[ab % gn.N_ACTIONS]}")
        for c in range(3):
            if tuple(ca[c]) != tuple(cb[c]):
                fa, fb = ca[c][0] % gn.N_FIELDS, cb[c][0] % gn.N_FIELDS
                if fa == fb == gn.F_TRUE:
                    continue
                out.append(f"{name} condition {c+1}: {_cond_txt(*ca[c])} -> {_cond_txt(*cb[c])}")
    return out


def hamming(a: bytes, b: bytes) -> int:
    return int(sum(x != y for x, y in zip(a[:-4], b[:-4])))
