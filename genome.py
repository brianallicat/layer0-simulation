"""Layer 0 genome: a tiny byte-coded rule program ("a recipe, not a blueprint").

Every block carries the same bytes. Nothing in here names a coordinate, a body
size, or a cell-by-cell layout. Each rule is purely LOCAL:

    IF <field> <op> <threshold> AND ... (up to 3 conditions) THEN <action>

where <field> is something the block can measure at its own position
(the strength of a morphogen field relayed by its neighbours).

Byte layout
-----------
header (7 bytes): magic 'N','0' | version | n_rules | p_divide(/255) | starve_T | apoptosis_flag(0xA5)
rules  (10 bytes each): f1 op1 t1  f2 op2 t2  f3 op3 t3  action
tail   (4 bytes): CRC32 of everything before it (the "self marker" used for proofreading,
                  self-check apoptosis and neighbour clearance)

Any byte value decodes to *something* (ids are taken modulo table size), so a random
copy error always yields a runnable - possibly harmful - phenotype, like a real mutation.
"""
from __future__ import annotations

import math
import struct
import zlib

import numpy as np

# ---- field ids a block can read locally -------------------------------------------
F_TRUE, F_C, F_A, F_P, F_AP = 0, 1, 2, 3, 4
N_FIELDS = 5
FIELD_NAMES = ["TRUE", "C(core)", "A(anterior)", "P(posterior)", "sqrt(A*P)"]

# ---- actions ----------------------------------------------------------------------
(NOP, DIVIDE, STARVE, BECOME_SCAFFOLD, BECOME_SHELL, BECOME_CONDUIT, BECOME_CORE,
 BECOME_SENSOR, BECOME_RADIATOR, ORG_A, ORG_P, ORG_C) = range(12)
N_ACTIONS = 12
ACTION_NAMES = ["NOP", "DIVIDE", "STARVE", "SCAFFOLD", "SHELL", "CONDUIT", "CORE",
                "SENSOR", "RADIATOR", "ORGANIZE_A", "ORGANIZE_P", "ORGANIZE_C"]

# tissue codes (what a block differentiates into)
EMPTY, SCAFFOLD, SHELL, CONDUIT, CORE, SENSOR, RADIATOR = range(7)
TISSUE_NAMES = ["empty", "scaffold", "shell", "conduit", "core", "sensor", "radiator"]
FATE_OF_ACTION = {BECOME_SCAFFOLD: SCAFFOLD, BECOME_SHELL: SHELL, BECOME_CONDUIT: CONDUIT,
                  BECOME_CORE: CORE, BECOME_SENSOR: SENSOR, BECOME_RADIATOR: RADIATOR}

HDR = 7
RULE_BYTES = 10
APOPTOSIS_ON = 0xA5
GT, LT = 0, 1

# thresholds are log-coded: byte b -> exp(-b/32)  (1.0 ... 3e-4, ~3 % resolution)
THR_TABLE = np.exp(-np.arange(256) / 32.0)


def thr_byte(t: float) -> int:
    return int(round(min(255, max(0, -32.0 * math.log(t)))))


def _design_rules():
    """The wild-type rule set. Thresholds are expressed as fractions of the body radius R
    only to *derive* them here, in field units; the block never knows R, its
    coordinates, or the body size. L = ln(1/t_grow) = R / lambda."""
    t_grow = 0.08
    L = math.log(1 / t_grow)
    f = lambda frac: math.exp(-frac * L)  # field value at distance frac*R from a source
    c_hi, c_lo = f(0.62), f(0.78)        # the "organizer ring" band
    T = F_TRUE
    rules = [
        # --- organizers (founder/regenerating signalling centres) ---
        ((F_A, GT, f(0.78)), (F_P, GT, f(0.78)), (T, GT, 1), ORG_C),  # both poles strong -> may become core organizer
        ((F_C, LT, c_hi), (F_C, GT, c_lo), (T, GT, 1), ORG_A),    # in ring band -> may become anterior organizer
        ((F_C, LT, c_hi), (F_C, GT, c_lo), (F_A, GT, 0.002), ORG_P),  # ring band and A heard -> may become posterior
        # --- growth control ---
        ((F_C, GT, t_grow), (T, GT, 1), (T, GT, 1), DIVIDE),      # core signal strong enough -> divide
        ((F_A, GT, f(1.05)), (F_P, GT, f(1.05)), (T, GT, 1), DIVIDE),  # wound healing: near both poles' midzone -> divide
        ((F_C, LT, f(1.15)), (T, GT, 1), (T, GT, 1), STARVE),     # core signal lost -> starve (die after starve_T steps)
        # --- regional modifiers (epigenome): first matching fate wins ---
        ((F_C, GT, f(0.25)), (T, GT, 1), (T, GT, 1), BECOME_CORE),
        ((F_A, GT, f(0.40)), (T, GT, 1), (T, GT, 1), BECOME_SENSOR),
        ((F_A, LT, f(1.25)), (F_A, GT, 0.002), (T, GT, 1), BECOME_RADIATOR),  # far from A (but A heard)
        ((F_AP, GT, f(0.75)), (T, GT, 1), (T, GT, 1), BECOME_CONDUIT),
        ((F_C, LT, f(0.85)), (T, GT, 1), (T, GT, 1), BECOME_SHELL),
        ((T, GT, 1), (T, GT, 1), (T, GT, 1), BECOME_SCAFFOLD),
    ]
    return rules


def build_genome(rules=None, p_divide=0.5, starve_T=150) -> bytes:
    rules = _design_rules() if rules is None else rules
    body = bytearray(b"N0")
    body += bytes([1, len(rules), int(p_divide * 255), starve_T, APOPTOSIS_ON])
    for rule in rules:
        conds, action = rule[:3], rule[3]
        for (fid, op, t) in conds:
            body += bytes([fid, op, thr_byte(t) if fid != F_TRUE else 0])
        body += bytes([action])
    crc = zlib.crc32(bytes(body)) & 0xFFFFFFFF
    return bytes(body) + struct.pack("<I", crc)


WILDTYPE = build_genome()
GENOME_LEN = len(WILDTYPE)
MAX_RULES = (GENOME_LEN - HDR - 4) // RULE_BYTES


def crc_ok(g: np.ndarray | bytes) -> bool:
    b = bytes(g)
    return (zlib.crc32(b[:-4]) & 0xFFFFFFFF) == struct.unpack("<I", b[-4:])[0]


def decode(g: bytes) -> str:
    """Human-readable listing of a genome."""
    lines = [f"header: magic={g[:2]!r} v{g[2]} n_rules={g[3]} p_div={g[4]/255:.2f} "
             f"starve_T={g[5]} apoptosis={'ON' if g[6]==APOPTOSIS_ON else 'off'}"]
    for i in range(min(g[3], MAX_RULES)):
        o = HDR + RULE_BYTES * i
        conds = []
        for c in range(3):
            fid, op, tb = g[o + 3 * c] % N_FIELDS, g[o + 3 * c + 1] % 2, g[o + 3 * c + 2]
            if fid == F_TRUE:
                continue
            conds.append(f"{FIELD_NAMES[fid]} {'>' if op == GT else '<'} {THR_TABLE[tb]:.3f}")
        lines.append(f"R{i+1:02d}: IF {' AND '.join(conds) or 'always'} THEN {ACTION_NAMES[g[o+9] % N_ACTIONS]}")
    lines.append(f"crc32 ok: {crc_ok(g)}")
    return "\n".join(lines)


def copy_genome(template: np.ndarray, k: int, mu: float, proofread: bool, proof_eff: float,
                rng: np.random.Generator):
    """Copy a genome into k redundant copies with per-byte error rate mu.
    Proofreading = compare each new copy against the template at write time and repair
    each mismatch with probability proof_eff. Returns (copies[k,L], consensus[L])."""
    L = template.shape[0]
    copies = np.broadcast_to(template, (k, L)).copy()
    if mu > 0:
        err = rng.random((k, L)) < mu
        if err.any():
            if proofread:
                err &= rng.random((k, L)) >= proof_eff  # proofreader caught & fixed the rest
            n = int(err.sum())
            if n:
                copies[err] = rng.integers(0, 256, n, dtype=np.uint8)
    return copies, consensus(copies)


def consensus(copies: np.ndarray) -> np.ndarray:
    """Byte-wise majority vote across redundant copies (k=1 -> identity)."""
    k = copies.shape[0]
    if k == 1:
        return copies[0].copy()
    if k == 3:
        a, b, c = copies
        return np.where((a == b) | (a == c), a, np.where(b == c, b, a))
    # general k: per-byte mode
    out = np.empty(copies.shape[1], dtype=np.uint8)
    for j in range(copies.shape[1]):
        vals, cnt = np.unique(copies[:, j], return_counts=True)
        out[j] = vals[np.argmax(cnt)]
    return out


if __name__ == "__main__":
    print(f"genome length: {GENOME_LEN} bytes, rules: {MAX_RULES}")
    print(decode(WILDTYPE))
