"""Evolved-rule comparison track for Layer 0 (an ADDITION; nothing in the original model,
experiments, tests or results is modified). Rules are discovered by a (mu+lambda) evolutionary
search over the existing 131-byte genome format and run by the unchanged simulator (sim.py).

Modules
  space     - the searchable rule space (decode/encode/mutate/crossover over genome bytes)
  protocol  - honest scoring (independent references, sustained-recovery rule, tumour test)
  evolve    - the (mu+lambda) search
  run       - full run: search on training seeds, held-out evaluation, report files
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
