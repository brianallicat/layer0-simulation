import copy, sys, numpy as np
sys.path.insert(0, '.')
from sim import Config, World
from experiments import damage_masks, _orgs, GROW_STEPS
from metrics import similarity, class_iou
import genome as gn
name = sys.argv[1] if len(sys.argv) > 1 else "core excision (incl. organizer C)"
base = World(Config(seed=0, lam=7.0)).run(GROW_STEPS)
P0 = base.pattern()
w = copy.deepcopy(base); w.rng = np.random.default_rng(12345)
m = damage_masks(base)[name]; w.kill(m)
print("base orgs", _orgs(base), "cells", base.alive.sum(), "N", base.N)
for k in range(400):
    w.step()
    if k % 10 == 0 or k < 10:
        p = w.pattern()
        iou = class_iou(p, P0)
        print(k, "cells", int(w.alive.sum()), "sim %.3f" % similarity(p, P0), _orgs(w),
              " ".join(f"{gn.TISSUE_NAMES[c][:3]}={(p==c).sum()}/{(P0==c).sum()}" for c in range(1,7)))
np.save("diag/final.npy", w.pattern()); np.save("diag/base.npy", P0)
