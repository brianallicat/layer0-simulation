import copy, sys, json, numpy as np
sys.path.insert(0, '.')
from sim import Config, World
from experiments import damage_masks, _orgs, GROW_STEPS
seed=int(sys.argv[1]); K=int(sys.argv[2])
base = World(Config(seed=seed, lam=7.0)).run(GROW_STEPS)
w = copy.deepcopy(base); w.rng = np.random.default_rng(12345)
w.kill(damage_masks(base)["core excision (incl. organizer C)"])
def show(a):
    a=np.where(np.isfinite(a),np.minimum(a,9.99),-0.01); print((a[18:41,18:41]*100).round().astype(int))
for k in range(K): w.step()
show(w.mem); show(w.recon)
