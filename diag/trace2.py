import copy, sys, numpy as np
sys.path.insert(0, '.')
from sim import Config, World
from experiments import damage_masks, _orgs, GROW_STEPS
np.set_printoptions(linewidth=250)
base = World(Config(seed=0, lam=7.0)).run(GROW_STEPS)
w = copy.deepcopy(base); w.rng = np.random.default_rng(12345)
w.kill(damage_masks(base)["core excision (incl. organizer C)"])
def show(a): 
    a=np.where(np.isfinite(a),a,-0.01); print((a[17:42:1,17:42:1]*100).round().astype(int))
show(w.mem)
for k in range(95): w.step()
show(w.mem); show(w.recon)
m=np.where(np.isfinite(w.mem),w.mem,9); print('min mem',m.min(),np.unravel_index(m.argmin(),m.shape), 'alive',w.alive.sum())
print(sorted(w.decay.values())[:3], len(w.decay))
