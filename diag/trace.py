import copy, sys, numpy as np
sys.path.insert(0, '.')
from sim import Config, World
from experiments import damage_masks, _orgs, GROW_STEPS
base = World(Config(seed=0, lam=7.0)).run(GROW_STEPS)
w = copy.deepcopy(base); w.rng = np.random.default_rng(12345)
w.kill(damage_masks(base)["core excision (incl. organizer C)"])
for k in range(200):
    w.step()
    o=_orgs(w)['C']
    if o or k%20==0:
        r=np.where(np.isfinite(w.recon),w.recon,-1)
        am=np.unravel_index(np.argmax(r),r.shape)
        print(k,o,'recon argmax',am,'%.3f'%r.max(),'alive hole', w.alive[22:37,22:37].sum(), 'qC', (w.qstab[0]>0.9).sum())
        if o and k>140: break
r=np.where(np.isfinite(w.recon),w.recon,0)
np.set_printoptions(linewidth=250,precision=2)
print((r[20:40,20:40]*100).astype(int))
