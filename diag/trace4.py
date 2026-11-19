import copy, sys, json, numpy as np
sys.path.insert(0, '.')
from sim import Config, World
from experiments import damage_masks, _orgs, GROW_STEPS
seed=int(sys.argv[1])
base = World(Config(seed=seed, lam=7.0)).run(GROW_STEPS)
w = copy.deepcopy(base); w.rng = np.random.default_rng(12345)
w.kill(damage_masks(base)["core excision (incl. organizer C)"])
prev=None
for k in range(300):
    w.step()
    o=_orgs(w)['C']
    if o!=prev or k%25==0:
        r=np.where(np.isfinite(w.recon),np.minimum(w.recon,1),-1); am=np.unravel_index(np.argmax(r),r.shape)
        q=(w.qstab[0]>0.9)&w.alive
        print(k,o,'recon@29,29=%.3f'%r[29,29],'argmax',am,'%.3f'%r.max(),'q@29,29',q[29,29],'alive@',w.alive[29,29],'hole',w.alive[22:37,22:37].sum(), 'mem@29,29', w.mem[29,29])
    prev=o
