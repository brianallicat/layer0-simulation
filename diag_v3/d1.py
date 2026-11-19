import sys, copy, time, numpy as np
sys.path.insert(0,'.')
from sim import Config, World
from experiments import damage_masks, _orgs
from metrics import similarity
seed=int(sys.argv[1]); cut=sys.argv[2]
t0=time.time()
base=World(Config(seed=seed)).run(200)
print('grow',time.time()-t0, _orgs(base))
m=[v for k,v in damage_masks(base).items() if k.startswith(cut)][0]
w=copy.deepcopy(base); w.rng=np.random.default_rng(12345); w.kill(m)
P0=base.pattern()
# memory map before
mem=np.where(np.isfinite(base.mem),base.mem,0); print('mem argmax',np.unravel_index(mem.argmax(),mem.shape), mem.max())
t0=time.time()
for k in range(400):
    w.step()
    if k%20==0 or k<10:
        r=np.where(np.isfinite(w.recon),w.recon,-1)
        print(k,w.alive.sum(),_orgs(w), 'reconpk',np.unravel_index(r.argmax(),r.shape),round(r.max(),3), round(similarity(w.pattern(),P0),3), 'intact', w.intact.sum(), 'qC',(w.qstab[0]>0.9).sum())
print('t',time.time()-t0)
