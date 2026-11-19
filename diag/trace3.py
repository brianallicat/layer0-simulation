import copy, sys, json, numpy as np
sys.path.insert(0, '.')
from sim import Config, World
from experiments import damage_masks, _orgs, GROW_STEPS
kw=json.loads(sys.argv[1])
base = World(Config(seed=0, lam=7.0, **kw)).run(GROW_STEPS)
w = copy.deepcopy(base); w.rng = np.random.default_rng(12345)
w.kill(damage_masks(base)["core excision (incl. organizer C)"])
for k in range(160):
    w.step()
    if k%10==0:
        q=(w.qstab[0]>0.9)
        print(k,int(w.alive.sum()),_orgs(w),'qC',q.sum(),'q&intact',(q&w.intact).sum(),'hole',w.alive[24:35,24:35].sum(), 'A@c %.3f P@c %.3f'%(w.read[1,29,29],w.read[2,29,29]))
