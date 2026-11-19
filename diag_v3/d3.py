import sys, copy, time, numpy as np
sys.path.insert(0,'.')
from ext_v3 import ConfigV3, WorldV3
from sim import Config, World
from experiments import damage_masks, _orgs
from metrics import similarity
seed=int(sys.argv[1]); cut=sys.argv[2]; import json; flags=dict(mem_v3=True, **json.loads(sys.argv[3] if len(sys.argv)>3 else "{}"))
base=WorldV3(ConfigV3(seed=seed,**flags)).run(200)
b0=World(Config(seed=seed)).run(200)
print('same growth', np.array_equal(base.pattern(), b0.pattern()))
m=[v for k,v in damage_masks(base).items() if k.startswith(cut)][0]
w=copy.deepcopy(base); w.rng=np.random.default_rng(12345); w.kill(m)
P0=base.pattern(); c=[]
for k in range(400):
    w.step(); c.append(similarity(w.pattern(),P0))
    if k%40==0: print(k,w.alive.sum(),_orgs(w),w.rederiving, round(c[-1],3))
print('orig',_orgs(base),'final',_orgs(w),'last50',np.mean(c[-50:]))
