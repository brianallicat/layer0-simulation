import sys, copy, numpy as np
sys.path.insert(0,'.')
from ext_v3 import ConfigV3, WorldV3
from experiments import damage_masks, _orgs
np.set_printoptions(linewidth=250, precision=3, suppress=True)
seed=int(sys.argv[1])
base=WorldV3(ConfigV3(seed=seed,mem_v3=True)).run(200)
true=base.fields[2].copy(); P=_orgs(base)['P'][0]
anc=base._anchors(2); r=base.m3[2][anc]/true[anc]
print('anchor mem/true', r.mean(), np.median(r), r.std())
m=[v for k,v in damage_masks(base).items() if k.startswith('post')][0]
w=copy.deepcopy(base); w.rng=np.random.default_rng(12345); w.kill(m)
for k in range(60):
    w.step()
    if k in (10,30,59):
        e=w._estimate(2); y,x=P
        print(k,_orgs(w)['P'], 'row', e[y, x-4:x+4], 'col', e[y-3:y+4, x])
