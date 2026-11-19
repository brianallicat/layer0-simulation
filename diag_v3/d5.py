import sys, copy, numpy as np
sys.path.insert(0,'.')
from ext_v3 import ConfigV3, WorldV3
from experiments import damage_masks, _orgs
seed=6
base=WorldV3(ConfigV3(seed=seed,mem_v3=True)).run(200)
true=base.fields[2].copy()
m=[v for k,v in damage_masks(base).items() if k.startswith('post')][0]
w=copy.deepcopy(base); w.rng=np.random.default_rng(12345); w.kill(m)
holders0 = w.alive & np.isfinite(w.m3[2])
for k in range(40): w.step()
H = w.alive & np.isfinite(w.m3[2])
ys,xs=np.nonzero(H)
d=np.hypot(ys-26,xs-18)
v=w.m3[2][ys,xs]*np.exp(d/7)
idx=np.argsort(v)[:8]
for i in idx: print(ys[i],xs[i],round(v[i],3),'mem',round(w.m3[2][ys[i],xs[i]],3),'true',round(true[ys[i],xs[i]],3),'orig holder',holders0[ys[i],xs[i]], 'n',w.n3[2][ys[i],xs[i]])
