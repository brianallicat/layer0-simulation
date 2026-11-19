import sys, copy, numpy as np
sys.path.insert(0,'.')
from ext_v3 import ConfigV3, WorldV3
from experiments import damage_masks, _orgs
np.set_printoptions(linewidth=250, precision=2, suppress=True)
seed=int(sys.argv[1]); cut=sys.argv[2]; t=int(sys.argv[3])
base=WorldV3(ConfigV3(seed=seed,mem_v3=True)).run(200)
true=base.fields[t].copy()
mm=base.m3[t]
ok=np.isfinite(mm)&base.alive&(true>0)
print('mem/true ratio: mean',np.mean(mm[ok]/true[ok]),'sd',np.std(mm[ok]/true[ok]))
m=[v for k,v in damage_masks(base).items() if k.startswith(cut)][0]
w=copy.deepcopy(base); w.rng=np.random.default_rng(12345); w.kill(m)
o=_orgs(base); oy,ox=o['CAP'[t]][0]
for k in range(int(sys.argv[4])): w.step()
est=np.where(np.isfinite(w.m3[t]),w.m3[t],w.est3[t])
print(_orgs(w))
sl=(slice(oy-5,oy+6),slice(max(ox-6,0),ox+6))
print('est'); print(np.where(np.isfinite(est[sl]),est[sl],-1))
print('true'); print(true[sl])
print('score'); print(w.scores()[t+1][sl])
print('q', (w.qstab[t][sl]>0.9).astype(int))
