import sys, json, numpy as np
sys.path.insert(0,'.')
from ext_v3 import ConfigV3, WorldV3
from metrics import similarity, resample
from concurrent.futures import ProcessPoolExecutor
kw=json.loads(sys.argv[1])
def job(a):
    lam,seed=a
    w=WorldV3(ConfigV3(lam=lam,seed=seed,**kw)).run(300)
    p=w.pattern(); n=(p>0).sum(); R=np.sqrt(n/np.pi); c=w.N//2
    o={k:[round(float(np.hypot(a-c,b-c)/R),2) for a,b in v] for k,v in w.organizer_positions().items()}
    return lam,seed,p,n,o,(w.s_trace[::50]+[w.s_trace[-1]] if w.s_trace else None)
jobs=[(7.0,0)]+[(l,s) for l in (4.0,7.0,11.0) for s in (1,2,3)]
with ProcessPoolExecutor(6) as ex: out=list(ex.map(job,jobs))
ref=resample(out[0][2],64)
for lam,seed,p,n,o,st in out[1:]:
    print(lam,seed,n,round(similarity(resample(p,64),ref),4),o,st and [round(x,3) for x in st])
