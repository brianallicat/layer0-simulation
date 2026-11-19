import sys, numpy as np
sys.path.insert(0,'.')
from sim import Config, World
from metrics import similarity, resample, class_iou
import genome as gn
ref=resample(World(Config(lam=7.0,seed=0)).run(300).pattern(),64)
for lam in (4.0,7.0,11.0):
    w=World(Config(lam=lam,seed=1)).run(300)
    p=w.pattern(); n=(p>0).sum(); R=np.sqrt(n/np.pi); c=w.N//2
    fr={gn.TISSUE_NAMES[i]:round((p==i).sum()/n,3) for i in range(1,7)}
    o={k:[((a-c)/R,(b-c)/R) for a,b in v] for k,v in w.organizer_positions().items()}
    print(lam,n,round(similarity(resample(p,64),ref),3),fr,{k:[(round(a,2),round(b,2)) for a,b in v] for k,v in o.items()})
    print({gn.TISSUE_NAMES[k]:round(v,2) for k,v in class_iou(resample(p,64),ref).items()})
