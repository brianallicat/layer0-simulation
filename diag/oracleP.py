import copy, sys, numpy as np
sys.path.insert(0, '.')
from sim import Config, World
from experiments import damage_masks, _orgs, GROW_STEPS
from metrics import similarity, aligned_similarity
from concurrent.futures import ProcessPoolExecutor
base = World(Config(seed=0, lam=7.0)).run(GROW_STEPS)
P0 = base.pattern()
name="posterior wedge (incl. organizer P)"
def job(pos):
    w = copy.deepcopy(base); w.rng = np.random.default_rng(12345)
    m=damage_masks(base)[name]; w.kill(m)
    log=[]
    for k in range(400):
        if pos and k==pos[2]:
            w._place(pos[0],pos[1], w.wt, clean=True); w.org[pos[:2]] = 3; w.org_born[pos[:2]] = -10**6
            w.cfg.organizer_regen = False
        w.step()
        if k%25==0: log.append((k,int(w.alive.sum()),_orgs(w)['P'], round(similarity(w.pattern(),P0),3)))
    p=w.pattern()
    return pos, np.mean([0]), similarity(p,P0), int(w.alive.sum()), _orgs(w), log, p
with ProcessPoolExecutor(4) as ex:
    res=list(ex.map(job, [(27,18,0),(27,17,0),(27,18,60),None]))
for r in res: print(r[:5]); 
print(res[3][5])
np.save('diag/wedge_final.npy',res[3][6]); np.save('diag/base.npy',P0)
