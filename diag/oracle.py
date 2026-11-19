import copy, sys, numpy as np
sys.path.insert(0, '.')
from sim import Config, World
from experiments import damage_masks, _orgs, GROW_STEPS
from metrics import similarity, aligned_similarity
from concurrent.futures import ProcessPoolExecutor
base = World(Config(seed=0, lam=7.0)).run(GROW_STEPS)
P0 = base.pattern()
def job(pos):
    w = copy.deepcopy(base); w.rng = np.random.default_rng(12345)
    w.kill(damage_masks(base)["core excision (incl. organizer C)"])
    if pos:
        w._place(*pos, w.wt, clean=True); w.org[pos] = 1; w.org_born[pos] = -10**6
        w.cfg.organizer_regen = False
    c=[]
    for k in range(400):
        w.step(); c.append(similarity(w.pattern(), P0))
    return pos, np.mean(c[-50:]), aligned_similarity(w.pattern(), P0), int(w.alive.sum()), _orgs(w)
with ProcessPoolExecutor(4) as ex:
    for r in ex.map(job, [(29,29),(28,30),(27,30),None]): print(r)
