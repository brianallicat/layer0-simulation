import copy, sys, json, numpy as np
sys.path.insert(0, '.')
from sim import Config, World
from experiments import damage_masks, _orgs, GROW_STEPS
from metrics import similarity, aligned_similarity
from concurrent.futures import ProcessPoolExecutor
kw = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
base = World(Config(seed=seed, lam=7.0, **kw)).run(GROW_STEPS)
P0 = base.pattern()
def job(name):
    w = copy.deepcopy(base); w.rng = np.random.default_rng(12345)
    if name != "twin": w.kill(damage_masks(base)[name])
    c=[]
    for k in range(400):
        w.step(); c.append(similarity(w.pattern(), P0))
    return name[:20], round(float(np.mean(c[-50:])),3), round(aligned_similarity(w.pattern(), P0),3), int(w.alive.sum()), _orgs(w)
names = ["twin"] + list(damage_masks(base))
print("base", int(base.alive.sum()), _orgs(base))
with ProcessPoolExecutor(5) as ex:
    for r in ex.map(job, names): print(r)
