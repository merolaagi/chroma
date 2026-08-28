"""Run one arm of E1 and checkpoint the result, so the experiment can span
sessions on a slow machine."""
import json, sys, time
sys.path.insert(0, '.')
import torch
from chroma import CHROMA, ChromaConfig, TactileWorld, Trainer, TrainConfig
from experiments.e1_to_e4 import _eval_regime

arm = sys.argv[1]        # "chroma" | "ablate_grn" | "ablate_hysteresis"
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 800
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0

cfg = ChromaConfig(seed=seed,
                   ablate_grn_dynamics=(arm == 'ablate_grn'),
                   ablate_hysteresis=(arm == 'ablate_hysteresis'))
model = CHROMA(cfg)
world = TactileWorld(n_regimes=3, objs_per_regime=4, seed=seed)
per = max(steps // 8, 1)
tr = Trainer(model, world, TrainConfig(
    steps_pluripotent=per, steps_specification=per,
    steps_memory_voting=2 * per, steps_continual=steps - 4 * per,
    regime_period=max(steps // 6, 1), log_every=max(steps // 8, 1)))
t0 = time.time()
tr.run()

# backward transfer: error on regime 0 after training through all regimes
res = dict(arm=arm, seed=seed, steps=steps,
           switch_rate=tr.switch_events / max(tr.t, 1),
           err_regime0=_eval_regime(model, world, 0),
           err_regime1=_eval_regime(model, world, 1),
           err_regime2=_eval_regime(model, world, 2),
           mean_D=float(model.diff.D.mean()),
           memory=model.memory.size(),
           wall_s=round(time.time() - t0, 1),
           log=tr.log)
if hasattr(model.regulator, 'enumerate_attractors'):
    c, _ = model.regulator.enumerate_attractors(n_init=512, steps=250, tol=0.35)
    res['n_basins'] = int(c.shape[0])
json.dump(res, open(f'results/e1_{arm}_s{seed}{"_smoke" if steps < 10000 else ""}.json', 'w'), indent=2, default=str)
print("SAVED", arm, {k: v for k, v in res.items() if k != 'log'})
