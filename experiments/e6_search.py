"""
E6 - does search buy faster recognition?

Three policies, matched task, measured as steps-to-recognition:

    random      what the trainer did before: uniform over displacements
    greedy_eig  one-step expected information gain
    mcts        PUCT tree search over action sequences, valued by info gain

Pre-registered claim: MCTS cuts steps-to-recognition by 2x relative to random.
It FAILS if it only ties greedy_eig -- that would mean the tree is buying
nothing over one-step lookahead and the honest answer is greedy.

Report includes wall-clock per decision, because a 2x reduction in steps that
costs 50x per step is not a win.
"""
from __future__ import annotations

import json, sys, time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chroma import CHROMA, ChromaConfig, GroupAction, TactileWorld, Trainer, TrainConfig
from chroma.search import ActionSet, InfoGainPolicy, MCTSPolicy, RandomPolicy


def build(steps=1500, seed=0):
    """Train a model far enough to have a usable object memory."""
    cfg = ChromaConfig(seed=seed)
    m = CHROMA(cfg)
    w = TactileWorld(n_regimes=3, objs_per_regime=4, seed=seed)
    per = max(steps // 4, 1)
    tr = Trainer(m, w, TrainConfig(steps_pluripotent=per, steps_specification=per,
                                   steps_memory_voting=per, steps_continual=per,
                                   log_every=10 ** 9))
    tr.run()
    return m, w


@torch.no_grad()
def episode(m, w, policy, max_steps=20, conf_target=0.60):
    """Steps until the pooled posterior puts conf_target on one object."""
    obj = w.sample_object()
    m.bus.reset()
    pose = GroupAction(torch.randint(-3, 4, (1, 2)).float(),
                       torch.randint(0, 4, (1,)) * 2)
    M = m.cfg.n_modules
    t0 = time.time()
    for step in range(max_steps):
        sv, sj = [], []
        for off in w.sensor_offsets:
            p = m.rep.compose(pose, off); sv.append(p.v); sj.append(p.j)
        x = w.read(obj, torch.cat(sv), torch.cat(sj))
        s = m.encoder(x)
        m.bus.accumulate_batch(s, w.sensor_offsets, m.memory)
        post = torch.softmax(m.bus.L.sum(0).reshape(-1), -1).reshape(m.cfg.n_objects, -1)
        by_obj = post.sum(-1)
        if float(by_obj.max()) >= conf_target:
            return step + 1, int(by_obj.argmax()) == obj, time.time() - t0
        d = policy.select(m.bus.L, w.sensor_offsets[0])
        pose = m.rep.compose(d, pose)
    return max_steps, False, time.time() - t0


def run(seeds=(0, 1), n_ep=12, train_steps=1500) -> dict:
    out = {}
    for seed in seeds:
        m, w = build(train_steps, seed)
        acts = ActionSet(m.rep)
        pols = [RandomPolicy(acts),
                InfoGainPolicy(m.rep, m.grid, acts, m.memory),
                MCTSPolicy(m.rep, m.grid, acts, m.memory, n_sim=16, depth=3)]
        for p in pols:
            steps, correct, secs = [], [], []
            for _ in range(n_ep):
                st, ok, sec = episode(m, w, p)
                steps.append(st); correct.append(ok); secs.append(sec)
            r = out.setdefault(p.name, {"steps": [], "acc": [], "s_per_ep": []})
            r["steps"].append(sum(steps) / len(steps))
            r["acc"].append(sum(correct) / len(correct))
            r["s_per_ep"].append(sum(secs) / len(secs))
    res = {}
    for k, v in out.items():
        res[k] = {kk: round(sum(vv) / len(vv), 4) for kk, vv in v.items()}
        print(f"{k:12s} steps={res[k]['steps']:6.2f}  acc={res[k]['acc']:.2f}  "
              f"sec/ep={res[k]['s_per_ep']:.3f}")
    rs, ms, gs = res["random"]["steps"], res["mcts"]["steps"], res["greedy_eig"]["steps"]
    res["speedup_vs_random"] = round(rs / max(ms, 1e-9), 2)
    res["mcts_vs_greedy"] = round(gs / max(ms, 1e-9), 2)
    res["verdict"] = ("search buys recognition speed"
                      if res["speedup_vs_random"] >= 2.0 and res["mcts_vs_greedy"] > 1.1
                      else "NOT SUPPORTED - greedy or random is enough")
    return res


if __name__ == "__main__":
    r = run()
    print(f"\nspeedup vs random {r['speedup_vs_random']}x   "
          f"mcts vs greedy {r['mcts_vs_greedy']}x\n{r['verdict']}")
    Path("results").mkdir(exist_ok=True)
    Path("results/e6_search.json").write_text(json.dumps(r, indent=2))
