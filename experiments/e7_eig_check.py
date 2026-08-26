"""
E7 - is the information-gain value function sound at all?

E6 measures whether search helps. This measures whether the quantity search is
maximising is even correlated with what it claims to predict. It has to pass
before any E6 number means anything, because MCTS uses EIG as both its prior
and its backup: a broken value function makes the tree numbers meaningless too.

The check is direct. For each state and each legal action:

    predicted  = expected_info_gain(L, d)          what the model claims
    realised   = H[posterior] - H[posterior after actually moving and looking]

If EIG is sound, these correlate positively, and the argmax-EIG action reduces
entropy more than a randomly chosen one. If the correlation is near zero or
negative, the value function is wrong and chroma/search.py is built on sand.

    python experiments/e7_eig_check.py
"""
from __future__ import annotations

import json, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chroma import GroupAction
from chroma.search import ActionSet, InfoGainPolicy
from experiments.e6_search import build


def _entropy(L: torch.Tensor) -> float:
    p = torch.softmax(L.sum(0).reshape(-1), -1)
    return float(-(p * (p + 1e-12).log()).sum())


def _spearman(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 3:
        return 0.0
    ra = torch.tensor(a).argsort().argsort().float()
    rb = torch.tensor(b).argsort().argsort().float()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = (ra.norm() * rb.norm()).clamp(min=1e-9)
    return float((ra * rb).sum() / d)


def run(seed: int = 0, train_steps: int = 1500, n_states: int = 24,
        warm_steps: int = 4) -> dict:
    # build() TRAINS, so it must not sit under no_grad -- the decorator was
    # originally on this function and silently killed the backward pass.
    m, w = build(train_steps, seed)
    with torch.no_grad():
        return _evaluate(m, w, n_states, warm_steps)


def _evaluate(m, w, n_states: int, warm_steps: int) -> dict:
    acts = ActionSet(m.rep)
    pol = InfoGainPolicy(m.rep, m.grid, acts, m.memory)
    print(f"memory nodes: {m.memory.size()}")

    pred_all, real_all, rhos = [], [], []
    best_gain, rand_gain = [], []

    for _ in range(n_states):
        obj = w.sample_object()
        m.bus.reset()
        pose = GroupAction(torch.randint(-3, 4, (1, 2)).float(),
                           torch.randint(0, 4, (1,)) * 2)

        def look(pz):
            sv, sj = [], []
            for off in w.sensor_offsets:
                p = m.rep.compose(pz, off); sv.append(p.v); sj.append(p.j)
            x = w.read(obj, torch.cat(sv), torch.cat(sj))
            return m.encoder(x)

        # warm the posterior so entropy has somewhere to fall from
        for _ in range(warm_steps):
            m.bus.accumulate_batch(look(pose), w.sensor_offsets, m.memory)
            pose = m.rep.compose(acts.get(int(torch.randint(acts.n, (1,)))), pose)

        L0 = m.bus.L.clone()
        h0 = _entropy(L0)
        pred, real = [], []
        for i in range(acts.n):
            d = acts.get(i)
            pred.append(pol.expected_info_gain(L0, d, w.sensor_offsets[0]))
            m.bus.L.copy_(L0)
            m.bus.accumulate_batch(look(m.rep.compose(d, pose)),
                                   w.sensor_offsets, m.memory)
            real.append(h0 - _entropy(m.bus.L))
        m.bus.L.copy_(L0)

        if max(pred) - min(pred) < 1e-9:
            continue                       # EIG is flat here; no ranking to test
        rhos.append(_spearman(pred, real))
        pred_all += pred
        real_all += real
        best_gain.append(real[int(torch.tensor(pred).argmax())])
        rand_gain.append(sum(real) / len(real))

    out = dict(
        n_states=len(rhos),
        spearman_pooled=round(_spearman(pred_all, real_all), 4),
        spearman_per_state=round(sum(rhos) / max(len(rhos), 1), 4),
        gain_argmax_eig=round(sum(best_gain) / max(len(best_gain), 1), 5),
        gain_random_action=round(sum(rand_gain) / max(len(rand_gain), 1), 5),
        memory_nodes=m.memory.size(),
    )
    out["advantage"] = round(out["gain_argmax_eig"] - out["gain_random_action"], 5)
    out["verdict"] = ("EIG is sound" if out["spearman_per_state"] > 0.2
                      and out["advantage"] > 0 else
                      "BROKEN - do not trust E6; search maximises a bad target")
    return out


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
    print("\n" + r["verdict"])
    Path("results").mkdir(exist_ok=True)
    Path("results/e7_eig_check.json").write_text(json.dumps(r, indent=2))
