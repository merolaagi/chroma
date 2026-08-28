"""
E8 - does recognition work at all?

The foundation test. No search, no regulatory arms, no ablations. One question:
given a trained model, how well can it identify an object by touching it?

Everything else in this repo sits on top of this number. E1 compared three
regulatory variants and E6 compared three action policies -- but with
recognition at 0.21 against a 1/12 = 0.083 chance baseline, both were comparing
variants of a system that barely works. This measures the base directly, and
reports it as a multiple of chance, because 0.21 sounds like a number until you
know the baseline is 0.083.

Also sweeps the memory settings, since object memory hit its 20,000 cap in every
E1 arm -- usually by step 10,000, so half of each run had frozen memory. That is
the leading suspect for weak recognition.

    python experiments/e8_recognition.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chroma import CHROMA, ChromaConfig, GroupAction, TactileWorld, Trainer, TrainConfig


def train(steps: int, seed: int, max_per_cell: int, capacity: int):
    cfg = ChromaConfig(seed=seed)
    m = CHROMA(cfg)
    m.memory.max_per_cell = max_per_cell
    m.memory.capacity = capacity
    w = TactileWorld(n_regimes=3, objs_per_regime=4, seed=seed)
    per = max(steps // 4, 1)
    Trainer(m, w, TrainConfig(steps_pluripotent=per, steps_specification=per,
                              steps_memory_voting=per, steps_continual=per,
                              log_every=10 ** 9)).run()
    return m, w


@torch.no_grad()
def accuracy(m, w, n_ep: int = 40, touches: int = 12) -> tuple[float, float]:
    """Top-1 object accuracy after a fixed number of random touches."""
    hits, conf = 0, 0.0
    for _ in range(n_ep):
        obj = w.sample_object()
        m.bus.reset()
        pose = GroupAction(torch.randint(-3, 4, (1, 2)).float(),
                           torch.randint(0, 4, (1,)) * 2)
        for _ in range(touches):
            sv, sj = [], []
            for off in w.sensor_offsets:
                p = m.rep.compose(pose, off)
                sv.append(p.v)
                sj.append(p.j)
            s = m.encoder(w.read(obj, torch.cat(sv), torch.cat(sj)))
            m.bus.accumulate_batch(s, w.sensor_offsets, m.memory)
            pose = m.rep.compose(w.sample_action(1), pose)
        by_obj = torch.softmax(m.bus.L.sum(0).reshape(-1), -1)
        by_obj = by_obj.reshape(m.cfg.n_objects, -1).sum(-1)
        hits += int(by_obj.argmax()) == obj
        conf += float(by_obj.max())
    return hits / n_ep, conf / n_ep


def run(steps: int = 1500, seeds=(0, 1),
        configs=((3, 20000), (1, 60000), (2, 60000))) -> dict:
    chance = 1.0 / 12
    out: dict = {}
    print(f"chance baseline = {chance:.3f}\n")
    print(f"{'max/cell':>9} {'capacity':>9} {'nodes':>7} {'acc':>6} "
          f"{'x chance':>9} {'conf':>6}")
    for mpc, cap in configs:
        accs, nodes, confs = [], [], []
        for sd in seeds:
            m, w = train(steps, sd, mpc, cap)
            a, c = accuracy(m, w)
            accs.append(a)
            nodes.append(m.memory.size())
            confs.append(c)
        a = sum(accs) / len(accs)
        key = f"max_per_cell={mpc},cap={cap}"
        out[key] = dict(acc=round(a, 4), x_chance=round(a / chance, 2),
                        nodes=int(sum(nodes) / len(nodes)),
                        conf=round(sum(confs) / len(confs), 4))
        print(f"{mpc:>9} {cap:>9} {out[key]['nodes']:>7} {a:>6.3f} "
              f"{out[key]['x_chance']:>9.2f} {out[key]['conf']:>6.3f}")
    best = max(out, key=lambda k: out[k]["acc"])
    out["best"] = best
    out["verdict"] = ("recognition works" if out[best]["acc"] > 0.60
                      else "STILL BROKEN - recognition is the bottleneck")
    return out


if __name__ == "__main__":
    r = run()
    print("\n" + r["verdict"])
    Path("results").mkdir(exist_ok=True)
    Path("results/e8_recognition.json").write_text(json.dumps(r, indent=2))
