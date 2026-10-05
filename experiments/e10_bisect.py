"""
E10 - where does the 0.71 recognition gap live?

E9 established that a Bayes-optimal observer reaches 1.000 accuracy in three
touches while CHROMA manages 0.287 in twelve. The gap is the architecture's, not
the task's. This experiment localises it.

The oracle differs from CHROMA in exactly three independent places:

    representation   raw 9-d patches          vs  learned 64-d latents
    hypothesis set   all 324 poses            vs  the 45-cell grid (extent 2)
    likelihood       exact Gaussian on error  vs  cosine similarity / tau

Each is a binary switch, so all eight combinations can be run. The oracle is
(raw, full, gaussian); CHROMA is (latent, grid, cosine). Whichever switch moves
accuracy most is where the defect lives.

A prior worth stating before looking: the trainer samples agent poses over
+/-4 (``TactileWorld.pose_range``), while ``HypothesisGrid`` spans +/-2. If the
true pose is routinely outside the hypothesis set, no evidence can ever
accumulate on it and accuracy is capped structurally, whatever the encoder does.

    python experiments/e10_bisect.py
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chroma import CHROMA, ChromaConfig, GroupAction, TactileWorld  # noqa: E402
from chroma.train import TrainConfig, Trainer  # noqa: E402


def pose_grid(extent: int) -> GroupAction:
    xs = torch.arange(-extent, extent + 1, dtype=torch.float32)
    vs = torch.stack(torch.meshgrid(xs, xs, indexing="ij"), -1).reshape(-1, 2)
    js = torch.tensor([0, 2, 4, 6])
    return GroupAction(vs.repeat_interleave(len(js), 0), js.repeat(vs.shape[0]))


def train(steps: int, seed: int):
    cfg = ChromaConfig(seed=seed)
    m = CHROMA(cfg)
    w = TactileWorld(n_regimes=3, objs_per_regime=4, seed=seed)
    w.bind_rep(m.rep)
    per = max(steps // 4, 1)
    Trainer(m, w, TrainConfig(steps_pluripotent=per, steps_specification=per,
                              steps_memory_voting=per, steps_continual=per,
                              log_every=10 ** 9)).run()
    return m, w


@torch.no_grad()
def build_templates(m, w, H: GroupAction, latent: bool):
    """What every (object, pose, sensor) would look like, from ground truth.

    IMPORTANT, and the reason this experiment's conclusion did not transfer.
    Building templates from the world rather than from the learned memory was
    meant to isolate the *comparison* from memory coverage. It does -- but that
    makes this an experiment about an idealised recogniser, not about CHROMA.

    The bisection found the hypothesis set dominant (0.200 -> 0.700 from
    widening the grid alone). End to end, through the real memory, widening the
    grid moved nothing: 0.250 at extent 2, 0.250 at extent 4. Whatever limits
    the shipped pipeline sits upstream of everything measured here.

    An ablation that removes the suspected bottleneck in order to study
    something else cannot then be used to predict the un-ablated system.
    """
    n_hyp, M = H.batch, m.cfg.n_modules
    out = []
    for k, off in enumerate(w.sensor_offsets):
        ob = GroupAction(off.v.expand(n_hyp, 2), off.j.expand(n_hyp))
        q = m.rep.compose(H, ob)
        per_obj = []
        for o in range(w.n_objects):
            x = w.read(o, q.v, q.j)                      # (n_hyp, 9)
            per_obj.append(m.encoder(x) if latent else x)
        out.append(torch.stack(per_obj))                 # (O, n_hyp, D)
    return out                                           # list over sensors


@torch.no_grad()
def recognise(m, w, latent: bool, extent: int, gaussian: bool,
              n_ep: int = 40, touches: int = 12, tau: float = 0.10,
              sigma: float = 0.02) -> float:
    H = pose_grid(extent)
    n_hyp = H.batch
    T = build_templates(m, w, H, latent)
    hits = 0
    for _ in range(n_ep):
        obj = w.sample_object()
        pose = GroupAction(torch.randint(-3, 4, (1, 2)).float(),
                           torch.randint(0, 4, (1,)) * 2)
        ll = torch.zeros(w.n_objects, n_hyp)
        cur = GroupAction(H.v.clone(), H.j.clone())
        for _t in range(touches):
            for k, off in enumerate(w.sensor_offsets):
                p = m.rep.compose(pose, off)
                x = w.read(obj, p.v, p.j)
                obs = (m.encoder(x) if latent else x)[0]      # (D,)
                tpl = T[k]                                    # (O, n_hyp, D)
                # templates were built at H; shift them to the current step by
                # rebuilding is too costly, so index by the transported grid
                if gaussian:
                    ll += -((tpl - obs) ** 2).sum(-1) / (2 * sigma ** 2)
                else:
                    ll += (F.normalize(tpl, dim=-1)
                           @ F.normalize(obs, dim=-1)) / tau
            d = w.sample_action(1)
            pose = m.rep.compose(d, pose)
            db = GroupAction(d.v.expand(n_hyp, 2), d.j.expand(n_hyp))
            cur = m.rep.compose(db, cur)
            T = build_templates(m, w, cur, latent)
        post = torch.softmax(ll.reshape(-1), -1).reshape(w.n_objects, n_hyp)
        hits += int(post.sum(-1).argmax()) == obj
    return hits / n_ep


def run(steps: int = 700, seed: int = 0, n_ep: int = 30,
        touches: int = 6) -> dict:
    m, w = train(steps, seed)
    chance = 1.0 / w.n_objects
    out: dict = {"chance": round(chance, 4)}
    print(f"chance = {chance:.3f}   (CHROMA as shipped = latent/grid/cosine)\n")
    print(f"{'repr':>7} {'poses':>6} {'likelihood':>11} {'acc':>7} {'x chance':>9}")
    for latent, extent, gaussian in itertools.product(
            (True, False), (2, 4), (False, True)):
        a = recognise(m, w, latent, extent, gaussian, n_ep=n_ep, touches=touches)
        key = f"{'latent' if latent else 'raw'}/{'grid' if extent == 2 else 'full'}/" \
              f"{'gauss' if gaussian else 'cosine'}"
        out[key] = round(a, 4)
        print(f"{'latent' if latent else 'raw':>7} "
              f"{'±2' if extent == 2 else '±4':>6} "
              f"{'gaussian' if gaussian else 'cosine':>11} {a:>7.3f} "
              f"{a / chance:>9.2f}")

    base = out["latent/grid/cosine"]
    out["delta_representation"] = round(out["raw/grid/cosine"] - base, 4)
    out["delta_hypothesis_set"] = round(out["latent/full/cosine"] - base, 4)
    out["delta_likelihood"] = round(out["latent/grid/gauss"] - base, 4)
    out["oracle"] = out["raw/full/gauss"]
    deltas = {"representation": out["delta_representation"],
              "hypothesis set": out["delta_hypothesis_set"],
              "likelihood": out["delta_likelihood"]}
    out["dominant"] = max(deltas, key=deltas.get)
    return out


if __name__ == "__main__":
    r = run()
    print(f"\nsingle-switch gains from the shipped configuration:")
    for k in ("representation", "hypothesis_set", "likelihood"):
        print(f"  {k:16s} {r['delta_' + k]:+.4f}")
    print(f"\noracle {r['oracle']:.3f}   dominant factor: {r['dominant']}")
    Path("results").mkdir(exist_ok=True)
    Path("results/e10_bisect.json").write_text(json.dumps(r, indent=2))
