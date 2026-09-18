"""
E9 - is 0.287 a model failure or a task ceiling?

E8 put recognition at 0.287 against a 1/12 = 0.083 chance baseline, and every
attempt to improve it by changing the model (memory capacity, evidence
normalisation, action policy) has produced little or nothing. Before touching
the model again, it is worth asking whether 0.287 is anywhere near what is
*achievable*.

That question is answerable exactly, because the generative model is fully
known: the canvases are ours, the sensor is deterministic apart from additive
Gaussian noise, and the agent's own displacements are known. So a Bayes-optimal
observer can be written down directly.

    P(object o | patches) proportional to
        sum over initial poses p0 of  prod over t  N(x_t ; patch(o, p0 . d_1..t), sigma)

Enumerating every (object, initial pose) hypothesis and scoring the true
likelihood gives the ceiling. Interpretation:

    oracle ~ 0.30   the task itself is near-impossible at this patch size and
                    step budget; CHROMA is close to optimal and the thing to
                    change is the WORLD (bigger receptive field, more distinct
                    shapes, more touches), not the architecture.

    oracle ~ 0.95   CHROMA has a real gap, and the architecture is where the
                    work is.

This is the diagnostic that should have been run before E6 and E7.

    python experiments/e9_ceiling.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chroma import GroupAction, SE2Rep, TactileWorld


def _hypotheses(rep: SE2Rep, extent: int = 4) -> GroupAction:
    """Every initial pose the agent could have started from."""
    xs = torch.arange(-extent, extent + 1, dtype=torch.float32)
    vs = torch.stack(torch.meshgrid(xs, xs, indexing="ij"), -1).reshape(-1, 2)
    js = torch.tensor([0, 2, 4, 6])
    v = vs.repeat_interleave(len(js), 0)
    j = js.repeat(vs.shape[0])
    return GroupAction(v, j)


@torch.no_grad()
def oracle_accuracy(w: TactileWorld, rep: SE2Rep, n_ep: int = 60,
                    touches: int = 12, sigma: float | None = None) -> dict:
    """Exact Bayesian recognition given the true generative model."""
    sigma = w.sensor_noise if sigma is None else sigma
    sigma = max(float(sigma), 1e-3)
    H = _hypotheses(rep)
    n_hyp = H.batch
    hits = hits_pose = 0

    for _ in range(n_ep):
        obj = w.sample_object()
        pose = GroupAction(torch.randint(-3, 4, (1, 2)).float(),
                           torch.randint(0, 4, (1,)) * 2)
        true0 = GroupAction(pose.v.clone(), pose.j.clone())

        # log-likelihood of every (object, initial pose) hypothesis
        ll = torch.zeros(w.n_objects, n_hyp)
        cur_h = GroupAction(H.v.clone(), H.j.clone())
        for _t in range(touches):
            # what the agent actually felt, at all sensors
            sv, sj = [], []
            for off in w.sensor_offsets:
                p = rep.compose(pose, off)
                sv.append(p.v); sj.append(p.j)
            x = w.read(obj, torch.cat(sv), torch.cat(sj))          # (M, P*P)

            # what each hypothesis predicts, at all sensors
            for k, off in enumerate(w.sensor_offsets):
                ob = GroupAction(off.v.expand(n_hyp, 2), off.j.expand(n_hyp))
                q = rep.compose(cur_h, ob)
                for o in range(w.n_objects):
                    pred = w.read(o, q.v, q.j)                     # (n_hyp, P*P)
                    ll[o] -= ((pred - x[k]) ** 2).sum(-1) / (2 * sigma ** 2)

            d = w.sample_action(1)
            pose = rep.compose(d, pose)
            db = GroupAction(d.v.expand(n_hyp, 2), d.j.expand(n_hyp))
            cur_h = rep.compose(db, cur_h)

        post = torch.softmax(ll.reshape(-1), -1).reshape(w.n_objects, n_hyp)
        hits += int(post.sum(-1).argmax()) == obj
        flat = int(post.argmax())
        if flat // n_hyp == obj:
            h = flat % n_hyp
            same = (torch.allclose(H.v[h], true0.v[0], atol=1.01)
                    and int(H.j[h]) == int(true0.j[0]))
            hits_pose += int(same)

    return dict(oracle_acc=round(hits / n_ep, 4),
                oracle_acc_with_pose=round(hits_pose / n_ep, 4),
                n_hypotheses=n_hyp, touches=touches, sigma=round(sigma, 4))


def run(touch_budgets=(4, 12, 24), n_ep: int = 40) -> dict:
    rep = SE2Rep(16, 3, 8)
    w = TactileWorld(n_regimes=3, objs_per_regime=4, seed=0)
    w.bind_rep(rep)
    chance = 1.0 / w.n_objects
    out: dict = {"chance": round(chance, 4), "chroma_e8_acc": 0.287}
    print(f"chance = {chance:.3f}   CHROMA (E8) = 0.287\n")
    print(f"{'touches':>8} {'oracle acc':>11} {'x chance':>9} {'+pose':>7}")
    for t in touch_budgets:
        r = oracle_accuracy(w, rep, n_ep=n_ep, touches=t)
        out[f"touches={t}"] = r
        print(f"{t:>8} {r['oracle_acc']:>11.3f} {r['oracle_acc']/chance:>9.2f} "
              f"{r['oracle_acc_with_pose']:>7.3f}")
    best = max(out[f"touches={t}"]["oracle_acc"] for t in touch_budgets)
    out["ceiling"] = best
    out["chroma_gap"] = round(best - 0.287, 4)
    out["verdict"] = (
        "TASK CEILING - CHROMA is near optimal; change the world, not the model"
        if best < 0.45 else
        "MODEL GAP - the architecture is leaving accuracy on the table")
    return out


if __name__ == "__main__":
    r = run()
    print(f"\nceiling {r['ceiling']:.3f}  CHROMA 0.287  gap {r['chroma_gap']:+.3f}")
    print(r["verdict"])
    Path("results").mkdir(exist_ok=True)
    Path("results/e9_ceiling.json").write_text(json.dumps(r, indent=2))
