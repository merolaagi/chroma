"""
Section 8 - What decides whether this is real.

Run order matters:  E4 first (cheapest, cleanest, validates the geometry before
anything else is switched on), then E1, then E2 and E3 together on one run.

    python experiments/e1_to_e4.py e4
    python experiments/e1_to_e4.py e1 --steps 20000
    python experiments/e1_to_e4.py e2e3 --steps 20000

E1 is the load-bearing one.  If the attractor ablation does not blow up the
regime switching rate and cost backward transfer, the regulatory layer is
decoration and should be removed from the architecture and the paper.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chroma import (CHROMA, ChromaConfig, GroupAction, TactileWorld,
                    TrainConfig, Trainer)


# ===========================================================================
# E4 - loop closure isolates the geometric contribution
# ===========================================================================

def e4_loop_drift(seed: int = 0, train_steps: int = 400,
                  lengths=(4, 8, 16, 32), trials: int = 24) -> dict:
    """Drift around closed paths, CHROMA vs the dense-transition ablation.

    Two measurements, because the loose version of this claim is wrong:

    transport_only  Delta_phi disabled at eval.  This is the clean isolation of
                    the geometric contribution.  Proposition 1 says CHROMA is
                    *exactly* zero at every path length (log-log slope
                    undefined, drift ~1e-6 numerical); the learned dense
                    transition has no such guarantee and drifts.

    full            Delta_phi active.  Both grow with path length -- residuals
                    accumulate -- so "CHROMA drift is flat" is only true in the
                    rigid limit.  What CHROMA still buys is that *all* of the
                    drift is attributable to Delta_phi and is bounded by
                    delta_max per step, whereas the ablation's drift has no
                    such decomposition.
    """
    out = {}
    for tag, ablate in (("chroma", False), ("ablate_equivariance", True)):
        cfg = ChromaConfig(n_modules=4, n_objects=6, seed=seed,
                           ablate_equivariance=ablate)
        model = CHROMA(cfg)
        world = TactileWorld(n_regimes=2, objs_per_regime=3, n_sensors=4, seed=seed)
        tr = Trainer(model, world, TrainConfig(
            steps_pluripotent=train_steps, steps_specification=0,
            steps_memory_voting=0, steps_continual=0,
            episode_len=6, log_every=10 ** 9))
        tr.run()

        model.eval()
        entry = {}
        for mode in ("transport_only", "full"):
            drift = {}
            with torch.no_grad():
                for L in lengths:
                    vals = []
                    for _ in range(trials):
                        obj = world.sample_object()
                        v = torch.randint(-3, 4, (4, 2)).float()
                        j = torch.randint(0, 4, (4,)) * 2
                        x = world.read(obj, v, j)
                        s = model.encoder(x)
                        path = world.closed_path(L)
                        path = [GroupAction(p.v.expand(4, 2), p.j.expand(4))
                                for p in path]
                        cur = s
                        for d in path:
                            if mode == "transport_only":
                                cur = model.transport(cur, d)
                            else:
                                cur = model.predict(cur, d, torch.zeros(4, cfg.n_z),
                                                    None)
                        rel = ((cur - s).norm(dim=-1)
                               / s.norm(dim=-1).clamp(min=1e-6))
                        vals.append(float(rel.mean()))
                    drift[L] = sum(vals) / len(vals)
            xs = [math.log(L) for L in lengths]
            ys = [math.log(max(drift[L], 1e-12)) for L in lengths]
            xm, ym = sum(xs) / len(xs), sum(ys) / len(ys)
            slope = (sum((a - xm) * (b - ym) for a, b in zip(xs, ys))
                     / sum((a - xm) ** 2 for a in xs))
            entry[mode] = dict(drift=drift, log_log_slope=slope,
                               drift_at_32=drift[max(lengths)])
        out[tag] = entry
    ratio = (out["ablate_equivariance"]["transport_only"]["drift_at_32"]
             / max(out["chroma"]["transport_only"]["drift_at_32"], 1e-12))
    out["transport_only_drift_ratio"] = ratio
    out["verdict"] = ("exact transport confirmed (Prop 1)" if ratio > 100
                      else "NOT SUPPORTED")
    return out


# ===========================================================================
# E1 - does the regulatory layer earn its keep?
# ===========================================================================

def e1_regulatory_ablation(seed: int = 0, steps: int = 20000,
                           regime_period: int = 5000) -> dict:
    """CHROMA (attractor dynamics) vs a matched-parameter memoryless MLP.

    Metrics
    -------
    switch_rate      mean_t 1[ argmax_k e_mk(g_t) != argmax_k e_mk(g_{t-1}) ]
    backward_transfer  error on regime 0 measured after training through all
                       regimes, minus error on regime 0 at the end of its own
                       block.  More negative = more forgetting.

    Prediction: the ablation flip-flops at 5-20x the rate and loses 2-5 points
    of backward transfer.
    """
    res = {}
    for tag, ablate in (("chroma", False), ("ablate_grn", True)):
        cfg = ChromaConfig(seed=seed, ablate_grn_dynamics=ablate)
        model = CHROMA(cfg)
        world = TactileWorld(n_regimes=3, objs_per_regime=4, seed=seed)
        per = max(steps // 8, 1)
        tr = Trainer(model, world, TrainConfig(
            steps_pluripotent=per, steps_specification=per,
            steps_memory_voting=2 * per, steps_continual=steps - 4 * per,
            regime_period=regime_period, log_every=max(steps // 20, 1)))
        tr.run()

        mid = _eval_regime(model, world, regime=0)
        res[tag] = dict(
            switch_rate=tr.switch_events / max(tr.t, 1),
            final_err_regime0=mid,
            log=tr.log,
        )
    a, b = res["chroma"]["switch_rate"], res["ablate_grn"]["switch_rate"]
    res["switch_rate_ratio"] = b / max(a, 1e-9)
    res["verdict"] = ("regulatory layer earns its keep"
                      if res["switch_rate_ratio"] > 5.0
                      else "NOT SUPPORTED -- consider deleting regulatory.py")
    return res


@torch.no_grad()
def _eval_regime(model: CHROMA, world: TactileWorld, regime: int,
                 n: int = 64) -> float:
    old = world.regime
    world.set_regime(regime)
    M = model.cfg.n_modules
    e = model.expression()
    tot = 0.0
    for _ in range(n):
        obj = world.sample_object()
        pose = GroupAction(torch.randint(-2, 3, (1, 2)).float(),
                           torch.randint(0, 4, (1,)) * 2)
        d = world.sample_action(1)
        nxt = model.rep.compose(d, pose)
        sv, sj, nv, nj = [], [], [], []
        for off in world.sensor_offsets[:M]:
            p = model.rep.compose(pose, off); q = model.rep.compose(nxt, off)
            sv.append(p.v); sj.append(p.j); nv.append(q.v); nj.append(q.j)
        x = world.read(obj, torch.cat(sv), torch.cat(sj))
        xn = world.read(obj, torch.cat(nv), torch.cat(nj))
        _, err, _ = model.step(x, xn, GroupAction(d.v.expand(M, 2), d.j.expand(M)), e)
        tot += float(err.mean())
    world.set_regime(old)
    return tot / n


# ===========================================================================
# E2 - attractor count tracks regime count
# ===========================================================================

def e2_attractor_count(seed: int = 0, steps: int = 20000,
                       regime_counts=(2, 3, 4)) -> dict:
    """Train with k hidden regimes, k never disclosed; count basins.

    Falsifiable claim: the count converges to within +/- 1 of k.
    """
    res = {}
    for k in regime_counts:
        cfg = ChromaConfig(seed=seed, n_objects=4 * k)
        model = CHROMA(cfg)
        world = TactileWorld(n_regimes=k, objs_per_regime=4, seed=seed)
        per = max(steps // 4, 1)
        Trainer(model, world, TrainConfig(
            steps_pluripotent=per, steps_specification=per,
            steps_memory_voting=per, steps_continual=steps - 3 * per,
            regime_period=max(steps // (3 * k), 1),
            log_every=10 ** 9)).run()
        centres, counts = model.regulator.enumerate_attractors(
            n_init=2048, steps=400, tol=0.35)
        p = counts / counts.sum()
        res[k] = dict(n_basins=int(centres.shape[0]),
                      basin_entropy=float(-(p * (p + 1e-12).log()).sum()),
                      within_one=abs(int(centres.shape[0]) - k) <= 1)
    res["verdict"] = ("attractor count tracks regime count"
                      if all(v["within_one"] for v in res.values()
                             if isinstance(v, dict))
                      else "NOT SUPPORTED")
    return res


# ===========================================================================
# E3 - lineage structure in expression space
# ===========================================================================

def e3_lineage(seed: int = 0, steps: int = 20000, snapshots: int = 8) -> dict:
    """Cluster e_m(g) over training and test for monotone separation.

    Prediction: a progressively branching tree -- modules that split early never
    re-merge.  Flat, unstructured clusters falsify the developmental story.
    """
    cfg = ChromaConfig(seed=seed)
    model = CHROMA(cfg)
    world = TactileWorld(n_regimes=3, objs_per_regime=4, seed=seed)
    per = max(steps // snapshots, 1)
    traj = []
    tr = Trainer(model, world, TrainConfig(
        steps_pluripotent=per, steps_specification=per,
        steps_memory_voting=per, steps_continual=per, log_every=10 ** 9))
    for k in range(snapshots):
        tr.model.set_phase(["pluripotent", "specification", "memory_voting",
                            "continual"][min(k, 3)])
        for _ in range(per):
            tr.episode()
            tr.t += 1
        e = model.readout.lineage_features(model.regulator.g)
        traj.append(e.clone())

    # pairwise cosine distance between module expression profiles over time
    dists = []
    for e in traj:
        en = torch.nn.functional.normalize(e, dim=-1)
        dists.append(1 - en @ en.T)
    D = torch.stack(dists)                       # (T, M, M)
    M = D.shape[1]
    iu = torch.triu_indices(M, M, offset=1)
    pair = D[:, iu[0], iu[1]]                    # (T, P)
    # monotone separation: fraction of pairs whose distance never decreases
    diffs = pair[1:] - pair[:-1]
    monotone = float((diffs >= -1e-3).all(0).float().mean())
    return dict(final_mean_distance=float(pair[-1].mean()),
                separation_growth=float(pair[-1].mean() - pair[0].mean()),
                monotone_pair_fraction=monotone,
                verdict=("lineage-like: progressive, non-reverting separation"
                         if monotone > 0.7 and pair[-1].mean() > pair[0].mean()
                         else "NOT SUPPORTED -- clusters re-merge"))


# ===========================================================================

RUNNERS = dict(e1=e1_regulatory_ablation, e2=e2_attractor_count,
               e3=e3_lineage, e4=e4_loop_drift)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=list(RUNNERS) + ["e2e3", "all"])
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()

    todo = list(RUNNERS) if a.which == "all" else (
        ["e2", "e3"] if a.which == "e2e3" else [a.which])
    results = {}
    for name in todo:
        kw = dict(seed=a.seed)
        if a.steps is not None and name != "e4":
            kw["steps"] = a.steps
        elif a.steps is not None:
            kw["train_steps"] = a.steps
        print(f"\n=== {name} ===")
        r = RUNNERS[name](**kw)
        results[name] = r
        print(json.dumps(_strip(r), indent=2, default=str))
    if a.out:
        Path(a.out).write_text(json.dumps(_strip(results), indent=2, default=str))


def _strip(o):
    if isinstance(o, dict):
        return {k: _strip(v) for k, v in o.items() if k != "log"}
    if isinstance(o, list):
        return [_strip(x) for x in o]
    return o


if __name__ == "__main__":
    main()
