"""
E5 - does the group buy sample efficiency on a fitness landscape?

Three models on the same combinatorially complete landscape:

    additive     the measured single-mutant effects, no learning at all.
                 A strong baseline: on published benchmarks, plain additive
                 extrapolation beats fancier models on most datasets.
    chroma       debiased additive transport + sparse learned Walsh residual
    mlp          unconstrained network, no group, no additive prior

Pre-registered claim: chroma reaches a given test RMSE from far fewer training
genotypes than the MLP, because additivity is given rather than learned. The
claim FAILS if chroma only ties the additive baseline - that would mean the
epistatic residual bought nothing, which is the same verdict the flat-landscape
arm delivered on the regulatory layer in E1.

    python experiments/e5_hypercube.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chroma.hypercube import AdditiveTransport, FitnessPredictor, MLPBaseline
from chroma.landscapes import HotspotLandscape, NKLandscape, split


def _train(model, Gtr, ytr, steps=2500, lr=0.05):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        opt.zero_grad()
        model.loss(Gtr, ytr).backward()
        opt.step()
    model.eval()
    return model


@torch.no_grad()
def _rmse(model, G, y):
    return float(((model(G) - y) ** 2).mean().sqrt())


def run(L=8, order=3, sizes=(40, 70, 100, 140, 200), seeds=(0, 1, 2),
        kind="hotspot") -> dict:
    out = {}
    for n in sizes:
        acc = {"additive": [], "chroma": [], "mlp": []}
        for s in seeds:
            land = (HotspotLandscape(L=L, n_hotspots=3, order=order,
                                     hotspot_scale=1.5, seed=s)
                    if kind == "hotspot" else NKLandscape(L=L, K=2, seed=s))
            f_wt, f_single = land.singles()
            b0, singles = AdditiveTransport.fit_from_singles(f_wt, f_single)
            Gtr, ytr, Gte, yte = split(land, n_train=n, seed=s)
            sd = float(land.y.std())

            add = AdditiveTransport(L, b0, singles, freeze=True)
            acc["additive"].append(_rmse(add, Gte, yte) / sd)

            m = FitnessPredictor(L, order, b0, singles, True, 3e-3, debias=True)
            acc["chroma"].append(_rmse(_train(m, Gtr, ytr), Gte, yte) / sd)

            mlp = MLPBaseline(L)
            acc["mlp"].append(_rmse(_train(mlp, Gtr, ytr, lr=3e-3), Gte, yte) / sd)
        out[n] = {k: round(sum(v) / len(v), 4) for k, v in acc.items()}
        print(f"n_train={n:4d}  " + "  ".join(
            f"{k}={out[n][k]:.4f}" for k in ("additive", "chroma", "mlp")))
    best = min(out, key=lambda n: out[n]["chroma"])
    out["verdict"] = (
        "group buys sample efficiency"
        if out[best]["chroma"] < 0.5 * min(out[best]["additive"], out[best]["mlp"])
        else "NOT SUPPORTED - the residual did not earn its place")
    return out


if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else "hotspot"
    print(f"=== E5 ({kind} landscape), RMSE normalised by landscape sd ===")
    res = run(kind=kind)
    print("\n" + res["verdict"])
    Path("results").mkdir(exist_ok=True)
    Path(f"results/e5_{kind}.json").write_text(json.dumps(res, indent=2))
