"""Every proposition in the spec, as an executable assertion.

If one of these fails, the corresponding claim in the write-up is false and the
write-up is what should change.
"""

import math
import torch

from chroma import (CHROMA, ChromaConfig, CanonicalMemory, GroupAction, SE2Rep,
                    Differentiation)
from chroma.losses import effective_rank

torch.manual_seed(0)
REP = SE2Rep(n_scalar=16, n_orbits=3, n_rot=8)
B = 32


def _rand_g(b=B):
    return GroupAction(torch.randn(b, 2) * 2.0, torch.randint(0, 8, (b,)))


def test_prop0_homomorphism():
    """rho(g1) rho(g2) = rho(g1 g2)."""
    s = torch.randn(B, REP.D)
    g1, g2 = _rand_g(), _rand_g()
    lhs = REP.act(REP.act(s, g2), g1)
    rhs = REP.act(s, REP.compose(g1, g2))
    err = (lhs - rhs).abs().max()
    assert err < 1e-4, f"homomorphism violated: {err}"
    return float(err)


def test_rho_unitary():
    """||rho(g) s|| = ||s||  -- the premise of Prop 1 and Prop 4."""
    s = torch.randn(B, REP.D)
    g = _rand_g()
    err = (REP.act(s, g).norm(dim=-1) - s.norm(dim=-1)).abs().max()
    assert err < 1e-4, f"rho not unitary: {err}"
    return float(err)


def test_inverse():
    s = torch.randn(B, REP.D)
    g = _rand_g()
    err = (REP.act_inv(REP.act(s, g), g) - s).abs().max()
    assert err < 1e-4
    return float(err)


def test_prop1_loop_closure():
    """Closed path with Delta_phi == 0 returns exactly s, at any length."""
    from chroma.predictor import Predictor
    pred = Predictor(REP, n_z=8, width=16, n_prog=4, rank=2, depth=2)
    for p in pred.parameters():
        torch.nn.init.zeros_(p)
    s = torch.randn(B, REP.D)
    worst = 0.0
    for L in (4, 8, 16, 32):
        acts, acc = [], GroupAction.identity(B)
        for _ in range(L - 1):
            a = _rand_g()
            acts.append(a)
            acc = REP.compose(a, acc)
        acts.append(REP.inverse(acc))
        out = pred.rollout(s, acts, None, None)
        drift = float((out - s).abs().max())
        worst = max(worst, drift)
        assert drift < 1e-3, f"loop length {L} drifted {drift}"
    return worst


def test_prop2_viewpoint_invariance():
    """s_tilde = rho(p)^{-1} s is invariant under global rigid motion h."""
    s = torch.randn(B, REP.D)
    p = _rand_g()
    h = _rand_g()
    canon = REP.act_inv(s, p)
    s2 = REP.act(s, h)
    p2 = REP.compose(h, p)
    canon2 = REP.act_inv(s2, p2)
    err = (canon - canon2).abs().max()
    assert err < 1e-4, f"canonical form not invariant: {err}"
    return float(err)


def test_prop3_bounded_drift():
    """1 - D_m decays exponentially and cumulative drift is finite."""
    diff = Differentiation(4, kappa=5e-2, lam=1e-2, p=2.0, ema=0.5)
    diff.eps_star.fill_(1.0)
    lows, cum = [], 0.0
    for t in range(600):
        diff.update(torch.full((4,), 0.5))     # ebar -> 0.5 = eps*(1-delta), delta=0.5
        lows.append(float(1 - diff.D[0]))
        cum += float(diff.lr_scale()[0])
    assert lows[-1] < 0.05, f"D did not saturate: {lows[-1]}"
    # analytic bound  eta0 G (1-D0)^p / (p kappa delta),  eta0=G=1
    bound = 1.0 / (2.0 * 5e-2 * 0.5)
    assert cum < bound, f"drift {cum:.2f} exceeded bound {bound:.2f}"
    # and it reopens when error spikes
    for _ in range(200):
        diff.update(torch.full((4,), 3.0))
    assert float(diff.D[0]) < 0.9, "gate failed to reopen on error spike"
    return dict(final_gate=lows[-1], cumulative=cum, bound=bound)


def test_prop4_norm_bound():
    """|| s_hat || stays within (1 +/- delta_max) || s ||."""
    from chroma.predictor import Predictor
    dm = 0.3
    pred = Predictor(REP, n_z=8, width=24, n_prog=4, rank=2, depth=2, delta_max=dm)
    with torch.no_grad():
        for p in pred.delta.gate.lin.parameters():
            p.mul_(0).add_(torch.randn_like(p) * 3.0)
    s = torch.randn(B, REP.D)
    z = torch.randn(B, 8)
    out = pred(s, _rand_g(), z, None)
    ratio = out.norm(dim=-1) / s.norm(dim=-1)
    assert ratio.max() < 1 + dm + 1e-3 and ratio.min() > 1 - dm - 1e-3, ratio
    return (float(ratio.min()), float(ratio.max()))


def test_predictor_equivariance():
    """rho(h) . Delta_phi(s) == Delta_phi(rho(h) s)  for h in the C_n subgroup."""
    from chroma.predictor import Predictor
    pred = Predictor(REP, n_z=8, width=24, n_prog=4, rank=2, depth=2)
    with torch.no_grad():          # gate head is zero-init; make the test real
        for p_ in pred.delta.gate.lin.parameters():
            p_.add_(torch.randn_like(p_))
        for p_ in pred.delta.scal.parameters():
            p_.add_(torch.randn_like(p_))
    s = torch.randn(B, REP.D)
    z = torch.randn(B, 8)
    h = _rand_g()
    a = REP.act(pred.delta(s, z, None), h)
    b = pred.delta(REP.act(s, h), z, None)
    assert a.abs().max() > 1e-3, 'vacuous test: Delta_phi is identically zero'
    err = (a - b).abs().max()
    assert err < 1e-3, f"Delta_phi not equivariant: {err}"
    return float(err.detach())


def test_memory_roundtrip():
    mem = CanonicalMemory(REP)
    s = torch.randn(REP.D)
    p = GroupAction(torch.tensor([[1.0, -2.0]]), torch.tensor([3]))
    mem.write(0, s, p)
    got, ok = mem.query(0, p.v, p.j)
    assert bool(ok[0])
    expect = REP.act_inv(s.unsqueeze(0), p).squeeze(0)
    expect = expect / expect.norm()
    assert float(got[0] @ expect) > 0.99
    return mem.size()


def test_attractor_count():
    """RegulatoryState should expose roughly n_proto basins; the MLP ablation
    exposes exactly one (no dynamics -> no hysteresis)."""
    from chroma.regulatory import MLPRegulator, RegulatoryState
    reg = RegulatoryState(n=16, n_proto=6, beta=6.0)
    centres, counts = reg.enumerate_attractors(n_init=400, steps=250, tol=0.5)
    mlp = MLPRegulator(n=16, n_proto=6)
    c2, _ = mlp.enumerate_attractors()
    assert centres.shape[0] >= 2, "landscape is unimodal -- beta too low"
    assert c2.shape[0] == 1
    return dict(chroma_basins=int(centres.shape[0]), ablation_basins=1)


def test_end_to_end():
    from chroma import TactileWorld, TrainConfig, Trainer
    cfg = ChromaConfig(n_modules=4, n_objects=6)
    model = CHROMA(cfg)
    world = TactileWorld(n_regimes=2, objs_per_regime=3, n_sensors=4)
    tc = TrainConfig(steps_pluripotent=3, steps_specification=3,
                     steps_memory_voting=3, steps_continual=3,
                     episode_len=4, log_every=1000)
    tr = Trainer(model, world, tc)
    tr.run()
    n = sum(p.numel() for p in model.parameters())
    return dict(params=n, memory=model.memory.size())


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    width = max(len(t.__name__) for t in tests)
    ok = True
    for t in tests:
        try:
            r = t()
            print(f"PASS  {t.__name__:<{width}}  {r}")
        except AssertionError as ex:
            ok = False
            print(f"FAIL  {t.__name__:<{width}}  {ex}")
    print("\nALL PASS" if ok else "\nFAILURES PRESENT")
