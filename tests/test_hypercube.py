"""Propositions for the hypercube retarget, as executable assertions.

Same discipline as tests/test_propositions.py: if one of these fails, a claim
about the (Z/2)^L transfer is false and the claim is what should change.
"""

import itertools
import torch

from chroma.hypercube import (AdditiveTransport, FitnessPredictor, WalshRep,
                              cycle_epistasis, cycle_epistasis_expected,
                              subsets_upto, walsh_transform)
from chroma.landscapes import HotspotLandscape, NKLandscape, all_genotypes, split

torch.manual_seed(0)
L = 8
REP = WalshRep(L, subsets_upto(L, 3))
B = 64


def _rand_g(b=B, n=L):
    return torch.randint(0, 2, (b, n)).float()


def test_h0_homomorphism():
    """rho(d1) rho(d2) = rho(d1 XOR d2), exactly."""
    s = torch.randn(B, REP.K)
    d1, d2 = _rand_g(), _rand_g()
    lhs = REP.act(REP.act(s, d2), d1)
    rhs = REP.act(s, REP.compose(d1, d2))
    err = (lhs - rhs).abs().max()
    assert err == 0, f"homomorphism violated: {err}"
    return float(err)


def test_h0b_unitary_and_involutive():
    """rho entries are +/-1, so rho is unitary and rho(d)^2 = I exactly."""
    s = torch.randn(B, REP.K)
    d = _rand_g()
    chi = REP.chi(d)
    assert set(chi.unique().tolist()) <= {1.0, -1.0}
    assert (REP.act(REP.act(s, d), d) - s).abs().max() == 0
    assert (REP.act(s, d).norm(dim=-1) - s.norm(dim=-1)).abs().max() < 1e-5
    return "exact"


def test_h1_cycle_closure_transport_only():
    """Any closed mutational cycle returns exactly under transport alone.

    The SE(2) analogue (Proposition 1) held to float precision; here it holds
    to zero, because the character values are exactly +/-1.
    """
    s = torch.randn(B, REP.K)
    worst = 0.0
    for n in (4, 8, 16, 32):
        acts, acc = [], torch.zeros(B, L)
        for _ in range(n - 1):
            d = _rand_g()
            acts.append(d)
            acc = REP.compose(acc, d)
        acts.append(acc)                      # d^{-1} = d in (Z/2)^L
        cur = s
        for d in acts:
            cur = REP.act(cur, d)
        drift = float((cur - s).abs().max())
        worst = max(worst, drift)
        assert drift == 0, f"cycle length {n} drifted {drift}"
    return worst


def test_h2_double_mutant_cycle_accounting():
    """The measured cycle equals the exact Walsh accounting.

    The naive claim eps_ij = 4 beta_ij is FALSE on a landscape with third-order
    or higher terms: every subset containing both sites contributes. That is
    the background dependence of measured epistasis, and this test pins the
    exact version.
    """
    land = NKLandscape(L=6, K=2, seed=1)
    w = land.true_walsh()
    worst_exact, worst_naive = 0.0, 0.0
    for i, j in itertools.combinations(range(6), 2):
        eps = cycle_epistasis(land.y, i, j, 6)
        worst_exact = max(worst_exact,
                          abs(eps - cycle_epistasis_expected(w, i, j, 6)))
        worst_naive = max(worst_naive,
                          abs(eps - 4 * float(w[(1 << i) | (1 << j)])))
    assert worst_exact < 1e-4, f"exact accounting off by {worst_exact}"
    assert worst_naive > 1e-3, "naive identity should fail on an NK landscape"

    # and the naive form IS exact when the landscape is purely pairwise
    flat = HotspotLandscape(L=6, n_hotspots=3, order=2, seed=7)
    wf = flat.true_walsh()
    worst_pair = max(
        abs(cycle_epistasis(flat.y, i, j, 6) - 4 * float(wf[(1 << i) | (1 << j)]))
        for i, j in itertools.combinations(range(6), 2))
    assert worst_pair < 1e-4, f"pairwise case off by {worst_pair}"
    return dict(exact=round(worst_exact, 8), naive_error=round(worst_naive, 4),
                pairwise_only=round(worst_pair, 8))


def test_h3_walsh_roundtrip():
    """The transform is exact: reconstructing from coefficients recovers y."""
    land = HotspotLandscape(L=7, n_hotspots=3, seed=2)
    w = walsh_transform(land.y, 7)
    G = all_genotypes(7)
    masks = torch.stack([torch.tensor([(i >> k) & 1 for k in range(7)],
                                      dtype=torch.float32)
                         for i in range(2 ** 7)])
    chi = 1.0 - 2.0 * torch.remainder(G @ masks.T, 2.0)
    err = (chi @ w - land.y).abs().max()
    assert err < 1e-4, f"roundtrip error {err}"
    return float(err)


def test_h4_additive_transport_is_given_not_learned():
    """Order-0/1 coefficients recovered from single-mutant assays reproduce the
    additive prediction exactly, with no fitting."""
    land = HotspotLandscape(L=8, n_hotspots=0, seed=3)   # purely additive
    f_wt, f_single = land.singles()
    b0, singles = AdditiveTransport.fit_from_singles(f_wt, f_single)
    add = AdditiveTransport(8, b0, singles, freeze=True)
    err = (add(land.genotypes()) - land.y).abs().max()
    assert err < 1e-4, f"additive landscape not reproduced: {err}"
    n_trainable = sum(p.numel() for p in add.parameters() if p.requires_grad)
    assert n_trainable == 0
    return dict(max_err=float(err), trainable_params=n_trainable)


def test_h6_debiasing_is_load_bearing():
    """The naive single-mutant estimate of beta_k absorbs every interaction
    involving that site.  Without the closed-form correction the 'exact'
    additive part is worse than useless."""
    land = HotspotLandscape(L=8, n_hotspots=3, order=3, hotspot_scale=1.5, seed=4)
    f_wt, f_single = land.singles()
    b0, singles = AdditiveTransport.fit_from_singles(f_wt, f_single)
    out = {}
    for debias in (False, True):
        torch.manual_seed(0)
        m = FitnessPredictor(8, 3, b0, singles, True, 3e-3, debias=debias)
        Gtr, ytr, Gte, yte = split(land, n_train=140, seed=4)
        opt = torch.optim.Adam(m.parameters(), lr=0.05)
        m.train()
        for _ in range(2000):
            opt.zero_grad(); m.loss(Gtr, ytr).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            out[debias] = float(((m(Gte) - yte) ** 2).mean().sqrt())
    assert out[True] < 0.2 * out[False], f"debiasing did not help: {out}"
    return {f"debias={k}": round(v, 4) for k, v in out.items()}


def test_h5_residual_recovers_known_hotspots():
    """On a sparse landscape the learned residual should find the planted
    interactions, not a smear across all subsets."""
    land = HotspotLandscape(L=8, n_hotspots=3, order=3, hotspot_scale=1.5, seed=4)
    f_wt, f_single = land.singles()
    b0, singles = AdditiveTransport.fit_from_singles(f_wt, f_single)
    m = FitnessPredictor(8, order=3, beta0=b0, singles=singles,
                         freeze_additive=True, l0_weight=3e-3)
    Gtr, ytr, Gte, yte = split(land, n_train=140, seed=4)
    opt = torch.optim.Adam(m.parameters(), lr=0.05)
    m.train()
    for _ in range(1500):
        opt.zero_grad()
        m.loss(Gtr, ytr).backward()
        opt.step()
    m.eval()
    found = {s for s, _ in m.residual.hotspots(top=4)}
    truth = {s for s, _ in land.true_hotspots()}
    hit = len(truth & found)
    with torch.no_grad():
        rmse = float(((m(Gte) - yte) ** 2).mean().sqrt())
    assert hit == len(truth), f"recovered {hit}/{len(truth)}: {found} vs {truth}"
    assert rmse < 0.25 * float(land.y.std()), f"test rmse {rmse} too high"
    return dict(recovered=f"{hit}/{len(truth)}", test_rmse=round(rmse, 4),
                landscape_sd=round(float(land.y.std()), 3))


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    w = max(len(t.__name__) for t in tests)
    ok = True
    for t in tests:
        try:
            print(f"PASS  {t.__name__:<{w}}  {t()}")
        except AssertionError as ex:
            ok = False
            print(f"FAIL  {t.__name__:<{w}}  {ex}")
    print("\nALL PASS" if ok else "\nFAILURES PRESENT")
