r"""
CHROMA on the mutation hypercube.

The same architecture, different group.  In the tactile world the group is
SE(2) and rho(d) is rigid transport.  On a combinatorially complete fitness
landscape the group is

    G = (Z/2)^L        genotypes {0,1}^L, mutations act by XOR

and rho(d) is the diagonal matrix of Walsh characters.  Everything that made
the robot version work carries over exactly, because (Z/2)^L is a genuine group
with a genuine exact representation -- unlike, say, an antigenic map, which is
fitted from data and therefore only approximately equivariant.

Walsh basis
-----------
For S a subset of sites, the character is

    chi_S(g) = (-1)^{sum_{i in S} g_i}

Any fitness function on the hypercube expands exactly as

    f(g) = sum_S  beta_S chi_S(g)

Write the state at genotype g as the vector with components s_S = beta_S chi_S(g).
Because chi_S(g XOR d) = chi_S(g) chi_S(d), a mutation acts as

    rho(d) = diag( chi_S(d) )          entries in {+1, -1}

which is diagonal, unitary, abelian and involutive.  Propositions 0, 1, 2 and 4
of the SE(2) case all hold here, and hold *exactly* rather than to float
precision.

The split that matters
----------------------
    order 0-1 coefficients  ->  the additive effect of each mutation.
                                Measured directly from single-mutant assays.
                                Never learned.  This is rho.
    order >= 2 coefficients ->  epistasis.  This is Delta_phi, and it is the
                                only thing the model spends capacity on.

Sparsity is the inductive bias: higher-order interaction in real landscapes is
concentrated in a few hotspots, not spread evenly, so the residual uses the same
hard-concrete gate machinery as the enhancer masks in ``chroma.regulatory``.

Loop closure changes meaning
----------------------------
In the robot, a closed path *must* return to its start -- that is a constraint,
and drift is error.  Here the cycle

    wt -> A -> AB -> B -> wt

deliberately does *not* close, and the gap is the double-mutant cycle that
experimentalists use to measure interaction:

    eps_AB = f(AB) - f(A) - f(B) + f(wt)

So the same code path that validated the robot's geometry becomes the readout
for the quantity of interest.  See :func:`cycle_epistasis`.

One correction the tests forced.  It is tempting to write eps_AB = 4 beta_{AB},
and that is what a first pass at this claimed.  It is false in general.  Every
subset S containing both A and B contributes, because the remaining sites are
held fixed and so contribute chi = +1 across all four corners of the cycle:

    eps_AB = 4 * sum_{S contains A and B, S \ {A,B} matches background} beta_S

The clean identity holds only when the landscape is purely pairwise.  This is
not a defect of the formalism -- it *is* the well-known background dependence
of measured epistasis, and the formalism makes the exact accounting explicit.
See :func:`cycle_epistasis_expected`.
"""

from __future__ import annotations

import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["subsets_upto", "WalshRep", "AdditiveTransport",
           "SparseEpistaticResidual", "FitnessPredictor", "MLPBaseline",
           "cycle_epistasis", "cycle_epistasis_expected", "walsh_transform"]


def subsets_upto(L: int, order: int, min_order: int = 0) -> Tensor:
    """All subsets of [L] with min_order <= |S| <= order, as a (K, L) 0/1 mask."""
    rows = []
    for k in range(min_order, order + 1):
        for S in itertools.combinations(range(L), k):
            m = [0] * L
            for i in S:
                m[i] = 1
            rows.append(m)
    return torch.tensor(rows, dtype=torch.float32)


class WalshRep(nn.Module):
    """Exact unitary representation of (Z/2)^L on the Walsh coefficient space.

    ``masks`` is a (K, L) 0/1 matrix listing the subsets S that index the latent.
    """

    def __init__(self, L: int, masks: Tensor):
        super().__init__()
        self.L = L
        self.register_buffer("masks", masks)
        self.K = masks.shape[0]

    def chi(self, g: Tensor) -> Tensor:
        """Characters chi_S(g) for every S. g: (B, L) 0/1 -> (B, K) in {+1,-1}."""
        parity = g @ self.masks.T                      # (B, K) integer counts
        return 1.0 - 2.0 * torch.remainder(parity, 2.0)

    def act(self, s: Tensor, d: Tensor) -> Tensor:
        """rho(d) s.  d: (B, L) mutation vector."""
        return s * self.chi(d)

    def act_inv(self, s: Tensor, d: Tensor) -> Tensor:
        """rho(d)^{-1} = rho(d): the group is its own inverse (d XOR d = 0)."""
        return self.act(s, d)

    @staticmethod
    def compose(d1: Tensor, d2: Tensor) -> Tensor:
        """Group product is XOR."""
        return torch.remainder(d1 + d2, 2.0)


class AdditiveTransport(nn.Module):
    """The exact, unlearned part: constant plus per-site additive effects.

    In a real deep mutational scan the single-mutant effects are measured, so
    these coefficients are *data*, not parameters.  ``freeze=True`` (the
    default) enforces that, which is what makes the comparison against an
    unconstrained model honest.
    """

    def __init__(self, L: int, beta0: float = 0.0,
                 singles: Tensor | None = None, freeze: bool = True):
        super().__init__()
        self.L = L
        b = torch.zeros(L) if singles is None else singles.clone().float()
        self.beta0 = nn.Parameter(torch.tensor(float(beta0)),
                                  requires_grad=not freeze)
        self.singles = nn.Parameter(b, requires_grad=not freeze)

    @staticmethod
    def fit_from_singles(f_wt: float, f_single: Tensor) -> tuple[float, Tensor]:
        """Convert measured wild-type and single-mutant fitness into Walsh
        order-0/1 coefficients.

        Since chi_S(wt) = +1 for every S, the wild-type value is the *sum* of
        all coefficients, not the constant term:

            f(wt)   = beta0 + sum_i beta_i
            f(e_k)  = f(wt) - 2 beta_k      =>  beta_k = (f(wt) - f(e_k)) / 2
            beta0   = f(wt) - sum_i beta_i
        """
        singles = (f_wt - f_single) / 2.0
        beta0 = f_wt - float(singles.sum())
        return beta0, singles

    def forward(self, g: Tensor) -> Tensor:
        chi1 = 1.0 - 2.0 * g                            # (B, L), chi_{i}(g)
        return self.beta0 + chi1 @ self.singles


class SparseEpistaticResidual(nn.Module):
    """Delta_phi: learned Walsh coefficients of order >= 2, sparsely gated.

    Equivariance is automatic and exact.  A function parameterised by its Walsh
    coefficients transforms under rho(d) by construction -- there is no
    architecture to constrain and no equivariance error to measure.
    """

    GAMMA, ZETA, BETA = -0.1, 1.1, 2.0 / 3.0

    def __init__(self, L: int, order: int = 3, l0_weight: float = 1e-2):
        super().__init__()
        masks = subsets_upto(L, order, min_order=2)
        self.rep = WalshRep(L, masks)
        self.K = masks.shape[0]
        self.coef = nn.Parameter(torch.zeros(self.K))
        self.log_alpha = nn.Parameter(torch.full((self.K,), -1.0))
        self.l0_weight = l0_weight

    def gates(self, hard: bool = False) -> Tensor:
        if hard or not self.training:
            s = torch.sigmoid(self.log_alpha) * (self.ZETA - self.GAMMA) + self.GAMMA
            return s.clamp(0, 1)
        u = torch.rand_like(self.log_alpha).clamp(1e-6, 1 - 1e-6)
        s = torch.sigmoid((u.log() - (1 - u).log() + self.log_alpha) / self.BETA)
        return (s * (self.ZETA - self.GAMMA) + self.GAMMA).clamp(0, 1)

    def l0(self) -> Tensor:
        p = torch.sigmoid(self.log_alpha
                          - self.BETA * math.log(-self.GAMMA / self.ZETA))
        return self.l0_weight * p.sum()

    def coefficients(self) -> Tensor:
        return self.gates(hard=True) * self.coef

    def forward(self, g: Tensor) -> Tensor:
        return (self.rep.chi(g) * (self.gates() * self.coef)).sum(-1)

    @torch.no_grad()
    def hotspots(self, top: int = 10) -> list[tuple[tuple[int, ...], float]]:
        """The interactions the model actually kept, largest first."""
        c = self.coefficients()
        idx = c.abs().argsort(descending=True)[:top]
        out = []
        for i in idx:
            sites = tuple(torch.nonzero(self.rep.masks[i]).flatten().tolist())
            out.append((sites, float(c[i])))
        return out


class FitnessPredictor(nn.Module):
    """f_hat(g) = additive transport (given, debiased) + sparse epistasis (learned).

    The debiasing step is the thing the tests forced, and it matters.

    A single-mutant assay does not measure the Walsh coefficient beta_k.  It
    measures

        (f(wt) - f(e_k)) / 2  =  sum over every S containing k of beta_S

    so the naive additive estimate silently absorbs every interaction involving
    that site.  On a landscape with 64% of its power above order 1 this makes
    the "exact" additive part actively worse than no model at all -- the
    residual variance came out *larger* than the raw variance before this was
    fixed.

    Because the learned coefficients c_S are exactly the contaminating terms,
    the correction is available in closed form and costs no parameters:

        beta_k_eff = measured_k - sum_{S containing k, |S| >= 2} c_S
        beta0_eff  = f(wt) - sum_k beta_k_eff - sum_S c_S

    Substituting the true coefficients for c_S recovers the true beta exactly,
    so the additive part remains given rather than fitted; it is just read in
    the right basis.
    """

    def __init__(self, L: int, order: int = 3, beta0: float = 0.0,
                 singles: Tensor | None = None, freeze_additive: bool = True,
                 l0_weight: float = 1e-2, debias: bool = True):
        super().__init__()
        self.additive = AdditiveTransport(L, beta0, singles, freeze_additive)
        self.residual = SparseEpistaticResidual(L, order, l0_weight)
        self.L, self.debias = L, debias
        # incidence[k, S] = 1 if site k belongs to subset S
        self.register_buffer("incidence", self.residual.rep.masks.T.contiguous())

    def effective_additive(self, c: Tensor) -> tuple[Tensor, Tensor]:
        if not self.debias:
            return self.additive.beta0, self.additive.singles
        beta_k = self.additive.singles - self.incidence @ c
        f_wt = self.additive.beta0 + self.additive.singles.sum()
        beta0 = f_wt - beta_k.sum() - c.sum()
        return beta0, beta_k

    def forward(self, g: Tensor) -> Tensor:
        c = self.residual.gates() * self.residual.coef
        beta0, beta_k = self.effective_additive(c)
        chi1 = 1.0 - 2.0 * g
        add = beta0 + chi1 @ beta_k
        return add + (self.residual.rep.chi(g) * c).sum(-1)

    def additive_only(self, g: Tensor) -> Tensor:
        return self.additive(g)

    def loss(self, g: Tensor, y: Tensor) -> Tensor:
        return F.mse_loss(self(g), y) + self.residual.l0()


class MLPBaseline(nn.Module):
    """Unconstrained control: no group, no additive prior, matched capacity."""

    def __init__(self, L: int, width: int = 128, depth: int = 3):
        super().__init__()
        layers, d = [], L
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.SiLU()]
            d = width
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, g: Tensor) -> Tensor:
        return self.net(1.0 - 2.0 * g).squeeze(-1)

    def loss(self, g: Tensor, y: Tensor) -> Tensor:
        return F.mse_loss(self(g), y)


# --------------------------------------------------------------- readouts

def cycle_epistasis(f: dict | Tensor, i: int, j: int, L: int,
                    background: Tensor | None = None) -> float:
    """Double-mutant cycle interaction between sites i and j.

        eps_ij = f(AB) - f(A) - f(B) + f(wt)

    ``f`` is either a callable-free lookup dict keyed by genotype tuple, or a
    (2**L,) tensor indexed by the integer encoding of the genotype.
    """
    base = torch.zeros(L) if background is None else background.clone().float()

    def get(gv: Tensor) -> float:
        if isinstance(f, dict):
            return float(f[tuple(int(x) for x in gv)])
        idx = int(sum(int(v) << k for k, v in enumerate(gv)))
        return float(f[idx])

    wt = base.clone()
    a = base.clone(); a[i] = 1 - a[i]
    b = base.clone(); b[j] = 1 - b[j]
    ab = a.clone(); ab[j] = 1 - ab[j]
    return get(ab) - get(a) - get(b) + get(wt)


def cycle_epistasis_expected(beta: Tensor, i: int, j: int, L: int,
                             background: Tensor | None = None) -> float:
    """The exact Walsh accounting for a double-mutant cycle.

    Sums 4*beta_S over every subset S that contains both i and j and whose
    remaining sites are all *set* in the background (those contribute chi = +1
    at all four corners; sites not in the background flip sign and cancel).
    With a wild-type background this is every S containing i and j.
    """
    bg = torch.zeros(L) if background is None else background
    total = 0.0
    for idx in range(2 ** L):
        S = [(idx >> k) & 1 for k in range(L)]
        if not (S[i] and S[j]):
            continue
        rest = [k for k in range(L) if S[k] and k not in (i, j)]
        sign = 1.0
        for k in rest:
            sign *= 1.0 if bg[k] == 0 else -1.0
        total += sign * float(beta[idx])
    return 4.0 * total


def walsh_transform(y: Tensor, L: int) -> Tensor:
    """Exact Walsh coefficients of a complete landscape.

    ``y`` is (2**L,) indexed so that bit k of the index is site k.  Returns
    (2**L,) coefficients indexed the same way (bit k set means site k in S).
    """
    a = y.clone().float().reshape([2] * L)
    for k in range(L):
        a = a.movedim(k, 0)
        s, d = a[0] + a[1], a[0] - a[1]
        a = torch.stack([s, d]).movedim(0, k)
    return a.reshape(-1) / (2 ** L)
