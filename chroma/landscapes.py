"""
Synthetic combinatorially complete landscapes with known ground truth.

Real combinatorially complete datasets exist and are the eventual target --
four-site saturation landscapes for binding (GB1) and for enzyme catalysis
(TrpB, 160k variants), plus combinatorially complete antibody landscapes.  But
those are measured, so the true epistatic coefficients are only known up to
assay noise.  For validating the architecture you want ground truth you
constructed yourself.

Two generators:

    NKLandscape       classic tunable ruggedness: each site interacts with K
                      others, so epistasis is dense up to order K+1
    HotspotLandscape  sparse interaction: additive backbone plus a handful of
                      designated high-order hotspots

The second is the honest test.  Real landscapes appear to concentrate
interaction in a few sites rather than spreading it evenly, and the sparse
Walsh residual in ``chroma.hypercube`` is built on exactly that assumption -- so
a model that only wins on HotspotLandscape and loses on NKLandscape has told
you something true about when to use it.
"""

from __future__ import annotations

import itertools

import torch
from torch import Tensor

from .hypercube import subsets_upto, walsh_transform

__all__ = ["NKLandscape", "HotspotLandscape", "all_genotypes", "split"]


def all_genotypes(L: int) -> Tensor:
    """(2**L, L) with bit k of the row index at column k."""
    idx = torch.arange(2 ** L)
    bits = [(idx >> k) & 1 for k in range(L)]
    return torch.stack(bits, dim=1).float()


class _Landscape:
    L: int
    y: Tensor

    def genotypes(self) -> Tensor:
        return all_genotypes(self.L)

    def true_walsh(self) -> Tensor:
        return walsh_transform(self.y, self.L)

    def singles(self) -> tuple[float, Tensor]:
        """Wild-type and single-mutant values, as an assay would report them."""
        f_wt = float(self.y[0])
        f_single = torch.tensor([float(self.y[1 << k]) for k in range(self.L)])
        return f_wt, f_single

    def true_epistasis_mass(self, min_order: int = 2) -> float:
        """Fraction of total Walsh power sitting at order >= min_order."""
        w = self.true_walsh()
        idx = torch.arange(2 ** self.L)
        order = torch.stack([(idx >> k) & 1 for k in range(self.L)]).sum(0)
        hi = (w[order >= min_order] ** 2).sum()
        lo = (w[(order >= 1) & (order < min_order)] ** 2).sum()
        return float(hi / (hi + lo + 1e-12))


class NKLandscape(_Landscape):
    """Kauffman NK: each site's contribution depends on itself and K neighbours."""

    def __init__(self, L: int = 10, K: int = 2, seed: int = 0,
                 noise: float = 0.0):
        g = torch.Generator().manual_seed(seed)
        self.L, self.K = L, K
        neigh = [torch.randperm(L, generator=g)[:K + 1] for _ in range(L)]
        neigh = [n if i in n.tolist() else torch.cat([torch.tensor([i]), n[:K]])
                 for i, n in enumerate(neigh)]
        tables = [torch.rand(2 ** (K + 1), generator=g) for _ in range(L)]
        G = all_genotypes(L)
        y = torch.zeros(2 ** L)
        for i in range(L):
            sub = G[:, neigh[i]]
            key = (sub * (2 ** torch.arange(K + 1).float())).sum(-1).long()
            y += tables[i][key]
        y /= L
        if noise:
            y += noise * torch.randn(y.shape, generator=g)
        self.y = y
        self.neighbourhoods = neigh


class HotspotLandscape(_Landscape):
    """Additive backbone plus a few designated high-order interaction hotspots."""

    def __init__(self, L: int = 10, n_hotspots: int = 4, order: int = 3,
                 hotspot_scale: float = 1.0, seed: int = 0,
                 noise: float = 0.0):
        g = torch.Generator().manual_seed(seed)
        self.L = L
        singles = torch.randn(L, generator=g) * 0.5
        pool = subsets_upto(L, order, min_order=2)
        pick = torch.randperm(pool.shape[0], generator=g)[:n_hotspots]
        self.hotspot_masks = pool[pick]
        self.hotspot_coefs = (torch.randn(n_hotspots, generator=g)
                              * hotspot_scale)
        G = all_genotypes(L)
        chi1 = 1.0 - 2.0 * G
        y = chi1 @ singles
        parity = G @ self.hotspot_masks.T
        chiS = 1.0 - 2.0 * torch.remainder(parity, 2.0)
        y = y + chiS @ self.hotspot_coefs
        if noise:
            y = y + noise * torch.randn(y.shape, generator=g)
        self.y = y
        self.true_singles = singles

    def true_hotspots(self) -> list[tuple[tuple[int, ...], float]]:
        out = []
        for m, c in zip(self.hotspot_masks, self.hotspot_coefs):
            sites = tuple(torch.nonzero(m).flatten().tolist())
            out.append((sites, float(c)))
        return sorted(out, key=lambda t: -abs(t[1]))


def split(land: _Landscape, n_train: int, seed: int = 0,
          include_singles: bool = True) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Train/test split.

    ``include_singles`` forces the wild type and all L single mutants into the
    training set, because a real programme always measures those first.  Both
    the constrained and the unconstrained model see them, so the comparison
    stays fair.
    """
    g = torch.Generator().manual_seed(seed)
    G, y, L = land.genotypes(), land.y, land.L
    N = G.shape[0]
    forced = ([0] + [1 << k for k in range(L)]) if include_singles else []
    rest = [i for i in range(N) if i not in set(forced)]
    perm = torch.randperm(len(rest), generator=g).tolist()
    need = max(0, n_train - len(forced))
    tr = forced + [rest[i] for i in perm[:need]]
    te = [rest[i] for i in perm[need:]]
    tr_i = torch.tensor(tr)
    te_i = torch.tensor(te)
    return G[tr_i], y[tr_i], G[te_i], y[te_i]
