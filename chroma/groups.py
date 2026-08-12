"""
Section 1 - Latent geometry and the group action.

Exact unitary representation rho of the discrete special Euclidean group

    G = Z^2  x|  C_n           (semidirect product, n-fold rotation)

acting on the latent space S = R^D.

Layout of a latent vector s in R^D:

    s = [ scalars (n_scalar real dims) | orbit blocks (n_orbits, n_rot) complex ]

    D = n_scalar + 2 * n_orbits * n_rot

Construction
------------
A faithful finite-dimensional unitary representation of the continuous group
SE(2) does not exist (SE(2) is non-compact), so we work with the discrete
subgroup Z^2 x| C_n, for which an exact one does.

Pick base frequencies k_1..k_m in R^2 and close each under the C_n action:

    freq[i, a] = R(2*pi*a/n) k_i,      a = 0..n-1

Translation by v acts diagonally by a phase, rotation by j cyclically permutes
the orbit index (because R_j maps freq[i,a] to freq[i,a+j]):

    (T_v z)_{i,a} = exp( i <freq[i,a], v> ) z_{i,a}
    (P_j z)_{i,a} = z_{i, a-j}

and we define

    rho(v, j) = T_v . P_j                                            (order matters)

Proposition 0 (homomorphism).  rho(g1) rho(g2) = rho(g1 g2) where the group
product is (v1, j1)(v2, j2) = (v1 + R_{j1} v2, j1 + j2).

    Proof.  From the definitions, P_j T_w = T_{R_j w} P_j.  Hence
        rho(v1,j1) rho(v2,j2) = T_{v1} P_{j1} T_{v2} P_{j2}
                              = T_{v1} T_{R_{j1} v2} P_{j1} P_{j2}
                              = T_{v1 + R_{j1} v2} P_{j1 + j2}
                              = rho( (v1,j1)(v2,j2) ).                       []

rho is unitary: T_v is diagonal with unit-modulus entries and P_j is a
permutation.  Unitarity is what Proposition 1 (loop closure) and Proposition 4
(collapse immunity) both rely on, so this file is deliberately kept exact
rather than learned.

Scalars carry the trivial representation and are fixed by every group element.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = ["GroupAction", "SE2Rep", "log_spaced_freqs"]


def log_spaced_freqs(n_orbits: int, k_min: float = 0.35, k_max: float = 2.2) -> Tensor:
    """Base frequencies for the orbit blocks, log-spaced in magnitude.

    Directions are offset by an irrational-ish angle so that no base frequency
    lands on another's C_n orbit (which would make rho non-faithful on that
    block).
    """
    mags = torch.logspace(math.log10(k_min), math.log10(k_max), n_orbits)
    # golden-angle offsets keep orbits distinct
    angs = torch.arange(n_orbits, dtype=torch.float32) * (math.pi * (3.0 - math.sqrt(5.0)))
    return torch.stack([mags * torch.cos(angs), mags * torch.sin(angs)], dim=-1)


@dataclass
class GroupAction:
    """A batch of group elements g = (v, j) in Z^2 x| C_n.

    v: (B, 2) real translation (continuous is fine; the lattice is only needed
       for the memory hash, not for rho).
    j: (B,) integer rotation index in [0, n_rot).
    """

    v: Tensor
    j: Tensor

    def __post_init__(self) -> None:
        if self.j.dtype not in (torch.int32, torch.int64):
            self.j = self.j.long()

    @property
    def batch(self) -> int:
        return self.v.shape[0]

    def to(self, *a, **kw) -> "GroupAction":
        return GroupAction(self.v.to(*a, **kw), self.j.to(*a, **kw))

    @staticmethod
    def identity(batch: int, device=None) -> "GroupAction":
        return GroupAction(
            torch.zeros(batch, 2, device=device),
            torch.zeros(batch, dtype=torch.long, device=device),
        )


def _rot2(theta: Tensor) -> Tensor:
    """(B,) angles -> (B, 2, 2) rotation matrices."""
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.stack(
        [torch.stack([c, -s], -1), torch.stack([s, c], -1)], dim=-2
    )


class SE2Rep(torch.nn.Module):
    """Exact unitary representation of Z^2 x| C_n on R^D.

    Parameters
    ----------
    n_scalar : dimension of the trivial (invariant) block.
    n_orbits : number of frequency orbits.
    n_rot    : order of the rotation group C_n.
    """

    def __init__(self, n_scalar: int = 16, n_orbits: int = 3, n_rot: int = 8):
        super().__init__()
        self.n_scalar = n_scalar
        self.n_orbits = n_orbits
        self.n_rot = n_rot
        self.D = n_scalar + 2 * n_orbits * n_rot

        base = log_spaced_freqs(n_orbits)                     # (n_orbits, 2)
        angs = torch.arange(n_rot, dtype=torch.float32) * (2 * math.pi / n_rot)
        R = _rot2(angs)                                       # (n_rot, 2, 2)
        # freq[i, a] = R_a @ base_i
        freq = torch.einsum("aij,oj->oai", R, base)           # (n_orbits, n_rot, 2)
        self.register_buffer("freq", freq)
        self.register_buffer("rot_mats", R)

    # ---------------------------------------------------------------- layout

    def split(self, s: Tensor) -> tuple[Tensor, Tensor]:
        """(B, D) -> (scalars (B, n_scalar), z (B, n_orbits, n_rot) complex)."""
        ns = self.n_scalar
        scal = s[..., :ns]
        rest = s[..., ns:].reshape(*s.shape[:-1], self.n_orbits, self.n_rot, 2)
        z = torch.view_as_complex(rest.contiguous())
        return scal, z

    def join(self, scal: Tensor, z: Tensor) -> Tensor:
        """Inverse of :meth:`split`."""
        rest = torch.view_as_real(z).reshape(*z.shape[:-2], self.n_orbits * self.n_rot * 2)
        return torch.cat([scal, rest], dim=-1)

    # ------------------------------------------------------------ group ops

    def compose(self, g1: GroupAction, g2: GroupAction) -> GroupAction:
        """(v1, j1) . (v2, j2) = (v1 + R_{j1} v2, j1 + j2)."""
        R = self.rot_mats[g1.j % self.n_rot]                  # (B, 2, 2)
        v = g1.v + torch.einsum("bij,bj->bi", R, g2.v)
        return GroupAction(v, (g1.j + g2.j) % self.n_rot)

    def inverse(self, g: GroupAction) -> GroupAction:
        """(v, j)^{-1} = (-R_{-j} v, -j)."""
        R = self.rot_mats[(-g.j) % self.n_rot]
        return GroupAction(-torch.einsum("bij,bj->bi", R, g.v), (-g.j) % self.n_rot)

    # ---------------------------------------------------------------- rho

    def phases(self, v: Tensor) -> Tensor:
        """exp(i <freq, v>) with shape (B, n_orbits, n_rot)."""
        ang = torch.einsum("oai,bi->boa", self.freq, v)
        return torch.polar(torch.ones_like(ang), ang)

    def act_z(self, z: Tensor, g: GroupAction) -> Tensor:
        """Apply rho(v, j) = T_v . P_j to the complex blocks only."""
        # P_j : shift index a -> a - j, i.e. gather from (a - j) mod n
        idx = (torch.arange(self.n_rot, device=z.device)[None, :] - g.j[:, None]) % self.n_rot
        idx = idx[:, None, :].expand(-1, self.n_orbits, -1)   # (B, n_orbits, n_rot)
        z = torch.gather(z, 2, idx)
        return z * self.phases(g.v)

    def act(self, s: Tensor, g: GroupAction) -> Tensor:
        """Apply rho(g) to a full latent vector.  Scalars are invariant."""
        scal, z = self.split(s)
        return self.join(scal, self.act_z(z, g))

    def act_inv(self, s: Tensor, g: GroupAction) -> Tensor:
        """Apply rho(g)^{-1} = rho(g^{-1})."""
        return self.act(s, self.inverse(g))

    # ----------------------------------------------------------- invariants

    def invariants(self, s: Tensor) -> Tensor:
        """Translation-invariant, rotation-equivariant power spectrum |z|^2.

        Shape (B, n_orbits, n_rot).  Under rho(v, j) this is unchanged by v and
        cyclically permuted by j -- which is exactly what the gate network in
        :mod:`chroma.predictor` needs.
        """
        _, z = self.split(s)
        return (z.real ** 2 + z.imag ** 2)
