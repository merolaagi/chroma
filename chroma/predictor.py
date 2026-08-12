"""
Section 2 - The predictor: geometry imposed, content learned.

    s_hat_{t+1} = rho(d_t) [ s_t + Delta_phi(s_t, z_t ; g) ]

rho(d_t) is exact geometric transport with no parameters.  Delta_phi is the
learned residual carrying everything non-rigid.  Delta_phi must itself be
G-equivariant, which pins down its form almost completely:

  * A *linear* map commuting with T_v for every v must be diagonal in the
    frequency basis (T_v has distinct eigenvalues per frequency).  Requiring it
    to also commute with P_j forces the diagonal to be constant on orbits.
    Linear equivariant maps are therefore nearly trivial.

  * The expressive route is *gating*.  Multiply component (i, a) by a complex
    gate (gamma + i delta)_{i,a} that is itself built from invariants.  The
    power spectrum |z_{i,a}|^2 is translation-invariant and rotation-permuted,
    so any C_n-equivariant network on the orbit index a (i.e. a circular
    convolution) produces a legal gate.  Cross-orbit mixing is unrestricted
    because gates are scalars.

Gene programs (Section 4.2) modulate only the weights of the gate network.
Nothing geometric is ever modulated.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .groups import GroupAction, SE2Rep

__all__ = ["ProgramLinear", "CyclicProgramConv", "EquivariantResidual", "Predictor"]


class ProgramLinear(nn.Module):
    """Linear layer whose weight is a gene-program mixture.

        W(e) = W0 + sum_k e_k U_k V_k^T

    ``U``/``V`` are the globally shared programs; only ``e`` is module specific.
    This is the parameter-sharing claim of Section 4.2 made concrete: M modules
    cost M * K_prog numbers of specificity, not M full weight matrices.
    """

    def __init__(self, d_in: int, d_out: int, n_prog: int, rank: int = 4):
        super().__init__()
        self.d_in, self.d_out, self.n_prog, self.rank = d_in, d_out, n_prog, rank
        self.W0 = nn.Parameter(torch.empty(d_out, d_in))
        nn.init.kaiming_uniform_(self.W0, a=5 ** 0.5)
        self.b = nn.Parameter(torch.zeros(d_out))
        self.U = nn.Parameter(torch.randn(n_prog, d_out, rank) * (1.0 / d_out) ** 0.5)
        self.V = nn.Parameter(torch.zeros(n_prog, d_in, rank))
        nn.init.normal_(self.V, std=(1.0 / d_in) ** 0.5)

    def forward(self, x: Tensor, e: Tensor | None) -> Tensor:
        """x: (B, d_in); e: (B, n_prog) or None (pluripotent phase 0)."""
        y = F.linear(x, self.W0, self.b)
        if e is None:
            return y
        xv = torch.einsum("bi,kir->bkr", x, self.V)
        xv = xv * e.unsqueeze(-1)
        return y + torch.einsum("bkr,kor->bo", xv, self.U)


class CyclicProgramConv(nn.Module):
    """Circular convolution over the C_n orbit index, gene-modulated.

    Equivariant to the cyclic shift P_j by construction (circular padding), and
    trivially commutes with T_v because it only ever sees invariants.
    """

    def __init__(self, c_in: int, c_out: int, n_rot: int, n_prog: int,
                 rank: int = 4, kernel: int = 3):
        super().__init__()
        self.c_in, self.c_out, self.n_rot, self.k = c_in, c_out, n_rot, kernel
        self.lin = ProgramLinear(c_in * kernel, c_out, n_prog, rank)

    def forward(self, x: Tensor, e: Tensor | None) -> Tensor:
        """x: (B, c_in, n_rot); e: (B, n_prog) -> (B, c_out, n_rot)."""
        B, _, A = x.shape
        pad = self.k // 2
        xp = F.pad(x, (pad, pad), mode="circular")
        xu = xp.unfold(-1, self.k, 1)                       # (B, c_in, A, k)
        xu = xu.permute(0, 2, 1, 3).reshape(B * A, self.c_in * self.k)
        ee = None if e is None else e.repeat_interleave(A, dim=0)
        y = self.lin(xu, ee)                                # (B*A, c_out)
        return y.reshape(B, A, self.c_out).permute(0, 2, 1).contiguous()


class EquivariantResidual(nn.Module):
    """Delta_phi.  Exactly equivariant; norm-bounded by ``delta_max``.

    The bound is what buys Proposition 4: since rho is unitary and the residual
    can change the norm by at most a factor delta_max per step, representational
    norm collapse is unreachable on a bounded horizon and the *variance* term of
    VICReg can be dropped.
    """

    def __init__(self, rep: SE2Rep, n_z: int = 8, width: int = 48,
                 n_prog: int = 12, rank: int = 4, depth: int = 3,
                 delta_max: float = 0.30):
        super().__init__()
        self.rep, self.n_z, self.delta_max = rep, n_z, delta_max
        c_in = rep.n_orbits + rep.n_scalar + n_z
        chans = [c_in] + [width] * (depth - 1)
        self.convs = nn.ModuleList(
            CyclicProgramConv(chans[i], chans[i + 1], rep.n_rot, n_prog, rank)
            for i in range(depth - 1)
        )
        # gate head: 2 channels (gamma, delta) per orbit
        self.gate = CyclicProgramConv(width, 2 * rep.n_orbits, rep.n_rot, n_prog, rank)
        # scalar head: reads the orbit-pooled (hence invariant) feature
        self.scal = ProgramLinear(width, rep.n_scalar, n_prog, rank)
        nn.init.zeros_(self.gate.lin.W0)
        nn.init.zeros_(self.scal.W0)

    def forward(self, s: Tensor, z: Tensor, e: Tensor | None) -> Tensor:
        rep = self.rep
        B = s.shape[0]
        scal, zc = rep.split(s)
        inv = rep.invariants(s)                              # (B, O, A) equivariant in a
        feats = [inv,
                 scal.unsqueeze(-1).expand(-1, -1, rep.n_rot),
                 z.unsqueeze(-1).expand(-1, -1, rep.n_rot)]
        x = torch.cat(feats, dim=1)
        for c in self.convs:
            x = F.silu(c(x, e))
        g = self.gate(x, e).reshape(B, 2, rep.n_orbits, rep.n_rot)
        gam, dlt = g[:, 0], g[:, 1]
        # hard norm projection -> |gate| <= delta_max  (Proposition 4)
        mag = torch.sqrt(gam ** 2 + dlt ** 2 + 1e-12)
        scale = (self.delta_max / mag).clamp(max=1.0)
        gate = torch.complex(gam * scale, dlt * scale)
        dz = gate * zc
        dscal = self.scal(x.mean(-1), e)                     # orbit-pool = invariant
        dscal = dscal.clamp(-self.delta_max, self.delta_max) * scal.abs().clamp(min=1e-3)
        return rep.join(dscal, dz)


class Predictor(nn.Module):
    """One-step action-conditioned prediction in latent space."""

    def __init__(self, rep: SE2Rep, n_z: int = 8, **kw):
        super().__init__()
        self.rep = rep
        self.delta = EquivariantResidual(rep, n_z=n_z, **kw)
        self.n_z = n_z

    def forward(self, s: Tensor, d: GroupAction, z: Tensor,
                e: Tensor | None = None) -> Tensor:
        return self.rep.act(s + self.delta(s, z, e), d)

    def rollout(self, s: Tensor, ds: list[GroupAction], zs: list[Tensor] | None,
                e: Tensor | None = None) -> Tensor:
        """Compose a path.  With Delta_phi == 0 and a closed path this returns
        exactly ``s`` (Proposition 1)."""
        for t, d in enumerate(ds):
            z = torch.zeros(s.shape[0], self.n_z, device=s.device) if zs is None else zs[t]
            s = self.forward(s, d, z, e)
        return s
