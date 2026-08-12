"""Shared sensor encoder E_theta and its EMA target E_ema.

The encoder is shared across all modules (theta^0 of Section 4.2); modules
differ only through the gene-expression modulation of the predictor.  That is
the parameter-sharing claim, and keeping the encoder shared is what makes it
testable.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["PatchEncoder", "EMATarget"]


class PatchEncoder(nn.Module):
    def __init__(self, patch: int = 3, d_out: int = 64, width: int = 128,
                 n_z: int = 8):
        super().__init__()
        d_in = patch * patch
        self.net = nn.Sequential(
            nn.Linear(d_in, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, d_out),
        )
        self.z_head = nn.Linear(width, n_z)
        self.d_out, self.n_z = d_out, n_z

    def forward(self, x: Tensor, want_z: bool = False):
        h = x
        for layer in self.net[:-1]:
            h = layer(h)
        s = self.net[-1](h)
        s = s / s.norm(dim=-1, keepdim=True).clamp(min=1e-6) * (self.d_out ** 0.5)
        if want_z:
            return s, self.z_head(h)
        return s


class EMATarget(nn.Module):
    """Frozen exponential-moving-average copy providing the prediction target."""

    def __init__(self, module: nn.Module, decay: float = 0.996):
        super().__init__()
        self.ema = copy.deepcopy(module)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, module: nn.Module) -> None:
        for pe, pm in zip(self.ema.parameters(), module.parameters()):
            pe.mul_(self.decay).add_((1 - self.decay) * pm.detach())
        for be, bm in zip(self.ema.buffers(), module.buffers()):
            be.copy_(bm)

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        return self.ema(x)
