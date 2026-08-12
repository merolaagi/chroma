"""
Section 5 - Losses, and Section 8 - collapse instrumentation.

    L = L_pred + alpha L_loop + beta L_vote + gamma L_reg

L_loop is the term the equivariant construction buys for free.  Closed paths are
trivially generatable by an embodied agent, the target is the agent's own
current state, and because rho is unitary there is no shrink-to-zero shortcut --
so it needs no target encoder and carries no collapse risk.

Note the deliberate omission: the *variance* term of VICReg is dropped.
Proposition 4 closes off norm collapse structurally (the residual is norm-bounded
and rho is unitary), so only the covariance term is needed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .groups import GroupAction, SE2Rep

__all__ = [
    "covariance_loss", "prediction_loss", "loop_loss", "vote_distillation_loss",
    "effective_rank", "CollapseMonitor",
]


def covariance_loss(s: Tensor) -> Tensor:
    """Off-diagonal covariance penalty (VICReg covariance term only)."""
    B, D = s.shape
    if B < 2:
        return torch.zeros((), device=s.device)
    z = s - s.mean(0, keepdim=True)
    C = (z.T @ z) / (B - 1)
    off = C - torch.diag(torch.diag(C))
    return (off ** 2).sum() / D


def prediction_loss(s_hat: Tensor, s_target: Tensor, z: Tensor,
                    lam_cov: float = 0.04, lam_z: float = 1e-3
                    ) -> tuple[Tensor, Tensor]:
    """Returns (total, per-sample energy).  ``s_target`` must already be
    stop-gradiented from the EMA encoder."""
    per = ((s_hat - s_target) ** 2).mean(-1)
    total = per.mean() + lam_cov * covariance_loss(s_hat) + lam_z * (z ** 2).mean()
    return total, per.detach()


def loop_loss(predictor, rep: SE2Rep, s: Tensor, path: list[GroupAction],
              e: Tensor | None = None) -> Tensor:
    """Closed-path consistency (Proposition 1).

    With Delta_phi == 0 this is identically zero for any closed path, so any
    non-zero value is a pure, unbiased measurement of residual error.  Drift is
    flat in path length, whereas an unconstrained learned transition drifts as
    O(sqrt(n)) -- that gap is experiment E4.
    """
    s_end = predictor.rollout(s, path, None, e)
    return ((s_end - s.detach()) ** 2).mean()


def vote_distillation_loss(L_pre: Tensor, L_post: Tensor, conf: Tensor) -> Tensor:
    """Pull each module toward the (stop-gradiented) consensus.

    ``L_post`` is detached here explicitly -- see the groupthink warning in
    :mod:`chroma.voting`.
    """
    M = L_pre.shape[0]
    tgt = torch.softmax(L_post.detach().reshape(M, -1), -1)
    pred = torch.log_softmax(L_pre.reshape(M, -1), -1)
    kl = F.kl_div(pred, tgt, reduction="none").sum(-1)
    w = conf / conf.sum().clamp(min=1e-8)
    return (w * kl).sum()


# ---------------------------------------------------------------- monitoring


def effective_rank(s: Tensor) -> Tensor:
    """exp(entropy of the normalised singular-value spectrum).

    Representational collapse alarm: below 0.3 * D.
    """
    if s.shape[0] < 2:
        return torch.zeros(())
    z = s - s.mean(0, keepdim=True)
    sv = torch.linalg.svdvals(z.float())
    p = sv / sv.sum().clamp(min=1e-12)
    return torch.exp(-(p * (p + 1e-12).log()).sum())


class CollapseMonitor:
    """The three-mode alarm table of Section 8.

    Only one of these three shows up in task loss.  Watching task loss alone
    ships a model that looks fine and has silently degenerated.
    """

    def __init__(self, D: int, rank_frac: float = 0.30,
                 min_basin_entropy: float = 0.6931, mi_window: int = 5000):
        self.D, self.rank_frac = D, rank_frac
        self.min_H, self.mi_window = min_basin_entropy, mi_window
        self.mi_hist: list[float] = []

    def check(self, s: Tensor, basin_entropy: float | None,
              vote_mi: float | None) -> dict:
        out = {}
        er = float(effective_rank(s))
        out["effective_rank"] = er
        out["alarm_representational"] = er < self.rank_frac * self.D
        if basin_entropy is not None:
            out["basin_entropy"] = basin_entropy
            out["alarm_attractor"] = basin_entropy < self.min_H
        if vote_mi is not None:
            self.mi_hist.append(vote_mi)
            self.mi_hist = self.mi_hist[-self.mi_window:]
            rising = (len(self.mi_hist) == self.mi_window
                      and self.mi_hist[-1] > self.mi_hist[0]
                      and sum(b > a for a, b in zip(self.mi_hist, self.mi_hist[1:]))
                      > 0.7 * self.mi_window)
            out["vote_mi"] = vote_mi
            out["alarm_groupthink"] = bool(rising)
        return out
