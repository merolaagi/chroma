"""
Section 5 - Losses, and Section 8 - collapse instrumentation.

    L = L_pred + alpha L_loop + beta L_vote + gamma L_reg

L_loop is the term the equivariant construction buys for free.  Closed paths are
trivially generatable by an embodied agent, the target is the agent's own
current state, and because rho is unitary there is no shrink-to-zero shortcut --
so it needs no target encoder and carries no collapse risk.

RETRACTED, 2026-08-13.  An earlier version of this file dropped the *variance*
term of VICReg, arguing that Proposition 4 closes off collapse structurally.
That argument is wrong, and it cost a 12.5 hour sweep.

Proposition 4 bounds ||s_hat|| relative to ||s||, i.e. it prevents the *norm*
going to zero.  But PatchEncoder already renormalises s to fixed norm sqrt(D),
so norm collapse was never reachable in the first place.  What remains reachable
is *dimensional* collapse: every sample mapping to nearly the same point on the
sphere.  Norm is preserved, per-dimension variance vanishes, and Proposition 4
says nothing about it.

Measured after 500 steps with the variance term absent: 62 of 64 latent
dimensions had standard deviation below 0.05.  Raising the covariance weight
from 0.04 to 1.0 changed nothing, because the covariance term penalises
off-diagonal structure, not per-dimension variance -- only the variance term
does that.  Prediction error then falls to ~1e-5 because there is nothing left
to distinguish, vote disagreement follows it to zero, the regulatory input u
goes to zero, and all three E1 arms tie because none of them is being driven by
anything.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .groups import GroupAction, SE2Rep

__all__ = [
    "covariance_loss", "variance_loss", "prediction_loss", "loop_loss", "vote_distillation_loss",
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


def variance_loss(s: Tensor, gamma: float = 1.0, eps: float = 1e-4) -> Tensor:
    """VICReg hinge on per-dimension standard deviation.

    The encoder normalises to ||s|| = sqrt(D), so a zero-mean, isotropic
    representation has per-dimension variance exactly 1.  gamma = 1 is therefore
    the natural target rather than a tuned constant.
    """
    if s.shape[0] < 2:
        return torch.zeros((), device=s.device)
    std = torch.sqrt(s.var(0) + eps)
    return F.relu(gamma - std).mean()


def prediction_loss(s_hat: Tensor, s_target: Tensor, z: Tensor,
                    lam_cov: float = 0.04, lam_z: float = 1e-3,
                    lam_var: float = 1.0
                    ) -> tuple[Tensor, Tensor]:
    """Returns (total, per-sample energy).  ``s_target`` must already be
    stop-gradiented from the EMA encoder."""
    per = ((s_hat - s_target) ** 2).mean(-1)
    total = (per.mean()
             + lam_var * variance_loss(s_hat)
             + lam_cov * covariance_loss(s_hat)
             + lam_z * (z ** 2).mean())
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

    RETRACTION 2: the representational alarm was miscalibrated as
    ``effective_rank < 0.3 * D``.  That threshold is unreachable in principle.
    A 3x3 tactile patch is 9-dimensional with measured effective rank ~4.6, so
    no encoder can produce more than 9 independent latent directions no matter
    how healthy it is.  0.3 * 64 = 19.2 could never be satisfied and the alarm
    fired on every step of every run.

    The meaningful question is not how much of D the latent fills, but whether
    the latent preserves the *input* manifold.  The alarm now compares against
    a running estimate of the input's own effective rank.  Measured healthy
    ratio: latent 5.16 against input 4.64, i.e. 1.11 -- the representation is
    keeping everything the input had.
    """

    def __init__(self, D: int, rank_frac: float = 0.30,
                 min_basin_entropy: float = 0.6931, mi_window: int = 5000,
                 input_rank_frac: float = 0.80):
        self.D, self.rank_frac = D, rank_frac
        self.input_rank_frac = input_rank_frac
        self.input_rank: float | None = None
        self.min_H, self.mi_window = min_basin_entropy, mi_window
        self.mi_hist: list[float] = []

    def observe_input(self, x: Tensor) -> None:
        """Track the input manifold's own effective rank as the reference."""
        r = float(effective_rank(x))
        self.input_rank = r if self.input_rank is None else (
            0.98 * self.input_rank + 0.02 * r)

    def check(self, s: Tensor, basin_entropy: float | None,
              vote_mi: float | None) -> dict:
        out = {}
        er = float(effective_rank(s))
        out["effective_rank"] = er
        if self.input_rank is not None:
            out["input_rank"] = round(self.input_rank, 2)
            out["rank_ratio"] = round(er / max(self.input_rank, 1e-6), 2)
            out["alarm_representational"] = er < self.input_rank_frac * self.input_rank
        else:
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
