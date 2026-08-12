"""
Section 4 - The regulatory layer.

Three mechanisms, each imported from gene regulation because it supplies a
property that softmax mixture-of-experts routing does not have:

  4.1  Multistable dynamics      -> hysteresis (regimes must be *escaped*)
  4.2  Sparse cis-regulation     -> combinatorial control, shared programs
  4.3  Error-gated differentiation -> a horizon-independent forgetting bound

Everything here is falsifiable by experiment E1: ablate the attractor dynamics
for a matched-parameter MLP and measure the regime switching rate.  If the
switching rate does not blow up, this file is decoration and should be deleted.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["RegulatoryState", "MLPRegulator", "EnhancerReadout", "Differentiation"]


class RegulatoryState(nn.Module):
    """Dense-associative-memory regulatory dynamics (option (b) of Section 4.1).

        E(g) = -(1/beta) lse(beta Xi^T g) + 1/2 ||g||^2 - u^T B^T g
        gdot = -grad E = Xi softmax(beta Xi^T g) - g + B^T u

    Xi holds K learned regime prototypes.  Capacity is exponential in n
    (Krotov-Hopfield), attractors are well separated, and -- unlike a random
    sparse W -- K is a design parameter you can inspect, which is what makes
    experiment E2 (attractor count tracks regime count) well posed.

    The landscape of a real gene-regulatory network is sculpted by selection,
    not sampled i.i.d.; this is the closer analogy as well as the better
    engineering choice.
    """

    def __init__(self, n: int = 16, n_proto: int = 6, n_input: int = 3,
                 beta: float = 4.0, tau_g: int = 200, dt: float = 0.25,
                 inner_steps: int = 8, noise: float = 0.01):
        super().__init__()
        self.n, self.K, self.beta = n, n_proto, beta
        self.tau_g, self.dt, self.inner_steps, self.noise = tau_g, dt, inner_steps, noise
        self.Xi = nn.Parameter(F.normalize(torch.randn(n, n_proto), dim=0) * math.sqrt(n))
        self.B = nn.Parameter(torch.randn(n, n_input) * 0.5)
        self.register_buffer("g", torch.zeros(n))
        self.register_buffer("_tick", torch.zeros((), dtype=torch.long))

    # ------------------------------------------------------------------ core

    def energy(self, g: Tensor, u: Tensor | None = None) -> Tensor:
        e = -(1.0 / self.beta) * torch.logsumexp(self.beta * (g @ self.Xi), dim=-1)
        e = e + 0.5 * (g ** 2).sum(-1)
        if u is not None:
            e = e - (g * (u @ self.B.T)).sum(-1)
        return e

    def drift(self, g: Tensor, u: Tensor | None = None) -> Tensor:
        p = torch.softmax(self.beta * (g @ self.Xi), dim=-1)     # (..., K)
        out = p @ self.Xi.T - g
        if u is not None:
            out = out + u @ self.B.T
        return out

    def relax(self, g: Tensor, u: Tensor | None = None, steps: int | None = None,
              noise: float = 0.0) -> Tensor:
        steps = self.inner_steps if steps is None else steps
        for _ in range(steps):
            g = g + self.dt * self.drift(g, u)
            if noise:
                g = g + noise * math.sqrt(self.dt) * torch.randn_like(g)
        return g

    def step(self, u: Tensor) -> Tensor:
        """Advance the slow state by one regulatory tick.  ``u`` is
        ``[mean prediction error, vote disagreement, reward]``."""
        g = self.relax(self.g.unsqueeze(0), u.unsqueeze(0), noise=self.noise).squeeze(0)
        self.g.copy_(g.detach())
        self._tick += 1
        return g

    def maybe_step(self, u: Tensor, t: int) -> Tensor:
        """Only fire every ``tau_g`` environment steps -- the timescale
        separation is load bearing, not cosmetic."""
        if t % self.tau_g == 0:
            return self.step(u)
        return self.g

    # ---------------------------------------------------- E2 instrumentation

    @torch.no_grad()
    def enumerate_attractors(self, n_init: int = 4096, steps: int = 400,
                             tol: float = 0.25) -> tuple[Tensor, Tensor]:
        """Relax from random initial conditions and cluster the fixed points.

        Returns ``(centres (P, n), basin_counts (P,))``.  Experiment E2 checks
        that ``P`` converges to the number of hidden regimes in the task stream.
        """
        g = torch.randn(n_init, self.n, device=self.Xi.device) * 2.0
        g = self.relax(g, None, steps=steps)
        centres, counts = [], []
        for i in range(g.shape[0]):
            gi = g[i]
            placed = False
            for j, c in enumerate(centres):
                if torch.norm(gi - c) < tol:
                    counts[j] += 1
                    placed = True
                    break
            if not placed:
                centres.append(gi.clone())
                counts.append(1)
            if len(centres) > 64:            # runaway -> landscape is rough
                break
        return torch.stack(centres), torch.tensor(counts, dtype=torch.float32)

    @torch.no_grad()
    def basin_entropy(self, **kw) -> Tensor:
        """Collapse metric: entropy of basin occupancy.  Alarm below log 2."""
        _, counts = self.enumerate_attractors(**kw)
        p = counts / counts.sum()
        return -(p * (p + 1e-12).log()).sum()


class FlatRegulator(RegulatoryState):
    """Third E1 arm: recurrent dynamics *with* the multistability removed.

    ``ablate_grn`` (MLPRegulator) removes recurrence and multistability at once,
    so a positive result there cannot say which one did the work.  Sending
    beta -> 0 flattens the Krotov-Hopfield landscape to a single quadratic basin
    while leaving the relaxation dynamics, the timescale tau_g and the input
    coupling B completely intact.

    Any advantage that survives this ablation is attributable to slow
    integration alone.  Any advantage that *disappears* is attributable to
    hysteresis, which is the specific claim of Section 4.1.
    """

    def __init__(self, *a, **kw):
        kw["beta"] = 1e-3
        super().__init__(*a, **kw)


class MLPRegulator(nn.Module):
    """Ablation control for experiment E1: same parameter budget, no dynamics.

    g = f(u).  Memoryless, therefore no hysteresis.  The prediction is that
    this flip-flops between regimes at 5-20x the rate of RegulatoryState under
    a noisy task stream.
    """

    def __init__(self, n: int = 16, n_proto: int = 6, n_input: int = 3, **_):
        super().__init__()
        hidden = n * n_proto // 2 + n
        self.f = nn.Sequential(nn.Linear(n_input, hidden), nn.SiLU(),
                               nn.Linear(hidden, hidden), nn.SiLU(),
                               nn.Linear(hidden, n))
        self.n, self.tau_g = n, 1
        self.register_buffer("g", torch.zeros(n))

    def step(self, u: Tensor) -> Tensor:
        g = self.f(u)
        self.g.copy_(g.detach())
        return g

    def maybe_step(self, u: Tensor, t: int) -> Tensor:
        return self.step(u)

    @torch.no_grad()
    def enumerate_attractors(self, **_):
        return self.g.unsqueeze(0), torch.ones(1)


class EnhancerReadout(nn.Module):
    """Sparse combinatorial cis-regulation (Section 4.2).

        e_m = softplus( A_m (M_m . g) - b_m )

    ``M_m`` is a binary enhancer mask learned by L0 relaxation with hard-concrete
    gates, targeting ~``target_occupancy`` open sites out of ``n`` -- matching the
    handful of transcription factors that actually occupy a real enhancer.
    """

    GAMMA, ZETA, BETA = -0.1, 1.1, 2.0 / 3.0

    def __init__(self, n_modules: int, n: int = 16, n_prog: int = 12,
                 target_occupancy: int = 4):
        super().__init__()
        self.M, self.n, self.K = n_modules, n, n_prog
        self.target = target_occupancy
        self.log_alpha = nn.Parameter(torch.randn(n_modules, n) * 0.5 - 1.0)
        self.A = nn.Parameter(torch.randn(n_modules, n_prog, n) * (1.0 / n) ** 0.5)
        self.b = nn.Parameter(torch.zeros(n_modules, n_prog))

    def mask(self, hard: bool = False) -> Tensor:
        if hard or not self.training:
            s = torch.sigmoid(self.log_alpha) * (self.ZETA - self.GAMMA) + self.GAMMA
            return s.clamp(0, 1)
        u = torch.rand_like(self.log_alpha).clamp(1e-6, 1 - 1e-6)
        s = torch.sigmoid((u.log() - (1 - u).log() + self.log_alpha) / self.BETA)
        return (s * (self.ZETA - self.GAMMA) + self.GAMMA).clamp(0, 1)

    def l0(self) -> Tensor:
        """Expected number of open sites, penalised toward ``target``."""
        p_open = torch.sigmoid(self.log_alpha - self.BETA * math.log(-self.GAMMA / self.ZETA))
        occ = p_open.sum(-1)
        return ((occ - self.target) ** 2).mean()

    def forward(self, g: Tensor, hard: bool = False) -> Tensor:
        """g: (n,) -> expression e: (M, n_prog), non-negative.

        ``hard=True`` uses the deterministic mask.  Any *measurement* of module
        identity must pass hard=True: the training-time mask is a stochastic
        hard-concrete sample, and reading argmax through it counts sampling
        noise as regime switching.  That bug inflated the E1 switch rate by
        roughly two orders of magnitude on first run.
        """
        gm = self.mask(hard=hard) * g.unsqueeze(0)            # (M, n)
        return F.softplus(torch.einsum("mkn,mn->mk", self.A, gm) - self.b)

    @torch.no_grad()
    def lineage_features(self, g: Tensor) -> Tensor:
        """Expression profile used by experiment E3 (lineage clustering)."""
        return self.forward(g, hard=True).detach()


class Differentiation(nn.Module):
    """Error-gated plasticity (Section 4.3).

        Ddot_m = kappa (1-D_m) [1 - ebar_m/eps*]_+  -  lambda D_m [ebar_m/eps* - 1]_+
        eta_m  = eta_0 (1 - D_m)^p

    Proposition 3.  If ebar_m(t) <= eps*(1-delta) for all t > t0 then
        1 - D_m(t) <= (1 - D_m(t0)) exp(-kappa delta (t - t0))
    and total parameter drift is bounded by
        eta_0 G (1-D_m(t0))^p / (p kappa delta)
    *independent of the horizon*.  Unlike EWC the anchor is dynamic and the gate
    reopens the moment competence is lost.
    """

    def __init__(self, n_modules: int, kappa: float = 1e-3, lam: float = 1e-2,
                 p: float = 2.0, ema: float = 0.99, eps_pct: float = 0.60,
                 write_thresh: float = 0.90):
        super().__init__()
        self.kappa, self.lam, self.p, self.ema = kappa, lam, p, ema
        self.eps_pct, self.write_thresh = eps_pct, write_thresh
        self.register_buffer("D", torch.zeros(n_modules))
        self.register_buffer("ebar", torch.zeros(n_modules))
        self.register_buffer("_warm", torch.zeros((), dtype=torch.long))
        self.register_buffer("eps_star", torch.ones(()))

    @torch.no_grad()
    def update(self, err: Tensor, enabled: bool = True) -> Tensor:
        """err: (M,) per-module prediction error this step."""
        self.ebar.mul_(self.ema).add_((1 - self.ema) * err.detach())
        self._warm += 1
        # eps* tracks a running quantile of the module error distribution
        q = torch.quantile(self.ebar, self.eps_pct)
        self.eps_star.mul_(0.995).add_(0.005 * q)
        if not enabled:
            return self.D
        r = self.ebar / self.eps_star.clamp(min=1e-8)
        grow = self.kappa * (1 - self.D) * (1 - r).clamp(min=0)
        shrink = self.lam * self.D * (r - 1).clamp(min=0)
        self.D.add_(grow - shrink).clamp_(0.0, 1.0)
        return self.D

    def lr_scale(self) -> Tensor:
        return (1 - self.D).clamp(min=0) ** self.p

    def may_write(self) -> Tensor:
        return self.D < self.write_thresh
