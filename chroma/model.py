"""
The assembled architecture.

Three timescales, one currency.

    fast    (per step)    equivariant sensorimotor prediction   -> gradient
    medium  (per episode) pose-consistent evidence + voting      -> confidence
    slow    (per tau_g)   regulatory attractor state             -> differentiation

The currency is prediction error in latent space.  It trains Delta_phi, weights
the votes, and drives D_m.  That single fact is what makes this an architecture
rather than three papers stapled together.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor

from .encoder import EMATarget, PatchEncoder
from .groups import GroupAction, SE2Rep
from .losses import CollapseMonitor, loop_loss, prediction_loss, vote_distillation_loss
from .memory import CanonicalMemory
from .predictor import Predictor
from .regulatory import (Differentiation, EnhancerReadout, FlatRegulator,
                         MLPRegulator, RegulatoryState)
from .voting import HypothesisGrid, VotingBus

__all__ = ["ChromaConfig", "CHROMA", "PHASES"]


PHASES = {
    "pluripotent":   dict(grn=False, masks=False, memory=False, voting=False, diff=False),
    "specification": dict(grn=True,  masks=True,  memory=False, voting=False, diff=False),
    "memory_voting": dict(grn=True,  masks=True,  memory=True,  voting=True,  diff=False),
    "continual":     dict(grn=True,  masks=True,  memory=True,  voting=True,  diff=True),
}


@dataclass
class ChromaConfig:
    # Section 1 - geometry
    n_scalar: int = 16
    n_orbits: int = 3
    n_rot: int = 8
    # Section 2 - predictor
    n_z: int = 8
    width: int = 48
    depth: int = 3
    delta_max: float = 0.30
    # Section 4 - regulation
    n_g: int = 16
    n_proto: int = 6
    n_prog: int = 12
    rank: int = 4
    enhancer_occupancy: int = 4
    tau_g: int = 200
    beta_hopfield: float = 4.0
    kappa: float = 1e-3
    lam_diff: float = 1e-2
    # agent
    n_modules: int = 8
    patch: int = 3
    n_objects: int = 12
    # Section 5 - loss weights
    alpha_loop: float = 1.0
    beta_vote: float = 0.3
    gamma_reg: float = 0.01
    lam_cov: float = 0.04
    lam_var: float = 0.50    # VICReg variance hinge. Tuned under the FULL phase
                             # schedule, not just the pluripotent phase -- the
                             # first tuning pass measured only phase 0 and chose
                             # a value that collapsed once memory and voting
                             # switched on. min per-dim std / error trend:
                             #   0.00 -> 0.005 / 0.28   collapsed
                             #   0.20 -> 0.043 / 0.33   marginal, alarm fires
                             #   0.50 -> 0.130 / 0.42   clears with margin
                             #   1.00 -> 0.353 / 0.71   learning stalled
    # ablation switches (experiment E1 / E4)
    ablate_grn_dynamics: bool = False     # E1 arm 2: no recurrence, no basins
    ablate_hysteresis: bool = False       # E1 arm 3: recurrence kept, basins off
    ablate_equivariance: bool = False     # E4: rho -> learned dense transition
    lr: float = 3e-4
    seed: int = 0


class DenseTransition(nn.Module):
    """E4 ablation: replace exact transport rho(d) with a learned dense map."""

    def __init__(self, D: int, n_rot: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D + 2 + n_rot, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, D))
        self.n_rot = n_rot

    def forward(self, s: Tensor, d: GroupAction) -> Tensor:
        j1h = torch.nn.functional.one_hot(d.j % self.n_rot, self.n_rot).float()
        return s + self.net(torch.cat([s, d.v, j1h], -1))


class CHROMA(nn.Module):
    def __init__(self, cfg: ChromaConfig):
        super().__init__()
        torch.manual_seed(cfg.seed)
        self.cfg = cfg

        # --- Section 1: exact unitary representation -------------------------
        self.rep = SE2Rep(cfg.n_scalar, cfg.n_orbits, cfg.n_rot)
        D = self.rep.D

        # --- Section 2: encoder + equivariant predictor ----------------------
        self.encoder = PatchEncoder(cfg.patch, D, n_z=cfg.n_z)
        self.target = EMATarget(self.encoder)
        self.predictor = Predictor(
            self.rep, n_z=cfg.n_z, width=cfg.width, n_prog=cfg.n_prog,
            rank=cfg.rank, depth=cfg.depth, delta_max=cfg.delta_max)
        self.dense_transition = (
            DenseTransition(D, cfg.n_rot) if cfg.ablate_equivariance else None)

        # --- Section 4: regulatory layer -------------------------------------
        Reg = (MLPRegulator if cfg.ablate_grn_dynamics else
               FlatRegulator if cfg.ablate_hysteresis else RegulatoryState)
        self.regulator = Reg(n=cfg.n_g, n_proto=cfg.n_proto, n_input=3,
                             beta=cfg.beta_hopfield, tau_g=cfg.tau_g)
        self.readout = EnhancerReadout(cfg.n_modules, cfg.n_g, cfg.n_prog,
                                       cfg.enhancer_occupancy)
        self.diff = Differentiation(cfg.n_modules, kappa=cfg.kappa, lam=cfg.lam_diff)

        # --- Section 3 / 6: memory and voting --------------------------------
        self.memory = CanonicalMemory(self.rep)
        self.grid = HypothesisGrid(self.rep, extent=2, stride=1.0)
        self.bus = VotingBus(self.rep, self.grid, cfg.n_modules, cfg.n_objects)

        self.monitor = CollapseMonitor(D)
        self.phase = "pluripotent"
        self._t = 0

    # ------------------------------------------------------------------ phase

    def set_phase(self, name: str) -> None:
        assert name in PHASES, name
        self.phase = name

    @property
    def flags(self) -> dict:
        return PHASES[self.phase]

    # ------------------------------------------------------------- expression

    def expression(self) -> Tensor | None:
        """e_m for every module, or None during the pluripotent phase."""
        if not self.flags["masks"]:
            return None
        return self.readout(self.regulator.g)

    # --------------------------------------------------------------- forward

    def transport(self, s: Tensor, d: GroupAction) -> Tensor:
        if self.dense_transition is not None:
            return self.dense_transition(s, d)
        return self.rep.act(s, d)

    def predict(self, s: Tensor, d: GroupAction, z: Tensor,
                e: Tensor | None) -> Tensor:
        if self.dense_transition is not None:
            return self.dense_transition(s + self.predictor.delta(s, z, e), d)
        return self.predictor(s, d, z, e)

    def step(self, x_t: Tensor, x_next: Tensor, d: GroupAction,
             e: Tensor | None) -> tuple[Tensor, Tensor, Tensor]:
        """One fast-loop step for all modules at once.

        x_t, x_next: (M, patch*patch).  Returns (loss, per-module error, s_t).
        """
        s, z = self.encoder(x_t, want_z=True)
        s_hat = self.predict(s, d, z, e)
        with torch.no_grad():
            s_tgt = self.target(x_next)
        loss, per = prediction_loss(s_hat, s_tgt, z, lam_cov=self.cfg.lam_cov,
                                    lam_var=self.cfg.lam_var)
        return loss, per, s

    def loop_term(self, x_t: Tensor, path: list[GroupAction],
                  e: Tensor | None) -> Tensor:
        s = self.encoder(x_t)
        M = s.shape[0]
        path_b = [GroupAction(p.v.expand(M, 2), p.j.expand(M)) for p in path]
        if self.dense_transition is not None:
            cur = s
            for d in path_b:
                cur = self.dense_transition(cur + self.predictor.delta(
                    cur, torch.zeros(M, self.cfg.n_z), e), d)
            return ((cur - s.detach()) ** 2).mean()
        return loop_loss(self.predictor, self.rep, s, path_b, e)

    # ---------------------------------------------------------- slow loop tick

    @torch.no_grad()
    def regulatory_tick(self, mean_err: float, disagree: float,
                        reward: float = 0.0) -> Tensor:
        u = torch.tensor([mean_err, disagree, reward], dtype=torch.float32)
        if self.flags["grn"]:
            return self.regulator.maybe_step(u, self._t)
        return self.regulator.g

    @torch.no_grad()
    def differentiation_tick(self, per_module_err: Tensor) -> Tensor:
        return self.diff.update(per_module_err, enabled=self.flags["diff"])

    # ---------------------------------------------- per-module plasticity gate

    @torch.no_grad()
    def apply_plasticity_gate(self) -> None:
        """eta_m = eta_0 (1 - D_m)^p, applied where module identity exists.

        Module-specific parameters (the enhancer rows) are gated per module;
        shared parameters are gated by the mean, which is the correct reduction
        because a shared weight's gradient is a sum over modules.
        """
        if not self.flags["diff"]:
            return
        scale = self.diff.lr_scale()                          # (M,)
        for p in (self.readout.log_alpha, self.readout.A, self.readout.b):
            if p.grad is not None:
                shape = [-1] + [1] * (p.grad.dim() - 1)
                p.grad.mul_(scale.reshape(shape))
        mean = float(scale.mean())
        for name, p in self.named_parameters():
            if name.startswith("readout.") or p.grad is None:
                continue
            p.grad.mul_(mean)

    # ------------------------------------------------------------------ stats

    @torch.no_grad()
    def snapshot(self, s: Tensor, true_obj: int | None = None,
                 cheap: bool = True) -> dict:
        basin = None
        if not cheap and hasattr(self.regulator, "basin_entropy"):
            basin = float(self.regulator.basin_entropy(n_init=256, steps=120))
        mi = None
        if self.flags["voting"] and true_obj is not None:
            mi = float(self.bus.mutual_information(true_obj))
        out = self.monitor.check(s, basin, mi)
        out["phase"] = self.phase
        out["mean_D"] = float(self.diff.D.mean())
        if s.shape[0] > 1:
            out["min_dim_std"] = round(float(s.std(0).min()), 4)
            out["alarm_dim_collapse"] = out["min_dim_std"] < 0.05
        out["memory_size"] = self.memory.size()
        e = self.expression()
        if e is not None:
            out["expression_argmax"] = e.argmax(-1).tolist()
        return out
