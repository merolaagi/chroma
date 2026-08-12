"""
Section 9 - The minimal instance: a 2D tactile world.

Deliberately not vision.  The agent has a small tactile receptive field and
moves over 2D shapes; actions are displacements in G = Z^2 x| C_n.  Everything
in Sections 1-6 is exercised, and the whole ablation suite E1-E4 runs on a
laptop CPU, which is the only scale at which those ablations will actually get
run.

Rotations are restricted to the C_4 subgroup (multiples of 90 degrees) so that
sensor rotation is *exact* on the pixel lattice.  The representation still
carries the full C_8; the world simply uses even rotation indices.  This keeps
Proposition 1 testable without interpolation error contaminating the drift
measurement in experiment E4.

Regimes are shape *families*.  Regime shifts are never announced to the model --
that is the point of experiment E2.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .groups import GroupAction

__all__ = ["TactileWorld", "Episode"]


def _bar(c: Tensor, k: int) -> None:
    c[6:10, 3 + k:13 + k] = 1.0


def _cross(c: Tensor, k: int) -> None:
    c[7:9, 3:13] = 1.0
    c[3 + k:13 + k, 7:9] = 1.0


def _ell(c: Tensor, k: int) -> None:
    c[4:12, 4:6] = 1.0
    c[10:12, 4:10 + k] = 1.0


def _ring(c: Tensor, k: int) -> None:
    yy, xx = torch.meshgrid(torch.arange(16.0), torch.arange(16.0), indexing="ij")
    r = ((yy - 8) ** 2 + (xx - 8) ** 2).sqrt()
    c[(r > 3.0 + 0.4 * k) & (r < 5.0 + 0.4 * k)] = 1.0


def _tee(c: Tensor, k: int) -> None:
    c[4:6, 3:13] = 1.0
    c[4:12 + k, 7:9] = 1.0


def _zig(c: Tensor, k: int) -> None:
    for i in range(4):
        y = 4 + 2 * i
        x = 4 + (2 * i if i % 2 == 0 else 2 * i + k)
        c[y:y + 2, x:x + 3] = 1.0


FAMILIES = [_bar, _cross, _ell, _ring, _tee, _zig]


class Episode:
    """One traversal of one object by one multi-sensor agent."""

    def __init__(self, obj_id: int, regime: int, canvas: Tensor):
        self.obj_id, self.regime, self.canvas = obj_id, regime, canvas


class TactileWorld:
    """Shapes on a 16x16 canvas; a k-sensor agent reads rotated patches."""

    def __init__(self, n_regimes: int = 3, objs_per_regime: int = 4,
                 canvas: int = 16, patch: int = 3, n_sensors: int = 8,
                 n_rot: int = 8, seed: int = 0):
        self.n_regimes, self.opr = n_regimes, objs_per_regime
        self.N, self.P, self.n_sensors, self.n_rot = canvas, patch, n_sensors, n_rot
        self.g = torch.Generator().manual_seed(seed)
        self.n_objects = n_regimes * objs_per_regime
        self.canvases = self._build()
        self.sensor_offsets = self._sensor_layout()
        self.regime = 0

    # ---------------------------------------------------------------- assets

    def _build(self) -> Tensor:
        out = torch.zeros(self.n_objects, self.N, self.N)
        for r in range(self.n_regimes):
            fam = FAMILIES[r % len(FAMILIES)]
            for k in range(self.opr):
                c = torch.zeros(self.N, self.N)
                fam(c, k)
                out[r * self.opr + k] = c
        return out

    def _sensor_layout(self) -> list[GroupAction]:
        """Fixed, known relative sensor poses -- the precondition for voting."""
        poses = []
        for i in range(self.n_sensors):
            ang = 2 * math.pi * i / self.n_sensors
            v = torch.tensor([[2.0 * math.cos(ang), 2.0 * math.sin(ang)]])
            j = torch.tensor([(2 * i) % self.n_rot])          # C_4 subgroup
            poses.append(GroupAction(v, j))
        return poses

    # --------------------------------------------------------------- regimes

    def set_regime(self, r: int) -> None:
        """Never surfaced to the model.  Experiment E2 asks whether the
        regulatory state discovers the regime count on its own."""
        self.regime = r % self.n_regimes

    def sample_object(self) -> int:
        lo = self.regime * self.opr
        return lo + int(torch.randint(self.opr, (1,), generator=self.g))

    # --------------------------------------------------------------- sensing

    def read(self, obj: int, v: Tensor, j: Tensor) -> Tensor:
        """Rotated tactile patch at object-frame pose (v, j).  (B, P*P)."""
        B = v.shape[0]
        out = torch.zeros(B, self.P * self.P)
        c = self.canvases[obj]
        h = self.P // 2
        for b in range(B):
            cy = int(torch.round(v[b, 1])) + self.N // 2
            cx = int(torch.round(v[b, 0])) + self.N // 2
            y0, x0 = cy - h, cx - h
            patch = torch.zeros(self.P, self.P)
            ys0, xs0 = max(0, y0), max(0, x0)
            ys1, xs1 = min(self.N, y0 + self.P), min(self.N, x0 + self.P)
            if ys1 > ys0 and xs1 > xs0:
                patch[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0] = c[ys0:ys1, xs0:xs1]
            q = (int(j[b]) // 2) % 4                          # C_4 -> exact rot90
            out[b] = torch.rot90(patch, -q, (0, 1)).reshape(-1)
        return out

    # --------------------------------------------------------------- actions

    def sample_action(self, batch: int) -> GroupAction:
        dv = torch.randint(-1, 2, (batch, 2), generator=self.g).float()
        dj = torch.randint(0, 4, (batch,), generator=self.g) * 2   # C_4 subgroup
        return GroupAction(dv, dj)

    def closed_path(self, length: int = 4) -> list[GroupAction]:
        """A path whose product is exactly the identity -- the free supervision
        for L_loop.

        Convention: displacements are *spatial* (world-frame), so poses update
        as p_{t+1} = d_t . p_t and the rollout operator is
        rho(d_L) ... rho(d_1) = rho(d_L . ... . d_1).  The accumulator must
        therefore left-multiply, and the closing action is its inverse.
        """
        rep = getattr(self, "_rep", None)
        assert rep is not None, "call world.bind_rep(rep) first"
        acts = [self.sample_action(1) for _ in range(length - 1)]
        acc = GroupAction.identity(1)
        for a in acts:
            acc = rep.compose(a, acc)
        acts.append(rep.inverse(acc))
        return acts

    def bind_rep(self, rep) -> None:
        self._rep = rep

    def new_episode(self) -> Episode:
        o = self.sample_object()
        return Episode(o, self.regime, self.canvases[o])
