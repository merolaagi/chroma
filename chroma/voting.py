"""
Section 6 - Lateral voting.

Each module maintains log-evidence over hypotheses (object identity, object
pose *in that module's own frame*).  Because module frames differ by a known
relative sensor pose T_nm, transferring evidence requires a genuine pushforward
of the hypothesis distribution, not a plain sum:

    L_n  <-  L_n + w_nm * T_{T_nm}[ sg L_m ],     w_nm = c_m / sum_m' c_m'

with confidence

    c_m = exp(-ebar_m / T) * (1 - H(L_m)/H_max)

The stop-gradient is not optional.  Without it the distillation term in
Section 5 produces groupthink within a few thousand steps and the collapse
metric ``vote_mutual_information`` climbs monotonically.

The disagreement signal returned here is the *only* path from the medium loop
back to the slow regulatory loop (Section 4.1 input u).
"""

from __future__ import annotations

import itertools

import torch
import torch.nn.functional as F
from torch import Tensor

from .groups import GroupAction, SE2Rep

__all__ = ["HypothesisGrid", "VotingBus"]


class HypothesisGrid:
    """Discretised pose hypothesis space over G = Z^2 x| C_n."""

    def __init__(self, rep: SE2Rep, extent: int = 2, stride: float = 1.0):
        self.rep = rep
        xs = torch.arange(-extent, extent + 1, dtype=torch.float32) * stride
        vs = torch.stack(torch.meshgrid(xs, xs, indexing="ij"), -1).reshape(-1, 2)
        js = torch.arange(rep.n_rot)
        self.v = vs.repeat_interleave(rep.n_rot, 0)          # (H, 2)
        self.j = js.repeat(vs.shape[0])                      # (H,)
        self.H = self.v.shape[0]
        self._perm_cache: dict[tuple, Tensor] = {}

    def nearest(self, v: Tensor, j: Tensor) -> Tensor:
        """Snap arbitrary poses onto the grid; returns flat indices."""
        d = ((self.v[None] - v[:, None]) ** 2).sum(-1)
        dj = (self.j[None] - j[:, None]).abs() % self.rep.n_rot
        dj = torch.minimum(dj, self.rep.n_rot - dj).float()
        return (d + dj ** 2).argmin(-1)

    def pushforward_index(self, T: GroupAction) -> Tensor:
        """Permutation h -> index of  T . pose_h  on the grid.

        Cached: relative sensor poses are fixed by construction, so this is
        computed once per module pair for the lifetime of the run.
        """
        key = (round(float(T.v[0, 0]), 4), round(float(T.v[0, 1]), 4), int(T.j[0]))
        if key in self._perm_cache:
            return self._perm_cache[key]
        Tb = GroupAction(T.v.expand(self.H, 2), T.j.expand(self.H))
        g = GroupAction(self.v, self.j)
        composed = self.rep.compose(Tb, g)
        idx = self.nearest(composed.v, composed.j)
        self._perm_cache[key] = idx
        return idx


class VotingBus:
    """Evidence accumulation, pushforward voting, and the disagreement signal."""

    def __init__(self, rep: SE2Rep, grid: HypothesisGrid, n_modules: int,
                 n_objects: int, gamma: float = 0.95, tau: float = 0.10,
                 conf_temp: float = 0.05):
        self.rep, self.grid = rep, grid
        self.M, self.O, self.H = n_modules, n_objects, grid.H
        self.gamma, self.tau, self.conf_temp = gamma, tau, conf_temp
        self.L = torch.zeros(n_modules, n_objects, grid.H)
        self.rel: dict[tuple[int, int], GroupAction] = {}

    def set_relative_poses(self, sensor_poses: list[GroupAction]) -> None:
        """T_nm = q_n^{-1} . q_m for every ordered pair of modules."""
        for n, m in itertools.permutations(range(self.M), 2):
            qn_inv = self.rep.inverse(sensor_poses[n])
            self.rel[(n, m)] = self.rep.compose(qn_inv, sensor_poses[m])

    def reset(self) -> None:
        self.L.zero_()

    # -------------------------------------------------------------- evidence

    @torch.no_grad()
    def accumulate(self, m: int, s: Tensor, q: GroupAction, memory) -> None:
        """Add one step of log-evidence for module ``m``.

        For each hypothesis (O, p0) the sensor's object-frame pose is p0 . q,
        and the evidence is the inner product between the canonicalised current
        latent and the stored canonical latent at that pose.
        """
        gv, gj = self.grid.v, self.grid.j
        qb = GroupAction(q.v.expand(self.H, 2), q.j.expand(self.H))
        p_sensor = self.rep.compose(GroupAction(gv, gj), qb)
        canon = self.rep.act_inv(s.expand(self.H, -1), p_sensor)
        canon = F.normalize(canon, dim=-1)
        for o in range(self.O):
            stored, ok = memory.query(o, p_sensor.v, p_sensor.j)
            ll = (canon * stored).sum(-1) / self.tau
            ll = torch.where(ok, ll, torch.zeros_like(ll))
            self.L[m, o] = self.gamma * self.L[m, o] + ll

    @torch.no_grad()
    def accumulate_batch(self, s_all: Tensor, sensor_poses: list[GroupAction],
                         memory) -> None:
        """Evidence for every module and every object in one pass.

        s_all: (M, D).  Semantically identical to calling :meth:`accumulate`
        once per module; the loops over modules, objects and hypotheses are
        collapsed into tensor ops because that loop nest was the dominant cost
        of the whole system.
        """
        M, H = self.M, self.H
        gv, gj = self.grid.v, self.grid.j
        pv, pj = [], []
        for m in range(M):
            q = sensor_poses[m]
            qb = GroupAction(q.v.expand(H, 2), q.j.expand(H))
            p = self.rep.compose(GroupAction(gv, gj), qb)
            pv.append(p.v)
            pj.append(p.j)
        pv = torch.cat(pv)                                    # (M*H, 2)
        pj = torch.cat(pj)
        canon = self.rep.act_inv(
            s_all.repeat_interleave(H, 0), GroupAction(pv, pj))
        canon = F.normalize(canon, dim=-1)                    # (M*H, D)

        stored, ok = memory.query_all(self.O, pv, pj)         # (O, M*H, D)
        ll = (canon[None] * stored).sum(-1) / self.tau        # (O, M*H)
        ll = torch.where(ok, ll, torch.zeros_like(ll))
        ll = ll.reshape(self.O, M, H).permute(1, 0, 2)        # (M, O, H)
        self.L.mul_(self.gamma).add_(ll)

    # ---------------------------------------------------------------- voting

    def posterior(self, m: int) -> Tensor:
        return torch.softmax(self.L[m].reshape(-1), -1).reshape(self.O, self.H)

    def confidence(self, ebar: Tensor) -> Tensor:
        p = torch.stack([self.posterior(m).reshape(-1) for m in range(self.M)])
        H = -(p * (p + 1e-12).log()).sum(-1)
        Hmax = torch.log(torch.tensor(float(self.O * self.H)))
        sharp = (1 - H / Hmax).clamp(min=0)
        return torch.exp(-ebar / self.conf_temp) * sharp + 1e-6

    @torch.no_grad()
    def vote(self, ebar: Tensor, strength: float = 0.5) -> Tensor:
        """One round of pushforward voting.  Returns the pre-vote L for the
        distillation loss (which applies stop-gradient to the consensus)."""
        c = self.confidence(ebar)
        L_in = self.L.clone()
        for n in range(self.M):
            acc = torch.zeros_like(self.L[n])
            wsum = 0.0
            for m in range(self.M):
                if m == n:
                    continue
                idx = self.grid.pushforward_index(self.rel[(n, m)])
                acc += float(c[m]) * L_in[m][:, idx]
                wsum += float(c[m])
            if wsum > 0:
                self.L[n] = self.L[n] + strength * acc / wsum
        return L_in

    # ---------------------------------------------------------- disagreement

    def disagreement(self) -> Tensor:
        """Mean pairwise Jensen-Shannon divergence -- the surprise signal that
        drives g out of its current basin (Section 4.1)."""
        ps = torch.stack([self.posterior(m).reshape(-1) for m in range(self.M)])
        tot, cnt = torch.zeros(()), 0
        for n, m in itertools.combinations(range(self.M), 2):
            p, q = ps[n], ps[m]
            mix = 0.5 * (p + q)
            js = 0.5 * (p * ((p + 1e-12) / (mix + 1e-12)).log()).sum() \
               + 0.5 * (q * ((q + 1e-12) / (mix + 1e-12)).log()).sum()
            tot = tot + js
            cnt += 1
        return tot / max(cnt, 1)

    def mutual_information(self, true_obj: int) -> Tensor:
        """Groupthink metric: I(vote_m ; vote_n | O), conditioned on truth.

        Alarm if this rises for 5k consecutive steps (Section 8).
        """
        ps = torch.stack([self.posterior(m)[true_obj] for m in range(self.M)])
        ps = ps / ps.sum(-1, keepdim=True).clamp(min=1e-12)
        tot, cnt = torch.zeros(()), 0
        for n, m in itertools.combinations(range(self.M), 2):
            joint = torch.outer(ps[n], ps[m])
            emp = 0.5 * (torch.diag(ps[n] * ps[m]).sum())
            tot = tot + (emp - joint.mean()).abs()
            cnt += 1
        return tot / max(cnt, 1)
