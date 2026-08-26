r"""
Active sensing: AlphaGo-style search over where to move next.

Until now ``world.sample_action()`` picked displacements at random.  That is the
weakest line in the codebase, and it silently limited every experiment: Monty's
whole premise is that *where you move* determines how fast you recognise
something, and CHROMA was moving blind.

What transfers from AlphaGo, and what does not
----------------------------------------------
Self-play needs an adversary; there is none here.  Value networks bootstrap from
terminal outcomes; "recognised the object" is not a return to propagate.  Policy
distillation needs expert trajectories; none exist for tactile exploration.

What does transfer is the search itself: PUCT tree search over actions, with a
value that is not "probability of winning" but **expected information gain**.

    EIG(d) = H[P] - E_obs[ H[P | obs] ]

where P is the posterior over hypotheses (object identity, object pose).  Pick
the displacement whose predicted observation most separates the surviving
hypotheses.

Why the rollout is trustworthy here
-----------------------------------
Search is only as good as its simulator.  In a learned world model, a 20-step
rollout is fiction and deep search amplifies the error.  Here rho(d) is *exact*
transport with zero drift (Proposition 1, measured at 9.5e-07 over 32 moves), so
the tree can be expanded to real depth without the simulator degrading.  This is
the one place where CHROMA's geometry gives search something it could not get
from a generic world model.

The trick that makes simulation cheap
-------------------------------------
A movement does not require re-deriving anything.  Hypotheses live on a grid of
G, and moving by d maps grid pose h to d . h -- a *permutation of the grid*,
already available as ``HypothesisGrid.pushforward_index``.  Simulating an action
is an index permutation plus one memory lookup, not a forward pass.

Top-K pruning plays the role of AlphaGo's policy network: the posterior is
concentrated on a handful of hypotheses, so the (Q x Q) likelihood matrix is
computed over the top K only.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from .groups import GroupAction, SE2Rep

__all__ = ["ActionSet", "InfoGainPolicy", "MCTSPolicy", "RandomPolicy"]


class ActionSet:
    """The legal moves.  Small and discrete, like a board game's move list."""

    def __init__(self, rep: SE2Rep, radius: int = 1, rotations=(0, 2)):
        vs, js = [], []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for j in rotations:
                    vs.append([float(dx), float(dy)])
                    js.append(j)
        self.v = torch.tensor(vs)
        self.j = torch.tensor(js)
        self.n = len(js)
        self.rep = rep

    def get(self, i: int) -> GroupAction:
        return GroupAction(self.v[i:i + 1], self.j[i:i + 1])


class RandomPolicy:
    """Baseline: what the trainer did before."""

    def __init__(self, actions: ActionSet):
        self.actions = actions
        self.name = "random"

    def select(self, *_, **__) -> GroupAction:
        return self.actions.get(int(torch.randint(self.actions.n, (1,))))


class InfoGainPolicy:
    """Greedy one-step expected information gain.

    Strong baseline, and the honest control for MCTS: if tree search cannot beat
    greedy EIG at matched compute, the search is not buying anything and the
    extra machinery should be dropped.
    """

    def __init__(self, rep: SE2Rep, grid, actions: ActionSet, memory,
                 top_k: int = 24, tau: float = 0.10):
        self.rep, self.grid, self.actions = rep, grid, actions
        self.memory, self.top_k, self.tau = memory, top_k, tau
        self.name = "greedy_eig"

    # ------------------------------------------------------------- internals

    @staticmethod
    def _entropy(p: Tensor) -> Tensor:
        return -(p * (p + 1e-12).log()).sum(-1)

    def _posterior(self, L: Tensor) -> tuple[Tensor, Tensor]:
        """Pool evidence across modules; return (top-K probs, flat indices)."""
        flat = L.sum(0).reshape(-1)
        k = min(self.top_k, flat.numel())
        vals, idx = flat.topk(k)
        return torch.softmax(vals, -1), idx

    def _predicted_latents(self, d: GroupAction, idx: Tensor,
                           sensor: GroupAction) -> tuple[Tensor, Tensor]:
        """What each surviving hypothesis predicts we would feel after move d.

        Returns (K, D) stored canonical latents and a (K,) validity mask.
        """
        H, O = self.grid.H, self.memory.rep.n_rot  # O unused; kept explicit
        objs = torch.div(idx, self.grid.H, rounding_mode="floor")
        hs = idx % self.grid.H
        gv, gj = self.grid.v[hs], self.grid.j[hs]
        K = idx.numel()
        db = GroupAction(d.v.expand(K, 2), d.j.expand(K))
        moved = self.rep.compose(db, GroupAction(gv, gj))
        sb = GroupAction(sensor.v.expand(K, 2), sensor.j.expand(K))
        p = self.rep.compose(moved, sb)
        out = torch.zeros(K, self.rep.D)
        ok = torch.zeros(K, dtype=torch.bool)
        for o in objs.unique():
            m = objs == o
            vals, valid = self.memory.query(int(o), p.v[m], p.j[m])
            out[m], ok[m] = vals, valid
        return out, ok

    def expected_info_gain(self, L: Tensor, d: GroupAction,
                           sensor: GroupAction) -> float:
        """H[P] - E_{h*~P} H[P | observation generated by h*]."""
        prob, idx = self._posterior(L)
        S, ok = self._predicted_latents(d, idx, sensor)
        if ok.sum() < 2:
            return 0.0
        S = F.normalize(S, dim=-1) * ok.unsqueeze(-1).float()
        lam = (S @ S.T) / self.tau                       # (K, K) log-likelihood
        prior = prob.log().unsqueeze(1)                  # (K, 1)
        post = torch.softmax(prior + lam, dim=0)         # column h* -> posterior
        h_post = self._entropy(post.T)                   # (K,)
        return float(self._entropy(prob) - (prob * h_post).sum())

    # ---------------------------------------------------------------- select

    def select(self, L: Tensor, sensor: GroupAction, **__) -> GroupAction:
        best, best_g = -1e18, 0
        for i in range(self.actions.n):
            g = self.expected_info_gain(L, self.actions.get(i), sensor)
            if g > best:
                best, best_g = g, i
        return self.actions.get(best_g)


class MCTSPolicy(InfoGainPolicy):
    """PUCT tree search over action sequences, valued by information gain.

    Selection follows AlphaGo's PUCT rule

        a = argmax  Q(s,a) + c_puct * P(a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))

    with the prior P(a) taken from one-step EIG rather than a policy network --
    there is no expert data to train one on, and EIG is a principled prior that
    costs one simulation per action.

    Backup is the discounted sum of information gained along the path, so the
    tree prefers action *sequences* that disambiguate, not just single moves
    that look good in isolation.
    """

    def __init__(self, *a, n_sim: int = 32, depth: int = 3,
                 c_puct: float = 1.4, gamma: float = 0.9, **kw):
        super().__init__(*a, **kw)
        self.n_sim, self.depth, self.c_puct, self.gamma = n_sim, depth, c_puct, gamma
        self.name = "mcts"

    def _apply(self, L: Tensor, d: GroupAction, sensor: GroupAction) -> Tensor:
        """Evidence after hypothetically moving by d and observing.

        The observation is taken to be the one predicted by the *current MAP*
        hypothesis -- a determinised rollout.  Sampling the observation instead
        would be more faithful but multiplies cost by the sample count, and the
        determinised version is what AlphaGo's own rollouts effectively did.
        """
        prob, idx = self._posterior(L)
        S, ok = self._predicted_latents(d, idx, sensor)
        if ok.sum() < 2:
            return L
        S = F.normalize(S, dim=-1) * ok.unsqueeze(-1).float()
        star = int(prob.argmax())
        lam = (S @ S[star]) / self.tau                   # (K,)
        out = L.clone()
        flat = out.sum(0).reshape(-1)
        flat = flat.clone()
        flat[idx] = flat[idx] + lam
        return flat.reshape(1, *out.shape[1:]).expand_as(out).contiguous()

    def select(self, L: Tensor, sensor: GroupAction, **__) -> GroupAction:
        n = self.actions.n
        prior = torch.tensor([self.expected_info_gain(L, self.actions.get(i), sensor)
                              for i in range(n)])
        prior = torch.softmax(prior / max(float(prior.std()), 1e-6), 0)
        N = torch.zeros(n)
        W = torch.zeros(n)

        for _ in range(self.n_sim):
            Q = torch.where(N > 0, W / N.clamp(min=1), torch.zeros(n))
            u = Q + self.c_puct * prior * math.sqrt(float(N.sum()) + 1) / (1 + N)
            a = int(u.argmax())
            # roll the chosen action forward `depth` steps, greedily thereafter
            # Rollout policy is a cheap sample from the prior, NOT the full
            # greedy search. Calling greedy select inside every rollout cost
            # 49 ms per step and made one MCTS decision ~1.8 s -- 36x greedy,
            # which no speedup in steps could repay. AlphaGo's rollouts used a
            # fast policy for the same reason.
            Ls, g, disc = L, 0.0, 1.0
            d = self.actions.get(a)
            for t in range(self.depth):
                g += disc * self.expected_info_gain(Ls, d, sensor)
                disc *= self.gamma
                if t < self.depth - 1:
                    Ls = self._apply(Ls, d, sensor)
                    d = self.actions.get(int(torch.multinomial(prior, 1)))
            N[a] += 1
            W[a] += g
        return self.actions.get(int(N.argmax()))
