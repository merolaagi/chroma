"""
Section 3 - Object memory: canonicalization, not storage.

Given latent ``s`` observed while the sensor sits at object-frame pose ``p``,
store the canonical form

    s_tilde = rho(p)^{-1} s

Proposition 2 (viewpoint invariance by construction).  Under a global rigid
motion h, s -> rho(h) s and p -> h . p, so

    s_tilde -> rho(h p)^{-1} rho(h) s = rho(p)^{-1} rho(h)^{-1} rho(h) s = s_tilde.

Consequence: each surface patch needs to be visited *once*, not once per
viewpoint.  Sample complexity O(|surface|) rather than O(|surface| x |views|).
This is the whole reason for building rho exactly in Section 1.

Writes are discrete and non-differentiable.  Nothing needs a gradient through
them -- gradients flow only through the encoder and Delta_phi.
"""

from __future__ import annotations

from collections import defaultdict

import torch
from torch import Tensor

from .groups import GroupAction, SE2Rep

__all__ = ["CanonicalMemory"]


class CanonicalMemory:
    """Per-object graph of (pose, canonical latent, visit count).

    Lookup is an O(1) hash on the discretised pose plus a fixed-radius sweep of
    neighbouring cells -- not an O(|M|) scan.  This is what keeps recognition
    cheap as models grow.
    """

    def __init__(self, rep: SE2Rep, cell: float = 1.0, merge_cos: float = 0.90,
                 capacity: int = 20000, max_per_cell: int = 3):
        self.rep, self.cell, self.merge_cos, self.capacity = rep, cell, merge_cos, capacity
        self.max_per_cell = max_per_cell
        # object id -> {cell key -> list of slot indices}
        self.index: dict[int, dict[tuple, list[int]]] = defaultdict(lambda: defaultdict(list))
        self.vals = torch.zeros(0, rep.D)
        self.counts = torch.zeros(0)
        self.owner = torch.zeros(0, dtype=torch.long)
        self.pose_v = torch.zeros(0, 2)
        self.pose_j = torch.zeros(0, dtype=torch.long)
        self._slot_cache: dict[int, Tensor] = {}
        self._dirty: dict[int, bool] = {}

    # --------------------------------------------------------------- hashing

    def _cell(self, v: Tensor, j: int) -> tuple:
        c = torch.round(v / self.cell).long().tolist()
        return (int(c[0]), int(c[1]), int(j) % self.rep.n_rot)

    def _neighbours(self, key: tuple, radius: int = 1):
        x, y, j = key
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dj in (-1, 0, 1):
                    yield (x + dx, y + dy, (j + dj) % self.rep.n_rot)

    # ----------------------------------------------------------------- write

    @torch.no_grad()
    def write(self, obj: int, s: Tensor, pose: GroupAction) -> None:
        """s: (D,) latent observed at object-frame ``pose`` (batch of 1)."""
        canon = self.rep.act_inv(s.unsqueeze(0), pose).squeeze(0)
        canon = canon / canon.norm().clamp(min=1e-6)
        key = self._cell(pose.v[0], int(pose.j[0]))
        bucket = self.index[obj][key]
        for slot in bucket:
            if float(self.vals[slot] @ canon) > self.merge_cos:
                n = self.counts[slot]
                self.vals[slot] = (self.vals[slot] * n + canon) / (n + 1)
                self.vals[slot] /= self.vals[slot].norm().clamp(min=1e-6)
                self.counts[slot] += 1
                return
        if len(bucket) >= self.max_per_cell:
            # finite capacity per pose cell: merge into the nearest prototype
            # rather than growing without bound.  Without this the graph grows
            # linearly in steps because early latents are unstable and never
            # clear the merge threshold.
            sims = torch.stack([self.vals[s] @ canon for s in bucket])
            slot = bucket[int(sims.argmax())]
            n = self.counts[slot]
            self.vals[slot] = (self.vals[slot] * n + canon) / (n + 1)
            self.vals[slot] /= self.vals[slot].norm().clamp(min=1e-6)
            self.counts[slot] += 1
            return
        if self.vals.shape[0] >= self.capacity:
            return
        slot = self.vals.shape[0]
        self.vals = torch.cat([self.vals, canon.unsqueeze(0)], 0)
        self.counts = torch.cat([self.counts, torch.ones(1)], 0)
        self.owner = torch.cat([self.owner, torch.tensor([obj])], 0)
        self.pose_v = torch.cat([self.pose_v, pose.v[:1].clone()], 0)
        self.pose_j = torch.cat([self.pose_j, pose.j[:1].clone()], 0)
        bucket.append(slot)
        self._dirty[obj] = True

    # ------------------------------------------------------------------ read

    def _slots(self, obj: int) -> Tensor:
        """Contiguous slot index for one object, rebuilt only when dirty.

        The hash in :meth:`write` handles deduplication; lookup is a single
        batched distance computation over that object's slots.  The earlier
        per-hypothesis Python sweep over neighbouring cells was ~100x slower
        than the rest of the model combined and made the voting phase the
        bottleneck of the whole system.
        """
        if not self._dirty.get(obj, True):
            return self._slot_cache[obj]
        idx = torch.tensor(
            sorted(s for b in self.index[obj].values() for s in b),
            dtype=torch.long) if obj in self.index else torch.zeros(0, dtype=torch.long)
        self._slot_cache[obj] = idx
        self._dirty[obj] = False
        return idx

    @torch.no_grad()
    def query(self, obj: int, pose_v: Tensor, pose_j: Tensor,
              max_dist: float = 2.5) -> tuple[Tensor, Tensor]:
        """Nearest stored canonical latent for each queried pose.

        pose_v: (H, 2), pose_j: (H,).  Returns (H, D) latents and (H,) validity.
        Matches beyond ``max_dist`` in the product metric are rejected rather
        than silently returning a far-away node.
        """
        H = pose_v.shape[0]
        out = torch.zeros(H, self.rep.D)
        ok = torch.zeros(H, dtype=torch.bool)
        slots = self._slots(obj)
        if slots.numel() == 0:
            return out, ok
        kv = self.pose_v[slots]                               # (N, 2)
        kj = self.pose_j[slots]                               # (N,)
        d = ((pose_v[:, None, :] - kv[None]) ** 2).sum(-1)     # (H, N)
        dj = (pose_j[:, None] - kj[None]).abs() % self.rep.n_rot
        dj = torch.minimum(dj, self.rep.n_rot - dj).float()
        d = d + dj ** 2
        best_d, best = d.min(-1)
        good = best_d <= max_dist ** 2
        out[good] = self.vals[slots[best[good]]]
        ok = good
        return out, ok

    @torch.no_grad()
    def query_all(self, n_objects: int, pose_v: Tensor, pose_j: Tensor,
                  max_dist: float = 2.5) -> tuple[Tensor, Tensor]:
        """Batched lookup for every object at once.

        pose_v: (Q, 2), pose_j: (Q,).  Returns (O, Q, D) latents, (O, Q) valid.

        The pose-distance matrix is computed for all Q sensor-hypothesis pairs
        at once but restricted to each object's own slots.  Batching over
        objects *without* that restriction is slower, not faster: the masked
        form costs O(Q x N_total) per object where the slot form costs
        O(Q x N_total / O).
        """
        Q = pose_v.shape[0]
        out = torch.zeros(n_objects, Q, self.rep.D)
        ok = torch.zeros(n_objects, Q, dtype=torch.bool)
        if self.vals.shape[0] == 0:
            return out, ok
        for o in range(n_objects):
            slots = self._slots(o)
            if slots.numel() == 0:
                continue
            kv, kj = self.pose_v[slots], self.pose_j[slots]
            d = ((pose_v[:, None, :] - kv[None]) ** 2).sum(-1)
            dj = (pose_j[:, None] - kj[None]).abs() % self.rep.n_rot
            d = d + torch.minimum(dj, self.rep.n_rot - dj).float() ** 2
            best_d, best = d.min(-1)
            good = best_d <= max_dist ** 2
            out[o][good] = self.vals[slots[best[good]]]
            ok[o] = good
        return out, ok

    def size(self) -> int:
        return int(self.vals.shape[0])

    def n_objects(self) -> int:
        return len(self.index)
