"""
Section 7 - The training algorithm.

The phase order is not optional.  Enabling memory before the encoder is trained
writes garbage nodes, which poison voting, which poisons the regulatory input.
The developmental curriculum happens to also be the correct optimisation
curriculum -- which is either a pleasing coincidence or the point.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from .groups import GroupAction
from .losses import vote_distillation_loss
from .model import CHROMA, ChromaConfig
from .world import TactileWorld

__all__ = ["TrainConfig", "Trainer"]


@dataclass
class TrainConfig:
    steps_pluripotent: int = 4000
    steps_specification: int = 4000
    steps_memory_voting: int = 6000
    steps_continual: int = 10000
    episode_len: int = 12
    loop_every: int = 4
    loop_len: int = 4
    regime_period: int = 5000          # undisclosed to the model
    accumulate_every: int = 3   # evidence is a discounted sum; subsampling it
                                # only rescales gamma, and it is the dominant
                                # cost of the medium loop
    write_every: int = 2
    log_every: int = 500
    deep_log_every: int = 2500
    grad_clip: float = 1.0
    err_floor: float = 1e-3     # below this the error signal is uninformative


class Trainer:
    def __init__(self, model: CHROMA, world: TactileWorld, tcfg: TrainConfig):
        self.model, self.world, self.tcfg = model, world, tcfg
        world.bind_rep(model.rep)
        self.opt = torch.optim.AdamW(model.parameters(), lr=model.cfg.lr,
                                     weight_decay=1e-4)
        self.bus_ready = False
        self.log: list[dict] = []
        self._prev_regime_state: Tensor | None = None
        self.switch_events = 0
        self.t = 0
        self._sbuf: list = []   # latent ring buffer for effective_rank

    # ------------------------------------------------------------------ setup

    def _ensure_bus(self) -> None:
        if not self.bus_ready:
            self.model.bus.set_relative_poses(self.world.sensor_offsets)
            self.bus_ready = True

    # ------------------------------------------------------------- one episode

    def episode(self) -> dict:
        m, w, tc = self.model, self.world, self.tcfg
        M = m.cfg.n_modules
        ep = w.new_episode()
        e = m.expression()

        # agent's own pose in the object frame
        r = getattr(w, "pose_range", 2)
        pose = GroupAction(torch.randint(-r, r + 1, (1, 2)).float(),
                           torch.randint(0, 4, (1,)) * 2)
        m.bus.reset()

        tot_loss = torch.zeros(())
        errs = torch.zeros(M)
        n = 0
        s_last = None

        for step in range(tc.episode_len):
            d = w.sample_action(1)
            nxt = m.rep.compose(d, pose)   # spatial displacement

            # sensor poses = agent pose composed with fixed known offsets
            sv, sj, nv, nj = [], [], [], []
            for off in w.sensor_offsets:
                p = m.rep.compose(pose, off)
                q = m.rep.compose(nxt, off)
                sv.append(p.v); sj.append(p.j); nv.append(q.v); nj.append(q.j)
            sv = torch.cat(sv); sj = torch.cat(sj)
            nv = torch.cat(nv); nj = torch.cat(nj)

            x_t = w.read(ep.obj_id, sv, sj)
            x_n = w.read(ep.obj_id, nv, nj)
            d_b = GroupAction(d.v.expand(M, 2), d.j.expand(M))

            loss, per, s = m.step(x_t, x_n, d_b, e)
            tot_loss = tot_loss + loss
            errs += per
            n += 1
            s_last = s
            self._sbuf.append(s.detach())
            m.monitor.observe_input(x_t.detach())
            self._sbuf = self._sbuf[-64:]

            if m.flags["memory"]:
                writable = m.diff.may_write()
                with torch.no_grad():
                    if step % tc.write_every == 0:
                        for k in range(M):
                            if writable[k]:
                                m.memory.write(ep.obj_id, s[k],
                                               GroupAction(sv[k:k + 1], sj[k:k + 1]))
                    if m.flags["voting"] and step % tc.accumulate_every == 0:
                        m.bus.accumulate_batch(s, w.sensor_offsets, m.memory)
            pose = nxt

        errs /= max(n, 1)
        tot_loss = tot_loss / max(n, 1)

        # --- closed-path consistency (Proposition 1 / experiment E4) ---------
        if self.t % tc.loop_every == 0:
            path = w.closed_path(tc.loop_len)
            x0 = w.read(ep.obj_id, sv, sj)
            tot_loss = tot_loss + m.cfg.alpha_loop * m.loop_term(x0, path, e)

        # --- voting + distillation (medium loop) -----------------------------
        disagree = 0.0
        if m.flags["voting"]:
            self._ensure_bus()
            conf = m.bus.confidence(m.diff.ebar)
            L_pre = m.bus.L.clone()
            L_post = m.bus.vote(m.diff.ebar)
            tot_loss = tot_loss + m.cfg.beta_vote * vote_distillation_loss(
                L_pre.requires_grad_(False), L_post, conf) * 0.0
            # NOTE: evidence tensors carry no gradient path in this
            # implementation (memory lookups are non-differentiable), so the
            # distillation term is computed for monitoring and scaled to zero.
            # Re-enable by making the evidence inner product differentiable
            # through the encoder.
            disagree = float(m.bus.disagreement())

        # --- regulatory penalties -------------------------------------------
        if m.flags["masks"]:
            tot_loss = tot_loss + m.cfg.gamma_reg * m.readout.l0()

        self.opt.zero_grad(set_to_none=True)
        tot_loss.backward()
        m.apply_plasticity_gate()
        torch.nn.utils.clip_grad_norm_(m.parameters(), tc.grad_clip)
        self.opt.step()
        m.target.update(m.encoder)

        # --- slow loop -------------------------------------------------------
        m._t = self.t
        m.differentiation_tick(errs)
        g_before = m.regulator.g.clone()
        m.regulatory_tick(float(errs.mean()), disagree)
        self._track_switch(g_before)

        return dict(loss=float(tot_loss.detach()), err=float(errs.mean()),
                    disagree=disagree, obj=ep.obj_id, s=s_last)

    # ------------------------------------------- E1 metric: switching rate

    def _track_switch(self, g_before: Tensor) -> None:
        m = self.model
        if not m.flags["masks"]:
            return
        with torch.no_grad():
            a = m.readout(g_before, hard=True).argmax(-1)
            b = m.readout(m.regulator.g, hard=True).argmax(-1)
        if self._prev_regime_state is None:
            self._prev_regime_state = b
            return
        # normalise per module so the rate is a probability, not a count
        self.switch_events += float((a != b).float().mean())
        self._prev_regime_state = b

    # ------------------------------------------------------------------- run

    def run(self) -> list[dict]:
        tc = self.tcfg
        schedule = [("pluripotent", tc.steps_pluripotent),
                    ("specification", tc.steps_specification),
                    ("memory_voting", tc.steps_memory_voting),
                    ("continual", tc.steps_continual)]
        for phase, n_steps in schedule:
            self.model.set_phase(phase)
            for _ in range(n_steps):
                if self.t % tc.regime_period == 0 and self.t > 0:
                    self.world.set_regime(
                        (self.world.regime + 1) % self.world.n_regimes)
                out = self.episode()
                if self.t % tc.log_every == 0:
                    deep = self.t % tc.deep_log_every == 0
                    sbuf = torch.cat(self._sbuf) if self._sbuf else out["s"]
                    snap = self.model.snapshot(sbuf, out["obj"], cheap=not deep)
                    snap.update(step=self.t, loss=out["loss"], err=out["err"],
                                disagree=out["disagree"],
                                switch_rate=self.switch_events / max(self.t, 1),
                                regime=self.world.regime)
                    if snap.get("alarm_dim_collapse"):
                        print(f"  !! dimensional collapse: min per-dim std "
                              f"{snap['min_dim_std']:.4f} < 0.05. Raise lam_var.")
                    if snap.get("alarm_representational"):
                        print(f"  !! representational collapse: latent rank "
                              f"{snap['effective_rank']:.1f} vs input rank "
                              f"{snap.get('input_rank', '?')} "
                              f"(ratio {snap.get('rank_ratio', '?')}). "
                              "Check lam_var, not the world.")
                    if snap["err"] < tc.err_floor:
                        snap["DEGENERATE"] = True
                        print("  !! prediction error below the floor "
                              f"({snap['err']:.2e} < {tc.err_floor:.0e}). Every "
                              "error-driven mechanism -- differentiation, vote "
                              "confidence, the regulatory input u -- is now being "
                              "fed ~zero. E1 arms will tie because none of them "
                              "is doing anything. Raise task difficulty "
                              "(TactileWorld graded/sensor_noise/pose_range) "
                              "before trusting this run.")
                    self.log.append(snap)
                    print(f"[{self.t:6d}] {phase:14s} loss={out['loss']:.4f} "
                          f"err={out['err']:.4f} dis={out['disagree']:.4f} "
                          f"D={snap['mean_D']:.3f} sd={snap.get('min_dim_std', 0):.3f} "
                          f"sw={snap['switch_rate']:.4f} mem={snap['memory_size']}")
                self.t += 1
        return self.log
