"""
CHROMA Lab - interactive demonstration server.

Runs the *real* model (not a JS reimplementation), so everything the UI shows is
a live forward pass: the same rho, the same memory, the same regulatory state
the experiments use.

    python webapp/server.py           # http://localhost:51847

Stdlib http.server only -- no framework dependency beyond what the package
already needs.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STATIC = Path(__file__).resolve().parent / "static"

from chroma import CHROMA, ChromaConfig, GroupAction, TactileWorld  # noqa: E402
from chroma.hypercube import (AdditiveTransport, FitnessPredictor,  # noqa: E402
                              cycle_epistasis, walsh_transform)
from chroma.landscapes import HotspotLandscape  # noqa: E402
from chroma.losses import effective_rank  # noqa: E402

PORT = 51847
LOCK = threading.Lock()


class Lab:
    """One live model plus the world it acts in."""

    def __init__(self):
        self.reset()

    def reset(self, seed: int = 0, train_steps: int = 0):
        torch.manual_seed(seed)
        self.cfg = ChromaConfig(seed=seed)
        self.model = CHROMA(self.cfg)
        self.world = TactileWorld(n_regimes=3, objs_per_regime=4, seed=seed)
        self.world.bind_rep(self.model.rep)
        self.model.set_phase("continual")
        self.model.bus.set_relative_poses(self.world.sensor_offsets)
        self.obj = self.world.sample_object()
        self.pose = GroupAction(torch.zeros(1, 2), torch.zeros(1, dtype=torch.long))
        self.history = []
        self.landscape = HotspotLandscape(L=8, n_hotspots=3, order=3,
                                          hotspot_scale=1.5, seed=seed)
        self.genotype = [0] * 8
        if train_steps:
            self.train(train_steps)

    # ------------------------------------------------------------------ world

    def train(self, steps: int):
        from chroma.train import TrainConfig, Trainer
        per = max(steps // 4, 1)
        tr = Trainer(self.model, self.world, TrainConfig(
            steps_pluripotent=per, steps_specification=per,
            steps_memory_voting=per, steps_continual=per, log_every=10 ** 9))
        tr.run()
        self.model.bus.set_relative_poses(self.world.sensor_offsets)

    def sensor_poses(self, pose=None):
        pose = self.pose if pose is None else pose
        vs, js = [], []
        for off in self.world.sensor_offsets:
            p = self.model.rep.compose(pose, off)
            vs.append(p.v); js.append(p.j)
        return torch.cat(vs), torch.cat(js)

    @torch.no_grad()
    def observe(self):
        sv, sj = self.sensor_poses()
        x = self.world.read(self.obj, sv, sj)
        s = self.model.encoder(x)
        return x, s, sv, sj

    @torch.no_grad()
    def step(self, dx: float, dy: float, dj: int, write: bool = True):
        """Move, predict, feel, compare -- one turn of the fast loop."""
        d = GroupAction(torch.tensor([[float(dx), float(dy)]]),
                        torch.tensor([int(dj) % self.cfg.n_rot]))
        M = self.cfg.n_modules
        x0, s0, sv0, sj0 = self.observe()
        e = self.model.expression()
        z = torch.zeros(M, self.cfg.n_z)
        pred = self.model.predict(s0, GroupAction(d.v.expand(M, 2), d.j.expand(M)), z, e)

        self.pose = self.model.rep.compose(d, self.pose)
        x1, s1, sv1, sj1 = self.observe()
        err = ((pred - s1) ** 2).mean(-1)

        if write:
            for k in range(M):
                self.model.memory.write(self.obj, s1[k],
                                        GroupAction(sv1[k:k + 1], sj1[k:k + 1]))
            self.model.bus.accumulate_batch(s1, self.world.sensor_offsets,
                                            self.model.memory)
        self.model.diff.update(err, enabled=True)
        self.history.append(float(err.mean()))
        self.history = self.history[-120:]
        return dict(patch=x1.tolist(), error=[round(float(v), 5) for v in err],
                    mean_error=round(float(err.mean()), 5))

    @torch.no_grad()
    def posterior(self):
        L = self.model.bus.L
        p = torch.softmax(L.sum(0).reshape(-1), -1).reshape(self.cfg.n_objects, -1)
        by_obj = p.sum(-1)
        return [round(float(v), 4) for v in by_obj]

    @torch.no_grad()
    def loop_drift(self, length: int = 8):
        """Proposition 1, live: walk a closed path, measure the return error."""
        path = self.world.closed_path(length)
        _, s, _, _ = self.observe()
        M = s.shape[0]
        cur = s
        traj = []
        for d in path:
            db = GroupAction(d.v.expand(M, 2), d.j.expand(M))
            cur = self.model.rep.act(cur, db)
            traj.append(round(float((cur - s).norm(dim=-1).mean()), 8))
        full = self.model.predictor.rollout(
            s, [GroupAction(d.v.expand(M, 2), d.j.expand(M)) for d in path],
            None, self.model.expression())
        return dict(transport_only=traj,
                    transport_final=traj[-1],
                    with_residual=round(float((full - s).norm(dim=-1).mean()), 6),
                    path=[[float(d.v[0, 0]), float(d.v[0, 1]), int(d.j[0])]
                          for d in path])

    # ------------------------------------------------------------------ genes

    @torch.no_grad()
    def gene_state(self):
        g = self.model.regulator.g
        e = self.model.readout(g, hard=True)
        mask = self.model.readout.mask(hard=True)
        D = self.model.diff.D
        return dict(
            g=[round(float(v), 4) for v in g],
            expression=[[round(float(v), 4) for v in row] for row in e],
            mask=[[int(v > 0.5) for v in row] for row in mask],
            differentiation=[round(float(v), 4) for v in D],
            plasticity=[round(float(v), 4) for v in self.model.diff.lr_scale()],
            ebar=[round(float(v), 5) for v in self.model.diff.ebar],
            eps_star=round(float(self.model.diff.eps_star), 5),
            energy=round(float(self.model.regulator.energy(g.unsqueeze(0))), 4)
            if hasattr(self.model.regulator, "energy") else None,
        )

    @torch.no_grad()
    def set_gene(self, idx: int, value: float):
        self.model.regulator.g[int(idx)] = float(value)
        return self.gene_state()

    @torch.no_grad()
    def set_all_genes(self, vals):
        self.model.regulator.g.copy_(torch.tensor(vals, dtype=torch.float32))
        return self.gene_state()

    @torch.no_grad()
    def relax_genes(self, steps: int = 40, u=None):
        """Let the regulatory state fall into whichever basin it is nearest."""
        g = self.model.regulator.g.unsqueeze(0)
        ut = None if u is None else torch.tensor([u], dtype=torch.float32)
        g = self.model.regulator.relax(g, ut, steps=steps)
        self.model.regulator.g.copy_(g.squeeze(0))
        return self.gene_state()

    @torch.no_grad()
    def attractors(self):
        """The basins, projected to 2D so the landscape can be drawn."""
        reg = self.model.regulator
        if not hasattr(reg, "enumerate_attractors"):
            return dict(centres=[], counts=[], current=[0, 0], basis=None)
        c, counts = reg.enumerate_attractors(n_init=512, steps=200, tol=0.35)
        pts = torch.cat([c, reg.g.unsqueeze(0)], 0)
        mu = pts.mean(0, keepdim=True)
        U, S, V = torch.pca_lowrank(pts - mu, q=min(2, pts.shape[0] - 1))
        proj = (pts - mu) @ V[:, :2]
        return dict(centres=[[round(float(a), 3), round(float(b), 3)]
                             for a, b in proj[:-1]],
                    counts=[int(v) for v in counts],
                    current=[round(float(proj[-1, 0]), 3),
                             round(float(proj[-1, 1]), 3)])

    @torch.no_grad()
    def jump_to_basin(self, i: int):
        reg = self.model.regulator
        c, _ = reg.enumerate_attractors(n_init=512, steps=200, tol=0.35)
        if 0 <= i < c.shape[0]:
            reg.g.copy_(c[i])
        return self.gene_state()

    # -------------------------------------------------------------- hypercube

    @torch.no_grad()
    def hypercube(self):
        """The fitness-landscape retarget: mutations as group elements."""
        land = self.landscape
        g = torch.tensor([self.genotype], dtype=torch.float32)
        y = float(land.y[sum(v << k for k, v in enumerate(self.genotype))])
        w = land.true_walsh()
        f_wt, f_single = land.singles()
        b0, singles = AdditiveTransport.fit_from_singles(f_wt, f_single)
        add = AdditiveTransport(8, b0, singles, freeze=True)
        additive = float(add(g))
        eps = []
        for i in range(8):
            for j in range(i + 1, 8):
                e = cycle_epistasis(land.y, i, j, 8,
                                    torch.tensor(self.genotype, dtype=torch.float32))
                if abs(e) > 1e-6:
                    eps.append(dict(sites=[i, j], value=round(e, 4)))
        eps.sort(key=lambda d: -abs(d["value"]))
        return dict(genotype=self.genotype, fitness=round(y, 4),
                    additive_only=round(additive, 4),
                    epistasis=round(y - additive, 4),
                    cycles=eps[:8],
                    hotspots=[dict(sites=list(s), value=round(v, 3))
                              for s, v in land.true_hotspots()],
                    landscape=[round(float(v), 3) for v in land.y])

    def toggle_site(self, i: int):
        self.genotype[int(i)] ^= 1
        return self.hypercube()

    # ------------------------------------------------------------------ state

    @torch.no_grad()
    def snapshot(self):
        x, s, sv, sj = self.observe()
        canvas = self.world.canvases[self.obj]
        return dict(
            canvas=[[round(float(v), 3) for v in row] for row in canvas],
            object=self.obj, n_objects=self.cfg.n_objects,
            regime=self.world.regime,
            pose=[float(self.pose.v[0, 0]), float(self.pose.v[0, 1]),
                  int(self.pose.j[0])],
            sensors=[[float(sv[k, 0]), float(sv[k, 1]), int(sj[k])]
                     for k in range(sv.shape[0])],
            patches=[[round(float(v), 3) for v in row] for row in x],
            posterior=self.posterior(),
            memory=self.model.memory.size(),
            effective_rank=round(float(effective_rank(s)), 2),
            min_dim_std=round(float(s.std(0).min()), 4) if s.shape[0] > 1 else 0,
            history=self.history,
            genes=self.gene_state(),
        )


LAB = Lab()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200, ctype="application/json"):
        body = (json.dumps(obj) if ctype == "application/json"
                else obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p in ("/", "/index.html"):
            return self._send((STATIC / "index.html").read_text(), 200,
                              "text/html; charset=utf-8")
        with LOCK:
            if p == "/api/state":
                return self._send(LAB.snapshot())
            if p == "/api/genes":
                return self._send(LAB.gene_state())
            if p == "/api/attractors":
                return self._send(LAB.attractors())
            if p == "/api/hypercube":
                return self._send(LAB.hypercube())
            if p == "/api/loop":
                return self._send(LAB.loop_drift())
        self._send({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or "{}")
        p = self.path.split("?")[0]
        with LOCK:
            try:
                if p == "/api/step":
                    LAB.step(body.get("dx", 0), body.get("dy", 0), body.get("dj", 0))
                    return self._send(LAB.snapshot())
                if p == "/api/gene":
                    LAB.set_gene(body["index"], body["value"])
                    return self._send(LAB.gene_state())
                if p == "/api/genes":
                    LAB.set_all_genes(body["values"])
                    return self._send(LAB.gene_state())
                if p == "/api/relax":
                    LAB.relax_genes(body.get("steps", 40), body.get("u"))
                    return self._send(LAB.gene_state())
                if p == "/api/basin":
                    LAB.jump_to_basin(body["index"])
                    return self._send(LAB.gene_state())
                if p == "/api/site":
                    return self._send(LAB.toggle_site(body["index"]))
                if p == "/api/object":
                    LAB.obj = int(body["index"]) % LAB.cfg.n_objects
                    LAB.model.bus.reset()
                    return self._send(LAB.snapshot())
                if p == "/api/reset":
                    LAB.reset(int(body.get("seed", 0)), int(body.get("train", 0)))
                    return self._send(LAB.snapshot())
                if p == "/api/train":
                    LAB.train(int(body.get("steps", 300)))
                    return self._send(LAB.snapshot())
            except Exception as ex:  # surface errors in the UI, not the console
                return self._send({"error": f"{type(ex).__name__}: {ex}"}, 500)
        self._send({"error": "not found"}, 404)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    print(f"CHROMA Lab  ->  http://localhost:{port}")
    print("  model is live: every panel is a real forward pass")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
