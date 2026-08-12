# CHROMA

[![propositions](https://github.com/merolaagi/chroma/actions/workflows/tests.yml/badge.svg)](https://github.com/merolaagi/chroma/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Chromatin-Regulated Hierarchy of Reference-frame Object Modules with Action-conditioned prediction**

> **Status: research pilot with two failing hypotheses.** The geometric half of
> the architecture (§1–§3) is proved and empirically confirmed. The
> gene-regulatory half (§4) is the novel part, and two of its three claims
> currently fail their own pre-registered tests — see
> [Results](#results-so-far--preliminary--400-steps-2-seeds-1-cpu). This repo is
> published in that state deliberately: the ablations that produce the negative
> results are the most useful thing in it.

A reference implementation of the three-timescale architecture: Monty-style
reference frames and voting, JEPA-style predictive representation learning, and
a gene-regulatory slow state that gates module identity, parameterisation, and
plasticity.

```
fast    (per step)     equivariant sensorimotor prediction   -> gradient
medium  (per episode)  pose-consistent evidence + voting      -> confidence
slow    (per tau_g)    regulatory attractor state             -> differentiation
```

One currency flows through all three: **prediction error in latent space**.

---

## Start here: the plain-language explainer

**[docs/CHROMA-explained.pdf](docs/CHROMA-explained.pdf)** — six pages, four
diagrams, no mathematics assumed. It uses a warehouse picking robot as the
running example and includes the scoreboard of which claims currently hold and
which do not.

Rebuild it from source with:

```bash
pip install reportlab svglib
python docs/make_figures.py && python docs/build_pdf.py
```

---

## Iterating

One command per iteration. It refuses to ship if a proposition test fails,
because a failing proposition means a claim in this README or in the explainer
PDF has become false.

```bash
make ship M="what changed"     # verify -> rebuild docs -> commit -> push
make dry  M="what changed"     # same, but stop before pushing
make test                      # proposition suite only
make docs                      # regenerate figures + PDF from results/
make results                   # full E1 sweep, 5 seeds x 3 arms (overnight)
```

`docs/figures/fig4_e1_bars.svg` is generated from whatever is in `results/`, not
hand-written. Re-run an experiment and the figure, its caption, and the PDF all
update to match — including the verdict line, which flips from "the distinct
modes add nothing" to "the distinct modes are doing work" if the numbers ever
separate. The document cannot silently drift from the evidence.

Two CI checks run on every push: the proposition suite must print `ALL PASS`,
and the committed figures must match what `make_figures.py` regenerates from the
committed results.

---

## Spec-to-code map

| Spec | File | Key object |
|---|---|---|
| §1 Latent geometry, exact unitary rep of `Z² ⋊ Cₙ` | `chroma/groups.py` | `SE2Rep`, `GroupAction` |
| §2 `ŝ = ρ(d)[s + Δ_φ]`, gene-modulated equivariant residual | `chroma/predictor.py` | `Predictor`, `EquivariantResidual`, `ProgramLinear` |
| §3 Canonical memory `s̃ = ρ(p)⁻¹s` | `chroma/memory.py` | `CanonicalMemory` |
| §4.1 Attractor dynamics (dense associative memory) | `chroma/regulatory.py` | `RegulatoryState` |
| §4.2 Sparse cis-regulation + shared gene programs | `chroma/regulatory.py` | `EnhancerReadout` |
| §4.3 Error-gated differentiation | `chroma/regulatory.py` | `Differentiation` |
| §5 Losses + collapse instrumentation | `chroma/losses.py` | `loop_loss`, `CollapseMonitor` |
| §6 Pushforward voting, JS disagreement | `chroma/voting.py` | `VotingBus`, `HypothesisGrid` |
| §7 Four-phase developmental curriculum | `chroma/train.py` | `Trainer` |
| §8 Experiments E1–E4 | `experiments/e1_to_e4.py` | |
| §9 Minimal tactile instance | `chroma/world.py` | `TactileWorld` |

Every proposition is an executable assertion in `tests/test_propositions.py`.

---

## Install and verify

```bash
pip install torch
PYTHONPATH=. python tests/test_propositions.py
```

Expected: eleven `PASS` lines and `ALL PASS`. If a proposition test fails, the
corresponding claim in the write-up is false and the write-up is what should
change.

## Train

```python
from chroma import CHROMA, ChromaConfig, TactileWorld, Trainer, TrainConfig

model = CHROMA(ChromaConfig())
world = TactileWorld(n_regimes=3, objs_per_regime=4)
Trainer(model, world, TrainConfig()).run()
```

## Experiments

```bash
python experiments/e1_to_e4.py e4                 # cheapest, run first
python experiments/e1_to_e4.py e1  --steps 20000  # the load-bearing one
python experiments/e1_to_e4.py e2e3 --steps 20000
```

**E1 is the experiment that decides whether this is a real idea.** If ablating
the attractor dynamics for a matched-parameter MLP does not blow up the regime
switching rate and cost backward transfer, then `regulatory.py` is decoration
and should be deleted.

---

## Corrections the code forced on the spec

- **"CHROMA loop drift is flat in path length" was stated too loosely.** It is
  flat only in the rigid limit `Δ_φ = 0`. With a trained residual, drift grows
  in path length for both CHROMA and the ablation. The clean isolation is
  drift with `Δ_φ` disabled at eval — measured in `e4_loop_drift` under
  `transport_only`, where CHROMA gives float32 noise (~1e-6) and the learned
  dense transition gives ~0.65 at length 32.
- **The vote-distillation gradient path does not exist yet.** Memory lookups
  are non-differentiable, so evidence tensors carry no gradient. `L_vote` is
  computed for monitoring and scaled to zero in `train.py`; enabling it
  requires making the evidence inner product differentiable through the
  encoder. This is flagged in-line, not hidden.
- **Memory lookup was the system bottleneck.** The per-hypothesis Python sweep
  over neighbouring hash cells cost ~5 s/episode. `CanonicalMemory.query` is
  now a single batched distance computation; the hash is retained only for
  write-time deduplication. 15× speedup, no change in semantics.

## Known gaps

- `RegulatoryState` may be doing nothing that an EMA on `u_t` would not do —
  see the E1 three-arm result above. This is the largest open question in the
  architecture.

- `L_vote` gradient path (above).
- `TactileWorld` restricts rotations to the `C₄` subgroup so that sensor
  rotation is exact on the pixel lattice. The representation carries the full
  `C₈`; extending the world requires interpolated patch rotation, which would
  contaminate the E4 drift measurement.
- No 3D / `SE(3)` instance. The `groups.py` construction generalises (Wigner
  D-matrices for the rotational part, the same Fourier construction for
  translation) but is not implemented.
- Single-object episodes only; no scene composition, no occlusion.

---

## Results so far (preliminary — 400 steps, 2 seeds, 1 CPU)

The spec calls for 20,000 steps and 5 seeds. Everything below is 1/50th of that
and should be read as a pilot, not an answer. It is reported as measured.

### E4 — loop closure: **supported**

Closed-path drift at length 32, with `Δ_φ` disabled to isolate transport:

| | drift |
|---|---|
| CHROMA (exact `ρ`) | 9.5e-07 (float32 noise) |
| learned dense transition | 0.649 |

Ratio 6.8e5. Proposition 1 confirmed empirically.

### E1 — three arms

The third arm is the discriminator. `ablate_grn` removes recurrence *and*
multistability together; `ablate_hysteresis` sends `β → 0`, flattening the
Krotov–Hopfield landscape to one basin while keeping the relaxation dynamics,
`τ_g`, and the input coupling `B` intact.

| arm | n | switch rate | mean err | basins |
|---|---|---|---|---|
| CHROMA (recurrence + basins) | 2 | 0.00234 ± 0.00022 | 0.0717 | 6.0 |
| flat landscape (recurrence only) | 2 | 0.00250 ± 0.00000 | 0.0707 | 1.0 |
| MLP (neither) | 2 | 0.01078 ± 0.00685 | 0.0790 | 1.0 |

- `ablate_grn / chroma` = **4.60×** — below the pre-registered 5× threshold.
- `ablate_hysteresis / chroma` = **1.07×** — indistinguishable.

**The flat-landscape arm matches CHROMA on every metric.** At this scale the
advantage over the memoryless MLP is attributable to *slow recurrent
integration* — the `τ_g` timescale — and not to multistability or hysteresis.
Section 4.1 of the write-up claims the opposite, and the claim is not supported
by this pilot.

If this survives at 20k steps and 5 seeds, the honest conclusion is that
`RegulatoryState` should be replaced by an EMA low-pass filter on `u_t`, which
gets the same benefit with none of the Hopfield machinery. The gene-regulatory
framing would then have earned exactly one thing — a timescale separation —
which is a much smaller claim than the one in the spec.

### E3 — lineage structure: **not supported**

| metric | value | threshold |
|---|---|---|
| monotone pair fraction | 0.214 | > 0.7 |
| separation growth | 0.025 | > 0 |
| final mean pairwise distance | 0.025 | — |

Separation does grow from zero, but module expression profiles re-merge
frequently and the final spread is tiny — modules are nearly identical in
expression space. No lineage-like progressive branching at this scale. Likely
under-trained (the `L0` enhancer target has not converged in 320 steps), but
reported as measured.

### Still to run

E2 (attractor count vs. hidden regime count) and the full-scale E1/E3.
`aggregate.py` reads every `results/e1_*.json` checkpoint, so runs can be
accumulated across sessions:

```bash
for s in 0 1 2 3 4; do
  for arm in chroma ablate_hysteresis ablate_grn; do
    python run_e1_chunk.py $arm 20000 $s
  done
done
python aggregate.py
```
