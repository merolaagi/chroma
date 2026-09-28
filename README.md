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

## CHROMA Lab (interactive)

```bash
./run.sh          # http://localhost:51847
```

Eight panels, all driven by a live model — every one is a real forward pass, not
a JavaScript reimplementation:

1. **Tactile world** — eight sensor pads on an object; click or arrow-key to move
2. **Predict → feel → error** — the fast loop, with what each pad actually feels
3. **Voting & posterior** — pooled evidence across pads, against the chance line
4. **Genes** — drag any of the 16 regulatory dimensions and watch downstream
5. **Expression, enhancers, differentiation** — which programs each module runs,
   its sparse enhancer mask, and how plastic it still is
6. **Attractor landscape** — the basins in 2D; click one to jump the cell into it
7. **Loop closure** — walk a closed path; transport lands exactly, residual doesn't
8. **Mutation hypercube** — toggle mutations, split fitness into additive vs epistasis

Panels 4–6 are the gene-manipulation surface: `g` is directly draggable, and
expression, enhancer occupancy, differentiation and plasticity all update live.
Panel 6 lets you move the cell between attractor basins by clicking.

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

## The hypercube retarget (`chroma/hypercube.py`)

Same architecture, different group. On a combinatorially complete fitness
landscape the group is `(Z/2)^L` — genotypes in `{0,1}^L`, mutations acting by
XOR — and `ρ(d)` is the diagonal matrix of Walsh characters. Every proposition
from the SE(2) case holds here, and holds *exactly* rather than to float
precision, because the character values are ±1.

| SE(2) / tactile | (Z/2)^L / fitness landscape |
|---|---|
| `ρ(d)` rigid transport | additive (non-epistatic) mutation effects |
| `Δ_φ` non-rigid residual | epistasis |
| closed path must return | double-mutant cycle — the gap **is** the interaction |
| sparse enhancer masks | sparse epistatic hotspots |

**E5 — where it works, and where it does not.** Test RMSE normalised by
landscape standard deviation, 3 seeds, L=8:

| n_train | | additive | chroma | mlp |
|---|---|---|---|---|
| 40 | sparse hotspots | 2.096 | **0.142** | 1.134 |
| 200 | sparse hotspots | 2.320 | **0.045** | 0.055 |
| 40 | dense NK | 1.546 | 1.546 | **0.939** |
| 200 | dense NK | 1.493 | 1.493 | **0.033** |

Eight-fold better than the unconstrained network at 40 training genotypes when
interaction is sparse — and *strictly worse* when it is dense, where the L0 gate
shuts the residual off entirely and the model collapses to the additive
baseline. That boundary is the useful finding, not a bug to tune away.

```bash
python experiments/e5_hypercube.py hotspot
python experiments/e5_hypercube.py nk
PYTHONPATH=. python tests/test_hypercube.py
```

**Two corrections the code forced.** First, `ε_ij = 4·β_ij` is false whenever
third-order or higher terms exist — every subset containing both sites
contributes, which is the background dependence of measured epistasis.
`cycle_epistasis_expected` pins the exact accounting. Second, a single-mutant
assay does not measure `β_k`; it measures `β_k` plus every interaction
involving site k. Left uncorrected, the "exact" additive part is worse than no
model at all (test RMSE 6.91 vs a landscape σ of 3.0). The learned coefficients
are exactly the contaminating terms, so the correction is closed-form and costs
no parameters — with it, RMSE drops to 0.08.

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

### E9: the ceiling is 1.0, so recognition is a MODEL failure (2026-08-28)

Before changing the model again it was worth asking whether 0.287 is anywhere
near achievable. That question is answerable exactly: the generative model is
fully known (our canvases, a deterministic sensor plus Gaussian noise, known
displacements), so a Bayes-optimal observer can be written down and run.

    P(o | patches) ∝ sum over initial poses p0 of
                     prod over t of N(x_t ; patch(o, p0·d_1..t), sigma)

Result, 324 pose hypotheses x 12 objects, sigma = 0.02:

| touches | oracle accuracy | x chance |
|---|---|---|
| 1 | 0.438 | 5.25x |
| 2 | 0.750 | 9.00x |
| **3** | **1.000** | **12.00x** |
| 12 | 1.000 | 12.00x |

**A single touch from the optimal observer (0.438) beats CHROMA's twelve
(0.287). Three touches saturate at perfect object AND pose recovery.**

So the task is not hard, the world is not degenerate, and there is no ceiling
anywhere near where CHROMA sits. The gap is **0.71 accuracy**, and it belongs
entirely to the architecture: encoder, canonical memory, or evidence
accumulation. Everything measured in E6, E7 and E8 was tuning around a defect
four times larger than any effect those experiments could have detected.

This is the diagnostic that should have been run first. It costs 90 seconds and
would have redirected the whole search effort.

Next: bisect the gap. The oracle differs from CHROMA in three places — it
compares raw patches rather than learned latents, it enumerates all 324 poses
rather than a 45-cell grid, and it uses an exact Gaussian likelihood rather than
cosine similarity in latent space. Swapping them in one at a time localises the
defect.

### Verdict on the search layer: cut it (2026-08-26)

Three measurements of the same value function, at increasing levels of
correctness:

| build | within-state Spearman | advantage over random |
|---|---|---|
| original | +0.050 | +0.065 |
| after the evidence-normalisation fix | **-0.280** | -0.0005 |
| after making the forward model consistent | -0.028 | -0.0001 |

The middle row was self-inflicted: `accumulate_batch` was changed to normalise
evidence by valid-lookup count, and `InfoGainPolicy` was not changed with it, so
the search was simulating dynamics the environment no longer had. A search whose
simulator disagrees with the environment is worse than no search — hence the
systematic *negative* correlation, which is not noise.

Fixing the inconsistency removed the harm. It did not create any benefit.
Within-state Spearman is ~0 in every correct build: **expected information gain,
as computed here, does not predict which action will actually reduce
uncertainty.** Candidate causes are the determinised observation model, memory
too sparse for predicted latents to mean anything, and top-K pruning discarding
the hypotheses that would discriminate.

The recommendation is to cut `chroma/search.py` rather than debug it further.
It has now consumed two retractions and produced no positive result, and the
thing it needs — a recognition system whose posterior means something — is the
open problem underneath it. If recognition ever reaches usable accuracy, the
search layer is worth reviving; until then it is a solution waiting for its
premise.

### E8: recognition was the bottleneck, and it was a bug (2026-08-26)

Recognition accuracy sat at 0.21 against a 1/12 = 0.083 chance baseline, which
made E1 and E6 comparisons between variants of a system that barely worked.
E8 measures the base directly, with no search and no regulatory arms.

The diagnostic was the *pairing*: accuracy 0.24 at confidence **0.80**. Not
uncertain -- confidently wrong. That points at evidence accumulation, not at
memory capacity (only ~2,800 nodes at this scale, far below any cap).

`VotingBus.accumulate_batch` summed raw similarities, so evidence magnitude
scaled with the **number of valid memory lookups** rather than with match
quality. A hypothesis whose queried poses happened to fall in a densely explored
region out-accumulated one that matched better but was queried where memory was
sparse. The system was biased toward well-explored objects, not correct ones.

Dividing by the valid-lookup count makes evidence a mean similarity, so
hypotheses compete on fit rather than coverage:

| | accuracy | x chance | confidence |
|---|---|---|---|
| sum (as shipped) | 0.240 | 2.88x | 0.797 |
| normalised (25 episodes, 1 seed) | 0.440 | 5.28x | 0.224 |
| **normalised (80 episodes, 2 seeds)** | **0.287** | **3.45x** | 0.233 |

The 0.440 figure was a single noisy measurement over 25 episodes and should not
have been reported as the result. At 80 episodes across 2 seeds the honest
number is **0.287**. The fix is real but modest: 0.21 -> 0.287, not a doubling.

What it did fix decisively is calibration — confidence 0.797 -> 0.233, from
wildly overconfident to roughly matched. Recognition is no longer actively
lying about its own certainty, but 0.287 is nowhere near usable.

Memory settings made no difference at all (identical 4,276 nodes and identical
accuracy across three configurations). That is correct rather than a bug: with
~4,300 nodes spread over ~16,000 pose cells the per-cell cap never binds. It
confirms capacity was never the bottleneck.

**This invalidates nothing in E1 -- all three arms shared the bug equally -- but
it does mean E6 was testing action policies on top of a miscalibrated evidence
model, which is the likeliest reason the EIG value function had no within-state
ranking power.** Re-run E7 before drawing any conclusion about search.

### E6: active sensing (`chroma/search.py`) — implemented, NOT validated

`world.sample_action()` picked displacements at random, which silently limited
every experiment: Monty's premise is that *where you move* determines how fast
you recognise something, and CHROMA was moving blind.

What transfers from AlphaGo is the search, not the rest. Self-play needs an
adversary; value networks need terminal returns; policy distillation needs
expert trajectories. None exist here. PUCT tree search does transfer, with the
value replaced by **expected information gain** — pick the displacement whose
predicted observation most separates the surviving hypotheses.

Two things make it cheap. A movement is a *permutation of the hypothesis grid*
(`pushforward_index`), so simulating an action is an index permutation plus a
memory lookup, not a forward pass. And because `rho(d)` is exact transport with
zero drift, the tree can be expanded to real depth without the simulator
degrading — which a learned world model cannot offer.

Pilot, 400 training steps, 6 episodes, 1 seed:

| policy | steps | accuracy | ms/decision |
|---|---|---|---|
| random | 10.17 | 0.17 | 261 |
| greedy EIG | 10.67 | 0.00 | 320 |
| MCTS | 11.17 | 0.50 | 500 |

**Run at 1500 steps / 2 seeds / 12 episodes: NOT SUPPORTED.**

| policy | steps | accuracy | sec/ep |
|---|---|---|---|
| random | 13.21 | 0.21 | 0.152 |
| greedy EIG | 13.21 | **0.33** | 0.307 |
| MCTS | 13.92 | 0.17 | 1.023 |

Search buys nothing: MCTS is worse than random on both metrics while costing
6.7x more. Two things are wrong with the experiment, though, and both were
found by reading these numbers.

*The metric was saturated.* All three finished at 13.2-13.9 steps against a cap
of 20, meaning the 0.60 confidence target was almost never reached. The
comparison was between three policies that all failed. Cap and target adjusted.

*The tree simulator had a real bug.* `MCTSPolicy._apply` summed evidence across
modules and broadcast the sum back to every module, multiplying total evidence
by M each application -- M^3 = 512x over a depth-3 rollout. The tree was scoring
hallucinated certainty. This is exactly why MCTS scored *below* its own greedy
prior: a search cannot beat its prior unless the simulator is sane. Fixed.

*The deeper read.* Random accuracy is 0.21 against a 1/12 = 0.083 chance
baseline. Recognition itself barely works, so no action policy can rescue it.
**The bottleneck is evidence quality, not action selection** -- which means E6
was testing the wrong thing, and object memory hitting its capacity cap in every
run is the more likely culprit.

**This does not support anything yet.** Steps-to-recognition is flat across all
three because the 0.60 confidence target is rarely reached inside the step cap,
so the headline metric is not discriminating. Accuracy differs (3/6 vs 1/6 vs
0/6) but n = 6.

The finding worth chasing: **greedy EIG scored worse than random.** That should
not happen if the information-gain calculation is sound. Either the predicted
latents are meaningless when memory is this sparse (~2,500 nodes), or greedy is
maximising discrimination among hypotheses that are all wrong. That needs
diagnosing before E6 is run at scale — a broken value function would make the
MCTS numbers meaningless too.

Also fixed during implementation: the rollout policy originally called full
greedy select inside every simulation, costing 49 ms per step and ~1.8 s per
MCTS decision. It now samples from the prior, as AlphaGo's fast rollout policy
did.

### E1, first valid run (2026-08-23)

Five seeds, three arms, 20k steps, on a build with no standing alarms.
Switch rate is instability of module identity; lower is better.

| arm | n | switch rate | mean err | basins |
|---|---|---|---|---|
| CHROMA (recurrence + basins) | 5 | 0.00010 +/- 0.00003 | 0.278 +/- 0.160 | 6 |
| flat landscape (recurrence only) | 5 | 0.00014 +/- 0.00004 | 0.214 +/- 0.125 | 1 |
| MLP (neither) | 5 | 0.00202 +/- 0.00052 | 0.175 +/- 0.029 | 1 |

**Slow integration: strongly supported.** Removing the slow dial entirely costs
**20.95x** on switch rate, against a bar of 5x set before the run. The
timescale separation is doing real work.

**Multistability: not supported.** The flat-landscape arm — recurrence and
`tau_g` intact, the Krotov-Hopfield basins flattened — differs by only
**1.44x**, well under the 5x bar. Paired by seed the direction is consistent
(4 of 4 non-tied seeds favour CHROMA) but a sign test gives p = 0.125, and the
effect is small either way.

So Section 4.1 splits. The claim that survives is "a slow timescale stabilises
regime identity." The claim that does not is "attractor multistability is what
provides that stability." The Hopfield energy can be flattened with almost no
cost, and the honest version of the architecture keeps `RegulatoryState` only
for its recurrence.

**A finding that cuts against the architecture:** CHROMA has the *worst* mean
prediction error (0.278) and by far the highest seed-to-seed variance (+/-0.160
against the MLP's +/-0.029). Two of five seeds came in near 0.44 while the rest
sat near 0.15. Being the most stable arm while predicting worst suggests the
stability is partly bought by under-adapting, and that trade is not one the
design ever argued for.

**Caveats on this run.** Memory hit its 20,000 cap in every arm, usually by step
10,000-12,500, so roughly half of each run had a frozen object memory. Two
transient dimensional-collapse warnings fired (chroma seed 2, MLP seed 2) and
both recovered by the next log window. Neither invalidates the comparison — all
three arms hit the cap equally — but capacity is now the obvious next thing to
fix.

### Second retraction: it was representational collapse (2026-08-13)

The full 15-run sweep (5 seeds x 3 arms, 20k steps, 12.5 hours) completed with
the `DEGENERATE` warning firing in almost every run. The graded-world fix did
not work, because the diagnosis was wrong.

Measured after 500 steps: **62 of 64 latent dimensions had standard deviation
below 0.05**. Raising the covariance weight from 0.04 to 1.0 changed nothing.

The cause is a design error in `losses.py`. It dropped VICReg's *variance* term,
arguing Proposition 4 closes off collapse structurally. Proposition 4 bounds
`||s_hat||` relative to `||s||` -- but `PatchEncoder` already renormalises to
fixed norm, so norm collapse was never reachable. What remains reachable is
*dimensional* collapse: every sample mapping to nearly the same point on the
sphere. Norm preserved, per-dimension variance gone, Proposition 4 silent.

With the variance hinge restored (`lam_var=1.0`), dimensions below 0.05 go from
62 to 0 and minimum per-dimension std from 0.008 to 0.34.

Sweep results, recorded but **not valid** — all three arms were being driven by
a signal that had collapsed to noise:

| arm | switch rate | mean err | basins |
|---|---|---|---|
| CHROMA | 0.00007 +/- 0.00001 | 0.0122 | 6 |
| flat landscape | 0.00014 +/- 0.00003 | 0.0095 | 1 |
| MLP | 0.00027 +/- 0.00018 | 0.0037 | 1 |

**Resolved (same day).** The metric was misleading. The alarm compared
`effective_rank` against `0.3 * D = 19.2`, a threshold unreachable in principle:
a 3x3 tactile patch is 9-dimensional with measured effective rank 4.64, so no
encoder can produce more than 9 independent latent directions however healthy it
is. The alarm fired on every step of every run and meant nothing.

The meaningful question is whether the latent preserves the *input* manifold, so
`CollapseMonitor` now tracks the input's own effective rank as the reference.
Measured after the variance fix: **latent 5.16 against input 4.64, ratio 1.11** —
the representation keeps everything the input had.

With `lam_var=1.0`, a 2000-step smoke run gives err 0.28-0.43 and disagreement
0.40, against 1e-5 and 0.0 before. Those are the two preconditions E1 needs, and
they are now met.

**Run `make smoke` before `make results`.** Two thousand steps, five minutes,
and it fails loudly rather than after twelve hours.

### A confound found mid-sweep (2026-08-12)

The first full-scale E1 run drove prediction error to ~1e-4 by step 10,000, with
vote disagreement at ~2e-4. Instrumenting the world explained why: the binary
3x3 tactile patch admits 512 readings and a run actually sees **26 of them**.
The encoder memorises a lookup table, and every error-driven mechanism in the
architecture is then fed a signal that is identically zero.

That invalidates the pilot conclusion as stated. The three E1 arms may have tied
because none of them was doing anything, which is a different finding from the
regulatory layer being useless.

Fix: `TactileWorld` now softens shapes to graded contact values, adds sensor
noise, and widens the pose range. Distinct observations go from 26 to 3,200. The
trainer prints a `DEGENERATE` warning whenever mean error falls below
`TrainConfig.err_floor`, so a future run says so rather than producing a
confident tie.

Not yet demonstrated: that error stays informative all the way to 20k steps.
That needs the long run, with the warning watched.

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
