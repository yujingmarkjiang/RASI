# R5 manuscript-to-code alignment

The supplied manuscripts are preserved byte-for-byte in `docs/manuscript/`. This implementation targets the main Methods subsection **“Randomisation-anchored self-improvement (RASI)”** and Supplementary Note 16 **“The method and its safe-improvement guarantee”**, not the two-component illustrative demonstration or the earlier permissive ablation gate.

## Implemented mapping

| Manuscript requirement | Active implementation |
|---|---|
| The same real-only model at round 0 | `engine.RASI.run`: `FitRequest(real, contrasts, contrast_weight=0)`; immutable `base.json`. |
| Score on reserved recorded contrasts, not factual loss alone | `data.Problem` checks partitions; only `audit.AuditGate` receives audit records during proposal selection. |
| `N^-1 ||X beta-y||² + lambda M^-1 ||D beta-d||² + eta ||beta||²` | `models.linear.LinearWorldModel.fit` builds a separate active-minus-comparator matrix; `cpp/src/solver.cpp` evaluates the identical normalised normal equations. |
| Neural contrast objective | `models.gru.GRUWorldModel.fit` adds differentiable free-rollout active-minus-comparator loss; observations and contrasts have separate normalisers. |
| AdamW, cosine learning-rate schedule, clipping at 1, visible factual early stopping | GRU adapter; default patience 20. Memory microbatches accumulate the exact complete objective before an optimiser step. |
| Select lambda and k once on visible pairs | `selection.select`: unit-blocked CV excludes all held-unit subjects from the fold's fitting and retrieval data; `selection.json` freezes the decision. |
| Five independently trained members, using current data | `retrieval.allocate`: independent seeds and patient-cluster bootstraps of the current real-plus-accepted-synthetic data. |
| Allocate rollouts proportional to contrast disagreement | Time-averaged across-member contrast variance; a multinomial allocation preserves the exact budget and proportionality in expectation. Zero allocations are allowed; all-zero disagreement uses a uniform allocation. |
| Average the k nearest real patients in the model's representation | `retrieval.nearest_mean`: real-only donor index, current `encode`, distinct patients, matching treatment/stratum, observed scheduled times only, average of exactly k donors. |
| Three anchor standard errors | `retrieval.verify`: recorded/trial-derived anchor estimates and their standard errors, not the round-0 prediction and not factual residual SD. All failed proposals are discarded. |
| Replacement or accumulation | `engine.RASI.run`: replacement by default; accumulated data contain accepted synthetic batches only. |
| Candidate retraining | Linear refits solve exactly; GRU candidates fine-tune an independent copy of the incumbent. The five ensemble fits are independently initialised. |
| Same rotated subset for incumbent and candidate | `AuditGate.consider` draws after candidate fitting and evaluates both models on that draw. It never preselects a candidate on the entire pool. |
| Accept only beyond the improvement margin | `AuditGate`: main-text standard-error rule or explicit fixed epsilon; never the generic positive-deterioration tolerance. |
| Revert and resample after rejection | Incumbent parameters and accepted synthetic buffer are unchanged; a new ensemble/bootstrap/allocation is sampled next round. Neural proposal tensors do not share storage with the incumbent. |
| Ten consecutive rejected rounds | Enforced in `RASIConfig` and the controller; successful acceptance resets the counter. |
| Export a permitted update | `accepted.json` contains the last accepted checkpoint. No “lowest full-pool audit score” checkpoint search. |
| Parallel-trial evidence identifies stratum means | `Contrast(scope='stratum')` predicts mean(active) minus mean(comparator); parallel-trial provenance is rejected for individual labels. |
| QTcF, period correction and replicate floor | `preprocess.prepare_qt` implements these from real QT/RR records; the old contrast-only CSV cannot establish a factual loss or replicate-derived floor. |
| Agentic proposals share the same acceptance boundary | `ProposalContext` contains only training/proposal information; the returned candidate goes through the same gate. No agent experiment or remote agent service is implemented. |
| Audit resumption must not reset adaptive reuse | `resume.json` persists exact audit/RNG state and the rejection counter; data and configuration hashes must match on resumption. |

## Deliberate resolutions of under-specified or different manuscript rules

### Full RASI versus the demonstration and generic gate

The demonstration uses individual retrieved donors, sensitivity shifts and a non-increase gate on 8 of 12 pairs. The generic self-training subsection permits deterioration up to tau. Neither is the full four-component algorithm. The active default uses averaged k-neighbour trajectories and strict improvement beyond a standard-error margin. The old demonstration shortcuts remain only in the historical archive.

### The main-text standard-error rule versus the Note-16 margin

The main Methods says improvement must exceed the audit statistic's own standard error; it does not uniquely define whether to use the candidate error's SE or the paired improvement's SE. Here the **candidate observation-weighted contrast MAE's cluster-robust SE** is used. The paired improvement SE is logged separately, not silently substituted.

For audit units `u`, let `S_u` be the sum of absolute contrast errors and `n_u` their number of scheduled observations. The primary score is

```
C_hat = sum(S_u) / sum(n_u)
SE(C_hat) = sqrt[ U/(U-1) * sum((S_u - C_hat*n_u)^2) ] / sum(n_u).
```

Thus the score retains the main Methods' observation weighting, but repeated times and shared patient comparators do not become fake independent sample units. With equal-sized units this reduces to the usual standard error of unit means. In standard-error mode the acceptance criterion is

```
C_hat(incumbent) - C_hat(candidate) > max(epsilon, SE(candidate)).
```

In fixed-margin mode it is the supplementary algorithm's literal inequality:

```
C_hat(candidate) <= C_hat(incumbent) - epsilon.
```

The Note-16 statement is **not** claimed for the empirical SE rule. Its reported positive step is

```
epsilon - 2*sigma_c*sqrt(log(2*R/delta)/n_a).
```

`require_certificate=true` fails configuration validation unless this is positive, `sigma_c` is externally supplied, independent units are explicitly asserted, the audit is subject-disjoint and equally weighted, and audit draws do not reuse units. The bound remains conditional on the supplied assumptions; code cannot certify randomisation, transportability or a sub-Gaussian law from a label. On acceptance these per-step bounds telescope; on rejection the deployed model is unchanged.

### Rotation and the finite reserved pool

Empirical rotated mode can reuse units across rounds but never within a draw, and records each unit's use count. Its output explicitly does not claim Note 16. With `allow_reuse=false`, exhaustion ends the run rather than silently reducing the audit size or recycling pairs. `fixed` and `held_out` are supported audit policies from Note 14, but are labelled alternatives to the main rotated loop. A held-out policy does not deploy any intermediate candidate: it applies one final gate against the original baseline.

### Patient effects versus population anchors

A published average treatment effect is not treated as a patient-specific true effect. Every anchor names its population and scope. A stratum anchor is checked against the candidate-updated mean in that population, without rollout-budget weighting. All proposals contributing to a failed stratum check are discarded together; overlapping anchors are rechecked after removals. Individual anchors check one named patient. Uncovered proposals are rejected.

For the supplied crossover importer, population means and SEs are calculated from **training contrasts only**, never audit pairs. The target-population transport assumption is recorded explicitly. A published trial can instead be supplied in `anchors` with its estimate, SE, trial-eligible population and source.

### Deterministic retrieved means and integer budgets

The full algorithm asks for averages of k real neighbours. It does not specify adding Gaussian noise to these means. Accordingly, repeated allocations to an arm are stored as a multiplicity/weight on that mean. The synthetic budget still affects relative training weights. Fresh bootstrap ensembles and multinomial allocations supply proposal variation on later attempts; no synthetic-noise or closest-failed-candidate rescue is used.

If every candidate is rejected by retrieval/verification, the implementation rejects the proposal round without fitting a synthetic-free substitute. This is an explicit conservative empty-batch policy. It counts toward the ten-rejection stop rule and never manufactures training targets.

### General models and available data

`outcome` mode implements a genuine factual objective plus a distinct paired contrast objective. Training contrast values must equal the recorded active-minus-comparator values; external published estimates belong in `anchors`, not fabricated factual records.

`contrast` mode is a separate coordinate representation for the supplied CSV: its active target is already a placebo-corrected contrast and its reference is algebraic zero. The “factual” term then measures fitted contrast targets and is not factual QT reconstruction. The output labels this limitation. No missing placebo waveform, hidden outcome, measurement floor or clinical result is invented.

The generic model contract is endpoint/trajectory-based. High-dimensional imaging or language world models require a real adapter that maps their outputs to the declared endpoint and implements training on the corresponding real/synthetic data. A named proxy model is not an implementation of a released backbone.

## Explicitly outside this delivery

The active package now includes engineering unit/smoke/hardware checks, but no manuscript experiment runner, artificial hidden-arm masks, E0 simulations, E1–E12 studies, 28-model benchmark, scaling sweep, agent-service experiment, federated-site experiment, ablation grid, figures, or research result files are implemented. Dataset names, anchor values, model strings and eligibility rules left as TBD in the manuscripts are not guessed. The existing v1 files are preserved for provenance but are not part of the active implementation.
