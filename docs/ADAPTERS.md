# Integrating a real world model or proposal operator

The controller is model-independent at its endpoint-trajectory boundary. The built-in adapters are a real linear model and an autoregressive GRU. Named regressors pretending to be transformers, convolutional networks, state-space models or imaging backbones are not part of the active implementation.

## World-model protocol

Implement the `WorldModel` protocol in `rasi.models.base`:

```python
class WorldModel(Protocol):
    def fork(self, seed: int) -> WorldModel: ...
    def fit(self, request: FitRequest) -> None: ...
    def predict(self, query: Query) -> np.ndarray: ...
    def encode(self, query: Query) -> np.ndarray: ...
    def state_dict(self) -> dict[str, Any]: ...
```

`predict` must return a finite vector of length `len(query.times)` on the declared outcome scale. Predict both interventions without reading their observed outcomes; a neural contrast term must retain gradients through both predictions during fitting. In contrast-coordinate mode, the output itself is a corrected contrast.

`encode` returns a finite, fixed-width one-dimensional representation used for patient retrieval. It must be determined by current model parameters and query inputs only, not the donor's future outcome or an audit label. The built-in linear adapter uses its current multi-treatment response profile plus standardized context; the GRU uses its current hidden representation. Retrieval averages recorded donor trajectories, not these representations or generated model outcomes.

`fork` creates an independent trainable copy with a new proposal seed. No parameters, optimizer buffers or mutable state may alias the incumbent. GRU proposals warm-start from this copy; linear fitting solves its convex objective anew. A factory `factory(seed)` instead returns a fresh independent model for round zero and each uncertainty-ensemble member.

`fit` receives only:

| FitRequest field | Meaning |
|---|---|
| `observations` | Real plus currently permitted synthetic trajectories. |
| `contrasts` | Visible recorded contrast constraints; never audit contrasts. |
| `contrast_weight` | Frozen lambda; zero only for real-only round zero. |
| `validation` | Separate visible early-stopping trajectories. |
| `bootstrap` | Optional per-patient ensemble sampling multiplicities. |

Use `request.observation_weight(row)` and `request.contrast_sample_weight(pair)` for effective bootstrap weights. Normalize factual squared error by the weighted count of factual scheduled observations, and contrast squared error separately by the weighted count of contrast observations. Do not implement lambda by simply multiplying every factual sample weight. Preserve any documented model regularizer; the linear adapter uses `ridge * ||beta||²`, while the GRU uses AdamW's declared weight decay.

The GRU runs autoregressively from its observed/learned initial state. No teacher forcing may expose a hidden target trajectory during contrast prediction. The standard training controls are AdamW, cosine learning-rate decay, gradient-norm clipping at 1 and early stopping on visible validation error with patience 20. The neural implementation accumulates memory microbatches before an optimizer step, so minibatch normalization does not inadvertently rescale the two objective terms.

## Supplying an adapter

Construct `RASI(problem, config, selection=frozen_selection, factory=factory)` and call `.run(output_directory)`. Supply a `Selection` containing positive contrast weight, retrieval k, selection source, visible training-data digest and recorded CV scores. External adapters must perform their own visible-only CV once or reuse previously frozen settings; the controller does not tune a linear surrogate on their behalf.

The same controller then uses your model for five-member contrast disagreement, real-neighbour representations, anchor checks, candidate fitting, reserved-audit comparison and rollback. An adapter must support every proposed and audit treatment/time/context, including unfamiliar patient IDs without fitting new embeddings to audit labels.

Use strict JSON-compatible `state_dict()` values: no NaN/infinity, arbitrary executable pickles or dynamically imported checkpoint class names. The controller can write an external model state, but the bundled CLI loader and automatic resume intentionally whitelist **only linear and GRU states**. An external integration must provide its own trusted checkpoint loader and resume integration. It must not label an unsupported model state as a built-in model merely to bypass the whitelist.

## Agent, aggregation or architecture proposals

For a custom update operator, pass `proposer` to `RASI`:

```python
def proposer(context: ProposalContext, incumbent: WorldModel) -> WorldModel:
    # incumbent is an isolated copy, never the live deployed object.
    candidate = incumbent.fork(context.seed)
    candidate.fit(FitRequest(
        context.observations,
        context.training.contrasts,
        context.selection.contrast_weight,
        context.training.validation,
    ))
    return candidate
```

The callback receives the restricted `TrainingView`, the verified current training observations, frozen selection, round number and proposal seed. The controller retains audit ownership, draws its subset after the callback returns and evaluates incumbent/candidate on that same subset. No custom update may deploy directly: it must return its candidate to this boundary. The callback can implement a genuine agentic architecture change or a federated aggregation proposal, but the repository does not implement an agent service, federated transport protocol or either experimental comparison.

## Trust and isolation

This is an **in-process trusted-code interface**, not an operating-system security sandbox. A malicious plugin could read the filesystem, close over the original `Problem`, use external data or mutate process globals. API separation and independent RNGs prevent accidental ordinary leakage; they do not prevent adversarial code from accessing an audit file. Run untrusted proposals in a separate process/container with training-only mounted data, disallow audit/network access, serialize the candidate, and perform audit evaluation in a separate trusted process. Such deployment isolation is outside this codebase.

Query contexts, anchors and externally supplied selections also remain caller-governed inputs. A provenance label cannot prove absence of carry-over, trial transportability, independent audit units or a sub-Gaussian error law. Do not represent an empirical audit acceptance as unconditional clinical or causal safety.
