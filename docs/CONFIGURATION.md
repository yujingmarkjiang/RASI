# Configuration reference

RASI configuration is strict JSON. Unknown keys are rejected, device selection is explicit, and manuscript-fixed constants such as the three-standard-error verifier and ten-rejection stopping rule cannot be changed through configuration.

## Creating and validating a config

```bash
rasi init-config --preset linear-cpu --output my-rasi.json
rasi show-config --config my-rasi.json
rasi doctor --config my-rasi.json --strict
```

Available presets are `linear-cpu`, `linear-native`, `gru-cpu`, and `gru-cuda`. The files in `configs/` are equivalent editable examples.

## Top-level fields

| Field | Meaning |
|---|---|
| `rounds` | Maximum number of RASI proposal rounds. |
| `rollouts_per_arm` | Mean rollout budget per unobserved target arm; disagreement allocation preserves the exact total budget. |
| `seed` | Master deterministic seed used by the controller. |
| `regime` | `replacement` keeps only the latest accepted synthetic proposal set; `accumulation` retains accepted synthetic data across rounds. |
| `synthetic_weight` | Base weight applied to verified retrieved trajectories; allocation multiplicity is applied separately. |
| `rejection_patience` | Fixed to 10 by the manuscript-aligned implementation. |
| `verifier_z` | Fixed to 3.0 for the three-anchor-standard-error verifier. |

## `model`

| Field | Meaning |
|---|---|
| `kind` | `linear` or `gru`. |
| `backend` | Linear only: `numpy` or optional `native`. |
| `device` | `cpu` or an explicit PyTorch device such as `cuda:0`; linear models are CPU-only. |
| `ridge` | Linear ridge penalty. |
| `hidden_size` | GRU hidden width. |
| `embedding_size` | GRU subject/treatment embedding width. |
| `epochs` | Maximum GRU training epochs per fit. |
| `batch_size` | GRU microbatch size. |
| `learning_rate` | AdamW learning rate. |
| `weight_decay` | AdamW weight decay. |
| `early_stopping_patience` | Visible-validation early-stopping patience. |

The GRU requires visible factual validation trajectories and never substitutes the reserved audit set for early stopping.

## `selection`

`lambda_grid` and `k_grid` are **training-only hyperparameter candidates**, selected using visible data before round 0 and then frozen. `folds` is the number of visible independent-unit folds. These are not experiment sweep controls.

## `audit`

| Field | Meaning |
|---|---|
| `size` | Number of independent audit units per audit evaluation. |
| `mode` | `rotated`, `fixed`, or `held_out`. |
| `allow_reuse` | Whether audit units may be reused across rounds when the selected mode permits it. |
| `margin` | `standard_error` for the main-method acceptance rule or `fixed` for a predeclared epsilon. |
| `epsilon` | Fixed improvement margin when `margin="fixed"`. |
| `delta` | Failure probability used only for the optional conditional bound. |
| `sigma_c` | Externally justified contrast bound/scale for the conditional certificate; never fitted from audit residuals. |
| `require_certificate` | Enforce the stronger Supplementary Note 16 preconditions. |
| `independent_units_asserted` | Explicit assertion that the audit units satisfy the required independence assumption. |

Do not turn on `require_certificate` by copying arbitrary values. The implementation checks structural conditions and the numerical epsilon penalty, but scientific justification for `sigma_c` and independence remains the user's responsibility.
