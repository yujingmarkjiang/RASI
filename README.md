# RASI — manuscript-aligned implementation

RASI implements the manuscript's randomisation-anchored self-improvement loop: it proposes synthetic trajectories from real-neighbour retrieval, verifies them against randomised anchors, retrains a world model, and accepts updates only when a reserved contrast audit improves. The repository provides linear and autoregressive GRU adapters, explicit CPU/CUDA execution, reproducible configuration, checkpoint/resume support, diagnostics, and engineering tests without adding manuscript experiments.

## 1. Quick start

### Linear CPU

```bash
python -m pip install -e .
rasi doctor --config configs/linear_cpu.json --strict
rasi smoke-test --config configs/linear_cpu.json

rasi prepare \
  --cohort data/e1_qt_cohort.csv \
  --output prepared.json

rasi train \
  --data prepared.json \
  --config configs/linear_cpu.json \
  --output runs/linear

rasi infer \
  --checkpoint runs/linear/accepted.json \
  --data prepared.json
```

### GRU with CUDA

```bash
python -m pip install -e '.[neural]'
rasi doctor --config configs/gru_cuda.json --strict
rasi test-gpu --device cuda:0
rasi smoke-test --config configs/gru_cuda.json --epochs 3

rasi train \
  --data prepared-outcomes.json \
  --config configs/gru_cuda.json \
  --output runs/gru
```

RASI does not silently emulate a requested GPU on CPU. If `cuda:0` is requested and PyTorch cannot use it, the command fails with a diagnostic message.

---

## 2. Repository layout

```text
rasi/
├── python/rasi/             # active Python package and shared RASI controller
│   ├── engine.py            # proposal / verification / audit / rollback loop
│   ├── audit.py             # reserved-audit logic
│   ├── retrieval.py         # ensemble disagreement, allocation, kNN retrieval
│   ├── selection.py         # visible-data lambda/k selection
│   ├── preprocess.py        # CSV/raw QT-RR preparation
│   ├── storage.py           # atomic checkpoints and resume state
│   ├── diagnostics.py       # installation, CUDA and model smoke checks
│   └── models/              # linear and GRU adapters
├── configs/                 # validated starting configurations
├── tests/                   # lightweight engineering tests
├── data/                    # supplied demonstration cohort
├── cpp/                     # optional native linear solver + CLI launcher
├── go/                      # optional process launcher
├── docs/                    # data, adapter, config, testing and alignment docs
├── scripts/                 # convenience shell entry points
└── legacy/v1/               # preserved earlier implementation; not active
```

`legacy/v1/` is retained for provenance only. The active package does not import it.

---

## 3. Requirements and installation

### Minimum CPU installation

Requirements:

- Python 3.10+
- NumPy 1.24+

Install in editable mode:

```bash
python -m pip install -e .
```

Verify the installation:

```bash
rasi doctor
rasi smoke-test --config configs/linear_cpu.json
```

### GRU / PyTorch installation

The GRU adapter requires PyTorch:

```bash
python -m pip install -e '.[neural]'
```

The package does not pin a particular CUDA wheel because the correct PyTorch build depends on the host's driver/runtime environment. After installing the appropriate PyTorch build, check it with:

```bash
rasi doctor --config configs/gru_cuda.json --strict
rasi test-gpu --device cuda:0
```

### Optional C++ and Go components

The NumPy backend is sufficient for the linear model. To build the native C++ solver and launchers:

```bash
make
```

Or build only one component:

```bash
make native
make coordinator
```

The native linear configuration is:

```bash
configs/linear_native.json
```

If the shared library is installed elsewhere, set:

```bash
export RASI_NATIVE_LIB=/absolute/path/to/librasi.so
```

---

## 4. Environment and hardware diagnostics

### General diagnostics

```bash
rasi doctor
```

This reports:

- RASI version;
- Python executable/version;
- operating system and architecture;
- NumPy version;
- PyTorch version, CUDA build, cuDNN status and visible GPUs;
- optional native C++ library status;
- available C++ compiler and Go toolchain.

Check whether a particular config can run on the machine:

```bash
rasi doctor --config configs/gru_cuda.json --strict
```

`--strict` is useful in CI and cluster batch scripts because an incompatible selected backend/device returns a nonzero exit code.

### Actual CUDA compute test

```bash
rasi test-gpu --device cuda:0
```

This is not merely an availability query. It allocates CUDA tensors, performs matrix multiplication and a backward pass, synchronizes the device, checks finite outputs and gradients, and reports device/memory/build information.

Useful options:

```bash
rasi test-gpu --device cuda:1 --matrix-size 1024 --repeats 3
rasi test-gpu --allow-missing      # for CPU-only CI where CUDA is optional
```

A successful GPU check means the basic CUDA/PyTorch execution path works; it is not a performance benchmark or a manuscript experiment.

---

## 5. Smoke tests and engineering tests

### Model smoke tests

Linear NumPy:

```bash
rasi smoke-test --config configs/linear_cpu.json
```

Native C++ solver:

```bash
make native
rasi smoke-test --config configs/linear_native.json
```

GRU CPU:

```bash
rasi smoke-test --config configs/gru_cpu.json --epochs 3
```

GRU CUDA:

```bash
rasi smoke-test --config configs/gru_cuda.json --epochs 3
```

The smoke test creates a tiny synthetic factual/contrast dataset, fits the requested adapter, checks finite predictions, serializes/reloads the model state, and verifies that predictions survive the checkpoint round-trip. `--epochs` only caps GRU epochs for this engineering check; it does not change the actual training config file.

### Standard-library test suite

No pytest dependency is required:

```bash
python -m unittest discover -s tests -v
```

CUDA testing is opt-in so ordinary CPU machines do not fail merely because no GPU exists:

```bash
RASI_TEST_CUDA=1 RASI_TEST_DEVICE=cuda:0 \
  python -m unittest tests.test_diagnostics.DiagnosticTests.test_cuda -v
```

### Make targets

```bash
make doctor
make test
make smoke
make smoke-gru
make test-gpu
make smoke-gru-gpu
make check
```

See [`docs/TESTING.md`](docs/TESTING.md) for the complete preflight workflow.

---

## 6. Configuration

RASI uses strict JSON configuration. Unknown keys are rejected rather than ignored.

Included starting points:

| Config | Use |
|---|---|
| `configs/linear_cpu.json` | Recommended default; NumPy linear solver on CPU. |
| `configs/linear_native.json` | Linear model with the optional C++ solver. |
| `configs/gru_cpu.json` | Autoregressive GRU on CPU. |
| `configs/gru_cuda.json` | Autoregressive GRU on `cuda:0`. |
| `configs/fixed_margin.json` | Fixed audit-improvement margin instead of the standard-error margin. |
| `configs/rasi.json` | Compatibility alias of the linear CPU starting config. |
| `configs/gru.json` | Compatibility alias of the GRU CPU starting config. |

### Generate a fresh config

```bash
rasi init-config --preset linear-cpu --output my-rasi.json
rasi init-config --preset gru-cuda --output my-gru.json
```

The command refuses to overwrite an existing file.

### Validate and normalize a config

```bash
rasi show-config --config my-rasi.json
rasi doctor --config my-rasi.json --strict
```

### Important configuration semantics

`selection.lambda_grid` and `selection.k_grid` are visible-training-data hyperparameter choices. They are selected before round 0 and frozen; they are not manuscript experiment grids.

`rejection_patience=10` and `verifier_z=3.0` are intentionally enforced by code to keep the active implementation aligned with the manuscript. The linear adapter is CPU-only; setting a CUDA device for a linear model is rejected.

For field-by-field documentation, see [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md).

---

## 7. Data preparation

RASI accepts three input routes.

### A. Prepared RASI JSON

If a dataset has already been partitioned into training, validation, targets, anchors and audit data:

```bash
rasi validate --data prepared.json
```

Then train directly with `--data prepared.json`.

The JSON schema and estimand rules are documented in [`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md).

### B. Supplied precomputed-contrast cohort

The included CSV contains precomputed contrasts rather than complete factual active/placebo outcome trajectories. Prepare it with:

```bash
rasi prepare \
  --cohort data/e1_qt_cohort.csv \
  --audit-subjects 12 \
  --validation-subjects 2 \
  --seed 20261001 \
  --output prepared.json
```

This produces an explicitly labelled contrast-mode dataset. It does not invent missing factual outcomes and therefore does not claim factual-outcome validation that the source table cannot support.

Optional genuinely unobserved targets can be supplied via an input-only target CSV:

```bash
rasi prepare \
  --cohort data/e1_qt_cohort.csv \
  --targets targets.csv \
  --output prepared.json
```

The target file must not contain hidden outcome labels.

### C. Raw replicate QT/RR data

For factual trajectory training, prepare raw replicate measurements:

```bash
rasi prepare \
  --raw-qt recordings.csv \
  --placebo Placebo \
  --audit-subjects 12 \
  --validation-subjects 2 \
  --output prepared-outcomes.json
```

Required columns are:

```text
participant,period,drug,time,QT,RR
```

QT and RR are expected in milliseconds. The importer computes QTcF per replicate, period-centres it against the pre-dose mean, constructs recorded active-minus-placebo contrasts, and propagates replicate measurement variance where available. Invalid measurements are excluded and missing outcomes are not replaced with zeros.

### Validate before training

Always inspect the prepared partition:

```bash
rasi validate --data prepared-outcomes.json
```

The output reports the representation type, training/validation sizes, number of targets, reserved audit contrasts, independent audit units and anchors.

---

## 8. Training

### Linear CPU

```bash
rasi train \
  --data prepared.json \
  --config configs/linear_cpu.json \
  --output runs/linear
```

### Native linear solver

```bash
make native
rasi train \
  --data prepared.json \
  --config configs/linear_native.json \
  --output runs/linear-native
```

### GRU CPU

```bash
rasi train \
  --data prepared-outcomes.json \
  --config configs/gru_cpu.json \
  --output runs/gru-cpu
```

### GRU CUDA

Run the GPU preflight first:

```bash
rasi doctor --config configs/gru_cuda.json --strict
rasi test-gpu --device cuda:0
rasi smoke-test --config configs/gru_cuda.json --epochs 3
```

Then train:

```bash
rasi train \
  --data prepared-outcomes.json \
  --config configs/gru_cuda.json \
  --output runs/gru-gpu
```

### Useful command-line overrides

The following are convenient for controlled operational changes without rewriting JSON:

```bash
rasi train --data prepared.json --config configs/linear_cpu.json \
  --output runs/tmp --rounds 5 --m 3 --seed 123
```

Model/backend/device overrides are also accepted:

```bash
rasi train --data prepared-outcomes.json --config configs/gru_cpu.json \
  --output runs/gru-gpu --device cuda:0
```

Use overrides for execution choices, not for post-hoc searching on the reserved audit set.

### Reuse a frozen selection

After selecting lambda/k on one visible training dataset, reuse them without retuning:

```bash
rasi train \
  --data another-prepared.json \
  --config configs/linear_cpu.json \
  --selection runs/linear/selection.json \
  --output runs/another
```

---

## 9. Resume behavior

Interrupted runs can be resumed:

```bash
rasi train \
  --data prepared.json \
  --config configs/linear_cpu.json \
  --output runs/linear \
  --resume
```

Resume requires the same dataset, configuration and audit pool. It restores the accepted synthetic data, rejection counter, audit state and random-number state rather than starting a new audit search.

Only one writer may use a run directory. A `.writer.lock` prevents concurrent writers. Remove a stale lock only after confirming that the previous process is no longer running.

---

## 10. Run outputs

A normal run directory contains:

| File | Meaning |
|---|---|
| `base.json` | Immutable real-only round-zero model. |
| `accepted.json` | Last accepted model and deployment metadata. |
| `selection.json` | Frozen lambda/k and visible-only selection provenance. |
| `events.jsonl` | Aggregate proposal, verifier, audit and rollback events. |
| `resume.json` | Private transactional state required for exact resume. |
| `summary.json` | Run status and visible-data/deployment summary. |

Generated state files are written with owner-only file permissions. Treat the whole run directory as sensitive research data; in particular, do not distribute `resume.json`, which can contain retrieved synthetic trajectories and provenance.

---

## 11. Inference

Use the last accepted checkpoint:

```bash
rasi infer \
  --checkpoint runs/linear/accepted.json \
  --data prepared.json
```

Write results to a file:

```bash
rasi infer \
  --checkpoint runs/linear/accepted.json \
  --data prepared.json \
  --output predictions.json
```

Or provide an explicit JSON list of query/target records:

```bash
rasi infer \
  --checkpoint runs/linear/accepted.json \
  --queries queries.json
```

For a GRU checkpoint, a device override is allowed at load time:

```bash
rasi infer \
  --checkpoint runs/gru/accepted.json \
  --queries queries.json \
  --device cuda:0
```

The inference command never retrains.

---

## 12. CPU/GPU behavior

- **Linear + NumPy:** CPU only.
- **Linear + native C++:** CPU only.
- **GRU + `device=cpu`:** PyTorch CPU.
- **GRU + `device=cuda:N`:** real PyTorch CUDA device; unavailable devices raise an error.
- The RASI statistical controller itself is not presented as a multi-GPU distributed system.
- The current implementation does not claim automatic data parallelism or distributed training.

For multi-GPU extension, integrate a model adapter that handles distributed execution internally while preserving the shared RASI adapter contract; see [`docs/ADAPTERS.md`](docs/ADAPTERS.md).

---

## 13. Audit modes and statistical boundaries

The default audit rule requires the candidate's observation-weighted contrast MAE to improve beyond the configured standard-error margin on the same audit subset used for both incumbent and candidate.

A fixed predeclared epsilon is available through `audit.margin="fixed"`. The stronger conditional statement described in Supplementary Note 16 additionally requires externally justified `sigma_c`, independent subject-disjoint audit units, no forbidden reuse, and a positive numerical bound. The code can enforce declared structural conditions, but it cannot scientifically justify independence or `sigma_c` for the user.

Insufficient audit capacity is reported rather than silently recycling data outside the configured policy.

---

## 14. Common problems

### `A CUDA device was requested but PyTorch cannot use CUDA`

Run:

```bash
rasi doctor --config configs/gru_cuda.json
rasi test-gpu --device cuda:0
```

Usually this means a CPU-only PyTorch build, an incompatible NVIDIA driver/runtime, an unavailable device index, or a scheduler/container that did not expose the GPU.

### `Native backend missing`

Build the library:

```bash
make native
```

Then check:

```bash
rasi doctor --config configs/linear_native.json --strict
rasi smoke-test --config configs/linear_native.json
```

If the library is outside the repository, set `RASI_NATIVE_LIB`.

### GRU says visible validation data are required

This is deliberate. The GRU's early stopping must use visible factual validation trajectories; the reserved audit data cannot be substituted for model-selection/early-stopping data.

### Selection fails because retrieval k lacks donor support

The visible-data cross-validation requires comparable donor support across candidate `k` values. Reduce the declared `k_grid` based on scientifically valid donor availability, provide more real donors, or reuse a previously frozen justified selection. Do not use audit outcomes to choose `k`.

### Run directory is locked

Do not start multiple training writers in one directory. Confirm the previous process exited before deleting a stale `.writer.lock`.

---

## 15. Convenience scripts and launchers

Without installing the Python package, the shell scripts set `PYTHONPATH` for you:

```bash
scripts/train.sh --data prepared.json --config configs/linear_cpu.json --output runs/linear
scripts/infer.sh --checkpoint runs/linear/accepted.json --data prepared.json
scripts/check.sh --config configs/linear_cpu.json --strict
scripts/test_gpu.sh --device cuda:0
scripts/smoke.sh --config configs/linear_cpu.json
```

The C++ and Go executables are thin launchers around the same Python controller; they do not contain separate scientific implementations. `RASI_PYTHON` selects the Python interpreter for the C++ launcher, and `RASI_ROOT` can point it to the repository root.
