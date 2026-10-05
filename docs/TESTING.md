# Diagnostics and testing

The active package includes lightweight checks for installation, CPU/native execution, checkpoint round-tripping, and CUDA execution. These checks are engineering validation only: they do not reproduce manuscript experiments and never consume reserved audit outcomes.

## 1. Installation diagnostics

```bash
rasi doctor
rasi doctor --config configs/linear_cpu.json --strict
rasi doctor --config configs/gru_cuda.json --strict
```

`doctor` reports the Python and NumPy versions, PyTorch/CUDA status, visible CUDA devices, cuDNN, the C++ compiler, Go, and whether the optional native linear library can be loaded. With `--config`, it also checks whether the selected backend/device is runnable on the current machine. `--strict` returns a nonzero exit status when the config is incompatible, which is useful in CI or batch-job prologues.

## 2. Model smoke tests

```bash
rasi smoke-test --config configs/linear_cpu.json
rasi smoke-test --config configs/linear_native.json
rasi smoke-test --config configs/gru_cpu.json --epochs 3
rasi smoke-test --config configs/gru_cuda.json --epochs 3
```

The smoke test constructs a tiny synthetic factual/contrast dataset, fits the selected adapter, checks for finite predictions, serializes and reloads the model state, and confirms that predictions survive the round-trip. For GRU it caps the number of epochs so the check remains short; this is not a training recommendation.

## 3. Focused GPU test

```bash
rasi test-gpu --device cuda:0
rasi test-gpu --device cuda:1 --matrix-size 1024 --repeats 3
```

The GPU test performs a real CUDA matrix multiplication plus backward pass, synchronizes the device, checks finite outputs/gradients, and reports the visible device, compute capability, CUDA/cuDNN build information, timings, and memory. It never silently switches to CPU. CPU-only CI can use `--allow-missing` when CUDA is optional.

## 4. Standard-library test suite

No third-party test runner is required:

```bash
python -m unittest discover -s tests -v
```

The CUDA unit check is intentionally opt-in:

```bash
RASI_TEST_CUDA=1 RASI_TEST_DEVICE=cuda:0 \
  python -m unittest tests.test_diagnostics.DiagnosticTests.test_cuda -v
```

## 5. Make targets

```bash
make doctor          # environment report
make test            # CPU-safe unit checks
make smoke           # linear NumPy smoke test
make smoke-gru       # short GRU CPU smoke test
make test-gpu        # require CUDA
make smoke-gru-gpu   # CUDA compute check + short GRU GPU fit
make check           # doctor + CPU tests + linear smoke
```

## Suggested preflight sequence

For a CPU linear run:

```bash
rasi doctor --config configs/linear_cpu.json --strict
rasi smoke-test --config configs/linear_cpu.json
rasi validate --data prepared.json
```

For a GPU GRU run:

```bash
rasi doctor --config configs/gru_cuda.json --strict
rasi test-gpu --device cuda:0
rasi smoke-test --config configs/gru_cuda.json --epochs 3
rasi validate --data prepared-outcomes.json
```

A passing hardware or smoke test only establishes that the requested execution path works. It does not establish statistical validity, clinical validity, convergence on a real cohort, or reproduction of any manuscript result.
