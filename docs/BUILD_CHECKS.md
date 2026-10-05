# Build and execution checks for this delivery

These are engineering checks, not manuscript experiments or scientific validation.

## Environment used

- Python 3.13.5; NumPy 2.3.5.
- PyTorch 2.10.0+cpu. This host exposed no CUDA device, so the new CUDA test path was verified for correct missing-CUDA reporting but could not be executed on real GPU hardware here.
- GNU C++ 14.2.0; Go 1.23.2; Linux amd64.

The package declares Python 3.10+, NumPy 1.24–2.x and optional PyTorch 2.2–2.x. Lower-version combinations were not individually exercised in this environment.

## Checks completed

- Compiled all active Python modules and the new test modules.
- Ran the standard-library suite: configuration invariants, data-safety invariants, environment diagnostics and linear model/checkpoint smoke test passed; the CUDA unit test skipped because CUDA is opt-in and unavailable on this host.
- Ran a short GRU CPU smoke fit and verified finite prediction plus state reload equivalence.
- Built the C++ launcher/shared solver and passed the native linear smoke test.
- Built the Go supervisor.
- Exercised `rasi doctor`, `rasi init-config`, `rasi show-config`, and CPU-safe `rasi test-gpu --allow-missing` behavior.
- Rechecked cohort preparation and validation after extending the CLI.

## CUDA limitation of this delivery check

`rasi test-gpu` is implemented to allocate tensors on the requested CUDA device, perform matrix multiplication and backward propagation, synchronize, check finite outputs/gradients, and report device/memory/build information. The execution environment used for packaging had a CPU-only PyTorch build, so a real CUDA pass could not be established here. Users should run `rasi doctor --config configs/gru_cuda.json --strict`, `rasi test-gpu --device cuda:0`, and `rasi smoke-test --config configs/gru_cuda.json --epochs 3` on the intended GPU host before a real run.

## Not established

These checks do not establish reproduction of manuscript results, model accuracy on external cohorts, clinical validity, raw ECG importer validity on a real replicate dataset, manuscript-scale GRU convergence, or GPU performance. The engineering test suite intentionally avoids consuming reserved audit outcomes or implementing manuscript experiments.
