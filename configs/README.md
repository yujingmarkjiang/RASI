# Configuration presets

These files are starting points, not tuned experiment settings. `linear_cpu.json` is the safest default; `linear_native.json` uses the optional C++ solver; `gru_cpu.json` uses PyTorch on CPU; `gru_cuda.json` requests a real CUDA device and fails rather than falling back; `fixed_margin.json` switches the audit to a predeclared fixed improvement margin.

Create a fresh copy with the CLI instead of editing a packaged preset in-place:

```bash
rasi init-config --preset linear-cpu --output my-rasi.json
rasi show-config --config my-rasi.json
rasi doctor --config my-rasi.json --strict
```

For the conditional Supplementary Note 16 certificate, do not copy arbitrary `sigma_c` or `epsilon` values from an example. Set `audit.require_certificate=true` only after externally justifying `sigma_c`, using subject-disjoint fresh audit units, disabling reuse, selecting `margin="fixed"`, and choosing an epsilon that passes the implementation's bound check.
