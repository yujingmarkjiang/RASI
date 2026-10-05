.PHONY: all native coordinator doctor check test test-gpu smoke smoke-gru smoke-gru-gpu clean

PYTHON ?= python
PYTHONPATH := python$(if $(PYTHONPATH),:$(PYTHONPATH))

all: native coordinator

native:
	$(MAKE) -C cpp

coordinator:
	mkdir -p go/build
	cd go && go build -o build/rasi-coord ./cmd/rasi-coord

doctor:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m rasi doctor

check: doctor test smoke

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest discover -s tests -v

test-gpu:
	PYTHONPATH=$(PYTHONPATH) RASI_TEST_CUDA=1 $(PYTHON) -m unittest tests.test_diagnostics.DiagnosticTests.test_cuda -v

smoke:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m rasi smoke-test --config configs/linear_cpu.json

smoke-gru:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m rasi smoke-test --config configs/gru_cpu.json --epochs 3

smoke-gru-gpu:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m rasi test-gpu --device cuda:0 --matrix-size 256 --repeats 1
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m rasi smoke-test --config configs/gru_cuda.json --epochs 3

clean:
	$(MAKE) -C cpp clean
	rm -rf go/build .coverage .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
