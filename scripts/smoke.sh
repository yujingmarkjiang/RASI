#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$root/python${PYTHONPATH:+:$PYTHONPATH}"
exec "${RASI_PYTHON:-python3}" -m rasi smoke-test "$@"
