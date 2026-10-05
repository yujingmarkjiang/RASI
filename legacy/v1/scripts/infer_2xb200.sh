#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
export RASI_BIN="${RASI_BIN:-$ROOT/cpp/build/rasi}"
COORD="$ROOT/go/build/rasi-coord"
if [ -x "$COORD" ]; then
  exec "$COORD" infer --config "$ROOT/configs/infer_2xb200.json" "$@"
fi
exec go run "$ROOT/go/cmd/rasi-coord" infer --config "$ROOT/configs/infer_2xb200.json" "$@"
