#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
export RASI_BIN="${RASI_BIN:-$ROOT/cpp/build/rasi}"
COORD="$ROOT/go/build/rasi-coord"
if [ -x "$COORD" ]; then
  exec "$COORD" train --config "$ROOT/configs/train_32xb200.json" "$@"
fi
exec go run "$ROOT/go/cmd/rasi-coord" train --config "$ROOT/configs/train_32xb200.json" "$@"
