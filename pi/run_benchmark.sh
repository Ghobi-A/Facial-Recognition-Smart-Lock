#!/usr/bin/env bash
# Reproduce the edge benchmark on a Raspberry Pi. Usage: bash pi/run_benchmark.sh [label]
set -euo pipefail
label="${1:-pi}"
cd "$(dirname "$0")/.."

edgeauth-study fetch
edgeauth-study manifest
edgeauth download-models --state .edgeauth-study
edgeauth-study benchmark --label "$label" --probes 200
edgeauth-study figures

if command -v vcgencmd >/dev/null; then
  echo "throttled: $(vcgencmd get_throttled)  (0x0 = no throttling during the run)"
  vcgencmd measure_temp
fi
