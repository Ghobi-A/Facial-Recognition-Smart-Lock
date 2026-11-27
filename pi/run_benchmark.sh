#!/usr/bin/env bash
# Reproduce the edge benchmark on a Raspberry Pi. Usage: bash pi/run_benchmark.sh [label] [seconds]
#   label   names the results file (default: pi)
#   seconds sustained-load duration per model for the thermal test (default: 300)
set -euo pipefail
label="${1:-pi}"
duration="${2:-300}"
cd "$(dirname "$0")/.."

echo "throttled before: $(vcgencmd get_throttled 2>/dev/null || echo n/a)"
edgeauth download-models --state .edgeauth-study
edgeauth-study prepare                      # LFW, ~180 MB on first run
edgeauth-study benchmark --label "$label" --probes 200 --duration "$duration"
edgeauth-study figures                      # tradeoff_<label>.png and thermal_<label>.png

if command -v vcgencmd >/dev/null; then
  echo "throttled after: $(vcgencmd get_throttled)  (0x0 = no throttling or under-voltage)"
  vcgencmd measure_temp
fi
echo "commit reports/results/benchmark_${label}.json and reports/figures/*_${label}.png"
