# Raspberry Pi: benchmark and live demo

The study's headline numbers are only persuasive if they come from the hardware. This folder
is everything needed to reproduce them on a Pi (tested target: Raspberry Pi 4/5, 64-bit
Raspberry Pi OS Bookworm).

## 1. Set up

```bash
sudo apt install -y python3-venv python3-dev cmake libopenblas-dev
python3 -m venv ~/edgeauth-venv && . ~/edgeauth-venv/bin/activate
git clone https://github.com/Ghobi-A/Facial-Recognition-Smart-Lock && cd Facial-Recognition-Smart-Lock
pip install -e '.[study]'
pip install -e '.[legacy]'   # optional: builds dlib from source, ~30-60 min on a Pi
```

## 2. Benchmark

```bash
bash pi/run_benchmark.sh pi5        # or pi4; the label names the results file
bash pi/run_benchmark.sh pi5 600    # optional: a 10-minute sustained run per model
```

The script downloads LFW (~180 MB) if needed. It then times 200 verification probes per
model, each model in a fresh process, and runs each model flat out for 5 minutes. During the
sustained run it records throughput, SoC temperature and board power every second, along
with the Pi's throttle flags. It writes:

* `reports/results/benchmark_pi5.json`: latency p50/p95, throughput, peak memory, model size,
  power, the per-second sustained series, throughput drift (first vs last minute), peak
  temperature and the throttle bitmask;
* `reports/figures/tradeoff_pi5.png`: accuracy against latency and model size;
* `reports/figures/thermal_pi5.png`: throughput and temperature over the sustained run.

Accuracy doesn't need re-running on the Pi. Embeddings are deterministic, so accuracy comes
from the full LFW run on any machine.

**Power:** a Pi 5 reports board power from its PMIC (`vcgencmd pmic_read_adc`). A Pi 4 has no
on-board sensor, so its power fields come back `null`. Record power with a USB-C meter instead
and add it to the JSON by hand, noting the meter you used.

**Fair comparison:**
* Close other programs and use the official power supply.
* Note whether a heatsink or fan is fitted. Thermal throttling is exactly what the sustained
  run is designed to expose, so report it either way.
* A throttle value other than `0x0` means the run hit under-voltage or thermal limits. Keep
  the result, and say so in the write-up.

## 3. Live demo (optional)

```bash
edgeauth init
edgeauth enrol alice a1.jpg a2.jpg a3.jpg          # three consented captures
python pi/door_demo.py alice --policy policy.json  # policy from `edgeauth evaluate --policy-out`
```

Wire the relay or lock driver to GPIO 23 (BCM) through a transistor or relay board; never
drive a lock coil from a GPIO pin directly. Without `RPi.GPIO` the relay is simulated.

The demo requires three consecutive accepted frames and always releases the relay on exit.
It is a demonstration of the pipeline, not a secure lock: see the security notes in the
main README.
