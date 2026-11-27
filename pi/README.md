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
bash pi/run_benchmark.sh pi5      # or pi4; the label names the results file
```

This downloads LFW (~180 MB) if needed and times 200 verification probes per model, each
model in a fresh process. It writes `reports/results/benchmark_pi5.json` and
`reports/figures/tradeoff_pi5.png`. Accuracy does not need re-running on the Pi: the
embeddings are deterministic, so accuracy comes from the full LFW run on any machine.

**Power:** on a Pi 5 the script samples board power from the PMIC (`vcgencmd pmic_read_adc`)
at idle and under load. On a Pi 4 there is no on-board sensor, so the power fields are
`null`; record them with a USB-C power meter instead and add them to the JSON by hand,
noting the meter used.

**Fair comparison:** close other programs, use the official power supply, add a heatsink or
fan (thermal throttling inflates p95), and check `vcgencmd get_throttled` returns `0x0`
after the run.

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
