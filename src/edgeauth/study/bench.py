"""Edge trade-off benchmark: latency, peak memory, model size and (on a Pi 5) power.

Each model runs in a fresh process so peak-RSS numbers are not polluted by other models.
Timings cover one full verification probe: decode, detect, align and embed.
"""

import multiprocessing as mp
import platform
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from ..models import RECOGNIZERS, YUNET
from . import embedders


def device() -> dict:
    info = {
        "machine": platform.machine(),
        "system": platform.system(),
        "python": platform.python_version(),
    }
    model_file = Path("/proc/device-tree/model")
    if model_file.exists():
        info["board"] = model_file.read_text(errors="ignore").strip("\x00\n ")
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        match = re.search(r"model name\s*:\s*(.+)", cpuinfo.read_text())
        if match:
            info["cpu"] = match.group(1).strip()
    try:
        import cv2

        info["opencv"] = cv2.__version__
    except ImportError:
        pass
    return info


def pi5_power_watts() -> float | None:
    """Total board power from the Pi 5 PMIC (sum of rail V*I); None elsewhere."""
    if not shutil.which("vcgencmd"):
        return None
    try:
        text = subprocess.run(
            ["vcgencmd", "pmic_read_adc"], capture_output=True, text=True, timeout=2
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    currents = dict(re.findall(r"(\w+)_A current\(\d+\)=([\d.]+)A", text))
    volts = dict(re.findall(r"(\w+)_V volt\(\d+\)=([\d.]+)V", text))
    rails = set(currents) & set(volts)
    return sum(float(currents[r]) * float(volts[r]) for r in rails) if rails else None


class PowerSampler(threading.Thread):
    def __init__(self, interval: float = 0.2):
        super().__init__(daemon=True)
        self.interval, self.samples, self.stopped = interval, [], threading.Event()

    def run(self) -> None:
        while not self.stopped.is_set():
            watts = pi5_power_watts()
            if watts is not None:
                self.samples.append(watts)
            time.sleep(self.interval)

    def mean(self) -> float | None:
        return float(np.mean(self.samples)) if self.samples else None


def _rss_mb() -> float:
    import resource  # Unix only; the benchmark targets Linux (Pi) and macOS

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if platform.system() == "Darwin" else peak / 1024


def _measure(model: str, paths: list[str], models_dir: str, warmup: int, queue) -> None:
    import cv2

    cv2.setNumThreads(1)
    before = _rss_mb()
    if model == "dlib-legacy":
        run = embedders.DlibEmbedder()
    else:
        sface = embedders.SFaceEmbedder(Path(models_dir))
        recognizer = sface.recognizers[model]

        def run(path):
            image = cv2.imread(str(path))
            face = sface.detect(image)
            if face is not None:
                recognizer.feature(recognizer.alignCrop(image, face))

    for path in paths[:warmup]:
        run(Path(path))
    idle = pi5_power_watts()
    sampler = PowerSampler()
    sampler.start()
    timings = []
    for path in paths[warmup:]:
        started = time.perf_counter()
        run(Path(path))
        timings.append((time.perf_counter() - started) * 1000)
    sampler.stopped.set()
    sampler.join()
    queue.put(
        {
            "probes": len(timings),
            "latency_ms_p50": float(np.percentile(timings, 50)),
            "latency_ms_p95": float(np.percentile(timings, 95)),
            "throughput_fps": float(1000 / np.mean(timings)),
            "peak_rss_mb": _rss_mb(),
            "model_rss_mb": _rss_mb() - before,
            "power_w_idle": idle,
            "power_w_load": sampler.mean(),
        }
    )


def model_bytes(model: str, models_dir: Path) -> int:
    if model == "dlib-legacy":
        import face_recognition_models as frm

        files = [
            frm.face_recognition_model_location(),
            frm.pose_predictor_five_point_model_location(),
        ]
        return sum(Path(f).stat().st_size for f in files)
    return (models_dir / RECOGNIZERS[model].filename).stat().st_size + (
        models_dir / YUNET.filename
    ).stat().st_size


def benchmark(images: list[Path], models_dir: Path, models: list[str], warmup: int = 10) -> dict:
    context = mp.get_context("spawn")
    results = {"device": device(), "warmup": warmup, "models": {}}
    for model in models:
        queue = context.Queue()
        process = context.Process(
            target=_measure, args=(model, [str(p) for p in images], str(models_dir), warmup, queue)
        )
        process.start()
        result = queue.get()
        process.join()
        result["model_bytes"] = model_bytes(model, models_dir)
        results["models"][model] = result
    return results
