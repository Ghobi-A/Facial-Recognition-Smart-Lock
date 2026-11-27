"""Single-thread CPU comparison of the shipped FP32 and INT8 recognition weights."""

import platform
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from .models import PAD, RECOGNIZERS, YUNET
from .vision import FacePipeline, PresentationDetector, cosine, read_image


def benchmark(directory: Path, capture: Path, runs: int, warmup: int) -> dict:
    if not 1 <= runs <= 1000 or not 0 <= warmup <= 100:
        raise ValueError("use 1–1000 measured runs and 0–100 warm-up runs")
    image = read_image(capture)
    detector = PresentationDetector(directory)
    results = {}
    baseline = None
    for variant, model in RECOGNIZERS.items():
        matcher = FacePipeline(directory, variant)
        face_times, pad_times, cumulative = [], [], []
        for iteration in range(warmup + runs):
            started = perf_counter()
            sample = matcher.analyse(image)
            face_finished = perf_counter()
            detector.live_score(sample.face_crop)
            finished = perf_counter()
            if iteration >= warmup:
                face_times.append((face_finished - started) * 1000)
                pad_times.append((finished - face_finished) * 1000)
                cumulative.append((finished - started) * 1000)
        if baseline is None:
            baseline = sample.embedding
        results[variant] = {
            "matcher": matcher.model_id,
            "recognition_weights_bytes": (directory / model.filename).stat().st_size,
            "cosine_with_fp32_embedding": cosine(baseline, sample.embedding),
            "detect_align_embed_p50_ms": float(np.percentile(face_times, 50)),
            "pad_p50_ms": float(np.percentile(pad_times, 50)),
            "pipeline_p50_ms": float(np.percentile(cumulative, 50)),
            "pipeline_p95_ms": float(np.percentile(cumulative, 95)),
        }
    return {
        "schema": "edgeauth/benchmark/v1",
        "system": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "opencv_threads": cv2.getNumThreads(),
        "image_shape_hwc": list(image.shape),
        "warmup_runs": warmup,
        "measured_runs": runs,
        "shared_weights_bytes": sum(
            (directory / model.filename).stat().st_size for model in (YUNET, PAD)
        ),
        "variants": results,
    }
