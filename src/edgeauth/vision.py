"""On-device image decoding, face quality, embeddings, and passive PAD."""

import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from .models import PAD, RECOGNIZERS, YUNET


class CaptureRejected(ValueError):
    """A capture could not be processed to a usable face sample."""


class Condition(str, Enum):
    CLEAN = "clean"
    LOW_LIGHT = "low-light"
    BLUR = "blur"
    LOW_RESOLUTION = "low-resolution"


@dataclass(frozen=True)
class FaceSample:
    embedding: np.ndarray
    face_crop: np.ndarray


def read_image(path: Path) -> np.ndarray:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path, formats=["JPEG", "PNG"]) as image:
                if image.width * image.height > 12_000_000:
                    raise CaptureRejected("image exceeds 12 megapixels")
                if getattr(image, "is_animated", False):
                    raise CaptureRejected("supply one still image")
                rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    except (
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as error:
        raise CaptureRejected("could not decode JPEG or PNG capture") from error
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def perturb(image: np.ndarray, condition: Condition) -> np.ndarray:
    if condition == Condition.LOW_LIGHT:
        return cv2.convertScaleAbs(image, alpha=0.45)
    if condition == Condition.BLUR:
        return cv2.GaussianBlur(image, (0, 0), sigmaX=4)
    if condition == Condition.LOW_RESOLUTION:
        height, width = image.shape[:2]
        reduced = cv2.resize(
            image, (max(1, width // 4), max(1, height // 4)), interpolation=cv2.INTER_AREA
        )
        return cv2.resize(reduced, (width, height), interpolation=cv2.INTER_LINEAR)
    return image


def unit_vector(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vector)
    if vector.size == 0 or not np.isfinite(norm) or norm <= 1e-8:
        raise ValueError("invalid embedding")
    return vector / norm


def cosine(reference: np.ndarray, probe: np.ndarray) -> float:
    return float(np.clip(unit_vector(reference) @ unit_vector(probe), -1.0, 1.0))


class FacePipeline:
    def __init__(self, directory: Path, variant: str):
        cv2.setNumThreads(1)
        model = RECOGNIZERS[variant]
        detector_buffer = np.frombuffer(YUNET.read(directory / YUNET.filename), np.uint8)
        recognition_buffer = np.frombuffer(model.read(directory / model.filename), np.uint8)
        self.detector = cv2.FaceDetectorYN.create(
            "onnx",
            detector_buffer,
            b"",
            (320, 320),
            0.8,
            0.3,
            100,
        )
        self.recognizer = cv2.FaceRecognizerSF.create("onnx", recognition_buffer, b"")
        self.model_id = f"{variant}:{model.digest}:{YUNET.digest}:quality-v1"

    def analyse(self, image: np.ndarray) -> FaceSample:
        height, width = image.shape[:2]
        if min(height, width) < 64:
            raise CaptureRejected("capture dimensions are too small")
        factor = min(1.0, 960 / max(height, width))
        size = (round(width * factor), round(height * factor))
        detection_image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        if min(size) < 32:
            raise CaptureRejected("capture aspect ratio is too narrow")
        self.detector.setInputSize(size)
        _, faces = self.detector.detect(detection_image)
        if faces is None or len(faces) != 1:
            raise CaptureRejected("capture must contain exactly one detected face")
        face = faces[0].copy()
        face[:14] /= factor
        x, y, w, h = face[:4]
        if min(w, h) < 64:
            raise CaptureRejected("face is too small")
        if x < 0 or y < 0 or x + w > width or y + h > height:
            raise CaptureRejected("face must be entirely inside the frame")
        aligned = self.recognizer.alignCrop(image, face)
        gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        if brightness < 40 or brightness > 220:
            raise CaptureRejected("face exposure is outside the quality range")
        if cv2.Laplacian(gray, cv2.CV_64F).var() < 35:
            raise CaptureRejected("face is too blurred")
        face_crop = image[int(y) : int(np.ceil(y + h)), int(x) : int(np.ceil(x + w))]
        return FaceSample(unit_vector(self.recognizer.feature(aligned)), face_crop)


class PresentationDetector:
    def __init__(self, directory: Path):
        buffer = np.frombuffer(PAD.read(directory / PAD.filename), np.uint8)
        self.net = cv2.dnn.readNetFromONNX(buffer)
        self.model_id = f"anti-spoof-mn3:{PAD.digest}:raw-face-crop-v1"

    def live_score(self, face_crop: np.ndarray) -> float:
        resized = cv2.resize(face_crop, (128, 128), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        mean = np.array([151.2405, 119.5950, 107.8395], dtype=np.float32)
        scale = np.array([63.0105, 56.4570, 55.0035], dtype=np.float32)
        blob = ((rgb - mean) / scale).transpose(2, 0, 1)[None].copy()
        self.net.setInput(blob)
        probabilities = self.net.forward().reshape(-1)
        if (
            probabilities.shape != (2,)
            or not np.all(np.isfinite(probabilities))
            or not np.all((0 <= probabilities) & (probabilities <= 1))
            or not np.isclose(probabilities.sum(), 1, atol=1e-3)
        ):
            raise ValueError("invalid PAD model output")
        return float(probabilities[0])
