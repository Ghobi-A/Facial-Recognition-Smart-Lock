"""Embedding extractors compared in the study.

* ``dlib-legacy``: the 2022 smart lock's pipeline (face_recognition HOG detector, 128-d
  dlib ResNet embedding, Euclidean distance) with its library defaults.
* ``sface-fp32`` / ``sface-int8``: the lab's YuNet detector, five-point alignment and the
  SFace recogniser at full and 8-bit precision.

LFW images often contain several faces; the protocol subject is the one nearest the centre.
A detection failure is a failure to acquire (FTA) and is kept, not dropped, so attempt-level
error rates stay honest.
"""

from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from ..models import RECOGNIZERS, YUNET, download_model

MODELS = ("dlib-legacy", "sface-fp32", "sface-int8")
# How each model's raw comparison becomes a similarity score (higher = same person).
SCORING = {"dlib-legacy": "negative euclidean", "sface-fp32": "cosine", "sface-int8": "cosine"}


def central(boxes: list[tuple[float, float, float, float]], width: int, height: int) -> int:
    """Index of the box (x, y, w, h) whose centre is nearest the image centre."""
    centre = np.array([width / 2, height / 2])
    distances = [np.linalg.norm(np.array([x + w / 2, y + h / 2]) - centre) for x, y, w, h in boxes]
    return int(np.argmin(distances))


def yaw_proxy(face: np.ndarray) -> float:
    """Nose offset from the eye midpoint, in inter-ocular units (0 = frontal)."""
    right_eye, left_eye, nose = face[4:6], face[6:8], face[8:10]
    spacing = np.linalg.norm(left_eye - right_eye)
    if spacing <= 1e-6:
        return float("nan")
    return float((nose[0] - (right_eye[0] + left_eye[0]) / 2) / spacing)


class SFaceEmbedder:
    def __init__(self, models_dir: Path):
        cv2.setNumThreads(1)
        for model in (YUNET, *RECOGNIZERS.values()):
            download_model(models_dir, model)
        self.detector = cv2.FaceDetectorYN.create(
            str(models_dir / YUNET.filename), "", (320, 320), 0.6, 0.3, 50
        )
        self.recognizers = {
            name: cv2.FaceRecognizerSF.create(str(models_dir / model.filename), "")
            for name, model in RECOGNIZERS.items()
        }

    def detect(self, image: np.ndarray) -> np.ndarray | None:
        height, width = image.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(image)
        if faces is None or len(faces) == 0:
            return None
        return faces[central([tuple(face[:4]) for face in faces], width, height)]

    def __call__(self, path: Path) -> dict:
        image = cv2.imread(str(path))
        face = None if image is None else self.detect(image)
        if face is None:
            return {"yaw": float("nan")}
        result = {"yaw": yaw_proxy(face)}
        for name, recognizer in self.recognizers.items():
            vector = recognizer.feature(recognizer.alignCrop(image, face)).reshape(-1)
            result[name] = vector / np.linalg.norm(vector)
        return result


class DlibEmbedder:
    """The 2022 pipeline: face_recognition defaults (HOG, 1x upsample, 'small' landmarks)."""

    def __init__(self, _models_dir: Path | None = None):
        import face_recognition

        self.fr = face_recognition

    def __call__(self, path: Path) -> dict:
        rgb = self.fr.load_image_file(str(path))
        boxes = self.fr.face_locations(rgb, model="hog")
        if not boxes:
            return {}
        height, width = rgb.shape[:2]
        # face_recognition boxes are (top, right, bottom, left).
        xywh = [(left, top, right - left, bottom - top) for top, right, bottom, left in boxes]
        box = boxes[central(xywh, width, height)]
        return {"dlib-legacy": self.fr.face_encodings(rgb, [box])[0]}


_worker = None


def _init(kind: str, models_dir: Path) -> None:
    global _worker
    _worker = (SFaceEmbedder if kind == "sface" else DlibEmbedder)(models_dir)


def _run(path: Path) -> dict:
    return _worker(path)


def embed_images(
    kind: str, root: Path, images: list[str], models_dir: Path, workers: int = 4
) -> list[dict]:
    paths = [root / image for image in images]
    with Pool(workers, initializer=_init, initargs=(kind, models_dir)) as pool:
        return pool.map(_run, paths, chunksize=32)


def save(path: Path, images: list[str], results: list[dict], model: str) -> None:
    dims = next(len(r[model]) for r in results if model in r)
    vectors = np.full((len(images), dims), np.nan, dtype=np.float32)
    for index, result in enumerate(results):
        if model in result:
            vectors[index] = result[model]
    extra = {}
    if "yaw" in results[0]:
        extra["yaw"] = np.array([r["yaw"] for r in results], dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, images=np.array(images), vectors=vectors, **extra)


def load(path: Path) -> tuple[dict[str, int], np.ndarray, np.ndarray | None]:
    data = np.load(path)
    index = {str(image): i for i, image in enumerate(data["images"])}
    return index, data["vectors"], data["yaw"] if "yaw" in data else None


def pair_scores(model: str, vectors: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Similarity per pair; NaN where either image failed to acquire."""
    left, right = vectors[a], vectors[b]
    if SCORING[model] == "cosine":
        return np.sum(left * right, axis=1)
    return -np.linalg.norm(left - right, axis=1)
