"""Pinned model downloads with byte limits and checksum verification."""

import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

_ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo"
_ZOO_REVISION = "48b10fa505aeb51fbd127fc60a5c660040e823b7"


@dataclass(frozen=True)
class Model:
    filename: str
    urls: tuple[str, ...]
    digest: str
    byte_limit: int
    algorithm: str = "sha256"

    def read(self, path: Path) -> bytes:
        with path.open("rb") as source:
            contents = source.read(self.byte_limit + 1)
        if len(contents) > self.byte_limit:
            raise ValueError(f"Model exceeds its size limit: {self.filename}")
        if hashlib.new(self.algorithm, contents).hexdigest() != self.digest:
            raise ValueError(f"Checksum mismatch for {self.filename}")
        return contents


def zoo_model(folder: str, filename: str, digest: str, byte_limit: int) -> Model:
    path = f"models/{folder}/{filename}"
    return Model(
        filename,
        (
            f"{_ZOO}/{_ZOO_REVISION}/{path}",
            f"https://huggingface.co/opencv/opencv_zoo/resolve/main/{path}",
        ),
        digest,
        byte_limit,
    )


YUNET = zoo_model(
    "face_detection_yunet",
    "face_detection_yunet_2023mar.onnx",
    "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    256_000,
)
RECOGNIZERS = {
    "sface-fp32": zoo_model(
        "face_recognition_sface",
        "face_recognition_sface_2021dec.onnx",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
        40_000_000,
    ),
    "sface-int8": zoo_model(
        "face_recognition_sface",
        "face_recognition_sface_2021dec_int8.onnx",
        "2b0e941e6f16cc048c20aee0c8e31f569118f65d702914540f7bfdc14048d78a",
        11_000_000,
    ),
}
PAD = Model(
    "anti-spoof-mn3.onnx",
    (
        "https://storage.openvinotoolkit.org/repositories/open_model_zoo/public/2022.1/"
        "anti-spoof-mn3/anti-spoof-mn3.onnx",
    ),
    "6de4534964b723397b3e8c995cadcf43bc007cc2f9930b95ae25f76adccece5d1d4d058d0b15117b9e4a9f758424f92a",
    12_270_179,
    "sha384",
)


def download_model(directory: Path, model: Model) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / model.filename
    if path.exists():
        model.read(path)
        return path
    errors = []
    for url in model.urls:
        temporary = None
        try:
            request = Request(url, headers={"User-Agent": "EdgeAuth-Lab/0.1"})
            with urlopen(request, timeout=30) as response:
                contents = response.read(model.byte_limit + 1)
            with tempfile.NamedTemporaryFile(dir=directory, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(contents)
            model.read(temporary)
            temporary.replace(path)
            return path
        except (OSError, ValueError) as error:
            errors.append(str(error))
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    raise ValueError(f"Could not download {model.filename}: {'; '.join(errors)}")
