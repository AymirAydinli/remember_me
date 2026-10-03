from dataclasses import dataclass

import cv2
import numpy as np

MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "retinaface"


class FaceProcessingError(Exception):
    """Base error for expected face-processing failures."""


class InvalidImageError(FaceProcessingError):
    pass


class NoFaceDetectedError(FaceProcessingError):
    pass


class MultipleFacesDetectedError(FaceProcessingError):
    pass


@dataclass(frozen=True)
class FaceBox:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class FaceAnalysis:
    embedding: list[float]
    face: FaceBox


def analyze_face(image_bytes: bytes) -> FaceAnalysis:
    if not image_bytes:
        raise InvalidImageError("The uploaded image is empty")

    encoded_image = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)

    if image is None:
        raise InvalidImageError("The uploaded file is not a valid image")

    # Import lazily so normal API startup does not initialize TensorFlow.
    from deepface import DeepFace

    try:
        representations = DeepFace.represent(
            img_path=image,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
            align=True,
        )
    except ValueError as error:
        if "face could not be detected" in str(error).lower():
            raise NoFaceDetectedError("No face was detected") from error

        raise FaceProcessingError("Face processing failed") from error
    except Exception as error:
        raise FaceProcessingError("Face processing failed") from error

    if not representations:
        raise NoFaceDetectedError("No face was detected")

    if len(representations) > 1:
        raise MultipleFacesDetectedError(
            "Multiple faces were detected; upload an image containing one face"
        )

    representation = representations[0]
    embedding = representation.get("embedding")
    facial_area = representation.get("facial_area")

    if embedding is None or len(embedding) == 0:
        raise FaceProcessingError("ArcFace did not return an embedding")

    if not isinstance(facial_area, dict):
        raise FaceProcessingError("Face coordinates were not returned")

    required_coordinates = {"x", "y", "w", "h"}

    if not required_coordinates.issubset(facial_area):
        raise FaceProcessingError("Face coordinates are incomplete")

    face = FaceBox(
        x=int(facial_area["x"]),
        y=int(facial_area["y"]),
        width=int(facial_area["w"]),
        height=int(facial_area["h"]),
    )

    return FaceAnalysis(
        embedding=[float(value) for value in embedding],
        face=face,
    )


def generate_face_embedding(image_bytes: bytes) -> list[float]:
    return analyze_face(image_bytes).embedding
