from collections.abc import Iterable

import numpy as np

from remember_me.face_service import MODEL_NAME
from remember_me.models import FaceEmbedding

MAX_COSINE_DISTANCE = 0.45


def cosine_distance(
    first_embedding: list[float],
    second_embedding: list[float],
) -> float | None:
    first = np.asarray(first_embedding, dtype=np.float32)
    second = np.asarray(second_embedding, dtype=np.float32)

    if first.ndim != 1 or second.ndim != 1:
        return None

    if first.size == 0 or first.shape != second.shape:
        return None

    denominator = np.linalg.norm(first) * np.linalg.norm(second)

    if denominator == 0:
        return None

    similarity = float(np.dot(first, second) / denominator)
    similarity = float(np.clip(similarity, -1.0, 1.0))

    return 1.0 - similarity


def find_matching_embedding(
    query_embedding: list[float],
    stored_embeddings: Iterable[FaceEmbedding],
    max_distance: float = MAX_COSINE_DISTANCE,
) -> FaceEmbedding | None:
    best_embedding: FaceEmbedding | None = None
    best_distance = float("inf")

    for stored_embedding in stored_embeddings:
        if stored_embedding.model_name != MODEL_NAME:
            continue

        distance = cosine_distance(
            query_embedding,
            stored_embedding.embedding,
        )

        if distance is not None and distance < best_distance:
            best_distance = distance
            best_embedding = stored_embedding

    if best_embedding is None or best_distance > max_distance:
        return None

    return best_embedding
