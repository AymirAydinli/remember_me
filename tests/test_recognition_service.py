from remember_me.models import FaceEmbedding
from remember_me.recognition_service import (
    cosine_distance,
    find_matching_embedding,
)


def make_embedding(
    values: list[float],
    *,
    embedding_id: int = 1,
    model_name: str = "ArcFace",
) -> FaceEmbedding:
    return FaceEmbedding(
        id=embedding_id,
        person_id=embedding_id,
        embedding=values,
        model_name=model_name,
    )


def test_cosine_distance() -> None:
    assert cosine_distance([1.0, 0.0], [1.0, 0.0]) == 0.0
    assert cosine_distance([1.0, 0.0], [0.0, 1.0]) == 1.0


def test_selects_best_acceptable_embedding() -> None:
    weaker_match = make_embedding(
        [0.8, 0.6],
        embedding_id=1,
    )
    stronger_match = make_embedding(
        [0.99, 0.1],
        embedding_id=2,
    )

    result = find_matching_embedding(
        [1.0, 0.0],
        [weaker_match, stronger_match],
    )

    assert result is stronger_match


def test_rejects_closest_embedding_when_match_is_weak() -> None:
    weak_match = make_embedding(
        [0.5, 0.866],
        embedding_id=1,
    )

    result = find_matching_embedding(
        [1.0, 0.0],
        [weak_match],
    )

    assert result is None


def test_returns_none_when_no_embeddings_exist() -> None:
    result = find_matching_embedding([1.0, 0.0], [])

    assert result is None


def test_skips_incompatible_embeddings() -> None:
    wrong_model = make_embedding(
        [1.0, 0.0],
        embedding_id=1,
        model_name="VGG-Face",
    )
    wrong_dimensions = make_embedding(
        [1.0, 0.0, 0.0],
        embedding_id=2,
    )

    result = find_matching_embedding(
        [1.0, 0.0],
        [wrong_model, wrong_dimensions],
    )

    assert result is None
