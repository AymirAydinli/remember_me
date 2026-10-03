from collections.abc import Generator, Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from remember_me.database import Base, get_db
from remember_me.face_service import (
    MultipleFacesDetectedError,
    NoFaceDetectedError,
)
from remember_me.main import app
from remember_me.models import FaceEmbedding, Person


@pytest.fixture
def registration_context() -> Iterator[tuple[TestClient, sessionmaker[Session]]]:
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    test_session_factory = sessionmaker(
        bind=test_engine,
        expire_on_commit=False,
    )
    Base.metadata.create_all(bind=test_engine)

    def override_get_db() -> Generator[Session, None, None]:
        db = test_session_factory()

        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    try:
        yield client, test_session_factory
    finally:
        app.dependency_overrides.clear()
        client.close()
        test_engine.dispose()


def test_register_person(
    registration_context: tuple[TestClient, sessionmaker[Session]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, test_session_factory = registration_context
    expected_embedding = [0.1, -0.2, 0.3]

    def fake_generate_embedding(image_bytes: bytes) -> list[float]:
        assert image_bytes == b"fake-image"
        return expected_embedding

    monkeypatch.setattr(
        "remember_me.main.generate_face_embedding",
        fake_generate_embedding,
    )

    response = client.post(
        "/people/register",
        data={
            "name": "Alice",
            "relationship": "Mother",
        },
        files={
            "image": ("alice.jpg", b"fake-image", "image/jpeg"),
        },
    )

    assert response.status_code == 201
    assert response.json()["name"] == "Alice"
    assert response.json()["relationship"] == "Mother"

    with test_session_factory() as db:
        people = db.scalars(select(Person)).all()
        embeddings = db.scalars(select(FaceEmbedding)).all()

    assert len(people) == 1
    assert len(embeddings) == 1
    assert embeddings[0].person_id == people[0].id
    assert embeddings[0].embedding == expected_embedding
    assert embeddings[0].model_name == "ArcFace"


def test_invalid_image_creates_no_records(
    registration_context: tuple[TestClient, sessionmaker[Session]],
) -> None:
    client, test_session_factory = registration_context

    response = client.post(
        "/people/register",
        data={
            "name": "Alice",
            "relationship": "Mother",
        },
        files={
            "image": ("invalid.jpg", b"not-an-image", "image/jpeg"),
        },
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "The uploaded file is not a valid image"}

    with test_session_factory() as db:
        assert db.scalars(select(Person)).all() == []
        assert db.scalars(select(FaceEmbedding)).all() == []


@pytest.mark.parametrize(
    ("processing_error", "expected_detail"),
    [
        (
            NoFaceDetectedError("No face was detected"),
            "No face was detected",
        ),
        (
            MultipleFacesDetectedError(
                "Multiple faces were detected; " "upload an image containing one face"
            ),
            "Multiple faces were detected; " "upload an image containing one face",
        ),
    ],
)
def test_face_detection_error_creates_no_records(
    registration_context: tuple[TestClient, sessionmaker[Session]],
    monkeypatch: pytest.MonkeyPatch,
    processing_error: Exception,
    expected_detail: str,
) -> None:
    client, test_session_factory = registration_context

    def fail_to_generate_embedding(image_bytes: bytes) -> list[float]:
        raise processing_error

    monkeypatch.setattr(
        "remember_me.main.generate_face_embedding",
        fail_to_generate_embedding,
    )

    response = client.post(
        "/people/register",
        data={
            "name": "Alice",
            "relationship": "Mother",
        },
        files={
            "image": ("alice.jpg", b"fake-image", "image/jpeg"),
        },
    )

    assert response.status_code == 422
    assert response.json() == {"detail": expected_detail}

    with test_session_factory() as db:
        assert db.scalars(select(Person)).all() == []
        assert db.scalars(select(FaceEmbedding)).all() == []


def test_recognizes_familiar_person(
    registration_context: tuple[TestClient, sessionmaker[Session]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, test_session_factory = registration_context

    with test_session_factory() as db:
        person = Person(
            name="Anna Kowalska",
            relationship="Daughter",
        )
        person.embeddings.append(
            FaceEmbedding(
                embedding=[1.0, 0.0],
                model_name="ArcFace",
            )
        )
        db.add(person)
        db.commit()

    def fake_generate_embedding(image_bytes: bytes) -> list[float]:
        assert image_bytes == b"query-image"
        return [0.99, 0.1]

    monkeypatch.setattr(
        "remember_me.main.generate_face_embedding",
        fake_generate_embedding,
    )

    response = client.post(
        "/api/recognize",
        files={
            "image": ("query.jpg", b"query-image", "image/jpeg"),
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "recognized": True,
        "name": "Anna Kowalska",
        "relationship": "Daughter",
    }


def test_returns_unknown_when_closest_match_is_weak(
    registration_context: tuple[TestClient, sessionmaker[Session]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, test_session_factory = registration_context

    with test_session_factory() as db:
        person = Person(
            name="Wrong Person",
            relationship="Stranger",
        )
        person.embeddings.append(
            FaceEmbedding(
                embedding=[0.5, 0.866],
                model_name="ArcFace",
            )
        )
        db.add(person)
        db.commit()

    monkeypatch.setattr(
        "remember_me.main.generate_face_embedding",
        lambda image_bytes: [1.0, 0.0],
    )

    response = client.post(
        "/api/recognize",
        files={
            "image": ("query.jpg", b"query-image", "image/jpeg"),
        },
    )

    assert response.status_code == 200
    assert response.json() == {"recognized": False}


def test_returns_unknown_when_database_has_no_embeddings(
    registration_context: tuple[TestClient, sessionmaker[Session]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _ = registration_context

    monkeypatch.setattr(
        "remember_me.main.generate_face_embedding",
        lambda image_bytes: [1.0, 0.0],
    )

    response = client.post(
        "/api/recognize",
        files={
            "image": ("query.jpg", b"query-image", "image/jpeg"),
        },
    )

    assert response.status_code == 200
    assert response.json() == {"recognized": False}
