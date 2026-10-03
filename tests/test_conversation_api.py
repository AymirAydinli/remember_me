from collections.abc import Generator, Iterator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from remember_me.database import Base, get_db
from remember_me.main import app, get_openai_service
from remember_me.models import Conversation, Person
from remember_me.openai_service import OpenAIService, OpenAIServiceError
from remember_me.schemas import ConversationSummaryOutput


@pytest.fixture
def conversation_context(
) -> Iterator[tuple[TestClient, sessionmaker[Session], MagicMock]]:
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
    openai_service = MagicMock(spec=OpenAIService)

    def override_get_db() -> Generator[Session, None, None]:
        db = test_session_factory()

        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_openai_service] = lambda: openai_service
    client = TestClient(app)

    try:
        yield client, test_session_factory, openai_service
    finally:
        app.dependency_overrides.clear()
        client.close()
        test_engine.dispose()


def create_person(test_session_factory: sessionmaker[Session]) -> int:
    with test_session_factory() as db:
        person = Person(name="Anna", relationship="Daughter")
        db.add(person)
        db.commit()
        return person.id


def test_creates_conversation_summary(
    conversation_context: tuple[TestClient, sessionmaker[Session], MagicMock],
) -> None:
    client, test_session_factory, openai_service = conversation_context
    person_id = create_person(test_session_factory)
    openai_service.transcribe_audio.return_value = "Anna discussed her trip."
    openai_service.summarize_transcript.return_value = ConversationSummaryOutput(
        summary="Anna discussed her upcoming trip.",
        topics=["travel"],
        follow_up="Ask Anna how the trip went.",
    )

    response = client.post(
        f"/api/people/{person_id}/conversations",
        files={
            "audio": ("conversation.webm", b"fake-audio", "audio/webm"),
        },
    )

    assert response.status_code == 201
    assert response.json()["person_id"] == person_id
    assert response.json()["summary"] == "Anna discussed her upcoming trip."
    assert response.json()["topics"] == ["travel"]
    assert response.json()["follow_up"] == "Ask Anna how the trip went."
    openai_service.transcribe_audio.assert_called_once_with(
        b"fake-audio",
        filename="conversation.webm",
        content_type="audio/webm",
    )
    openai_service.summarize_transcript.assert_called_once_with(
        "Anna discussed her trip."
    )

    with test_session_factory() as db:
        conversations = db.scalars(select(Conversation)).all()

    assert len(conversations) == 1
    assert conversations[0].person_id == person_id
    assert conversations[0].summary == "Anna discussed her upcoming trip."


def test_conversation_rejects_missing_person(
    conversation_context: tuple[TestClient, sessionmaker[Session], MagicMock],
) -> None:
    client, _, openai_service = conversation_context

    response = client.post(
        "/api/people/999/conversations",
        files={
            "audio": ("conversation.webm", b"fake-audio", "audio/webm"),
        },
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Person not found"}
    openai_service.transcribe_audio.assert_not_called()


@pytest.mark.parametrize(
    ("audio_bytes", "content_type", "expected_detail"),
    [
        (b"", "audio/webm", "The audio recording is empty"),
        (b"fake-audio", "text/plain", "Unsupported audio type"),
    ],
)
def test_conversation_rejects_invalid_audio(
    conversation_context: tuple[TestClient, sessionmaker[Session], MagicMock],
    audio_bytes: bytes,
    content_type: str,
    expected_detail: str,
) -> None:
    client, test_session_factory, openai_service = conversation_context
    person_id = create_person(test_session_factory)

    response = client.post(
        f"/api/people/{person_id}/conversations",
        files={
            "audio": ("conversation.webm", audio_bytes, content_type),
        },
    )

    assert response.status_code == 400
    assert response.json() == {"detail": expected_detail}
    openai_service.transcribe_audio.assert_not_called()


def test_openai_failure_creates_no_conversation(
    conversation_context: tuple[TestClient, sessionmaker[Session], MagicMock],
) -> None:
    client, test_session_factory, openai_service = conversation_context
    person_id = create_person(test_session_factory)
    openai_service.transcribe_audio.side_effect = OpenAIServiceError(
        "Provider failure"
    )

    response = client.post(
        f"/api/people/{person_id}/conversations",
        files={
            "audio": ("conversation.webm", b"fake-audio", "audio/webm"),
        },
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Conversation processing is unavailable"
    }

    with test_session_factory() as db:
        assert db.scalars(select(Conversation)).all() == []
