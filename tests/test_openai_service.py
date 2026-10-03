from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from openai import OpenAIError

from remember_me.config import OpenAISettings
from remember_me.openai_service import (
    EmptyTranscriptionError,
    OpenAIService,
    OpenAIServiceError,
)


@pytest.fixture
def settings() -> OpenAISettings:
    return OpenAISettings(
        api_key="test-api-key",
        transcription_model="test-transcription-model",
        summary_model="test-summary-model",
    )


def test_transcribes_audio(
    settings: OpenAISettings,
) -> None:
    client = MagicMock()
    client.audio.transcriptions.create.return_value = SimpleNamespace(
        text="  Anna discussed her upcoming trip.  "
    )
    service = OpenAIService(
        settings=settings,
        client=client,
    )

    transcript = service.transcribe_audio(
        b"fake-audio",
        filename="conversation.webm",
        content_type="audio/webm",
    )

    assert transcript == "Anna discussed her upcoming trip."
    client.audio.transcriptions.create.assert_called_once_with(
        model="test-transcription-model",
        file=(
            "conversation.webm",
            b"fake-audio",
            "audio/webm",
        ),
    )


def test_rejects_empty_audio(
    settings: OpenAISettings,
) -> None:
    client = MagicMock()
    service = OpenAIService(
        settings=settings,
        client=client,
    )

    with pytest.raises(
        EmptyTranscriptionError,
        match="audio recording is empty",
    ):
        service.transcribe_audio(
            b"",
            filename="conversation.webm",
            content_type="audio/webm",
        )

    client.audio.transcriptions.create.assert_not_called()


def test_rejects_empty_transcription(
    settings: OpenAISettings,
) -> None:
    client = MagicMock()
    client.audio.transcriptions.create.return_value = SimpleNamespace(text="   ")
    service = OpenAIService(
        settings=settings,
        client=client,
    )

    with pytest.raises(
        EmptyTranscriptionError,
        match="did not contain any speech",
    ):
        service.transcribe_audio(
            b"silent-audio",
            filename="conversation.webm",
            content_type="audio/webm",
        )


def test_translates_openai_errors(
    settings: OpenAISettings,
) -> None:
    client = MagicMock()
    client.audio.transcriptions.create.side_effect = OpenAIError("Provider failure")
    service = OpenAIService(
        settings=settings,
        client=client,
    )

    with pytest.raises(
        OpenAIServiceError,
        match="Audio transcription failed",
    ):
        service.transcribe_audio(
            b"fake-audio",
            filename="conversation.webm",
            content_type="audio/webm",
        )
