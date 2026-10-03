import pytest

from remember_me.config import (
    OpenAIConfigurationError,
    load_openai_settings,
)


def test_loads_openai_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv(
        "OPENAI_TRANSCRIPTION_MODEL",
        "test-transcription-model",
    )
    monkeypatch.setenv(
        "OPENAI_SUMMARY_MODEL",
        "test-summary-model",
    )

    settings = load_openai_settings()

    assert settings.api_key == "test-key"
    assert settings.transcription_model == "test-transcription-model"
    assert settings.summary_model == "test-summary-model"


def test_reports_missing_openai_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv(
        "OPENAI_TRANSCRIPTION_MODEL",
        raising=False,
    )
    monkeypatch.delenv(
        "OPENAI_SUMMARY_MODEL",
        raising=False,
    )

    with pytest.raises(OpenAIConfigurationError) as error:
        load_openai_settings()

    message = str(error.value)

    assert "OPENAI_API_KEY" in message
    assert "OPENAI_TRANSCRIPTION_MODEL" in message
    assert "OPENAI_SUMMARY_MODEL" in message
