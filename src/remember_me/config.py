from dataclasses import dataclass
import os


class OpenAIConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class OpenAISettings:
    api_key: str
    transcription_model: str
    summary_model: str


def load_openai_settings() -> OpenAISettings:
    values = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "").strip(),
        "OPENAI_TRANSCRIPTION_MODEL": os.getenv(
            "OPENAI_TRANSCRIPTION_MODEL",
            "",
        ).strip(),
        "OPENAI_SUMMARY_MODEL": os.getenv(
            "OPENAI_SUMMARY_MODEL",
            "",
        ).strip(),
    }

    missing = [
        name
        for name, value in values.items()
        if not value or value == "replace-with-your-openai-api-key"
    ]

    if missing:
        missing_names = ", ".join(missing)
        raise OpenAIConfigurationError(
            f"Missing required environment variables: {missing_names}"
        )

    return OpenAISettings(
        api_key=values["OPENAI_API_KEY"],
        transcription_model=values["OPENAI_TRANSCRIPTION_MODEL"],
        summary_model=values["OPENAI_SUMMARY_MODEL"],
    )
