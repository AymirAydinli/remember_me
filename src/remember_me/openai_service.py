from openai import OpenAI, OpenAIError

from remember_me.config import OpenAISettings, load_openai_settings


class OpenAIServiceError(RuntimeError):
    pass


class EmptyTranscriptionError(OpenAIServiceError):
    pass


class OpenAIService:
    def __init__(
        self,
        settings: OpenAISettings | None = None,
        client: OpenAI | None = None,
    ) -> None:
        self.settings = settings or load_openai_settings()
        self.client = client or OpenAI(api_key=self.settings.api_key)

    def transcribe_audio(
        self,
        audio_bytes: bytes,
        *,
        filename: str,
        content_type: str,
    ) -> str:
        if not audio_bytes:
            raise EmptyTranscriptionError("The audio recording is empty")

        try:
            transcription = self.client.audio.transcriptions.create(
                model=self.settings.transcription_model,
                file=(
                    filename,
                    audio_bytes,
                    content_type,
                ),
            )
        except OpenAIError as error:
            raise OpenAIServiceError("Audio transcription failed") from error

        transcript = transcription.text.strip()

        if not transcript:
            raise EmptyTranscriptionError("The recording did not contain any speech")

        return transcript
