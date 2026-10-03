from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, joinedload
from remember_me.config import OpenAIConfigurationError
from remember_me.database import engine, get_db
from remember_me.models import Person, FaceEmbedding, Conversation
from remember_me.face_service import (
    MODEL_NAME,
    FaceProcessingError,
    InvalidImageError,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
    analyze_face,
    generate_face_embedding,
)
from remember_me.openai_service import (
    EmptyTranscriptionError,
    OpenAIService,
    OpenAIServiceError,
)
from remember_me.recognition_service import find_matching_embedding

WEB_DIR = Path(__file__).parent / "web"
MAX_IMAGE_SIZE = 10 * 1024 * 1024
MAX_AUDIO_SIZE = 20 * 1024 * 1024

AUDIO_EXTENSIONS = {
    "audio/flac": ".flac",
    "audio/m4a": ".m4a",
    "audio/mp4": ".mp4",
    "audio/mpeg": ".mp3",
    "audio/mpga": ".mpga",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
    "audio/webm": ".webm",
    "audio/x-m4a": ".m4a",
    "audio/x-wav": ".wav",
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    Person.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="Remember Me",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


def get_openai_service() -> OpenAIService:
    try:
        return OpenAIService()
    except OpenAIConfigurationError as error:
        raise HTTPException(
            status_code=503,
            detail="Conversation service is unavailable",
        ) from error


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def homepage():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/register", include_in_schema=False)
def registration_page() -> FileResponse:
    return FileResponse(WEB_DIR / "register.html")


@app.post("/people/register", status_code=201)
def register_person(
    name: Annotated[str, Form()],
    relationship: Annotated[str, Form()],
    image: Annotated[UploadFile, File()],
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    clean_name = name.strip()
    clean_relationship = relationship.strip()
    if not clean_name:
        raise HTTPException(status_code=422, detail="Name cannot be empty")

    if len(clean_name) > 100:
        raise HTTPException(
            status_code=422,
            detail="Name cannot exceed 100 characters",
        )

    if not clean_relationship:
        raise HTTPException(
            status_code=422,
            detail="Relationship cannot be empty",
        )

    if len(clean_relationship) > 100:
        raise HTTPException(
            status_code=422,
            detail="Relationship cannot exceed 100 characters",
        )
    image_bytes = image.file.read(MAX_IMAGE_SIZE + 1)

    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image cannot exceed 10 MB",
        )

    try:
        embedding = generate_face_embedding(image_bytes)
    except InvalidImageError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except NoFaceDetectedError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except MultipleFacesDetectedError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except FaceProcessingError as error:
        raise HTTPException(
            status_code=503,
            detail="Face processing is unavailable",
        ) from error
    person = Person(
        name=clean_name,
        relationship=clean_relationship,
    )
    person.embeddings.append(
        FaceEmbedding(
            embedding=embedding,
            model_name=MODEL_NAME,
        )
    )

    try:
        db.add(person)
        db.commit()
        db.refresh(person)
    except SQLAlchemyError as error:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Could not save the person",
        ) from error

    return {
        "id": person.id,
        "name": person.name,
        "relationship": person.relationship,
    }


@app.post("/api/recognize")
def recognize_face(
    image: Annotated[UploadFile, File()],
    db: Annotated[Session, Depends(get_db)],
) -> dict[str, object]:
    image_bytes = image.file.read(MAX_IMAGE_SIZE + 1)

    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image cannot exceed 10 MB",
        )

    try:
        analysis = analyze_face(image_bytes)
    except InvalidImageError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except NoFaceDetectedError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except MultipleFacesDetectedError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except FaceProcessingError as error:
        raise HTTPException(
            status_code=503,
            detail="Face processing is unavailable",
        ) from error

    try:
        stored_embeddings = db.scalars(
            select(FaceEmbedding)
            .where(FaceEmbedding.model_name == MODEL_NAME)
            .options(joinedload(FaceEmbedding.person))
        ).all()
    except SQLAlchemyError as error:
        raise HTTPException(
            status_code=500,
            detail="Could not load familiar people",
        ) from error

    face = {
        "x": analysis.face.x,
        "y": analysis.face.y,
        "width": analysis.face.width,
        "height": analysis.face.height,
    }

    matched_embedding = find_matching_embedding(
        analysis.embedding,
        stored_embeddings,
    )

    if matched_embedding is None:
        return {
            "recognized": False,
            "face": face,
        }

    person = matched_embedding.person

    try:
        latest_conversation = db.scalars(
            select(Conversation)
            .where(Conversation.person_id == person.id)
            .order_by(
                Conversation.occurred_at.desc(),
                Conversation.id.desc(),
            )
            .limit(1)
        ).first()
    except SQLAlchemyError as error:
        raise HTTPException(
            status_code=500,
            detail="Could not load conversation history",
        ) from error

    last_conversation = None

    if latest_conversation is not None:
        last_conversation = {
            "summary": latest_conversation.summary,
            "follow_up": latest_conversation.follow_up,
            "occurred_at": latest_conversation.occurred_at.isoformat(),
        }

    return {
        "recognized": True,
        "person_id": person.id,
        "name": person.name,
        "relationship": person.relationship,
        "face": face,
        "last_conversation": last_conversation,
    }


@app.post(
    "/api/people/{person_id}/conversations",
    status_code=201,
)
def create_conversation(
    person_id: int,
    audio: Annotated[UploadFile, File()],
    db: Annotated[Session, Depends(get_db)],
    openai_service: Annotated[
        OpenAIService,
        Depends(get_openai_service),
    ],
) -> dict[str, object]:
    try:
        person = db.get(Person, person_id)
    except SQLAlchemyError as error:
        raise HTTPException(
            status_code=500,
            detail="Could not load the person",
        ) from error

    if person is None:
        raise HTTPException(
            status_code=404,
            detail="Person not found",
        )

    content_type = (audio.content_type or "").split(";")[0].lower()

    if content_type not in AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported audio type",
        )

    audio_bytes = audio.file.read(MAX_AUDIO_SIZE + 1)

    if not audio_bytes:
        raise HTTPException(
            status_code=400,
            detail="The audio recording is empty",
        )

    if len(audio_bytes) > MAX_AUDIO_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Audio cannot exceed 20 MB",
        )

    filename = f"conversation{AUDIO_EXTENSIONS[content_type]}"

    try:
        transcript = openai_service.transcribe_audio(
            audio_bytes,
            filename=filename,
            content_type=content_type,
        )
        summary = openai_service.summarize_transcript(transcript)
    except EmptyTranscriptionError as error:
        raise HTTPException(
            status_code=422,
            detail="No speech was detected",
        ) from error
    except OpenAIServiceError as error:
        raise HTTPException(
            status_code=503,
            detail="Conversation processing is unavailable",
        ) from error

    conversation = Conversation(
        person_id=person.id,
        summary=summary.summary,
        topics=summary.topics,
        follow_up=summary.follow_up,
        occurred_at=datetime.now(UTC).replace(tzinfo=None),
    )

    try:
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
    except SQLAlchemyError as error:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Could not save the conversation",
        ) from error

    return {
        "id": conversation.id,
        "person_id": conversation.person_id,
        "summary": conversation.summary,
        "topics": conversation.topics,
        "follow_up": conversation.follow_up,
        "occurred_at": conversation.occurred_at.isoformat(),
    }
