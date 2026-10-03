from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated


from pathlib import Path
from fastapi import FastAPI, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, joinedload
from remember_me.database import engine, get_db
from remember_me.models import Person, FaceEmbedding
from remember_me.face_service import (
    MODEL_NAME,
    FaceProcessingError,
    InvalidImageError,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
    analyze_face,
    generate_face_embedding,
)
from remember_me.recognition_service import find_matching_embedding

WEB_DIR = Path(__file__).parent / "web"
MAX_IMAGE_SIZE = 10 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    Person.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="Remember Me",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


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

    return {
        "recognized": True,
        "name": person.name,
        "relationship": person.relationship,
        "face": face,
    }
