from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from remember_me.database import engine
from remember_me.models import Person

WEB_DIR = Path(__file__).parent / "web"


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
