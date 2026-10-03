from fastapi.testclient import TestClient

from remember_me.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_homepage() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "Remember Me" in response.text


def test_stylesheet() -> None:
    response = client.get("/static/styles.css")

    assert response.status_code == 200
    assert "text/css" in response.headers["content-type"]
