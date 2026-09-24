"""
API tests using FastAPI's TestClient (in-process, no server needed).

The `with TestClient(app) as client:` form is required so the lifespan
handler runs and the model is actually loaded.

Run from the project root:
    pytest tests/ -v
"""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:   # lifespan -> model loads here
        yield c


def make_image_bytes(fmt="JPEG", color=(200, 30, 30), size=(128, 128)):
    """A valid image generated in memory — no test fixtures on disk."""
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format=fmt)
    buf.seek(0)
    return buf.getvalue()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert len(body["classes"]) == 5
    assert "daisy" in body["classes"]


def test_predict_valid_image(client):
    files = {"file": ("flower.jpg", make_image_bytes(), "image/jpeg")}
    r = client.post("/predict", files=files)
    assert r.status_code == 200
    body = r.json()
    # contract check: schema fields present and sane
    assert body["class_name"] in {"daisy", "dandelion", "rose", "sunflower", "tulip"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["probabilities"].keys()) == set(
        ["daisy", "dandelion", "rose", "sunflower", "tulip"])
    # probabilities must sum to ~1
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-3


def test_wrong_content_type_rejected(client):
    files = {"file": ("notes.txt", b"not an image", "text/plain")}
    r = client.post("/predict", files=files)
    assert r.status_code == 415


def test_garbage_bytes_rejected(client):
    # right MIME type, undecodable body -> 400 (never crash the server)
    files = {"file": ("fake.jpg", b"\xff\xd8\xff garbage", "image/jpeg")}
    r = client.post("/predict", files=files)
    assert r.status_code == 400


def test_index_page_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Flower Classifier" in r.text
