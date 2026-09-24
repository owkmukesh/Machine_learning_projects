"""
FastAPI inference service.

Design notes (interview talking points):
    * The model is loaded ONCE in the lifespan handler and stored on
      app.state -> weights are never re-read per request.
    * Input is validated in layers: declared MIME type -> size cap ->
      actual decodability. Never trust the client.
    * The response schema is a Pydantic model -> automatic validation,
      serialization, and Swagger documentation.
    * torch runs on CPU here; swap in a device check to serve on GPU.
"""

import io
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image

from app.model import FlowerPredictor

MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
MAX_FILE_BYTES = 10 * 1024 * 1024            # 10 MB upload cap
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
STATIC_DIR = Path(__file__).resolve().parent / "static"


class PredictionResponse(BaseModel):
    class_name: str
    confidence: float
    probabilities: dict[str, float]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once at startup: expensive work (loading weights) happens here.
    app.state.predictor = FlowerPredictor(MODEL_DIR)
    yield


app = FastAPI(title="Flower Classifier API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    ok = hasattr(app.state, "predictor")
    return {"status": "ok" if ok else "model not loaded"}


@app.get("/model-info")
def model_info():
    p = app.state.predictor
    return {"classes": p.classes,
            "transform": str(p.transform).replace("\n", " ")}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # -- validation layer 1: declared content type --
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. "
                   f"Send JPEG, PNG or WebP.")

    data = await file.read()

    # -- validation layer 2: size cap (protects memory) --
    if len(data) == 0 or len(data) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413,
                            detail="Empty or oversized file (max 10 MB).")

    # -- validation layer 3: actually decodable as an image --
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400,
                            detail="File could not be decoded as an image.")

    return app.state.predictor.predict(img)


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")
