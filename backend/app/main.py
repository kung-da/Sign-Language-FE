from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .model import ModelUnavailableError, get_predictor, load_baseline_metadata


app = FastAPI(title="Sign Language Model Test API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    sequence: Any = Field(..., description="Keypoints shaped [60, 634] or [batch, 60, 634].")
    top_k: int = Field(default=5, ge=1, le=50)


class BenchmarkRequest(BaseModel):
    runs: int = Field(default=30, ge=1, le=500)
    warmup: int = Field(default=5, ge=0, le=100)


@app.get("/health")
def health() -> dict[str, Any]:
    metadata = load_baseline_metadata()
    return {
        "status": "ok",
        "checkpoint_exists": metadata["checkpoint_exists"],
        "torch_available": metadata["runtime"]["torch_available"],
    }


@app.get("/model/metadata")
def model_metadata() -> dict[str, Any]:
    return load_baseline_metadata()


@app.post("/model/predict")
def predict(payload: PredictRequest) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        predictor = get_predictor()
        result = predictor.predict(payload.sequence, top_k=payload.top_k)
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "top_k": result.top_k,
        "batch_size": result.batch_size,
        "model_inference_ms": result.inference_ms,
        "end_to_end_ms": (time.perf_counter() - start) * 1000,
    }


@app.post("/model/benchmark")
def benchmark(payload: BenchmarkRequest) -> dict[str, Any]:
    try:
        predictor = get_predictor()
        return predictor.benchmark(runs=payload.runs, warmup=payload.warmup)
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
