"""FastAPI serving layer over otda.inference.

Thin wrapper: Pydantic validation in, otda.inference.predict() does
the actual work, Pydantic response out. Keeping the model logic in
inference.py (not here) is what lets Stage 9's SageMaker /invocations
adapter reuse it without duplicating a second implementation.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from otda.inference import PredictionResult, load_model, predict

_model_holder: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _model_holder["model"] = load_model()
    yield
    _model_holder.clear()


app = FastAPI(
    title="Orbital Threat Assessment API",
    description=(
        "Predicts whether an asteroid is a Potentially Hazardous "
        "Asteroid (PHA) from its orbital elements, using the "
        "Experiment 2 (MOID-included) Decision Tree — the model that "
        "reconstructs NASA's own two-condition PHA rule "
        "(MOID <= 0.05 AU and H <= 22.0)."
    ),
    lifespan=lifespan,
)


class AsteroidFeatures(BaseModel):
    e: float = Field(ge=0, lt=1, description="Eccentricity")
    q: float = Field(gt=0, description="Perihelion distance (AU)")
    H: float = Field(gt=-5, lt=40, description="Absolute magnitude")
    is_neo: bool = Field(description="Near-Earth object flag")
    class_code: int = Field(
        ge=0, le=12,
        description="Orbit class code (see dbt/seeds/asteroid_class_map.csv)",
    )
    moid: float = Field(ge=0, description="Minimum Orbit Intersection Distance (AU)")


class PredictionResponse(BaseModel):
    is_pha: bool
    probability: float
    explanation: str

    @classmethod
    def from_result(cls, result: PredictionResult) -> "PredictionResponse":
        return cls(
            is_pha=result.is_pha,
            probability=result.probability,
            explanation=result.explanation,
        )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "model" in _model_holder}


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(features: AsteroidFeatures) -> PredictionResponse:
    result = predict(_model_holder["model"], features.model_dump())
    return PredictionResponse.from_result(result)
