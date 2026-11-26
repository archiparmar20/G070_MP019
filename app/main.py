"""FastAPI service for MP019 EV charging energy prediction."""
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.prediction_logging import log_prediction

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.joblib"

app = FastAPI(title="MP019 EV charging energy")


class SessionRequest(BaseModel):
    """The eight model inputs, known when the vehicle plugs in."""

    model_config = ConfigDict(extra="forbid")  # unknown fields -> 422

    battery_start_pct: float
    session_minutes: float
    charger_kw: float = Field(..., le=100)
    temp_c: float
    vehicle_age_yr: float = Field(..., ge=0)
    queue_before: int
    vehicle: str
    station_type: str


@lru_cache(maxsize=1)
def load_model():
    """Load the saved pipeline once, on the first prediction (not at import)."""
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model file not found")
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    """Liveness check: needs no model or data files."""
    return {"status": "ok"}


@app.post("/predict")
def predict(request: SessionRequest):
    """Predict the energy delivered in the session, in kWh."""
    row = pd.DataFrame([request.model_dump()])
    prediction = float(load_model().predict(row)[0])
    log_prediction(prediction)  # Ayanna's JSON logger: one line per success
    return {"prediction": prediction}
