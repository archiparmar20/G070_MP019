"""Validation tests for POST /predict. Invalid requests are rejected with 422
before the endpoint runs, so no saved model or CSV file is needed."""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _valid_body(**overrides):
    body = {
        "battery_start_pct": 20,
        "session_minutes": 60,
        "charger_kw": 22.0,
        "temp_c": 25.0,
        "vehicle_age_yr": 2.0,
        "queue_before": 1,
        "vehicle": "car",
        "station_type": "Mall",
    }
    body.update(overrides)
    return body


def test_charger_kw_above_100_is_rejected():
    response = client.post("/predict", json=_valid_body(charger_kw=101))
    assert response.status_code == 422


def test_negative_vehicle_age_is_rejected():
    response = client.post("/predict", json=_valid_body(vehicle_age_yr=-1))
    assert response.status_code == 422


def test_extra_field_is_rejected():
    response = client.post("/predict", json=_valid_body(kwh_billed_in_app=15.0))
    assert response.status_code == 422
