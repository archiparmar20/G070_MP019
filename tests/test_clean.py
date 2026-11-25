"""Tests for clean_train, using small DataFrames built here (no CSV needed)."""
import pandas as pd

from src.mp019.clean import clean_train

# Same values as the cleaning section of params.yaml
CONFIG = {
    "cleaning": {
        "vehicle_age": {"station_type": "Mall", "conversion_factor": 12},
        "session_minutes": {"sentinel_value": 9999},
    },
    "target_fix": {"threshold": 110.485, "factor": 10},
}


def _rows(**overrides):
    """Two sample rows (Mall, Highway) with every train.csv column filled in."""
    data = {
        "record_id": ["T001", "T002"],
        "date": ["2026-01-01", "2026-01-02"],
        "battery_start_pct": [20, 40],
        "session_minutes": [60, 90],
        "charger_kw": [22.0, 50.0],
        "temp_c": [25.0, 30.0],
        "vehicle_age_yr": [2.0, 4.0],
        "queue_before": [1, 2],
        "vehicle": ["car", "two_wheeler"],
        "station_type": ["Mall", "Highway"],
        "kwh_billed_in_app": [15.0, 20.0],
        "energy_kwh": [15.0, 20.0],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def _by_id(df):
    return df.set_index("record_id")


def test_mall_vehicle_age_months_converted_to_years():
    out = _by_id(clean_train(_rows(vehicle_age_yr=[24, 4]), CONFIG))
    assert out.loc["T001", "vehicle_age_yr"] == 2   # Mall: 24 months -> 2 years
    assert out.loc["T002", "vehicle_age_yr"] == 4   # Highway: already years


def test_session_minutes_sentinel_becomes_missing():
    out = _by_id(clean_train(_rows(session_minutes=[9999, 90]), CONFIG))
    assert pd.isna(out.loc["T001", "session_minutes"])  # 9999 -> missing
    assert out.loc["T002", "session_minutes"] == 90     # normal value kept
