import pandas as pd
NUMERIC_FEATURES = [
    "battery_start_pct",
    "session_minutes",
    "charger_kw",
    "temp_c",
    "vehicle_age_yr",
    "queue_before",
]

CATEGORICAL_FEATURES = ["vehicle", "station_type"]
TARGET = "energy_kwh"


def validate_training_data(df):
    required_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    for column in NUMERIC_FEATURES:
        original = df[column]
        numeric = pd.to_numeric(original, errors="coerce")

        if (original.notna() & numeric.isna()).any():
            raise ValueError(f"Invalid numeric value in {column}")

    target = pd.to_numeric(df[TARGET], errors="coerce")
    if target.isna().any():
        raise ValueError(f"{TARGET} contains missing or non-numeric values")