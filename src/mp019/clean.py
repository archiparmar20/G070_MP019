
import pandas as pd

# Import Ayanna's category cleaning function
try:
    from .clean_categories import clean_categories
except ImportError:
    from mp019.clean_categories import clean_categories


# Exactly 8 permitted model input columns
MODEL_INPUT_COLUMNS = [
    "battery_start_pct",
    "session_minutes",
    "charger_kw",
    "temp_c",
    "vehicle_age_yr",
    "queue_before",
    "vehicle",
    "station_type"
]


def clean_train(df, config):
    """
    Clean training data without modifying the original DataFrame.

    Cleaning operations:
    1. Fix vehicle category spellings.
    2. Convert Mall vehicle age from months to years.
    3. Replace sentinel value 9999 in session_minutes.
    4. Correct energy_kwh values above the configured threshold.
    """

    # Always create a copy
    out = df.copy(deep=True)

    # 1. Fix vehicle category spellings
    out = clean_categories(out)

    # 2. Mall vehicle age conversion
    age_config = config["cleaning"]["vehicle_age"]

    affected_station = age_config["station_type"]
    conversion_factor = age_config["conversion_factor"]

    mall_mask = (
        out["station_type"].astype("string").str.strip().str.lower()
        == affected_station.lower()
    )

    out.loc[mall_mask, "vehicle_age_yr"] = (
        pd.to_numeric(
            out.loc[mall_mask, "vehicle_age_yr"],
            errors="coerce"
        ) / conversion_factor
    )

    # 3. Replace sentinel value in session_minutes
    sentinel_config = config["cleaning"]["session_minutes"]

    sentinel_value = sentinel_config["sentinel_value"]

    out["session_minutes"] = out["session_minutes"].replace(
        sentinel_value, pd.NA
    )

    # 4. Harshit's target correction
    target_config = config["target_fix"]

    threshold = target_config["threshold"]
    factor = target_config["factor"]

    energy_values = pd.to_numeric(
        out["energy_kwh"], errors="coerce"
    )

    target_mask = energy_values > threshold

    corrected_count = int(target_mask.sum())

    out.loc[target_mask, "energy_kwh"] = (
        energy_values.loc[target_mask] / factor
    )

    print("Target rows corrected:", corrected_count)

    # Return cleaned copy
    return out
