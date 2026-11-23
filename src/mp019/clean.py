import pandas as pd
from .clean_categories import clean_categories


def clean_train(df, config):
    out = clean_categories(df.copy())

    cleaning = config["cleaning"]

    age = cleaning["vehicle_age"]
    mask = out["station_type"] == age["station_type"]

    out.loc[mask, "vehicle_age_yr"] = (
        out.loc[mask, "vehicle_age_yr"] /
        age["conversion_factor"]
    )

    sentinel = cleaning["session_minutes"]["sentinel_value"]

    out["session_minutes"] = out["session_minutes"].replace(
        sentinel, float("nan")
    )

    return out