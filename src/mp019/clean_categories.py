"""Fix spelling variants in the categorical columns of train.csv.

Mapping comes from the Stage 1 audit (reports/audit_ayanna.md): 17 wrong
spellings of `vehicle`, all mapping to one of 3 real levels. `station_type`
had no variants, so it is left alone.
"""
import pandas as pd

VEHICLE_MAP = {
    # car
    "Car": "car", "CAR": "car", "car ": "car", " car": "car", "car.": "car",
    # three_wheeler
    "three_wheeler ": "three_wheeler", " three_wheeler": "three_wheeler",
    "Three_Wheeler": "three_wheeler", "THREE_WHEELER": "three_wheeler",
    "three wheeler": "three_wheeler", "thr.": "three_wheeler",
    # two_wheeler
    "two_wheeler ": "two_wheeler", " two_wheeler": "two_wheeler",
    "Two_Wheeler": "two_wheeler", "TWO_WHEELER": "two_wheeler",
    "two wheeler": "two_wheeler", "two.": "two_wheeler",
}


def clean_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of df; the input is never modified.

    Correct spellings, unknown values and missing values (NaN) are kept as is.
    """
    out = df.copy()
    if "vehicle" in out.columns:
        out["vehicle"] = out["vehicle"].replace(VEHICLE_MAP)
    return out
