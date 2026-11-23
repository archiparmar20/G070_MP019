import numpy as np
import pandas as pd

from src.mp019.clean_categories import clean_categories


def _sample():
    return pd.DataFrame({
        "vehicle": ["car.", "two.", "THREE_WHEELER", " car", "two_wheeler", np.nan],
        "station_type": ["Mall", "Highway", "Residential", "Mall", "Highway", "Mall"],
    })


def test_fixes_wrong_spellings_keeps_correct_and_missing():
    out = clean_categories(_sample())
    assert out["vehicle"].tolist()[:5] == [
        "car", "two_wheeler", "three_wheeler", "car", "two_wheeler"]
    assert pd.isna(out["vehicle"].iloc[5])


def test_original_is_not_changed():
    df = _sample()
    before = df.copy()
    clean_categories(df)
    pd.testing.assert_frame_equal(df, before)


def test_other_columns_untouched():
    df = _sample()
    out = clean_categories(df)
    assert out["station_type"].tolist() == df["station_type"].tolist()
