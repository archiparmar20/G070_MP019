
import pandas as pd
from pathlib import Path

# Get project directory
ROOT = Path(__file__).resolve().parents[2]

# Load training dataset
DATA_PATH = ROOT / "data" / "raw" / "train.csv"
df = pd.read_csv(DATA_PATH)

print("DATASET INFORMATION")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# Check station_type
if "station_type" not in df.columns:
    raise ValueError("station_type column not found")

# Select numeric input columns
numeric_cols = df.select_dtypes(include="number").columns.tolist()
numeric_cols = [c for c in numeric_cols if c != "station_type"]

# UNIT MIX-UP CHECK
print("\n===== UNIT MIX-UP CHECK =====")

for col in numeric_cols:
    print("\nColumn:", col)

    stats = df.groupby("station_type")[col].agg(
        minimum="min",
        average="mean",
        maximum="max"
    )

    print(stats)

# SENTINEL VALUE CHECK
print("\n===== SENTINEL VALUE CHECK =====")

candidates = [-1, -999, 999, 9999]

for col in numeric_cols:
    counts = df[col].value_counts()

    for value in candidates:
        if value in counts.index:
            print(
                f"Column: {col}, "
                f"Value: {value}, "
                f"Count: {counts[value]}"
            )

print("\n===== AUDIT COMPLETED =====")


print("\n===== FINAL VERIFICATION =====")

# Check vehicle age values for Mall stations
mall_age = df.loc[
    df["station_type"] == "Mall",
    "vehicle_age_yr"
]

print("\nMall vehicle age value counts:")
print(mall_age.value_counts().sort_index().to_string())

# Check sentinel value count
print("\nSession minutes value counts:")
print(df["session_minutes"].value_counts().head(15))