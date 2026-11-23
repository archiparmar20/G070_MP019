
import pandas as pd
from pathlib import Path

# Only the original training CSV is required
DATA_PATH = Path("data/raw/train.csv")
TARGET = "energy_kwh"


def main():
    if not DATA_PATH.exists():
        print(f"ERROR: Cannot find {DATA_PATH}")
        print("Make sure train.csv is inside data/raw/")
        return

    df = pd.read_csv(DATA_PATH)

    if TARGET not in df.columns:
        print(f"ERROR: Column '{TARGET}' not found.")
        return

    values = pd.to_numeric(df[TARGET], errors="coerce").dropna()

    sorted_values = values.sort_values().reset_index(drop=True)

    print("\n========== ENERGY TARGET AUDIT ==========")
    print("Total valid values:", len(sorted_values))

    print("\n--- 30 LARGEST VALUES ---")
    print(sorted_values.tail(30).iloc[::-1].to_string(index=False))

    # Calculate neighbouring gaps
    gaps = sorted_values.diff()

    gap_table = pd.DataFrame({
        "lower": sorted_values.shift(1),
        "upper": sorted_values,
        "gap": gaps
    })

    print("\n--- 20 LARGEST NEIGHBOURING GAPS ---")
    print(
        gap_table.nlargest(20, "gap").to_string(index=False)
    )

    # Inspect the largest gap
    largest_gap_index = gaps.idxmax()

    lower = sorted_values.iloc[largest_gap_index - 1]
    upper = sorted_values.iloc[largest_gap_index]
    gap = upper - lower

    threshold = (lower + upper) / 2

    suspicious = values[values > threshold]

    print("\n--- SUSPICIOUS GROUP ---")
    print(f"Lower boundary: {lower}")
    print(f"Upper boundary: {upper}")
    print(f"Largest gap: {gap}")
    print(f"Candidate threshold: {threshold}")
    print(f"Candidate affected rows: {len(suspicious)}")

    print("\n--- ORIGINAL vs DIVIDED BY 10 ---")

    for index, value in suspicious.items():
        print(
            f"Row {index + 2}: "
            f"Original={value}, "
            f"Divided by 10={value / 10}"
        )

    print("\n========== AUDIT COMPLETE ==========")
    print("Original CSV was not modified.")


if __name__ == "__main__":
    main()