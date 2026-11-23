"""Stage 1 audit (Atharv): stuck sensor check.

A stuck sensor shows up as the same reading repeated in many consecutive
rows once the data is put in time order. This script:

1. converts `date` to a date type and sorts the rows by date,
2. for every numeric input column, finds runs of the identical value in
   consecutive rows,
3. for the longest runs, reports how many rows and how many distinct dates
   they cover, and whether readings just before and just after the run vary
   (a real sensor reading should vary around a stuck stretch).

Usage (from repo root):
    python -m src.mp019.audit_sensor --train data/raw/train.csv
"""
import argparse

import pandas as pd

DATE_COL = "date"
TARGET = "energy_kwh"
# not inputs: the target, the row id, and the column filled in after the session
NOT_INPUTS = (TARGET, "record_id", "kwh_billed_in_app")
NUMERIC_FEATURES = [
    "battery_start_pct",
    "session_minutes",
    "charger_kw",
    "temp_c",
    "vehicle_age_yr",
    "queue_before",
]
CONTEXT_ROWS = 10  # rows to show either side of a run


def load_sorted(path):
    df = pd.read_csv(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="raise").dt.normalize()
    # stable sort keeps the file order for rows sharing a date
    return df.sort_values(DATE_COL, kind="mergesort").reset_index(drop=True)


def find_runs(series):
    """Return a DataFrame of runs of identical consecutive values."""
    s = series.reset_index(drop=True)
    # NaN != NaN, so treat missing values as their own breaks
    run_id = (s != s.shift()).cumsum()
    runs = (
        pd.DataFrame({"value": s, "run_id": run_id, "pos": range(len(s))})
        .dropna(subset=["value"])
        .groupby("run_id")
        .agg(value=("value", "first"), start=("pos", "min"),
             end=("pos", "max"), rows=("pos", "size"))
        .reset_index(drop=True)
    )
    return runs.sort_values("rows", ascending=False, kind="mergesort")


def describe_run(df, col, run):
    start, end = int(run.start), int(run.end)
    block = df.iloc[start:end + 1]
    before = df[col].iloc[max(0, start - CONTEXT_ROWS):start]
    after = df[col].iloc[end + 1:end + 1 + CONTEXT_ROWS]
    return {
        "column": col,
        "value": run.value,
        "rows": int(run.rows),
        "start_date": block[DATE_COL].min().strftime("%Y-%m-%d"),
        "end_date": block[DATE_COL].max().strftime("%Y-%m-%d"),
        "distinct_dates": int(block[DATE_COL].nunique()),
        "before_unique": int(before.nunique()),
        "after_unique": int(after.nunique()),
        "before": before.tolist(),
        "after": after.tolist(),
        "start_row": start,
        "end_row": end,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/raw/train.csv")
    ap.add_argument("--top", type=int, default=3, help="runs to list per column")
    a = ap.parse_args()

    df = load_sorted(a.train)
    print("rows:", len(df), "| date range:",
          df[DATE_COL].min().date(), "to", df[DATE_COL].max().date(),
          "| distinct dates:", df[DATE_COL].nunique())

    cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    extra = [c for c in df.select_dtypes("number").columns
             if c not in cols and c not in NOT_INPUTS]
    cols += extra  # any other numeric inputs present in the file

    print("\n== LONGEST RUNS OF THE SAME VALUE (consecutive rows, sorted by date) ==")
    summary = []
    for col in cols:
        runs = find_runs(df[col])
        n_unique = df[col].nunique()
        print(f"\n-- {col} (distinct values: {n_unique}) --")
        print(runs.head(a.top).to_string(index=False))
        summary.append(describe_run(df, col, runs.iloc[0]))

    summary.sort(key=lambda r: r["rows"], reverse=True)
    print("\n== LONGEST RUN PER COLUMN ==")
    print(pd.DataFrame(summary)[["column", "value", "rows", "distinct_dates",
                                 "start_date", "end_date",
                                 "before_unique", "after_unique"]].to_string(index=False))

    top = summary[0]
    second = summary[1]["rows"] if len(summary) > 1 else 0
    print("\n== MOST SUSPICIOUS RUN ==")
    for k in ("column", "value", "rows", "distinct_dates", "start_date",
              "end_date", "start_row", "end_row"):
        print(f"{k}: {top[k]}")
    print(f"{CONTEXT_ROWS} readings before:", top["before"])
    print(f"{CONTEXT_ROWS} readings after: ", top["after"])
    print("next-longest run in any column:", second, "rows")

    print("\nRows in the run (with neighbours):")
    lo = max(0, top["start_row"] - 3)
    hi = top["end_row"] + 4
    print(df.iloc[lo:hi][[DATE_COL, top["column"]]].to_string())


if __name__ == "__main__":
    main()
