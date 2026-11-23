"""Stage 1 audit (Ayanna): leakage, duplicate rows, spelling variants.

Usage (from repo root):
    python -m src.mp019.audit --train data/raw/train.csv --later data/raw/later.csv \
        --holdout /path/to/extracted_zip/data/holdout_features.csv
"""
import argparse
import re

import pandas as pd

TARGET = "energy_kwh"
TEXT_COLS = ["vehicle", "station_type"]


def check_leakage(train, holdout):
    feats = [c for c in train.columns if c != TARGET]
    missing_in_holdout = [c for c in feats if c not in holdout.columns]
    print("== LEAKAGE ==")
    print("Train columns (excl. target) missing from holdout:", missing_in_holdout)
    res = {"missing_in_holdout": missing_in_holdout}
    for col in missing_in_holdout:
        d = pd.to_numeric(train[col], errors="coerce")
        y = pd.to_numeric(train[TARGET], errors="coerce")
        ok = d.notna() & y.notna()
        corr = d[ok].corr(y[ok])
        abs_err = (d[ok] - y[ok]).abs()
        # ignore the few 10x target typos when judging closeness
        typo = y[ok] > 3 * y[ok].median()  # likely 10x typos in the target
        print(f"{col}: corr with {TARGET} = {corr:.4f}")
        print(f"  mean |diff| = {abs_err.mean():.3f} kWh, median |diff| = {abs_err.median():.3f} kWh")
        print(f"  within 1 kWh: {(abs_err <= 1).mean():.1%} of rows")
        print(f"  rows with likely target typos excluded from the check below: {int(typo.sum())}")
        corr_clean = d[ok][~typo].corr(y[ok][~typo])
        print(f"  corr excluding those rows = {corr_clean:.4f}")
        print(f"  within 1 kWh excluding those rows: {(abs_err[~typo] <= 1).mean():.1%}")
        print(f"  median |diff| excluding likely target typos: {abs_err[~typo].median():.3f}")
        res[col] = dict(corr=corr, mean_abs=abs_err.mean(), median_abs=abs_err.median(),
                        within1=(abs_err <= 1).mean())
    return res


def check_duplicates(train):
    n = int(train.duplicated().sum())
    n_id = int(train["record_id"].duplicated().sum())
    print("\n== DUPLICATES ==")
    print("df.duplicated().sum() (extra copies, all columns):", n)
    print("Duplicates ignoring record_id:", int(train.drop(columns='record_id').duplicated().sum()))
    print("Duplicated record_id values:", n_id)
    if n:
        print(train[train.duplicated(keep=False)].sort_values("record_id").head(6).to_string())
    return n


def check_spelling(train, later):
    print("\n== SPELLING VARIANTS ==")
    out = {}
    for col in TEXT_COLS:
        print(f"\n-- {col} (train, raw repr) --")
        vc = train[col].map(repr).value_counts()
        print(vc.to_string())
        clean = sorted(later[col].dropna().unique())
        print(f"clean categories in later.csv ({len(clean)}):", clean)
        mapping = {}
        for raw in train[col].dropna().unique():
            if raw in clean:
                continue
            norm = re.sub(r"[^a-z]", "", raw.lower())
            match = [c for c in clean if re.sub(r"[^a-z]", "", c.lower()) == norm]
            if not match and norm:  # truncated forms like 'thr.' / 'two.'
                match = [c for c in clean if re.sub(r"[^a-z]", "", c.lower()).startswith(norm)]
                if len(match) != 1:
                    match = []
            mapping[raw] = match[0] if match else "?? (no exact match; check by hand)"
        counts = train[col].value_counts()
        print("wrong -> correct (rows):")
        for k, v in mapping.items():
            print(f"  {k!r} -> {v!r}  ({counts[k]} rows)")
        out[col] = dict(clean=clean, mapping=mapping,
                        affected_rows=int(sum(counts[k] for k in mapping)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/raw/train.csv")
    ap.add_argument("--later", default="data/raw/later.csv")
    ap.add_argument("--holdout", default="data/holdout_features.csv")
    a = ap.parse_args()
    train, later, holdout = (pd.read_csv(p) for p in (a.train, a.later, a.holdout))
    print("train", train.shape, "later", later.shape, "holdout", holdout.shape)
    check_leakage(train, holdout)
    check_duplicates(train)
    check_spelling(train, later)


if __name__ == "__main__":
    main()
