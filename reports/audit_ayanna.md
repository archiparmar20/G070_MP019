# Stage 1 audit: Ayanna (G070, MP019)

Script: `src/mp019/audit.py`. Full printed output: `reports/audit_ayanna_output.txt`.
Data: original `train.csv` (1000 rows), `later.csv` (300 rows), `holdout_features.csv` (250 rows).

| Check | Status |
|---|---|
| Leaking column | **PRESENT**: `kwh_billed_in_app` |
| Duplicate rows | **ABSENT** (0 extra copies) |
| Spelling variants | **PRESENT**: `vehicle` (3 real levels); `station_type` is clean |

## 1. Leaking column: PRESENT
- **Column:** `kwh_billed_in_app`
- **Evidence 1 (availability):** every train column except the target is also in `holdout_features.csv`, with one exception: `kwh_billed_in_app`. The data dictionary says it is the energy billed in the app *after the session*, so it is not known when the vehicle plugs in and the prediction has to be made.
- **Evidence 2 (closeness to target):** after excluding the 14 rows whose target looks 10x too big (target > 3x median), correlation with `energy_kwh` is **0.9989**. The median absolute difference is **0.31 kWh**, and **94.9%** of rows are within 1 kWh. (With the typo rows included, correlation drops to 0.265, which is the typo effect and not a weak link.)
- **Action:** do not use it as a feature.

## 2. Duplicate rows: ABSENT
- `train.duplicated().sum()` on the original file = **0**.
- Also 0 when ignoring `record_id`, and 0 repeated `record_id` values.

## 3. Spelling variants: PRESENT
- **Column:** `vehicle`. `station_type` has only Residential 382, Mall 331, Highway 287, which matches the clean spellings in `later.csv`.
- **Real categories (`true_levels`): 3**: `car`, `three_wheeler`, `two_wheeler` (as in `later.csv`).
- **Rows affected:** 150 of 1000 (17 wrong spellings).
- Wrong -> correct (rows):
  - `car`: `Car` (9), `CAR` (5), `car ` (9), ` car` (8), `car.` (8)
  - `three_wheeler`: `three_wheeler ` (14), ` three_wheeler` (10), `Three_Wheeler` (9), `THREE_WHEELER` (12), `three wheeler` (10), `thr.` (5)
  - `two_wheeler`: `two_wheeler ` (11), ` two_wheeler` (11), `Two_Wheeler` (5), `TWO_WHEELER` (10), `two wheeler` (7), `two.` (7)
- `thr.` and `two.` are truncated forms. They are matched by prefix, since each prefix fits only one clean category.

## Side note for the data lead
Not part of my checks, but seen while auditing: the 14 target values above 100 kWh (157.3 to 447.3, with a gap below 63.7) look like the 10x `target_typos`.
