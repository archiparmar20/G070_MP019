# Stage 1: Stuck Sensor Audit

**Auditor:** Atharv Arora
**Branch:** stage1-atharv-audit
**Dataset:** Original `train.csv` (1000 rows, 2026-01-01 to 2026-06-30, 181 dates; MD5 `c227a345ae95f6bed694d9237adeef25`, matching `data/raw/train.csv.dvc`)
**Script:** `src/mp019/audit_sensor.py`

## Finding

**Status: ABSENT**

No numeric input column shows the same reading on consecutive rows for a long run of days. The longest run in any column is **3 rows**, all on a single date. Readings before and after it vary normally.

## Method

1. Converted `date` to a date type with `pd.to_datetime` and sorted the rows by date using a stable sort. The file was already in date order.
2. For each numeric input (`battery_start_pct`, `session_minutes`, `charger_kw`, `temp_c`, `vehicle_age_yr`, `queue_before`), found every run of the identical value in consecutive rows.
3. For the longest run in each column, counted its rows and distinct dates, and checked whether the 10 readings before and after it vary.
4. Robustness checks:
   - Re-ran the run search with the `session_minutes` sentinel `9999` removed, and with Mall `vehicle_age_yr` divided by 12 to undo the unit mix-up. Neither hides a longer run; the longest stays 3 rows.
   - Checked whether any column's most frequent value is concentrated in one block of dates. None is; the top values are spread across the whole six months.
   - The columns have no missing values that could split a run.

`record_id`, `energy_kwh` (target) and `kwh_billed_in_app` (the leaking column) are not inputs, so they were not audited.

## Evidence

Longest run of the same value per column, after sorting by date:

| Column | Repeated value | Rows | Dates spanned | Start date | End date | Distinct values in 10 rows before / after |
|---|---:|---:|---:|---|---|---|
| `battery_start_pct` | 56 | 3 | 1 | 2026-03-30 | 2026-03-30 | 9 / 9 |
| `queue_before` | 8 | 3 | 2 | 2026-01-07 | 2026-01-08 | 7 / 6 |
| `session_minutes` | 183 | 2 | 2 | 2026-01-19 | 2026-01-20 | 10 / 10 |
| `charger_kw` | 46.0 | 2 | 1 | 2026-03-05 | 2026-03-05 | 10 / 10 |
| `temp_c` | 31.1 | 2 | 2 | 2026-01-04 | 2026-01-05 | 10 / 10 |
| `vehicle_age_yr` | 2.8 | 2 | 1 | 2026-01-05 | 2026-01-05 | 9 / 10 |

The most suspicious run is `battery_start_pct` = 56 for 3 rows on 2026-03-30:

- 10 readings before: 32, 44, 74, 80, 15, 66, 25, 20, 20, 26
- 10 readings after: 39, 35, 33, 43, 25, 43, 38, 64, 62, 37

Three equal values in a row is normal for a whole-number percentage with only 76 distinct values. `queue_before` has just 11 possible values, so 3-row repeats are expected there too. A stuck sensor would repeat one value on every row for weeks (dozens of rows); nothing like that exists in this data.

## Conclusion

Stuck sensor: **absent**. For `reports/findings.yaml`:

```yaml
stuck_sensor:
  present: false
  column: null
  start_date: null
  end_date: null
```

The original `train.csv` was kept local and was not committed.
