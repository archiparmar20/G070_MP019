# Stage 1 – Numeric Data Quality Audit

**Audited by:** Vanshika Sharma  
**Repository:** G070_MP019  
**Branch:** stage1-vanshika-audit  
**Dataset:** train.csv

## 1. Unit Mix-up Audit

**Status: PRESENT**

### Finding

- **Column:** `vehicle_age_yr`
- **Affected station type:** `Mall`
- **Expected unit:** Years
- **Observed unit:** Months
- **Exact conversion factor:** 12

### Evidence

The grouped statistics showed:

| Station Type | Minimum | Average | Maximum |
|---|---:|---:|---:|
| Highway | 0.1 | 4.002439 | 8.0 |
| Mall | 1.2 | 47.818731 | 96.0 |
| Residential | 0.0 | 4.100785 | 8.0 |

Mall values are approximately 12 times the expected year-based scale.

The Mall values are consistent with vehicle age being recorded in months instead of years.

**Conversion:** Divide Mall values by 12 to convert months into years.

## 2. Sentinel Value Audit

**Status: PRESENT**

### Finding

- **Column:** `session_minutes`
- **Sentinel value:** `9999`
- **Count:** 33

### Evidence

Pandas `value_counts()` identified the value 9999 occurring 33 times in the original training dataset.

The column represents planned charging session length in minutes. A repeated value of 9999 is an implausible session duration and is consistent with an invalid or missing-value marker.

## 3. Methodology

The audit was performed using Python and pandas.

1. Loaded the original training dataset using `pd.read_csv()`.
2. Grouped numeric columns by `station_type`.
3. Calculated minimum, average and maximum values using pandas aggregation.
4. Used `value_counts()` to investigate unusual repeated numeric values.
5. Investigated the suspicious values against the project column definitions.

## 4. Conclusion

Two data quality issues were identified:

1. Unit mix-up in `vehicle_age_yr` for Mall stations, with a conversion factor of 12.
2. Sentinel value 9999 in `session_minutes`, occurring 33 times.

The original training CSV was kept local and was not committed to GitHub.

## 5. Files Submitted

- `src/mp019/audit_numeric.py`
- `reports/audit_vanshika.md`
