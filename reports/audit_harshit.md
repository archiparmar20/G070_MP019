# Stage 1: Target Column Typo Audit

**Auditor:** Harshit Singh  
**Column audited:** `energy_kwh`  
**Dataset:** Original `train.csv`

## 1. Methodology

1. Loaded the original training dataset.
2. Sorted `energy_kwh` values in ascending order.
3. Examined the 30 largest target values.
4. Calculated gaps between neighbouring sorted values.
5. Identified unusually high values separated from the main distribution by a large gap.
6. Checked whether dividing the suspicious values by 10 produced plausible values.

## 2. Finding

**Status: PRESENT**

**Number of suspicious rows: 14**

A large gap of **93.63** was identified between:

- Lower value: 63.67
- Upper value: 157.30

The midpoint threshold was calculated as:

`(63.67 + 157.30) / 2 = 110.485`

All 14 values above this threshold were flagged for further investigation.

## 3. Evidence

The suspicious values ranged from 157.30 to 447.30.

After dividing these values by 10, the corrected range became:

**15.73 to 44.73**

These values fall within the existing target distribution, making a possible 10x scaling error plausible.

The 14 suspicious CSV data rows were:

30, 48, 51, 205, 270, 394, 433, 521, 530, 615, 725, 770, 930, 982.

## 4. Conclusion

The audit identified 14 suspicious `energy_kwh` values that may represent 10x scaling errors.

The finding is marked PRESENT based on the unusually large neighbouring gap and the plausibility of the values after division by 10.

The original `train.csv` was inspected without modification and remains local. No CSV changes were made.