# Challenge 4: Loan Default Prediction

## Overview

A lending team needs to predict which loans will default and when. Early default (within the first year or two) is especially costly because the bank has already funded the loan but recovers little. Your task is to build a model that predicts (1) **days until early default** (regression) and (2) **whether the loan will default at all** (binary classification).

This is a **dual-target problem** similar to cash flow shortfall: regression (days) is primary; classification (default yes/no) is secondary.

---

## Description

You are given three related tables: borrower profiles, loan terms, and payment history. These tables are linked by `borrower_id` and `loan_id` and must be **joined, aggregated, and feature-engineered** before modeling.

### Key challenges:
- **Payment history requires aggregation.** Each loan has up to 12 monthly payment records. You must decide how to summarize them (on-time rate? late count? trend?).
- **Not all borrowers have the same number of loans.** Some have one; some have multiple. Aggregate borrower-level features carefully.
- **Missing values and messy data.** Nulls in financial columns, inconsistent formatting in categoricals.
- **Default rate is ~18%.** Moderately imbalanced classification.
- **Days-to-default is censored for non-defaulters.** Non-defaulting loans have `days_to_early_default` equal to the loan term (e.g. 1095 for a 36-month loan).

---

## Data Dictionary

### borrowers.csv (~71k rows)

| Column | Type | Description |
|--------|------|-------------|
| `borrower_id` | string | **Primary key.** Unique borrower identifier |
| `credit_score` | int | Credit score at origination (300–850) |
| `annual_income` | float | Annual income ($) |
| `debt_to_income_ratio` | float | DTI at origination (%) |
| `employment_years` | int | Years in current job |
| `num_existing_loans` | int | Number of other loans at origination |
| `previous_defaults` | int | Count of prior defaults |
| `months_since_last_inquiry` | int | Months since last credit inquiry |
| `city` | string | Borrower city |
| `state` | string | Borrower state |

### loans.csv (~182k rows)

| Column | Type | Description |
|--------|------|-------------|
| `loan_id` | string | **Primary key.** Unique loan identifier |
| `borrower_id` | string | **Foreign key** to borrowers.csv |
| `principal` | float | Loan amount ($) |
| `interest_rate` | float | Annual interest rate (%) |
| `term_months` | int | Loan term in months (12, 24, 36, 48, 60) |
| `loan_type` | string | personal, auto, home_equity, etc. |
| `monthly_payment` | float | Monthly payment amount ($) |
| `payment_to_income_ratio` | float | Monthly payment / monthly income |
| `origination_date` | date | Loan origination date |

### payment_history.csv (~628k rows)

| Column | Type | Description |
|--------|------|-------------|
| `loan_id` | string | **Foreign key** to loans.csv |
| `month_number` | int | Months since origination (1–12) |
| `payment_due_date` | date | Due date for the payment |
| `amount_due` | float | Scheduled payment amount |
| `amount_paid` | float | Amount actually paid |
| `days_delinquent` | int | Days past due (0 = on time) |

**Note:** Each loan has up to 12 monthly records. Aggregate to get: on-time rate (days_delinquent == 0), total days delinquent, late payment count, payment ratio (amount_paid / amount_due).

### train.csv (~126k rows)

| Column | Type | Description |
|--------|------|-------------|
| `loan_id` | string | Loan identifier |
| `days_to_early_default` | int | **Primary target (regression).** Days from origination to default. For non-defaulters, equals full term in days. |
| `default_flag` | int | **Secondary target (classification).** 1 = defaulted, 0 = did not default. |

### eval.csv (~54k rows)

| Column | Type | Description |
|--------|------|-------------|
| `loan_id` | string | Loan identifier (predict for these) |

---

## Evaluation Criteria

### Track 1: Regression (primary)

Predict `days_to_early_default` for each loan in `eval.csv`.

| Metric | Description |
|--------|-------------|
| **RMSE** | Root Mean Squared Error (in days) |
| **MAE** | Mean Absolute Error (in days) |
| **R-squared** | Proportion of variance explained |

### Track 2: Classification (secondary)

Predict `default_flag` for each loan in `eval.csv`.

| Metric | Description |
|--------|-------------|
| **AUC-ROC** | Discrimination between default and non-default |
| **F1 Score** | Balanced precision-recall for the default class |

### Baseline Benchmarks

See `benchmark.txt`. Linear Regression achieves R² ~0.21 on days. Logistic Regression achieves AUC ~0.82. These are baselines on clean flat data—your scores on messy multi-table data may initially be lower.

---

## Submission Format

Submit a CSV file with **exactly three columns**, one row per `loan_id` in `eval.csv`:

```csv
loan_id,predicted_days_to_default,predicted_default_flag
L12345-67890,1095,0
L98765-43210,180,1
...
```

- `loan_id` must match the IDs in `eval.csv`
- `predicted_days_to_default` is an integer (days; for non-defaulters use full term, e.g. 1095)
- `predicted_default_flag` is 0 or 1

---

## Tips

1. **Aggregate payment history.** On-time rate, max days late, count of missed payments, and trend over months are likely predictive.
2. **Join borrowers + loans + payment aggregates.** Use `loan_id` as the primary key for your modeling table.
3. **Handle non-defaulters correctly.** For regression, predicting the full term (e.g. 36 months = 1095 days) for loans that don't default is correct.
4. **Feature engineering.** Payment-to-income ratio, DTI, credit score, and loan type are classic default predictors.
5. **Watch for data quality.** Inconsistent status values, orphaned loan_ids, nulls in key columns.
