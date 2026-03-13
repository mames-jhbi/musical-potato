# Challenge 3: Small Business Cash Flow Shortfall Prediction

## Overview

A community bank's small business lending team needs to proactively identify which business customers are at risk of a cash flow shortfall in the next 30 days. Early identification allows bankers to reach out with credit line adjustments, payment restructuring, or advisory services -- before the business misses a payment or bounces a check.

This is a **dual-target problem**: predict both the **dollar amount** of the expected shortfall (regression) and whether a shortfall will occur at all (binary classification). The regression target is primary; the binary flag is secondary.

---

## Description

You are given three related tables: business profiles, monthly financial snapshots, and loan records. These tables are linked by `business_id` and must be **joined, aggregated, and feature-engineered** before modeling.

### Key challenges:
- **Financials require aggregation.** Each business has ~3 monthly snapshots. You must decide how to summarize them (latest values? trends? averages?).
- **Not all businesses have loans.** The loans table only contains rows for businesses with active debt. A LEFT JOIN will produce NULLs for ~40% of businesses -- you need to handle this.
- **Heavy missing data.** Financial columns have ~15% null values. Business profiles have ~10%.
- **Noise features.** Several columns are not predictive. Feature selection matters.
- **This is a hard regression problem.** The baseline Linear Regression achieves only R-squared ~0.34 on clean data. With messy multi-table data, expect lower out of the box.

---

## Data Dictionary

### businesses.csv (~202k rows)

| Column | Type | Description |
|--------|------|-------------|
| `business_id` | string | **Primary key.** Unique business identifier |
| `business_name` | string | Business name |
| `owner_name` | string | Business owner name |
| `industry` | string | Industry sector: retail, restaurant, construction, healthcare, professional_services, manufacturing, transportation, tech_services |
| `city` | string | Business city |
| `state` | string | Business state |
| `years_in_business` | int | Years since business was established |
| `num_employees` | int | Number of employees |

### financials.csv (~606k rows)

| Column | Type | Description |
|--------|------|-------------|
| `business_id` | string | **Foreign key** to businesses.csv |
| `reporting_month` | string | Month of the financial snapshot (2025-10, 2025-11, or 2025-12) |
| `monthly_revenue` | float | Revenue for the month ($) |
| `monthly_expenses` | float | Expenses for the month ($) |
| `accounts_receivable` | float | Outstanding amounts owed to the business ($) |
| `accounts_payable` | float | Outstanding amounts the business owes ($) |
| `cash_on_hand` | float | Available cash ($) |

**Note:** Each business has up to 3 monthly records. The most recent month (2025-12) is the closest to the prediction window. Consider computing:
- Latest-month values
- Month-over-month trends (is revenue growing or shrinking?)
- Averages or min/max across months

### loans.csv (~157k rows)

| Column | Type | Description |
|--------|------|-------------|
| `loan_id` | string | **Primary key.** Unique loan identifier |
| `business_id` | string | **Foreign key** to businesses.csv |
| `loan_type` | string | Loan category: term_loan, line_of_credit, sba_loan, equipment_financing |
| `outstanding_balance` | float | Current loan balance ($) |
| `monthly_payment` | float | Monthly payment amount ($) |
| `credit_line_utilization_pct` | float | Percent of credit line currently used (0-100) |
| `interest_rate` | float | Annual interest rate (%) |

**Important:** Not every business has a loan. ~40% of businesses will have no rows in this table. When joining, this means:
- Use a **LEFT JOIN** from businesses to loans
- NULL loan columns likely mean the business has no debt (impute accordingly)
- Some businesses have multiple loans (aggregate: total balance, total payment, max utilization, etc.)

### train.csv (140k rows)

| Column | Type | Description |
|--------|------|-------------|
| `business_id` | string | Business identifier |
| `cashflow_shortfall_amount` | float | **Primary target (regression).** Dollar amount of projected shortfall. Negative = shortfall, positive = healthy. |
| `shortfall_flag` | int | **Secondary target (classification).** 1 = shortfall expected, 0 = no shortfall. |

### eval.csv (60k rows)

| Column | Type | Description |
|--------|------|-------------|
| `business_id` | string | Business identifier (predict for these) |

---

## Evaluation Criteria

This challenge has **two evaluation tracks**:

### Track 1: Regression (primary)

Predict `cashflow_shortfall_amount` for each business in `eval.csv`.

| Metric | Description |
|--------|-------------|
| **RMSE** | Root Mean Squared Error -- penalizes large errors heavily |
| **MAE** | Mean Absolute Error -- average magnitude of errors |
| **R-squared** | Proportion of variance explained (higher is better; 1.0 is perfect) |

### Track 2: Classification (secondary)

Predict `shortfall_flag` for each business in `eval.csv`.

| Metric | Description |
|--------|-------------|
| **AUC-ROC** | Discrimination between shortfall and non-shortfall businesses |
| **F1 Score** | Balanced precision-recall for the shortfall class |

### Baseline Benchmarks

See `benchmark.txt`. Linear Regression achieves R-squared ~0.34. Logistic Regression achieves AUC ~0.78. These are baselines on clean flat data -- your scores on the messy multi-table data may initially be lower. The goal is to match or exceed these through strong data preparation and feature engineering.

---

## Submission Format

Submit a CSV file with **exactly three columns**, one row per `business_id` in `eval.csv`:

```csv
business_id,predicted_shortfall_amount,predicted_shortfall_flag
B12345-67890,-12340.50,1
B98765-43210,45000.00,0
...
```

- `business_id` must match the IDs in `eval.csv`
- `predicted_shortfall_amount` is a float (negative = shortfall, positive = healthy)
- `predicted_shortfall_flag` is 0 or 1

---

## Tips

1. **Aggregate financials thoughtfully.** The three monthly snapshots are your richest data source. Consider: latest month values, 3-month averages, revenue-expense ratios, month-over-month trends (is cash_on_hand declining?), and volatility measures.
2. **Handle missing loan data as a feature, not a bug.** Businesses with no loans are fundamentally different from those carrying debt. The absence of data is informative. Consider adding a `has_loan` binary feature and imputing loan columns as zero for businesses without loans.
3. **Engineer financial ratios.** Classic indicators like `monthly_revenue / monthly_expenses`, `accounts_receivable / accounts_payable`, and `cash_on_hand / monthly_expenses` can be more predictive than raw values.
4. **This problem has a linear component.** The underlying relationship has linear structure, so do not overlook linear models with well-engineered features -- they may outperform tree-based methods.
5. **Use both targets strategically.** The regression and classification targets are related. A good regression model can derive the binary flag, or you can train separate models for each track.
6. **Watch for data quality issues.** Inconsistent industry labels, whitespace in names, null values in key financial columns, and orphaned business_ids in child tables all need handling.
7. **Shortfall rate is ~25%.** This is moderately imbalanced. Consider threshold tuning for the classification track.
# musical-potato
