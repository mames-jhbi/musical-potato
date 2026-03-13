# Challenge 2: Fraud Detection

## Overview

A payment processing team needs a fraud risk scoring system that can flag suspicious transactions in near-real-time. The challenge: fraud is rare (~4% of transactions), so the model must detect anomalies without drowning the review team in false positives.

This is an **imbalanced binary classification** problem. Standard accuracy is meaningless here -- a model that predicts "not fraud" for everything would be 96% accurate but completely useless. You must optimize for precision-recall trade-offs.

---

## Description

You are given three related tables: customer profiles, transaction records, and merchant information. Transactions reference both `customer_id` and `merchant_id` as foreign keys, so you must **join all three tables** to assemble the complete feature set.

The target variable is `is_fraud` (1 = fraudulent, 0 = legitimate), available only in `transactions_train.csv`. The evaluation transactions in `transactions_eval.csv` have the label stripped.

### Key challenges:
- Severe class imbalance (~96% legitimate, ~4% fraud)
- Three tables with different granularities must be joined
- Orphaned foreign keys in transaction tables do not match customer or merchant records
- Missing values and inconsistent formatting in categorical fields
- Some features are noise -- not every column carries signal
- Standard metrics like accuracy are misleading; you need imbalance-aware evaluation

---

## Data Dictionary

### customers.csv (~80k rows)

| Column | Type | Description |
|--------|------|-------------|
| `customer_id` | string | **Primary key.** Unique customer identifier |
| `customer_name` | string | Customer name |
| `card_last_four` | string | Last four digits of primary card |
| `home_city` | string | Customer home city |
| `home_state` | string | Customer home state |
| `account_age_days` | int | Days since account was opened (30-3650) |

### merchants.csv (~5k rows)

| Column | Type | Description |
|--------|------|-------------|
| `merchant_id` | string | **Primary key.** Unique merchant identifier |
| `merchant_name` | string | Merchant business name |
| `merchant_category` | string | Business category: grocery, electronics, travel, dining, gas, online_retail, entertainment, healthcare, clothing, utilities |
| `merchant_city` | string | Merchant location city |
| `merchant_state` | string | Merchant location state |
| `is_international` | string | Whether merchant is international (Yes/No) |

### transactions_train.csv (~353k rows)

| Column | Type | Description |
|--------|------|-------------|
| `transaction_id` | string | **Primary key.** Unique transaction identifier |
| `customer_id` | string | **Foreign key** to customers.csv |
| `merchant_id` | string | **Foreign key** to merchants.csv |
| `transaction_amount` | float | Transaction amount in dollars |
| `transaction_datetime` | datetime | Timestamp of transaction |
| `is_online` | string | Whether the transaction was online (Yes/No) |
| `card_present` | string | Whether the physical card was present (Yes/No) |
| `distance_from_home_miles` | float | Distance from customer's home to transaction location |
| `hours_since_last_txn` | float | Hours since this customer's previous transaction |
| `ratio_to_median_purchase` | float | Transaction amount divided by customer's median purchase |
| `num_txns_last_24h` | int | Number of transactions by this customer in the past 24 hours |
| `is_fraud` | int | **Target:** 1 = fraudulent, 0 = legitimate |

### transactions_eval.csv (~151k rows)

Same schema as transactions_train.csv **except** `is_fraud` is not included.

**Note:** The eval file may contain duplicate rows -- this is a data quality issue you need to handle.

### answer_key.csv (150k rows)

| Column | Type | Description |
|--------|------|-------------|
| `transaction_id` | string | Transaction identifier |
| `is_fraud` | int | True label |

---

## Evaluation Criteria

Given the severe class imbalance, **precision-recall metrics** are primary:

| Metric | Weight | Description |
|--------|--------|-------------|
| **AUC-PR** | Primary | Area under the Precision-Recall curve. The single most important metric for this challenge. |
| **F1 Score** | Secondary | Harmonic mean of precision and recall at the optimal threshold |
| **Recall@5% FPR** | Secondary | How much fraud do you catch if you can only flag 5% of all transactions? This reflects a real operational constraint. |

### Why not AUC-ROC?

With only ~4% fraud, AUC-ROC can look deceptively high even for weak models. AUC-PR focuses on the precision-recall trade-off in the positive (fraud) class, which is what the business actually cares about.

### Baseline Benchmarks

See `benchmark.txt`. A Logistic Regression baseline scores ~0.11 AUC-PR (essentially failing to detect fraud). Random Forest reaches ~0.48 AUC-PR. There is substantial room for improvement.

---

## Submission Format

Submit a CSV file with **exactly two columns**, one row per `transaction_id` in `transactions_eval.csv`:

```csv
transaction_id,fraud_probability
T12345-67890,0.92
T98765-43210,0.01
...
```

- `transaction_id` must match the IDs in `transactions_eval.csv` (after deduplication)
- `fraud_probability` must be a float between 0 and 1
- Higher values = more likely to be fraud

---

## Tips

1. **Do not use accuracy as your metric.** A model predicting all-legitimate gets 96% accuracy but 0% fraud detection. Focus on AUC-PR and F1.
2. **Join all three tables.** Merchant category and international status are in `merchants.csv`, not the transaction table. Account age is in `customers.csv`.
3. **Handle class imbalance explicitly.** Consider: oversampling (SMOTE), undersampling, class weights, threshold tuning, or anomaly detection approaches.
4. **Deduplicate the eval set.** `transactions_eval.csv` contains some duplicate rows.
5. **Engineer features from timestamps.** Transaction hour-of-day, day-of-week, and time patterns can be informative.
6. **Watch for data quality issues.** Categorical values have inconsistent casing ("Yes" vs "yes" vs "YES"). Standardize them.
7. **Some features are noise.** Not all 10+ columns carry signal. Feature importance analysis can help you focus.
8. **Think like a fraud analyst.** High transaction amounts, unusual distances from home, rapid successive transactions, and international merchants are common fraud indicators in real data.
