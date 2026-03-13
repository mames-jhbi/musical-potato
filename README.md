# Banking Analytics Challenge Series

Three independent data science challenges designed around realistic synthetic banking data. Each challenge presents a different problem type, evaluation methodology, and set of data engineering hurdles that mirror what analysts encounter in production banking environments.

---

## Challenges at a Glance

| # | Challenge | Problem Type | Primary Metric | Rows (approx.) |
|---|-----------|-------------|----------------|-----------------|
| 1 | [Product Recommendation](output/product_recommendation/) | Classification + Ranking | AUC-ROC, Precision@K, MAP@K | 250k customers |
| 2 | [Fraud Detection](output/fraud_detection/) | Imbalanced Classification | AUC-PR, F1, Recall@5% FPR | 500k transactions |
| 3 | [Cash Flow Shortfall](output/cashflow_shortfall/) | Regression + Classification | RMSE, MAE, R-squared, AUC-ROC | 200k businesses |

---

## What to Expect

Each challenge provides **multiple related CSV files** that must be joined, cleaned, and transformed before modeling. You will encounter:

- **Missing values** scattered across feature columns (5-15%)
- **Inconsistent formatting** in categorical fields (mixed casing, extra whitespace)
- **Duplicate rows** in some tables
- **Orphaned foreign keys** that do not match any record in the parent table
- **Numeric outliers** that require investigation and handling
- **Multi-table relationships** requiring JOIN operations and aggregation logic

This is intentional. Real banking data is messy, and the ability to prepare data systematically is as important as model selection.

---

## Competition Format

Each challenge follows a Kaggle-style structure:

```
output/<challenge_name>/
    <table_1>.csv            # Feature tables (may need joining)
    <table_2>.csv
    ...
    train.csv                # Training IDs with target labels
    eval.csv                 # Evaluation IDs (no labels)
    answer_key.csv           # True labels for scoring (DO NOT PEEK)
    benchmark.txt            # Baseline model scores to beat
```

1. **Join** the feature tables together using the provided keys.
2. **Clean** the data (handle nulls, fix formatting, deduplicate, manage outliers).
3. **Engineer features** from the raw columns.
4. **Train** your model on `train.csv` IDs only.
5. **Predict** on `eval.csv` IDs and format your submission.
6. **Score** against `answer_key.csv` using the challenge-specific metrics.

---

## Getting Started

### Prerequisites

```
Python 3.10+
pandas
numpy
scikit-learn
```

### Quick Start

```bash
# Look at the challenge you want to tackle
cd output/product_recommendation/   # or fraud_detection/ or cashflow_shortfall/

# Read the challenge README for full details
cat README.md

# Check the benchmark you need to beat
cat benchmark.txt

# Start exploring the data
python3 -c "import pandas as pd; print(pd.read_csv('customers.csv').head())"
```

---

## Rules

1. **No peeking at `answer_key.csv`** during development. Use it only for final scoring.
2. Train exclusively on IDs listed in `train.csv`. Do not use evaluation IDs for training.
3. Your submission must follow the exact format specified in each challenge README.
4. You may use any Python libraries or modeling approaches.
5. Feature engineering, ensembles, and creative data transformations are encouraged.

---

## Challenge Details

Detailed instructions, data dictionaries, evaluation criteria, and submission formats are in each challenge folder:

- **[Challenge 1: Product Recommendation](output/product_recommendation/README.md)**
- **[Challenge 2: Fraud Detection](output/fraud_detection/README.md)**
- **[Challenge 3: Cash Flow Shortfall](output/cashflow_shortfall/README.md)**

Good luck!
