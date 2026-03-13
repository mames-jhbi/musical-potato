#!/usr/bin/env python3
"""
Dataset 4 — Loan Default Prediction
Problem type: Regression (primary) + Binary Classification (secondary)
Tables: borrowers.csv, loans.csv, payment_history.csv
Targets:
  days_to_early_default   (continuous, days from origination to default; non-defaulters = term_days)
  default_flag            (0/1, ~18% positive)
Evaluation:
  Regression  — RMSE, MAE, R²
  Classification — AUC-ROC, F1
"""

import pandas as pd
import numpy as np
import random
import os

from synthetic_utils import (
    generate_base_regression, map_to_range, map_to_int_range,
    map_to_categories, generate_ids, sample_pool, inject_nulls,
    inject_messiness, add_orphaned_keys, train_eval_split,
    benchmark_loan_default, write_loan_default_benchmark, fake,
)

RANDOM_STATE = 42
N_LOANS = 180_000
OUTPUT_DIR = "output/loan_default"

LOAN_TYPES = [
    "personal", "auto", "home_equity", "credit_line",
    "small_business", "student",
]


def main():
    print("=" * 60)
    print("  Loan Default Dataset Generator")
    print("=" * 60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)

    # ------------------------------------------------------------------
    # 1. Base predictive features (make_regression for days + derive flag)
    # ------------------------------------------------------------------
    print("\n[1/9] Generating base features (make_regression)…")
    X, y_raw = generate_base_regression(
        n_samples=N_LOANS, n_features=12, n_informative=6,
        noise=180.0, random_state=RANDOM_STATE,
    )

    # y_raw drives both targets: low y = early default, high y = no default
    # Map to days: defaulters get 30–720 days; non-defaulters get term_days (e.g. 1095)
    term_days = 1095  # 36 months typical
    p18 = np.percentile(y_raw, 18)
    default_flag = (y_raw < p18).astype(int)
    default_rate = default_flag.mean()

    # days_to_early_default: defaulters 30–720, non-defaulters = term_days
    days_for_defaulters = map_to_int_range(
        pd.Series(y_raw[default_flag == 1]), 30, 720
    ).values
    days_for_nondefaulters = np.full((default_flag == 0).sum(), term_days)

    days_to_default = np.zeros(N_LOANS, dtype=int)
    days_to_default[default_flag == 1] = days_for_defaulters
    days_to_default[default_flag == 0] = days_for_nondefaulters

    print(f"      Loans: {N_LOANS:,}  |  Default rate: {default_rate:.1%}")

    # ------------------------------------------------------------------
    # 2. Map features to realistic domains
    # ------------------------------------------------------------------
    print("[2/9] Mapping features to realistic domains…")

    credit_score       = map_to_int_range(pd.Series(X[:, 0]), 300, 850)
    annual_income      = map_to_range(pd.Series(X[:, 1]), 25_000, 250_000)
    debt_to_income     = map_to_range(pd.Series(X[:, 2]), 10, 60, decimals=1)
    employment_years   = map_to_int_range(pd.Series(X[:, 3]), 0, 35)
    principal          = map_to_range(pd.Series(X[:, 4]), 5_000, 150_000)
    interest_rate      = map_to_range(pd.Series(X[:, 5]), 5, 28, decimals=2)
    term_months        = map_to_int_range(pd.Series(X[:, 6]), 12, 84)
    loan_type          = map_to_categories(pd.Series(X[:, 7]), LOAN_TYPES)
    num_existing_loans = map_to_int_range(pd.Series(X[:, 8]), 0, 8)
    payment_to_income  = map_to_range(pd.Series(X[:, 9]), 0.02, 0.35, decimals=3)
    previous_defaults  = map_to_int_range(pd.Series(X[:, 10]), 0, 3)
    months_since_inquiry = map_to_int_range(pd.Series(X[:, 11]), 0, 48)

    # Unique borrowers (some have multiple loans)
    n_borrowers = int(N_LOANS * 0.65)
    borrower_ids = generate_ids(n_borrowers, prefix="BR")

    # Assign borrower_id to each loan (some borrowers have 2–3 loans)
    loan_to_borrower = np.random.choice(
        np.arange(n_borrowers), size=N_LOANS, replace=True,
        p=np.random.dirichlet(np.ones(n_borrowers)),
    )
    borrower_ids_per_loan = borrower_ids[loan_to_borrower]

    # ------------------------------------------------------------------
    # 3. Build borrowers.csv
    # ------------------------------------------------------------------
    print("[3/9] Building borrowers.csv…")
    borrower_unique = np.unique(borrower_ids_per_loan, return_inverse=True)[0]
    n_unique = len(borrower_unique)

    # Aggregate borrower-level features (take first loan's mapped values per borrower)
    first_loan_idx = {}
    for i, bid in enumerate(borrower_ids_per_loan):
        if bid not in first_loan_idx:
            first_loan_idx[bid] = i

    bid_order = [borrower_ids_per_loan[k] for k in sorted(first_loan_idx.values())]
    borrowers = pd.DataFrame({
        "borrower_id": borrower_unique,
        "credit_score": np.array([credit_score[first_loan_idx[b]] for b in borrower_unique]),
        "annual_income": np.array([annual_income[first_loan_idx[b]] for b in borrower_unique]),
        "debt_to_income_ratio": np.array([debt_to_income[first_loan_idx[b]] for b in borrower_unique]),
        "employment_years": np.array([employment_years[first_loan_idx[b]] for b in borrower_unique]),
        "num_existing_loans": np.array([num_existing_loans[first_loan_idx[b]] for b in borrower_unique]),
        "previous_defaults": np.array([previous_defaults[first_loan_idx[b]] for b in borrower_unique]),
        "months_since_last_inquiry": np.array([months_since_inquiry[first_loan_idx[b]] for b in borrower_unique]),
        "city": sample_pool("cities", n_unique),
        "state": sample_pool("states", n_unique),
    })
    print(f"      {len(borrowers):,} rows")

    # ------------------------------------------------------------------
    # 4. Build loans.csv
    # ------------------------------------------------------------------
    print("[4/9] Building loans.csv…")
    monthly_payment = np.round(principal * (interest_rate / 100 / 12) *
                               (1 + interest_rate / 100 / 12) ** term_months /
                               ((1 + interest_rate / 100 / 12) ** term_months - 1), 2)

    loan_ids = generate_ids(N_LOANS, prefix="L")
    origination_dates = pd.to_datetime("2023-01-01") + pd.to_timedelta(
        np.random.randint(0, 730, N_LOANS), unit="D"
    )

    loans = pd.DataFrame({
        "loan_id": loan_ids,
        "borrower_id": borrower_ids_per_loan,
        "principal": np.round(principal, 2),
        "interest_rate": interest_rate,
        "term_months": term_months,
        "loan_type": loan_type,
        "monthly_payment": monthly_payment,
        "payment_to_income_ratio": payment_to_income,
        "origination_date": [d.strftime("%Y-%m-%d") for d in pd.to_datetime(origination_dates)],
    })
    print(f"      {len(loans):,} rows")

    # ------------------------------------------------------------------
    # 5. Build payment_history.csv (monthly status per loan)
    # ------------------------------------------------------------------
    print("[5/9] Building payment_history.csv…")
    # Up to 12 months of history per loan
    hist_rows = []
    for i in range(N_LOANS):
        term = term_months[i]
        n_months = min(12, max(1, term // 12))
        for m in range(n_months):
            hist_rows.append({
                "loan_id": loan_ids[i],
                "month_number": m + 1,
                "payment_due_date": f"2023-{1 + (m % 12):02d}-15",
                "amount_due": round(monthly_payment[i], 2),
                "amount_paid": round(monthly_payment[i] * np.random.uniform(0.7, 1.05), 2)
                if m < n_months - 1 or default_flag[i] == 0
                else round(monthly_payment[i] * np.random.uniform(0, 0.5), 2),
                "days_delinquent": max(0, int(np.random.exponential(2))) if default_flag[i] == 1 and m >= n_months - 2 else 0,
            })

    payment_history = pd.DataFrame(hist_rows)
    print(f"      {len(payment_history):,} rows")

    # ------------------------------------------------------------------
    # 6. Inject data-quality issues
    # ------------------------------------------------------------------
    print("[6/9] Injecting data-quality issues…")
    borrowers = inject_nulls(borrowers, pct=0.08, exclude_cols=["borrower_id"])
    borrowers = inject_messiness(
        borrowers,
        categorical_cols=["state"],
        string_cols=[],
        numeric_cols=["credit_score", "annual_income", "debt_to_income_ratio", "employment_years"],
    )

    loans = inject_nulls(loans, pct=0.06, exclude_cols=["loan_id", "borrower_id"])
    loans = inject_messiness(
        loans,
        categorical_cols=["loan_type"],
        numeric_cols=["principal", "interest_rate", "monthly_payment", "payment_to_income_ratio"],
    )
    loans = add_orphaned_keys(loans, "borrower_id", pct=0.004)

    payment_history = inject_nulls(
        payment_history, pct=0.10,
        exclude_cols=["loan_id", "month_number"],
    )

    # ------------------------------------------------------------------
    # 7. Train/eval split
    # ------------------------------------------------------------------
    print("[7/9] Splitting train/eval…")
    target_df = pd.DataFrame({
        "loan_id": loan_ids,
        "days_to_early_default": days_to_default,
        "default_flag": default_flag,
    })
    train, evaluation, answer_key, train_ids, eval_ids = train_eval_split(
        target_df, "loan_id",
        ["days_to_early_default", "default_flag"],
    )

    # ------------------------------------------------------------------
    # 8. Benchmarks
    # ------------------------------------------------------------------
    print("[8/9] Running benchmarks…")
    train_mask = np.isin(loan_ids, train_ids)
    eval_mask = np.isin(loan_ids, eval_ids)

    feature_cols = [
        credit_score, annual_income, debt_to_income, employment_years,
        principal, interest_rate, term_months, payment_to_income,
        num_existing_loans, previous_defaults, months_since_inquiry,
    ]
    X_flat = np.column_stack([c.values if hasattr(c, "values") else c for c in feature_cols])

    bench = benchmark_loan_default(
        X_flat[train_mask], days_to_default[train_mask],
        X_flat[eval_mask], days_to_default[eval_mask],
        X_flat[train_mask], default_flag[train_mask],
        X_flat[eval_mask], default_flag[eval_mask],
    )
    write_loan_default_benchmark(bench, OUTPUT_DIR, default_rate)

    # ------------------------------------------------------------------
    # 9. Write CSVs
    # ------------------------------------------------------------------
    print("[9/9] Writing CSVs…")
    borrowers.to_csv(f"{OUTPUT_DIR}/borrowers.csv", index=False)
    loans.to_csv(f"{OUTPUT_DIR}/loans.csv", index=False)
    payment_history.to_csv(f"{OUTPUT_DIR}/payment_history.csv", index=False)
    train.to_csv(f"{OUTPUT_DIR}/train.csv", index=False)
    evaluation.to_csv(f"{OUTPUT_DIR}/eval.csv", index=False)
    answer_key.to_csv(f"{OUTPUT_DIR}/answer_key.csv", index=False)

    print(f"\nDone!  Files in {OUTPUT_DIR}/")
    print(f"  borrowers.csv      : {len(borrowers):>10,} rows")
    print(f"  loans.csv          : {len(loans):>10,} rows")
    print(f"  payment_history.csv: {len(payment_history):>10,} rows")
    print(f"  train.csv          : {len(train):>10,} rows")
    print(f"  eval.csv           : {len(evaluation):>10,} rows")
    print(f"  answer_key.csv     : {len(answer_key):>10,} rows")


if __name__ == "__main__":
    main()
