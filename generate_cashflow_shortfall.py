#!/usr/bin/env python3
"""
Dataset 3 — Cash Flow Shortfall Prediction
Problem type: Regression (primary) + Binary Classification (secondary)
Tables: businesses.csv, financials.csv, loans.csv
Targets:
  cashflow_shortfall_amount  (continuous, negative = shortfall)
  shortfall_flag             (0/1, ~25 % positive)
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
    benchmark_cashflow, write_cashflow_benchmark, fake,
)

RANDOM_STATE = 42
N_BUSINESSES = 200_000
OUTPUT_DIR = "output/cashflow_shortfall"

INDUSTRIES = [
    "retail", "restaurant", "construction", "healthcare",
    "professional_services", "manufacturing", "transportation",
    "tech_services",
]


def main():
    print("=" * 60)
    print("  Cash Flow Shortfall Dataset Generator")
    print("=" * 60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)

    # ------------------------------------------------------------------
    # 1. Base predictive features (make_regression)
    # ------------------------------------------------------------------
    print("\n[1/8] Generating base features (make_regression)…")
    X, y_raw = generate_base_regression(
        n_samples=N_BUSINESSES, n_features=12, n_informative=6,
        noise=250.0, random_state=RANDOM_STATE,
    )

    # Rescale y so ~25 % are negative (shortfall)
    p25 = np.percentile(y_raw, 25)
    y_shifted = y_raw - p25
    y_max = y_shifted.max()
    y_min = y_shifted.min()
    cashflow_amount = np.where(
        y_shifted >= 0,
        y_shifted / y_max * 100_000,
        y_shifted / abs(y_min) * 50_000,
    )
    cashflow_amount = np.round(cashflow_amount, 2)
    shortfall_flag = (cashflow_amount < 0).astype(int)

    shortfall_rate = shortfall_flag.mean()
    print(f"      Businesses: {N_BUSINESSES:,}  |  Shortfall rate: {shortfall_rate:.1%}")

    # ------------------------------------------------------------------
    # 2. Map features to realistic domains
    # ------------------------------------------------------------------
    print("[2/8] Mapping features to realistic domains…")
    business_ids = generate_ids(N_BUSINESSES, prefix="B")

    monthly_revenue     = map_to_range(pd.Series(X[:, 0]), 5_000, 500_000)
    monthly_expenses    = map_to_range(pd.Series(X[:, 1]), 4_000, 480_000)
    accts_receivable    = map_to_range(pd.Series(X[:, 2]), 0, 200_000)
    accts_payable       = map_to_range(pd.Series(X[:, 3]), 0, 180_000)
    cash_on_hand        = map_to_range(pd.Series(X[:, 4]), 1_000, 250_000)
    num_employees       = map_to_int_range(pd.Series(X[:, 5]), 1, 200)
    industry            = map_to_categories(pd.Series(X[:, 6]), INDUSTRIES)
    outstanding_bal     = map_to_range(pd.Series(X[:, 7]), 0, 500_000)
    monthly_payment     = map_to_range(pd.Series(X[:, 8]), 0, 15_000)
    credit_util         = map_to_range(pd.Series(X[:, 9]), 0, 100, decimals=1)
    years_in_biz        = map_to_int_range(pd.Series(X[:, 10]), 0, 40)
    interest_rate       = map_to_range(pd.Series(X[:, 11]), 3, 25, decimals=2)

    # ------------------------------------------------------------------
    # 3. Build businesses.csv
    # ------------------------------------------------------------------
    print("[3/8] Building businesses.csv…")
    businesses = pd.DataFrame({
        "business_id": business_ids,
        "business_name": sample_pool("companies", N_BUSINESSES),
        "owner_name": [
            f"{fn} {ln}" for fn, ln in
            zip(sample_pool("first_names", N_BUSINESSES),
                sample_pool("last_names", N_BUSINESSES))
        ],
        "industry": industry,
        "city": sample_pool("cities", N_BUSINESSES),
        "state": sample_pool("states", N_BUSINESSES),
        "years_in_business": years_in_biz,
        "num_employees": num_employees,
    })
    print(f"      {len(businesses):,} rows")

    # ------------------------------------------------------------------
    # 4. Build financials.csv (3 monthly snapshots per business)
    # ------------------------------------------------------------------
    print("[4/8] Building financials.csv (~3 months per business)…")
    months = ["2025-10", "2025-11", "2025-12"]
    fin_frames = []

    base_arrays = {
        "monthly_revenue": monthly_revenue.values,
        "monthly_expenses": monthly_expenses.values,
        "accounts_receivable": accts_receivable.values,
        "accounts_payable": accts_payable.values,
        "cash_on_hand": cash_on_hand.values,
    }

    for mi, month in enumerate(months):
        noise_std = 0.35 * (2 - mi)  # older months → more noise
        frame_data = {
            "business_id": business_ids,
            "reporting_month": month,
        }
        for col, base_vals in base_arrays.items():
            noise = np.random.normal(1.0, max(noise_std, 0.02),
                                     size=N_BUSINESSES)
            frame_data[col] = np.round(np.maximum(0, base_vals * noise), 2)

        fin_frames.append(pd.DataFrame(frame_data))

    financials = pd.concat(fin_frames, ignore_index=True)
    print(f"      {len(financials):,} rows")

    # ------------------------------------------------------------------
    # 5. Build loans.csv (~60 % of businesses have ≥1 loan)
    # ------------------------------------------------------------------
    print("[5/8] Building loans.csv…")
    # Businesses with higher outstanding_bal signal are more likely to have loans
    has_loan = outstanding_bal.values > np.percentile(outstanding_bal.values, 40)
    biz_with_loans = business_ids[has_loan]
    n_with_loans = len(biz_with_loans)

    # ~30 % of those get a second loan
    has_second = np.random.random(n_with_loans) < 0.30
    loan_biz_ids = np.concatenate([
        biz_with_loans,
        biz_with_loans[has_second],
    ])
    n_loans = len(loan_biz_ids)

    loan_types = ["term_loan", "line_of_credit", "sba_loan",
                  "equipment_financing"]

    # For first loans: use mapped sklearn features directly
    # For second loans: add noise to the sklearn features
    first_bal = outstanding_bal.values[has_loan]
    second_bal = first_bal[has_second] * np.random.normal(0.5, 0.15,
                                                          size=has_second.sum())
    all_bal = np.round(np.maximum(0, np.concatenate([first_bal, second_bal])), 2)

    first_pmt = monthly_payment.values[has_loan]
    second_pmt = first_pmt[has_second] * np.random.normal(0.5, 0.15,
                                                           size=has_second.sum())
    all_pmt = np.round(np.maximum(0, np.concatenate([first_pmt, second_pmt])), 2)

    first_util = credit_util.values[has_loan]
    second_util = np.clip(
        first_util[has_second] * np.random.normal(1.0, 0.2,
                                                   size=has_second.sum()),
        0, 100)
    all_util = np.round(np.concatenate([first_util, second_util]), 1)

    first_rate = interest_rate.values[has_loan]
    second_rate = np.clip(
        first_rate[has_second] + np.random.normal(1.5, 0.5,
                                                   size=has_second.sum()),
        3, 25)
    all_rate = np.round(np.concatenate([first_rate, second_rate]), 2)

    loans = pd.DataFrame({
        "loan_id": generate_ids(n_loans, prefix="L"),
        "business_id": loan_biz_ids,
        "loan_type": np.random.choice(loan_types, size=n_loans),
        "outstanding_balance": all_bal,
        "monthly_payment": all_pmt,
        "credit_line_utilization_pct": all_util,
        "interest_rate": all_rate,
    })
    loans = loans.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"      {len(loans):,} rows  ({n_with_loans:,} businesses with ≥1 loan, "
          f"{has_second.sum():,} with 2 loans)")

    # ------------------------------------------------------------------
    # 6. Inject data-quality issues
    # ------------------------------------------------------------------
    print("[6/8] Injecting data-quality issues…")
    businesses = inject_nulls(businesses, pct=0.10,
                              exclude_cols=["business_id"])
    businesses = inject_messiness(
        businesses,
        categorical_cols=["industry", "state"],
        string_cols=["business_name", "owner_name", "city"],
        numeric_cols=["years_in_business", "num_employees"],
    )

    financials = inject_nulls(financials, pct=0.15,
                              exclude_cols=["business_id", "reporting_month"])
    financials = inject_messiness(
        financials,
        numeric_cols=["monthly_revenue", "monthly_expenses",
                      "accounts_receivable", "accounts_payable",
                      "cash_on_hand"],
        duplicate_pct=0.01,
    )
    financials = add_orphaned_keys(financials, "business_id", pct=0.005)

    loans = inject_nulls(loans, pct=0.08,
                         exclude_cols=["loan_id", "business_id"])
    loans = inject_messiness(
        loans,
        categorical_cols=["loan_type"],
        numeric_cols=["outstanding_balance", "monthly_payment",
                      "credit_line_utilization_pct"],
    )
    loans = add_orphaned_keys(loans, "business_id", pct=0.005)

    # ------------------------------------------------------------------
    # 7. Train/eval split
    # ------------------------------------------------------------------
    print("[7/8] Splitting train/eval…")
    target_df = pd.DataFrame({
        "business_id": business_ids,
        "cashflow_shortfall_amount": cashflow_amount,
        "shortfall_flag": shortfall_flag,
    })
    train, evaluation, answer_key, train_ids, eval_ids = train_eval_split(
        target_df, "business_id",
        ["cashflow_shortfall_amount", "shortfall_flag"])

    # ------------------------------------------------------------------
    # 8. Benchmarks
    # ------------------------------------------------------------------
    print("[8/8] Running benchmarks…")
    train_mask = np.isin(business_ids, train_ids)
    eval_mask = np.isin(business_ids, eval_ids)

    bench = benchmark_cashflow(
        X[train_mask], cashflow_amount[train_mask],
        X[eval_mask], cashflow_amount[eval_mask],
        X[train_mask], shortfall_flag[train_mask],
        X[eval_mask], shortfall_flag[eval_mask],
    )
    write_cashflow_benchmark(bench, OUTPUT_DIR, shortfall_rate)

    # ------------------------------------------------------------------
    # Write CSVs
    # ------------------------------------------------------------------
    print("\nWriting CSVs…")
    businesses.to_csv(f"{OUTPUT_DIR}/businesses.csv", index=False)
    financials.to_csv(f"{OUTPUT_DIR}/financials.csv", index=False)
    loans.to_csv(f"{OUTPUT_DIR}/loans.csv", index=False)
    train.to_csv(f"{OUTPUT_DIR}/train.csv", index=False)
    evaluation.to_csv(f"{OUTPUT_DIR}/eval.csv", index=False)
    answer_key.to_csv(f"{OUTPUT_DIR}/answer_key.csv", index=False)

    print(f"\nDone!  Files in {OUTPUT_DIR}/")
    print(f"  businesses.csv  : {len(businesses):>10,} rows")
    print(f"  financials.csv  : {len(financials):>10,} rows")
    print(f"  loans.csv       : {len(loans):>10,} rows")
    print(f"  train.csv       : {len(train):>10,} rows")
    print(f"  eval.csv        : {len(evaluation):>10,} rows")
    print(f"  answer_key.csv  : {len(answer_key):>10,} rows")


if __name__ == "__main__":
    main()
