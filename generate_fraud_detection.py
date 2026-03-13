#!/usr/bin/env python3
"""
Dataset 2 — Fraud Detection
Problem type: Imbalanced Binary Classification
Tables: customers.csv, transactions_train.csv, transactions_eval.csv, merchants.csv
Target: is_fraud (0/1, ~3 % positive)
Evaluation: AUC-PR, F1, Recall@5 % FPR
"""

import pandas as pd
import numpy as np
import random
import os

from synthetic_utils import (
    generate_base_classification, map_to_range, map_to_int_range,
    map_to_categories, map_to_binary, generate_ids, sample_pool,
    inject_nulls, inject_messiness, add_orphaned_keys,
    benchmark_fraud, write_fraud_benchmark, fake, POOLS,
)

RANDOM_STATE = 42
N_TRANSACTIONS = 500_000
N_CUSTOMERS = 80_000
N_MERCHANTS = 5_000
OUTPUT_DIR = "output/fraud_detection"

MERCHANT_CATEGORIES = [
    "grocery", "electronics", "travel", "dining", "gas",
    "online_retail", "entertainment", "healthcare", "clothing", "utilities",
]


def main():
    print("=" * 60)
    print("  Fraud Detection Dataset Generator")
    print("=" * 60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)

    # ------------------------------------------------------------------
    # 1. Base predictive features (transaction-level)
    # ------------------------------------------------------------------
    print("\n[1/8] Generating base features (make_classification)…")
    X, y = generate_base_classification(
        n_samples=N_TRANSACTIONS, n_features=10,
        weights=[0.97, 0.03], flip_y=0.03, class_sep=0.4,
        n_informative=7, random_state=RANDOM_STATE,
    )
    fraud_rate = y.mean()
    print(f"      Transactions: {N_TRANSACTIONS:,}  |  Fraud rate: {fraud_rate:.2%}")

    # ------------------------------------------------------------------
    # 2. Map features
    # ------------------------------------------------------------------
    print("[2/8] Mapping features to realistic domains…")

    txn_amount        = map_to_range(pd.Series(X[:, 0]), 1, 9_500)
    dist_home         = map_to_range(pd.Series(X[:, 1]), 0, 5_000)
    hours_since       = map_to_range(pd.Series(X[:, 2]), 0, 720)
    ratio_median      = map_to_range(pd.Series(X[:, 3]), 1, 50, decimals=1)
    num_txn_24h       = map_to_int_range(pd.Series(X[:, 4]), 0, 30)
    acct_age_signal   = map_to_int_range(pd.Series(X[:, 5]), 30, 3_650)
    merch_cat_signal  = map_to_categories(pd.Series(X[:, 6]),
                                          MERCHANT_CATEGORIES)
    is_online         = map_to_categories(pd.Series(X[:, 7]),
                                          ["No", "Yes"])
    card_present      = map_to_categories(pd.Series(X[:, 8]),
                                          ["Yes", "No"])
    is_intl_signal    = map_to_binary(X[:, 9], threshold_pct=85)

    # ------------------------------------------------------------------
    # 3. Assign customer IDs (power-law frequency)
    # ------------------------------------------------------------------
    print("[3/8] Assigning customer IDs…")
    customer_ids_pool = generate_ids(N_CUSTOMERS, prefix="CU")
    weights = np.random.pareto(a=1.5, size=N_CUSTOMERS) + 1
    weights /= weights.sum()
    txn_customer_ids = np.random.choice(
        customer_ids_pool, size=N_TRANSACTIONS, p=weights)

    # ------------------------------------------------------------------
    # 4. Build customers.csv
    # ------------------------------------------------------------------
    print("[4/8] Building customers.csv…")
    acct_age_series = pd.Series(acct_age_signal.values, dtype="float64")
    cust_acct_age = (
        pd.DataFrame({"customer_id": txn_customer_ids,
                       "account_age_days": acct_age_series})
        .groupby("customer_id")["account_age_days"].median().astype(int)
    )
    customers = pd.DataFrame({
        "customer_id": customer_ids_pool,
        "customer_name": sample_pool("first_names", N_CUSTOMERS),
        "card_last_four": np.random.randint(1000, 9999, size=N_CUSTOMERS)
                             .astype(str),
        "home_city": sample_pool("cities", N_CUSTOMERS),
        "home_state": sample_pool("states", N_CUSTOMERS),
    })
    customers = customers.merge(
        cust_acct_age.rename("account_age_days").reset_index(),
        on="customer_id", how="left")
    na_mask = customers["account_age_days"].isna()
    customers.loc[na_mask, "account_age_days"] = np.random.randint(
        30, 3650, size=na_mask.sum())
    customers["account_age_days"] = customers["account_age_days"].astype(int)
    print(f"      {len(customers):,} unique customers")

    # ------------------------------------------------------------------
    # 5. Build merchants.csv (category + is_international aligned to signal)
    # ------------------------------------------------------------------
    print("[5/8] Building merchants.csv…")
    merchants_per_cat = N_MERCHANTS // len(MERCHANT_CATEGORIES)
    merchant_frames = []
    merchant_cat_lookup = {}
    merchant_intl_lookup = {}

    for cat in MERCHANT_CATEGORIES:
        n_intl = max(1, int(merchants_per_cat * 0.15))
        n_dom = merchants_per_cat - n_intl
        m_ids_dom = generate_ids(n_dom, prefix="M")
        m_ids_intl = generate_ids(n_intl, prefix="M")

        for m_ids, intl_flag in [(m_ids_dom, "No"), (m_ids_intl, "Yes")]:
            frame = pd.DataFrame({
                "merchant_id": m_ids,
                "merchant_name": sample_pool("companies", len(m_ids)),
                "merchant_category": cat,
                "merchant_city": sample_pool("cities", len(m_ids)),
                "merchant_state": sample_pool("states", len(m_ids)),
                "is_international": intl_flag,
            })
            merchant_frames.append(frame)
            for mid in m_ids:
                merchant_cat_lookup[mid] = cat
                merchant_intl_lookup[mid] = intl_flag

    merchants = pd.concat(merchant_frames, ignore_index=True)

    # Build mapping: (category, intl_flag) → list of merchant_ids
    merch_group = (
        merchants.groupby(["merchant_category", "is_international"])
        ["merchant_id"].apply(list).to_dict()
    )

    # ------------------------------------------------------------------
    # 6. Assign merchant_id per transaction (matching category + intl)
    # ------------------------------------------------------------------
    print("[6/8] Assigning merchants to transactions…")
    intl_labels = np.where(is_intl_signal == 1, "Yes", "No")
    txn_merchant_ids = np.empty(N_TRANSACTIONS, dtype=object)

    txn_df_temp = pd.DataFrame({
        "idx": np.arange(N_TRANSACTIONS),
        "category": merch_cat_signal.values,
        "intl": intl_labels,
    })
    for (cat, intl), group in txn_df_temp.groupby(["category", "intl"]):
        pool = merch_group.get((cat, intl))
        if pool is None:
            pool = merch_group.get((cat, "No"), merch_group.get((cat, "Yes")))
        txn_merchant_ids[group["idx"].values] = np.random.choice(
            pool, size=len(group))

    # ------------------------------------------------------------------
    # 7. Build transaction DataFrames + inject messiness
    # ------------------------------------------------------------------
    print("[7/8] Building transaction tables & injecting data-quality issues…")
    datetime_pool = pd.date_range("2025-07-01", "2025-12-31 23:59",
                                   freq="min")
    txn_datetimes = np.random.choice(datetime_pool, size=N_TRANSACTIONS)

    transactions = pd.DataFrame({
        "transaction_id": generate_ids(N_TRANSACTIONS, prefix="T"),
        "customer_id": txn_customer_ids,
        "merchant_id": txn_merchant_ids,
        "transaction_amount": txn_amount,
        "transaction_datetime": txn_datetimes,
        "is_online": is_online,
        "card_present": card_present,
        "distance_from_home_miles": dist_home,
        "hours_since_last_txn": hours_since,
        "ratio_to_median_purchase": ratio_median,
        "num_txns_last_24h": num_txn_24h,
        "is_fraud": y,
    })

    # Train/eval split at transaction level (70/30)
    n_train = int(N_TRANSACTIONS * 0.70)
    shuffled = transactions.sample(frac=1, random_state=RANDOM_STATE)
    txn_train = shuffled.iloc[:n_train].copy()
    txn_eval_full = shuffled.iloc[n_train:].copy()
    txn_eval = txn_eval_full.drop(columns=["is_fraud"])
    answer_key = txn_eval_full[["transaction_id", "is_fraud"]].copy()

    # Data quality
    txn_train = inject_nulls(txn_train, pct=0.08,
                             exclude_cols=["transaction_id", "customer_id",
                                           "merchant_id", "is_fraud"])
    txn_train = inject_messiness(
        txn_train,
        categorical_cols=["is_online", "card_present"],
        numeric_cols=["transaction_amount", "distance_from_home_miles"],
    )

    txn_eval = inject_nulls(txn_eval, pct=0.08,
                            exclude_cols=["transaction_id", "customer_id",
                                          "merchant_id"])
    txn_eval = inject_messiness(
        txn_eval,
        categorical_cols=["is_online", "card_present"],
        numeric_cols=["transaction_amount", "distance_from_home_miles"],
    )

    txn_train = add_orphaned_keys(txn_train, "customer_id", pct=0.003)
    txn_train = add_orphaned_keys(txn_train, "merchant_id", pct=0.003)
    txn_eval = add_orphaned_keys(txn_eval, "customer_id", pct=0.003)
    txn_eval = add_orphaned_keys(txn_eval, "merchant_id", pct=0.003)

    customers = inject_nulls(customers, pct=0.08,
                             exclude_cols=["customer_id"])
    customers = inject_messiness(
        customers, string_cols=["customer_name", "home_city"])

    merchants = inject_nulls(merchants, pct=0.03,
                             exclude_cols=["merchant_id"])
    merchants = inject_messiness(
        merchants,
        categorical_cols=["merchant_category", "is_international"],
        string_cols=["merchant_name"],
    )

    # ------------------------------------------------------------------
    # 8. Benchmarks
    # ------------------------------------------------------------------
    print("[8/8] Running benchmarks…")
    train_idx = shuffled.index[:n_train]
    eval_idx = shuffled.index[n_train:]

    flat_train_mask = np.isin(np.arange(N_TRANSACTIONS),
                              transactions.index.get_indexer(train_idx))
    flat_eval_mask = ~flat_train_mask

    bench = benchmark_fraud(
        X[flat_train_mask], y[flat_train_mask],
        X[flat_eval_mask], y[flat_eval_mask])
    write_fraud_benchmark(bench, OUTPUT_DIR, fraud_rate)

    # ------------------------------------------------------------------
    # Write CSVs
    # ------------------------------------------------------------------
    print("\nWriting CSVs…")
    customers.to_csv(f"{OUTPUT_DIR}/customers.csv", index=False)
    merchants.to_csv(f"{OUTPUT_DIR}/merchants.csv", index=False)
    txn_train.to_csv(f"{OUTPUT_DIR}/transactions_train.csv", index=False)
    txn_eval.to_csv(f"{OUTPUT_DIR}/transactions_eval.csv", index=False)
    answer_key.to_csv(f"{OUTPUT_DIR}/answer_key.csv", index=False)

    print(f"\nDone!  Files in {OUTPUT_DIR}/")
    print(f"  customers.csv          : {len(customers):>10,} rows")
    print(f"  merchants.csv          : {len(merchants):>10,} rows")
    print(f"  transactions_train.csv : {len(txn_train):>10,} rows")
    print(f"  transactions_eval.csv  : {len(txn_eval):>10,} rows")
    print(f"  answer_key.csv         : {len(answer_key):>10,} rows")


if __name__ == "__main__":
    main()
