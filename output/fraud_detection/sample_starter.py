"""
Challenge 2: Fraud Detection -- Sample Starter
Pipeline: load → join → clean → feature engineer → train → predict → submit → score
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_curve, auc, f1_score, roc_curve
import warnings

warnings.filterwarnings("ignore")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 1. Load ─────────────────────────────────────────────────────────────────

customers = pd.read_csv(os.path.join(DATA_DIR, "customers.csv"))
merchants = pd.read_csv(os.path.join(DATA_DIR, "merchants.csv"))
txn_train = pd.read_csv(os.path.join(DATA_DIR, "transactions_train.csv"))
txn_eval = pd.read_csv(os.path.join(DATA_DIR, "transactions_eval.csv"))
answer_key = pd.read_csv(os.path.join(DATA_DIR, "answer_key.csv"))

print(f"Customers:        {customers.shape}")
print(f"Merchants:        {merchants.shape}")
print(f"Txn Train:        {txn_train.shape}")
print(f"Txn Eval:         {txn_eval.shape}")
print(f"\nFraud rate (train): {txn_train['is_fraud'].mean():.2%}")

# ── 2. Quick EDA ────────────────────────────────────────────────────────────

print("\n--- Train nulls ---")
print(txn_train.isnull().sum())
print(f"\n--- Eval duplicates ---")
print(f"Eval rows: {len(txn_eval)}, Unique IDs: {txn_eval['transaction_id'].nunique()}")
print(f"Duplicate rows: {txn_eval.duplicated().sum()}")

orphan_cust = txn_train[~txn_train["customer_id"].isin(customers["customer_id"])]
orphan_merch = txn_train[~txn_train["merchant_id"].isin(merchants["merchant_id"])]
print(f"\nOrphan customer_ids: {len(orphan_cust)}")
print(f"Orphan merchant_ids: {len(orphan_merch)}")

# ── 3. Clean ────────────────────────────────────────────────────────────────


def clean_str(s):
    return str(s).strip().lower() if pd.notna(s) else s


for col in ["is_online", "card_present"]:
    txn_train[col] = txn_train[col].apply(clean_str)
    txn_eval[col] = txn_eval[col].apply(clean_str)

merchants["merchant_category"] = merchants["merchant_category"].apply(clean_str)
merchants["is_international"] = merchants["is_international"].apply(clean_str)

txn_eval = txn_eval.drop_duplicates(subset=["transaction_id"], keep="first")
print(f"\nEval after dedup: {len(txn_eval):,}")

# ── 4. Join Tables ──────────────────────────────────────────────────────────


def enrich_transactions(txn, cust, merch):
    txn = txn.merge(
        cust[["customer_id", "account_age_days"]], on="customer_id", how="left"
    )
    txn = txn.merge(
        merch[["merchant_id", "merchant_category", "is_international"]],
        on="merchant_id",
        how="left",
    )
    return txn


txn_train = enrich_transactions(txn_train, customers, merchants)
txn_eval = enrich_transactions(txn_eval, customers, merchants)

print(f"Enriched train: {txn_train.shape}")
print(f"Enriched eval:  {txn_eval.shape}")

# ── 5. Feature Engineering ──────────────────────────────────────────────────

cat_cols = ["is_online", "card_present", "merchant_category", "is_international"]
encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    txn_train[col] = txn_train[col].fillna("unknown")
    txn_eval[col] = txn_eval[col].fillna("unknown")
    all_vals = pd.concat([txn_train[col], txn_eval[col]]).unique()
    le.fit(all_vals)
    txn_train[col + "_enc"] = le.transform(txn_train[col])
    txn_eval[col + "_enc"] = le.transform(txn_eval[col])
    encoders[col] = le

feature_cols = [
    "transaction_amount", "distance_from_home_miles", "hours_since_last_txn",
    "ratio_to_median_purchase", "num_txns_last_24h", "account_age_days",
    "is_online_enc", "card_present_enc", "merchant_category_enc",
    "is_international_enc",
]

X_train = txn_train[feature_cols].fillna(0)
y_train = txn_train["is_fraud"]
X_eval = txn_eval[feature_cols].fillna(0)

print(f"\nFeatures: {len(feature_cols)}")
print(f"X_train: {X_train.shape}  |  X_eval: {X_eval.shape}")

# ── 6. Train Model ──────────────────────────────────────────────────────────

model = RandomForestClassifier(
    n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)

train_proba = model.predict_proba(X_train)[:, 1]
prec, rec, _ = precision_recall_curve(y_train, train_proba)
print(f"\nTrain AUC-PR: {auc(rec, prec):.4f}")

# ── 7. Generate Submission ──────────────────────────────────────────────────

eval_proba = model.predict_proba(X_eval)[:, 1]

submission = pd.DataFrame({
    "transaction_id": txn_eval["transaction_id"],
    "fraud_probability": eval_proba,
})
submission.to_csv(os.path.join(DATA_DIR, "submission.csv"), index=False)
print(f"Submission written: {len(submission):,} rows")

# ── 8. Score Against Answer Key ─────────────────────────────────────────────

scored = submission.merge(answer_key, on="transaction_id", how="inner")
y_true = scored["is_fraud"].values
y_prob = scored["fraud_probability"].values
y_pred = (y_prob >= 0.5).astype(int)

prec, rec, _ = precision_recall_curve(y_true, y_prob)
auc_pr = auc(rec, prec)
f1 = f1_score(y_true, y_pred)

fpr, tpr, _ = roc_curve(y_true, y_prob)
valid = np.where(fpr <= 0.05)[0]
recall_at_5fpr = tpr[valid[-1]] if len(valid) > 0 else 0.0

print(f"\nAUC-PR:       {auc_pr:.4f}")
print(f"F1:           {f1:.4f}")
print(f"Recall@5%FPR: {recall_at_5fpr:.4f}")
print("Compare to benchmark.txt -- can you beat it?")
