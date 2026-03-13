"""
Challenge 4: Loan Default Prediction -- Sample Starter
Pipeline: load → join → clean → feature engineer → train → predict → submit → score
Dual targets: regression (days to default) + classification (default flag)
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, f1_score,
)
import warnings

warnings.filterwarnings("ignore")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 1. Load ─────────────────────────────────────────────────────────────────

borrowers = pd.read_csv(os.path.join(DATA_DIR, "borrowers.csv"))
loans = pd.read_csv(os.path.join(DATA_DIR, "loans.csv"))
payment_history = pd.read_csv(os.path.join(DATA_DIR, "payment_history.csv"))
train_labels = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
eval_ids = pd.read_csv(os.path.join(DATA_DIR, "eval.csv"))
answer_key = pd.read_csv(os.path.join(DATA_DIR, "answer_key.csv"))

print(f"Borrowers:       {borrowers.shape}")
print(f"Loans:           {loans.shape}")
print(f"Payment history: {payment_history.shape}")
print(f"Train:           {train_labels.shape}")
print(f"Eval:            {eval_ids.shape}")
print(f"\nDefault rate (train): {train_labels['default_flag'].mean():.1%}")

# ── 2. Quick EDA ────────────────────────────────────────────────────────────

print("\n--- Borrower nulls ---")
print(borrowers.isnull().sum())
print("\n--- Loan nulls ---")
print(loans.isnull().sum())
print("\n--- Payment status distribution ---")
print(payment_history["days_delinquent"].describe())

orphan_loans = loans[~loans["borrower_id"].isin(borrowers["borrower_id"])]
print(f"\nOrphaned loans (no borrower): {len(orphan_loans)}")

# ── 3. Clean ────────────────────────────────────────────────────────────────


def clean_str(s):
    return str(s).strip().lower() if pd.notna(s) else s


borrowers["state"] = borrowers["state"].apply(clean_str)
loans["loan_type"] = loans["loan_type"].apply(clean_str)

# Keep all loans (including those with orphan borrower_ids) so we can predict every eval loan
payment_history = payment_history[payment_history["loan_id"].isin(loans["loan_id"])]

borrowers = borrowers.drop_duplicates(subset=["borrower_id"], keep="first")
loans = loans.drop_duplicates(subset=["loan_id"], keep="first")
payment_history = payment_history.drop_duplicates(
    subset=["loan_id", "month_number"], keep="first"
)

print(f"\nCleaned borrowers: {len(borrowers):,}")
print(f"Cleaned loans:     {len(loans):,}")
print(f"Cleaned payment:   {len(payment_history):,}")

# ── 4. Feature Engineering -- Payment History ─────────────────────────────────
# payment_history has: loan_id, month_number, payment_due_date, amount_due,
# amount_paid, days_delinquent (0 = on time)

ph_agg = payment_history.groupby("loan_id").agg(
    on_time_count=("days_delinquent", lambda x: (x.fillna(999) == 0).sum()),
    total_days_delinquent=("days_delinquent", lambda x: x.fillna(0).sum()),
    n_months=("month_number", "count"),
).reset_index()

ph_agg["on_time_rate"] = ph_agg["on_time_count"] / ph_agg["n_months"].replace(0, np.nan)
ph_agg["delinquent_rate"] = 1 - ph_agg["on_time_rate"]

# ── 5. Join Everything ──────────────────────────────────────────────────────

df = loans.merge(
    borrowers[
        [
            "borrower_id",
            "credit_score",
            "annual_income",
            "employment_years",
            "debt_to_income_ratio",
            "num_existing_loans",
            "previous_defaults",
            "months_since_last_inquiry",
            "state",
        ]
    ],
    on="borrower_id",
    how="left",
)
df = df.merge(ph_agg, on="loan_id", how="left")

# Encode categoricals
for col in ["loan_type", "state"]:
    df[col] = df[col].fillna("unknown")
    le = LabelEncoder()
    df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))

# Fill missing payment agg with 0 (no history = assume on time)
df["on_time_rate"] = df["on_time_rate"].fillna(1.0)
df["delinquent_rate"] = df["delinquent_rate"].fillna(0.0)
df["total_days_delinquent"] = df["total_days_delinquent"].fillna(0)

print(f"\nJoined dataset: {df.shape}")

# ── 6. Prepare Train / Eval ──────────────────────────────────────────────────

feature_cols = [
    "principal",
    "interest_rate",
    "term_months",
    "monthly_payment",
    "payment_to_income_ratio",
    "credit_score",
    "annual_income",
    "employment_years",
    "debt_to_income_ratio",
    "num_existing_loans",
    "previous_defaults",
    "months_since_last_inquiry",
    "loan_type_enc",
    "state_enc",
    "on_time_rate",
    "delinquent_rate",
    "total_days_delinquent",
]

train_df = df.merge(train_labels, on="loan_id", how="inner")
eval_df = df.merge(eval_ids, on="loan_id", how="inner")

X_train = train_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
y_train_reg = train_df["days_to_early_default"]
y_train_cls = train_df["default_flag"]
X_eval = eval_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)

print(f"X_train: {X_train.shape}  |  X_eval: {X_eval.shape}")

# ── 7. Train Models ─────────────────────────────────────────────────────────

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_eval_s = scaler.transform(X_eval)

reg_model = LinearRegression()
reg_model.fit(X_train_s, y_train_reg)
train_pred_reg = reg_model.predict(X_train_s)
print(f"\nTrain RMSE (days): {np.sqrt(mean_squared_error(y_train_reg, train_pred_reg)):.1f}")
print(f"Train R2:          {r2_score(y_train_reg, train_pred_reg):.4f}")

cls_model = LogisticRegression(max_iter=1000, random_state=42)
cls_model.fit(X_train_s, y_train_cls)
train_proba_cls = cls_model.predict_proba(X_train_s)[:, 1]
print(f"\nTrain AUC: {roc_auc_score(y_train_cls, train_proba_cls):.4f}")

# ── 8. Generate Submission ───────────────────────────────────────────────────

eval_pred_days = np.round(reg_model.predict(X_eval_s)).astype(int)
eval_pred_days = np.clip(eval_pred_days, 1, 3650)  # reasonable range
eval_pred_flag = (cls_model.predict_proba(X_eval_s)[:, 1] >= 0.5).astype(int)

submission = pd.DataFrame({
    "loan_id": eval_df["loan_id"],
    "predicted_days_to_default": eval_pred_days,
    "predicted_default_flag": eval_pred_flag,
})
sub_path = os.path.join(DATA_DIR, "submission.csv")
submission.to_csv(sub_path, index=False)
print(f"\nSubmission written: {len(submission):,} rows")

# ── 9. Score Against Answer Key ─────────────────────────────────────────────

scored = submission.merge(answer_key, on="loan_id", how="inner")

y_true_reg = scored["days_to_early_default"].values
y_pred_reg = scored["predicted_days_to_default"].values.astype(float)
rmse = np.sqrt(mean_squared_error(y_true_reg, y_pred_reg))
mae = mean_absolute_error(y_true_reg, y_pred_reg)
r2 = r2_score(y_true_reg, y_pred_reg)
print(f"\n=== Regression (days_to_early_default) ===")
print(f"RMSE: {rmse:.1f} days")
print(f"MAE:  {mae:.1f} days")
print(f"R2:   {r2:.4f}")

y_true_cls = scored["default_flag"].values
y_pred_cls = scored["predicted_default_flag"].values
auc_roc = roc_auc_score(y_true_cls, y_pred_cls)
f1 = f1_score(y_true_cls, y_pred_cls)
print(f"\n=== Classification (default_flag) ===")
print(f"AUC-ROC: {auc_roc:.4f}")
print(f"F1:      {f1:.4f}")
print("\nCompare to benchmark.txt -- can you beat it?")
