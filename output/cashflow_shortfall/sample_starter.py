"""
Challenge 3: Small Business Cash Flow Shortfall -- Sample Starter
Pipeline: load → join → clean → feature engineer → train → predict → submit → score
Dual targets: regression (shortfall amount) + classification (shortfall flag)
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

businesses = pd.read_csv(os.path.join(DATA_DIR, "businesses.csv"))
financials = pd.read_csv(os.path.join(DATA_DIR, "financials.csv"))
loans = pd.read_csv(os.path.join(DATA_DIR, "loans.csv"))
train_labels = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
eval_ids = pd.read_csv(os.path.join(DATA_DIR, "eval.csv"))
answer_key = pd.read_csv(os.path.join(DATA_DIR, "answer_key.csv"))

print(f"Businesses:  {businesses.shape}")
print(f"Financials:  {financials.shape}")
print(f"Loans:       {loans.shape}")
print(f"Train:       {train_labels.shape}")
print(f"Eval:        {eval_ids.shape}")
print(f"\nShortfall rate (train): {train_labels['shortfall_flag'].mean():.1%}")

# ── 2. Quick EDA ────────────────────────────────────────────────────────────

print("\n--- Financials nulls ---")
print(financials.isnull().sum())
print(f"\n--- Months available ---")
print(financials["reporting_month"].value_counts().sort_index())

biz_with_loans = loans["business_id"].nunique()
total_biz = businesses["business_id"].nunique()
print(f"\nBusinesses with loans: {biz_with_loans:,} / {total_biz:,} ({biz_with_loans/total_biz:.0%})")

orphans = financials[~financials["business_id"].isin(businesses["business_id"])]
print(f"Orphaned rows in financials: {len(orphans)}")

# ── 3. Clean ────────────────────────────────────────────────────────────────


def clean_str(s):
    return str(s).strip().lower() if pd.notna(s) else s


businesses["industry"] = businesses["industry"].apply(clean_str)
businesses["state"] = businesses["state"].apply(clean_str)
loans["loan_type"] = loans["loan_type"].apply(clean_str)

valid_biz_ids = set(businesses["business_id"].dropna())
financials = financials[financials["business_id"].isin(valid_biz_ids)]
loans = loans[loans["business_id"].isin(valid_biz_ids)]

businesses = businesses.drop_duplicates(subset=["business_id"], keep="first")
financials = financials.drop_duplicates(
    subset=["business_id", "reporting_month"], keep="first"
)

print(f"\nCleaned businesses:  {len(businesses):,}")
print(f"Cleaned financials:  {len(financials):,}")
print(f"Cleaned loans:       {len(loans):,}")

# ── 4. Feature Engineering -- Financials ────────────────────────────────────

fin_cols = [
    "monthly_revenue", "monthly_expenses", "accounts_receivable",
    "accounts_payable", "cash_on_hand",
]

latest = financials[financials["reporting_month"] == "2025-12"].copy()
latest = latest.rename(columns={c: c + "_latest" for c in fin_cols})
latest = latest[["business_id"] + [c + "_latest" for c in fin_cols]]

fin_avg = financials.groupby("business_id")[fin_cols].mean()
fin_avg.columns = [c + "_avg" for c in fin_cols]
fin_avg = fin_avg.reset_index()

earliest = financials[financials["reporting_month"] == "2025-10"].copy()
trend = latest[["business_id", "monthly_revenue_latest"]].merge(
    earliest[["business_id", "monthly_revenue"]].rename(
        columns={"monthly_revenue": "revenue_earliest"}
    ),
    on="business_id",
    how="inner",
)
trend["revenue_trend"] = trend["monthly_revenue_latest"] - trend["revenue_earliest"]
trend = trend[["business_id", "revenue_trend"]]

ratios = latest[
    ["business_id", "monthly_revenue_latest", "monthly_expenses_latest", "cash_on_hand_latest"]
].copy()
ratios["revenue_expense_ratio"] = (
    ratios["monthly_revenue_latest"] / ratios["monthly_expenses_latest"].replace(0, np.nan)
)
ratios["cash_runway_months"] = (
    ratios["cash_on_hand_latest"] / ratios["monthly_expenses_latest"].replace(0, np.nan)
)
ratios = ratios[["business_id", "revenue_expense_ratio", "cash_runway_months"]]

print(f"\nLatest month features:  {latest.shape}")
print(f"Average features:       {fin_avg.shape}")
print(f"Trend features:         {trend.shape}")

# ── 5. Feature Engineering -- Loans ─────────────────────────────────────────

loan_agg = loans.groupby("business_id").agg(
    num_loans=("loan_id", "count"),
    total_outstanding=("outstanding_balance", "sum"),
    total_monthly_payment=("monthly_payment", "sum"),
    max_utilization=("credit_line_utilization_pct", "max"),
    avg_interest_rate=("interest_rate", "mean"),
).reset_index()

print(f"Loan features: {loan_agg.shape}")

# ── 6. Join Everything ──────────────────────────────────────────────────────

le = LabelEncoder()
businesses["industry"] = businesses["industry"].fillna("unknown")
businesses["industry_enc"] = le.fit_transform(businesses["industry"])

df = businesses[["business_id", "years_in_business", "num_employees", "industry_enc"]].copy()
df = df.merge(latest, on="business_id", how="left")
df = df.merge(fin_avg, on="business_id", how="left")
df = df.merge(trend, on="business_id", how="left")
df = df.merge(ratios, on="business_id", how="left")
df = df.merge(loan_agg, on="business_id", how="left")

df["has_loan"] = df["num_loans"].notna().astype(int)
loan_fill_cols = [
    "num_loans", "total_outstanding", "total_monthly_payment",
    "max_utilization", "avg_interest_rate",
]
df[loan_fill_cols] = df[loan_fill_cols].fillna(0)

print(f"\nJoined dataset: {df.shape}")
print(f"Nulls remaining:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# ── 7. Prepare Train / Eval Sets ───────────────────────────────────────────

feature_cols = [
    "years_in_business", "num_employees", "industry_enc",
    "monthly_revenue_latest", "monthly_expenses_latest",
    "accounts_receivable_latest", "accounts_payable_latest",
    "cash_on_hand_latest",
    "monthly_revenue_avg", "monthly_expenses_avg",
    "accounts_receivable_avg", "accounts_payable_avg", "cash_on_hand_avg",
    "revenue_trend", "revenue_expense_ratio", "cash_runway_months",
    "has_loan", "num_loans", "total_outstanding", "total_monthly_payment",
    "max_utilization", "avg_interest_rate",
]

train_df = df.merge(train_labels, on="business_id", how="inner")
eval_df = df.merge(eval_ids, on="business_id", how="inner")

X_train = train_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
y_train_reg = train_df["cashflow_shortfall_amount"]
y_train_cls = train_df["shortfall_flag"]
X_eval = eval_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)

print(f"\nX_train: {X_train.shape}  |  X_eval: {X_eval.shape}")

# ── 8. Train Models ─────────────────────────────────────────────────────────

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_eval_s = scaler.transform(X_eval)

reg_model = LinearRegression()
reg_model.fit(X_train_s, y_train_reg)
train_pred_reg = reg_model.predict(X_train_s)
print(f"\nTrain RMSE: ${np.sqrt(mean_squared_error(y_train_reg, train_pred_reg)):,.2f}")
print(f"Train R2:   {r2_score(y_train_reg, train_pred_reg):.4f}")

cls_model = LogisticRegression(max_iter=1000, random_state=42)
cls_model.fit(X_train_s, y_train_cls)
train_proba_cls = cls_model.predict_proba(X_train_s)[:, 1]
print(f"Train AUC:  {roc_auc_score(y_train_cls, train_proba_cls):.4f}")

# ── 9. Generate Submission ──────────────────────────────────────────────────

eval_pred_amount = reg_model.predict(X_eval_s)
eval_pred_flag = (cls_model.predict_proba(X_eval_s)[:, 1] >= 0.5).astype(int)

submission = pd.DataFrame({
    "business_id": eval_df["business_id"],
    "predicted_shortfall_amount": np.round(eval_pred_amount, 2),
    "predicted_shortfall_flag": eval_pred_flag,
})
submission.to_csv(os.path.join(DATA_DIR, "submission.csv"), index=False)
print(f"\nSubmission written: {len(submission):,} rows")

# ── 10. Score Against Answer Key ────────────────────────────────────────────

scored = submission.merge(answer_key, on="business_id", how="inner")

y_true_reg = scored["cashflow_shortfall_amount"].values
y_pred_reg = scored["predicted_shortfall_amount"].values

rmse = np.sqrt(mean_squared_error(y_true_reg, y_pred_reg))
mae = mean_absolute_error(y_true_reg, y_pred_reg)
r2 = r2_score(y_true_reg, y_pred_reg)

print("\n=== Regression (cashflow_shortfall_amount) ===")
print(f"RMSE: ${rmse:,.2f}")
print(f"MAE:  ${mae:,.2f}")
print(f"R2:   {r2:.4f}")

y_true_cls = scored["shortfall_flag"].values
y_pred_cls = scored["predicted_shortfall_flag"].values

auc_roc = roc_auc_score(y_true_cls, y_pred_cls)
f1 = f1_score(y_true_cls, y_pred_cls)

print(f"\n=== Classification (shortfall_flag) ===")
print(f"AUC-ROC: {auc_roc:.4f}")
print(f"F1:      {f1:.4f}")
print("\nCompare to benchmark.txt -- can you beat it?")
