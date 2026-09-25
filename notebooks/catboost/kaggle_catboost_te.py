# Paste this into a Kaggle notebook cell.
# Local twin: ../train_cv_te.py (same logic, local paths, imports prepare.py).

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import TargetEncoder
from catboost import CatBoostClassifier

DATA = "/kaggle/input/competitions/playground-series-s6e9"
CAT_COLS = ["Gender", "City_Type", "Current_Car_Type"]
SMOOTH_KEYS = ["income1000", "commute10", "age_int"]
TE_COLS = CAT_COLS + SMOOTH_KEYS
N_FOLDS = 5
PARAMS = {
    "iterations": 10000,
    "learning_rate": 0.03,
    "depth": 4,
    "early_stopping_rounds": 200,
}


def prepare(df):
    df = df.copy()
    df = df.drop(columns=["id"])
    df["Home_Charging_Possible"] = (df["Home_Charging_Possible"] == "Yes").astype(int)
    df["Subsidy_Available"] = (df["Subsidy_Available"] == "Yes").astype(int)
    df["Range_Anxiety_Level"] = df["Range_Anxiety_Level"].map({"Low": 0, "Medium": 1, "High": 2})
    df["Income_x_Concern"] = df["Annual_Income_USD"] * df["Environmental_Concern_Level"]
    df["Income_Per_Car"] = df["Annual_Income_USD"] / df["Number_of_Cars_Owned"]
    if "Will_Buy_EV" in df.columns:
        df["Will_Buy_EV"] = (df["Will_Buy_EV"] == "Yes").astype(int)
    return df


def add_smooth_keys(df):
    """Turn numeric columns into coarse string categories so they can be
    target-encoded: the model then sees the buy rate of each income band."""
    df = df.copy()
    df["income1000"] = np.floor(df["Annual_Income_USD"] / 1000).astype(int).astype(str)
    df["commute10"] = np.floor(df["Daily_Commute_km"] / 10).astype(int).astype(str)
    df["age_int"] = df["Age"].astype(str)
    return df


train = add_smooth_keys(prepare(pd.read_csv(f"{DATA}/train.csv")))
test = pd.read_csv(f"{DATA}/test.csv")
test_ids = test["id"]
X_test_raw = add_smooth_keys(prepare(test))

X = train.drop(columns=["Will_Buy_EV"])
y = train["Will_Buy_EV"]

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof = np.zeros(len(X))
test_pred = np.zeros(len(X_test_raw))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    X_tr, y_tr = X.iloc[tr_idx].copy(), y.iloc[tr_idx]
    X_va, y_va = X.iloc[va_idx].copy(), y.iloc[va_idx]
    X_te = X_test_raw.copy()

    # Fit the encoder on this fold's training rows only, then transform
    # validation and test. That asymmetry is what prevents leakage.
    te = TargetEncoder(cv=5, smooth="auto")
    X_tr[TE_COLS] = te.fit_transform(X_tr[TE_COLS], y_tr)
    X_va[TE_COLS] = te.transform(X_va[TE_COLS])
    X_te[TE_COLS] = te.transform(X_te[TE_COLS])

    # Every column is numeric now, so CatBoost needs no cat_features.
    model = CatBoostClassifier(**PARAMS, verbose=0)
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va))

    oof[va_idx] = model.predict_proba(X_va)[:, 1]
    test_pred += model.predict_proba(X_te)[:, 1] / N_FOLDS

    print(f"Fold {fold}: AUC={roc_auc_score(y_va, oof[va_idx]):.5f}  best_iter={model.get_best_iteration()}")

print(f"OOF AUC: {roc_auc_score(y, oof):.5f}")

pd.DataFrame({"id": test_ids, "Will_Buy_EV": test_pred}).to_csv("submission.csv", index=False)
np.save("oof_catboost_te.npy", oof)
