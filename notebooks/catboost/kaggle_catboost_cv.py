# Paste this into a Kaggle notebook cell.
# Local twin: ../train_cv.py (same logic, local paths, imports prepare.py).
# Result: OOF AUC 0.94205, public LB 0.94192.

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

DATA = "/kaggle/input/competitions/playground-series-s6e9"
CAT_COLS = ["Gender", "City_Type", "Current_Car_Type"]
N_FOLDS = 5
PARAMS = {
    "iterations": 6000,
    "learning_rate": 0.03,
    "depth": 4,
    "early_stopping_rounds": 100,
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


train = prepare(pd.read_csv(f"{DATA}/train.csv"))
test = pd.read_csv(f"{DATA}/test.csv")
test_ids = test["id"]
X_test = prepare(test)

X = train.drop(columns=["Will_Buy_EV"])
y = train["Will_Buy_EV"]

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
oof = np.zeros(len(X))
test_pred = np.zeros(len(X_test))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    model = CatBoostClassifier(**PARAMS, cat_features=CAT_COLS, verbose=0)
    model.fit(X.iloc[tr_idx], y.iloc[tr_idx], eval_set=(X.iloc[va_idx], y.iloc[va_idx]))
    oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
    test_pred += model.predict_proba(X_test)[:, 1] / N_FOLDS
    print(f"Fold {fold}: AUC={roc_auc_score(y.iloc[va_idx], oof[va_idx]):.5f}  best_iter={model.get_best_iteration()}")

print(f"OOF AUC: {roc_auc_score(y, oof):.5f}")

pd.DataFrame({"id": test_ids, "Will_Buy_EV": test_pred}).to_csv("submission.csv", index=False)
np.save("oof_catboost.npy", oof)
