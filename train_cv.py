import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

from prepare import prepare, CAT_COLS

SAMPLE = None          # None = all rows
N_FOLDS = 5
PARAMS = {
    "iterations": 6000,
    "learning_rate": 0.03,
    "depth": 4,
    "early_stopping_rounds": 100,
}

train = prepare(pd.read_csv("data/train.csv", nrows=SAMPLE))
test = pd.read_csv("data/test.csv")
test_ids = test["id"]
X_test = prepare(test)

X = train.drop(columns=["Will_Buy_EV"])
y = train["Will_Buy_EV"]

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof = np.zeros(len(X))
test_pred = np.zeros(len(X_test))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
    X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

    model = CatBoostClassifier(**PARAMS, cat_features=CAT_COLS, verbose=0)
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va))

    oof[va_idx] = model.predict_proba(X_va)[:, 1]
    test_pred += model.predict_proba(X_test)[:, 1] / N_FOLDS

    print(f"Fold {fold}: AUC={roc_auc_score(y_va, oof[va_idx]):.5f}  best_iter={model.get_best_iteration()}")

auc = roc_auc_score(y, oof)
print(f"OOF AUC: {auc:.5f}")

pd.DataFrame({"id": test_ids, "Will_Buy_EV": test_pred}).to_csv("submissions/catboost/catboost_cv5.csv", index=False)
np.save("oof_catboost.npy", oof)

with open("experiments.jsonl", "a") as f:
    f.write(json.dumps({
        "time": datetime.now().isoformat(timespec="minutes"),
        "model": "catboost_cv5",
        "sample": SAMPLE,
        "params": PARAMS,
        "features": list(X.columns),
        "auc": round(auc, 5),
    }) + "\n")