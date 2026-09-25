import json
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

from prepare import prepare, CAT_COLS

PARAMS = {
    "iterations": 6000,
    "learning_rate": 0.03,
    "depth": 4,
    "early_stopping_rounds": 100,
}

train = pd.read_csv("data/train.csv")
train = prepare(train)

X = train.drop(columns=["Will_Buy_EV"])
y = train["Will_Buy_EV"]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = CatBoostClassifier(
    **PARAMS,
    cat_features=CAT_COLS,
    verbose=200,
)
model.fit(X_train, y_train, eval_set=(X_valid, y_valid))

pred = model.predict_proba(X_valid)[:, 1]
auc = roc_auc_score(y_valid, pred)
print(f"Validation AUC: {auc:.4f}")
print(model.get_feature_importance(prettified=True))

model.save_model("model.cbm")

with open("experiments.jsonl", "a") as f:
    f.write(json.dumps({
        "time": datetime.now().isoformat(timespec="minutes"),
        "params": PARAMS,
        "features": list(X.columns),
        "best_iteration": model.get_best_iteration(),
        "auc": round(auc, 5),
    }) + "\n")
