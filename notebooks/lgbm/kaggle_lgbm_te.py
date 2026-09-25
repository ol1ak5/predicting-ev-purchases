# Paste this into a Kaggle notebook cell and run.
#
# View "te": no digits, no artifact flags. Instead, triple target encoding —
# every key encoded at three smoothing strengths at once (auto, 10, 100), so
# the model can pick whichever resolution helps at each split. This is the
# recipe behind the best public single-model notebook (OOF 0.94626).
#
# Companion run: kaggle_lgbm_raw.py. Already run: "full" view, OOF 0.94617
# (see experiments.jsonl for its params).

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import TargetEncoder
from scipy.stats import rankdata
import lightgbm as lgb

DATA = "/kaggle/input/competitions/playground-series-s6e9"
CAT_COLS = ["Gender", "City_Type", "Current_Car_Type"]
SMOOTH_KEYS = ["income_exact", "income100", "income1000", "commute10", "age_int"]
TE_COLS = CAT_COLS + SMOOTH_KEYS
SMOOTHINGS = ["auto", 10.0, 100.0]
N_FOLDS = 10
SEED = 42
PARAMS = dict(
    n_estimators=20000,
    learning_rate=0.02,
    max_depth=5,
    num_leaves=32,
    min_child_samples=10,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.3,
    reg_alpha=0.071,
    reg_lambda=2.0,
    max_bin=1024,
    feature_pre_filter=False,
    n_jobs=-1,
    verbose=-1,
)


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
    df = df.copy()
    inc = df["Annual_Income_USD"].astype("int64")
    df["income_exact"] = inc.astype(str)
    df["income100"] = (inc // 100).astype(str)
    df["income1000"] = (inc // 1000).astype(str)
    df["commute10"] = np.floor(df["Daily_Commute_km"] / 10).astype(int).astype(str)
    df["age_int"] = df["Age"].astype(str)
    return df


def build(df):
    return add_smooth_keys(prepare(df))


rank = lambda v: rankdata(v) / len(v)

train = build(pd.read_csv(f"{DATA}/train.csv"))
test_raw = pd.read_csv(f"{DATA}/test.csv")
test_ids = test_raw["id"]
X_test_raw = build(test_raw)

X = train.drop(columns=["Will_Buy_EV"])
y = train["Will_Buy_EV"]

constant = [c for c in X.columns if X[c].nunique() <= 1]
X = X.drop(columns=constant)
X_test_raw = X_test_raw.drop(columns=constant)
print(f"VIEW=te | features: {X.shape[1]} | TE keys: {len(TE_COLS)} x {len(SMOOTHINGS)} smoothings")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof = np.zeros(len(X))
test_pred = np.zeros(len(X_test_raw))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    X_tr, y_tr = X.iloc[tr_idx].copy(), y.iloc[tr_idx]
    X_va, y_va = X.iloc[va_idx].copy(), y.iloc[va_idx]
    X_te = X_test_raw.copy()

    # One encoder per smoothing level, each adding its own set of columns.
    for smooth in SMOOTHINGS:
        tag = str(smooth).replace(".0", "")
        te = TargetEncoder(cv=5, smooth=smooth)
        enc_tr = te.fit_transform(X_tr[TE_COLS], y_tr)
        enc_va = te.transform(X_va[TE_COLS])
        enc_te = te.transform(X_te[TE_COLS])
        for i, col in enumerate(TE_COLS):
            X_tr[f"{col}_te{tag}"] = enc_tr[:, i].astype("float32")
            X_va[f"{col}_te{tag}"] = enc_va[:, i].astype("float32")
            X_te[f"{col}_te{tag}"] = enc_te[:, i].astype("float32")

    # The original string columns are replaced by their encodings.
    X_tr = X_tr.drop(columns=TE_COLS)
    X_va = X_va.drop(columns=TE_COLS)
    X_te = X_te.drop(columns=TE_COLS)

    model = lgb.LGBMClassifier(random_state=SEED, **PARAMS)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(300, verbose=False)],
    )

    pred_va = model.predict_proba(X_va)[:, 1]
    oof[va_idx] = rank(pred_va)
    test_pred += rank(model.predict_proba(X_te)[:, 1]) / N_FOLDS

    print(f"Fold {fold}: AUC={roc_auc_score(y_va, pred_va):.5f}  best_iter={model.best_iteration_}")

print(f"OOF AUC: {roc_auc_score(y, oof):.5f}")

pd.DataFrame({"id": test_ids, "Will_Buy_EV": test_pred}).to_csv("submission.csv", index=False)
np.save("oof_lgbm_te.npy", oof)
np.save("test_lgbm_te.npy", test_pred)
