# Paste this into a Kaggle notebook cell and run.
#
# Same as kaggle_lgbm.py (the "full" view, our best model, OOF 0.94617 /
# LB 0.94624), plus three new target-encoded keys: the three columns that
# dominate the buying decision (Environmental_Concern_Level, Subsidy_Available,
# Range_Anxiety_Level) joined together, not encoded one at a time.
#
# Why: encoding them separately hands the tree three numbers and leaves it to
# find the right combination through splits. With colsample_bytree=0.3, any
# single tree only sees 30% of the columns, so it is not guaranteed to see
# all three at once. A joint key (e.g. "3_1_0" for concern=3, subsidy=yes,
# anxiety=low) hands the tree the exact combination's own buy rate directly,
# regardless of which columns that particular tree happened to sample.

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
INTERACTION_KEYS = ["concern_x_anxiety", "concern_x_subsidy", "attitude_trigram"]
TE_COLS = CAT_COLS + SMOOTH_KEYS + INTERACTION_KEYS
DIGIT_SOURCES = [
    "Age",
    "Annual_Income_USD",
    "Daily_Commute_km",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
]
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


def add_interaction_keys(df):
    df = df.copy()
    concern = df["Environmental_Concern_Level"].fillna(-1).astype(int).astype(str)
    subsidy = df["Subsidy_Available"].astype(str)
    anxiety = df["Range_Anxiety_Level"].fillna(-1).astype(int).astype(str)
    df["concern_x_anxiety"] = concern + "_" + anxiety
    df["concern_x_subsidy"] = concern + "_" + subsidy
    df["attitude_trigram"] = concern + "_" + subsidy + "_" + anxiety
    return df


def add_digits(df):
    df = df.copy()
    for col in DIGIT_SOURCES:
        for k in range(-1, 6):
            df[f"{col}_d{k}"] = (df[col].fillna(0) // (10 ** k) % 10).astype("int8")
    return df


def add_artifact_flags(df):
    df = df.copy()
    df["is_30k_spike"] = (df["Annual_Income_USD"] == 30000.0).astype("int8")
    df["is_income_cliff"] = (df["Annual_Income_USD"] >= 170537.0).astype("int8")
    df["is_dead_zone"] = (
        (df["Annual_Income_USD"] >= 38000.0) & (df["Annual_Income_USD"] <= 42000.0)
    ).astype("int8")
    df["is_env_hater"] = (df["Environmental_Concern_Level"] == 1).astype("int8")
    return df


def add_counts(train_df, test_df, cols):
    train_df, test_df = train_df.copy(), test_df.copy()
    both = pd.concat([train_df[cols], test_df[cols]], ignore_index=True)
    for col in cols:
        freq = both[col].value_counts(normalize=True)
        cnt = both[col].value_counts()
        for d in (train_df, test_df):
            d[f"{col}_fe"] = d[col].map(freq).astype("float32").fillna(0)
            d[f"{col}_cnt"] = d[col].map(cnt).astype("int32").fillna(0)
    return train_df, test_df


def build(df):
    return add_artifact_flags(add_digits(add_interaction_keys(add_smooth_keys(prepare(df)))))


rank = lambda v: rankdata(v) / len(v)

train = build(pd.read_csv(f"{DATA}/train.csv"))
test_raw = pd.read_csv(f"{DATA}/test.csv")
test_ids = test_raw["id"]
X_test_raw = build(test_raw)

train, X_test_raw = add_counts(train, X_test_raw, TE_COLS)

X = train.drop(columns=["Will_Buy_EV"])
y = train["Will_Buy_EV"]

constant = [c for c in X.columns if X[c].nunique() <= 1]
X = X.drop(columns=constant)
X_test_raw = X_test_raw.drop(columns=constant)
print(f"VIEW=trigram | features: {X.shape[1]} | TE keys: {len(TE_COLS)} (incl. {len(INTERACTION_KEYS)} interaction keys)")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof = np.zeros(len(X))
test_pred = np.zeros(len(X_test_raw))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    X_tr, y_tr = X.iloc[tr_idx].copy(), y.iloc[tr_idx]
    X_va, y_va = X.iloc[va_idx].copy(), y.iloc[va_idx]
    X_te = X_test_raw.copy()

    te = TargetEncoder(cv=5, smooth="auto")
    X_tr[TE_COLS] = te.fit_transform(X_tr[TE_COLS], y_tr)
    X_va[TE_COLS] = te.transform(X_va[TE_COLS])
    X_te[TE_COLS] = te.transform(X_te[TE_COLS])

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
np.save("oof_lgbm_trigram.npy", oof)
np.save("test_lgbm_trigram.npy", test_pred)
