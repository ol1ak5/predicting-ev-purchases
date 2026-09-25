# Paste this into a Kaggle notebook cell and run.
#
# Same as kaggle_lgbm.py (the "full" view, our best single model so far),
# plus one addition: neighbourhood window encoding for income and commute.
#
# Regular target encoding on income/commute buckets (income_exact, income100,
# income1000) draws a hard line at each bucket edge -- two people a dollar
# apart on opposite sides of a bucket boundary can get very different
# encoded values, even though nothing about them actually differs by much.
# A window encoding instead looks at everyone within +/- R of a person's own
# value and uses their average buy rate -- a smooth, continuous estimate
# with no artificial edges. Multiple radii (small and large R) are computed
# at once so the tree can pick whichever resolution helps.
#
# This mirrors the (also continuous) way income actually enters the
# competition's known data-generating formula, which is why it's worth
# testing on top of the existing feature set rather than another encoding
# of the same discrete buckets.

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

# Window-encoding config: radius in whole dollars for income, in tenths of a
# km for commute (so a radius of 10 there means +/- 1.0 km). WIN_PRIOR is
# how many pseudo-observations of the global mean to blend in -- keeps a
# window with few neighbours from swinging wildly.
INCOME_RADII = [50, 500, 5000]
COMMUTE_RADII = [2, 10]
WIN_PRIOR = 10.0
INNER_FOLDS = 5


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
    # integer versions used only for the window encoder below
    df["_income_int"] = inc
    df["_commute_tenths"] = np.round(df["Daily_Commute_km"] * 10).astype(int)
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
    return add_artifact_flags(add_digits(add_smooth_keys(prepare(df))))


def window_stats(vals, y, val_min, n_bins):
    idx = vals - val_min + 1
    counts = np.bincount(idx, minlength=n_bins).astype(np.int64)
    sums = np.bincount(idx, weights=y, minlength=n_bins).astype(np.float64)
    return np.cumsum(counts), np.cumsum(sums)


def window_lookup(cum_counts, cum_sums, vals, val_min, radius, prior, global_mean):
    idx = vals - val_min + 1
    top = len(cum_counts) - 1
    lo = np.clip(idx - radius - 1, 0, top)
    hi = np.clip(idx + radius, 0, top)
    cnt = cum_counts[hi] - cum_counts[lo]
    s = cum_sums[hi] - cum_sums[lo]
    return ((s + prior * global_mean) / (cnt + prior)).astype("float32")


def add_window_encodings(X_tr, y_tr, X_va, X_te, col, radii, prefix, seed):
    vals_tr = X_tr[col].to_numpy()
    val_min = min(vals_tr.min(), X_va[col].min(), X_te[col].min())
    val_max = max(vals_tr.max(), X_va[col].max(), X_te[col].max())
    n_bins = int(val_max - val_min) + 3
    global_mean = y_tr.mean()
    y_tr_arr = y_tr.to_numpy()

    out_tr = {r: np.zeros(len(X_tr), dtype="float32") for r in radii}
    inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed)
    for inner_tr_idx, inner_va_idx in inner.split(vals_tr, y_tr_arr):
        cc, cs = window_stats(vals_tr[inner_tr_idx], y_tr_arr[inner_tr_idx], val_min, n_bins)
        for r in radii:
            out_tr[r][inner_va_idx] = window_lookup(
                cc, cs, vals_tr[inner_va_idx], val_min, r, WIN_PRIOR, global_mean
            )

    cc_full, cs_full = window_stats(vals_tr, y_tr_arr, val_min, n_bins)
    for r in radii:
        X_tr[f"{prefix}_win{r}"] = out_tr[r]
        X_va[f"{prefix}_win{r}"] = window_lookup(
            cc_full, cs_full, X_va[col].to_numpy(), val_min, r, WIN_PRIOR, global_mean
        )
        X_te[f"{prefix}_win{r}"] = window_lookup(
            cc_full, cs_full, X_te[col].to_numpy(), val_min, r, WIN_PRIOR, global_mean
        )


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
print(f"VIEW=window | features: {X.shape[1]} (+{len(INCOME_RADII) + len(COMMUTE_RADII)} window cols per fold)")

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

    add_window_encodings(X_tr, y_tr, X_va, X_te, "_income_int", INCOME_RADII, "income", SEED + fold)
    add_window_encodings(X_tr, y_tr, X_va, X_te, "_commute_tenths", COMMUTE_RADII, "commute", SEED + fold)

    drop_cols = ["_income_int", "_commute_tenths"]
    X_tr = X_tr.drop(columns=drop_cols)
    X_va = X_va.drop(columns=drop_cols)
    X_te = X_te.drop(columns=drop_cols)

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
np.save("oof_lgbm_window.npy", oof)
np.save("test_lgbm_window.npy", test_pred)
