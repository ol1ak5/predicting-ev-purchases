# Paste this into a Kaggle notebook cell.
#
# A neural network instead of trees. It will almost certainly score worse on
# its own - boosting wins on tabular data - but it learns in a completely
# different way, so it should be far less correlated with CatBoost and
# LightGBM. That decorrelation is the whole point: the blend is what we are
# after, not this model's own score.
#
# Same folds (StratifiedKFold(10, shuffle=True, random_state=42)) so the OOF
# array lines up row for row with the others.
#
# Two things trees do not need but a network does:
#   - every column must be a number (no text)
#   - every column must be on a similar scale

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import TargetEncoder, QuantileTransformer
from sklearn.neural_network import MLPClassifier
from scipy.stats import rankdata

DATA = "/kaggle/input/competitions/playground-series-s6e9"
CAT_COLS = ["Gender", "City_Type", "Current_Car_Type"]
SMOOTH_KEYS = ["income_exact", "income100", "income1000", "commute10", "age_int"]
TE_COLS = CAT_COLS + SMOOTH_KEYS
N_FOLDS = 10
SEED = 42
PARAMS = dict(
    hidden_layer_sizes=(256, 128, 64),
    activation="relu",
    alpha=1e-4,
    batch_size=1024,
    learning_rate_init=1e-3,
    max_iter=60,
    early_stopping=True,
    n_iter_no_change=8,
    validation_fraction=0.1,
    random_state=SEED,
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


# No digit features here. Digits are arbitrary integers with no order
# (a 9 is not "more" than a 1), which trees can split on but a network would
# read as a quantity. They would be noise.

rank = lambda v: rankdata(v) / len(v)

train = add_smooth_keys(prepare(pd.read_csv(f"{DATA}/train.csv")))
test_raw = pd.read_csv(f"{DATA}/test.csv")
test_ids = test_raw["id"]
X_test_raw = add_smooth_keys(prepare(test_raw))

X = train.drop(columns=["Will_Buy_EV"])
y = train["Will_Buy_EV"]
print(f"Features: {X.shape[1]}")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof = np.zeros(len(X))
test_pred = np.zeros(len(X_test_raw))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    X_tr, y_tr = X.iloc[tr_idx].copy(), y.iloc[tr_idx]
    X_va, y_va = X.iloc[va_idx].copy(), y.iloc[va_idx]
    X_te = X_test_raw.copy()

    # 1. Text columns become numbers, exactly as in the tree models.
    te = TargetEncoder(cv=5, smooth="auto")
    X_tr[TE_COLS] = te.fit_transform(X_tr[TE_COLS], y_tr)
    X_va[TE_COLS] = te.transform(X_va[TE_COLS])
    X_te[TE_COLS] = te.transform(X_te[TE_COLS])

    # 2. Put every column on the same scale. Income runs to 188,000 while
    #    concern runs 1-5; without this the network would let income
    #    dominate purely because its numbers are bigger. Fitted on the
    #    training rows only, like any transformer that learns.
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000, random_state=SEED)
    X_tr_s = qt.fit_transform(X_tr)
    X_va_s = qt.transform(X_va)
    X_te_s = qt.transform(X_te)

    model = MLPClassifier(**PARAMS)
    model.fit(X_tr_s, y_tr)

    pred_va = model.predict_proba(X_va_s)[:, 1]
    oof[va_idx] = rank(pred_va)
    test_pred += rank(model.predict_proba(X_te_s)[:, 1]) / N_FOLDS

    print(f"Fold {fold}: AUC={roc_auc_score(y_va, pred_va):.5f}  epochs={model.n_iter_}")

print(f"OOF AUC: {roc_auc_score(y, oof):.5f}")

# What the network actually learned: one weight matrix per layer.
print("\nWeight matrices (rows x columns):")
for i, w in enumerate(model.coefs_):
    print(f"  layer {i}: {w.shape}")
print(f"Total weights: {sum(w.size for w in model.coefs_):,}")

pd.DataFrame({"id": test_ids, "Will_Buy_EV": test_pred}).to_csv("submission.csv", index=False)
np.save("oof_mlp.npy", oof)
np.save("test_mlp.npy", test_pred)
