"""Blend several submission files in rank space.

AUC only cares about the order of the predictions, so averaging ranks is
more robust than averaging raw probabilities: a model whose scores are
systematically higher cannot drag the average around.

Weights are equal by default. That is the honest choice when the models
score the same on OOF - there is no evidence to favour either one.
"""

import sys

import pandas as pd
from scipy.stats import rankdata, spearmanr

INPUTS = {
    "submissions/catboost/catboost_cv10_tuned_oof94609.csv": 0.5,
    "submissions/lgbm/lgbm_cv10_oof94617.csv": 0.5,
}
OUTPUT = "submissions/blend/blend_cat50_lgbm50_oof946199.csv"

frames = {path: pd.read_csv(path) for path in INPUTS}

ids = None
for path, df in frames.items():
    if ids is None:
        ids = df["id"].values
    elif not (df["id"].values == ids).all():
        sys.exit(f"ids in {path} are not aligned with the others")

blend = 0.0
for path, weight in INPUTS.items():
    values = frames[path]["Will_Buy_EV"].values
    blend = blend + weight * (rankdata(values) / len(values))

paths = list(INPUTS)
for i, a in enumerate(paths):
    for b in paths[i + 1:]:
        rho = spearmanr(frames[a]["Will_Buy_EV"], frames[b]["Will_Buy_EV"]).statistic
        print(f"Spearman {a.split('/')[-1]} vs {b.split('/')[-1]}: {rho:.5f}")

pd.DataFrame({"id": ids, "Will_Buy_EV": blend}).to_csv(OUTPUT, index=False)
print(f"\nWrote {OUTPUT} ({len(ids):,} rows)")
