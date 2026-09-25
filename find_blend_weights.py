"""Find the blend weights by measuring them on the OOF predictions.

Because every model here was trained on the same 10 folds, their OOF arrays
line up row for row and can be compared against the real answers. That means
the blend can be scored offline, without spending a submission.
"""

import itertools

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

MODELS = ["catboost_tuned", "lgbm", "lgbm_te", "lgbm_te_org"]
SHORT_NAME = {"catboost_tuned": "cat", "lgbm": "lgbm", "lgbm_te": "lgbmte", "lgbm_te_org": "lgbmteorg", "mlp": "mlp"}
STEP = 0.05
OUTPUT_DIR = "submissions/blend"

y = (pd.read_csv("data/train.csv")["Will_Buy_EV"] == "Yes").astype(int).values
test_ids = pd.read_csv("data/test.csv")["id"].values

rank = lambda v: rankdata(v) / len(v)

oof = {m: rank(np.load(f"oof/oof_{m}.npy")) for m in MODELS}
test = {m: rank(np.load(f"oof/test_{m}.npy")) for m in MODELS}

print("Individual OOF AUC:")
for m in MODELS:
    print(f"  {m:16} {roc_auc_score(y, oof[m]):.6f}")

print("\nSpearman correlation between models:")
for a, b in itertools.combinations(MODELS, 2):
    corr = spearmanr(oof[a], oof[b]).statistic
    print(f"  {a} / {b}: {corr:.5f}")


def simplex_grid(n, step):
    """All weight tuples of length n, each a multiple of `step`, summing to 1."""
    steps = round(1 / step)
    for combo in itertools.product(range(steps + 1), repeat=n - 1):
        if sum(combo) > steps:
            continue
        last = steps - sum(combo)
        yield tuple(c * step for c in combo) + (last * step,)


results = []
for weights in simplex_grid(len(MODELS), STEP):
    blend = sum(w * oof[m] for w, m in zip(weights, MODELS))
    auc = roc_auc_score(y, blend)
    results.append((weights, auc))

results.sort(key=lambda r: -r[1])

print("\nTop 5 weight combinations:")
for weights, auc in results[:5]:
    w_str = ", ".join(f"{m}={w:.2f}" for m, w in zip(MODELS, weights))
    print(f"  {w_str}  ->  AUC={auc:.6f}")

best_weights, best_auc = results[0]
best_single = max(roc_auc_score(y, oof[m]) for m in MODELS)
print(f"\nBest blend: {dict(zip(MODELS, best_weights))} -> OOF AUC {best_auc:.6f}")
print(f"Gain over the best single model: {best_auc - best_single:+.6f}")

# A flat top-5 means the exact maximum is noise, not a real optimum -- look at
# how much the top combos actually differ before trusting the winner.
spread = best_auc - results[min(4, len(results) - 1)][1]
print(f"Spread across the top 5 combinations: {spread:.6f}")

blend_test = sum(w * test[m] for w, m in zip(best_weights, MODELS))
tag = "_".join(f"{SHORT_NAME[m]}{int(round(w * 100))}" for w, m in zip(best_weights, MODELS) if w > 0)
out_path = f"{OUTPUT_DIR}/blend_{tag}_oof{int(round(best_auc * 1e6))}.csv"
pd.DataFrame({"id": test_ids, "Will_Buy_EV": blend_test}).to_csv(out_path, index=False)
print(f"\nWrote {out_path}")
