"""Generates the three charts embedded in README.md, into assets/.

Run locally after `oof/` has the model arrays this script reads.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 10

# ---------------------------------------------------------------------------
# 1. Score progression across the project
# ---------------------------------------------------------------------------
steps = [
    ("CatBoost\nbaseline", 0.94178),
    ("CatBoost\n5-fold", 0.94205),
    ("+ target\nencoding", 0.94237),
    ("+ digits,\nflags, counts", 0.94366),
    ("CatBoost\ntuned, 10-fold", 0.94609),
    ("LightGBM\n(full)", 0.94617),
    ("Blend\nCat+LGBM", 0.94620),
    ("LightGBM\n(te view)", 0.94608),
    ("LightGBM\n(raw view)", 0.94328),
    ("LightGBM\n(te + org-mean)", 0.94608),
    ("Blend\n3-way", 0.946225),
    ("Blend\n4-way", 0.94623),
    ("LightGBM\n+ window enc.", 0.94617),
    ("LightGBM\n+ trigram TE", 0.94616),
    ("LightGBM\n+ orig rows", 0.94609),
]
labels = [s[0] for s in steps]
scores = [s[1] for s in steps]

fig, ax = plt.subplots(figsize=(13, 5))
x = np.arange(len(steps))
colors = ["#f28e2b" if s < 0.944 else "#4e79a7" for s in scores]
ax.bar(x, scores, color=colors, width=0.65)
ax.set_ylim(0.930, 0.9475)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel("OOF ROC AUC")
ax.set_title("Score progression: a steep climb, then a long, flat ceiling")
ax.axhline(0.9461, color="#999999", linestyle="--", linewidth=1)
ax.text(len(steps) - 1, 0.9463, "~0.9461-0.9463 ceiling", ha="right",
        fontsize=8, color="#666666")
for i, s in enumerate(scores):
    ax.text(i, s + 0.0004, f"{s:.4f}", ha="center", fontsize=6.5, rotation=90)
plt.tight_layout()
plt.savefig("assets/score_progression.png")
plt.close()
print("Wrote assets/score_progression.png")

# ---------------------------------------------------------------------------
# 2. Income distribution with generator artifacts
# ---------------------------------------------------------------------------
train = pd.read_csv("data/train.csv")
income = train["Annual_Income_USD"]

fig, ax = plt.subplots(figsize=(11, 5))
ax.hist(income, bins=300, color="#4e79a7", alpha=0.85)
ax.axvline(30000, color="#e15759", linestyle="--", linewidth=1.5)
ax.text(30000, ax.get_ylim()[1] * 0.92, " $30K spike", color="#e15759", fontsize=9)
ax.axvspan(38000, 42000, color="#f28e2b", alpha=0.35)
ax.text(40000, ax.get_ylim()[1] * 0.78, "dead zone\n$38K-$42K", color="#b8710a",
        fontsize=8, ha="center")
ax.axvline(170537, color="#59a14f", linestyle="--", linewidth=1.5)
ax.text(170537, ax.get_ylim()[1] * 0.5, " cliff\n $170,537", color="#3d7a37", fontsize=8)
ax.set_xlabel("Annual_Income_USD")
ax.set_ylabel("Count")
ax.set_title("Income distribution: generator fingerprints, not a natural population")
plt.tight_layout()
plt.savefig("assets/income_artifacts.png")
plt.close()
print("Wrote assets/income_artifacts.png")

# ---------------------------------------------------------------------------
# 3. Correlation heatmap between models' OOF predictions
# ---------------------------------------------------------------------------
models = ["catboost_tuned", "lgbm", "lgbm_te", "lgbm_te_org", "mlp", "logreg"]
display_names = ["CatBoost", "LightGBM\n(full)", "LightGBM\n(te)", "LightGBM\n(te+org)", "MLP", "LogReg"]
oof = {m: np.load(f"oof/oof_{m}.npy") for m in models}

n = len(models)
corr = np.eye(n)
for i in range(n):
    for j in range(n):
        if i != j:
            corr[i, j] = spearmanr(oof[models[i]], oof[models[j]]).statistic

fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(corr, cmap="RdYlGn", vmin=0.97, vmax=1.0)
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(display_names, fontsize=8, rotation=45, ha="right")
ax.set_yticklabels(display_names, fontsize=8)
for i in range(n):
    for j in range(n):
        ax.text(j, i, f"{corr[i, j]:.3f}", ha="center", va="center", fontsize=7.5,
                 color="black")
ax.set_title("Spearman correlation between models' OOF predictions")
fig.colorbar(im, ax=ax, shrink=0.8, label="correlation")
plt.tight_layout()
plt.savefig("assets/model_correlation.png")
plt.close()
print("Wrote assets/model_correlation.png")
