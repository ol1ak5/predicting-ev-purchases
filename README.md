# Predicting EV Purchases — Kaggle Playground Series S6E9

## Introduction

This repository trains binary classifiers to predict `Will_Buy_EV` — whether
a person will buy an electric vehicle — for [Kaggle's Playground Series
Season 6, Episode 9](https://www.kaggle.com/competitions/playground-series-s6e9).
The data is a synthetic tabular dataset (14 demographic, geographic, and
psychological features per person) generated to resemble a small real-world
survey. The competition is scored by ROC AUC: how well the model *ranks*
buyers above non-buyers, regardless of the exact probabilities it outputs.

The work here goes through the usual arc of a tabular Kaggle problem —
baseline model, cross-validation, feature engineering, hyperparameter
tuning, a second model family, blending — and then a deliberate attempt to
find out *why* the score stopped improving once it did. That last part
(feature "views" designed to decorrelate models, checks against the
original real-world dataset, and a direct measurement of the data's own
noise floor) turned out to matter as much as the modeling itself: see
[Where the ceiling comes from](#11-where-the-ceiling-comes-from).

Best result: a single LightGBM model, 10-fold cross-validated, scoring
**0.94628** on the public leaderboard (OOF 0.94616).

## Repository structure

```
prepare.py              shared feature-preparation function, used by the local scripts below
explore.py              initial EDA: column types, missing values, target balance
train.py                first model: a single 80/20 train/validation split
train_cv.py             upgrades train.py to 5-fold cross-validation
train_cv_te.py          adds target encoding on top of train_cv.py
predict.py              loads a saved model and writes a submission file
blend.py                rank-blends two or more finished submission CSVs
find_blend_weights.py   sweeps blend weights against out-of-fold predictions
experiments.jsonl       one line per run: parameters, feature count, OOF score, public score
notebooks/catboost/     CatBoost scripts written to be pasted into a Kaggle notebook
notebooks/lgbm/         LightGBM scripts, same purpose
notebooks/mlp/          a neural-network baseline used to test blend diversity
```

`data/`, `oof/`, and `submissions/` are gitignored: `data/` holds the
competition's CSVs, `oof/` holds each model's out-of-fold and test-set
prediction arrays (needed to search blend weights without spending a
submission), and `submissions/` holds the CSVs actually uploaded to Kaggle.

The local scripts (`train.py` through `train_cv_te.py`) were the first
experiments, sized to run on a laptop without a GPU. Once the feature set
grew and 10-fold cross-validation became the standard, training moved to
Kaggle's own notebooks (which provide more CPU/RAM and both CatBoost and
LightGBM) — the `notebooks/` scripts are the versions pasted there. Local
work only needs `uv sync` and `uv run python <script>.py`; nothing here
requires a GPU.

## Approach

### 1. Baseline: a single train/validation split

The first model is deliberately simple: one 80/20 split, a handful of raw
features, a shallow CatBoost model (`depth=4`). The goal at this stage is a
working end-to-end pipeline — load data, train, validate, predict, submit —
not a good score. It scored 0.94178 locally and 0.94158 on the public
leaderboard, close enough to confirm the local validation number can be
trusted.

### 2. Cross-validation instead of a single split

A single 80/20 split means the validation score depends on which 20% of
rows happened to land in the validation set — a different split can move
the score by a few thousandths, which is large relative to how close
competitors sit on this leaderboard. Standard k-fold cross-validation
solves this two ways at once: it averages 5 (later 10) different train/
validation splits into one more stable score, and it produces an
**out-of-fold (OOF)** prediction for every training row — each row is
scored by the one fold where it was held out. That full-length OOF array is
what makes blend-weight search possible later without needing new Kaggle
submissions to test it.

### 3. Target encoding: turning numbers into categories

A tree can only split a numeric column on thresholds (`income > 80000`); it
cannot look up "people earning exactly 92-93k buy at 21% historically" the
way a lookup table could. Target encoding builds that lookup table
directly: it replaces a category (or a binned numeric range, like income
rounded to the nearest thousand) with the average target value observed for
that category, and hands the tree that average as a feature.

The risk is leakage: if a row's own label leaks into the average computed
for its category, validation scores look great and the public leaderboard
collapses. The fix used throughout is cross-fitting — every encoder is
fit only on the current fold's training rows and only *applied* to
validation/test rows, via `sklearn.preprocessing.TargetEncoder(cv=...)` —
so the encoding a row receives never depends on that row's own answer.

### 4. Reading the data's own fingerprints: digit features and artifact flags

Plotting the distributions of the numeric columns turned up several sharp,
non-natural patterns: a spike of rows with income at *exactly* $30,000, a
hard ceiling around $170,537 above which behavior looked different, and a
flat "dead zone" between $38,000 and $42,000. These are not patterns a real
population would produce — they look like a data generator clipping or
special-casing values at fixed boundaries. Each was encoded as its own
binary flag, so the model gets the boundary directly instead of having to
discover it from thresholds.

The same reasoning extended further: if a generator is working from
rounded or bucketed inputs, that structure should show up in the low-order
digits of the numeric columns. Splitting every numeric column into its
individual digits (units, tens, hundreds, and the digits after the decimal
point) and handing each digit its own column let the model pick up on
exactly that kind of quantization. This was the single largest
feature-engineering gain measured (OOF 0.94237 → 0.94366 once digits,
flags, and frequency/count encoding were added together).

### 5. Hyperparameter tuning

With the feature set grown from 15 to 58 columns, the shallow `depth=4`
model from step 1 was under-fitting. Retuning `depth` (4 → 6) and the
column-sampling rate — `rsm` in CatBoost, `colsample_bytree` in LightGBM —
gave the single biggest jump measured in this project (OOF +0.0024).
Beyond that point, tuning saturated: sweeping the sampling rate further
(0.2 vs 0.3 vs 0.5) moved the score by about 0.00002, well inside run-to-run
noise, and a deliberately more aggressive LightGBM configuration
(`num_leaves=255`, smaller leaves, more trees) scored *worse* (0.94565).
Both results say the same thing: this configuration had reached the point
where more tuning was not going to move the needle.

### 6. A second model family: LightGBM

CatBoost grows trees symmetrically (every leaf at a given depth splits on
the same rule); LightGBM grows trees leaf-wise, picking whichever leaf
promises the biggest improvement next. Given the same 58 features and the
same 10 folds, that structural difference is enough that the two models
make different mistakes on different rows — which is exactly what a blend
needs. LightGBM turned out to be both the stronger single model (OOF
0.94617, public 0.94624) and, paired with CatBoost, a source of blend
diversity.

### 7. Blending: averaging in rank space, and testing weights offline

Because ROC AUC only depends on the *order* of predictions, averaging raw
probabilities can be distorted by one model's scores running systematically
higher or lower than another's. Converting every model's predictions to
their rank (position if sorted, scaled to [0, 1]) before averaging removes
that distortion. Blend weights are chosen by sweeping them against the OOF
arrays — scoring every candidate weight against the real training labels —
so the weight that will actually be submitted is picked with evidence, not
a submission spent guessing.

### 8. Feature "views": looking for a genuinely different model

CatBoost and LightGBM, on the same 58 columns, ended up correlating at
0.997 — for blending purposes, nearly the same model twice. The next idea
was to keep the algorithm fixed (LightGBM) and change what it saw: three
different column sets ("views"), each trained the same way with the same
10 folds, so a blend of the three views might decorrelate more than two
algorithms on identical columns ever could.

- The **full** view is everything described above (58 columns).
- The **te** view drops the digit/flag/count features entirely and instead
  target-encodes every key at three smoothing strengths at once (light,
  medium, heavy), letting the tree pick whichever resolution helps. It
  matched the full view's OOF (0.9461) with a third of the columns.
- The **raw** view keeps only the 14 original columns plus two ratio
  features, deliberately starved of engineering, to force a different kind
  of relationship. It scored meaningfully lower (OOF 0.9433) — too weak to
  help a blend regardless of how differently it errs.

The disappointing result: even the *te* view, despite having almost no
overlap in which columns it used, correlated with the full view at 0.998 —
more correlated than CatBoost and LightGBM were with each other on
identical columns. Changing the columns barely changed what the model
learned, which is itself informative (see below).

### 9. Does the original real-world dataset help?

The competition's synthetic data is generated to resemble an independently
published, real 10,000-row survey of the same population. Two ways of
using that real data were tested:

- **Per-category real-world averages.** For every column, the average
  real-world buy rate for each of its values was computed from the
  10,000-row dataset and added as a feature. It made no measurable
  difference (OOF 0.94608 vs 0.94607 without it) — the target encoding
  already computed on the much larger synthetic training set estimates the
  same underlying rate more precisely, using 65x more rows. The real-data
  average is not wrong, it is simply a less precise version of information
  the model already had.
- **Row-level matching.** If any synthetic row were a near-exact copy of a
  real row, that row's true outcome would no longer be noise, it would be
  recoverable. Matching all 668,665+286,571 competition rows against the
  10,000 real rows, first on exact values and then on progressively looser
  tolerances, found zero exact matches and only chance-level near matches
  — no evidence of row-level duplication to exploit.

Both checks came back negative, which is itself a useful (negative)
result: whatever separates the strongest leaderboard scores from this
one, it is not obviously recoverable from the published real-world data.

### 10. Benchmarking against a stronger public result

A public notebook using XGBoost and roughly the same feature families as
this project scored 0.94646 on the leaderboard, noticeably above the
0.94624 reached so far. Rather than copy its ~340-column pipeline wholesale,
three of its individually-motivated ideas were isolated and tested one at a
time against this project's own best model, to find out which specific
mechanism, if any, explains the gap.

- **Neighbourhood window encoding.** Target encoding on income/commute
  *buckets* draws a hard edge at every bucket boundary — two people a
  dollar apart on opposite sides of a boundary can get very different
  encoded values. Window encoding instead averages the buy rate of everyone
  within a radius of a person's own value (several radii at once,
  cross-fitted the same way as target encoding), producing a smooth
  estimate with no artificial edges. It made no OOF difference (0.94617,
  identical to the plain model) — the existing digit and multi-scale bucket
  features evidently already captured enough of that smoothness.
- **Joint encoding of the three attitude columns.** Environmental concern,
  subsidy availability, and range anxiety drive most of the buying
  decision, and their combined effect is not additive. Encoding all three
  together as one key, instead of one column at a time, hands the tree
  their interaction directly instead of leaving it to rediscover through
  splits — useful in principle since each tree only samples 30% of columns
  at a time (`colsample_bytree=0.3`), so no single tree is guaranteed to
  see all three columns together. This also made no measurable difference
  (0.94616) — with thousands of trees built across 10 folds, the model
  evidently had enough chances across trees, if not within any single one,
  to learn the interaction anyway.
- **Concatenating the original dataset into training.** The 10,000 real
  rows carry no generator noise; folding a 10-fold split of them into each
  fold's *training* set only (never validation) gives the model a small
  amount of unambiguous signal to train on directly, instead of only ever
  learning from a synthetic approximation of it. It scored slightly lower
  (0.94609) than not doing this at all — consistent with a separate check
  (recomputing the competition's own recovered generating formula directly
  on the synthetic columns, which reproduced the true buy rate almost
  exactly) showing the synthetic data already follows the same underlying
  relationship as the real data, just resampled at 65x the volume. A small
  slice of real rows does not correct anything that is not already present,
  in far larger and lower-variance form, in the synthetic data itself.

None of the three moved OOF outside of measurement noise. Two of them
(window encoding, the attitude interaction) scored marginally *higher* on
the public leaderboard (0.94626, 0.94628) despite a flat or slightly lower
OOF — more evidence of split noise than of real improvement, as the next
section explains.

### 11. Where the ceiling comes from

Grouping rows by their most predictive features exposes the real limit
directly: one group of nearly 5,000 people, identical on the four
strongest features, splits 88.1% buy / 11.9% do not. Nothing observable in
the data explains that split — it is noise built into how the labels were
generated, not a signal a model failed to find. Every model tried here —
different algorithms, different feature sets, blends of both, and the
three targeted ideas in the previous section — converges to essentially
the same ceiling for the same reason: they are all estimating the same
underlying rate, and the remaining error, at this point, is largely
irreducible.

## Results

| Step | Model | Features | OOF AUC | Public LB |
|---|---|---:|---:|---:|
| 1 | CatBoost, single 80/20 split, depth=4 | 15 | 0.94178 | 0.94158 |
| 2 | CatBoost, 5-fold CV, depth=4 | 15 | 0.94205 | 0.94192 |
| 3 | CatBoost, 5-fold CV + target encoding | 21 | 0.94237 | — |
| 4 | CatBoost, 5-fold CV + digit features + artifact flags + frequency/count encoding | 52 | 0.94366 | 0.94373 |
| 5 | CatBoost, 10-fold CV, tuned (depth=6, rsm=0.3) | 58 | 0.94609 | 0.94606 |
| 6 | **LightGBM, 10-fold CV, same 58 features** | 58 | **0.94617** | **0.94624** |
| 7 | Rank blend: 30% CatBoost (step 5) + 70% LightGBM (step 6) | 58 | 0.94620 | 0.94621 |
| 8 | LightGBM, "te" view: triple target encoding, no digits/flags | 20 | 0.94607–0.94610 | — |
| 9 | LightGBM, "raw" view: 14 original columns + 2 ratios | 15 | 0.94328 | — |
| 10 | LightGBM, "te" view + real-dataset category averages | 33 | 0.94608 | — |
| 11 | Rank blend: CatBoost + LightGBM (steps 5, 6, 8, 10) | — | 0.94623 | — |
| 12 | LightGBM, step 6 + income/commute neighbourhood-window encoding | 60 | 0.94617 | 0.94626 |
| 13 | **LightGBM, step 6 + joint target encoding of the three attitude columns** | 67 | 0.94616 | **0.94628** |
| 14 | LightGBM, step 6 + original dataset rows concatenated into training | 58 | 0.94609 | 0.94616 |

Step 13 is the best public result to date, and it makes the case for
treating these last few thousandths as noise rather than progress: its OOF
score (0.94616) is the *lowest* of steps 6, 12 and 13, yet its public score
is the *highest*. Combined with step 7's blend, which scored lower publicly
than a higher-OOF model, the pattern is consistent — OOF and the public
split disagree on the ranking of these near-identical models, which is
exactly what "within measurement noise" means in practice. Steps 8-14 were
run to test whether a different view of the features, an outside data
source, or a specific idea borrowed from a stronger public result could
break past the ceiling described in the closing section; none moved the
OOF score outside of measurement noise, confirming rather than breaking it.
