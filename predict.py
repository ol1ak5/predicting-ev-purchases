import pandas as pd
from catboost import CatBoostClassifier

from prepare import prepare

test = pd.read_csv("data/test.csv")
test_ids = test["id"]

X_test = prepare(test)

model = CatBoostClassifier()
model.load_model("model.cbm")

pred = model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    "id": test_ids,
    "Will_Buy_EV": pred,
})
submission.to_csv("submissions/catboost/catboost_single_val94178.csv", index=False)

print(submission.shape)
print(submission.head())