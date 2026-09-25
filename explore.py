import pandas as pd
from prepare import prepare

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

print(train.shape, test.shape)
print(train.dtypes)
print(train.isna().sum())
print(train["Will_Buy_EV"].value_counts(normalize=True))
print(train.describe())

cat_cols = train.select_dtypes(include=["str", "object"]).columns.tolist()
print(cat_cols)

for col in cat_cols:
    print(train[col].value_counts())
    print()

print("--- train prepared:")
print(prepare(train).head())
print("--- test prepared:")
print(prepare(test).head())