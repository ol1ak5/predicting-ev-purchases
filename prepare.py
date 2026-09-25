import pandas as pd

CAT_COLS = ["Gender", "City_Type", "Current_Car_Type"]

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