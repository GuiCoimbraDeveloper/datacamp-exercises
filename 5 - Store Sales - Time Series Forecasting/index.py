import sys
print(f'Python Version: {sys.version}')

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib

ARTIFACT_PATH = "/kaggle/input/jwt/other/joblib/1/"

model = joblib.load(ARTIFACT_PATH + "lgbm_model.joblib")
FEATURES = joblib.load(ARTIFACT_PATH + "features.joblib")
last_lags = joblib.load(ARTIFACT_PATH + "last_lags.joblib")

# -------------------------
# LOAD TEST DATA
# -------------------------
DATA_PATH = "/kaggle/input/store-sales-time-series-forecasting/"

test = pd.read_csv(DATA_PATH + "test.csv")
oil = pd.read_csv(DATA_PATH + "oil.csv")
holidays = pd.read_csv(DATA_PATH + "holidays_events.csv")
stores = pd.read_csv(DATA_PATH + "stores.csv")

# -------------------------
# PREPROCESS
# -------------------------
for df in [test, oil, holidays]:
    df["date"] = pd.to_datetime(df["date"])

oil["dcoilwtico"] = oil["dcoilwtico"].ffill()

holidays = holidays[
    (holidays["transferred"] == False) &
    (holidays["type"].isin(["Holiday", "Event"]))
][["date"]]
holidays["is_holiday"] = 1

def merge_features(df):
    df = df.merge(oil, on="date", how="left")
    df = df.merge(holidays, on="date", how="left")
    df = df.merge(stores, on="store_nbr", how="left")
    df["is_holiday"] = df["is_holiday"].fillna(0)
    return df

test = merge_features(test)

# -------------------------
# FEATURES
# -------------------------
test["dayofweek"] = test["date"].dt.dayofweek
test["week"] = test["date"].dt.isocalendar().week.astype(int)
test["month"] = test["date"].dt.month

test = test.merge(
    last_lags,
    on=["store_nbr", "family"],
    how="left"
)

# -------------------------
# PREDICTION
# -------------------------
X_test = test[FEATURES]

test["sales"] = np.expm1(model.predict(X_test))
test["sales"] = test["sales"].clip(0)

# -------------------------
# SUBMISSION
# -------------------------
submission = test[["id", "sales"]]
submission.to_csv("submission.csv", index=False)

submission.head()