import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Read training and testing data from CSV files
df_train = pd.read_csv("df_train.csv")
df_test = pd.read_csv("df_test.csv")

# Display structure of the training data
print(df_train.info())

# Convert 'date' column to datetime and 'day_in_week' column to category in both datasets
df_train['date'] = pd.to_datetime(df_train['date'], format="%m/%d/%Y")
df_test['date'] = pd.to_datetime(df_test['date'], format="%m/%d/%Y")
df_train['day_in_week'] = df_train['day_in_week'].astype('category')
df_test['day_in_week'] = df_test['day_in_week'].astype('category')

# One-hot encode 'day_in_week' in both datasets
df_onehot_train = pd.get_dummies(df_train['day_in_week'], prefix='day_in_week')
df_onehot_test = pd.get_dummies(df_test['day_in_week'], prefix='day_in_week')

# Align columns in case some categories are missing in test or train
df_onehot_train, df_onehot_test = df_onehot_train.align(df_onehot_test, join='outer', axis=1, fill_value=0)

# Combine one-hot encoded columns with the original datasets and remove the 'day_in_week' column
df_train = pd.concat([df_train.drop('day_in_week', axis=1), df_onehot_train], axis=1)
df_test = pd.concat([df_test.drop('day_in_week', axis=1), df_onehot_test], axis=1)

# Separate features and target variable for both training and testing datasets
train_x = df_train.drop(['power_consumption', 'date'], axis=1)
train_y = df_train['power_consumption']
test_x = df_test.drop(['power_consumption', 'date'], axis=1)
test_y = df_test['power_consumption']

# Train models, predict on test dataset and calculate RMSE for each model.

## Linear regression
lm_model = LinearRegression()
lm_model.fit(train_x, train_y)
lm_pred = lm_model.predict(test_x)
lm_rmse = np.sqrt(mean_squared_error(test_y, lm_pred))

## Random forest
rf_model = RandomForestRegressor(
    n_estimators=1000,
    max_features=3,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(train_x, train_y)
rf_pred = rf_model.predict(test_x)
rf_rmse = np.sqrt(mean_squared_error(test_y, rf_pred))

## XGBoost
xgb_model = XGBRegressor(
    n_estimators=800,
    learning_rate=0.2,  
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42,
    verbosity=0
)

xgb_model.fit(train_x, train_y)
xgb_pred = xgb_model.predict(test_x)
xgb_rmse = np.sqrt(mean_squared_error(test_y, xgb_pred))

# RMSE scores
rmse_df = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest", "XGBoost"],
    "RMSE": [lm_rmse, rf_rmse, xgb_rmse]
})
print(rmse_df)

# Get the lowest RMSE and assign it to selected_rmse
selected_rmse = min(lm_rmse, rf_rmse, xgb_rmse)
print(f"selected_rmse: {selected_rmse:.3f} kW")

# Add predictions to the test dataset for plotting (using Random Forest as in original code)
df_test['Predicted'] = rf_pred

# Plot actual vs predicted power consumption over time to check for trend similarity
plt.figure(figsize=(12, 6))
plt.plot(df_test['date'], df_test['power_consumption'], color='green', linewidth=1.1, label='Original')
plt.plot(df_test['date'], df_test['Predicted'], color='brown', linewidth=1, label='Predicted')
plt.title("Power Consumption: Original and Predicted")
plt.xlabel("Date")
plt.ylabel("Power Consumption")
plt.legend()
plt.grid(True, axis='x', linestyle='--', color='grey', alpha=0.5)
plt.tight_layout()
plt.show()

trend_similarity = "Yes"
print("trend_similarity:", trend_similarity)