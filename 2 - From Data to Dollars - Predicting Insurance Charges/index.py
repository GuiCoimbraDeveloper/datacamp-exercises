import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

# Load the dataset
insurance_data_path = 'insurance.csv'
insurance = pd.read_csv(insurance_data_path)

def clean_dataset(insurance):
    """
    Cleans the insurance dataset by performing several preprocessing tasks:
    - Corrects the 'sex' column values to a standard format ('male', 'female').
    - Removes the dollar sign from the 'charges' column and converts it to float.
    - Drops negative 'age' values.
    - Converts negative 'children' values to zero.
    - Converts 'region' values to lowercase.
    - Drops rows with any missing values.
    
    Parameters:
    - insurance: pandas DataFrame, the insurance dataset.
    
    Returns:
    - DataFrame after cleaning.
    """
    insurance['sex'] = insurance['sex'].replace({'M': 'male', 'man': 'male', 'F': 'female', 'woman': 'female'})
    insurance['charges'] = insurance['charges'].replace({'\$': ''}, regex=True).astype(float)
    insurance = insurance[insurance["age"] > 0]
    insurance.loc[insurance["children"] < 0, "children"] = 0
    insurance["region"] = insurance["region"].str.lower()

    return insurance.dropna()

def create_and_evaluate_regression_model(insurance):
    """
    Prepares the data, fits a linear regression model, and evaluates it using cross-validation.
    
    Parameters:
    - insurance: pandas DataFrame, the cleaned insurance dataset.
    
    Returns:
    - A tuple containing the fitted sklearn Pipeline object, mean MSE, and mean R2 scores.
    """
    # Preprocessing
    X = insurance.drop('charges', axis=1)
    y = insurance['charges']
    categorical_features = ['sex', 'smoker', 'region']
    numerical_features = ['age', 'bmi', 'children']
    
    # Convert categorical variables to dummy variables
    X_categorical = pd.get_dummies(X[categorical_features], drop_first=True)
    
    # Combine numerical features with dummy variables
    X_processed = pd.concat([X[numerical_features], X_categorical], axis=1)
    # Scaling numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_processed)
    # Linear regression model
    lin_reg = LinearRegression()
    
    # Pipeline
    steps = [("scaler", scaler), ("lin_reg", lin_reg)]
    insurance_model_pipeline = Pipeline(steps)
    
    # Fitting the model
    insurance_model_pipeline.fit(X_scaled, y)
    
    # Evaluating the model
    mse_scores = -cross_val_score(insurance_model_pipeline, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
    r2_scores = cross_val_score(insurance_model_pipeline, X_scaled, y, cv=5, scoring='r2')
    mean_mse = np.mean(mse_scores)
    mean_r2 = np.mean(r2_scores)
    
    return insurance_model_pipeline, mean_mse, mean_r2

# Usage example
cleaned_insurance = clean_dataset(insurance)
insurance_model, mean_mse, r2_score = create_and_evaluate_regression_model(cleaned_insurance)
print("Mean MSE:", mean_mse)
print("Mean R2:", r2_score)

# Predict on validation data
validation_data_path = 'validation_dataset.csv'
validation_data = pd.read_csv(validation_data_path)

# Ensure categorical variables are properly transformed
validation_data_processed = pd.get_dummies(validation_data, columns=['sex', 'smoker', 'region'], drop_first=True)

# Make predictions using the trained model
validation_predictions = insurance_model.predict(validation_data_processed)

# Add predicted charges to the validation data
validation_data['predicted_charges'] = validation_predictions

# Adjust predictions to ensure minimum charge is $1000
validation_data.loc[validation_data['predicted_charges'] < 1000, 'predicted_charges'] = 1000

# Display the updated dataframe
validation_data.head()