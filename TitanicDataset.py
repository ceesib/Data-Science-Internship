import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import numpy as np
from xgboost import XGBClassifier

# File path to Titanic dataset
file_path = r"C:\Users\ceejay sibeko\Downloads\Data Science Interships\Titanic-Dataset.csv"

# Read the CSV file
read_data = pd.read_csv(file_path, encoding='latin1')

# Fill in missing values
read_data['Age'].fillna(read_data['Age'].median(), inplace=True)
read_data['Fare'].fillna(read_data['Fare'].median(), inplace=True)
read_data['Cabin'].fillna('Unknown', inplace=True)

# Define target and features
y = read_data.Survived
X = read_data[['Age', 'Sex', 'Ticket', 'Pclass', 'Fare', 'Cabin']]

# Divide data into training and validation subsets
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, ['Age', 'Fare']),
        ('cat', categorical_transformer, ['Sex', 'Ticket', 'Pclass', 'Cabin'])
    ])

# Define the RandomForestClassifier model pipeline
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(n_estimators=100, random_state=0))
])

# Define the XGBoost model pipeline
xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', XGBClassifier(n_estimators=2000, learning_rate=0.05, n_jobs=4, use_label_encoder=False, eval_metric='logloss'))
])

# Perform cross-validation for RandomForestClassifier
rf_scores = -1 * cross_val_score(rf_pipeline, X, y, cv=7, scoring='neg_mean_squared_error')
rf_rmse_scores = np.sqrt(rf_scores)
print("RandomForestClassifier RMSE scores:\n", rf_rmse_scores)
print("Mean RMSE score (RandomForestClassifier):", np.mean(rf_rmse_scores))

# Perform cross-validation for XGBoost
xgb_scores = -1 * cross_val_score(xgb_pipeline, X, y, cv=7, scoring='neg_mean_squared_error')
xgb_rmse_scores = np.sqrt(xgb_scores)
print("XGBoost RMSE scores:\n", xgb_rmse_scores)
print("Mean RMSE score (XGBoost):", np.mean(xgb_rmse_scores))

# Select the model with the lowest RMSE score
if np.mean(rf_rmse_scores) < np.mean(xgb_rmse_scores):
    selected_model = rf_pipeline
    selected_model_name = "RandomForestClassifier"
else:
    selected_model = xgb_pipeline
    selected_model_name = "XGBoost"

print("Selected model:", selected_model_name)

# Fit the selected model on the entire training data
selected_model.fit(X_train, y_train)

# Make predictions on the validation set
predictions = selected_model.predict(X_val)

# Calculate the classification matrix (confusion matrix and classification report)
conf_matrix = confusion_matrix(y_val, predictions)
class_report = classification_report(y_val, predictions)

print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)