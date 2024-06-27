import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
file_path = r"C:\Users\ceejay sibeko\Downloads\Data Science Interships\advertising.csv"

# Read the CSV file with error handling
read_data = pd.read_csv(file_path, encoding='latin1')

# Display the data description and columns
print(read_data.describe())
print(read_data.columns)
# Checking Null values
print(read_data.isnull().sum()*100/read_data.shape[0])
# Drop rows with missing values
#read_data = read_data.dropna(axis=0)
print(read_data.head())
print(read_data.shape)

# how is Sales are related with other variables using scatter plot.
sns.pairplot(read_data, x_vars=['TV', 'Newspaper', 'Radio'], y_vars='Sales', height=4, aspect=1, kind='reg')

plt.show()

# Correlation matrix .
sns.heatmap(read_data.corr(), cmap="YlGnBu", annot = True)
plt.show()
# Sales is more correlated to TV

# TV relationship
X = read_data[['TV', 'Radio', 'Newspaper']]
y = read_data['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 100)

# intercept
X_train_sm = sm.add_constant(X_train)
# Fit the resgression line using 'OLS'
mlr = sm.OLS(y_train, X_train_sm).fit()
print(mlr.summary())

#1. The coefficient for TV is 0.054, with a very low p value
#2. The coefficient for Radio is 0.11, with a very low p value
#   These coefficient is statistically significant.
#3. The coefficient Newspaper for is 0.007, with a very high p value, not significant
# 2. R - squared is 0.91
# Meaning that 91.6% of the variance in Sales is explained by all the variables
# This is a decent R-squared value.
# 3. F statistic has a very low p value (practically low)
# Meaning that the model fit is statistically significant
X = read_data[['TV', 'Radio']]
y = read_data['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 100)
# intercept
X_train_sm = sm.add_constant(X_train)
# Fit the resgression line using 'OLS'
mlr = sm.OLS(y_train, X_train_sm).fit()
print(mlr.summary())
# Plot Actual vs. Predicted values for training data
y_train_pred = mlr.predict(X_train_sm)
fig = plt.figure(figsize=(12, 8))
plt.scatter(y_train, y_train_pred)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)  # Adding a red dashed trend line
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales (Training Data)')
plt.show()


#TEST DATA
# intercept
X_test_sm = sm.add_constant(X_test)
# Predict the y values corresponding to X_test_sm
y_pred = mlr.predict(X_test_sm)
#RMSE
rmse= np.sqrt(mean_squared_error(y_test, y_pred))
# Predict on test data
y_test_pred = mlr.predict(X_test_sm)

print(f"RMSE= {rmse}")
r_squared = r2_score(y_test, y_pred)
print(f"R_squared= {r_squared}")

# Plot Actual vs. Predicted values for test data
fig = plt.figure(figsize=(12, 8))
plt.scatter(y_test, y_test_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  # Adding a red dashed trend line
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales (Test Data)')
plt.show()

# Residuals plot for test data
residuals_test = y_test - y_test_pred
# Plot distribution of residuals
fig = plt.figure(figsize=(12, 8))
sns.histplot(residuals_test, kde=True, color='blue', bins=20)
plt.axvline(x=np.mean(residuals_test), color='red', linestyle='--', linewidth=2)  # Adding a red dashed line for mean
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.show()
##The residuals are following the normally distributed with a mean 0
