import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

housing = pd.read_csv("housing.csv")
print(housing.info())
housing['total_bedrooms'].fillna(housing['total_bedrooms'].mean(), inplace=True)


def cap_outliers(df, column):
      Q1 = df[column].quantile(0.25)
      Q3 = df[column].quantile(0.75)
      IQR = Q3 - Q1
      lower_bound = Q1 - 1.5 * IQR
      upper_bound = Q3 + 1.5 * IQR
      df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
      df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
      return df

numeric_columns = housing.select_dtypes(include=[np.number]).columns
for column in numeric_columns:
    housing = cap_outliers(housing, column)
print(housing)


housing = pd.get_dummies(housing, drop_first=True)
housing
X = housing.drop('median_house_value', axis=1)
y = housing['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = LinearRegression()
model.fit(X_train_scaled, y_train)


y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f'Training MSE: {train_mse:.2f}')
print(f'Test MSE: {test_mse:.2f}')
print(f'Training R^2: {train_r2:.2f}')
print(f'Test R^2: {test_r2:.2f}')
