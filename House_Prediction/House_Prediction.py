

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from xgboost import XGBRegressor
from sklearn import metrics

"""Importing the California House Price Dataset"""

# load dataset
house_price_dataset = fetch_california_housing(as_frame=True)

print(house_price_dataset)

# Loading the dataset to a Pandas DataFrame
house_price_dataframe = pd.DataFrame(house_price_dataset.data, columns = house_price_dataset.feature_names)

# Print First 5 rows of our DataFrame
house_price_dataframe.head()

# add the target (price) column to the DataFrame
house_price_dataframe['price'] = house_price_dataset.target

# Distribution of target (house prices)
sns.histplot(house_price_dataframe['price'], bins=50, kde=True)
plt.title("House Price Distribution")
plt.show()

# Relationship between Median Income and Price
plt.scatter(house_price_dataframe['MedInc'], house_price_dataframe['price'], alpha=0.3)
plt.xlabel("Median Income")
plt.ylabel("House Price")
plt.title("Median Income vs House Price")
plt.show()

house_price_dataframe.head()

# checking the number of rows and Columns in the data frame
house_price_dataframe.shape

# check for missing values
house_price_dataframe.isnull().sum()

# statistical measures of the dataset
house_price_dataframe.describe()

correlation = house_price_dataframe.corr()

# constructing a heatmap to nderstand the correlation
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')

"""Splitting the data and Target"""

X = house_price_dataframe.drop(['price'], axis=1)
Y = house_price_dataframe['price']

print(X)
print(Y)

"""Splitting the data into Training data and Test data"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2)

print(X.shape, X_train.shape, X_test.shape)

"""Model Training

XGBoost Regressor
"""

# loading the model
model = XGBRegressor()

# training the model with X_train
model.fit(X_train, Y_train)

"""Evaluation

Prediction on training data
"""

# accuracy for prediction on training data
training_data_prediction = model.predict(X_train)

print(training_data_prediction)

# R squared error
score_1 = metrics.r2_score(Y_train, training_data_prediction)

# Mean Absolute Error
score_2 = metrics.mean_absolute_error(Y_train, training_data_prediction)

print("R squared error : ", score_1)
print('Mean Absolute Error : ', score_2)

"""Visualizing the actual Prices and predicted prices"""

plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Price vs Preicted Price")
plt.show()

"""Prediction on Test Data"""

# accuracy for prediction on test data
test_data_prediction = model.predict(X_test)

# R squared error
score_1 = metrics.r2_score(Y_test, test_data_prediction)

# Mean Absolute Error
score_2 = metrics.mean_absolute_error(Y_test, test_data_prediction)

print("R squared error : ", score_1)
print('Mean Absolute Error : ', score_2)

