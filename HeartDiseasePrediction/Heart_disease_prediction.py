# Heart Disease Prediction 🫀
# Beginner-friendly Machine Learning project to predict the likelihood of heart disease
# based on patient medical data such as age, blood pressure, cholesterol, and other features.
# Uses Support Vector Machine (SVM) for training and prediction.
# Author: Rithanya Ramesh Babu
# GitHub: https://github.com/RithanyaRameshBabu



import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

"""Data Collection and Analysis"""

# loading the dataset to a pandas DataFrame
heart_dataset = pd.read_csv('/content/heart_v2.csv')

pd.read_csv?

# printing the first 5 rows of the dataset
heart_dataset.head()

# number of rows and Columns in this dataset
heart_dataset.shape

# getting the statistical measures of the data
heart_dataset.describe()

heart_dataset['heart disease'].value_counts()

"""0 --> No Heart Disease

1 --> Heart Disease
"""

heart_dataset.groupby('heart disease').mean()

# separating the data and labels
X = heart_dataset.drop(columns = 'heart disease', axis=1)
Y = heart_dataset['heart disease']

print(X)

print(Y)

"""Data Standardization"""

scaler = StandardScaler()

scaler.fit(X)

standardized_data = scaler.transform(X)

print(standardized_data)

X = standardized_data
Y = heart_dataset['heart disease']

print(X)
print(Y)

"""Train Test Split"""

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

"""Training the Model"""

classifier = svm.SVC(kernel='linear')

#training the support vector Machine Classifier
classifier.fit(X_train, Y_train)

"""Model Evaluation

Accuracy Score
"""

# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy score of the training data : ', training_data_accuracy)

# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of the test data : ', test_data_accuracy)

"""Making a Predictive System"""

input_data = (63,0,150,407)

# changing the input_data to numpy array
input_Numpy = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_reshaped = input_Numpy.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('No heart disease')
else:
  print('The person has heart disease')

