# -*- coding: utf-8 -*-

# Artificial Neural Network

# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf
tf.__version__

# Part 1 - Data Preprocessing

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values
print(X)
print(y)

# Encoding categorical data
# Label Encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder
labelencoder_x_1 = LabelEncoder()
X[:, 1] = labelencoder_x_1.fit_transform(X[:, 1])
labelencoder_x_2 = LabelEncoder()
X[:, 2] = labelencoder_x_2.fit_transform(X[:, 2])

# One Hot Encoding the "Geography" column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense


# Training the ANN on the Training set
classifier.fit(X_train, y_train, batch_size = 32, epochs = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

"""
Homework:
Use our ANN model to predict if the customer with the following informations will leave the bank: 
Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $ 60000
Number of Products: 2
Does this customer have a credit card? Yes
Is this customer an Active Member: Yes
Estimated Salary: $ 50000
So, should we say goodbye to that customer?

Solution:
"""

new_prediction=classifier.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]))
new_prediction=(new_prediction>0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)

# Evaluating, Improving and Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units=6,activation='relu'))
    classifier.add(Dense(units=6,activation='relu'))
    classifier.add(Dense(units=1,activation='sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
    
classifier = KerasClassifier(build_fn=build_classifier,batch_size=10,epochs=100)
accuracies = cross_val_score( estimator = classifier, X = X_train, y = y_train , cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()

# Tuning The ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units=6,activation='relu'))
    classifier.add(Dense(units=6,activation='relu'))
    classifier.add(Dense(units=1,activation='sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
    
classifier = KerasClassifier(build_fn=build_classifier)
parameters = {'batch_size':[25,32],
              'epochs':[100,500],
              'optimizer':['adam','rmsprop']}

grid_search = GridSearchCV(estimator=classifier,param_grid=parameters,scoring='accuracy',cv=10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_




























