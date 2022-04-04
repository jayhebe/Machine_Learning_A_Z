# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 13:56:32 2022

@author: WBY00EQ
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("./3.Regression/4.Simple_Linear_Regression/Salary_Data.csv")
X = data[["YearsExperience"]].copy()
y = data["Salary"].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Fitting Simple Linear Regression to the Training Set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the test set
y_prediction = regressor.predict(X_test)
print(y_prediction)

# Visualising the training set
plt.scatter(X_train, y_train, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title("Salary VS Experience (training set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

plt.scatter(X_test, y_test, color="green")
plt.plot(X_test, y_prediction, color="blue")
plt.title("Salary VS Experience (test set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()
