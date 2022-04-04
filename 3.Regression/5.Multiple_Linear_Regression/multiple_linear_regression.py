# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 21:09:57 2022

@author: WBY00EQ
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


data = pd.read_csv("./3.Regression/5.Multiple_Linear_Regression/50_Startups.csv")

X = data[["R&D Spend", "Administration", "Marketing Spend", "State"]].copy()
y = data["Profit"].copy()


state_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
data_state = pd.DataFrame(
    state_encoder.fit_transform(X[["State"]]),
    columns=["California", "Florida", "New York"]
)

X = pd.concat([data_state, X[["R&D Spend", "Administration", "Marketing Spend"]]], axis=1)
X = X.drop(X[["California"]], axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_prediction = regressor.predict(X_test)

X_train.insert(0, "Constant", value=1)


import statsmodels.api as sm

X_opt = X_train.iloc[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y_train, exog=X_opt).fit()
print(regressor_OLS.summary())

X_opt = X_train.iloc[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y_train, exog=X_opt).fit()
print(regressor_OLS.summary())

X_opt = X_train.iloc[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y_train, exog=X_opt).fit()
print(regressor_OLS.summary())

X_opt = X_train.iloc[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog=y_train, exog=X_opt).fit()
print(regressor_OLS.summary())

X_opt = X_train.iloc[:, [0, 3]]
regressor_OLS = sm.OLS(endog=y_train, exog=X_opt).fit()
print(regressor_OLS.summary())