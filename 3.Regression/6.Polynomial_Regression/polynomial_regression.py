# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 14:33:53 2022

@author: Jay
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


data = pd.read_csv("./3.Regression/6.Polynomial_Regression/Position_Salaries.csv")

X = data[["Level"]].copy()
y = data["Salary"].copy()

lin_reg = LinearRegression()
lin_reg.fit(X, y)


poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)


plt.scatter(X, y, color="red")
plt.plot(X, lin_reg.predict(X))
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()


X_grid = np.arange(min(X.values), max(X.values), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color="blue")
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)))
plt.title("Truth or Bluff (Polynomial Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

lin_reg.predict(np.array(6.5).reshape((-1, 1)))
lin_reg_2.predict(poly_reg.fit_transform(np.array(6.5).reshape((-1, 1))))
