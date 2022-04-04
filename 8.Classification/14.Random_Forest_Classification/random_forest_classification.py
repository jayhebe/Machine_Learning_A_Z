# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 12:56:51 2022

@author: Jay
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

ads_data = pd.read_csv("./8.Classification/14.Random_Forest_Classification/Social_Network_Ads.csv")
X = ads_data[["Age", "EstimatedSalary"]].copy()
y = ads_data["Purchased"].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

ads_scaler = StandardScaler()
X_train = ads_scaler.fit_transform(X_train)
X_test = ads_scaler.transform(X_test)

classifier = RandomForestClassifier(n_estimators=100, random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# ---------------------------------------
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

# ---------------------------------------
from matplotlib.colors import ListedColormap

X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, 
                                stop=X_set[:, 0].max() + 1, step=0.01),
                      np.arange(start=X_set[:, 1].min() - 1, 
                                stop=X_set[:, 1].max() + 1, step=0.01)
                      )
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
              alpha=0.75, cmap=ListedColormap(("red", "green")))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(("orange", "blue"))(i), label=j, s=15)
plt.title("Random Forest (Training set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()

X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, 
                                stop=X_set[:, 0].max() + 1, step=0.01),
                      np.arange(start=X_set[:, 1].min() - 1, 
                                stop=X_set[:, 1].max() + 1, step=0.01)
                      )
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
              alpha=0.75, cmap=ListedColormap(("red", "green")))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(("orange", "blue"))(i), label=j, s=15)
plt.title("Random Forest (Test set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()
