# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 15:03:49 2022

@author: Jay
"""

import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

ads_data = pd.read_csv(
    "./30.Model_Selection/32.Grid_Search/Social_Network_Ads.csv"    
)
X = ads_data[["Age", "EstimatedSalary"]].copy()
y = ads_data["Purchased"].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

ads_scaler = StandardScaler()
X_train = ads_scaler.fit_transform(X_train)
X_test = ads_scaler.transform(X_test)

classifier = SVC(kernel="rbf", random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV

parameters = [
    {
     "C": [1, 10, 100, 1000],
     "kernel": ["linear"]
    },
    {
     "C": [1, 10, 100, 1000],
     "kernel": ["rbf"],
     "gamma": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    },
]
grid_search = GridSearchCV(
    estimator=classifier,
    param_grid=parameters,
    scoring="accuracy",
    cv=10,
    n_jobs=-1
)

grid_search = grid_search.fit(X_train, y_train)

best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

