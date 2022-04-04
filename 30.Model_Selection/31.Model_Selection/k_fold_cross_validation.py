# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 14:44:11 2022

@author: Jay
"""

import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

ads_data = pd.read_csv(
    "./30.Model_Selection/31.Model_Selection/Social_Network_Ads.csv"    
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
