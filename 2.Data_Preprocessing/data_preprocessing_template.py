# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 13:56:32 2022

@author: Jay
"""

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Reading data file
data_path = "./Data_Preprocessing/"
data_file = "Data.csv"
data = pd.read_csv(data_path + data_file)

X = data[["Country", "Age", "Salary"]].copy()
y = data["Purchased"].copy()

# Imputing missing data
simple_imputer = SimpleImputer(strategy="mean", missing_values=np.nan)
X[["Age", "Salary"]] = simple_imputer.fit_transform(X[["Age", "Salary"]])

# Encoding categorical data
onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
data_countries = pd.DataFrame(
    onehot_encoder.fit_transform(X[["Country"]]),
    columns=["is_france", "is_germany", "is_spain"]
)
X = pd.concat([data_countries, X[["Age", "Salary"]]], axis=1)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Splitting training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardize data
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)