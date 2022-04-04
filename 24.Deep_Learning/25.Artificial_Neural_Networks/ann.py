# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 10:59:56 2022

@author: Jay
"""

import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Importing the Keras Libraries and packages
from keras.models import Sequential
from keras.layers import Dense

churn_data = pd.read_csv("./24.Deep_Learning/25.Artificial_Neural_Networks/Churn_Modelling.csv")
X = churn_data.drop(
    ["RowNumber", "CustomerId", "Surname", "Exited"],
    axis=1
).select_dtypes(exclude="object").copy()
y = churn_data["Exited"].copy()

geo_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
geo_data = pd.DataFrame(
    geo_encoder.fit_transform(churn_data[["Geography"]]),
    columns=["France", "Germany", "Spain"]
)

gender_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
gender_data = pd.DataFrame(
    gender_encoder.fit_transform(churn_data[["Gender"]]),
    columns=["Female", "Male"]
)

X = pd.concat([X, geo_data, gender_data], axis=1)
X = X.drop(["Spain"], axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

churn_scaler = StandardScaler()
X_train = churn_scaler.fit_transform(X_train)
X_test = churn_scaler.transform(X_test)

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(
    Dense(units=6, activation="relu", kernel_initializer="uniform", input_dim=12)
)

# Adding the second hidden layer
classifier.add(
    Dense(units=6, activation="relu", kernel_initializer="uniform")
)

# Adding the output layer
# 如果分类结果有多个，units要对应结果的个数，激活函数要设置为softmax
classifier.add(
    Dense(units=1, activation="sigmoid", kernel_initializer="uniform")
)

# Compiling the ANN
# 分类结果若为3个及以上，损失函数为categorical_crossentropy
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

cm = confusion_matrix(y_test, y_pred)
