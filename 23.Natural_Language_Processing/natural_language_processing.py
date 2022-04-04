# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 19:51:58 2022

@author: Jay
"""

import numpy as np
import pandas as pd

import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
# from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

def clean_data(data):
    # Remove non-letters
    data = re.sub("[^a-zA-Z]", " ", data)
    # Convert to lower case
    data = data.lower()
    # Split data into list and remove stopwords
    ps = PorterStemmer()
    data = data.split()
    # data = [
    #     ps.stem(word) for word in data if word == "not" or word not in set(stopwords.words("english"))
    # ]
    data = [
        ps.stem(word) for word in data if word not in set(stopwords.words("english"))
    ]
    
    return " ".join(data)

res_data = pd.read_csv(
    "./23.Natural_Language_Processing/Restaurant_Reviews.tsv",
    sep='\t',
    quoting=3
)
cleaned_data = res_data.copy()
cleaned_data["Review"] = cleaned_data["Review"].apply(clean_data)

# Creating the Bag of Words model
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(cleaned_data["Review"]).toarray()
y = np.array(cleaned_data["Liked"].copy())

# Predicting with Naive Bayes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
classifier = GaussianNB()
# classifier = RandomForestClassifier(n_estimators=100, random_state=0)
# Naive Bayes: [[55, 42], [12, 91]]
# Random Forest: [[85, 12], [48, 55]]
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
