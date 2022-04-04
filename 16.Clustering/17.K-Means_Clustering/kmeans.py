# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 21:46:27 2022

@author: Jay
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

mall_data = pd.read_csv("./16.Clustering/17.K-Means_Clustering/Mall_Customers.csv")

X = mall_data[["Annual Income (k$)", "Spending Score (1-100)"]].copy()

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, max_iter=300, n_init=10, init="k-means++", random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()


kmeans = KMeans(n_clusters=5, max_iter=300, n_init=10, init="k-means++", random_state=0)
y_kmeans = kmeans.fit_predict(X)

# Visualizing the clusters
plt.scatter(X.iloc[y_kmeans == 0, 0], X.iloc[y_kmeans == 0, 1], s=30, c="red", label="Careful")
plt.scatter(X.iloc[y_kmeans == 1, 0], X.iloc[y_kmeans == 1, 1], s=30, c="blue", label="Standard")
plt.scatter(X.iloc[y_kmeans == 2, 0], X.iloc[y_kmeans == 2, 1], s=30, c="green", label="Target")
plt.scatter(X.iloc[y_kmeans == 3, 0], X.iloc[y_kmeans == 3, 1], s=30, c="grey", label="Careless")
plt.scatter(X.iloc[y_kmeans == 4, 0], X.iloc[y_kmeans == 4, 1], s=30, c="magenta", label="Sensible")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=50, c="yellow", label="Centroids")
plt.title("Clusters of Clients")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()
