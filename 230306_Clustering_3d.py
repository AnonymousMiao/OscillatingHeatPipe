import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN

from func_preprocess import preprocess_data

# Read in the Excel file
file_path = 'data_combine.xlsx'
X = preprocess_data(file_path)

# Define the number of clusters
k = 3

# K-means clustering
kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
kmeans_labels = kmeans.labels_

# Gaussian mixture model clustering
gmm = GaussianMixture(n_components=k, random_state=0).fit(X)
gmm_labels = gmm.predict(X)

# Density-based clustering
dbscan = DBSCAN(eps=0.3, min_samples=5).fit(X)
dbscan_labels = dbscan.labels_

# Hierarchical clustering
hierarchical = AgglomerativeClustering(n_clusters=k).fit(X)
hierarchical_labels = hierarchical.labels_

# Plot the data points with colors corresponding to their assigned labels
fig = plt.figure(figsize=(12, 12))

ax = fig.add_subplot(2, 2, 1, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=kmeans_labels)
ax.set_title('K-means clustering')

ax = fig.add_subplot(2, 2, 2, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=gmm_labels)
ax.set_title('Gaussian mixture model clustering')

ax = fig.add_subplot(2, 2, 3, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=dbscan_labels)
ax.set_title('Density-based clustering')

ax = fig.add_subplot(2, 2, 4, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=hierarchical_labels)
ax.set_title('Hierarchical clustering')

plt.show()
