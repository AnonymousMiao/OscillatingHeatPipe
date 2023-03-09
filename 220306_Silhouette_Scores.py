import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from func_preprocess import preprocess_data

file_path = 'data_combine.xlsx'
X = preprocess_data(file_path)

# Calculate the silhouette score and inertia for different values of k
silhouette_scores = []
inertias = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))
    inertias.append(kmeans.inertia_)
    print(f'k={k}: silhouette score = {silhouette_scores[-1]:.3f}, inertia = {inertias[-1]:.3f}')

# Plot the results of both methods
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,6))

# Plot the Silhouette Method results
ax[0].plot(range(2, 11), silhouette_scores)
ax[0].set_xlabel('Number of clusters')
ax[0].set_ylabel('Silhouette score')
ax[0].set_title('Silhouette Method')

# Plot the Elbow Method results
ax[1].plot(range(2, 11), inertias)
ax[1].set_xlabel('Number of clusters')
ax[1].set_ylabel('Inertia')
ax[1].set_title('Elbow Method')

plt.show()
