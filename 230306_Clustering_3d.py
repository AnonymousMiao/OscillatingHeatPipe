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

n_kmeans_labels = np.ones(len(X[:, 0]))
for i in range(len(kmeans_labels)):
    if kmeans_labels[i] == 2:
        n_kmeans_labels[i] = 1
    if kmeans_labels[i] == 1:
        n_kmeans_labels[i] = 2
    if kmeans_labels[i] == 0:
        n_kmeans_labels[i] = 0

# Gaussian mixture model clustering
gmm = GaussianMixture(n_components=k, random_state=0).fit(X)
gmm_labels = gmm.predict(X)

n_gmm_labels = np.ones(len(X[:, 0]))
for i in range(len(gmm_labels)):
    if gmm_labels[i] == 2:
        n_gmm_labels[i] = 0
    if gmm_labels[i] == 1:
        n_gmm_labels[i] = 2
    if gmm_labels[i] == 0:
        n_gmm_labels[i] = 1

# Density-based clustering
dbscan = DBSCAN(eps=0.3, min_samples=5).fit(X)
dbscan_labels = dbscan.labels_

# Hierarchical clustering
hierarchical = AgglomerativeClustering(n_clusters=k).fit(X)
hierarchical_labels = hierarchical.labels_

n_hierarchical_labels = np.ones(len(X[:, 0]))
for i in range(len(hierarchical_labels)):
    if hierarchical_labels[i] == 0:
        n_hierarchical_labels[i] = 0
    if hierarchical_labels[i] == 1:
        n_hierarchical_labels[i] = 2
    if hierarchical_labels[i] == 2:
        n_hierarchical_labels[i] = 1

LL_manu = np.ones(len(X[:, 0]))

for i in range(len(X[:, 0])):
    if X[i, 0] < 0.15:
        LL_manu[i] = 0
    elif X[i, 0] > 0.85:
        LL_manu[i] = 2

# Convert X to a DataFrame
# Replace 'Column1', 'Column2', 'Column3' with actual column names
X_df = pd.DataFrame(X, columns=['Column1', 'Column2', 'Column3'])

# Add the additional variables as new columns to X_df
X_df['LL_manu'] = LL_manu
X_df['KMeans'] = n_kmeans_labels
X_df['GMM'] = n_gmm_labels
X_df['DBSCAN'] = dbscan_labels
X_df['Hierarchical'] = n_hierarchical_labels

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


# Save X_df to an Excel file
output_file_path = 'output_data.xlsx'
# Save X_df to the Excel file without including the index
X_df.to_excel(output_file_path, index=False)
