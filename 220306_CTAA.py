import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from pyclustertend import hopkins
from pyclustertend import vat, ivat

from func_preprocess import preprocess_data

# Read in the Excel file
file_path = 'data_combine.xlsx'
X = preprocess_data(file_path)

# Cluster tendency assessment
hopkins_statistic = hopkins(X, X.shape[0] - 1)
print(f'Hopkins statistic: {hopkins_statistic:.3f}')

fig_vat = vat(X,return_odm= True)
fig_ivat = ivat(X,return_odm= True)
plt.close('all')

# Create a new figure with two subplots
fig_VAT_IVAT, axs = plt.subplots(1, 2, figsize=(12, 6))

# Display the VAT figure in the left subplot
axs[0].imshow(fig_vat, cmap='gray')
axs[0].set_title('VAT')

# Display the IVAT figure in the right subplot
axs[1].imshow(fig_ivat, cmap='gray')
axs[1].set_title('IVAT')

# Show the figure
plt.show()
fig_VAT_IVAT.savefig('fig_VAT_IVAT.png')

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


# Print the cluster labels for each algorithm
print('K-means labels:', kmeans_labels)
print('GMM labels:', gmm_labels)
print('DBSCAN labels:', dbscan_labels)
print('Hierarchical labels:', hierarchical_labels)

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
fig.savefig('clustering_results.png')
