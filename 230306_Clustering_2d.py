# K-means clustering using PyTorch

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read in the Excel file
data = pd.read_excel('data_combine.xlsx')

dataset_1 = data                                                       # ALL
dataset_2 = data[(data.iloc[:, 0] >= 15) & (data.iloc[:, 0] <= 85)]    # 15-85
dataset_3 = data[(data.iloc[:, 0] < 15) |  (data.iloc[:, 0] > 85)]     # 0-15 + 85-100

dataset=dataset_1.sample(frac=1).reset_index(drop=True)

# Convert the dataset to a PyTorch tensor
X = torch.tensor(dataset.values, dtype=torch.float)

# Define the number of clusters
k = 3

# Initialize the centroids randomly
centroids = X[np.random.choice(X.shape[0], k, replace=False)]

# Loop until convergence
for i in range(100):
    # Calculate the distances between each point and each centroid
    distances = torch.cdist(X, centroids)
    
    # Find the closest centroid for each point
    labels = torch.argmin(distances, dim=1)
    
    # Update the centroids
    for j in range(k):
        centroids[j] = X[labels == j].mean(dim=0)

# Plot the data points with colors corresponding to their assigned labels
colors = ['r', 'g', 'b']
for i in range(k):
    plt.scatter(X[labels == i, 0], X[labels == i, 1], c=colors[i], label=f'Cluster {i+1}')
plt.scatter(centroids[:, 0], centroids[:, 1], c='k', marker='x', s=100, linewidths=3, label='Centroids')
plt.legend()
plt.show()
