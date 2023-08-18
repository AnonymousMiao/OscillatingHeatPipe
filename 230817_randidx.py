from sklearn.metrics import rand_score
import numpy as np


def matrix_to_labels(matrix):
    # Convert matrix to a numpy array
    matrix_np = np.array(matrix)
    # Get the labels by finding the column index of the maximum value in each row
    labels = matrix_np.argmax(axis=1)
    return labels


# Given matrices
Manual = [
    [0.0037,	0.0614,	0.0064,	0.0131,	0.038],
    [0.0308,	0.0521,	0.0257,	0.107,	0.062],
    [0.0079,	0.0886,	0.0071,	0.0177,	0.1011],
    [0.0625,	0.233,	0.0213,	0.016,	0.2104],
    [0.0576,	0.0743,	0.0279,	0.0288,	0.0633]
]

Kmean = [
    [0.0044,	0.0066,	0.0323,	0.0074,	0.0153],
    [0.0863,	0.0029,	0.0128,	0.1609,	0.0605],
    [0.0437,	0.0509,	0.0883,	0.0638,	0.0762],
    [0.0236,	0.0436,	0.0522,	0.0143,	0.0313],
    [0.018,	0.002,	0.0677,	0.0158,	0.0124]
]

GMM = [
    [0.0114,	0.0302,	0.0205,	0.0067,	0.0523],
    [0.0915,	0.0439,	0.1283,	0.0306,	0.0362],
    [0.0915,	0.1712,	0.0201,	0.1214,	0.1628],
    [0.1848,	0.0516,	0.1566,	0.0038,	0.004],
    [0.056,	0.0395,	0.2001,	0.0063,	0.0052]
]

Hier = [
    [0.0896,	0.0289,	0.0226,	0.0038,	0.0287],
    [0.1043,	0.009,	0.1729,	0.0138,	0.0174],
    [0.1314,	0.0658,	0.0033,	0.1926,	0.1905],
    [0.1699,	0.036,	0.1671,	0.018,	0.0649],
    [0.0876,	0.0629,	0.1616,	0.0177,	0.0138]
]

# Convert matrices to clustering labels
manual_labels = matrix_to_labels(Manual)
kmean_labels = matrix_to_labels(Kmean)
gmm_labels = matrix_to_labels(GMM)
hier_labels = matrix_to_labels(Hier)


# Compute Rand index between the manual partition and the other partitions
rand_kmean = rand_score(manual_labels, kmean_labels)
rand_gmm = rand_score(manual_labels, gmm_labels)
rand_hier = rand_score(manual_labels, hier_labels)
rand_manu = rand_score(manual_labels, manual_labels)

print("Rand index (Manual vs. Kmean):", rand_kmean)
print("Rand index (Manual vs. GMM):", rand_gmm)
print("Rand index (Manual vs. Hier):", rand_hier)
print("Rand index (Manual vs. Manual):", rand_manu)
