import numpy as np
import pandas as pd

def preprocess_data(file_path):
    # Read in the Excel file
    data = pd.read_excel(file_path)

    dataset_1 = data                                                       # ALL
    dataset_2 = data[(data.iloc[:, 0] >= 15) & (data.iloc[:, 0] <= 85)]    # 15-85
    dataset_3 = data[(data.iloc[:, 0] < 15) |  (data.iloc[:, 0] > 85)]     # 0-15 + 85-100

    dataset = dataset_1.sample(frac=1).reset_index(drop=True)

    # Convert the dataset to a numpy array
    X = dataset.values
    X[:,0]/=100
    X[:,1]/=50
    X[:,2]/=3.5

    return X
