import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# pip install -U scikit-learn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# Define the neural network architecture

class Net_1(torch.nn.Module):
    def __init__(self):
        super(Net_1, self).__init__()
        self.fc1 = torch.nn.Linear(2, 5)
        self.fc2 = torch.nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Net_2(torch.nn.Module):
    def __init__(self):
        super(Net_2, self).__init__()
        self.fc1 = torch.nn.Linear(2, 10)
        self.fc2 = torch.nn.Linear(10, 20)
        self.fc3 = torch.nn.Linear(20, 10)
        self.fc4 = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Define the dataset class
class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)
def main():
    # Read the data from the Excel file
    data = pd.read_excel('data_combine.xlsx')

    dataset_1 = data                                                       # ALL
    dataset_2 = data[(data.iloc[:, 0] >= 15) & (data.iloc[:, 0] <= 85)]    # 15-85
    dataset_3 = data[(data.iloc[:, 0] < 15) |  (data.iloc[:, 0] > 85)]     # 0-15 + 85-100

    dataset=dataset_2.sample(frac=1).reset_index(drop=True)
    # we can get absolutely error 0.30 error relative 11.0% with dataset_1
    # we can get absolutely error 0.06 error relative 10.5% with dataset_2
    # we can get absolutely error 1.40 error relative 18.9% with dataset_3  

    x = dataset.iloc[:, :2]  # Input features
    y = dataset.iloc[:, 2]   # Output variable

    # Split the data into training and validation sets

    val_size = 11
    train_size = len(dataset)-val_size 
    train_x, val_x, train_y, val_y = train_test_split(dataset.iloc[:, :2], dataset.iloc[:, 2], test_size=val_size)

    # Create the dataset and data loader
    train_dataset = MyDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataset = MyDataset(val_x, val_y)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # Create the neural network and the optimizer
    net = Net_1()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.001)

    # Initialize the plot
    plt.ion()
    fig, ax = plt.subplots()

    train_loss_history = []
    val_loss_history = []
    epoch_history = []

    # Train the neural network
    for epoch in range(1000):
        net.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = net(batch_x)
            loss = torch.nn.functional.mse_loss(output.squeeze(), batch_y)
            
            # Add L2 regularization to the loss function
            l2_reg = 0
            for param in net.parameters():
                l2_reg += torch.norm(param)
            loss += 0.02 * l2_reg
            
            loss.backward()
            optimizer.step()

            train_loss_history.append(loss.item())
            epoch_history.append(epoch + len(train_loss_history) / len(train_dataset))

        net.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                output = net(batch_x)
                val_loss += torch.nn.functional.mse_loss(output.squeeze(), batch_y)
            val_loss /= len(val_loader)

            val_loss_history.append(val_loss.item())
            
        epochs_perPlot=200
        if epoch % epochs_perPlot == (epochs_perPlot-1) :
            # Update the plot with the latest loss values
            ax.clear()
            ax.plot(epoch_history, train_loss_history, label='Train Loss')
            ax.plot(range(0, len(val_loss_history)), val_loss_history, label='Val Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            plt.draw()

        epochs_perPrint=20
        if epoch % epochs_perPrint == (epochs_perPrint-1) :
            print(f"Epoch {epoch+1}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

    net.eval()
    with torch.no_grad():
        output = net(torch.tensor(x.values, dtype=torch.float32))
        error_1 = torch.abs(output.squeeze() - torch.tensor(y.values, dtype=torch.float32)).mean() / torch.tensor(y.values, dtype=torch.float32).mean()
        error_2 = torch.abs((output.squeeze() - torch.tensor(y.values, dtype=torch.float32))/torch.tensor(y.values, dtype=torch.float32)).mean()
        
        print(f'Relative Error 1: {error_1:.2%}')
        print(f'Relative Error 2: {error_2:.2%}')

    plt.show()
    plt.ioff()
    print('Done')

