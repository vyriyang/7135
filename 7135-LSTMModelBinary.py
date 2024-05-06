import datetime
import time
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.decomposition import PCA
from itertools import product
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

X = pd.read_csv('csv/7135_X.csv')
Y = pd.read_csv('csv/7135_Y.csv')
Y_batch = pd.read_csv('csv/7135_Y_batch.csv')
# 输出前几行以确认读取正确
print(Y_batch.head())

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=142)
Y_batch_train, Y_batch_test = train_test_split(Y_batch, test_size=0.2, random_state=189)
# 查看分割后的数据集大小
print("Training data size:")
print(X_train.shape, Y_train.shape)

print("Test data size:")
print(X_test.shape, Y_test.shape)
print("Y_batch data size:")
print(Y_batch_train.shape, Y_batch_test.shape)

X_train_numpy = X_train.to_numpy()
Y_train_numpy = Y_train.to_numpy()
Y_batch_train_numpy = Y_batch_train.to_numpy()

# Now you can create torch tensors from the numpy arrays
X_train_tensor = torch.from_numpy(X_train_numpy).float().unsqueeze(1)
Y_train_tensor = torch.from_numpy(Y_train_numpy).float()
Y_batch_train_tensor = torch.from_numpy(Y_batch_train_numpy).float()

X_test_numpy = X_test.to_numpy()
Y_test_numpy = Y_test.to_numpy()
Y_batch_test_numpy = Y_batch_test.to_numpy()

# Now you can create torch tensors from the numpy arrays
X_test_tensor = torch.from_numpy(X_test_numpy).float().unsqueeze(1)
Y_test_tensor = torch.from_numpy(Y_test_numpy).float()
Y_batch_test_tensor = torch.from_numpy(Y_batch_test_numpy).float()

print("Training data size:")
print(X_train_tensor.shape, Y_train_tensor.shape)

print("Test data size:")
print(X_test_tensor.shape, Y_test_tensor.shape)

print("Y_batch data size:")
print(Y_batch_train_tensor.shape, Y_batch_test_tensor.shape)

# 创建TensorDatasets
Y_batch_train_tensor = Y_batch_train_tensor.squeeze()
train_batch_dataset = TensorDataset(X_train_tensor, Y_batch_train_tensor)
test_batch_dataset = TensorDataset(X_test_tensor, Y_batch_test_tensor)

# 定义批量大小
batch_size = 64

# 创建DataLoaders
train_loader = DataLoader(dataset=train_batch_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_batch_dataset, batch_size=batch_size, shuffle=False)

# 打印数据集大小确认
print("Training data size:", len(train_loader.dataset), "Test data size:", len(test_loader.dataset))

for X_batch, Y_batch in train_loader:
    print("X_batch shape:", X_batch.shape)  # 应该是 [batch_size, features...]
    print("Y_batch shape:", Y_batch.shape)  # 应该是 [batch_size]
    print("First few Y values:", Y_batch[:5])  # 打印前几个标签
    break  # 只查看第一个批次
    
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  # output_size现在为2

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 只取最后一个时间步的输出
        return out

# 定义模型参数
input_size = 20
hidden_size = 100  # 最优隐藏层大小
num_layers = 2  # 最优层数
output_size = 2  # 输出大小为2，对应于二元分类
lr = 0.001  # 最优学习率
num_epochs = 200

# 实例化模型
model = LSTMModel(input_size, hidden_size, num_layers, output_size)
save_path = 'pth/Binary_model_parameters.pth'
criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 训练模型并记录损失
epoch_losses = []
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    epoch_loss = 0
    for X_batch, Y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        Y_batch = Y_batch.long()
        loss = criterion(outputs, Y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= len(train_loader)
    epoch_losses.append(epoch_loss)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}')

# Save the model's state
torch.save(model.state_dict(), save_path)
print("Model saved to", save_path)

# Plot the training loss
plt.figure(figsize=(10, 5))
plt.plot(epoch_losses, label='Training Loss')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

from torch.utils.data import DataLoader, TensorDataset

# Assuming test_batch_dataset is already created and similar to train_batch_dataset
test_loader = DataLoader(dataset=test_batch_dataset, batch_size=batch_size, shuffle=False)

# After training, switch to evaluation mode and evaluate on test data
model.eval()  # Set the model to evaluation mode
test_loss = 0
correct = 0
total = 0

with torch.no_grad():
    for X_batch, Y_batch in test_loader:
        outputs = model(X_batch)
        Y_batch = Y_batch.squeeze()
        Y_batch = Y_batch.squeeze().long()
        loss = criterion(outputs, Y_batch)
        test_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total += Y_batch.size(0)
        correct += (predicted == Y_batch).sum().item()

test_loss /= len(test_loader)
accuracy = 100 * correct / total
print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')

# Optionally, save the trained model
torch.save(model.state_dict(), save_path)
print("Model saved to", save_path)
