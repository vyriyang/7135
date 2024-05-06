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

# Create TensorDatasets
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

# Define the batch size
batch_size = 64

# Create DataLoaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Now, train_loader and test_loader will yield batches of (X, Y) pairs
print("Training data size:", len(train_loader.dataset), "Test data size:", len(test_loader.dataset))

# 假设每个时间点的输入特征维度是20，因为您的X_train的第二维大小为20
input_size = 20
# 定义隐藏层的特征维度
hidden_size = 50
# 定义LSTM层的堆叠数（层数）
num_layers = 2
# 输出维度，对于回归任务通常是1
output_size = 1
# 序列长度，对于逐点预测通常是1
seq_length = 1

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # 将LSTM的输出维度转换成我们需要的输出维度
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # 前向传播LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # 取最后一个时间点的输出并通过全连接层
        out = self.fc(out[:, -1, :])
        return out
        
# 实例化模型
model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10  # 或任何你选择的epoch数量
for epoch in range(num_epochs):
    for X_batch, Y_batch in train_loader:
        outputs = model(X_batch)
        loss = criterion(outputs, Y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
    
learning_rates = [0.01, 0.001, 0.0001]
hidden_sizes = [50, 100, 150]
num_layers_options = [1, 2, 3]

best_loss = float('inf')
best_params = {}

def validate(model, criterion, data_loader):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    count = 0

    with torch.no_grad():  # No need to track gradients during validation
        for X_batch, Y_batch in data_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            total_loss += loss.item()
            count += 1

    average_loss = total_loss / count
    return average_loss


for lr in learning_rates:
    for hidden_size in hidden_sizes:
        for num_layers in num_layers_options:
            model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            for epoch in range(num_epochs):
                for X_batch, Y_batch in train_loader:
                    outputs = model(X_batch)
                    loss = criterion(outputs, Y_batch)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
            avg_loss = validate(model, criterion, test_loader)

            print(f'LR: {lr}, Hidden Size: {hidden_size}, Layers: {num_layers}, Loss: {avg_loss}')

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_params = {'lr': lr, 'hidden_size': hidden_size, 'num_layers': num_layers}

print(f'Best Loss: {best_loss}, Best Params: {best_params}')

# 定义模型参数
input_size = 20
hidden_size = 100  # 最优隐藏层大小
num_layers = 2  # 最优层数
output_size = 1
lr = 0.001  # 最优学习率
num_epochs = 100

# 实例化模型
save_path = 'pth/model_parameters.pth'
torch.save(model.state_dict(), save_path)
model.load_state_dict(torch.load(save_path))
model = LSTMModel(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 训练模型并记录损失
epoch_losses = []
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    epoch_loss = 0
    for X_batch, Y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, Y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= len(train_loader)
    epoch_losses.append(epoch_loss)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}')
    
plt.plot(epoch_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Across Epochs')
plt.legend()
#plt.show()

# 训练模型并记录损失及相对预测误差
epoch_losses = []
relative_errors = []
num_epochs = 150
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for X_batch, Y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, Y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= len(train_loader)
    epoch_losses.append(epoch_loss)

    # 在评估模式下计算相对预测误差
    model.eval()
    total_error = 0
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            outputs = model(X_batch)
            error = np.abs(outputs.numpy().flatten() - Y_batch.numpy().flatten())
            total_error += np.sum(error / np.abs(Y_batch.numpy().flatten()))
    relative_error = total_error / len(test_loader.dataset)
    relative_errors.append(relative_error)

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}, Relative Prediction Error: {relative_error}')
    
    save_path = 'pth/stock_parameters.pth'
torch.save(model.state_dict(), save_path)

model.load_state_dict(torch.load(save_path))
model.eval()
all_predictions = []
all_actuals = []

with torch.no_grad():
    for X_batch, Y_batch in test_loader:
        # Predict
        outputs = model(X_batch).squeeze().cpu().numpy()  # Adjust based on your model's output
        all_predictions.extend(outputs)
        all_actuals.extend(Y_batch.squeeze().cpu().numpy())  # Adjust if necessary
        
# Convert lists to arrays for easier handling
all_predictions = np.array(all_predictions)
all_actuals = np.array(all_actuals)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(all_actuals, label='Actual Data', color='blue')
plt.plot(all_predictions, linestyle='--', color='red', label='Predicted Data')
# Only the first call to plot should have the label argument.

plt.title('Comparison of Actual and Predicted Data')
plt.xlabel('Time')
plt.ylabel('Normalized Stock Price')
plt.legend()
plt.show()
