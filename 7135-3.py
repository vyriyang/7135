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


X = pd.read_csv('/Users/vyriYang/documents/research/7135/csv/7135_X.csv')
Y = pd.read_csv('/Users/vyriYang/documents/research/7135/csv/7135_Y.csv')
Y_batch = pd.read_csv('/Users/vyriYang/documents/research/7135/csv/7135_Y_batch.csv')
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

