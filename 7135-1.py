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

directory = "/Users/vyriYang/documents/research/7135/data"

start_date = pd.to_datetime('2010-07-16').date()
end_date = pd.to_datetime('2020-12-24').date()
dataframes = {}
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(directory, filename)
        df = pd.read_csv(file_path)
        df = df.iloc[:, [0, 4]]
        #print(df.shape)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Date'] = df['Date'].dt.date
        mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
        filtered_df = df.loc[mask]
        print(filtered_df.shape)
        print(filtered_df.isna().sum())
        df_name = filename[:-4]
        print(df_name)
        dataframes[df_name] = filtered_df
        exec(f"{df_name} = filtered_df")
        
# 创建图形和轴
plt.figure(figsize=(14, 7))

# 对于字典中的每个DataFrame
for stock_name, df in dataframes.items():
    # 确保日期列是datetime类型并按日期排序
    df.sort_values('Date', inplace=True)
    plt.plot(df['Date'], df['Close'], label=stock_name)

# 添加图例
plt.legend()

# 添加标题和轴标签
plt.title('Stock Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price')

# 显示图表
plt.show()

#################################
# We need to standardize the data

for stock_name, df in dataframes.items():
    print(stock_name)
    close_mean = df['Close'].mean()
    close_std = df['Close'].std()
    df['Close_std'] = (df['Close'] - close_mean) / close_std
    
print(AAPL.head())

# 创建图形和轴
plt.figure(figsize=(14, 7))

# 对于字典中的每个DataFrame
for stock_name, df in dataframes.items():
    df.sort_values('Date', inplace=True)
    plt.plot(df['Date'], df['Close_std'], label=stock_name)

# 添加图例
plt.legend()

# 添加标题和轴标签
plt.title('Stock Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price')

# 显示图表
plt.show()
