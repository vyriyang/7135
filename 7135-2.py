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

stock_names = ['AAPL', 'AMZN', 'BAC', 'BH', 'JPM', 'MSFT', 'TM', 'VZ', 'WFC', 'XOM']

for stock_name in stock_names:
    try:
        close_std_series = eval(stock_name)['Close_std']
        close_std_df = pd.DataFrame({f"{stock_name}_std": close_std_series.values})
        exec(f"{stock_name} = close_std_df")
    except NameError as e:
        print(f"Variable {stock_name} does not exist: {e}")
    except KeyError as e:
        print(f"DataFrame for {stock_name} does not contain 'Close_std': {e}")

print(AAPL.head(20))

# 创建一个字典来存储新的DataFrame
stock_features = {}

for stock_name in stock_names:
    try:
        df = eval(stock_name).copy()
        
        for i in range(19, -1, -1):
            df[f'Close_std_D-{i+1}'] = df[f'{stock_name}_std'].shift(i)
        # 创建预测目标
        df['Predicted_Close_std'] = df[f'{stock_name}_std'].shift(-1)
        # 删除因shift操作产生的NaN行
        df = df.dropna()
        df.drop(df.columns[0], axis=1, inplace=True)
        # 将处理后的DataFrame存入字典
        stock_features[stock_name] = df
    except NameError as e:
        print(f"Variable {stock_name} does not exist: {e}")
    except KeyError as e:
        print(f"DataFrame for {stock_name} does not contain 'Close_std': {e}")

# 查看一个样本股票的新DataFrame
sample_stock = list(stock_features.keys())[0]
print(f"Features and Predict Price for {sample_stock}:")
print(stock_features[sample_stock].head(10))

for stock_name in stock_names:
    try:
        df = eval(stock_name).copy()

        # 添加历史收盘价的标准化值
        for i in range(19, -1, -1):
            df[f'Close_std_D-{i+1}'] = df[f'{stock_name}_std'].shift(i)

        # 创建未来的预测目标
        df['Predicted_Close_std'] = df[f'{stock_name}_std'].shift(-1)

        # 生成涨跌标签
        df['Y_batch'] = (df['Predicted_Close_std'] > df['Close_std_D-1']).astype(int)

        # 删除因shift操作产生的NaN行
        df = df.dropna()
        df.drop(df.columns[0], axis=1, inplace=True)

        # 将处理后的DataFrame存入字典
        stock_features[stock_name] = df
    except NameError as e:
        print(f"Variable {stock_name} does not exist: {e}")
    except KeyError as e:
        print(f"DataFrame for {stock_name} does not contain 'Close_std': {e}")

# 查看一个样本股票的新DataFrame，包括Y_batch列
sample_stock = list(stock_features.keys())[0]
print(f"Features and Predict Price for {sample_stock}:")
print(stock_features[sample_stock].head(10))

# 首先创建一个空的DataFrame来存储所有的Y_batch数据
all_Y_batch = pd.DataFrame()

for stock_name in stock_names:
    if stock_name in stock_features:
        # 从每个股票的DataFrame中提取Y_batch列，并添加到新的DataFrame中
        all_Y_batch = pd.concat([all_Y_batch, stock_features[stock_name]['Y_batch']], ignore_index=True)

# 将合并后的Y_batch数据保存到CSV文件中
all_Y_batch.to_csv('/Users/vyriYang/documents/research/7135/csv-/7135_Y_batch.csv', index=False)

print("Y_batch data saved successfully.")

combined_features = pd.DataFrame()
# 遍历stock_features中的每个DataFrame
for stock_name, df in stock_features.items():
    # 提取特定的特征列
    feature_columns = df.iloc[:, :20]
    # 合并到大的DataFrame中
    combined_features = pd.concat([combined_features, feature_columns])

# 由于合并可能打乱了索引，重新设置索引为1,2,3,4,...
combined_features.index = range(1, len(combined_features) + 1)

# 查看合并后的DataFrame
print(combined_features.head())
# 存储DataFrame到CSV文件
combined_features.to_csv('/Users/vyriYang/documents/research/7135/csv-/7135_X.csv', index=False)

# 使用列表来收集所有的Predicted_Close_std值
predicted_values = []

for stock_name, df in stock_features.items():
    predicted_values.append(df['Predicted_Close_std'])

Y = pd.concat(predicted_values)
Y.to_csv('/Users/vyriYang/documents/research/7135/csv-/7135_Y.csv', index=False)

# 查看合并后的Y
print(Y.head())
print(len(Y))
