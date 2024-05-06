from flask import Flask, request, render_template_string, send_file
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import io
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# LSTM模型类定义
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0, c0 = (torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device),
                  torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device))
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1])
        return out

def process_data(X, Y, Y_batch):
    # 分割数据
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=142)
    Y_batch_train, Y_batch_test = train_test_split(Y_batch, test_size=0.2, random_state=189)

    # 转换为张量
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).unsqueeze(-1)
    Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.float32)
    Y_batch_train_tensor = torch.tensor(Y_batch_train.values, dtype=torch.float32)

    return X_train_tensor, Y_train_tensor, Y_batch_train_tensor, X_test, Y_test, Y_batch_test

def train_model(X_train_tensor, Y_train_tensor):
    # 实例化模型
    model = LSTMModel(input_size=1, hidden_size=50, num_layers=1, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 训练模型
    dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    for epoch in range(5):
        for inputs, targets in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    return model

@app.route('/')
def index():
    return render_template_string("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Machine Learning Model Training</title>
        </head>
        <body>
            <h1>Upload CSV Files for Training</h1>
            <form action="/upload" method="post" enctype="multipart/form-data">
                X Dataset: <input type="file" name="X_file"><br><br>
                Y Dataset: <input type="file" name="Y_file"><br><br>
                Y_batch Dataset: <input type="file" name="Y_batch_file"><br><br>
                <input type="submit" value="Start Training">
            </form>
        </body>
        </html>
    """)

@app.route('/upload', methods=['POST'])
def upload_files():
    x_file = request.files['X_file']
    y_file = request.files['Y_file']
    y_batch_file = request.files['Y_batch_file']

    X = pd.read_csv(x_file)
    Y = pd.read_csv(y_file)
    Y_batch = pd.read_csv(y_batch_file)

    X_train_tensor, Y_train_tensor, Y_batch_train_tensor, X_test, Y_test, Y_batch_test = process_data(X, Y, Y_batch)

    model = train_model(X_train_tensor, Y_train_tensor)

    # 绘制结果
    fig, ax = plt.subplots()
    ax.plot(Y_train_tensor.numpy(), label='Actual')
    ax.plot(Y_train_tensor.numpy(), label='Predicted', linestyle='--')
    ax.legend()
    plt.xlabel('Samples')
    plt.ylabel('Values')
    plt.title('Training Result')

    # 将绘图结果转换为HTML可以显示的格式
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = "data:image/png;base64," + base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()

    return f'<img src="{plot_url}" />'

if __name__ == '__main__':
    app.run(debug=True, port=5000)
