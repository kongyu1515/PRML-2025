# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib as mpl

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
mpl.style.use('dark_background')  # 使用dark_background样式

# 设置随机种子
torch.manual_seed(0)
np.random.seed(0)

# 1. 读取并预处理数据
df = pd.read_csv("LSTM-Multivariate_pollution.csv")
df.columns = df.columns.str.strip()  # 去除列名前后的空格
print("列名：", df.columns.tolist())

# 选取需要的特征
data = df[['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']]
data = data.copy()
data.dropna(inplace=True)

# 编码风向
encoder = LabelEncoder()
data['wnd_dir'] = encoder.fit_transform(data['wnd_dir'])

# 归一化
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)


# 2. 构建监督学习数据：用过去 24 小时预测下一小时
def create_dataset(data, lookback=24):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback])  # 预测所有特征
    return np.array(X), np.array(y)


lookback = 24
X, y = create_dataset(scaled, lookback)

# 划分训练集和测试集
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 转换为 Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


# 3. 构建 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)  # 输出大小与输入相同

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # 只取最后时刻的输出


model = LSTMModel(input_size=X.shape[2])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
print("开始训练模型...")
loss_history = []
for epoch in range(10):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        output = model(batch_X)
        loss = criterion(output, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(train_loader)
    loss_history.append(avg_loss)
    print(f"Epoch {epoch + 1}: 平均损失 = {avg_loss:.4f}")

# 绘制训练损失曲线
plt.figure(figsize=(10, 5))
plt.plot(loss_history, 'o-', color='#FF6B6B', linewidth=2, markersize=8)
plt.title('训练损失变化曲线', fontsize=14)
plt.xlabel('训练轮次', fontsize=12)
plt.ylabel('MSE损失', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(range(10), range(1, 11))
plt.tight_layout()
plt.show()

# 4. 预测测试集 MSE
model.eval()
with torch.no_grad():
    test_pred = model(X_test_tensor)
    mse = mean_squared_error(y_test_tensor.numpy(), test_pred.numpy())
    print(f"测试集MSE损失: {mse:.4f}")

# 5. 预测未来 24 小时
input_seq = torch.tensor(scaled[-lookback:], dtype=torch.float32).unsqueeze(0)  # shape: [1, 24, features]
cols_to_predict = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
feature_names = ['污染指数', '露点温度', '气温', '气压', '风向', '风速', '降雪量', '降雨量']
future_preds = {col: [] for col in cols_to_predict}  # 存储所有特征的预测值

model.eval()
with torch.no_grad():
    for _ in range(24):
        pred = model(input_seq)  # 预测所有特征
        for i, col in enumerate(cols_to_predict):
            future_preds[col].append(pred[0][i].item())

        # 构造下一步输入
        next_input = input_seq[:, 1:, :].clone()  # 移除最早一个
        next_features = input_seq[:, -1, :].clone()
        next_features[0] = pred[0].detach()  # 用预测的所有特征替代
        next_input = torch.cat((next_input, next_features.unsqueeze(1)), dim=1)
        input_seq = next_input

# 反归一化预测值
future_preds_inv = {}
for col in cols_to_predict:
    feature_index = cols_to_predict.index(col)
    dummy = np.zeros((24, len(cols_to_predict)))
    dummy[:, feature_index] = np.ravel(np.array(future_preds[col]))
    inv = scaler.inverse_transform(dummy)
    future_preds_inv[col] = inv[:, feature_index]

# 6. 绘制精美的预测图表
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#A37EBD', '#FFA07A', '#98D8C8', '#F06292', '#7986CB']
line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']

plt.figure(figsize=(16, 12))
plt.suptitle('多变量LSTM模型对未来24小时的预测结果', fontsize=16, y=1.02)

for i, (col, name) in enumerate(zip(cols_to_predict, feature_names)):
    ax = plt.subplot(3, 3, i + 1)

    # 绘制预测曲线
    ax.plot(range(1, 25), future_preds_inv[col],
            color=colors[i], linestyle=line_styles[i],
            linewidth=2, marker='o', markersize=6,
            label=f'{name}预测值')

    # 添加填充区间（模拟置信区间）
    lower = np.array(future_preds_inv[col]) * 0.95
    upper = np.array(future_preds_inv[col]) * 1.05
    ax.fill_between(range(1, 25), lower, upper,
                    color=colors[i], alpha=0.2)

    # 设置图表元素
    ax.set_title(f'{name}预测', fontsize=12)
    ax.set_xlabel('未来小时数', fontsize=10)
    ax.set_ylabel(name, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_facecolor('#F5F5F5')  # 设置背景色

    # 根据不同类型设置不同y轴范围
    if col == 'pollution':
        ax.set_ylim(0, max(future_preds_inv[col]) * 1.2)
    elif col in ['temp', 'dew']:
        ax.set_ylim(min(future_preds_inv[col]) - 5, max(future_preds_inv[col]) + 5)

# 调整布局
plt.tight_layout()
plt.subplots_adjust(top=0.92, hspace=0.4, wspace=0.3)

# 添加总结信息框
summary_text = f"""模型性能总结:
- 测试集MSE损失: {mse:.4f}
- 训练轮次: 10
- 预测时间范围: 24小时
"""
plt.figtext(0.5, 0.02, summary_text, ha='center', fontsize=11,
            bbox=dict(facecolor='#F8F9F9', edgecolor='#D5DBDB', boxstyle='round,pad=0.5'))

plt.show()