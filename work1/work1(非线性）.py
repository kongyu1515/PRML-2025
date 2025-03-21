import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd

# 加载数据
# data = np.loadtxt('Data4Regression.xlsx') # 原始代码
data = pd.read_excel('Data4Regression.xlsx') # 修正后的代码
x = data.iloc[:, 0].values.reshape(-1, 1)
y = data.iloc[:, 1].values

# 划分训练集和测试集
train_size = int(0.8 * len(x))
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 创建并训练神经网络回归模型
model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=2000, random_state=42)
model.fit(x_train, y_train)

# 预测
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

# 计算损失
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
print(f"训练集均方误差: {train_mse:.6f}")
print(f"测试集均方误差: {test_mse:.6f}")

# 绘制拟合曲线
plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train, color='blue', label='训练数据点', alpha=0.6)
plt.scatter(x_test, y_test, color='green', label='测试数据点', alpha=0.6)
plt.plot(x, model.predict(x), color='red', label='拟合曲线', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('神经网络回归拟合结果')
plt.legend()
plt.show()