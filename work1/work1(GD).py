import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# 加载数据
train_data = pd.read_excel('Data4Regression.xlsx', sheet_name='Training Data')
test_data = pd.read_excel('Data4Regression.xlsx', sheet_name='Test Data')

X_train = train_data['x'].values.reshape(-1, 1)
y_train = train_data['y_complex'].values

X_test = test_data['x_new'].values.reshape(-1, 1)
y_test = test_data['y_new_complex'].values

# 梯度下降法
def gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    m = len(y)
    X_b = np.c_[np.ones((m, 1)), X]  # 添加偏置项
    theta = np.zeros(X_b.shape[1])  # 初始化参数
    for epoch in range(epochs):
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
        theta -= learning_rate * gradients
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, MSE: {mean_squared_error(y, X_b.dot(theta)):.4f}")
    return theta

# 训练模型
theta_gd = gradient_descent(X_train, y_train, learning_rate=0.01, epochs=1000)

# 模型预测
X_train_b = np.c_[np.ones((len(X_train), 1)), X_train]
y_pred_train = X_train_b.dot(theta_gd)
train_mse = mean_squared_error(y_train, y_pred_train)

X_test_b = np.c_[np.ones((len(X_test), 1)), X_test]
y_pred_test = X_test_b.dot(theta_gd)
test_mse = mean_squared_error(y_test, y_pred_test)

# 绘制拟合曲线
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='green', label='Test Data')
plt.plot(X_train, y_pred_train, color='red', label='Linear Fit with Gradient Descent')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with Gradient Descent')
plt.legend()
plt.show()

print(f"梯度下降法 - 训练误差: {train_mse:.4f}, 测试误差: {test_mse:.4f}")