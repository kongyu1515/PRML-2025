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

# 牛顿法进行线性回归
def newton_method(X, y, max_iter=100, tol=1e-6):
    # 添加偏置项
    X_b = np.c_[np.ones((len(X), 1)), X]
    # 初始化参数
    theta = np.zeros(X_b.shape[1])
    # 迭代更新
    for _ in range(max_iter):
        # 计算梯度（一阶导数）
        gradient = X_b.T.dot(X_b.dot(theta) - y)
        # 计算Hessian矩阵（二阶导数）
        Hessian = X_b.T.dot(X_b)
        # 牛顿法更新规则
        theta_new = theta - np.linalg.inv(Hessian).dot(gradient)
        # 收敛判断
        if np.linalg.norm(theta_new - theta) < tol:
            break
        theta = theta_new
    return theta

# 使用牛顿法训练模型
theta_newton = newton_method(X_train, y_train)

# 模型预测
X_train_b = np.c_[np.ones((len(X_train), 1)), X_train]
y_pred_train = X_train_b.dot(theta_newton)
train_mse = mean_squared_error(y_train, y_pred_train)

X_test_b = np.c_[np.ones((len(X_test), 1)), X_test]
y_pred_test = X_test_b.dot(theta_newton)
test_mse = mean_squared_error(y_test, y_pred_test)

# 绘制拟合曲线
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='green', label='Test Data')
plt.plot(X_train, y_pred_train, color='red', label='Linear Fit with Newton Method')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with Newton Method')
plt.legend()
plt.show()

print(f"牛顿法 - 训练误差: {train_mse:.4f}, 测试误差: {test_mse:.4f}")