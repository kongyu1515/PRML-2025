# Generating 3D make-moons data

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def make_moons_3d(n_samples=500, noise=0.1):
    # Generate the original 2D make_moons data
    t = np.linspace(0, 2 * np.pi, n_samples)
    x = 1.5 * np.cos(t)
    y = np.sin(t)
    z = np.sin(2 * t)  # Adding a sinusoidal variation in the third dimension

    # Concatenating the positive and negative moons with an offset and noise
    X = np.vstack([np.column_stack([x, y, z]), np.column_stack([-x, y - 1, -z])])
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])

    # Adding Gaussian noise
    X += np.random.normal(scale=noise, size=X.shape)

    return X, y

# Generate the data (1000 datapoints)
X, labels = make_moons_3d(n_samples=1000, noise=0.2)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='viridis', marker='o')
legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
ax.add_artist(legend1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('3D Make Moons')
plt.show()
# 数据准备与可视化
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC

# 生成训练数据 (1000个样本)
X_train, y_train = make_moons_3d(n_samples=1000, noise=0.2)

# 生成测试数据 (500个样本)
X_test, y_test = make_moons_3d(n_samples=250, noise=0.2)  # 注意这里实际生成500个样本

# 模型训练与评估函数
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

# 初始化各分类器
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "AdaBoost (DT)": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=200,
        random_state=42
    ),
    "SVM (Linear)": SVC(kernel='linear', random_state=42),
    "SVM (RBF)": SVC(kernel='rbf', gamma='scale', random_state=42),
    "SVM (Poly)": SVC(kernel='poly', degree=3, gamma='scale', random_state=42)
}

# 训练与评估
results = {}
for name, model in models.items():
    acc = evaluate_model(model, X_train, y_train, X_test, y_test)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")

# 结果可视化
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
plt.bar(results.keys(), results.values(), color=colors)
plt.ylim(0.8, 1.0)
plt.ylabel('Accuracy')
plt.title('Classifier Performance Comparison on 3D Moons Dataset')
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, v in enumerate(results.values()):
    plt.text(i-0.1, v+0.01, f"{v:.4f}", color='black', fontweight='bold')
plt.show()