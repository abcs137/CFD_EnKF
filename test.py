import numpy as np
from pyDOE2 import lhs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 参数设置部分
param_ranges = (0, 15)
x_range = (0, 1)
observed_variance = 0.2

# 设置随机种子
# np.random.seed(42)

def y_true(x, y):
    return 0.6 * (6 * (x + y) - 2)**2 * np.sin(12 * (x + y) - 4) + 2 * (x + y - 0.5) - 1

def y_model(x, y, params):
    a, b, c, d = params
    return (a * (x + y) - b)**2 * np.sin(c * (x + y) - d)

# 生成参数集合
num_samples = 50
num_params = 4
lhs_samples = lhs(num_params, samples=num_samples)

# 将样本映射到指定范围
params_samples = param_ranges[0] + (param_ranges[1] - param_ranges[0]) * lhs_samples

# 定义 x 和 y 的值
x_values = np.linspace(x_range[0], x_range[1], 100)
y_values = np.linspace(x_range[0], x_range[1], 100)
X, Y = np.meshgrid(x_values, y_values)

# 初始化结果矩阵
result_matrix = np.zeros((X.shape[0], X.shape[1], num_samples))

# 计算 y_model 的结果
for i, params in enumerate(params_samples):
    result_matrix[:, :, i] = y_model(X, Y, params)

# 计算 y_true 的结果
true_values = y_true(X, Y)

# 绘图
fig = plt.figure(figsize=(14, 7))

# 绘制 y_true
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, true_values, cmap='viridis')
ax1.set_title('True Function')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# 绘制 y_model 的第一个样本
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, result_matrix[:, :, 0], cmap='viridis')
ax2.set_title('Model Function (Sample 1)')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

plt.show()
