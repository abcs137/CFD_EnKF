import numpy as np
from pyDOE2 import lhs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置随机种子
#np.random.seed(42)

def y_true(x, y):
    return 0.6 * (6 * x - 2)**2 * np.sin(12 * y - 4) + 2 * (x - 0.5) - 1

def y_model(x, y, params):
    a, b, c, d, e, f = params
    return (a * x - b)**2 * np.sin(c * y - d) + e * x + f

# 定义参数范围
param_ranges = (0, 15)

# 生成100个参数集合
num_samples = 100
num_params = 6
lhs_samples = lhs(num_params, samples=num_samples)

# 将样本映射到指定范围
params_samples = param_ranges[0] + (param_ranges[1] - param_ranges[0]) * lhs_samples

# 定义 x 和 y 的值，创建二维网格
x_values = np.linspace(0, 1, 20)
y_values = np.linspace(0, 1, 20)
X_grid, Y_grid = np.meshgrid(x_values, y_values)

# 展平网格以简化计算
X_flat = X_grid.flatten()
Y_flat = Y_grid.flatten()

# 初始化结果矩阵
result_matrix = np.zeros((len(X_flat) + num_params, num_samples))

# 计算 y_model 的结果
for i, params in enumerate(params_samples):
    result_matrix[:len(X_flat), i] = y_model(X_flat, Y_flat, params)
    result_matrix[len(X_flat):, i] = params

# 计算观测值
y_observed = y_true(X_flat, Y_flat)

# 假设状态向量维度 n，观测向量维度 m，集合成员数 N
n = len(X_flat) + num_params  # 状态向量的维度
m = len(X_flat)
N = num_samples

# 使用 result_matrix 作为预测状态矩阵 X_f
X_f = result_matrix  # 预测状态矩阵（n x N）
y = y_observed       # 观测向量（m,）

# 初始化观测矩阵 H
H = np.eye(m, n)

# 初始化协方差矩阵 P_f 和观测误差协方差矩阵 R
R = np.eye(m) * 0.2  # 观测误差协方差矩阵（m x m）

for iteration in range(10):
    P_f = np.cov(X_f)  # 状态协方差矩阵（n x n）
    # 计算增益矩阵 K
    K = P_f @ H.T @ np.linalg.inv(H @ P_f @ H.T + R)
    # 更新集合成员
    X_f = X_f + K @ (y[:, None] - H @ X_f)

row_means = np.mean(X_f, axis=1)
row_means_top_10 = row_means[:len(X_flat)-6]

# 初始化最小距离和索引
min_distance = float('inf')
min_index = -1

# 遍历 result_matrix 的列，计算每列的欧氏距离
for i in range(result_matrix.shape[1]):
    column_top_10 = result_matrix[:len(X_flat)-6, i]  # 提取当前列的前 len(X_flat) 个元素
    distance = np.linalg.norm(row_means_top_10 - column_top_10)  # 计算欧氏距离
    if distance < min_distance:
        min_distance = distance
        min_index = i

# 输出最相似列的索引以及该列的最后六个元素
params_from_means = result_matrix[-6:, min_index]
print("Extracted Parameters:", params_from_means)

# 计算 y_model 和 y_true 的结果
y_model_from_means = y_model(X_flat, Y_flat, params_from_means)
y_true_values = y_true(X_flat, Y_flat)
y_raw = y_model(X_flat, Y_flat, [7, 7, 7, 7, 7, 7])

# 重塑回网格以便于3D绘图
y_true_values = y_true_values.reshape(X_grid.shape)
y_model_from_means = y_model_from_means.reshape(X_grid.shape)
y_raw = y_raw.reshape(X_grid.shape)

# 绘制3D图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制真实值的曲面
#ax.plot_surface(X_grid, Y_grid, y_true_values, cmap='viridis', alpha=0.7, label='Observed Data')

# 绘制模型估计值的曲面
#ax.plot_surface(X_grid, Y_grid, y_model_from_means, cmap='Blues', alpha=0.7, label='Calibrated Data')

# 绘制未处理数据的曲面
#ax.plot_surface(X_grid, Y_grid, y_raw, cmap='spring', alpha=0.7, label='Raw Data')

# 绘制未处理数据的曲面
ax.plot_surface(X_grid, Y_grid, y_true_values-y_model_from_means, cmap='inferno', alpha=0.7, label='diff')

# 设置轴标签和标题
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Comparison of y_true, y_model, and raw_data')

plt.show()
