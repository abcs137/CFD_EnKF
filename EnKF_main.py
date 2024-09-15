import numpy as np
from pyDOE2 import lhs
import matplotlib.pyplot as plt


##参数设置部分

# 定义参数范围
param_ranges = (0, 15)

测试区间起点 = -1

测试区间终点 = 1

#观测的方差看介于(0,1]
观测的方差 = 0.5

迭代次数 = 10


# 设置随机种子
#np.random.seed(42)
def y_true(x):
    return 0.6 * (6 * x - 2)**2 * np.sin(12 * x - 4) + 2 * (x - 0.5) - 1

def y_model(x, params):
    a, b, c, d = params
    return (a * x - b)**2 * np.sin(c * x - d)


# 生成参数集合
num_samples = 50
num_params = 4
lhs_samples = lhs(num_params, samples=num_samples)
k=num_params
# 将样本映射到指定范围
params_samples = param_ranges[0] + (param_ranges[1] - param_ranges[0]) * lhs_samples

# 定义 x 的值
x_values =  np.linspace(测试区间起点, 测试区间终点, 100)

# 初始化结果矩阵
result_matrix = np.zeros((len(x_values) + num_params, num_samples))

# 计算 y_model 的结果，enumerate是遍历行，不转置代表遍历列
for i, params in enumerate(params_samples):
    result_matrix[:len(x_values), i] = y_model(x_values, params)
    result_matrix[len(x_values):, i] = params

# 计算观测值
y_observed = y_true(x_values)


# 假设状态向量维度 n = 14，观测向量维度 m = 10，集合成员数 N = 100
n = len(x_values) + num_params  # 状态向量的维度
m = len(x_values)
N = num_samples

print(m)
print(n)
###########################################################################

# 使用 result_matrix 作为预测状态矩阵 X_f
X_f = result_matrix  # 预测状态矩阵（n x N）
y = y_observed       # 观测向量（m,）

# 初始化观测矩阵 H
H = np.eye(m, n)
print(H.ndim)
print(H)
# 初始化协方差矩阵 P_f 和观测误差协方差矩阵 R

R = np.eye(m) * 观测的方差      # 观测误差协方差矩阵（m x m）
print(R.ndim)
print(R)
for iteration in range(迭代次数):

    P_f = np.cov(X_f)   # 状态协方差矩阵（n x n）
    # 计算增益矩阵 K
    K = P_f @ H.T @ np.linalg.inv(H @ P_f @ H.T + R)

    # 更新集合成员
    X_f = X_f + K @ (y[:, None] - H @ X_f)


row_means = np.mean(X_f, axis=1)
# 提取 row_means 的前十个元素
row_means_top_10 = row_means[:result_matrix.shape[0]-k]

# 初始化最小距离和索引
min_distance = float('inf')
min_index = -1

# 遍历 result_matrix 的列，计算每列前十个元素与 row_means_top_10 的欧氏距离
for i in range(result_matrix.shape[1]):
    column_top_10 = result_matrix[:result_matrix.shape[0]-k, i]  # 提取当前列的前十个元素
    distance = np.linalg.norm(row_means_top_10 - column_top_10)  # 计算欧氏距离
    if distance < min_distance:
        min_distance = distance
        min_index = i



# 提取最后四个均值作为参数 a, b, c, d
params_from_means = result_matrix[-k:, min_index]
print("Extracted Parameters:", params_from_means)

# 计算 y_model 和 y_true 的结果
y_model_from_means = y_model(x_values, params_from_means)
y_true_values = y_true(x_values)

# 绘制二维折线图
plt.plot(x_values, y_true_values, label='observe_data', color='blue')
plt.plot(x_values, y_model_from_means, label='calibrated_data', color='red')
plt.plot(x_values, y_model(x_values,[7.5,7.5,7.5,7.5]), label='raw_data', color='green')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('EnKF Comparison of y_true and y_model')
plt.show()