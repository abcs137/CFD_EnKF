import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc

# 定义边界
bounds = np.array([[1.0, 2.0], [3.0, 4.0], [1.0, 3.0], [1.0, 3.0], [1.0, 3.0], [1.0, 3.0], [2.0, 5.0]])

# 获取维度数
dim = bounds.shape[0]

# 创建拉丁超立方采样器
sampler = qmc.LatinHypercube(d=dim)

# 生成 [0, 1) 区间内的样本点
sample = sampler.random(n=100)

# 将样本点映射到指定的边界范围内
scaled_sample = qmc.scale(sample, bounds[:, 0], bounds[:, 1])

# 投影到2维空间 (例如，选择第1列和第2列)
projected_sample = scaled_sample[:, :2]

# 绘制散点图
plt.scatter(projected_sample[:, 0], projected_sample[:, 1])
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('2D Projection of Latin Hypercube Sample')
plt.grid(True)
plt.show()
