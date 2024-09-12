import numpy as np
from scipy.stats import qmc

# 定义变量
A1_Coefficient = 0.31
BetaStar_Coefficient = 0.09

K_EPSILON_REGIME_OMEGA_COEFFICIENTS_Alpha_Coefficient = 0.440355
K_EPSILON_REGIME_OMEGA_COEFFICIENTS_Beta_Coefficient = 0.0828

K_OMEGA_REGIME_OMEGA_COEFFICIENTS_Alpha_Coefficient = 0.553167
K_OMEGA_REGIME_OMEGA_COEFFICIENTS_Beta_Coefficient = 0.075

# 定义边界
def get_bounds(value):
    lower_bound = value * 0.4  # 减少20%
    upper_bound = value * 2.5  # 增加500%
    return lower_bound, upper_bound

bounds = np.array([
    get_bounds(A1_Coefficient),
    get_bounds(BetaStar_Coefficient),
    get_bounds(K_EPSILON_REGIME_OMEGA_COEFFICIENTS_Alpha_Coefficient),
    get_bounds(K_EPSILON_REGIME_OMEGA_COEFFICIENTS_Beta_Coefficient),
    get_bounds(K_OMEGA_REGIME_OMEGA_COEFFICIENTS_Alpha_Coefficient),
    get_bounds(K_OMEGA_REGIME_OMEGA_COEFFICIENTS_Beta_Coefficient)
])

# 获取维度数
dim = bounds.shape[0]

# 创建拉丁超立方采样器
sampler = qmc.LatinHypercube(d=dim)

# 生成区间内的样本点
n_samples = 50
sample = sampler.random(n=n_samples)

# 将样本点映射到指定的边界范围内
scaled_sample = qmc.scale(sample, bounds[:, 0], bounds[:, 1])
np.savetxt('scaled_sample_output.csv', scaled_sample, delimiter=',', fmt='%f')
# 保存到文件
with open("scaled_sample_output.txt", "w") as file:
    for i in range(n_samples):
        file.write(f"TURBULENCE MODEL:\n")
        file.write(f"    A1 Coefficient = {scaled_sample[i][0]:.6f}\n")
        file.write(f"    BetaStar Coefficient = {scaled_sample[i][1]:.6f}\n")
        file.write(f"    Option = SST\n")
        file.write(f"    K EPSILON REGIME:\n")
        file.write(f"      OMEGA COEFFICIENTS:\n")
        file.write(f"        Alpha Coefficient = {scaled_sample[i][2]:.6f}\n")
        file.write(f"        Beta Coefficient = {scaled_sample[i][3]:.6f}\n")
        file.write(f"      END\n")
        file.write(f"    END\n")
        file.write(f"    K OMEGA REGIME:\n")
        file.write(f"      OMEGA COEFFICIENTS:\n")
        file.write(f"        Alpha Coefficient = {scaled_sample[i][4]:.6f}\n")
        file.write(f"        Beta Coefficient = {scaled_sample[i][5]:.6f}\n")
        file.write(f"      END\n")
        file.write(f"    END\n")
        file.write(f"END\n\n")

print("样本已保存到 'scaled_sample_output.txt'")
