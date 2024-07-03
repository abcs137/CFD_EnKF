import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize

# 定义Ztrue函数
def Ztrue(x, y, params):
    a, b, c, d, e, f, g = params
    return a * (x**2 + y**2) + b * np.sin(c * x + d * y) + e * np.log(f * x**2 + g * y**2 + 1)

# 定义Ztest函数
def Zmodel(x, y, params):
    a, b, c, d = params
    return a * (x**2 + y**2) + b * np.sin(c * x + d * y)

# 创建测试点
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x, y)

# 计算Ztrue值
params_true = [1, 10, 5, 2, 0.5, 2, 3]
Z_true = Ztrue(X, Y, params_true)

# 定义误差函数
def error_function(params):
    Z_test = Zmodel(X, Y, params)
    return np.mean((Z_test - Z_true) ** 2)

# 使用SciPy的minimize函数进行优化
initial_guess = [1, 10, 5, 2]
result = minimize(error_function, initial_guess, method='L-BFGS-B')
params_optimized = result.x

# 计算优化后的Ztest值
Z_test_optimized = Zmodel(X, Y, params_optimized)

Zraw = Zmodel(X, Y, params_true[:4])
Zoptimized = Z_test_optimized
# 创建一个新的3D图形对象
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制第一个表面图，设置颜色和透明度
surf1 = ax.plot_surface(X, Y, Zraw, color='blue', alpha=0.5, label='Zraw')

# 绘制第二个表面图，设置颜色和透明度
surf2 = ax.plot_surface(X, Y, Zoptimized, color='red', alpha=0.5, label='Zoptimized')

# 添加图例
# 注意：matplotlib 的 plot_surface 不直接支持 label 属性，所以我们需要自定义图例
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color='blue', lw=4, label='Zraw'),
                   Line2D([0], [0], color='red', lw=4, label='Zoptimized')]
ax.legend(handles=legend_elements)

# 设置轴标签
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# 显示图形
plt.show()

# 输出优化后的参数
print("Optimized parameters:", params_optimized)
