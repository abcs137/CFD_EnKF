import numpy as np
import matplotlib.pyplot as plt

# 定义网格范围和分辨率
x = np.linspace(-10, 10, 20)
y = np.linspace(-10, 10, 20)
X, Y = np.meshgrid(x, y)

# 定义速度场函数
U = -Y
V = X

# 计算每个矢量的大小（范数）
magnitude = np.sqrt(U**2 + V**2)

# 绘制速度场，并用颜色表示场强
plt.figure()
quiver = plt.quiver(X, Y, U, V, magnitude, cmap='plasma')  # 使用 viridis colormap
plt.colorbar(quiver, label='Magnitude')  # 添加颜色条
plt.title('2D Velocity Field with Color-coded Magnitude')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.show()
