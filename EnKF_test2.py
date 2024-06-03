import numpy as np
import matplotlib.pyplot as plt
# 定义真实函数和模型函数
def y_true(x):
    return 0.6 * (6 * x - 2)**2 * np.sin(12 * x - 4) + 2 * (x - 0.5) - 1

def y_model(x, params):
    a, b, c, d = params
    return (a * x - b)**2 * np.sin(c * x - d)

# 初始化参数
theta_0 = np.array([6, 2, 12, 5])
num_ensembles = 100
param_dim = len(theta_0)
x = np.linspace(0, 1, 11)
observations = y_true(x)

# 创建初始集合
ensembles = np.random.normal(loc=theta_0, scale=1.0, size=(num_ensembles, param_dim))

# 定义噪声协方差矩阵
R = np.eye(len(x)) * 0.1

# 集合卡尔曼滤波
for iteration in range(50):  # 假设迭代 50 次
    # 预测步骤
    predictions = np.array([y_model(x, ensemble) for ensemble in ensembles])
    
    # 计算集合平均
    ensemble_mean = np.mean(predictions, axis=0)
    param_mean = np.mean(ensembles, axis=0)
    
    # 计算协方差矩阵
    P_yy = np.cov(predictions, rowvar=False)  # 预测输出的协方差矩阵
    P_xy = np.cov(ensembles.T, predictions.T)[:param_dim, param_dim:]  # 参数与预测输出的协方差矩阵
    
    # 计算卡尔曼增益矩阵 K
    HPHTR_inv = np.linalg.inv(P_yy + R)
    K = P_xy @ HPHTR_inv
    
    # 更新参数集合
    for i in range(num_ensembles):
        ensembles[i] += K @ (observations - predictions[i])

# 计算最终参数平均值
final_params = np.mean(ensembles, axis=0)
print(f'Final parameters: {final_params}')




# 计算初始模型函数值
initial_predictions = y_model(x, theta_0)

# 计算校准后的模型函数值
calibrated_predictions = y_model(x, final_params)

# 绘制对比图
plt.figure(figsize=(10, 6))
plt.plot(x, observations, 'ro', label='Observations (y_true)')
plt.plot(x, initial_predictions, 'b-', label='Initial Model (y_model with initial params)')
plt.plot(x, calibrated_predictions, 'g--', label='Calibrated Model (y_model with calibrated params)')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Comparison of y_true, Initial y_model, and Calibrated y_model')
plt.show()
