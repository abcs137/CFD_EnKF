import numpy as np

def Ztrue(x, y, params):
    a, b, c, d, e, f, g = params
    return a * (x**2 + y**2) + b * np.sin(c * x + d * y) + e * np.log(f * x**2 + g * y**2 + 1)

def Zmodel(x, y, params):
    a, b, c, d = params
    return a * (x**2 + y**2) + b * np.sin(c * x + d * y)


true_params = [1.0, 2.0, 3.0, 4.0, 0.1, 0.2, 0.3]
x_data = np.linspace(-5, 5, 100)
y_data = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x_data, y_data)
Z_obs = Ztrue(X, Y, true_params).flatten()  # 展平成一维数组



def initialize_ensemble(N, param_dim):
    ensemble = np.random.rand(N, param_dim)  # 随机初始化参数集合
    return ensemble

def predict(ensemble, x, y):
    N = ensemble.shape[0]
    Z_pred = np.array([Zmodel(x, y, ensemble[i]).flatten() for i in range(N)])
    return Z_pred


def update(ensemble, Z_pred, Z_obs, R):
    N = ensemble.shape[0]
    param_dim = ensemble.shape[1]
    obs_dim = Z_obs.shape[0]
    
    ensemble_mean = np.mean(ensemble, axis=0)
    Z_pred_mean = np.mean(Z_pred, axis=0)
    
    P = np.cov(ensemble.T)
    H = np.eye(param_dim, obs_dim)  # 确保H矩阵维度正确
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)  # 卡尔曼增益
    
    for i in range(N):
        ensemble[i] += K @ (Z_obs - Z_pred[i])
    
    return ensemble


def ensemble_kalman_filter(x, y, Z_obs, initial_params, R, iterations=10, N=100):
    param_dim = len(initial_params)
    ensemble = initialize_ensemble(N, param_dim)
    
    for _ in range(iterations):
        Z_pred = predict(ensemble, x, y)
        ensemble = update(ensemble, Z_pred, Z_obs, R)
    
    return np.mean(ensemble, axis=0)


initial_params = [0.5, 1.0, 2.0, 3.0]  # 初始参数猜测
R = np.eye(10000)  # 观测误差协方差矩阵，根据展平后的观测数据维度
estimated_params = ensemble_kalman_filter(X, Y, Z_obs, initial_params, R)

print("Estimated parameters:", estimated_params)
