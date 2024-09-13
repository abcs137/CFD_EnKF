import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree

class PressureEstimator:
    coordinates = 0

    Btree = 0

    def __init__(self, data, k=5):
        """
        初始化 PressureEstimator 类，并构建 Ball 树。

        参数：
        data (pd.DataFrame): 包含数据的 DataFrame，必须包含 'X [ m ]', 'Y [ m ]', 'Z [ m ]', ' Total Pressure [ Pa ]', 列。
        k (int): 最近邻的点的数量。
        """
        self.data = data
        self.k = k
        
        if(not isinstance(PressureEstimator.coordinates,np.ndarray)):
            PressureEstimator.coordinates = data[['X [ m ]', 'Y [ m ]', 'Z [ m ]']].values
        self.pressures = data['Total Pressure [ Pa ]'].values
        
        if(not isinstance(PressureEstimator.Btree,BallTree)):   
            PressureEstimator.Btree = BallTree(PressureEstimator.coordinates)

    def estimate_pressure(self, target_x, target_y, target_z):
        """
        估计指定坐标点的压力值

        参数：
        target_x (float): 目标点的 x 坐标。
        target_y (float): 目标点的 y 坐标。
        target_z (float): 目标点的 z 坐标。

        返回：
        float: 估计的压力值。
        """
        distances, indices = PressureEstimator.Btree.query([[target_x, target_y, target_z]], k=self.k)
        nearest_pressures = self.pressures[indices[0]]


        # 使用加权平均来估计压力值
        weights = 1 / (distances[0] + 1e-10)  # 加一个小常数避免除零错误
        estimated_pressure = np.dot(weights, nearest_pressures) / np.sum(weights)

        return estimated_pressure
"""
# 示例用法
# 创建一个示例数据框（请根据实际数据调整列名和数据）
data = pd.DataFrame({
    'X [ m ]': [0.0, 1.0, 2.0, 3.0, 4.0],
    'Y [ m ]': [0.0, 1.0, 2.0, 3.0, 4.0],
    'Z [ m ]': [0.0, 1.0, 2.0, 3.0, 4.0],
    'Total Pressure [ Pa ]': [100, 150, 200, 250, 300],

})

estimator = PressureEstimator(data, k=3)
estimated_pressure = estimator.estimate_pressure(1.5, 1.5, 1.5)
print(f"Estimated Pressure: {estimated_pressure}")
"""