import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree


class PressureEstimator:
    def __init__(self, data, k=5):
        """
        初始化 PressureEstimator 类，并构建 Ball 树。

        参数：
        data (pd.DataFrame): 包含数据的 DataFrame，必须包含 'x-coordinate', 'y-coordinate', 'pressure', 'cellnumber' 列。
        k (int): 最近邻的点的数量。
        """
        self.data = data
        self.k = k
        self.coordinates = data[['x-coordinate', 'y-coordinate']].values
        self.pressures = data['pressure'].values
        self.cellnumbers = data['cellnumber'].values
        self.tree = BallTree(self.coordinates)

    def estimate_pressure(self, target_x, target_y):
        """
        估计指定坐标点的压力值，并打印出距离该点最近的 k 个点的编号（cellnumber）。

        参数：
        target_x (float): 目标点的 x 坐标。
        target_y (float): 目标点的 y 坐标。

        返回：
        float: 估计的压力值。
        """
        distances, indices = self.tree.query([[target_x, target_y]], k=self.k)
        nearest_pressures = self.pressures[indices[0]]

        # 打印这些点的编号
        #nearest_cells = self.cellnumbers[indices[0]]
        #print("The nearest cell numbers are:", nearest_cells.tolist())

        # 使用这些点的 pressure 值估计目标点的 pressure 值（简单平均）
        estimated_pressure = np.mean(nearest_pressures)

        return estimated_pressure
