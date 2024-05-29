import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree

class PressurePropagator:
    def __init__(self, data, k=5):
        """
        初始化 PressurePropagator 类，并构建 Ball 树。

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
    
    def modify_and_propagate(self, target_x, target_y, new_pressure):
        """
        修改指定点的压力值，并根据距离远近将这个修改传播到最近邻的 k 个点。

        参数：
        target_x (float): 目标点的 x 坐标。
        target_y (float): 目标点的 y 坐标。
        new_pressure (float): 目标点的新压力值。

        返回：
        pd.DataFrame: 修改后的 DataFrame。
        """
        # 找到距离目标坐标最近的 k 个点
        distances, indices = self.tree.query([[target_x, target_y]], k=self.k)
        nearest_indices = indices[0]
        nearest_distances = distances[0]

        # 计算权重（距离越近，权重越高）
        weights = 1 / nearest_distances
        weights /= weights.sum()  # 归一化权重，使其和为1

        # 打印这些点的编号
        # nearest_cells = self.cellnumbers[nearest_indices]
        # print("The nearest cell numbers are:", nearest_cells.tolist())

        # 修改目标点的压力值
        target_index = self.data[(self.data['x-coordinate'] == target_x) & (self.data['y-coordinate'] == target_y)].index[0]
        original_pressure = self.data.at[target_index, 'pressure']
        self.data.at[target_index, 'pressure'] = new_pressure

        # 计算压力变化量
        pressure_change = new_pressure - original_pressure

        # 将压力变化传播到最近邻的 k 个点
        for i, idx in enumerate(nearest_indices):
            if idx != target_index:  # 排除目标点本身
                self.data.at[idx, 'pressure'] += pressure_change * weights[i]

        return self.data
