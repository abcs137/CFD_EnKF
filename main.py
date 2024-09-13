import pandas as pd
import numpy as np
import os
from PressureEstimator import PressureEstimator
from read_csv_files import get_csv_files_from_folder, read_csv_files
from 读取实验数据 import read_dat_file
import time
if __name__ == '__main__':
    # 记录开始时间
    start_time = time.perf_counter()
    from multiprocessing import freeze_support
    freeze_support()
    estimator = 0

    x_values=[]
    y_values=[]
    z_values=[]


    # Example usage:

    参数表=pd.read_csv('scaled_sample_output.csv',header=None)
    参数表=参数表.to_numpy()

    #参数数量
    k=6


    实验数据 = read_dat_file('i0_MA=0.69_压力面.dat')


    m=实验数据.shape[0] + 1
    n=m+k


    y_observed=实验数据.loc[:,['Ps_ps']].to_numpy().flatten()


    for row in 实验数据.loc[:,['x','y','z']].itertuples(index=True, name='Pandas'):
        x_values.append(row.x)
        y_values.append(row.y)
        z_values.append(row.z)

    result_matrix=np.zeros((n-1,50))

    folder_path = '.\data'  # Replace with the path to your folder containing CSV files
    csv_files = get_csv_files_from_folder(folder_path)
    dataframes_from_folder = read_csv_files(csv_files)
    for index, df in enumerate(dataframes_from_folder):
        estimator = PressureEstimator(df, k=10)
        当前列压力=[]
        for row in 实验数据.loc[:,['x','y','z']].itertuples(index=True, name='Pandas'):
            当前列压力.append(estimator.estimate_pressure(row.x , row.y , row.z) / 1000)  
            
        result_matrix[0:m-1,index]=当前列压力

    result_matrix[-k:,:]=参数表.T


    #np.savetxt('test111output.csv', result_matrix, delimiter=',', fmt='%f')


    #观测的方差看介于(0,1]
    观测的方差 = 0.5

    迭代次数 = 10
    #之前是迭代量，现在是数量
    m = m-1
    n = m + k
    ###########################################################################

    # 使用 result_matrix 作为预测状态矩阵 X_f
    X_f = result_matrix  # 预测状态矩阵（n x N）
    y = y_observed       # 观测向量（m,）

    # 初始化观测矩阵 H
    H = np.eye(m, n)

    # 初始化协方差矩阵 P_f 和观测误差协方差矩阵 R

    R = np.eye(m) * 观测的方差      # 观测误差协方差矩阵（m x m）



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
    # 记录结束时间
    end_time = time.perf_counter()

    # 计算运行时间
    elapsed_time = end_time - start_time
    print(f"程序运行时间: {elapsed_time:.6f} 秒")
