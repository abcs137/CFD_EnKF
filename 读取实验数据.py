import pandas as pd
import numpy as np
def read_dat_file(file_path):
    # 初始化空数据列表
    headers = []
    values = []
    coordinates = []

    # 读取文件内容
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 处理标题行
    headers = lines[0].strip().split()
    
    # 处理数据行
    data_lines = lines[1:]
    for i in range(0, len(data_lines), 3):
        if i + 1 < len(data_lines):
            # 读取参数值
            value_line = data_lines[i].strip().split()
            # 读取坐标值
            coord_line = data_lines[i + 1].strip().split()

            # 确保数据长度匹配
            if len(value_line) == len(headers) and len(coord_line) == 3:
                values.append(list(map(float, value_line)))
                coordinates.append(list(map(float, coord_line)))
            else:
                print("Data length mismatch, skipping this block.")

    # 创建 DataFrame
    df = pd.DataFrame(values, columns=headers,dtype=np.float64)
    df[['x', 'y', 'z']] = pd.DataFrame(coordinates, columns=['x', 'y', 'z'],dtype=np.float64)

    return df

def get_parameter(df, parameter_name, coordinate='y'):
    """
    获取指定参数和指定坐标的数据
    :param df: Pandas DataFrame，包含所有数据
    :param parameter_name: 需要查询的参数名称（如 'Ps_ps'）
    :param coordinate: 需要查询的坐标名称，'x'、'y' 或 'z'
    :return: 返回指定参数和指定坐标的数据列表
    """
    if coordinate not in ['x', 'y', 'z']:
        raise ValueError("Coordinate must be one of 'x', 'y', or 'z'")
    
    if parameter_name not in df.columns:
        raise ValueError(f"Parameter {parameter_name} not found in DataFrame columns")
    
    return df.loc[:, [parameter_name, coordinate]]

# 使用示例
file_path = 'i0_MA=0.69_压力面.dat'  # 替换为你的文件路径
df = read_dat_file(file_path)

# 读取 'Ps_ps' 参数和 'y' 坐标数据
#result = get_parameter(df, 'Ps_ps', 'y')
