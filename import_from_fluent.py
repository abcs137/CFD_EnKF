import pandas as pd
import numpy as np
from PIL import Image

from PressureEstimator import PressureEstimator


def preprocess_csv(csv_file):
    # 读取CSV文件
    df = pd.read_csv(csv_file, delim_whitespace=False,on_bad_lines='skip', header=0, index_col=False,dtype={'cellnumber': int, 'x-coordinate': float, 'y-coordinate': float, 'pressure': float})
    
    # 过滤缺少列、多余列和异常值的行
    df = df.dropna()  # 去除NaN值
    # 可以根据实际情况添加更多的过滤条件，比如对x、y的范围进行限制
    
    return df


def linear_estimate_range(data):
    # 线性估测x和y的范围
    min_x, max_x = data['x-coordinate'].min(), data['x-coordinate'].max()
    min_y, max_y = data['y-coordinate'].min(), data['y-coordinate'].max()
    return min_x, max_x, min_y, max_y


def log_estimate_pressure(data):
    # 对数估计value的值
    data['log_value'] = data['pressure']
    return data






def map_value_to_color(value, min_value, max_value):
    """
    将数值映射到颜色上。

    参数：
    value (float): 要映射的数值。
    min_value (float): 数值的最小值。
    max_value (float): 数值的最大值。

    返回：
    tuple: RGB 颜色值。
    """
    # 简单线性映射
    ratio = (value - min_value) / (max_value - min_value)
    blue = int(255 * (1 - ratio))
    red = int(255 * ratio)
    return (red, 0, blue)


def render_bitmap(data, min_x, max_x, min_y, max_y, output_size, k=5):
    """
    渲染一幅位图图像，表示指定区域内的压力分布。

    参数：
    data (pd.DataFrame): 包含数据的 DataFrame，必须包含 'x-coordinate', 'y-coordinate', 'log_value' 列。
    min_x (float): 区域的最小 x 坐标。
    max_x (float): 区域的最大 x 坐标。
    min_y (float): 区域的最小 y 坐标。
    max_y (float): 区域的最大 y 坐标。
    output_size (tuple): 输出图像的大小（宽，高）。
    k (int): 最近邻的点的数量。

    返回：
    PIL.Image: 渲染的图像。
    """
    # 创建一个空白的图像
    img = Image.new('RGB', output_size, color='white')
    pixels = img.load()

    # 获取 value 的最小值和最大值
    min_value, max_value = data['log_value'].min(), data['log_value'].max()

    # 初始化 PressureEstimator
    estimator = PressureEstimator(data, k=k)

    # 遍历图像的每个像素点，使用 Ball Tree 估计其 log_value 并赋予颜色
    for i in range(output_size[0]):
        for j in range(output_size[1]):
            # 将图像坐标转换为实际坐标
            x = min_x + (i / (output_size[0] - 1)) * (max_x - min_x)
            y = min_y + (j / (output_size[1] - 1)) * (max_y - min_y)

            # 估计目标点的 log_value
            estimated_value = estimator.estimate_pressure(x, y)

            # 将估计的压力值转换为颜色
            color = map_value_to_color(estimated_value, min_value, max_value)
            pixels[i, j] = color

    return img





def main(csv_file, output_size):
    # 预处理CSV文件
    data = preprocess_csv(csv_file)

    #columns = data.columns
    #print(columns)

    #print(data)

    # 线性估测x和y的范围
    min_x, max_x, min_y, max_y = linear_estimate_range(data)
    
    # 对数估计value的值
    data = log_estimate_pressure(data)
    data=data.dropna()

    # 渲染位图
    img = render_bitmap(data, min_x, max_x, min_y, max_y, output_size)
    
    # 显示位图或保存位图到文件
    img.show()
    img.save('output_bitmap.png')

# 示例用法
if __name__ == "__main__":
    csv_file = 'raoliu.csv'  # 替换为你的CSV文件路径
    output_size = (1000, 600)  # 输出图像的尺寸
    main(csv_file, output_size)
