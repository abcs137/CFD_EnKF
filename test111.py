import pandas as pd
import numpy as np

# 创建一个简单的 DataFrame
data = {
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
}
df = pd.DataFrame(data)

# 获取 DataFrame 的值
values = df.values

# 输出类型
print(type(values))  # 输出: <class 'numpy.ndarray'>

# 输出数组内容
print(values)
# 输出:
# [[1 4 7]
#  [2 5 8]
#  [3 6 9]]


# 获取 DataFrame 的值，推荐使用 to_numpy()
values = df.to_numpy()

# 输出类型
print(type(values))  # 输出: <class 'numpy.ndarray'>

# 输出数组内容
print(values)
# 输出:
# [[1 4 7]
#  [2 5 8]
#  [3 6 9]]
