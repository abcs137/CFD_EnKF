# 读取scaled_sample_output.txt文件，并按段落分隔
with open('scaled_sample_output.txt', 'r') as f:
    # 读取所有行
    lines = f.readlines()

# 初始化段落列表
segments = []
current_segment = []

# 遍历读取的每一行
for line in lines:
    if line.strip():  # 如果这一行不为空
        current_segment.append(line)
    else:  # 遇到空行时，说明一段结束
        if current_segment:  # 如果当前段落有内容
            segments.append(current_segment)
            current_segment = []

# 如果最后一个段落没有被添加，添加它
if current_segment:
    segments.append(current_segment)

# 读取A1-baseline-Ma0.69-4-i0.ccl文件内容
with open('A1-baseline-Ma0.69-4-i0.ccl', 'r') as f:
    ccl_lines = f.readlines()

# 替换指定行（636到652行，索引是635到651）
start_line = 635  # Python的索引从0开始
end_line = 652    # 包含到652行

# 对每个段落进行替换
for index, segment in enumerate(segments):
    # 将段落内容转换为字符串
    segment_str = ''.join(segment)
    
    # 替换指定行的内容
    ccl_lines[start_line:start_line + len(segment)] = segment

    # 生成新文件名
    new_filename = f'A1-baseline-Ma0.69-4-i0.{index + 1:03d}.ccl'
    
    # 写入新文件
    with open(new_filename, 'w') as new_file:
        new_file.writelines(ccl_lines)

print("文件生成完成！")
