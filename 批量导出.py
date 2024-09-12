import os
import subprocess

# 获取当前工作目录
working_dir = os.getcwd()

# 获取所有 .res 文件
res_files = [f for f in os.listdir(working_dir) if f.endswith('.res')]

for res_file in res_files:
    # 去掉文件扩展名
    base_name = os.path.splitext(res_file)[0]
    # 替换短横杠为空格
    case_name = base_name.replace('-', ' ')
    # 构建脚本文件路径
    script_path = os.path.join(working_dir, 'script.cse')
    
    # 读取脚本文件内容并修改
    with open(script_path, 'r') as file:
        lines = file.readlines()
    
    # 修改指定行
    lines[20] = f"Case Name = Case {case_name}\n"
    lines[23] = f"Export File = {base_name}.csv\n"
    
    # 写回修改后的内容
    with open(script_path, 'w') as file:
        file.writelines(lines)
    
    # 执行命令
    command = f'"C:\\Program Files\\ANSYS Inc\\v241\\CFX\\bin\\cfx5post.exe" -results {res_file} -line -session script.cse'
    process = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # 等待进程完成或超时
    try:
        process.wait(timeout=30)  # 等待最多300秒
    except subprocess.TimeoutExpired:
        print(f"Command for {res_file} timed out. Sending 'quit' command.")
        process.stdin.write(b'quit\n')
        process.stdin.flush()
        try:
            process.wait(timeout=30)  # 再等待30秒让进程有机会干净退出
        except subprocess.TimeoutExpired:
            print(f"Command for {res_file} did not exit after sending 'quit'. Terminating process.")
            process.terminate()