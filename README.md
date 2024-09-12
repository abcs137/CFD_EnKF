使用EnKF来自动调整k-ω SST湍流模型的参数
对应ANSYS 2024R1 CFX5




第一步，使用“LHS生成全部参数csv.py”生成对应的参数表，输出文件“scaled_sample_output.csv”与“scaled_sample_output.txt”


第二步，准备“A1-baseline-Ma0.69-4-i0.ccl”作为模板ccl文件，以及对应的“工况表csv”，并执行“生成不同工况脚本.py”
csv必须为utf-8 ascii编码，第一行为空行，路径不能有中文


第三步，和runC.ps1和def文件放一个文件夹内，进行计算


第四步，在cfx5post.exe录制script.cse脚本 执行“批量导出.py”将res转换成csv，此处必须精细的测试运行时间并+5后填入timeout参数