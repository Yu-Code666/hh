import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl.styles.stylesheet")

import os
import pandas as pd
from multiprocessing import Pool, cpu_count
from glob import glob
from tqdm import tqdm

# 设置进程数为 CPU 核心数 - 1（至少为1）
num_processes = max(1, cpu_count() - 1)

# 输入输出路径
input_path = "../../../data-hh/2025/haihangSalesProcess/海航系销售过程数据/"
output_path = "../../../data-hh/2025/haihangSalesProcess/海航系销售过程数据_日期版/"
os.makedirs(output_path, exist_ok=True)

# 查找所有 .xlsx 文件
xlsx_files = glob(os.path.join(input_path, "*.xlsx"))

# 定义转换函数（注意参数必须是可 picklable 的，即通常是字符串）
def convert_xlsx_to_csv(file_path):
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        base_name = os.path.basename(file_path).replace('.xlsx', '.csv')
        out_path = os.path.join(output_path, base_name)
        df.to_csv(out_path, index=False, encoding='utf-8')
        return f"✅ 成功转换：{base_name}"
    except Exception as e:
        return f"❌ 失败：{file_path}，错误：{str(e)}"

# 多进程执行并配合 tqdm 显示进度条
if __name__ == '__main__':
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap_unordered(convert_xlsx_to_csv, xlsx_files),
                            total=len(xlsx_files),
                            desc=f"转换进度（进程数: {num_processes}）"))

    # 打印所有结果
    for res in results:
        print(res)
