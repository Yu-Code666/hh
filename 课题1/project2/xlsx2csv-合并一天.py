import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl.styles.stylesheet")

import os
import pandas as pd
import re
from multiprocessing import Pool, cpu_count
from glob import glob
from tqdm import tqdm
from collections import defaultdict

# 设置进程数为 CPU 核心数 - 1（至少为1）
num_processes = max(1, cpu_count() - 1)

# 输入输出路径
input_path = "../../../data-hh/2025/haihangSalesProcess/海航系销售过程数据/"
output_path = "../../../data-hh/2025/haihangSalesProcess/海航系销售过程数据_日期版/"
os.makedirs(output_path, exist_ok=True)

# 查找所有 .xlsx 文件
xlsx_files = glob(os.path.join(input_path, "*.xlsx"))

# 按日期分组文件
def group_files_by_date(files):
    date_pattern = re.compile(r'_(\d{4}-\d{2}-\d{2})_')
    date_groups = defaultdict(list)
    
    for file in files:
        base_name = os.path.basename(file)
        match = date_pattern.search(base_name)
        if match:
            date = match.group(1)
            date_groups[date].append(file)
        else:
            # 对于不符合命名规则的文件，单独处理
            date_groups[f"无日期_{base_name}"].append(file)
    
    return date_groups

# 定义合并并转换函数
def merge_and_convert_to_csv(date_files):
    date, files = date_files
    try:
        # 合并同一天的所有文件
        all_data = []
        for file in files:
            df = pd.read_excel(file, engine='openpyxl')
            all_data.append(df)
        
        if not all_data:
            return f"⚠️ 警告：日期 {date} 没有有效数据"
        
        # 合并所有数据框
        merged_df = pd.concat(all_data, ignore_index=True)
        
        # 保存为CSV
        out_file = f"{date}.csv" if not date.startswith("无日期") else date.replace("无日期_", "") + ".csv"
        out_path = os.path.join(output_path, out_file)
        merged_df.to_csv(out_path, index=False, encoding='utf-8')
        
        return f"✅ 成功合并并转换：{date}（{len(files)}个文件）"
    except Exception as e:
        return f"❌ 失败：{date}，错误：{str(e)}"

# 主程序
if __name__ == '__main__':
    print("正在按日期分组文件...")
    date_groups = group_files_by_date(xlsx_files)
    print(f"找到 {len(date_groups)} 个不同日期的文件组")
    
    # 准备多进程处理的参数
    date_files_pairs = list(date_groups.items())
    
    # 多进程执行并配合 tqdm 显示进度条
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap_unordered(merge_and_convert_to_csv, date_files_pairs),
                            total=len(date_files_pairs),
                            desc=f"合并转换进度（进程数: {num_processes}）"))

    # # 打印所有结果
    # for res in results:
    #     print(res)
