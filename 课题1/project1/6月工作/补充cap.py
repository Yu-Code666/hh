import pandas as pd

# 1. 读数据
df = pd.read_csv('../../../../data-hh/2025/2025-5/航班销售结果数据_2023-01-01_2025-07-01_encrypt.csv', index_col=0)

# 2. 先对 leg_no=1/2 且 cap>0 的航班，按 (flt_date, flt_no) 求 cap 的平均值
mean_caps = (
    df.loc[(df.leg_no.isin([1,2])) & (df.cap>0)]
      .groupby(['flt_date','flt_no'], as_index=False)['cap']
      .mean()
      .rename(columns={'cap':'mean_cap'})
)

# 3. 把这个 mean_cap 按 (flt_date,flt_no) 左合并回原 df
df = df.merge(mean_caps, on=['flt_date','flt_no'], how='left')

# 4. 只更新 leg_no=3 且 原 cap=0 的行
mask3 = (df.leg_no==3) & (df.cap==0)
df.loc[mask3, 'cap'] = df.loc[mask3, 'mean_cap'].fillna(0)

# 5. 删除临时列
df.drop(columns='mean_cap', inplace=True)

# 6. 保存
df.to_csv('../../../../data-hh/2025/2025-5/result/cap_补全后.csv')
