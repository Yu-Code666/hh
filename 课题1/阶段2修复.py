#!/usr/bin/env python
# coding: utf-8

# ## 加载模型和编码器

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


def load_model_and_encoders():
    """
    加载所有模型、编码器和标准化器。
    
    返回：
    - model: XGBoost模型
    - label_encoders: 分类特征的标签编码器字典
    - scaler_x: 特征标准化器
    - scaler_y: 目标值标准化器
    - city_map: 城市频率编码映射
    - city_labels: 城市标签编码映射
    - city_embeddings: 城市嵌入编码字典
    """
    import xgboost as xgb
    import joblib
    import json
    import pandas as pd
    
    # 1. 加载XGBoost模型
    model = xgb.XGBRegressor()
    model.load_model("../../data-hh/my/模型文件/频率编码/归一化_xgboost_model_1000.json")
    
    # 2. 加载标签编码器
    categorical_columns = ['flt_no', 'bd_type', 'aircraft']
    label_encoders = {}
    for col in categorical_columns:
        label_encoders[col] = joblib.load(f"../../data-hh/my/encoder/{col}_encoder_all.pkl")
    
    # 3. 加载标准化器
    scaler_x = joblib.load('../../data-hh/my/encoder/standard_scaler_x.pkl')
    scaler_y = joblib.load('../../data-hh/my/encoder/standard_scaler_y.pkl')
    
    # 4. 加载城市频率编码映射
    with open('../../data-hh/my/encoder/city_map_频率编码.json', 'r') as f:
        city_map = json.load(f)
    
    # 5. 加载城市标签编码映射
    with open('../../data-hh/my/encoder/city_labels_航班频率加权图标签.json', 'r') as f:
        city_labels = json.load(f)
    
    # 6. 加载城市嵌入编码（修改了文件路径）
    with open('../../data-hh/my/encoder/城市嵌入编码_航班频率加权图.json', 'r') as f:
        city_embeddings = json.load(f)
    
    print("所有模型和编码器加载完成")
    
    return model, label_encoders, scaler_x, scaler_y, city_map, city_labels, city_embeddings


# ## 数据预处理

# In[3]:


def preprocess_data(flt_date, flt_no, bd_type, dep_time, ab_cap, bc_cap, ac_cap, aircraft, 
                   ab_duration, bc_duration, ac_duration, 
                   a, b, c, ab_price, bc_price, ac_price, 
                   label_encoders, scaler_x, city_map, city_labels, city_embeddings):
    """
    对输入数据进行预处理，生成模型输入。
    """
    import pandas as pd
    import numpy as np
    from datetime import datetime
    
    # 创建三条记录的DataFrame（对应三个leg_no）
    data = []
    
    # 处理日期和时间
    flt_datetime = datetime.strptime(f"{flt_date} {dep_time}", "%Y-%m-%d %H:%M:%S")
    year = flt_datetime.year
    month = flt_datetime.month
    day = flt_datetime.day
    weekday = flt_datetime.weekday()
    hour = flt_datetime.hour
    minute = flt_datetime.minute
    second = flt_datetime.second
    
    # 处理unit_price, duration和cap
    leg_info = {
        1: {'price': ab_price, 'duration': ab_duration, 'from': a, 'to': b, 'cap': ab_cap},
        2: {'price': bc_price, 'duration': bc_duration, 'from': b, 'to': c, 'cap': bc_cap},
        3: {'price': ac_price, 'duration': ac_duration, 'from': a, 'to': c, 'cap': ac_cap}
    }
    
    # 为每个leg_no创建一条记录
    for leg_no in [1, 2, 3]:
        record = {
            # 基础特征编码
            'flt_no': label_encoders['flt_no'].transform([flt_no])[0],
            'bd_type': label_encoders['bd_type'].transform([bd_type])[0],
            'cap': float(leg_info[leg_no]['cap']) if not pd.isna(leg_info[leg_no]['cap']) else 0.0,
            'aircraft': label_encoders['aircraft'].transform([aircraft])[0],
            'legs': 3,
            'leg_no': leg_no,
            'duration': float(leg_info[leg_no]['duration']) if not pd.isna(leg_info[leg_no]['duration']) else 0.0,
            
            # 城市频率编码
            'a': city_map.get(a, 0),
            'b': city_map.get(b, 0),
            'c': city_map.get(c, 0),
            'from': city_map.get(leg_info[leg_no]['from'], 0),
            'to': city_map.get(leg_info[leg_no]['to'], 0),
            
            # 时间特征
            'year': year,
            'month': month,
            'day': day,
            'weekday': weekday,
            'hour': hour,
            'minute': minute,
            'second': second,
            
            # 城市标签
            'a_label': city_labels.get(a, -1),
            'b_label': city_labels.get(b, -1),
            'c_label': city_labels.get(c, -1),
            'from_label': city_labels.get(leg_info[leg_no]['from'], -1),
            'to_label': city_labels.get(leg_info[leg_no]['to'], -1),
            
            # 票价
            'unit_price': float(leg_info[leg_no]['price']) if not pd.isna(leg_info[leg_no]['price']) else 0.0
        }
        
        # 添加城市嵌入编码
        city_values = {
            'a': a,
            'b': b,
            'c': c,
            'from': leg_info[leg_no]['from'],
            'to': leg_info[leg_no]['to']
        }
        
        for prefix, city_id in city_values.items():
            if city_id in city_embeddings:
                record[f'{prefix}_embedding_1'] = city_embeddings[city_id][0]
                record[f'{prefix}_embedding_2'] = city_embeddings[city_id][1]
            else:
                record[f'{prefix}_embedding_1'] = 0.0
                record[f'{prefix}_embedding_2'] = 0.0
        
        data.append(record)
    
    # 转换为DataFrame
    df = pd.DataFrame(data)
    
    # 确保列的顺序与训练时一致
    expected_columns = ['flt_no', 'bd_type', 'cap', 'aircraft', 'legs', 'leg_no', 'duration', 
                       'a', 'b', 'c', 'year', 'month', 'day', 'weekday', 'hour', 'minute', 'second', 
                       'from', 'to', 'unit_price', 'a_label', 'b_label', 'c_label', 'from_label', 'to_label',
                       'a_embedding_1', 'a_embedding_2', 'b_embedding_1', 'b_embedding_2', 
                       'c_embedding_1', 'c_embedding_2', 'from_embedding_1', 'from_embedding_2', 
                       'to_embedding_1', 'to_embedding_2']
    df = df[expected_columns]
    
    # 使用scaler进行标准化
    scaled_data = scaler_x.transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
    
    return scaled_df


# ## 客流预测

# In[4]:


import multiprocessing

def predict_pax(processed_data, model, scaler_y):
    """
    使用模型进行客流预测，并反向标准化。
    
    参数:
    - processed_data: DataFrame，经过预处理的数据
    - model: XGBoost模型
    - scaler_y: 目标值的标准化器
    """
    # 获取CPU核心数
    cpu_cores = multiprocessing.cpu_count()
    # 设置线程数为CPU核心数-1，保留一个核心处理其他任务
    optimal_threads = max(1, cpu_cores - 1)
    
    # 设置线程数
    model.set_params(n_jobs=optimal_threads)
    
    # 使用模型进行预测
    scaled_predictions = model.predict(processed_data)
    scaled_predictions_2d = scaled_predictions.reshape(-1, 1)
    predictions = scaler_y.inverse_transform(scaled_predictions_2d)
    
    result = {
        'AB_PAX': max(0, round(float(predictions[0].item()))),
        'BC_PAX': max(0, round(float(predictions[1].item()))),
        'AC_PAX': max(0, round(float(predictions[2].item())))
    }
    
    return result


# ## 座位分配

# In[5]:


def allocate_seats(pax_predictions, ab_price, bc_price, ac_price, cap):
    """
    根据收益最大化策略，分配航段 AB, BC, AC 的客舱容量。
    
    输入:
    - pax_predictions: 字典，包含 AB, BC, AC 的乘客数预测值，例如 {'AB_PAX': 120, 'BC_PAX': 110, 'AC_PAX': 130}。
    - ab_price: float，航段 AB 的票价。
    - bc_price: float，航段 BC 的票价。
    - ac_price: float，航段 AC 的票价。
    - cap: int，总客舱容量。

    输出:
    - tuple: (allocation, revenue)
        - allocation: 字典，包含分配给 AB, BC, AC 的舱位数量
        - revenue: float，预计总收入
    """
    # 获取航段预测乘客数
    ab_pax = round(pax_predictions['AB_PAX'])
    bc_pax = round(pax_predictions['BC_PAX'])
    ac_pax = round(pax_predictions['AC_PAX'])

    # 短途类（S 类）的最大容量
    s_max = max(ab_pax, bc_pax)
    s_min = min(ab_pax, bc_pax)

    # 分配结果初始化
    allocation = {'AB': 0, 'BC': 0, 'AC': 0}

    # 判断是否满座
    if s_max + ac_pax <= cap:
        # 未满座：按比例分配 S 类和 L 类
        total_pax = s_max + ac_pax
        s_cap = round(cap * (s_max / total_pax))  # S 类分配的容量
        l_cap = cap - s_cap  # L 类分配的容量

        # S 类：直接分配 s_cap（AB 和 BC 同时销售）
        allocation['AB'] = s_cap
        allocation['BC'] = s_cap

        # L 类：分配剩余容量
        allocation['AC'] = l_cap
    else:
        # 已满座：根据收益优先级分配
        s_revenue = ab_price + bc_price
        l_revenue = ac_price

        if s_revenue > l_revenue:
            # 优先分配给 S 类的 min(AB_PAX, BC_PAX)
            s_cap = min(s_min, cap)
            allocation['AB'] = s_cap
            allocation['BC'] = s_cap

            # 剩余容量优先分配给 L 类
            remaining_cap = cap - s_cap
            allocation['AC'] = min(ac_pax, remaining_cap)

            # 如果 L 类分配后还有剩余容量,直接分配给 S 类
            remaining_after_l = remaining_cap - allocation['AC']
            if remaining_after_l > 0:
                allocation['AB'] += remaining_after_l
                allocation['BC'] += remaining_after_l
        else:
            # 优先分配给 L 类
            allocation['AC'] = min(ac_pax, cap)  # L 类尽量满足 AC 的预测乘客数
            remaining_cap = cap - allocation['AC']  # 剩余容量分配给 S 类
            allocation['AB'] = remaining_cap
            allocation['BC'] = remaining_cap

    # 计算预期收入
    revenue = (min(ab_pax, allocation['AB']) * ab_price + 
              min(bc_pax, allocation['BC']) * bc_price + 
              min(ac_pax, allocation['AC']) * ac_price)

    return allocation, revenue


# ## 测试数据

# In[6]:


# # 测试数据
# test_data = {
#     'flt_date': '2023-1-1',
#     'flt_no': 'P6p42wyZFEI=',  # 从图片中第一行获取
#     'bd_type': '窄体',
#     'dep_time': '08:50:00',  # 从图片中第一行获取
#     'ab_cap': 132.0,  # 从图片中获取
#     'bc_cap': 132.0,  # 从图片中获取
#     'ac_cap': np.nan,  # 从图片中获取
#     'aircraft': '319',  # 从图片中获取
#     'ab_duration': 2.38,    # 从图片中第三行leg_no=1的duration获取
#     'bc_duration': 2.68,    # 从图片中第二行leg_no=2的duration获取
#     'ac_duration': 0.0,     # 从图片中第一行leg_no=3的duration获取
#     'a': 'tKjndGSl9NQ=',      # 从图片中航班号的第一部分获取
#     'b': 'KETr2NAmAHE=',   # 从图片中航班号的中间部分获取
#     'c': '5t+HPO9Mu/w=',      # 从图片中航班号的最后部分获取
#     'pax_ab':50, 
#     'pax_bc':50, 
#     'pax_ac':50,
#     'ab_price': 1048,    # 这些价格可以保持不变
#     'bc_price': 1259,    # 因为图片中没有显示价格信息
#     'ac_price': 1682,
#     # 'ab_price': 1048,    # 这些价格可以保持不变
#     # 'bc_price': 2000,    # 因为图片中没有显示价格信息
#     # 'ac_price': 2000
#     'tkt_rev_ab':5000,
#     'tkt_rev_bc':5000, 
#     'tkt_rev_ac':5000
# }


# In[7]:


# # 加载所有模型和编码器
# model, label_encoders, scaler_x, scaler_y, city_map, city_labels, city_embeddings = load_model_and_encoders()


# In[8]:


# import numpy as np

# # 直接传递所有参数，而不是使用字典解包
# processed_data = preprocess_data(
#     flt_date=test_data['flt_date'],
#     flt_no=test_data['flt_no'],
#     bd_type=test_data['bd_type'],
#     dep_time=test_data['dep_time'],
#     ab_cap=test_data['ab_cap'],
#     bc_cap=test_data['bc_cap'],
#     ac_cap=test_data['ac_cap'],
#     aircraft=test_data['aircraft'],
#     ab_duration=test_data['ab_duration'],
#     bc_duration=test_data['bc_duration'],
#     ac_duration=test_data['ac_duration'],
#     a=test_data['a'],
#     b=test_data['b'],
#     c=test_data['c'],
#     ab_price=test_data['ab_price'],
#     bc_price=test_data['bc_price'],
#     ac_price=test_data['ac_price'],
#     label_encoders=label_encoders,
#     scaler_x=scaler_x,
#     city_map=city_map,
#     city_labels=city_labels,
#     city_embeddings=city_embeddings
# )


# In[9]:


# # 进行预测
# predictions = predict_pax(processed_data, model, scaler_y)
# print("预测结果:", predictions)


# In[10]:


# # 从test_data中选择不为0或nan的cap值
# caps = [
#     test_data['ab_cap'],
#     test_data['bc_cap'],
#     test_data['ac_cap']
# ]
# cap = next(c for c in caps if c == c and c != 0)  # 获取第一个不为nan且不为0的值
# print(cap)
# # 调用座位分配函数



# In[11]:


# allocation, revenue = allocate_seats(
#     pax_predictions=predictions,
#     ab_price=test_data['ab_price'],
#     bc_price=test_data['bc_price'],
#     ac_price=test_data['ac_price'],
#     cap=cap
# )

# # 打印分配结果
# print("座位分配结果:")
# print(f"AB航段分配座位: {allocation['AB']}")
# print(f"BC航段分配座位: {allocation['BC']}")
# print(f"AC航段分配座位: {allocation['AC']}")
# print(f"预计总收入: {revenue}")


# In[12]:


# #     'ab_price': 1048,    # 这些价格可以保持不变
# #     'bc_price': 1259,    # 因为图片中没有显示价格信息
# #     'ac_price': 1682

# # 预测结果: {'AB_PAX': 116, 'BC_PAX': 61, 'AC_PAX': 19}
# 1048*113+1259*61+19*1682


# ## 票价预测

# In[13]:


def optimize_prices(test_data, model, label_encoders, scaler_x, scaler_y, city_map, city_labels, city_embeddings, price_stats):
    """
    优化票价以获得最大收入。
    
    参数:
    - test_data: dict, 包含航班基本信息的测试数据
    - model, label_encoders, scaler_x, scaler_y, city_map, city_labels, city_embeddings: 模型和编码器
    - price_stats: DataFrame, 包含航线价格统计信息
    
    返回:
    - dict: 包含最优票价组合、预测客流、座位分配和预期收入
    """
    # 输出原始票价收入
    if all(key in test_data for key in ['tkt_rev_ab', 'tkt_rev_bc', 'tkt_rev_ac']):
        total_original_revenue = test_data['tkt_rev_ab'] + test_data['tkt_rev_bc'] + test_data['tkt_rev_ac']
    
    # 获取各航段的价格统计
    ab_mean, ab_std = get_price_stats(test_data['a'], test_data['b'], price_stats)
    bc_mean, bc_std = get_price_stats(test_data['b'], test_data['c'], price_stats)
    ac_mean, ac_std = get_price_stats(test_data['a'], test_data['c'], price_stats)
    
    # 生成候选价格
    ab_prices = get_candidate_prices(ab_mean, ab_std)
    bc_prices = get_candidate_prices(bc_mean, bc_std)
    ac_prices = get_candidate_prices(ac_mean, ac_std)
    
    # 从test_data中获取cap值
    caps = [test_data['ab_cap'], test_data['bc_cap'], test_data['ac_cap']]
    cap = next(c for c in caps if c == c and c != 0)  # 获取第一个不为nan且不为0的值
    
    # 存储最优结果
    best_revenue = 0
    best_allocation = None
    best_prices = None
    best_predictions = None
    
    # 遍历所有价格组合
    for ac_p in ac_prices:
        valid_ab_prices = [p for p in ab_prices if p < ac_p]
        valid_bc_prices = [p for p in bc_prices if p < ac_p]
        
        if not valid_ab_prices or not valid_bc_prices:
            continue
            
        for ab_p in valid_ab_prices:
            for bc_p in valid_bc_prices:
                # 预处理数据
                processed_data = preprocess_data(
                    flt_date=test_data['flt_date'],
                    flt_no=test_data['flt_no'],
                    bd_type=test_data['bd_type'],
                    dep_time=test_data['dep_time'],
                    ab_cap=test_data['ab_cap'],
                    bc_cap=test_data['bc_cap'],
                    ac_cap=test_data['ac_cap'],
                    aircraft=test_data['aircraft'],
                    ab_duration=test_data['ab_duration'],
                    bc_duration=test_data['bc_duration'],
                    ac_duration=test_data['ac_duration'],
                    a=test_data['a'],
                    b=test_data['b'],
                    c=test_data['c'],
                    ab_price=ab_p,
                    bc_price=bc_p,
                    ac_price=ac_p,
                    label_encoders=label_encoders,
                    scaler_x=scaler_x,
                    city_map=city_map,
                    city_labels=city_labels,
                    city_embeddings=city_embeddings
                )
                
                # 预测客流
                curr_predictions = predict_pax(processed_data, model, scaler_y)
                
                # 分配座位
                curr_allocation, curr_revenue = allocate_seats(
                    pax_predictions=curr_predictions,
                    ab_price=ab_p,
                    bc_price=bc_p,
                    ac_price=ac_p,
                    cap=cap
                )
                
                # 更新最优结果
                if curr_revenue > best_revenue:
                    best_revenue = curr_revenue
                    best_allocation = curr_allocation
                    best_prices = {'AB': ab_p, 'BC': bc_p, 'AC': ac_p}
                    best_predictions = curr_predictions
    
    # 计算收入提升
    if all(key in test_data for key in ['tkt_rev_ab', 'tkt_rev_bc', 'tkt_rev_ac']):
        total_original_revenue = test_data['tkt_rev_ab'] + test_data['tkt_rev_bc'] + test_data['tkt_rev_ac']
        revenue_improvement = (best_revenue - total_original_revenue) / total_original_revenue * 100
    
    # 返回最优结果
    # 返回最优结果
    return {
        'optimal_prices': best_prices,
        'predicted_pax': best_predictions,
        'seat_allocation': best_allocation,
        'expected_revenue': best_revenue,
        'original_revenue': total_original_revenue if 'tkt_rev_ab' in test_data else None,
        'revenue_improvement': revenue_improvement if 'tkt_rev_ab' in test_data else None,
        'revenue_improvement_amount': (best_revenue - total_original_revenue) if 'tkt_rev_ab' in test_data else None
    }


# In[14]:


# # 测试数据
# test_data = {
#     'flt_date': '2023-1-1',
#     'flt_no': 'P6p42wyZFEI=',  # 从图片中第一行获取
#     'bd_type': '窄体',
#     'dep_time': '08:50:00',  # 从图片中第一行获取
#     'ab_cap': 132.0,  # 从图片中获取
#     'bc_cap': 132.0,  # 从图片中获取
#     'ac_cap': np.nan,  # 从图片中获取
#     'aircraft': '319',  # 从图片中获取
#     'ab_duration': 2.38,    # 从图片中第三行leg_no=1的duration获取
#     'bc_duration': 2.68,    # 从图片中第二行leg_no=2的duration获取
#     'ac_duration': 0.0,     # 从图片中第一行leg_no=3的duration获取
#     'a': 'tKjndGSl9NQ=',      # 从图片中航班号的第一部分获取
#     'b': 'KETr2NAmAHE=',   # 从图片中航班号的中间部分获取
#     'c': '5t+HPO9Mu/w=',      # 从图片中航班号的最后部分获取
#     'ab_price': 1048,    # 这些价格可以保持不变
#     'bc_price': 1259,    # 因为图片中没有显示价格信息
#     'ac_price': 1682,
#     'tkt_rev_ab':5000,
#     'tkt_rev_bc':5000, 
#     'tkt_rev_ac':5000
# }

# # 运行优化
# result = optimize_prices(
#     test_data,
#     model,
#     label_encoders,
#     scaler_x,
#     scaler_y,
#     city_map,
#     city_labels,
#     city_embeddings,
#     price_stats
# )

# # 打印结果
# print("最优票价组合:")
# print(f"AB航段票价: {result['optimal_prices']['AB']}")
# print(f"BC航段票价: {result['optimal_prices']['BC']}")
# print(f"AC航段票价: {result['optimal_prices']['AC']}")

# print("\n对应的客流预测:")
# print(f"AB航段预测客流: {result['predicted_pax']['AB_PAX']}")
# print(f"BC航段预测客流: {result['predicted_pax']['BC_PAX']}")
# print(f"AC航段预测客流: {result['predicted_pax']['AC_PAX']}")

# print("\n最优座位分配:")
# print(f"AB航段分配座位: {result['seat_allocation']['AB']}")
# print(f"BC航段分配座位: {result['seat_allocation']['BC']}")
# print(f"AC航段分配座位: {result['seat_allocation']['AC']}")
# print(f"最大预计总收入: {result['expected_revenue']}")


# In[15]:


# result


# In[16]:


# # 测试数据
# test_data = {
#     'flt_date': '2023-1-1',
#     'flt_no': 'P6p42wyZFEI=',  # 从图片中第一行获取
#     'bd_type': '窄体',
#     'dep_time': '08:50:00',  # 从图片中第一行获取
#     'ab_cap': 132.0,  # 从图片中获取
#     'bc_cap': 132.0,  # 从图片中获取
#     'ac_cap': np.nan,  # 从图片中获取
#     'aircraft': '319',  # 从图片中获取
#     'ab_duration': 2.38,    # 从图片中第三行leg_no=1的duration获取
#     'bc_duration': 2.68,    # 从图片中第二行leg_no=2的duration获取
#     'ac_duration': 0.0,     # 从图片中第一行leg_no=3的duration获取
#     'a': 'tKjndGSl9NQ=',      # 从图片中航班号的第一部分获取
#     'b': 'KETr2NAmAHE=',   # 从图片中航班号的中间部分获取
#     'c': '5t+HPO9Mu/w=',      # 从图片中航班号的最后部分获取
#     'ab_price': 1048,    # 这些价格可以保持不变
#     'bc_price': 1259,    # 因为图片中没有显示价格信息
#     'ac_price': 1682,
#     # 'ab_price': 1048,    # 这些价格可以保持不变
#     # 'bc_price': 2000,    # 因为图片中没有显示价格信息
#     # 'ac_price': 2000
#     'tkt_rev_ab':5000,
#     'tkt_rev_bc':5000, 
#     'tkt_rev_ac':5000
# }

# # 读取价格统计信息
# price_stats = pd.read_csv('../../data-hh/my/hh_result/route_price_stats.csv')

# 获取城市间价格的均值和标准差
def get_price_stats(from_city, to_city,price_stats):
    row = price_stats[(price_stats['from'] == from_city) & (price_stats['to'] == to_city)]
    if len(row) > 0:
        return row['平均价格'].values[0], row['价格标准差'].values[0]
    return None, None



# In[17]:


# # 获取A-B, B-C, A-C航线的价格统计
# ab_mean, ab_std = get_price_stats(test_data['a'], test_data['b'])  # 使用test_data中的城市代码
# bc_mean, bc_std = get_price_stats(test_data['b'], test_data['c'])
# ac_mean, ac_std = get_price_stats(test_data['a'], test_data['c'])

# 使用3sigma原则生成候选价格
def get_candidate_prices(mean, std):
    if mean is None or std is None:
        return []
    return [
        round(mean - 3*std),
        round(mean - 2*std), 
        round(mean - std),
        round(mean),
        round(mean + std),
        round(mean + 2*std),
        round(mean + 3*std)
    ]

# ab_prices = get_candidate_prices(ab_mean, ab_std)
# bc_prices = get_candidate_prices(bc_mean, bc_std)
# ac_prices = get_candidate_prices(ac_mean, ac_std)

# # 存储最优结果
# best_revenue = 0
# best_allocation = None
# best_prices = None
# best_predictions = None

# # 遍历所有价格组合，确保AB和BC的价格都低于AC
# for ac_p in ac_prices:
#     # 只选择小于ac_p的ab和bc价格
#     valid_ab_prices = [p for p in ab_prices if p < ac_p]
#     valid_bc_prices = [p for p in bc_prices if p < ac_p]
    
#     # 如果没有有效的价格组合，跳过当前AC价格
#     if not valid_ab_prices or not valid_bc_prices:
#         continue
        
#     for ab_p in valid_ab_prices:
#         for bc_p in valid_bc_prices:
#             # 更新测试数据的价格
#             test_data_copy = test_data.copy()
#             test_data_copy['ab_price'] = ab_p
#             test_data_copy['bc_price'] = bc_p 
#             test_data_copy['ac_price'] = ac_p
            
#                         # 预处理数据
#             processed_data = preprocess_data(
#                 flt_date=test_data_copy['flt_date'],
#                 flt_no=test_data_copy['flt_no'],
#                 bd_type=test_data_copy['bd_type'],
#                 dep_time=test_data_copy['dep_time'],
#                 ab_cap=test_data_copy['ab_cap'],
#                 bc_cap=test_data_copy['bc_cap'],
#                 ac_cap=test_data_copy['ac_cap'],
#                 aircraft=test_data_copy['aircraft'],
#                 ab_duration=test_data_copy['ab_duration'],
#                 bc_duration=test_data_copy['bc_duration'],
#                 ac_duration=test_data_copy['ac_duration'],
#                 a=test_data_copy['a'],
#                 b=test_data_copy['b'],
#                 c=test_data_copy['c'],
#                 ab_price=ab_p,
#                 bc_price=bc_p,
#                 ac_price=ac_p,
#                 label_encoders=label_encoders,
#                 scaler_x=scaler_x,
#                 city_map=city_map,
#                 city_labels=city_labels,
#                 city_embeddings=city_embeddings
#             )
#             # 预测客流
#             curr_predictions = predict_pax(processed_data, model, scaler_y)
            
#             # 分配座位
#             curr_allocation, curr_revenue = allocate_seats(
#                 pax_predictions=curr_predictions,
#                 ab_price=ab_p,
#                 bc_price=bc_p,
#                 ac_price=ac_p,
#                 cap=cap
#             )
            
#             # 更新最优结果
#             if curr_revenue > best_revenue:
#                 best_revenue = curr_revenue
#                 best_allocation = curr_allocation
#                 best_prices = {'AB': ab_p, 'BC': bc_p, 'AC': ac_p}
#                 best_predictions = curr_predictions

# print("最优票价组合:")
# print(f"AB航段票价: {best_prices['AB']}")
# print(f"BC航段票价: {best_prices['BC']}")
# print(f"AC航段票价: {best_prices['AC']}")

# print("\n对应的客流预测:")
# print(f"AB航段预测客流: {best_predictions['AB_PAX']}")
# print(f"BC航段预测客流: {best_predictions['BC_PAX']}")
# print(f"AC航段预测客流: {best_predictions['AC_PAX']}")

# print("\n最优座位分配:")
# print(f"AB航段分配座位: {best_allocation['AB']}")
# print(f"BC航段分配座位: {best_allocation['BC']}")
# print(f"AC航段分配座位: {best_allocation['AC']}")
# print(f"最大预计总收入: {best_revenue}")


# ## 批量测试

# In[18]:


# # 加载测试数据
# test_df = pd.read_csv('../../data-hh/my/hh_result/hh_result_2024_6_merged.csv')

# # 显示数据基本信息
# print("测试数据形状:", test_df.shape)
# print("\n数据列名:", test_df.columns.tolist())
# print("\n前5行数据:")
# pd.set_option('display.max_columns', None)  # 显示所有列
# print(test_df.head())

# # 显示基本统计信息
# print("\n数值列的基本统计信息:")
# print(test_df.describe())

# # 检查是否有缺失值
# print("\n各列的缺失值数量:")
# print(test_df.isnull().sum())


# ### 从DataFrame的一行创建test_data字典

# In[19]:


def create_test_data(row):
    """
    从DataFrame的一行创建test_data字典
    
    参数:
    - row: DataFrame的一行数据
    
    返回:
    - dict: 包含所需字段的test_data字典
    """
    return {
        'flt_date': row['flt_date'],
        'flt_no': row['flt_no'],
        'bd_type': row['bd_type'],
        'dep_time': row['dep_time_ab'],
        'ab_cap': row['cap_ab'],
        'bc_cap': row['cap_bc'],
        'ac_cap': row['cap_ac'],
        'aircraft': row['aircraft'],
        'ab_duration': row['duration_ab'],
        'bc_duration': row['duration_bc'],
        'ac_duration': row['duration_ac'],
        'a': row['a'],
        'b': row['b'],
        'c': row['c'],
        'ab_price': row['unit_price_ab'],
        'bc_price': row['unit_price_bc'],
        'ac_price': row['unit_price_ac'],
        'tkt_rev_ab': row['tkt_rev_ab'],
        'tkt_rev_bc': row['tkt_rev_bc'],
        'tkt_rev_ac': row['tkt_rev_ac']
    }


# ### 批量优化票价

# In[20]:


def batch_optimize_prices(test_df, model, label_encoders, scaler_x, scaler_y, city_map, city_labels, city_embeddings, price_stats):
    """
    批量优化票价
    
    参数:
    - test_df: DataFrame, 测试数据集
    - model, label_encoders, scaler_x, scaler_y, city_map, city_labels, city_embeddings: 模型和必要的编码器
    - price_stats: DataFrame, 包含航线价格统计信息
    
    返回:
    - DataFrame: 包含原始数据和优化结果的数据框
    """
    # 存储结果的列表
    results = []
    
    # 显示进度条
    from tqdm import tqdm
    
    # 对每一行数据进行处理
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="优化票价"):
        # 创建test_data
        test_data = create_test_data(row)


        # 运行优化
        result = optimize_prices(
            test_data,
            model,
            label_encoders,
            scaler_x,
            scaler_y,
            city_map,
            city_labels,
            city_embeddings,
            price_stats
        )

        # 构建结果字典
        row_result = {
            # 航班基本信息
            'flt_no': row['flt_no'],
            'flt_date': row['flt_date'],
            'aircraft': row['aircraft'],

            # 原始价格
            'original_ab_price': row['unit_price_ab'],
            'original_bc_price': row['unit_price_ab'],
            'original_ac_price': row['unit_price_ab'],

            # 原始收入
            'original_ab_revenue': row['tkt_rev_ab'],
            'original_bc_revenue': row['tkt_rev_bc'],
            'original_ac_revenue': row['tkt_rev_ac'],
            'original_total_revenue': row['tkt_rev_ab'] + row['tkt_rev_bc'] + row['tkt_rev_ac'],

            # 优化后的价格
            'optimal_ab_price': result['optimal_prices']['AB'],
            'optimal_bc_price': result['optimal_prices']['BC'],
            'optimal_ac_price': result['optimal_prices']['AC'],

            # 优化后的客流预测
            'predicted_ab_pax': result['predicted_pax']['AB_PAX'],
            'predicted_bc_pax': result['predicted_pax']['BC_PAX'],
            'predicted_ac_pax': result['predicted_pax']['AC_PAX'],

            # 优化后的座位分配
            'allocated_ab_seats': result['seat_allocation']['AB'],
            'allocated_bc_seats': result['seat_allocation']['BC'],
            'allocated_ac_seats': result['seat_allocation']['AC'],

            # 收入和提升
            'expected_revenue': result['expected_revenue'],
            'revenue_improvement': result['revenue_improvement'],
            'revenue_improvement_amount': result['revenue_improvement_amount']
        }

        results.append(row_result)
            

    
    # 转换为DataFrame
    results_df = pd.DataFrame(results)
    
    # 计算收入提升的统计信息
    total_flights = len(results_df)
    improved_flights = len(results_df[results_df['revenue_improvement'] > 0])
    improvement_ratio = improved_flights / total_flights * 100
    
    # 添加统计信息
    print("\n优化结果统计：")
    print(f"处理的航班数量: {total_flights}")
    print(f"收入提升的航班数量: {improved_flights}")
    print(f"收入提升航班占比: {improvement_ratio:.2f}%")
    print(f"\n收入提升详情:")
    print(f"平均收入提升: {results_df['revenue_improvement'].mean():.2f}%")
    print(f"最大收入提升: {results_df['revenue_improvement'].max():.2f}%")
    print(f"最小收入提升: {results_df['revenue_improvement'].min():.2f}%")
    print(f"总收入提升金额: {results_df['revenue_improvement_amount'].sum():,.2f}")
    
    # 添加分布统计
    print("\n收入提升分布:")
    improvement_ranges = [
        (results_df['revenue_improvement'] <= 0).sum(),
        ((results_df['revenue_improvement'] > 0) & (results_df['revenue_improvement'] <= 10)).sum(),
        ((results_df['revenue_improvement'] > 10) & (results_df['revenue_improvement'] <= 50)).sum(),
        ((results_df['revenue_improvement'] > 50) & (results_df['revenue_improvement'] <= 100)).sum(),
        (results_df['revenue_improvement'] > 100).sum()
    ]
    
    print(f"无提升航班数: {improvement_ranges[0]} ({improvement_ranges[0]/total_flights*100:.2f}%)")
    print(f"提升0-10%航班数: {improvement_ranges[1]} ({improvement_ranges[1]/total_flights*100:.2f}%)")
    print(f"提升10-50%航班数: {improvement_ranges[2]} ({improvement_ranges[2]/total_flights*100:.2f}%)")
    print(f"提升50-100%航班数: {improvement_ranges[3]} ({improvement_ranges[3]/total_flights*100:.2f}%)")
    print(f"提升>100%航班数: {improvement_ranges[4]} ({improvement_ranges[4]/total_flights*100:.2f}%)")
    
    return results_df


# In[21]:


# # 加载测试数据
# test_df = pd.read_csv('../../data-hh/my/hh_result/hh_result_2024_6_merged.csv')
# # 设置随机种子，确保结果可重现
# np.random.seed(42)

# # 随机抽取100行数据
# sampled_test_df = test_df.sample(n=100, random_state=42)

# # 重置索引
# sampled_test_df = sampled_test_df.reset_index(drop=True)
# # 现在可以使用sampled_test_df替代原来的test_df进行后续操作
# test_df = sampled_test_df


# # 加载所需的模型和编码器
# model, label_encoders, scaler_x, scaler_y, city_map, city_labels, city_embeddings = load_model_and_encoders()

# # 加载价格统计信息
# price_stats = pd.read_csv('../../data-hh/my/hh_result/route_price_stats.csv')




# In[22]:


# test_df.loc[120]


# In[23]:


# # 运行批量优化
# results_df = batch_optimize_prices(
#     test_df,
#     model,
#     label_encoders,
#     scaler_x,
#     scaler_y,
#     city_map,
#     city_labels,
#     city_embeddings,
#     price_stats
# )

# # 保存结果
# results_df.to_csv('optimization_results.csv', index=False)


# In[24]:


from multiprocessing import Pool, cpu_count
import numpy as np
from tqdm import tqdm
import pandas as pd

def process_single_flight(row, model, label_encoders, scaler_x, scaler_y, city_map, city_labels, city_embeddings, price_stats):
    """处理单个航班的优化任务"""
    try:
        test_data = create_test_data(row)
        result = optimize_prices(
            test_data, model, label_encoders, scaler_x, scaler_y, 
            city_map, city_labels, city_embeddings, price_stats
        )
        
        return {
            'flt_no': row['flt_no'],
            'flt_date': row['flt_date'],
            'original_ab_price': row['unit_price_ab'],
            'original_bc_price': row['unit_price_bc'],
            'original_ac_price': row['unit_price_ac'],
            'original_ab_revenue': row['tkt_rev_ab'],
            'original_bc_revenue': row['tkt_rev_bc'],
            'original_ac_revenue': row['tkt_rev_ac'],
            'original_total_revenue': row['tkt_rev_ab'] + row['tkt_rev_bc'] + row['tkt_rev_ac'],
            'optimal_ab_price': result['optimal_prices']['AB'],
            'optimal_bc_price': result['optimal_prices']['BC'],
            'optimal_ac_price': result['optimal_prices']['AC'],
            'predicted_ab_pax': result['predicted_pax']['AB_PAX'],
            'predicted_bc_pax': result['predicted_pax']['BC_PAX'],
            'predicted_ac_pax': result['predicted_pax']['AC_PAX'],
            'allocated_ab_seats': result['seat_allocation']['AB'],
            'allocated_bc_seats': result['seat_allocation']['BC'],
            'allocated_ac_seats': result['seat_allocation']['AC'],
            'expected_revenue': result['expected_revenue'],
            'revenue_improvement': result['revenue_improvement'],
            'revenue_improvement_amount': result['revenue_improvement_amount']
        }
    except Exception as e:
        print(f"处理航班 {row['flt_no']} 时出错: {str(e)}")
        return None

if __name__ == '__main__':
    # 记录开始时间
    import time
    start_time = time.time()
    print(f"开始运行时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 加载数据和模型
    test_df = pd.read_csv('../../data-hh/my/hh_result/hh_result_2024_6_merged.csv')
    model, label_encoders, scaler_x, scaler_y, city_map, city_labels, city_embeddings = load_model_and_encoders()
    price_stats = pd.read_csv('../../data-hh/my/hh_result/route_price_stats.csv')

    # 随机抽样100条数据
    # sampled_df = test_df.sample(n=1000, random_state=42)
    sampled_df= test_df
    
    # 设置进程数
    n_processes = max(1, cpu_count() - 1)
    print(f"使用 {n_processes} 个CPU核心进行并行处理")
    
    # 创建进程池并执行
    with Pool(processes=n_processes) as pool:
        results = list(tqdm(
            pool.starmap(
                process_single_flight,
                [(row, model, label_encoders, scaler_x, scaler_y, city_map, city_labels, city_embeddings, price_stats) 
                 for _, row in sampled_df.iterrows()]
            ),
            total=len(sampled_df),
            desc="优化票价"
        ))
    
    # 过滤掉None结果
    results = [r for r in results if r is not None]
    
    # 转换为DataFrame
    results_df = pd.DataFrame(results)
    
    # 计算统计信息
    total_flights = len(results_df)
    improved_flights = len(results_df[results_df['revenue_improvement'] > 0])
    improvement_ratio = improved_flights / total_flights * 100
    
    # 输出统计信息
    print("\n优化结果统计：")
    print(f"处理的航班数量: {total_flights}")
    print(f"收入提升的航班数量: {improved_flights}")
    print(f"收入提升航班占比: {improvement_ratio:.2f}%")
    print(f"\n收入提升详情:")
    print(f"平均收入提升: {results_df['revenue_improvement'].mean():.2f}%")
    print(f"最大收入提升: {results_df['revenue_improvement'].max():.2f}%")
    print(f"最小收入提升: {results_df['revenue_improvement'].min():.2f}%")
    print(f"总收入提升金额: {results_df['revenue_improvement_amount'].sum():,.2f}")
    
    # 保存结果
    results_df.to_csv('parallel_optimization_results.csv', index=False)
    
    # 记录结束时间
    end_time = time.time()
    print(f"\n结束运行时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总运行时长: {(end_time - start_time) / 60:.2f} 分钟")


# In[ ]:


# # 加载数据和模型
# test_df = pd.read_csv('../../data-hh/my/hh_result/hh_result_2024_6_merged.csv')
# model, label_encoders, scaler_x, scaler_y, city_map, city_labels, city_embeddings = load_model_and_encoders()
# price_stats = pd.read_csv('../../data-hh/my/hh_result/route_price_stats.csv')

# # 随机抽样100条数据
# sampled_df = test_df.sample(n=100, random_state=42)

# # 并行处理优化
# results_df = parallel_optimize_prices(
#     sampled_df,
#     model,
#     label_encoders,
#     scaler_x,
#     scaler_y,
#     city_map,
#     city_labels,
#     city_embeddings,
#     price_stats,
#     n_processes=6  # 使用6个CPU核心
# )

# # 保存结果
# results_df.to_csv('parallel_optimization_results.csv', index=False)

