import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sklearn.preprocessing import LabelEncoder
import datetime
import pickle
import re
import json
import os
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from functools import partial
from concurrent.futures import ProcessPoolExecutor  # ✅ 就是这个


# 定义常量
STATIC_CATEGORICAL_FIELDS = ["flt_no", "a", "b", "c", "from", "to"]
STATIC_NUMERIC_FIELDS = ["year", "month", "day", "weekday"]
STATIC_FIELDS = STATIC_CATEGORICAL_FIELDS + STATIC_NUMERIC_FIELDS

DCP_RANGE = list(range(29, -2, -1))  # DCP from 29 to -1

# 设备选择
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义清理列表字符串的函数
def clean_dcp_list(dcp_str):
    try:
        # 使用正则表达式清理列表字符串，只保留数字、逗号和负号
        cleaned = re.sub(r'[^\d,-]', '', str(dcp_str))
        return cleaned
    except Exception as e:
        return ""

def clean_pax_list(pax_str):
    try:
        # 使用正则表达式清理列表字符串，只保留数字、逗号、小数点和负号
        cleaned = re.sub(r'[^\d,.,-]', '', str(pax_str))
        return cleaned
    except Exception as e:
        return ""

# 高效数值列表解析函数
def parse_numeric_list_fast(list_str):
    """更高效的数值列表解析函数"""
    try:
        # 如果已经是列表，直接返回
        if isinstance(list_str, list):
            return list_str
            
        # 尝试直接JSON解析
        try:
            result = json.loads(list_str)
            # 确保结果是列表
            if isinstance(result, list):
                return result
            else:
                return [result]  # 单个值包装为列表
        except:
            pass
            
        # 清理并手动分割
        clean_str = re.sub(r"[^\d,.-]", "", str(list_str))
        
        # 检查是否为空
        if not clean_str:
            return []
            
        # 解析数值
        result = []
        for x in clean_str.split(','):
            if x.strip():
                try:
                    if '.' in x:
                        result.append(float(x))
                    else:
                        result.append(int(x))
                except:
                    continue
        return result
    except Exception as e:
        return []  # 任何错误，返回空列表

# 计算SMAPE指标
def calculate_smape(pred, true):
    """计算单个样本的对称平均绝对百分比误差"""
    if pred == true:
        return 0.0
    
    # 处理零值和负值情况
    if pred <= 0 and true <= 0:
        return 0.0
    
    # 标准SMAPE计算
    abs_diff = abs(pred - true)
    abs_sum = (abs(pred) + abs(true)) / 2.0
    
    # 防止除零
    if abs_sum < 1e-10:
        return 1.0  # 最大误差
    
    return abs_diff / abs_sum


# 将此函数作为模块级别函数，但需要将df作为参数传入
def process_feature(feature_encoder_pair, df):
    feature, encoder = feature_encoder_pair
    if feature not in df.columns:  # 修复了df不在作用域内的问题
        return feature, {}
    
    # 创建特征值到编码值的映射字典
    unique_values = df[feature].astype(str).unique()
    mapping = {}
    
    for val in tqdm(unique_values, desc=f"处理'{feature}'特征值", leave=False):
        try:
            encoded_val = encoder.transform([val])[0]
            mapping[val] = encoded_val
        except ValueError:
            mapping[val] = 0
    
    return feature, mapping

# 预处理所有编码器数据
def preprocess_encoders(encoders, df):
    """预处理所有编码器数据，一次性完成所有转换"""
    print("开始预处理编码器数据...")
    encoded_values = {}
    
    # 使用多进程处理
    from multiprocessing import Pool, cpu_count
    
    # 创建一个带有df参数的部分函数
    process_feature_with_df = partial(process_feature, df=df)
    
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(
            pool.imap(process_feature_with_df, encoders.items()),
            desc="编码特征处理",
            total=len(encoders)
        ))
    
    # 合并结果
    for feature, mapping in results:
        if mapping:  # 只添加非空映射
            encoded_values[feature] = mapping
    
    return encoded_values

# 并行处理单个样本
def process_sample(args):
    """处理单个样本的函数，用于并行处理"""
    idx, row, encoders_dict, static_dim_saved, dcp_range = args
    
    try:
        # 解析列表
        dcp_list = parse_numeric_list_fast(row['clean_dcp_list'])
        pax_list = parse_numeric_list_fast(row['clean_pax_list'])
        
        # 有效性检查
        if not dcp_list or not pax_list or len(dcp_list) != len(pax_list):
            return None
        
        # 对于负PAX值进行过滤 (可选)
        if any(pax < 0 for pax in pax_list):
            # 或者可以将负值替换为0: pax_list = [max(0, pax) for pax in pax_list]
            return None
        
        # 排序 - 保持原有的排序方式：DCP从大到小
        paired = sorted(zip(dcp_list, pax_list), key=lambda x: x[0], reverse=True)
        if len(paired) < 6:
            return None
        
        # 处理特征
        first5 = paired[:5]  # DCP值较大的前5个点(离起飞日期较远)
        target_dcp, true_pax = paired[-1]  # DCP值较小的点(接近起飞日期)
        
        # 构造DCP特征
        values, mask = [], []
        d2p = dict(first5)
        
        for dcp in dcp_range:
            if dcp in d2p:
                values.append(float(d2p[dcp]))
                mask.append(1)
            else:
                values.append(0.0)
                mask.append(0)
        
        # 检查是否有足够的有效特征
        if sum(mask) < 3:  # 至少需要3个有效的DCP值
            return None
        
        # 标准化PAX值 (可选)
        # max_pax = max([p for _, p in first5] + [1.0])
        # values = [v / max_pax for v in values]
        # true_pax = true_pax / max_pax
        
        # 构造静态特征
        static_vals = []
        
        # 使用预计算的编码值
        for feature, mapping in encoders_dict.items():
            if feature in row:
                val = str(row[feature])
                static_vals.append(mapping.get(val, 0))
            else:
                static_vals.append(0)
        
        # 添加剩余特征
        remaining_features = static_dim_saved - len(static_vals)
        if remaining_features > 0:
            numeric_features = ["year", "month", "day", "weekday"]
            for f in numeric_features:
                if len(static_vals) < static_dim_saved and f in row:
                    static_vals.append(float(row[f]))
            
            # 填充剩余维度
            static_vals.extend([0.0] * (static_dim_saved - len(static_vals)))
        
        # 截断超出的维度
        static_vals = static_vals[:static_dim_saved]
        
        return (idx, values, mask, static_vals, true_pax)
    except Exception as e:
        return None

def create_args_for_chunk(chunk_data):
    chunk, encoders_dict, static_dim_saved, dcp_range = chunk_data
    return [(idx, row, encoders_dict, static_dim_saved, dcp_range) 
            for idx, row in chunk.iterrows()]

# 改进的数据集类
class ImprovedPaxPredictionDataset(Dataset):
    def __init__(self, df, encoders, static_dim_saved, dcp_range, max_workers=None):
        self.valid_samples = []
        self.sample_indices = []
        
        # 确定工作进程数量
        if max_workers is None:
            max_workers = mp.cpu_count()-1  # 使用所有可用的CPU核心
        
        print(f"使用 {max_workers} 个工作进程预处理数据")
        
        # 预处理所有编码器数据
        encoders_dict = preprocess_encoders(encoders, df)
        
        # 使用多进程加速准备参数
        print("准备样本处理参数...")
        # 将DataFrame分块以加速参数准备
        chunk_size = 10000
        df_chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]

        # 准备完整的参数列表
        chunk_data_list = [(chunk, encoders_dict, static_dim_saved, dcp_range) for chunk in df_chunks]

        
        args_list = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 并行处理每个数据块
            chunk_args_list = list(tqdm(
                executor.map(create_args_for_chunk, chunk_data_list),  # 传递完整的参数
                total=len(df_chunks),
                desc="并行准备参数"
            ))
            
            # 合并所有参数列表
            for chunk_args in chunk_args_list:
                args_list.extend(chunk_args)
        
        # 使用进程池并行处理样本
        print(f"开始并行处理 {len(args_list)} 个样本...")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(
                executor.map(process_sample, args_list, chunksize=1000),
                total=len(args_list),
                desc="并行处理数据集"
            ))
        
        # 收集有效结果
        print("筛选有效样本...")
        valid_count = 0
        for result in tqdm(results, desc="收集处理结果"):
            if result is not None:
                idx, values, mask, static_vals, true_pax = result
                self.valid_samples.append((values, mask, static_vals, true_pax))
                self.sample_indices.append(idx)
                valid_count += 1
        
        print(f"创建了包含 {len(self.valid_samples)} 个有效样本的数据集 (有效率: {len(self.valid_samples)/len(df)*100:.2f}%)")
    
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        values, mask, static_vals, target = self.valid_samples[idx]
        orig_idx = self.sample_indices[idx]
        return (
            torch.tensor(values, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.bool),
            torch.tensor(static_vals, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
            orig_idx
        )

# 改进的静态特征编码器
class ImprovedStaticFeatureEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super().__init__()
        # 更复杂的网络架构
        self.fc = nn.Sequential(
            nn.Linear(input_dim, emb_dim*2),
            nn.BatchNorm1d(emb_dim*2),  # 添加批归一化
            nn.ReLU(),
            nn.Dropout(0.2),  # 添加dropout防止过拟合
            nn.Linear(emb_dim*2, emb_dim)
        )

    def forward(self, x):
        return self.fc(x)

# 改进的Transformer模型
class ImprovedSalesTransformerModel(nn.Module):
    def __init__(self, static_dim, emb_dim=128, nhead=16, num_layers=6, ff_dim=512, dropout=0.15):
        super().__init__()
        # 静态特征编码器
        self.static_encoder = ImprovedStaticFeatureEncoder(static_dim, emb_dim)
        
        # 时间序列特征处理
        self.embedding = nn.Sequential(
            nn.Linear(1, emb_dim),
            nn.LayerNorm(emb_dim)  # 添加层归一化
        )
        
        # 修复Transformer编码器 - 设置batch_first=True
        encoder_layer = TransformerEncoderLayer(
            d_model=emb_dim, 
            nhead=nhead, 
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True  # 关键修改
        )
        
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 池化和解码器
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # 扩展解码器网络
        self.decoder = nn.Sequential(
            nn.Linear(2 * emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, 1)
        )
        
        # 初始化权重 - Xavier初始化
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, seq_x, mask_x, static_x):
        # 编码静态特征
        static_feat = self.static_encoder(static_x)  # (B, emb_dim)
        
        # 嵌入时间序列特征
        seq_emb = self.embedding(seq_x.unsqueeze(-1))  # (B, seq_len, emb_dim)
        
        # 由于batch_first=True，直接传入Transformer
        seq_encoded = self.transformer(seq_emb, src_key_padding_mask=~mask_x)  # (B, seq_len, emb_dim)
        
        # 池化 - 调整维度顺序以适应池化操作
        seq_encoded = seq_encoded.transpose(1, 2)  # (B, emb_dim, seq_len)
        seq_pooled = self.pool(seq_encoded).squeeze(-1)  # (B, emb_dim)
        
        # 合并特征
        combined = torch.cat([seq_pooled, static_feat], dim=1)  # (B, 2*emb_dim)
        
        # 解码并确保输出非负
        output = self.decoder(combined).squeeze(-1)  # (B,)
        
        # 使用ReLU确保PAX值非负
        return torch.relu(output)  # 确保PAX值非负

# 
def train_model(model, train_loader, val_loader, optimizer, scheduler, loss_fn, device, epochs=10, patience=3):
    """
    训练模型的函数，包含早停和验证集评估
    """
    print(f"开始训练，共{epochs}轮，设备: {device}")
    
    best_val_loss = float('inf')
    no_improve_epochs = 0
    best_model_state = None
    
    for epoch in range(epochs):
        print(f"\n开始训练第 {epoch+1}/{epochs} 轮...")
        
        # 训练阶段
        model.train()
        train_loss = 0
        
        train_pbar = tqdm(train_loader, desc=f"轮次 {epoch+1}", ncols=100)
        
        for batch_idx, batch_data in enumerate(train_pbar):
            # 只取前4个元素
            x, mask, static_x, y = batch_data[:4]
            # 移到设备
            x, mask, static_x, y = x.to(device), mask.to(device), static_x.to(device), y.to(device)
            
            # 前向传播
            outputs = model(x, mask, static_x)
            
            # 计算损失
            loss = loss_fn(outputs, y)
            
            # 更新模型
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 更新损失总和
            train_loss += loss.item()
            
            # 更新进度条显示的损失
            if batch_idx % 10 == 0:
                # 监控预测值范围
                avg_pred = outputs.mean().item()
                min_pred = outputs.min().item()
                max_pred = outputs.max().item() 
                avg_target = y.mean().item()
                
                train_pbar.set_postfix({
                    "loss": f"{loss.item():.2f}",
                    "avg_pred": f"{avg_pred:.2f}",
                    "pred_range": f"[{min_pred:.2f}, {max_pred:.2f}]",
                    "avg_target": f"{avg_target:.2f}"
                })
        
        # 计算平均训练损失
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch_data in tqdm(val_loader, desc="验证中"):
                # 只取前4个元素
                x, mask, static_x, y = batch_data[:4]
                x, mask, static_x, y = x.to(device), mask.to(device), static_x.to(device), y.to(device)
                
                # 前向传播
                outputs = model(x, mask, static_x)
                
                # 计算损失
                loss = loss_fn(outputs, y)
                val_loss += loss.item()
                
                # 收集预测值和真实值用于计算SMAPE
                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(y.cpu().numpy())
        
        # 计算平均验证损失和SMAPE
        avg_val_loss = val_loss / len(val_loader)
        
        # 计算SMAPE
        smapes = []
        for pred, true in zip(val_preds, val_targets):
            smapes.append(calculate_smape(pred, true))
        avg_smape = np.mean(smapes) * 100  # 转为百分比
        
        # 打印训练和验证结果
        print(f"[轮次 {epoch+1}/{epochs}] 训练损失: {avg_train_loss:.4f}, 验证损失: {avg_val_loss:.4f}, SMAPE: {avg_smape:.2f}%")
        
        # 学习率调整
        scheduler.step(avg_val_loss)
        
        # 早停检查
        if avg_val_loss < best_val_loss:
            print(f"验证损失从 {best_val_loss:.4f} 改善到 {avg_val_loss:.4f}，保存模型")
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            print(f"验证损失未改善，已经 {no_improve_epochs} 轮未改善")
            
            if no_improve_epochs >= patience:
                print(f"早停！已经 {patience} 轮未见改善")
                break
    
    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("已恢复到最佳模型")
    
    return model

def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 数据文件路径
    csv_path = "/home/zhanyu/hh-experiment/课题1/project2/merged_2024.csv"
    
    # 确保输出目录存在
    trans_dir = "trans_dir_all"
    if not os.path.exists(trans_dir):
        os.makedirs(trans_dir)
    
    # 加载数据
    print(f"加载数据文件: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"原始数据行数: {len(df)}")
    
    # 基本过滤
    df = df.dropna(subset=["dcp_list", "pax_list", "flt_date", "segment"])
    print(f"过滤后数据行数: {len(df)}")
    
    # 拆分 segment
    if "from" not in df.columns or "to" not in df.columns:
        df[["from", "to"]] = df["segment"].str.split("-", expand=True)
    
    # 处理日期
    df["flt_date"] = pd.to_datetime(df["flt_date"], errors="coerce")
    df["year"] = df["flt_date"].dt.year
    df["month"] = df["flt_date"].dt.month
    df["day"] = df["flt_date"].dt.day
    df["weekday"] = df["flt_date"].dt.weekday
    
    # 清理数据
    print("正在批量清理DCP和PAX列表...")
    df['clean_dcp_list'] = df['dcp_list'].apply(clean_dcp_list)
    df['clean_pax_list'] = df['pax_list'].apply(clean_pax_list)
    
    # 过滤掉清理后为空的行
    df = df[df['clean_dcp_list'] != ""]
    df = df[df['clean_pax_list'] != ""]
    print(f"清理后剩余数据行数: {len(df)}")
    
    # 创建并拟合编码器
    encoders = {}
    for f in STATIC_CATEGORICAL_FIELDS:
        le = LabelEncoder()
        le.fit(df[f].astype(str).fillna('UNKNOWN'))
        encoders[f] = le
    
    # 创建数据集
    print("开始创建数据集...")
    dataset = ImprovedPaxPredictionDataset(
        df=df,
        encoders=encoders,
        static_dim_saved=len(STATIC_FIELDS),
        dcp_range=DCP_RANGE,
        max_workers=os.cpu_count()-1
    )
    # # 保存数据集到磁盘，避免每次重新处理
    # dataset_path = os.path.join(trans_dir, "processed_dataset.pkl")
    # print(f"保存处理后的数据集到 {dataset_path}")
    # try:
    #     with open(dataset_path, "wb") as f:
    #         pickle.dump(dataset, f)
    #     print("数据集保存成功！")
    # except Exception as e:
    #     print(f"保存数据集时出错: {e}")
    #     print("将继续执行但不保存数据集")
    
    # 添加加载已处理数据集的功能
    # 使用示例:
    # if os.path.exists(dataset_path):
    #     print(f"加载已处理的数据集 {dataset_path}")
    #     with open(dataset_path, "rb") as f:
    #         dataset = pickle.load(f)
    #     print(f"成功加载数据集，包含 {len(dataset)} 个样本")
    # else:
    #     print("未找到已处理的数据集，将重新创建...")
    #     dataset = ImprovedPaxPredictionDataset(...)
    
    # 保存编码器
    print(f"保存编码器到 {trans_dir}/static_label_encoders.pkl")
    with open(os.path.join(trans_dir, "static_label_encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    
    # 创建数据加载器
    # 为训练集创建DataLoader，用于批量加载数据
    train_loader = DataLoader(
        train_dataset,  # 训练数据集
        batch_size=1024,  # 每批处理1024个样本，较大的批量可以提高训练效率
        shuffle=True,  # 打乱数据顺序，防止模型学习到数据顺序相关的模式
        num_workers=min(64, os.cpu_count()),  # 使用多进程加载数据，但不超过8个或CPU核心数
        pin_memory=True  # 将数据直接加载到CUDA固定内存中，加速GPU训练
    )
    
    # 为验证集创建DataLoader
    val_loader = DataLoader(
        val_dataset,  # 验证数据集
        batch_size=128,  # 验证时使用更大的批量，因为不需要计算梯度，可以节省内存
        shuffle=False,  # 验证集不需要打乱顺序
        num_workers=min(64, os.cpu_count()),  # 同样使用多进程加载数据
        pin_memory=True  # 同样使用CUDA固定内存加速
    )
    
    # 创建模型
    # 实例化改进版的销售预测Transformer模型
    model = ImprovedSalesTransformerModel(
        static_dim=len(STATIC_FIELDS),  # 静态特征的维度，由STATIC_FIELDS列表长度决定
        emb_dim=256,  # 嵌入维度为128，决定了特征表示的丰富程度
        nhead=16,  # 多头注意力机制中的头数为16，可以学习不同方面的特征关系
        num_layers=6,  # Transformer编码器的层数为6，增加模型深度和表达能力
        ff_dim=1024,  # 前馈神经网络的隐藏层维度为512，控制模型复杂度
        dropout=0.15  # 丢弃率为0.15，防止过拟合
    ).to(DEVICE)  # 将模型移动到指定设备(CPU或GPU)上
    
    # 打印模型结构
    print(f"模型结构:\n{model}")
    
    # 定义优化器和学习率调度器
    # 使用Adam优化器，初始学习率为1e-4，权重衰减(L2正则化)为1e-5
    # Adam优化器结合了动量和自适应学习率，适合大多数深度学习任务
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # 创建学习率调度器，当验证损失不再下降时降低学习率
    # mode='min'表示监控的指标是越小越好（如损失值）
    # factor=0.5表示每次降低学习率为原来的一半
    # patience=2表示连续2个epoch验证损失没有改善才降低学习率
    # min_lr=1e-6设置学习率的下限，防止学习率过小
    # verbose=True表示在调整学习率时打印信息
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=2,
        min_lr=1e-7,
        verbose=True
    )
    
    # 定义损失函数 - 组合MSE和平滑L1
    def combined_loss(pred, target):
        mse_loss = nn.MSELoss()(pred, target)
        l1_loss = nn.SmoothL1Loss()(pred, target)
        return 0.7 * mse_loss + 0.3 * l1_loss
    
    # 训练模型
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=combined_loss,
        device=DEVICE,
        epochs=100,
        patience=3
    )
    
    # 保存模型
    model_path = os.path.join(trans_dir, "improved_pax_predictor.pt")
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到: {model_path}")
    
    # 评估最终模型
    model.eval()
    all_preds = []
    all_targets = []
    
    print("对验证集进行最终评估...")
    with torch.no_grad():
        for x, mask, static_x, y, _ in tqdm(DataLoader(val_dataset, batch_size=2048)):
            x, mask, static_x, y = x.to(DEVICE), mask.to(DEVICE), static_x.to(DEVICE), y.to(DEVICE)
            outputs = model(x, mask, static_x)
            
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    
    # 计算SMAPE
    smapes = []
    for pred, true in zip(all_preds, all_targets):
        smapes.append(calculate_smape(pred, true))
    
    final_smape = np.mean(smapes) * 100
    print(f"最终验证集SMAPE: {final_smape:.2f}%")
    
    # 保存一些评估结果样本
    sample_results = []
    for i in range(min(20, len(all_preds))):
        sample_results.append({
            "prediction": float(all_preds[i]),
            "target": float(all_targets[i]),
            "smape": float(smapes[i] * 100)
        })
    
    print("\n样本预测结果:")
    for i, res in enumerate(sample_results[:5]):
        print(f"样本 {i+1}: 预测值={res['prediction']:.2f}, 真实值={res['target']:.2f}, SMAPE={res['smape']:.2f}%")
    
    print("\n训练完成!")

if __name__ == "__main__":
    main()