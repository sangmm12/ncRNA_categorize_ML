#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd

# 1. 参数设置
#input_file  = 'mammalian_multispecies/mammalian_train_dev_pre_BPENV1368_limit30.csv'
#output_file = 'mammalian_multispecies/mammalian_train_dev_pre_Avg_bpe.csv'

input_file  = 'mammalian_multispecies/multispecies_train_dev_pre_BPENV1368_limit30.csv'
output_file = 'mammalian_multispecies/multispecies_train_dev_pre_Avg_bpe.csv'
# 2. 特征分组定义（划分为三类）
# 类别1: 基础统计量 (V1-V8)
# 类别2: 短/中长度 k-mer (V9-V344)
# 类别3: 长序列 k-mer (V345-V1368)
groups = {
    'basic_stats':  list(range(1, 5)),      # V1 到 V8
    'short_kmers':  list(range(5, 9)),    # V9 到 V344
    'long_kmers':   list(range(9, 1369)), # V345 到 V1368
}

# 3. 加载主文件
df = pd.read_csv(input_file)

# 自动获取所有特征列名 (V1, V2, ..., V1368)
feature_cols = [f'V{i}' for i in range(1, 1369)]

# 4. 按 Label 聚合求平均值
# 只对 Label 进行分组，计算所有 V 列的均值
print(f"正在对 {len(feature_cols)} 个特征进行分组聚合并计算均值...")
grouped = df.groupby('Label')[feature_cols].mean().reset_index()

# 5. 分类最大值归一化处理 (Row-wise Max Normalization)
# 在每一行（每个 Label）内，针对三个不同的特征组分别进行归一化
print("执行三类特征的组内归一化...")
for group_name, idxs in groups.items():
    # 构造当前组的列名列表
    current_group_cols = [f'V{i}' for i in idxs]
    
    # 计算当前行、当前组内的最大值，并将 0 替换为 1 避免除零错误
    row_max = grouped[current_group_cols].max(axis=1).replace(0, 1)
    
    # 执行归一化：当前组数据 / 当前组最大值
    grouped[current_group_cols] = grouped[current_group_cols].div(row_max, axis=0)
    
    print(f"  - {group_name} (包含 {len(current_group_cols)} 列) 处理完毕")

# 6. 输出保存
grouped.to_csv(output_file, index=False)
print(f"✅ 处理完成！结果已保存至: {output_file}")
