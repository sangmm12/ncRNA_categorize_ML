import pandas as pd
import numpy as np

# Load the CSV file
input_file = 'rnacentral/rnacentral_active.csv'
df = pd.read_csv(input_file)

# === 1) 先对 lncRNA 进行长度过滤：删除 Label='lncRNA' 且 Length<200 的记录 ===
# 确保 Length 可数值化
df['Length'] = pd.to_numeric(df['Length'], errors='coerce')
before_rows = len(df)
mask_lnc = (df['Label'] == 'lncRNA')
mask_short = mask_lnc & (df['Length'] < 200)
df = df[~mask_short].copy()
after_rows = len(df)
print(f"[INFO] Removed {before_rows - after_rows} rows where Label='lncRNA' and Length<200. Remaining rows: {after_rows}")

# Group by 'Label' and calculate the count and average length
result = df.groupby('Label').agg(
    Record_Count=('GeneName', 'count'),
    Average_Length=('Length', 'mean')
).reset_index()

# Sort the result by Record_Count in descending order
result = result.sort_values(by='Record_Count', ascending=False)

# Display the result
print("Label Summary:")
print(result)

# === 定义指定的10个标签集合 ===
label_set_10 = {'miRNA', 'siRNA', 'lncRNA', 'rRNA', 'tRNA', 'snRNA', 'snoRNA', 'Y_RNA', 'SRP_RNA', 'pre_miRNA'}

# 过滤出指定标签的数据
df_label_set = df[df['Label'].isin(label_set_10)].copy()
print(f"[INFO] Selected labels: {label_set_10}")
print(f"[INFO] Total records in specified label set: {len(df_label_set)}")

# 显示每个标签的记录数量
label_counts = df_label_set['Label'].value_counts()
print("Record counts for each label in the set:")
for label in label_set_10:
    count = label_counts.get(label, 0)
    print(f"  {label}: {count}")

# === Task: 对指定的10个标签，每个标签随机选 20,000 条；连续生成 10 次，用不同种子 ===
seeds = list(range(42, 52))
for seed in seeds:
    df_label_set_2w = (
        df_label_set.groupby('Label', group_keys=False)
        .apply(lambda x: x.sample(n=min(len(x), 20_000), random_state=seed))
        .reset_index(drop=True)
    )
    file_label_set_2w = f'rnacentral/rnacentral_active_label_set10_2w_seed{seed}.csv'
    df_label_set_2w.to_csv(file_label_set_2w, index=False)
    
    # 统计每个标签的实际采样数量
    sample_counts = df_label_set_2w['Label'].value_counts()
    print(f"[OK] Seed {seed}: Dataset with specified 10 labels and up to 20,000 records per label saved to {file_label_set_2w}")
    print(f"      Sample counts: {dict(sample_counts)}")

print("\n[COMPLETE] Generated 10 datasets with specified label set, each label sampled up to 20,000 records.")
