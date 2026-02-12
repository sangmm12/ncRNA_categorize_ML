import pandas as pd
import numpy as np

# Load the CSV file
#input_file = 'rnacentral/rnacentral_active.csv'
#input_file = 'rnacentral_active_NV1368_lncRNA_less200_afterFilter.csv'
input_file = 'rnacentral_active_NV1368_lncRNA_less200_afterFilter_2.csv'
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

# Keep only the top 20 labels
top_20_labels = result['Label'].head(20).tolist()
df_top20 = df[df['Label'].isin(top_20_labels)].copy()
'''
# Task 1: For the label 'rRNA', randomly select 3,400,000 records (or all if fewer)
df_top20_rRNA = df_top20.copy()
if 'rRNA' in df_top20['Label'].unique():
    rRNA_all = df_top20[df_top20['Label'] == 'rRNA']
    n_pick = min(len(rRNA_all), 3_400_000)
    if n_pick < 3_400_000:
        print(f"[WARN] rRNA only has {len(rRNA_all)} rows; picking all ({n_pick}).")
    rRNA_subset = rRNA_all.sample(n=n_pick, random_state=42)
    # 其余非 rRNA + 采样后的 rRNA
    df_top20_rRNA = pd.concat([df_top20[df_top20['Label'] != 'rRNA'], rRNA_subset], ignore_index=True)
else:
    print("[INFO] 'rRNA' not in top 20 labels; Task 1 will just keep df_top20 unchanged.")

# Save the dataset with the top 20 labels and reduced 'rRNA' records
file_top20_rRNA = 'rnacentral/rnacentral_active_top20_rRNA.csv'
df_top20_rRNA.to_csv(file_top20_rRNA, index=False)
print(f"Dataset with top 20 labels and reduced 'rRNA' records saved to {file_top20_rRNA}")
'''
# === 2) Task 2: 对 top 20 labels，每个标签随机选 20,000 条；连续生成 10 次，用不同种子 ===
# 这里使用种子列表 42~51，你可以按需改动
seeds = list(range(42, 52))
for seed in seeds:
    df_top20_2w = (
        df_top20.groupby('Label', group_keys=False)
        .apply(lambda x: x.sample(n=min(len(x), 20_000), random_state=seed))
        .reset_index(drop=True)
    )
    #file_top20_2w = f'rnacentral/rnacentral_active_top20_2w_seed{seed}.csv'
    #file_top20_2w = f'rnacentral_active_NV1368_lncRNA_less200_afterFilter_top20_2w_seed{seed}.csv'
    file_top20_2w = f'rnacentral_active_NV1368_lncRNA_less200_afterFilter_2_top20_2w_seed{seed}.csv'
    df_top20_2w.to_csv(file_top20_2w, index=False)
    print(f"[OK] Dataset with top 20 labels and 20,000 records per label saved to {file_top20_2w}")

'''
import pandas as pd

# Load the CSV file
input_file = 'rnacentral_active.csv'
df = pd.read_csv(input_file)

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

# Keep only the top 20 labels
top_20_labels = result['Label'].head(20).tolist()
df_top20 = df[df['Label'].isin(top_20_labels)]

# Task 1: For the label 'rRNA', randomly select 3,400,000 records
if 'rRNA' in df_top20['Label'].unique():
    rRNA_subset = df_top20[df_top20['Label'] == 'rRNA'].sample(n=3400000, random_state=42)
    # Keep the rest of the dataset excluding 'rRNA'
    df_top20_rRNA = pd.concat([df_top20[df_top20['Label'] != 'rRNA'], rRNA_subset])

# Save the dataset with the top 20 labels and reduced 'rRNA' records
file_top20_rRNA = 'rnacentral_active_top20_rRNA.csv'
df_top20_rRNA.to_csv(file_top20_rRNA, index=False)
print(f"Dataset with top 20 labels and reduced 'rRNA' records saved to {file_top20_rRNA}")

# Task 2: For all top 20 labels, randomly select 20,000 records per label
df_top20_2w = (
    df_top20.groupby('Label', group_keys=False)
    .apply(lambda x: x.sample(n=min(len(x), 20000), random_state=42))
)

# Save the dataset with top 20 labels and 20,000 records per label
file_top20_2w = 'rnacentral_active_top20_2w.csv'
df_top20_2w.to_csv(file_top20_2w, index=False)
print(f"Dataset with top 20 labels and 20,000 records per label saved to {file_top20_2w}")
'''
