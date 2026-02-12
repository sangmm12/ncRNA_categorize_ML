import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA # 建议添加此导入

# Load data
#data = pd.read_csv('rnacentral/rnacentral_active_label_set10_2w_seed42_NV1368.csv', index_col=0)
#data = pd.read_csv('rnacentral/rnacentral_active_top20_2w_seed42_NV1368.csv', index_col=0)
#data = pd.read_csv('mammalian_multispecies/mammalian_train_dev_pre_NV.csv', index_col=0)
#data = pd.read_csv('mammalian_multispecies/mammalian_train_dev_pre_NV1368.csv', index_col=0)
data = pd.read_csv('mammalian_multispecies/multispecies_train_dev_pre_NV.csv', index_col=0)
#data = pd.read_csv('mammalian_multispecies/multispecies_train_dev_pre_NV1368.csv', index_col=0)

#data = pd.read_csv('mammalian_multispecies/mammalian_train_dev_pre_BPENV1368_limit30.csv', index_col=0)
#data = pd.read_csv('mammalian_multispecies/multispecies_train_dev_pre_BPENV1368_limit30.csv', index_col=0)
print(data.head())
print(data.shape)

# Shuffle the data
data = shuffle(data, random_state=42)

# Define global label order and color mapping
GLOBAL_LABEL_ORDER = [
    'rRNA', 'Y_RNA', 'snRNA', 'snoRNA', 'SRP_RNA', 'tRNA', 'pre_miRNA',
    'miRNA', 'siRNA', 'lncRNA', 'misc_RNA', 'sRNA', 'ncRNA', 'piRNA',
    'tmRNA', 'hammerhead_ribozyme', 'RNase_P_RNA', 'antisense_RNA', 
    'other', 'ribozyme'
]

# Generate fixed colors for all possible labels
'''
GLOBAL_COLOR_MAP = {}
colors = [mcolors.hsv_to_rgb((i / len(GLOBAL_LABEL_ORDER), 1.0, 1.0)) for i in range(len(GLOBAL_LABEL_ORDER))]
for i, label in enumerate(GLOBAL_LABEL_ORDER):
    GLOBAL_COLOR_MAP[label] = colors[i]
'''
'''
import matplotlib.pyplot as plt
# 替换颜色生成部分
GLOBAL_COLOR_MAP = {}
tab20_colors = plt.cm.tab20.colors
for i, label in enumerate(GLOBAL_LABEL_ORDER):
    if i < len(tab20_colors):
        GLOBAL_COLOR_MAP[label] = tab20_colors[i]
    else:
        # 如果标签超过20个，使用Set3调色板补充
        set3_colors = plt.cm.Set3.colors
        GLOBAL_COLOR_MAP[label] = set3_colors[(i - len(tab20_colors)) % len(set3_colors)]
'''
'''
# 替换颜色生成部分
GLOBAL_COLOR_MAP = {}
set3_colors = plt.cm.Set3.colors
for i, label in enumerate(GLOBAL_LABEL_ORDER):
    GLOBAL_COLOR_MAP[label] = set3_colors[i % len(set3_colors)]
'''

# 替换颜色生成部分
CUSTOM_COLORS = [
    # ←—— 从图像中抽取的 7 种类别色 ——→
    '#E93323',  # rRNA
    '#EBDF5C',  # Y_RNA
    '#82FB55',  # snRNA
    '#75FA9D',  # snoRNA
    '#3F92F3',  # SRP_RNA
    '#3D0AEE',  # tRNA
    '#E634D3',  # pre_miRNA

    # ←—— 其余色：鲜亮 hue-style 等间隔 ——→
    '#F22424', '#F25A24', '#F28F24', '#F2C524', '#E9F224',
    '#B8F224', '#7DF224', '#42F224', '#24F24D', '#24F288',
    '#24F2C3', '#24DDF2', '#249FF2', '#2462F2', '#2824F2',
    '#6524F2', '#A224F2', '#DF24F2', '#F224C1', '#F22484',
    '#F22447', '#F27824', '#C8F224'
]
GLOBAL_COLOR_MAP = {}
for i, label in enumerate(GLOBAL_LABEL_ORDER):
    if i < len(CUSTOM_COLORS):
        GLOBAL_COLOR_MAP[label] = CUSTOM_COLORS[i]
    else:
        # 生成随机颜色作为备用
        import random
        GLOBAL_COLOR_MAP[label] = (random.random(), random.random(), random.random())

# Define label mappings for different datasets
# For mammalian
mammalian_label_map = {
    0: 'rRNA',
    1: 'Y_RNA',
    2: 'snRNA',
    3: 'snoRNA',
    4: 'SRP_RNA',
    5: 'tRNA',
    6: 'pre_miRNA'
}

# For multispecies
multispecies_label_map = {
    6: 'rRNA',
    9: 'Y_RNA',
    5: 'snRNA',
    3: 'snoRNA',
    2: 'SRP_RNA',
    0: 'tRNA',
    1: 'pre_miRNA'
}

# Define label sets (using numeric labels for filtering, will convert to string later)
label_set_7_mammalian = {0, 1, 2, 3, 4, 5, 6}
label_set_7_multispecies = {0, 1, 2, 3, 5, 6, 9}
label_set_10 = {"rRNA","Y_RNA","snRNA","snoRNA","SRP_RNA","tRNA","pre_miRNA","miRNA","siRNA","lncRNA"}
label_set_20 = {"rRNA","Y_RNA","snRNA","snoRNA","SRP_RNA","tRNA","pre_miRNA","miRNA","siRNA","lncRNA",
    "misc_RNA", "sRNA","ncRNA","piRNA","tmRNA","hammerhead_ribozyme","RNase_P_RNA","antisense_RNA","other", "ribozyme"}

#label_set_10_numeric = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}  # Adjust based on your actual data
#label_set_20_numeric = set(range(20))  # Adjust based on your actual data

# Select which label set and mapping to use
# For mammalian data:
label_set_numeric = label_set_7_mammalian
label_map = mammalian_label_map

# For multispecies data:
#label_set_numeric = label_set_7_multispecies
#label_map = multispecies_label_map

# For other datasets (adjust as needed):
#label_set_numeric = label_set_10#label_set_10_numeric
#label_set_numeric = label_set_10
#label_map = None  # If labels are already strings

# Filter data based on the numeric label set
filtered_data = data[data['Label'].isin(label_set_numeric)]
print(f"Filtered data shape: {filtered_data.shape}")

# Check if we have any data after filtering
if len(filtered_data) == 0:
    raise ValueError("No data found after filtering! Check your label set and data.")

# Count records per label
label_counts = filtered_data['Label'].value_counts()
print("Label counts after filtering:")
print(label_counts)

min_records = label_counts.min()
print(f"Minimum records for labels in label_set: {min_records}")

# Sample min_records for each label (ensure we have enough samples)
if min_records > 0:
    balanced_data = (
        filtered_data.groupby('Label', group_keys=False)
        .apply(lambda x: x.sample(n=min_records, random_state=42, replace=False))
        .reset_index(drop=True)
    )
else:
    raise ValueError("No samples available for some labels after filtering!")

print("Balanced data label counts:")
print(balanced_data['Label'].value_counts())

balanced_data = shuffle(balanced_data, random_state=42)

# Convert numeric labels to string labels using the appropriate mapping
if label_map is not None:
    balanced_data['Label'] = balanced_data['Label'].map(label_map)
    # Remove any rows that couldn't be mapped
    balanced_data = balanced_data.dropna(subset=['Label'])
    balanced_data['Label'] = balanced_data['Label'].astype(str)

print("Final label distribution:")
print(balanced_data['Label'].value_counts())

# Separate features and labels
X = balanced_data.iloc[:, -1368:]
y = balanced_data['Label']

# Define the feature groups
feature_groups = {
    '24-dim': pd.concat([X.iloc[:, 0:24]], axis=1),
    '88-dim': pd.concat([X.iloc[:, 0:88]], axis=1),
    '344-dim': pd.concat([X.iloc[:, 0:344]], axis=1),
    '1368-dim': pd.concat([X.iloc[:, 0:1368]], axis=1),
}

# Perform t-SNE and plot each feature group
for group_name, group_data in feature_groups.items():
    print(f"Processing {group_name} with shape {group_data.shape}")

    # --- 修复核心：数据预处理 ---
    # 1. 标准化：解决 overflow 问题的关键
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(group_data)

    # 2. PCA 预降维：处理高维数据（如 344 和 1368 维）的官方推荐做法
    # 如果维度大于 50，先降到 50 维，这样可以避免数值计算风险并提速
    if scaled_data.shape[1] > 50:
        pca = PCA(n_components=50, random_state=42)
        input_data = pca.fit_transform(scaled_data)
    else:
        input_data = scaled_data

    input_data = group_data    
    # --- 运行 t-SNE ---
    n_samples = len(input_data)
    perplexity = 50#min(30, n_samples - 1)

    # 增加 init='pca' 提升稳定性
    tsne_model = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=200,
        init='pca',
        random_state=42,
        n_jobs=-1 # 使用所有 CPU 核心加速
    )

    try:
        tsne_results = tsne_model.fit_transform(input_data)
    except Exception as e:
        print(f"Error processing {group_name}: {e}")
        continue

    # --- 后续绘图部分保持不变 ---
    tsne_df = pd.DataFrame(tsne_results, columns=['tSNE1', 'tSNE2'])
    tsne_df['Label'] = y.values
    '''    
    # Perform t-SNE and plot each feature group
    for group_name, group_data in feature_groups.items():
    print(f"Processing {group_name} with shape {group_data.shape}")
    
    # Adjust perplexity based on sample size
    n_samples = len(group_data)
    perplexity = min(30, n_samples - 1)  # Ensure perplexity is less than n_samples
    
    # Perform t-SNE for dimensionality reduction
    tsne_model = TSNE(n_components=2, perplexity=perplexity, learning_rate=200, random_state=42, n_jobs=1)
    tsne_results = tsne_model.fit_transform(group_data)

    # Create a DataFrame with the t-SNE results and labels
    tsne_df = pd.DataFrame(tsne_results, columns=['tSNE1', 'tSNE2'])
    tsne_df['Label'] = y.values
    '''
    # Create a new figure for this feature group
    plt.figure(figsize=(10, 10))
    
    # Get the actual labels present in this dataset
    present_labels = sorted(set(y), key=lambda x: GLOBAL_LABEL_ORDER.index(x) if x in GLOBAL_LABEL_ORDER else len(GLOBAL_LABEL_ORDER))
    
    # Plot each label separately to ensure correct coloring
    for label in present_labels:
        label_data = tsne_df[tsne_df['Label'] == label]
        if len(label_data) > 0:
            color = GLOBAL_COLOR_MAP.get(label, 'gray')  # Use gray for unknown labels
            plt.scatter(label_data['tSNE1'], label_data['tSNE2'], c=[color], label=label, s=2, alpha=0.95)
    
    plt.title(f"t-SNE Visualization - {group_name}", fontsize=14)
    plt.xlabel('tSNE1', fontsize=12)
    plt.ylabel('tSNE2', fontsize=12)
    plt.gca().set_facecolor('white')

    # Create legend using global label order (only include labels present in current set)
    current_labels = [label for label in GLOBAL_LABEL_ORDER if label in present_labels]
    handles = [plt.Line2D([0], [0], marker='o', color='w', 
                         markerfacecolor=GLOBAL_COLOR_MAP[label], markersize=8) 
              for label in current_labels]
    
    plt.legend(handles, current_labels, title='RNA Type', loc='best', 
               fontsize='medium', title_fontsize='medium')

    # Save the plot for this feature group
    plt.tight_layout()
    #plt.savefig(f'PNG/RNAcentral_Label10_{group_name}_tsne.pdf', dpi=300, bbox_inches='tight')
    #plt.savefig(f'PNG/RNAcentral_Label20_{group_name}_tsne.pdf')
    #plt.savefig(f'PNG/RNAcentral_{group_name}_tsne.png', dpi=300, bbox_inches='tight')

    plt.savefig(f'PNG/mammalian_{group_name}_tsne.pdf')
    #plt.savefig(f'PNG/multispecies_{group_name}_tsne_bpe.pdf')
    
    #plt.savefig(f'PNG/mammalian_{group_name}_tsne_bpe.pdf')
    #plt.savefig(f'PNG/multispecies_{group_name}_tsne_bpe.pdf')
    
    plt.close()  # Close the figure to free memory
    
    print(f"Completed {group_name}")

print("All t-SNE visualizations completed!")
