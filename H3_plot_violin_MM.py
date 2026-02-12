import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
#df_mammalian = pd.read_csv('mammalian_multispecies/mammalian_train_dev_pre_Avg.csv')
#df_multispecies = pd.read_csv('mammalian_multispecies/multispecies_train_dev_pre_Avg.csv')
df_mammalian = pd.read_csv('mammalian_multispecies/mammalian_train_dev_pre_Avg_bpe.csv')
df_multispecies = pd.read_csv('mammalian_multispecies/multispecies_train_dev_pre_Avg_bpe.csv')

species_order = ['rRNA','Y_RNA','snRNA','snoRNA','SRP_RNA','tRNA','pre_miRNA']

# Define label mappings
mammalian_label_mapping = {
    0: 'rRNA',
    1: 'Y_RNA',
    2: 'snRNA',
    3: 'snoRNA',
    4: 'SRP_RNA',
    5: 'tRNA',
    6: 'pre_miRNA'
}

multispecies_label_mapping = {
    6: 'rRNA',
    9: 'Y_RNA',
    5: 'snRNA',
    3: 'snoRNA',
    2: 'SRP_RNA',
    0: 'tRNA',
    1: 'pre_miRNA'
}

# Apply label mappings
df_mammalian['Label'] = df_mammalian['Label'].map(mammalian_label_mapping)
df_multispecies['Label'] = df_multispecies['Label'].map(multispecies_label_mapping)

# Get headers and define k-mer features
headers = list(df_mammalian.columns)
features_dict = {
    '2-mer': headers[1:17],      # 4^2 = 16 features
    '3-mer': headers[1:65],      # 4^3 = 64 features  
    '4-mer': headers[1:257],     # 4^4 = 256 features
    '5-mer': headers[1:1025],    # 4^5 = 1024 features
}

# Create figure with 2 rows (mammalian top, multispecies bottom) and 7 columns
fig, axes = plt.subplots(2, 7, figsize=(20, 8))

# Create color palette
palette = sns.color_palette("deep", n_colors=len(species_order))
color_map = dict(zip(species_order, palette))

# Process mammalian data (top row)
df_mammalian_grouped = df_mammalian.groupby('Label').mean().reset_index()
df_mammalian_grouped = df_mammalian_grouped[df_mammalian_grouped['Label'].isin(species_order)]
df_mammalian_grouped['Label'] = pd.Categorical(df_mammalian_grouped['Label'], categories=species_order, ordered=True)
df_mammalian_grouped = df_mammalian_grouped.sort_values('Label')

# Process multispecies data (bottom row)
df_multispecies_grouped = df_multispecies.groupby('Label').mean().reset_index()
df_multispecies_grouped = df_multispecies_grouped[df_multispecies_grouped['Label'].isin(species_order)]
df_multispecies_grouped['Label'] = pd.Categorical(df_multispecies_grouped['Label'], categories=species_order, ordered=True)
df_multispecies_grouped = df_multispecies_grouped.sort_values('Label')

# Plot mammalian data (top row)
for col_idx, label in enumerate(species_order):
    ax = axes[0, col_idx]
    
    # Prepare plot data for all k-mer features - 保持原来的数据结构
    plot_data = []
    row_data = df_mammalian_grouped[df_mammalian_grouped['Label'] == label].iloc[0]
    for kmer_features in features_dict.values():
        for col in kmer_features:
            plot_data.append({'Label': label, 'Value': row_data[col]})
    plot_df = pd.DataFrame(plot_data)
    
    # 保持原来的violinplot代码不变
    subset = plot_df[plot_df['Label'] == label]
    sns.violinplot(
        y=subset['Value'],
        palette=[color_map[label]],
        ax=ax,
        cut=0,
        inner='box',
        linewidth=1.5,
        edgecolor='cyan'
    )
    
    ax.set_title(f'Mammalian\n{label}', fontsize=12, fontweight='bold')
    ax.set_xticks([])
    
    if col_idx == 0:
        ax.set_ylabel('All k-mer Values', fontsize=11)
    else:
        ax.set_ylabel('')

# Plot multispecies data (bottom row)
for col_idx, label in enumerate(species_order):
    ax = axes[1, col_idx]
    
    # Prepare plot data for all k-mer features - 保持原来的数据结构
    plot_data = []
    row_data = df_multispecies_grouped[df_multispecies_grouped['Label'] == label].iloc[0]
    for kmer_features in features_dict.values():
        for col in kmer_features:
            plot_data.append({'Label': label, 'Value': row_data[col]})
    plot_df = pd.DataFrame(plot_data)
    
    # 保持原来的violinplot代码不变
    subset = plot_df[plot_df['Label'] == label]
    sns.violinplot(
        y=subset['Value'],
        palette=[color_map[label]],
        ax=ax,
        cut=0,
        inner='box',
        linewidth=1.5,
        edgecolor='cyan'
    )
    
    ax.set_title(f'Multispecies\n{label}', fontsize=12, fontweight='bold')
    ax.set_xticks([])
    
    if col_idx == 0:
        ax.set_ylabel('All k-mer Values', fontsize=11)
    else:
        ax.set_ylabel('')

#plt.suptitle('Average k-mer Distribution by ncRNA Type: Mammalian (Top) vs Multispecies (Bottom)', fontsize=16, y=0.95)
plt.suptitle('Average 1358 BPE Distribution by ncRNA Type: Mammalian (Top) vs Multispecies (Bottom)', fontsize=16, y=0.95)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('mammalian_multispecies/violinplot_kmer_comparison_bpe.png', dpi=300, bbox_inches='tight')
plt.savefig('mammalian_multispecies/violinplot_kmer_comparison_bpe.pdf', bbox_inches='tight')
plt.close()
print("✅ Comparison plot saved successfully.")
