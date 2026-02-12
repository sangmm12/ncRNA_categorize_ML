import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置字体 - 使用系统默认字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans']
plt.rcParams['axes.unicode_minus'] = False

# 定义20种标签集合

label_set_20 = {
    "rRNA", "Y_RNA", "snRNA", "snoRNA", "SRP_RNA", "tRNA", "pre_miRNA", "miRNA", "siRNA", "lncRNA",
    "misc_RNA", "sRNA", "ncRNA", "piRNA", "tmRNA", "hammerhead_ribozyme", "RNase_P_RNA", "antisense_RNA", "other", "ribozyme"
}
# 定义完整的颜色映射表
FIXED_COLOR_MAPPING = {
    # 7种固定颜色的RNA类型
    'rRNA': '#FF0000',
    'Y_RNA': '#FFDB00', 
    'snRNA': '#49FF00',
    'snoRNA': '#00FF92',
    'SRP_RNA': '#0092FF',
    'tRNA': '#4900FF',
    'pre_miRNA': '#FF00DB',
    # 其他13种RNA类型的颜色映射
    'miRNA': '#F22424',
    'siRNA': '#F25A24',
    'lncRNA': '#F28F24',
    'misc_RNA': '#F2C524',
    'sRNA': '#E9F224',
    'ncRNA': '#B8F224',
    'piRNA': '#7DF224',
    'tmRNA': '#42F224',
    'hammerhead_ribozyme': '#24F24D',
    'RNase_P_RNA': '#24F288',
    'antisense_RNA': '#24F2C3',
    'other': '#24DDF2',
    'ribozyme': '#249FF2'
}

# 读取数据
df = pd.read_csv('rnacentral_active_predlncRNA_ncRNA_less200.csv')
df = pd.read_csv('rnacentral_active_predlncRNA_lncRNA_less200.csv')

# 检查数据
print("Data preview:")
print(df.head())
print(f"\nData shape: {df.shape}")
print(f"Number of unique Predict_Label: {df['Predict_Label'].nunique()}")
print(f"All Predict_Labels: {sorted(df['Predict_Label'].unique())}")

# 计算每个预测标签在总预测中的占比
total_predictions = len(df)
label_counts = df['Predict_Label'].value_counts()
label_percentages = (label_counts / total_predictions * 100).round(1)

# 设置图形布局：3行6列
fig, axes = plt.subplots(3, 5, figsize=(24, 12))
axes = axes.flatten()

# 获取所有唯一的Predict_Label并按数量排序
unique_labels = df['Predict_Label'].unique()

# 为每个标签创建分布图
for i, label in enumerate(unique_labels):
    if i >= 18:  # 只处理前18个标签
        break
        
    ax = axes[i]
    
    # 获取当前标签的数据
    label_data = df[df['Predict_Label'] == label]['Predict_Probability']
    
    # 计算统计信息
    total_count = len(label_data)
    prob_ge_05 = len(label_data[label_data >= 0.5])
    prob_lt_05 = len(label_data[label_data < 0.5])
    percentage_ge_05 = prob_ge_05 / total_count * 100
    percentage_lt_05 = prob_lt_05 / total_count * 100
    
    # 计算该标签在总预测中的占比
    label_percentage = label_percentages.get(label, 0)
    
    # 创建分布图，使用指定的颜色
    color = FIXED_COLOR_MAPPING.get(label, '#2462F2')  # 默认颜色
    sns.histplot(data=label_data, ax=ax, bins=20, kde=True, color=color, alpha=0.7)
    
    # 添加0.5红线
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold=0.5')
    
    # 设置标题和标签
    title = f'{label}\nCount: {total_count} ({label_percentage}% of total)'
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Predict Probability')
    ax.set_ylabel('Count')
    
    # 添加图例
    legend_text = f'≥0.5: {prob_ge_05} ({percentage_ge_05:.1f}%)\n<0.5: {prob_lt_05} ({percentage_lt_05:.1f}%)'
    ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=9)
    
    # 设置x轴范围
    ax.set_xlim(0, 1)
    
    # 添加网格
    ax.grid(True, alpha=0.3)

# 隐藏多余的子图（如果标签不足18个）
for i in range(len(unique_labels), 15):
    axes[i].set_visible(False)

# 调整布局
plt.tight_layout()

# 修改总标题，加入文件总数
total_title = f'Predict_Probability Distribution by Predict_Label (Total: {total_predictions} sequences)'
plt.suptitle(total_title, fontsize=16, fontweight='bold', y=1.02)
plt.savefig('probability_distribution_by_label_lncRNA.png', dpi=300, bbox_inches='tight')
plt.savefig('probability_distribution_by_label_lncRNA.pdf', dpi=300, bbox_inches='tight')
#plt.show()

# 打印每个标签的详细统计信息
print("\nDetailed statistics by label:")
stats_summary = []
for label in unique_labels:
    label_data = df[df['Predict_Label'] == label]['Predict_Probability']
    total = len(label_data)
    ge_05 = len(label_data[label_data >= 0.5])
    lt_05 = len(label_data[label_data < 0.5])
    pct_ge_05 = ge_05 / total * 100
    pct_of_total = total / total_predictions * 100
    
    stats_summary.append({
        'Predict_Label': label,
        'Color': FIXED_COLOR_MAPPING.get(label, 'Not defined'),
        'Total': total,
        'Percentage_of_Total': f'{pct_of_total:.1f}%',
        '≥0.5': ge_05,
        '<0.5': lt_05,
        '≥0.5%': f'{pct_ge_05:.1f}%'
    })

# 创建统计摘要表格
stats_df = pd.DataFrame(stats_summary)
print(stats_df.to_string(index=False))

# 打印颜色映射信息
print(f"\nColor mapping used:")
for label, color in FIXED_COLOR_MAPPING.items():
    print(f"{label}: {color}")
