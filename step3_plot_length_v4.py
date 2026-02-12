import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. 读取 RNACentral 数据
input_file = 'rnacentral/rnacentral_active.csv'
df = pd.read_csv(input_file)

# === 1) 先对 lncRNA 进行长度过滤：删除 Label='lncRNA' 且 Length<200 的记录 ===
df['Length'] = pd.to_numeric(df['Length'], errors='coerce')
before_rows = len(df)
mask_lnc = (df['Label'] == 'lncRNA')
mask_short = mask_lnc & (df['Length'] < 200)
df = df[~mask_short].copy()
after_rows = len(df)
print(f"[INFO] Removed {before_rows - after_rows} rows where Label='lncRNA' and Length<200. Remaining rows: {after_rows}")

# 2. 定义完整的颜色映射表（7 + 13）
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

# 3. 严格按顺序仅对下面 18 个 label 做图
label_order_18 = [
    "rRNA", "Y_RNA", "snRNA", "snoRNA", "SRP_RNA", "tRNA",
    "pre_miRNA", "lncRNA", "miRNA", "siRNA", "antisense_RNA",
    "hammerhead_ribozyme", "misc_RNA", "piRNA", "ribozyme",
    "RNase_P_RNA", "sRNA", "tmRNA"
]

label_set_18 = set(label_order_18)

# 4. 过滤数据，只保留我们关注的18种标签
df18 = df[df['Label'].isin(label_set_18)].copy()

# 统计每个标签的数量（不改变顺序，只用于计数）
label_counts = df18['Label'].value_counts()

print(f"总标签数量: {len(df['Label'].unique())}")
print(f"关注的18种标签: {len(label_set_18)}")
print(f"18种标签的数据量: {len(df18)}")

# 5. 因为颜色映射已经完整，所以直接使用 FIXED_COLOR_MAPPING
complete_color_mapping = FIXED_COLOR_MAPPING

# 6. 计算动态x轴范围函数
def calculate_dynamic_xlim(data, percentile=95):
    """根据数据的百分位数动态计算x轴范围"""
    if len(data) == 0:
        return (0, 500)  # 默认值

    p95 = np.percentile(data, percentile)
    max_limit = np.ceil(p95 / 100) * 100

    max_limit = max(200, max_limit)
    max_limit = min(10000, max_limit)

    print(f"  数据范围: {data.min()}-{data.max()}, 95%分位数: {p95:.2f}, 选择范围: 0-{max_limit}")
    return (0, max_limit)

# 7. 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.formatter.useoffset'] = False
plt.rcParams['axes.formatter.use_mathtext'] = False

# 8. 创建 3×6 画布（18 个子图）
fig, axes = plt.subplots(3, 6, figsize=(24, 13))

# 9. 设置主标题
fig.suptitle('Length Distribution of 18 RNA Types (RNACentral Dataset)',
             fontsize=16,
             y=0.97)

# 10. 调整整体布局
plt.subplots_adjust(
    top=0.90,
    bottom=0.08,
    left=0.05,
    right=0.97,
    hspace=0.55,
    wspace=0.3
)

xlim_info = []

# 11. 绘制18种标签的长度分布
print(f"\n=== 开始绘制18种RNA类型的长度分布 ===")

for i, label in enumerate(label_order_18):
    row = i // 6  # 0-2 行
    col = i % 6   # 0-5 列
    ax = axes[row, col]

    subset = df18[df18['Label'] == label]
    count = len(subset)
    color = complete_color_mapping.get(label, '#000000')

    print(f"{i+1:2d}. {label:25s}: {count:8,}条数据 [颜色: {color}]")

    if count > 0:
        xlim = calculate_dynamic_xlim(subset['Length'])

        # 生物学特性调整（起点保留为0，目前都用0）
        if label in ['rRNA']:
            xmin = 0
            xmax = xlim[1]
            xlim = (xmin, xmax)
            print(f"  调整{label}范围: {xmin}-{xmax}")
        elif label in ['tRNA']:
            xmin = 0
            xmax = xlim[1]
            xlim = (xmin, xmax)
            print(f"  调整{label}范围: {xmin}-{xmax}")
        elif label in ['miRNA', 'pre_miRNA', 'siRNA', 'piRNA']:
            xmin = 0
            xmax = xlim[1]
            xlim = (xmin, xmax)
            print(f"  调整{label}范围: {xmin}-{xmax}")
        elif label in ['snRNA', 'snoRNA', 'Y_RNA', 'SRP_RNA', 'sRNA']:
            xmin = 0
            xmax = xlim[1]
            xlim = (xmin, xmax)
            print(f"  调整{label}范围: {xmin}-{xmax}")
        elif label in ['lncRNA']:
            xmin = 200
            xmax = xlim[1]
            xlim = (xmin, xmax)
            print(f"  调整{label}范围: {xmin}-{xmax}")
        elif label in ['hammerhead_ribozyme', 'ribozyme', 'RNase_P_RNA']:
            xmin = 0
            xmax = xlim[1]
            xlim = (xmin, xmax)
            print(f"  调整{label}范围: {xmin}-{xmax}")

        xlim_info.append((f"{label}", f"{xlim[0]}-{xlim[1]}"))

        data_range = xlim[1] - xlim[0]
        bins = min(50, max(20, int(data_range / 50)))

        sns.histplot(subset['Length'],
                     bins=bins,
                     kde=True,
                     color=color,
                     ax=ax,
                     edgecolor='black',
                     linewidth=0.5,
                     binrange=xlim)

        median_len = np.median(subset['Length'])
        mean_len = np.mean(subset['Length'])
        max_len = np.max(subset['Length'])
        ax.axvline(median_len, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
        ax.axvline(mean_len, color='green', linestyle='-', linewidth=1.5, alpha=0.8)

        ax.set_title(f"{label}\n(n={count:,}, max={max_len:,})", pad=12)
        ax.set_xlabel(f'Length ({int(xlim[0])}-{int(xlim[1])})', labelpad=8)
        ax.set_ylabel('Frequency', labelpad=8)

        ax.text(0.95, 0.85,
                f"Med: {median_len:.1f}\nMean: {mean_len:.1f}\nMax: {max_len:,}",
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round'),
                fontsize=8)

        ax.set_xlim(xlim)
        ax.ticklabel_format(style='plain', axis='x')
        ax.ticklabel_format(style='plain', axis='y')
        ax.get_xaxis().set_major_formatter(
            plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    else:
        ax.text(0.5, 0.5, 'No Data',
                transform=ax.transAxes,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=12)
        ax.set_title(f"{label}\n(n=0)", pad=12)
        ax.set_xlim(0, 500)

    ax.tick_params(axis='both', which='major', pad=6)

# 12. 隐藏多余子图（如果将来 label 少于 18）
actual_count = len(label_order_18)
for i in range(actual_count, 18):
    row = i // 6
    col = i % 6
    axes[row, col].set_visible(False)
    print(f"隐藏空子图: ({row},{col})")

# 13. 颜色图例保留原来的注释状态（如需可取消注释）
legend_elements = [plt.Rectangle((0, 0), 1, 1,
                                 facecolor=complete_color_mapping[label],
                                 edgecolor='black',
                                 label=f"{label} ({label_counts.get(label, 0):,})")
                   for label in label_order_18]

'''
fig.legend(handles=legend_elements,
           loc='center right',
           bbox_to_anchor=(0.98, 0.5),
           frameon=True,
           facecolor='white',
           title='RNA Types (Count)',
           title_fontsize=10,
           fontsize=7,
           ncol=1)
'''

# 14. 添加统计图例
fig.legend(['Median', 'Mean'],
           loc='lower center',
           ncol=2,
           bbox_to_anchor=(0.5, 0.02),
           frameon=True,
           facecolor='white',
           title='Statistics',
           title_fontsize=10,
           fontsize=10)

# 15. 保存图像
plt.savefig('rnacentral_18_rna_types_length_distributions.pdf', dpi=300, bbox_inches='tight')
plt.savefig('rnacentral_18_rna_types_length_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# 16. 打印详细统计信息
print("\n=== 详细统计信息 ===")
print(f"数据集总序列数: {len(df):,}")
print(f"18种标签总序列数: {len(df18):,}")
print(f"覆盖率: {len(df18)/len(df)*100:.2f}%")

print("\n18种标签分布（按给定顺序）:")
for i, label in enumerate(label_order_18, 1):
    count = label_counts.get(label, 0)
    percentage = (count / len(df18) * 100) if len(df18) > 0 else 0.0
    print(f"{i:2d}. {label:25s}: {count:8,}条 ({percentage:6.2f}%) - {complete_color_mapping[label]}")

print("\n各标签长度统计:")
length_stats = df18.groupby('Label')['Length'].agg(['count', 'min', 'max', 'median', 'mean']).round(2)
print(length_stats)

length_stats.to_csv('rnacentral_18_types_length_statistics.csv')
print("\n统计结果已保存到: rnacentral_18_types_length_statistics.csv")

# 18. 打印完整的颜色映射表
print("\n" + "="*60)
print("完整的颜色映射表（只打印18种关注标签）")
print("="*60)
for i, label in enumerate(label_order_18, 1):
    count = label_counts.get(label, 0)
    print(f"{i:2d}. {label:15s}: {complete_color_mapping[label]} (n={count:,})")

# 19. 保存颜色映射表到文件
color_mapping_df = pd.DataFrame([
    {
        'RNA_Type': label,
        'Color': complete_color_mapping[label],
        'Color_Type': 'Fixed' if label in FIXED_COLOR_MAPPING else 'Other',
        'Count': label_counts.get(label, 0)
    }
    for label in label_order_18
])
color_mapping_df.to_csv('rna_color_mapping_table_18.csv', index=False)
print("\n颜色映射表已保存到: rna_color_mapping_table_18.csv")

