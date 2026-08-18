# ConvexHull-RNA 修稿总计划（中文）

## 0. 当前状态

我们已经基本完成审稿人要求的主要补充实验。CD-HIT 聚类去冗余实验仍在运行中，本文档先将 CD-HIT 部分留作占位。

### 已完成结果

1. **长度混杂控制（RNACentral label_set10）**
   - length-only accuracy 约 0.59。
   - full ACNV accuracy 约 0.90。
   - full ACNV residualized by log(length) 后 accuracy 仍约 0.875-0.887。
   - drop first 8 basic features 后 accuracy 仍约 0.893-0.897。
   - drop first 8 + residualized by log(length) 后 accuracy 仍约 0.876-0.885。
   - 结论：长度确实有信号，但 ACNV 的分类能力不是由长度单独驱动。

2. **高维分类器控制**
   - true ACNV features 在 24/88/344/1368 维均明显高于随机。
   - label permutation 和 random Gaussian features 的 accuracy 约 0.10，接近 10 类随机水平。
   - 结论：分类器性能不是单纯由高维随机特征造成。

3. **convex-hull 高维几何控制**
   - mammalian/multispecies 七类数据已用正确 `x_start=3` 重跑。
   - RNACentral top20 和 ENA cRNA/ncRNA 也已补充 convex-hull control。
   - 关键结论：低维 24 维 ACNV 的 observed disjointness 明显高于 random/permuted controls；但 88/344/1368 维时 random/permuted controls 也逐渐分离，说明“高维完全凸包分离”不能单独作为生物学可分性的证明，只能作为几何诊断指标。

4. **k-mer/BPE/fusion 实验**
   - top20 sampled:
     - k-mer only accuracy ≈ 0.8304
     - BPE only accuracy ≈ 0.8231
     - k-mer+BPE concat accuracy ≈ 0.8301
   - 结论：BPE 是替代表示；简单拼接没有超过 5-mer ACNV，说明两者部分冗余，不能再强写“互补融合提升”。

5. **coding vs non-coding 任务补充**
   - ENA cRNA/ncRNA convex-hull:
     - observed 在 24/88/344/1368 维均 disjoint rate = 1.0。
     - random/permuted 在 24/88 维为 0，在 344/1368 维也变为 1.0。
   - 已生成 PCA/UMAP 图。
   - UMAP 图必须注明：使用 ENA deduplicated binary dataset 中 balanced subset，共 **10,000 samples（5,000 cRNA + 5,000 ncRNA）**。
   - 结论：二分类模块应作为 online server 的 first-stage screening，不作为本文主贡献，也不声称替代 CPAT/CPC2/RNAsamba 等专业工具。

6. **CD-HIT 聚类去冗余**
   - 当前仍在运行。
   - manuscript 和 rebuttal 中先保留占位：待 `cdhit_id0.8.fa.clstr` 生成后补入 cluster-level split performance。

## 1. 论文主线修改建议

原稿中对 convex-hull 的表述偏强，容易被审稿人质疑为“高维几何 artifact”。修订后建议主线改为：

> ACNV captures biologically structured sequence-organization signals that are already visible in low-dimensional spaces. Complete high-dimensional convex-hull disjointness is treated as a geometric diagnostic rather than stand-alone proof of biological separability. The main evidence is instead the convergence of classifier performance, length controls, random/permutation controls, dimensionality controls, and redundancy-aware validation.

中文理解：

> ACNV 在低维空间中已经表现出真实类别结构；高维完全凸包分离只能作为几何诊断，而不能单独证明生物学可分性。真正的证据来自分类器性能、长度控制、随机/置换控制、维度控制和去冗余验证的共同支持。

## 2. 需要新增/修改的表格

### Table 1. RNACentral 10-family length-control results
来源：
`reviewer_revision_results_missing/length_controls_v2_label10_seed42/length_control_v2_metrics.csv`

建议列：
- experiment
- seed
- n_train
- n_test
- accuracy
- macro_f1
- mcc

重点报告：
- length_only
- full_acnv
- full_acnv_residualized_by_log_length
- drop_first_8_basic_features
- drop_first_8_residualized_by_log_length

### Table 2. k-mer/BPE/fusion performance on RNACentral top20
来源：
`reviewer_revision_results_missing/fused_top20_seed42/fused_feature_metrics.csv`

建议列：
- experiment
- n_classes
- n_features
- accuracy
- macro_f1
- mcc

### Table 3. Low-dimensional convex-hull control summary
来源：
- `reviewer_revision_results_followup/convex_hull_mammalian_xstart3/convex_hull_control_summary_v2.csv`
- `reviewer_revision_results_followup/convex_hull_multispecies_xstart3/convex_hull_control_summary_v2.csv`
- `reviewer_revision_results_followup/convex_hull_top20_seed42/convex_hull_control_summary_v2.csv`
- `reviewer_revision_results_followup/convex_hull_cRNA_ncRNA/convex_hull_control_summary_v2.csv`

建议主表只列 24/88 维，344/1368 放 Supplementary：
- dataset
- dim
- observed mean disjoint rate
- permuted mean disjoint rate
- random Gaussian mean disjoint rate

### Table 4. CD-HIT cluster-aware split performance（暂留空）
来源待补：
`reviewer_revision_results_followup/cdhit_cluster_split_train_label10/cdhit_cluster_split_metrics.csv`

## 3. 需要新增/修改的图

### Figure 1. Revised workflow
建议重画为三条路径：
A. ACNV feature construction  
B. ncRNA family classification  
C. online server: cRNA/ncRNA screening + ncRNA subtype prediction

### Figure 2. Length-control performance
用 bar plot 显示：
- length_only
- full_acnv
- full_acnv_residualized_by_log_length
- drop_first_8_basic_features
- drop_first_8_residualized_by_log_length

### Figure 3. Convex-hull controls
显示 observed、permuted-label 和 random-feature disjoint rates across 24/88/344/1368 dimensions。重点强调低维 24 维 observed > controls，高维 controls 也会分离。

### Supplementary Figure. cRNA/ncRNA UMAP
来源：
`reviewer_revision_results_followup/binary_cRNA_ncRNA_visualization/binary_umap_scatter.png`

图注必须写：
“UMAP visualization was generated from a balanced subset of 10,000 ENA sequences, including 5,000 cRNA and 5,000 ncRNA samples.”

### Supplementary Figure. cRNA/ncRNA PCA
来源：
`reviewer_revision_results_followup/binary_cRNA_ncRNA_visualization/binary_pca_scatter.png`

## 4. Manuscript 需要改的关键段落

### Methods 增加
1. Length-control analysis  
2. Label-permutation and random-feature controls  
3. Convex-hull control interpretation  
4. k-mer/BPE/fusion experiment  
5. Online predictor deployment and model card  
6. CD-HIT cluster-aware validation（暂留空）

### Results 增加
1. Length alone is insufficient  
2. Random/permutation controls confirm classifier signal  
3. Low-dimensional ACNV contains meaningful hull organization  
4. High-dimensional hull disjointness is not overinterpreted  
5. k-mer and BPE are alternative, partly redundant representations  
6. cRNA/ncRNA task is a first-stage screening module  

### Discussion 增加
1. 限制 convex-hull 生物学解释  
2. 不把 coding-potential module 作为主贡献  
3. 不声称替代专门 coding-potential tools  
4. annotation refinement 改为 “computational prioritization for manual curation”  
5. CD-HIT 去冗余作为 supplementary validation  
