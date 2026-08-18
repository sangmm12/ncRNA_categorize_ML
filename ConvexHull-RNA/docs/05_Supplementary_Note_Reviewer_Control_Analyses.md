# Supplementary Note: Reviewer-Control Analyses

## S1. Length-control analyses

We evaluated whether transcript length could explain the RNACentral label_set10 classification performance. Five models were compared:

1. Length-only baseline.
2. Full ACNV.
3. Full ACNV residualized by log(length).
4. ACNV after removing the first eight basic count/position features.
5. ACNV after removing the first eight features and residualizing by log(length).

The length-only classifier achieved approximately 0.59 accuracy. Full ACNV achieved approximately 0.90 accuracy. After length residualization and feature ablation, ACNV still retained approximately 0.88 accuracy. These results demonstrate that length contributes to the task but does not fully explain ACNV performance.

Source table:
`reviewer_revision_results_missing/length_controls_v2_label10_seed42/length_control_v2_metrics.csv`

## S2. Random-feature and label-permutation controls

To test whether high-dimensional classification performance could arise by chance, we trained classifiers using true ACNV features, permuted labels, and random Gaussian features. Controls were evaluated at 24, 88, 344, and 1368 dimensions. True ACNV features substantially outperformed controls, while random and permuted models remained near chance for the 10-class task.

Source table:
`reviewer_revision_results/highdim_controls_seed42/highdim_control_metrics.csv`

## S3. Convex-hull dimensionality controls

Convex-hull disjointness was evaluated using linear programming. We tested observed labels, permuted labels, and random Gaussian features. Datasets included:

1. Mammalian benchmark.
2. Multispecies benchmark.
3. RNACentral top20.
4. ENA cRNA/ncRNA binary dataset.

The main conclusion is that low-dimensional ACNV spaces show meaningful observed class organization, while high-dimensional complete hull disjointness can also occur in random or permuted controls. Therefore, high-dimensional hull disjointness is treated as a geometric diagnostic.

Source tables:
- `reviewer_revision_results_followup/convex_hull_mammalian_xstart3/convex_hull_control_summary_v2.csv`
- `reviewer_revision_results_followup/convex_hull_multispecies_xstart3/convex_hull_control_summary_v2.csv`
- `reviewer_revision_results_followup/convex_hull_top20_seed42/convex_hull_control_summary_v2.csv`
- `reviewer_revision_results_followup/convex_hull_cRNA_ncRNA/convex_hull_control_summary_v2.csv`

## S4. k-mer/BPE/fusion comparison

We compared 5-mer ACNV, BPE-based ACNV, and concatenated k-mer+BPE features on a sampled RNACentral top20 task. Concatenation did not outperform 5-mer ACNV alone, suggesting partial redundancy under the current classifier setting.

Source table:
`reviewer_revision_results_missing/fused_top20_seed42/fused_feature_metrics.csv`

## S5. cRNA/ncRNA visualization

A balanced subset of 10,000 ENA sequences was sampled, including 5,000 cRNA and 5,000 ncRNA sequences. ACNV features were standardized and projected using PCA and UMAP. The UMAP plot shows strong separation between the two coarse classes and supports the role of the binary module as a first-stage screening component.

Source files:
- `reviewer_revision_results_followup/binary_cRNA_ncRNA_visualization/binary_umap_scatter.png`
- `reviewer_revision_results_followup/binary_cRNA_ncRNA_visualization/binary_pca_scatter.png`
- `reviewer_revision_results_followup/binary_cRNA_ncRNA_visualization/binary_pca_explained_variance.csv`

## S6. CD-HIT cluster-aware validation

CD-HIT-est clustering at 80% identity is currently running. After the `.clstr` output is generated, cluster-aware train/test splits will be created and evaluated.

Placeholder output files:
- `reviewer_revision_results_missing/cdhit_cluster_split_seed42/cluster_split_all_rows.csv`
- `reviewer_revision_results_followup/cdhit_cluster_split_train_label10/cdhit_cluster_split_metrics.csv`
