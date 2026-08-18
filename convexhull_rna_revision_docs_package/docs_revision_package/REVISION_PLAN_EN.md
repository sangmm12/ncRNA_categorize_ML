# ConvexHull-RNA Revision Plan (English)

## 0. Current status

Most reviewer-requested control analyses have been completed. The CD-HIT redundancy-control analysis is still running, so the CD-HIT section is intentionally left as a placeholder.

### Completed analyses

1. **Length-confounding controls for RNACentral label_set10**
   - Length-only accuracy is approximately 0.59.
   - Full ACNV accuracy is approximately 0.90.
   - Full ACNV residualized by log(length) remains approximately 0.875-0.887.
   - Dropping the first eight basic count/position features remains approximately 0.893-0.897.
   - Dropping the first eight features plus log(length) residualization remains approximately 0.876-0.885.
   - Conclusion: transcript length is informative but does not solely explain ACNV performance.

2. **High-dimensional classifier controls**
   - True ACNV features substantially outperform random controls.
   - Label-permutation and random Gaussian features yield near-chance accuracy (~0.10 for the 10-class task).
   - Conclusion: classifier performance is not explained by high dimensionality alone.

3. **Convex-hull dimensionality controls**
   - Corrected mammalian/multispecies analyses were rerun with `x_start=3`.
   - RNACentral top20 and ENA cRNA/ncRNA analyses were also added.
   - Key conclusion: low-dimensional ACNV spaces show meaningful observed class organization, but high-dimensional complete hull disjointness can also occur under random or permuted controls. Therefore, high-dimensional hull disjointness should be treated as a geometric diagnostic rather than stand-alone evidence of biological separability.

4. **k-mer/BPE/fusion experiment**
   - On sampled RNACentral top20:
     - k-mer only accuracy ≈ 0.8304
     - BPE only accuracy ≈ 0.8231
     - k-mer+BPE concatenation accuracy ≈ 0.8301
   - Conclusion: BPE is an alternative representation, but simple concatenation does not improve over 5-mer ACNV.

5. **Coding vs non-coding RNA module**
   - ENA cRNA/ncRNA convex-hull analysis:
     - observed disjoint rate = 1.0 at 24, 88, 344 and 1368 dimensions.
     - random/permuted controls are 0 at 24/88 dimensions but become 1.0 at 344/1368 dimensions.
   - PCA/UMAP visualizations have been generated.
   - The UMAP figure must state that it uses **10,000 samples total: 5,000 cRNA + 5,000 ncRNA**.
   - Conclusion: the binary module should be framed as a first-stage screening component for the online server, not as a replacement for specialized coding-potential tools.

6. **CD-HIT redundancy control**
   - Still running.
   - The manuscript and rebuttal should temporarily include a placeholder for cluster-aware split validation.

## 1. Revised manuscript narrative

The revised narrative should be:

> ACNV captures biologically structured sequence-organization signals already visible in low-dimensional spaces. Complete high-dimensional convex-hull disjointness is interpreted as a geometric diagnostic, not as stand-alone proof of biological separability. The main evidence comes from the convergence of classifier performance, length controls, random/permutation controls, dimensionality controls, and redundancy-aware validation.

## 2. Required tables

### Main Table 1. RNACentral 10-family length-control results
Source:
`reviewer_revision_results_missing/length_controls_v2_label10_seed42/length_control_v2_metrics.csv`

Recommended columns:
- experiment
- seed
- n_train
- n_test
- accuracy
- macro_f1
- mcc

### Main Table 2. k-mer/BPE/fusion performance on RNACentral top20
Source:
`reviewer_revision_results_missing/fused_top20_seed42/fused_feature_metrics.csv`

Recommended columns:
- experiment
- n_classes
- n_features
- accuracy
- macro_f1
- mcc

### Main Table 3. Low-dimensional convex-hull control summary
Sources:
- `reviewer_revision_results_followup/convex_hull_mammalian_xstart3/convex_hull_control_summary_v2.csv`
- `reviewer_revision_results_followup/convex_hull_multispecies_xstart3/convex_hull_control_summary_v2.csv`
- `reviewer_revision_results_followup/convex_hull_top20_seed42/convex_hull_control_summary_v2.csv`
- `reviewer_revision_results_followup/convex_hull_cRNA_ncRNA/convex_hull_control_summary_v2.csv`

Recommended columns:
- dataset
- dim
- observed mean disjoint rate
- permuted mean disjoint rate
- random Gaussian mean disjoint rate

The main text should emphasize 24- and 88-dimensional results. Full 344/1368-dimensional results should be placed in Supplementary Tables.

### Main/Supplementary Table 4. CD-HIT cluster-aware split performance
Placeholder source:
`reviewer_revision_results_followup/cdhit_cluster_split_train_label10/cdhit_cluster_split_metrics.csv`

This section remains pending until CD-HIT finishes.

## 3. Required figures

### Revised Figure 1. Workflow
Recommended panels:
A. ACNV feature construction  
B. ncRNA family classification  
C. online server: cRNA/ncRNA screening + ncRNA subtype prediction

### Figure 2. Length-control / residualization performance
Bar plot showing:
- length_only
- full_acnv
- full_acnv_residualized_by_log_length
- drop_first_8_basic_features
- drop_first_8_residualized_by_log_length

### Figure 3. Convex-hull controls
Show observed, permuted-label and random-feature disjoint rates across 24/88/344/1368 dimensions. The key message is that low-dimensional observed disjointness is meaningful, whereas high-dimensional complete disjointness must be interpreted cautiously.

### Supplementary Figure. cRNA/ncRNA UMAP
Source:
`reviewer_revision_results_followup/binary_cRNA_ncRNA_visualization/binary_umap_scatter.png`

Required caption statement:
“UMAP visualization was generated from a balanced subset of 10,000 ENA sequences, including 5,000 cRNA and 5,000 ncRNA samples.”

### Supplementary Figure. cRNA/ncRNA PCA
Source:
`reviewer_revision_results_followup/binary_cRNA_ncRNA_visualization/binary_pca_scatter.png`

## 4. Manuscript sections to revise

### Methods
Add:
1. Length-control analysis  
2. Label-permutation and random-feature controls  
3. Convex-hull control interpretation  
4. k-mer/BPE/fusion experiment  
5. Online predictor deployment and model card  
6. CD-HIT cluster-aware validation placeholder  

### Results
Add:
1. Length alone is insufficient to explain ACNV performance  
2. Classifier performance remains strong under length and feature-ablation controls  
3. Random/permutation controls confirm that classifier accuracy is not high-dimensional chance  
4. Low-dimensional convex-hull organization is meaningful  
5. High-dimensional hull disjointness is not overinterpreted  
6. k-mer and BPE are alternative but partly redundant representations  
7. cRNA/ncRNA is a first-stage screening module  

### Discussion
Add:
1. Limitations of convex-hull interpretation in high dimensions  
2. Coding/non-coding classification is not the central claim  
3. ConvexHull-RNA does not replace specialized coding-potential tools  
4. Annotation refinement should be reframed as computational prioritization for manual curation  
5. CD-HIT redundancy control as supplementary validation  
