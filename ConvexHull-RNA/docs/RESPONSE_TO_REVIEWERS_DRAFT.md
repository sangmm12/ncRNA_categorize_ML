# Response to Reviewers Draft Blocks

## High-dimensional convex-hull concern
We thank the reviewer for raising the important concern that complete convex-hull disjointness in high-dimensional spaces may arise from high-dimensional geometry rather than biological structure alone. We have added random Gaussian feature controls, label-permutation controls, dimensionality controls and repeated subsampling analyses. These analyses showed that observed labels were already strongly organized in low-dimensional ACNV spaces, particularly at 24 dimensions, whereas random and permuted controls showed no disjointness at that dimension. However, at higher dimensions, random and permuted controls also became increasingly disjoint. We therefore revised the manuscript to interpret high-dimensional hull disjointness as a geometric diagnostic rather than stand-alone proof of biological separability.

## Length confounding
We agree that transcript length can contribute to ncRNA family discrimination. We therefore added several length-control analyses, including a length-only classifier, log(length)-residualized ACNV features, removal of the first eight basic count/position features and a combined ablation/residualization analysis. The length-only model achieved approximately 0.59 accuracy, whereas full ACNV achieved approximately 0.90 accuracy. After log(length) residualization and removal of basic features, ACNV still retained approximately 0.88 accuracy. These results indicate that ACNV performance is not explained solely by transcript length.

## k-mer versus BPE
We added a direct comparison of 5-mer ACNV, BPE-based ACNV and concatenated k-mer+BPE features on the sampled RNACentral top20 task. The concatenated representation did not outperform 5-mer ACNV alone. We therefore revised the manuscript to describe BPE as an alternative representation that is partly redundant with fixed k-mer ACNV under the current classifier setting, rather than claiming that simple feature fusion necessarily improves performance.

## Coding versus non-coding task
We thank the reviewer for noting that coding versus non-coding RNA classification is distinct from ncRNA family classification. We have repositioned this task as a secondary demonstration of ACNV generality and as the first-stage screening module of the online server. We also added a dedicated convex-hull analysis and PCA/UMAP visualization for the deduplicated ENA cRNA/ncRNA dataset. The UMAP visualization was generated from a balanced subset of 10,000 sequences, including 5,000 cRNA and 5,000 ncRNA samples. The revised manuscript clarifies that this module is not intended to replace specialized coding-potential tools.

## GitHub and reproducibility
We have reorganized the code into a GitHub-ready structure with README documentation, installation commands, example input/output files, training and prediction commands, reviewer-control scripts and an online-server model card. These additions are intended to improve reproducibility and allow users to rerun the ACNV feature construction, training, prediction and reviewer-control workflows.

## CD-HIT redundancy control
We agree that exact duplicate removal is insufficient to rule out homology-driven performance inflation. We have initiated CD-HIT-est clustering at 80% sequence identity and will add cluster-aware train/test split results once the clustering finishes. This analysis will be reported in the revised supplementary material.
